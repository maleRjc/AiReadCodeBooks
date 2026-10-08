# Chapter 24: Architectural Evolution and Future Directions: From Static Communication to Programmable Communication

In the previous chapter we saw how the community builds a surrounding ecosystem around the NCCL core: Python bindings, Rust bindings, expert-parallel communication, ultra-bandwidth primitives, and communication checkpoints. These projects all reuse NCCL's stable API, but their demands have already gone beyond the scope of traditional collective communication—expert parallelism requires fine-grained point-to-point send/receive, checkpoints require pausing/resuming communication state, and ultra-bandwidth primitives require bypassing standard collective operations to directly operate the network. These demands point to the same problem: NCCL's fixed collective operation model is being stretched to the breaking point by more flexible communication needs. In this chapter we will no longer look at a single module, but instead start from the traces of evolution that have already appeared in the source code and discuss where NCCL is heading. Specifically, we will analyze three intertwined forces of evolution: communication primitives moving from fixed collectives to programmable—the RMA task scheduling in src/rma/rma.cc allows upper layers to compose Put/Signal/WaitSignal primitives instead of only calling AllReduce; network initiation moving from host proxy to GPU direct issue—the GIN backend management in src/gin/gin_host.cc allows GPU kernels to directly drive the NIC; and the memory model moving from registered buffers to symmetric memory—the symmetric memory kernel selection in src/sym_kernels.cc allows all ranks to use the same set of virtual addresses to access each other's buffers. These three forces are not isolated; they share the same infrastructure: the team abstraction in src/nccl_device/core.cc and the versioned DevComm in src/devcomm/devcomm_v23100.cc. Understanding how they mesh together means understanding the evolution logic of NCCL from a "collective communication library" to a "programmable communication engine."

# 1. Programmable Communication Primitives: How RMA Turns a "Fixed Recipe" into a "Buffet"

## Intuitive model

Traditional NCCL collective communication is like a fixed set meal: you order AllReduce, and the kitchen just follows the AllReduce procedure to completion. But in expert parallelism (MoE) scenarios, each token needs to be sent to a different expert, and the sending pattern is completely unknown at compile time—this is like a buffet, where you have to decide what to take, how much to take, and when to take it.

RMA is the "buffet counter" that NCCL provides to upper layers: Put (write data to the peer's memory), Signal (notify the peer), WaitSignal (wait for the peer's signal). Upper-layer frameworks can freely combine these three primitives to implement arbitrary communication patterns.

Without RMA, MoE's all-to-all can only be simulated through multiple small-scale collective operations, each requiring the full kernel launch and synchronization process, resulting in unacceptably high latency.

## Data Structures and Memory Layout

The core data structures of RMA are`ncclTaskRma`(task description) and`ncclRmaArgs`(plan parameters). Let's first look at`ncclRmaArgs`'s fields, which are initialized in`scheduleRmaTasksToPlan`.

[FACT:src/rma/rma.cc:166-171]

```cpp
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
plan->rmaArgs->nRmaTasks = 0;
plan->rmaArgs->nRmaTasksProxy = 0;
plan->rmaArgs->nRmaTasksCe = 0;
```

The key fields here are`nRmaTasksProxy`and`nRmaTasksCe`. They split RMA tasks into two execution paths:

- **CE path**(Copy Engine): The target rank is within the LSA (Local Symmetric Access) range, which can be completed directly using the GPU's copy engine without needing the network.
- **Proxy path**: The target rank is not within the LSA range and must go through a host proxy thread to drive the network.

> **[Design Inference & Architectural Trade-offs]**
> The motivation behind this dichotomy is straightforward: communication within the LSA range goes over NVLink or PCIe, which has high bandwidth and low latency, making asynchronous copy with CE the most cost-effective; cross-machine communication must go through the NIC and can only be driven by proxy threads. Only by scheduling the two types of tasks separately can CE and proxy execute in parallel, rather than waiting serially.

`ncclTaskRma`itself contains`peers`、`nsignals`、`signalIdxs`three array pointers, recording the peer rank, signal count, and signal index respectively. For WaitSignal tasks, one task can wait for multiple peers; for Put/Signal tasks, one task targets only one peer.

## Step-by-Step Walkthrough: Scheduling of a WaitSignal

Let's plug in a concrete scenario: rank 0 calls`ncclWaitSignal`, waiting for signals from rank 1 and rank 3. Assume rank 1 is within the LSA range and rank 3 is not.

**Step 1: Find the first non-empty context queue.**

[FACT:src/rma/rma.cc:148-158]

```cpp
int ctx = -1;
for (int i = 0; i config.numRmaCtx; i++) {
  if (!ncclIntruQueueEmpty(&planner->rmaTaskQueues[i])) {
    ctx = i;
    break;
  }
}
if (ctx == -1) return ncclSuccess;
```

RMA tasks are queued by context, and each context is an independent RMA channel. Here, find the first context that has tasks and take out its queue.

**Step 2: Take out the first task and determine its type.**

[FACT:src/rma/rma.cc:163-168]

```cpp
struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
```

`firstTask->func`is`ncclFuncWaitSignal`, enter the WaitSignal branch.

**Step 3: Split peers by LSA reachability.**

[FACT:src/rma/rma.cc:187-204]

```cpp
for (int i = 0; i npeers; i++) {
  int peerRank = firstTask->peers[i];
  bool lsaAccessible = isLsaAccessible(comm, peerRank);
  if (lsaAccessible) {
    peersCe[npeersCe] = peerRank;
    nsignalsCe[npeersCe] = firstTask->nsignals[i];
    signalIdxsCe[npeersCe] = firstTask->signalIdxs[i];
    npeersCe++;
  } else {
    peersProxy[npeersProxy] = peerRank;
    nsignalsProxy[npeersProxy] = firstTask->nsignals[i];
    signalIdxsProxy[npeersProxy] = firstTask->signalIdxs[i];
    npeersProxy++;
  }
}
```

`isLsaAccessible`iterates over`comm->devrState.lsaRankList`, determining whether the peer is within the LSA team. Rank 1 is within LSA, so it goes into the CE list; rank 3 is not, so it goes into the Proxy list.

**Step 4: Create a new task for each of CE and Proxy.**

[FACT:src/rma/rma.cc:206-246]

```cpp
if (npeersCe > 0) {
  struct ncclTaskRma* waitSignalTaskCe = ...;
  waitSignalTaskCe->peers = peersCe;
  waitSignalTaskCe->npeers = npeersCe;
  ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, waitSignalTaskCe);
  plan->rmaArgs->nRmaTasksCe = 1;
}
if (npeersProxy > 0) {
  struct ncclTaskRma* waitSignalTaskProxy = ...;
  waitSignalTaskProxy->peers = peersProxy;
  waitSignalTaskProxy->npeers = npeersProxy;
  ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, waitSignalTaskProxy);
  plan->rmaArgs->nRmaTasksProxy = 1;
}
```

The original single WaitSignal task is split into two: the CE task waits for rank 1, and the Proxy task waits for rank 3. The two tasks can execute in parallel—the CE path waits on the GPU, and the Proxy path waits on the host thread.

**Step 5: Release the original task.**

[FACT:src/rma/rma.cc:249-251]

```cpp
planner->nTasksRma -= 1;
ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
```

The original task has already been split into two new tasks, so it is released back to the memory pool.

## Concurrency Control and Hardware Interaction

The parallel execution of RMA is reflected in`ncclRmaWaitSignal`.

[FACT:src/rma/rma.cc:43-74]

```cpp
if (plan->rmaArgs->nRmaTasksProxy > 0 && plan->rmaArgs->nRmaTasksCe > 0) {
  cudaStream_t ceStream = comm->rmaState.rmaCeState.ceStream;
  cudaEvent_t ceEvent = comm->rmaState.rmaCeState.ceEvent;
  CUDACHECKGOTO(cudaEventRecord(ceEvent, stream), ret, fail);
  CUDACHECKGOTO(cudaStreamWaitEvent(ceStream, ceEvent, 0), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyWaitLaunch(comm, plan, stream), ret, fail);
  NCCLCHECKGOTO(ncclRmaCeWaitLaunch(comm, plan, ceStream), ret, fail);
  CUDACHECKGOTO(cudaEventRecord(ceEvent, ceStream), ret, fail);
  CUDACHECKGOTO(cudaStreamWaitEvent(stream, ceEvent, 0), ret, fail);
}
```

This code uses CUDA events for inter-stream synchronization: first record an event on the input stream, let the CE stream wait for this event, then launch the proxy and CE tasks on the two streams respectively, and finally let the input stream wait for the CE stream's event. In this way, the two paths advance in parallel, but externally it appears as a single synchronous operation.

> **[Design Inference & Architectural Trade-offs]**
> The design trade-off here is: parallel execution can reduce latency, but it introduces additional event recording and stream synchronization overhead. For small messages, this overhead may exceed the parallel benefit; for large messages, the parallel benefit is significant. NCCL does not make an adaptive judgment here, but uniformly takes the parallel path—because the typical scenario for RMA is fine-grained communication of large messages.

## Production Pitfall Avoidance Guide

**Pitfall 1: Incorrect LSA reachability judgment causes tasks to take the wrong path.** `isLsaAccessible`iterates over`lsaRankList`, if`lsaSize`is 0 (for example, a single-rank communication domain), all peers will be judged unreachable and all will take the Proxy path. This will not be exposed during small-scale testing, but it will cause a sharp performance drop in large-scale deployments. The troubleshooting method is to look at`scheduleRmaTasksToPlan`'s INFO logs for the ratio of`nRmaTasksProxy`and`nRmaTasksCe`.

**Pitfall 2: The lifetime of the peer array after a WaitSignal task is split.**The CE path's`peersCe`uses`ncclMemoryStackAlloc`allocation, and its lifetime follows`comm->memScoped`; the Proxy path's`peersProxy`uses`ncclCalloc`allocation, and after the task finishes executing it needs to be manually`free`. If Proxy task creation fails,`fail`branch will release these arrays.

[FACT:src/rma/rma.cc:302-308]

```cpp
exit:
  return ret;
fail:
  free(peersProxy);
  free(nsignalsProxy);
  free(signalIdxsProxy);
  goto exit;
```

**Pitfall 3: Cross-context batching of Put/Signal tasks.**In the Put/Signal branch, NCCL pulls put/signal tasks from all contexts into the same plan, but stops when it encounters a WaitSignal.

[FACT:src/rma/rma.cc:279-295]

```cpp
for (int c = 0; c config.numRmaCtx; c++) {
  struct ncclIntruQueue* q = &planner->rmaTaskQueues[c];
  while (!ncclIntruQueueEmpty(q)) {
    struct ncclTaskRma* task = ncclIntruQueueHead(q);
    if (!isRmaPutOrSignal(task->func)) break;
    ncclIntruQueueDequeue(q);
    ...
  }
}
```

The intent of this design is: a single kernel launch covers put/signal for all contexts, reducing launch overhead. But each context's queue only consumes up to the first WaitSignal, ensuring per-context FIFO ordering. If the upper layer alternates put and waitSignal calls within the same context, the batching effect is greatly diminished—this is a pattern to watch out for when using RMA.

---

# II. GPU Direct Network: How GIN Lets Kernels Bypass the Host Proxy

## Intuitive Model

Traditional NCCL network communication is like mailing a letter: the GPU kernel puts data into a buffer, the host proxy thread hands the data to the NIC, and the NIC sends it out. GIN, on the other hand, lets the GPU kernel drop the letter directly into the recipient's mailbox—the kernel writes directly to the NIC's send queue, and the NIC reads directly from GPU memory.

Without GIN, every network communication must go through host memory as an intermediary, adding at least one PCIe round-trip of latency. For fine-grained communication like MoE, this latency is fatal.

## Data Structures and Memory Layout

The core state of GIN is`ncclGinState`, which manages multiple backends and multiple DevComms. Let's first look at the backend version compatibility table.

[FACT:src/gin/gin_host.cc:27-33]

```cpp
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5), NCCL_VERSION(2, 32, 0)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
constexpr int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0), NCCL_VERSION(2, 32, 0)};
```

The index of these arrays is the backend version number, and the value is the minimum compatible NCCL version. For example,`proxyBackendMinVersions[3]`corresponds to backend version 3, requiring NCCL at least 2.32.0. This design allows NCCL to select the appropriate backend version at runtime based on the device code version, rather than binding at compile time.

> **[Design Inference & Architectural Trade-offs]**
> The motivation behind this version compatibility table design is: the GIN backend (NIC driver, firmware) and the NCCL library evolve at different paces. If version requirements were hardcoded, an upgrade on either side would cause incompatibility. Using arrays for version mapping allows dynamic selection at runtime, maintaining backward compatibility with older backends.

`ncclGinStateDevComm`is the GIN state for each DevComm, containing`contextCount`、`backendIndex`、`ginCtx[]`、`devHandles[]`and other fields. It is chained into a linked list attached to`ginState->devComms`.

## Step-by-Step Walkthrough: Establishing a GIN Connection

Let's walk through a scenario: rank 0 initializes the communication domain and needs to establish a GIN connection.

**Step 1: Check whether GIN is enabled and supported.**

[FACT:src/gin/gin_host.cc:96-107]

```cpp
if (ginState->connected) return ncclSuccess;
if (ncclParamGinEnable() == 0) {
  WARN("GIN is disabled.");
  return ncclInternalError;
}
if (!ginState->supported) {
  WARN("GIN not supported.");
  return ncclInvalidUsage;
}
```

`ncclParamGinEnable()`reads the environment variable`NCCL_GIN_ENABLE`, defaulting to 1. If the user explicitly disables it, return an error directly.

**Step 2: Check symmetric memory support.**

[FACT:src/gin/gin_host.cc:111-114]

```cpp
if (!comm->symmetricSupport) {
  WARN("Communicator does not support symmetric memory!");
  return ncclInternalError;
}
```

GIN relies on symmetric memory—because the GPU kernel needs to know the virtual address of the peer's buffer, and only symmetric memory can guarantee address consistency.

**Step 3: Get the local GIN device list.**

[FACT:src/gin/gin_host.cc:116-122]

```cpp
int nLocalGinDevs;
int localGinDevs[NCCL_TOPO_MAX_NODES];
NCCLCHECK(ncclTopoGetLocalGinDevs(comm, localGinDevs, &nLocalGinDevs));
if (nLocalGinDevs > NCCL_GIN_MAX_CONNECTIONS) {
  ATTN("Found %d local devices, but GIN supports at most %d connections. Using the first %d connections.",
       nLocalGinDevs, NCCL_GIN_MAX_CONNECTIONS, NCCL_GIN_MAX_CONNECTIONS);
}
```

`ncclTopoGetLocalGinDevs`finds all NICs that support GIN from the topology graph. If it exceeds`NCCL_GIN_MAX_CONNECTIONS`, only take the first few and print a warning.

**Step 4: Compute the GIN team.**

[FACT:src/gin/gin_host.cc:138-149]

```cpp
ginTeam = ncclTeamWorld(comm);
if (ginState->ginConnectionType != NCCL_GIN_CONNECTION_FULL) {
  ginTeam = {
    .nRanks = comm->nRanks / comm->contiguousRanksPerHost,
    .rank = comm->rank / comm->contiguousRanksPerHost,
    .stride = comm->contiguousRanksPerHost,
  };
}
for (int r = 0; r numActiveBackends; backendIdx++) {
  backend = &ginState->backends[backendIdx];
  NCCLCHECKGOTO(backend->ncclGin->devices(&ndev), ret, fail);
  ...
  for (int commIdx = 0; commIdx ginCommCount; commIdx++) {
    NCCLCHECKGOTO(backend->ncclGin->listen(...), ret, fail);
    NCCLCHECKGOTO(backend->ncclGin->getProperties(...), ret, fail);
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret, fail);
    NCCLCHECKGOTO(backend->ncclGin->connect(...), ret, fail);
    NCCLCHECKGOTO(backend->ncclGin->closeListen(...), ret, fail);
  }
}
```

Each backend first calls`devices`to get the device count, then performs the listen→getProperties→allGather→connect→closeListen flow for each connection.`bootstrapAllGather`exchanges handles among all ranks, so that each rank knows the connection information of its peers.

## Concurrency Control and Hardware Interaction

The GIN progress thread is the core concurrency mechanism.

[FACT:src/gin/gin_host.cc:56-87]

```cpp
void* ncclGinProgress(struct ncclGinState* ginState, int threadIdx) {
  if (ncclOsCpuCount(ginState->cpuAffinity)) {
    ncclOsSetAffinity(ginState->cpuAffinity);
  }
  while (1) {
    if (ginState->proxyThreadStopSignal.load()) return NULL;
    if (ginState->writePending.load()) {
      std::this_thread::yield();
      continue;
    }
    {
      std::shared_lock rlock(ginState->devCommRwMutex);
      struct ncclGinStateDevComm* dc = ginState->devComms;
      while (dc) {
        struct ncclGinBackendState* backend = &ginState->backends[dc->backendIndex];
        for (int commIdx = threadIdx; commIdx ginCommCount; commIdx += ginState->proxyNthreads) {
          if (dc->devHandles[commIdx]->needsProxyProgress) {
            ncclResult_t ret = backend->ncclGin->ginProgress(dc->ginCtx[commIdx]);
            if (ret != ncclSuccess) {
              COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
              return NULL;
            }
          }
        }
        dc = dc->next;
      }
    }
    std::this_thread::yield();
  }
}
```

There are several key design points here:

1. **CPU Affinity**：`ncclOsSetAffinity`binds the progress thread to a specified CPU core, avoiding cache invalidation caused by thread migration.

2. **Write Lock Backoff**：`writePending`is an atomic flag. When the main thread needs to modify the`devComms`linked list, it sets this flag first, and the progress thread yields voluntarily upon seeing it, avoiding lock contention.

3. **Read-Write Lock**：`devCommRwMutex`is`shared_timed_mutex`. The progress thread holds the read lock to traverse the linked list, and the main thread holds the write lock to modify the linked list.

4. **Thread Division of Labor**: thread t is responsible for connections t, t+proxyNthreads, t+2*proxyNthreads, ..., achieving load balancing through a stride loop.

[FACT:src/gin/gin_host.cc:43-47]

```cpp
static void ginProgressWriteLock(struct ncclGinState* ginState) {
  ginState->writePending.store(true);
  ginState->devCommRwMutex.lock();
}
static void ginProgressWriteUnlock(struct ncclGinState* ginState) {
  ginState->devCommRwMutex.unlock();
  ginState->writePending.store(false);
}
```

This write lock implementation assumes there is only one writer (the main thread), so no additional mutex is needed.`writePending`sets the flag first before acquiring the lock, ensuring the progress thread can see the write intent before acquiring the lock and voluntarily back off.

## Production Pitfall Guide

**Pitfall 1: Mismatched GIN connection counts causing AllGather deadlock.**Each rank's`ginCommCount`may differ (depending on the number of local NICs), and NCCL takes the minimum across all ranks via`bootstrapAllGather`.

[FACT:src/gin/gin_host.cc:176-180]

```cpp
ginCommCountHandles[comm->rank] = backend->ginCommCount;
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
for (int r = 0; r nRanks; r++) {
  backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
}
```

If a rank has fewer NICs than other ranks, all ranks drop to the minimum. This guarantees connection symmetry but wastes NIC resources.

**Pitfall 2: proxyNthreads exceeding ginCommCount causes thread spinning.**If the user sets`NCCL_GIN_PROXY_NTHREADS`greater than`ginCommCount`the excess threads will spin in the stride loop.

[FACT:src/gin/gin_host.cc:181-183]

```cpp
// After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
// on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
// Extra threads simply idle in the stride loop; no correctness issue.
```

This is not a correctness issue, but it wastes CPU resources. The way to troubleshoot is to check whether`NCCL_GIN_PROXY_NTHREADS`is greater than the actual number of NICs.

**Pitfall 3: race condition when releasing DevComm.** `ncclGinDevCommFree`First remove DevComm from the linked list, then destroy the context.

[FACT:src/gin/gin_host.cc:464-475]

```cpp
ginProgressWriteLock(ginState);
if (prevDc) prevDc->next = dc->next;
else ginState->devComms = dc->next;
ginProgressWriteUnlock(ginState);
struct ncclGinBackendState* backend = &ginState->backends[dc->backendIndex];
for (int commIdx = 0; commIdx ginCommCount; commIdx++) {
  NCCLCHECK(backend->ncclGin->destroyContext(dc->ginCtx[commIdx]));
}
```

After removal, the progress thread can no longer see this DevComm, so destroying the context is safe. However, if there are in-flight network operations during destruction, it may lead to undefined behavior—this is what must be ensured when using GIN: before releasing DevComm, all operations must be confirmed complete.

---

# III. Symmetric memory kernel: from "registered buffers" to "unified address space"

## Intuitive model

Traditional NCCL buffers are "registration-based": each rank registers its own buffer, and addresses are exchanged via handles during communication. Symmetric memory, by contrast, is a "unified address space": all ranks agree on the same set of virtual addresses; address A on rank 0 and address A on rank 1 point to their respective physical memory, but the same address can be used in code to access them.

This is like everyone agreeing that "row 3, seat 5" refers to the same location in each person's home, so when looking for something you don't have to first ask "where is row 3, seat 5 in your home?"

Without symmetric memory, every kernel would have to resolve the peer address first, increasing instruction overhead and register pressure.

## Data structures and memory layout

The core of the symmetric memory kernel is the kernel mask—a bitmap that marks which kernels are available in the current communication domain.

[FACT:src/sym_kernels.cc:17-63]

```cpp
constexpr uint32_t kernelMask_STMC =
  1  **[Design Inference & Architectural Trade-offs]**
> The advantage of this bitmap design is that bitwise operations can quickly filter available kernels. For example,`kmask &= ~kernelMask_STMC`a single line can disable all STMC kernels without traversing the list.

## Step-by-Step Walkthrough: one kernel mask computation

Let's plug in a scenario: rank 0 wants to execute AllReduce, the data type is float16, the message size is 1MB, the communication domain has 8 ranks, and all are interconnected via NVLink.

**Step 1: Get the base mask corresponding to the operation.**

[FACT:src/sym_kernels.cc:304-306]

```cpp
uint32_t kmask = kernelMask_coll(coll);
```

`kernelMask_coll(ncclFuncAllReduce)`returns`kernelMask_AR`containing 5 AllReduce kernels.

**Step 2: Check STMC and LDMC availability.**

[FACT:src/sym_kernels.cc:308-334]

```cpp
bool hasSTMC = comm->symkState.hasLsaMultimem;
bool hasLDMC = false;
if (comm->symkState.hasLsaMultimem) {
  switch (ty) {
  case ncclFloat16:
  case ncclBfloat16:
    hasLDMC = red == ncclDevSum || red == ncclDevMinMax || red == ncclDevSumPostDiv;
    break;
  ...
  }
}
if (!hasSTMC) kmask &= ~kernelMask_STMC;
if (!hasLDMC) kmask &= ~kernelMask_LDMC;
```

`hasLsaMultimem`is computed in`ncclSymkInitOnce`requiring NVLS symmetric multicast to be available and the LSA team to have more than 2 ranks. float16 supports LDMC, so if`hasLsaMultimem`is true, the LDMC kernel is retained.

**Step 3: Check the message size limit.**

[FACT:src/sym_kernels.cc:336-342]

```cpp
size_t nBytes = alignUp(nElts * ncclTypeSize(ty), NCCL_SYM_KERNEL_CELL_SIZE);
size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks) * nBytes;
if (nBusBytes >= (size_t(2) = 32 * (size_t(2) nRanks;
kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
```

If the LSA team covers all ranks, GIN is not needed; otherwise, only the GIN kernel is retained.

## Concurrency control and hardware interaction

Initialization of the symmetric memory kernel involves DevComm creation and resource allocation.

[FACT:src/sym_kernels.cc:185-264]

```cpp
ncclResult_t ncclSymkInitOnce(struct ncclComm* comm) {
  NCCLCHECK(ncclDevrInitOnce(comm));
  struct ncclSymkState* symk = &comm->symkState;
  if (!symk->initialized) {
    symk->initialized = true;
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    symk->hasLsaMultimem = ncclNvlsSymmetricMultimemEnabled(comm) && ncclTeamLsa(comm).nRanks > 2 && !comm->p2pCrossClique;
    reqs.lsaMultimem = symk->hasLsaMultimem;
    reqs.lsaBarrierCount = ncclSymkMaxBlocks;
    ...
    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &symk->kcomm.devComm, /*isInternal=*/true, /*deviceCodeVersion=*/NCCL_VERSION_CODE));
  }
  return ncclSuccess;
}
```

The key here is`ncclDevrCommCreateInternal`which creates an internal DevComm containing resources such as LSA multicast, GIN inbox/outbox, and signals.`reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL`specifies that GIN uses rail connection mode.

[FACT:src/sym_kernels.cc:257-261]

```cpp
symk->kcomm.workStarted = comm->profiler.symWorkStarted;
symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
symk->kcomm.workPhases = comm->profiler.symWorkPhases;
```

The symmetric memory kernel uses an independent profiler buffer to avoid interleaving with the workCounter of regular kernels.

## Production pitfall avoidance guide

**Pitfall 1: SMEM requirements of the TMA kernel.**TMA requires about 8KB of SMEM scratch per warp, so 16 warps means 128KB.

[FACT:src/sym_kernels.cc:135-142]

```cpp
bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  if (comm->maxSharedMemOptin minCompCap >= 100 && ncclParamSymTmaEnable();
}
```

If the GPU's SMEM capacity is insufficient (such as in a MIG instance), the TMA kernel will be disabled. The way to troubleshoot is to check whether`maxSharedMemOptin`is less than`ncclTmaShmemScratchWarpSize() * 16`。

**Pitfall 2: boundaries of the GIN chunk size.**The chunk size of the ReduceScatter GIN kernel has upper and lower limits.

[FACT:src/sym_kernels.cc:148-153]

```cpp
static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128  0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}
```

If the user sets`NCCL_SYM_RS_GIN_CHUNK_SIZE`exceeding 1GB, it will be truncated to 1GB; if it is less than 128 bytes, it will be raised to 128 bytes. The final value will also be rounded down to a power of 2.

**Pitfall 3: Symmetric memory registration type mismatch.** `ncclGetSymRegType`Based on the flags of sendWin and recvWin,`NCCL_WIN_COLL_SYMMETRIC`determine the registration type.

[FACT:src/sym_kernels.cc:395-412]

```cpp
if (!isSendSymmReg && !isRecvSymmReg) {
  *winRegType = ncclSymSendNonregRecvNonreg;
} else if (isSendSymmReg && !isRecvSymmReg) {
  *winRegType = ncclSymSendRegRecvNonreg;
} else if (!isSendSymmReg && isRecvSymmReg) {
  *winRegType = ncclSymSendNonregRecvReg;
} else if (isSendSymmReg && isRecvSymmReg) {
  *winRegType = ncclSymSendRegRecvReg;
}
```

If the registration types of send and recv are inconsistent, the kernel needs to take different code paths. This affects performance but does not cause errors.

---

# IV. Team Abstraction and Versioned DevComm: Evolving Infrastructure

## Intuitive Model

The Team abstraction is like "grouping": the world team is the whole class, the LSA team is deskmates, and the Rail team is seats in the same column. Different communication patterns require different grouping perspectives.

Versioned DevComm is like a "translator": different versions of device code speak different "dialects," and the DevComm compatibility layer handles translation so old and new code can understand each other.

Without the Team abstraction, every kernel would have to compute rank mappings itself; without versioned DevComm, any ABI change would force all device code to be recompiled.

## Data Structures and Memory Layout

A Team is a simple triple:`nRanks`、`rank`、`stride`。

[FACT:src/nccl_device/core.cc:13-19]

```cpp
ncclTeam_t ncclTeamWorld(ncclComm_t comm) {
  ncclTeam_t ans;
  ans.nRanks = comm->nRanks;
  ans.rank = comm->rank;
  ans.stride = 1;
  return ans;
}
```

The world team's stride is 1 because all ranks are arranged consecutively.

[FACT:src/nccl_device/core.cc:70-79]

```cpp
ncclTeam_t ncclTeamRail(ncclComm_t comm) {
  if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
  ncclTeam_t ans;
  ans.nRanks = comm->nRanks / comm->devrState.lsaSize;
  ans.rank = comm->rank / comm->devrState.lsaSize;
  ans.stride = comm->devrState.lsaSize;
  return ans;
}
```

The Rail team's stride is`lsaSize`, because ranks on each rail are separated by the size of one LSA team.

The core of versioned DevComm is the`ncclDevCommCompat`structure.

[FACT:src/devcomm/devcomm_v23100.cc:10-17]

```cpp
struct ncclDevCommCompat ncclDevCommCompat_v23100 = {
  NCCL_VERSION(2, 31, 0), // minVersion
  NCCL_VERSION_CODE, // maxVersion
  nullptr,           // commPropertiesFilter
  nullptr,           // devCommRequirementsFilter
  nullptr,           // devCommCopyNewToOld
  nullptr,           // devCommCopyOldToNew
};
```

This structure defines the compatibility rules for version 2.31.0.`minVersion`and`maxVersion`define the applicable version range, and the following four function pointers define attribute filtering and structure conversion logic. If all are nullptr, it means this version has no special compatibility requirements.

## Step-by-Step Walkthrough: A Team Conversion

Let's use a scenario: rank 5 in an 8-rank communication domain, with an LSA team size of 4. We want to compute rank 5's rank in the Rail team.

**Step 1: Initialize DevR state.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
```

`ncclDevrInitOnce`Computes derived information such as the LSA team and CFT team. If it fails, returns an empty team.

**Step 2: Compute Rail team parameters.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
ncclTeam_t ans;
ans.nRanks = comm->nRanks / comm->devrState.lsaSize;  // 8 / 4 = 2
ans.rank = comm->rank / comm->devrState.lsaSize;       // 5 / 4 = 1
ans.stride = comm->devrState.lsaSize;                  // 4
```

Rank 5's rank in the Rail team is 1, the team has 2 ranks, and the stride is 4.

**Step 3: Convert back to world rank.**

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

If you want to convert Rail rank 0 to a world rank:`5 + (0 - 1) * 4 = 1`. Verification: rank 1 and rank 5 are on the same rail (separated by 4).

## Concurrency Control and Hardware Interaction

The Team abstraction itself is stateless and requires no concurrency control. But`ncclDevrInitOnce`is lazily loaded, and all derived information is computed on the first call.

[FACT:src/nccl_device/core.cc:22-33]

```cpp
ncclTeam_t ncclTeamLsa(ncclComm_t comm) {
  if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
  ncclTeam_t ans;
  ans.nRanks = comm->devrState.lsaSize;
  ans.rank = comm->devrState.lsaSelf;
  ans.stride = 1;
  return ans;
}
```

The comment says "Ignoring errors since if it fails ncclDevrInitOnce will try again" — if initialization fails, it returns an empty team, and the next call will retry.

## Production Pitfall Guide

**Pitfall 1: Stride assumptions in Team conversion.** `ncclTeamRankToWorld`assumes that ranks within a team form an arithmetic sequence.

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

If the team is not an arithmetic sequence (for example, a custom arbitrary grouping), this function will compute incorrectly. NCCL currently only supports regular teams.

**Pitfall 2: Null pointers in versioned DevComm.** `ncclDevCommCompat_v23100`All function pointers in are nullptr, indicating there is no special compatibility logic. If a future version requires conversion, these functions must be implemented; otherwise, old and new code cannot interoperate.

**Pitfall 3: Hierarchy modes of the CFT team.** `ncclTeamCft`supports three modes: FLAT, HIER_MULTIMEM, and HIER_LSA.

[FACT:src/nccl_device/core.cc:36-55]

```cpp
if (mode == NCCL_CFT_TEAM_FLAT) return flatTeam;
int innerSize;
if (mode == NCCL_CFT_TEAM_HIER_MULTIMEM) {
  innerSize = comm->devrState.cftMcSize;
} else if (mode == NCCL_CFT_TEAM_HIER_LSA) {
  innerSize = comm->devrState.lsaSize;
} else {
  return ncclTeam_t{};
}
return ncclTeamOuterFactor(flatTeam, innerSize);
```

If an invalid mode is passed in, it returns an empty team. When using a CFT team, you need to ensure the mode is correct.

---

# Design Reflections

**Why does NCCL support three evolution paths simultaneously: RMA, GIN, and symmetric memory?**

> **[Design Inference & Architectural Trade-offs]**
> These three paths solve problems at different levels:

- **RMA**solves the problem of "fixed communication patterns" — allowing upper layers to compose primitives and implement arbitrary communication patterns.
- **GIN**solves the problem of "high network latency" — allowing the GPU to directly drive the NIC, bypassing the host proxy.
- **Symmetric memory**solves the problem of "address resolution overhead" — allowing the kernel to directly access peer memory using a unified address.

They are not substitutes but complements. RMA can use GIN as the underlying transport, and GIN relies on symmetric memory to provide address consistency. Together, the three form the infrastructure of a "programmable communication engine."

**What is the design philosophy of versioned DevComm?**

> **[Design Inference & Architectural Trade-offs]**
> The core idea of versioned DevComm is "stable ABI, evolving API." Device code (kernels) is compiled and embedded in binaries and cannot be recompiled as the NCCL library upgrades. Therefore, NCCL must ensure that old device code can run on the new library.`ncclDevCommCompat`The structure is the entry point of the compatibility layer: the new library selects the appropriate compatibility rules based on the device code version and performs structure conversion when necessary.

---

# Chapter Summary

In this chapter, starting from the traces of evolution in the source code, we analyzed the three forces driving NCCL from a collective communication library toward a programmable communication engine:

1. **RMA**（`src/rma/rma.cc`): Through the combination of Put/Signal/WaitSignal primitives, upper layers can implement arbitrary communication patterns. The core design splits tasks into two parallel execution paths, CE and Proxy, based on LSA reachability.

2. **GIN**（`src/gin/gin_host.cc`): Through GPU direct network transmission, bypassing the host proxy. The core design includes multi-backend management, version compatibility tables, and a progress thread pool.

3. **Symmetric memory kernel**（`src/sym_kernels.cc`): Through a unified address space, eliminating address resolution overhead. The core design includes kernel mask bitmaps and TMA/GIN hardware acceleration.

4. **Team abstraction and versioned DevComm**（`src/nccl_device/core.cc`、`src/devcomm/devcomm_v23100.cc`): Providing infrastructure for evolution. Team provides a grouping perspective, and versioned DevComm provides ABI compatibility.

The impact of these changes on upper-layer frameworks is profound: PyTorch's ProcessGroup can directly call RMA primitives to implement custom communication patterns; Megatron's expert parallelism can leverage GIN to reduce all-to-all latency; symmetric memory makes kernel code more concise.

# Chapter Review and Self-Test

Q1: If the`scheduleRmaTasksToPlan`LSA reachability check in the WaitSignal branch is removed, and all peers go through the Proxy path, what would be the consequences? In what scenarios would this trigger a performance disaster?

**Reference Analysis**：

The LSA reachability check is in[FACT:src/rma/rma.cc:187-204], which divides peers into two groups: CE and Proxy. If this check is removed, all peers go through the Proxy path,`nRmaTasksCe`is always 0.

The consequence is: the CE path is completely unused, and all WaitSignal operations poll the network through host proxy threads. For peers within LSA range (same-machine NVLink interconnect), which could originally use GPU copy engines for asynchronous waiting, now become host thread polling, with latency rising from microseconds to milliseconds.

Performance disaster scenario: In MoE training, each token needs to wait for signals from multiple experts. If all signals go through Proxy, the host thread becomes the bottleneck, and the GPU spends a large amount of time waiting for host polling. On an 8-GPU all-NVLink machine, this degradation is especially pronounced—all communication that could originally go through CE now crowds onto the host.

Troubleshooting method: Check the`scheduleRmaTasksToPlan`INFO logs. If`nRmaTasksCe`is always 0 while`nRmaTasksProxy`is very large, it indicates a problem with the LSA check.

Q2：`ncclGinProgress`In`writePending`flag and`devCommRwMutex`read-write lock coordination, if the`writePending`check is removed and only the read-write lock is kept, what problems would arise?

**Reference Analysis**：

`writePending`The check is in[FACT:src/gin/gin_host.cc:63-66], which makes the progress thread actively yield when the main thread wants to write. If this check is removed, the progress thread will directly attempt to acquire the read lock.

The problem is:`std::shared_timed_mutex`'s read lock is shared, and multiple progress threads can hold it simultaneously. If the main thread wants to acquire the write lock, it must wait for all read locks to be released. Under high load, progress threads frequently acquire read locks, and the main thread may be unable to acquire the write lock for a long time, causing`ncclGinDevCommSetup`or`ncclGinDevCommFree`to block.

More seriously: if the main thread first sets`ginProgressWriteLock`in`writePending`before acquiring the lock, and the progress thread does not check`writePending`, then the progress thread may still acquire the read lock after the main thread sets the flag, causing unpredictable wait times for the main thread.

`writePending`The purpose of is a "soft notification": telling progress threads "I'm about to write, please yield." This is more efficient than relying solely on lock fairness, because progress threads can actively yield rather than block on the lock.

Q3：`ncclSymkMask`In`nBusBytes >= 32 * (size_t(2) << 30)`, if all kernels are disabled when`kmask = 0`), at this point`ncclSymkAvailable`returns false, what path will NCCL fall back to? What performance impact does this fallback path have?

**Reference Analysis**：

`kmask = 0`In[FACT:src/sym_kernels.cc:342], at this point`ncclSymkAvailable`returns false ([FACT:src/sym_kernels.cc:354-361]）。

The fallback path is: NCCL will use traditional collective communication kernels (non-symmetric memory kernels). These kernels access peer memory through registered buffers, requiring address resolution first, with higher instruction overhead.

Performance impact: For very large messages (exceeding 64GB bus bytes), the address resolution overhead of traditional kernels is a small proportion, because data transfer itself dominates. But in boundary cases (just exceeding 64GB), traditional kernels may be 10-20% slower than symmetric memory kernels.

The root cause of this limitation is: symmetric memory kernels use 32-bit integers to track unrolled loop chunks, with each chunk being at least 32 bytes, so the maximum addressable range is 32 * 2^31 = 64GB. Exceeding this range causes integer overflow.

In actual production, scenarios where a single collective communication exceeds 64GB are rare (usually all-reduce after gradient accumulation), but not impossible. If such a scenario is encountered, consider sharded communication or using traditional kernels.

---

# Chapter Transition

In this chapter, we have seen NCCL moving from "fixed collective operations" toward a "programmable communication engine": RMA provides primitive composition, GIN provides GPU direct transmission, symmetric memory provides a unified address space, and Team and versioned DevComm provide infrastructure.

These evolutions are not isolated; they collectively point toward one goal:**Enable upper-layer frameworks to implement custom communication patterns with lower latency and greater flexibility**. For frameworks like PyTorch and Megatron, this means they can directly build complex communication patterns such as MoE all-to-all, pipeline parallelism, and expert parallelism on top of NCCL, without needing to bypass NCCL and implement the network layer themselves.

The next chapter is the final chapter of the book. We will walk through the complete path of a single AllReduce once again—starting from the`ncclAllReduce`call, going through task enqueue, algorithm selection, kernel launch, proxy progression, network transmission, until the result is returned. This review will connect the knowledge points from the previous 24 chapters into a complete cognitive map.

At this point, we have seen the three main lines of NCCL's evolution from fixed collective operations to a programmable communication engine: RMA primitive composition, GPU direct network transmission, symmetric memory model, and the team abstraction and versioned DevComm that support them. These mechanisms together point toward a more flexible communication future that is closer to hardware capabilities. However, no matter how the architecture evolves, the complete path of a single AllReduce remains the cornerstone of understanding NCCL. In the next chapter, we will not introduce new code, but instead re-narrate the end-to-end flow from Chapter 3 to Chapter 10—from the ncclAllReduce call, to communicator establishment, topology search, algorithm selection, task enqueue, kernel launch, device-side primitive execution, and result write-back. You will reassemble the mechanisms scattered across chapters into a complete mental model, and obtain an index of "which chapter to check when encountering a problem."
