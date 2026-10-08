# Chapter 8: Kernel Launch and Device-Side Execution: From Host-Side Invocation to GPU Thread Block Startup

In the previous chapter, we broke down how tasks are partitioned across multiple channels, how kernel launch parameters are generated, and the mechanisms for batch submission and dependency ordering under group semantics. Now, the launch plan is ready, but it is still only a host-side data structure. The core question this chapter answers is:`ncclKernelPlan`How does it become a grid actually running on the GPU? We will follow the`ncclLaunchKernel`call chain to see how parameters are packed into kernel args, how kernel variants are selected,`cuLaunchKernelEx`how it is invoked, and how the device-side`ncclKernelMain`reads the work description from shared memory and dispatches it to the concrete implementation.

# From Plan to Grid: A Panorama of the Launch Path

Before diving into details, let us first build an overall mental model. Think of`ncclKernelPlan`as a "construction blueprint": it records how many channels (how many blocks) to launch this time, how many threads per block, which work items to execute, and which kernel function to use. And`ncclLaunchKernel`is the action of "the construction crew entering the site"—it translates the information on the blueprint into`CUlaunchConfig`that the CUDA driver can understand, and then calls`cuLaunchKernelEx`to actually launch the grid onto the GPU.

Without this layer, all host-side scheduling (the previous chapter's channel partitioning, batch organization, and proxy op ordering) would be nothing but talk on paper; no kernel would run on the GPU, and communication would never happen. This is the final link in the end-to-end backbone, and also the boundary between host and device.

The entire launch path can be summarized in three stages:

1. **Parameter preparation**（`finishPlan` + `uploadWork`): organize the work structs, batch descriptors, and kernel args into a contiguous block of memory, deciding whether to place them in kernel parameters, in the FIFO, or in a persistent buffer.

2. **Kernel launch**（`ncclLaunchKernel`): compute grid/block dimensions, assemble launch attributes (CGA cluster, mem sync domain, launch completion event), and call`cuLaunchKernelEx`。

3. **Device-side entry**（`ncclKernelMain`): each block determines its own channelId based on`blockIdx.x`loads the work batch from args or the FIFO into shared memory, and then dispatches through`ncclDevFuncTable`to the concrete algorithm/protocol implementation.

The figure below shows the complete control flow from plan to grid, including the key branch decisions:

```mermaid
flowchart TD
    plan["ncclKernelPlanchannelMask / workBytes / kernelFn"]
    finish["finishPlan()决定 workStorageType"]
    check_budget{"sizeof(args)+batchBytes+workBytes work 直接放 kernel 参数"]
    fifo_type["workStorageType = Fifo/Persistentwork 放外部缓冲区"]
    upload["uploadWork()拷贝 work 到目标缓冲区"]
    launch["ncclLaunchKernel()组装 CUlaunchConfig"]
    check_cluster{"compCap >= 90且 clusterSize > 0?"}
    add_cluster["添加 CLUSTER_DIMENSION+ SPREAD 调度策略"]
    no_cluster["不添加 cluster 属性"]
    check_event{"userKernelEvent且 driver >= 12030?"}
    add_event["添加 LAUNCH_COMPLETION_EVENT"]
    no_event["无 completion event"]
    cu_launch["cuLaunchKernelEx()发射 grid 到 GPU"]

    plan --> finish --> check_budget
    check_budget -->|是| args_type
    check_budget -->|否| fifo_type
    args_type --> upload
    fifo_type --> upload
    upload --> launch --> check_cluster
    check_cluster -->|是| add_cluster
    check_cluster -->|否| no_cluster
    add_cluster --> check_event
    no_cluster --> check_event
    check_event -->|是| add_event
    check_event -->|否| no_event
    add_event --> cu_launch
    no_event --> cu_launch
```

This figure anchors the three core functions of this chapter:`finishPlan`、`uploadWork`、`ncclLaunchKernel`. Next, we will break them down one by one.

# Parameter Preparation: How the Work Struct Finds Its Place

## Intuitive model

`finishPlan`'s role is similar to the "packer" at a courier sorting center. It faces a pile of scattered work structs (one for each collective or p2p operation) and needs to decide: should these work items be stuffed into the "carry-on backpack" of kernel parameters, placed on the "conveyor belt" of the FIFO, or put into the "warehouse" of a persistent buffer?

If this decision is made incorrectly—for example, if the work is too large to fit into kernel parameters but is forced in anyway—the kernel launch will fail outright. If the work is placed in the wrong location, the device side will read garbage data, and the communication result will be completely wrong.

## Data Structures and Memory Layout

First look at`ncclDevKernelArgs`'s structure; it is the "envelope" between host and device:

[FACT:src/include/device.h:514-522]

```c
struct alignas(16) ncclDevKernelArgs {
  struct ncclKernelComm* comm;      // 指向设备侧通信器元数据
  uint64_t channelMask;             // 哪些 channel 有工作
  enum ncclDevWorkStorageType workStorageType;  // work 存在哪里
  uint32_t workMask;                // FIFO 环形缓冲区的掩码
  void* workBuf;                    // work 缓冲区指针
  // struct ncclDevWorkBatch batches[];  // 紧随其后的是 batch 数组
};
```

This struct has only 5 fields, but each field carries critical information.`channelMask`is a 64-bit mask, with each bit corresponding to a channel; the device side computes`__popcll`to determine`blockIdx.x`'s corresponding channelId.`workStorageType`determines where the device side reads work from:`Args`means the work is in the kernel parameters,`Fifo`means it is in the ring buffer,`Persistent`means it is in the persistent buffer.

`ncclDevWorkBatch`is the batch descriptor, which tells the device side "where the work for this channel is and how many there are":

[FACT:src/include/device.h:400-421]

```c
struct alignas(16) ncclDevWorkBatch {
  union {
    struct {
      uint32_t nextJump:14, nextExtends:1;
      uint32_t workType:2, funcId : NCCL_DEV_WORK_BATCH_FUNC_ID_BITS, func : NCCL_DEV_WORK_BATCH_FUNC_BITS;
    };
    uint32_t flags;
  };
  uint32_t offsetBase;    // work 在 FIFO 中的起始偏移
  uint64_t offsetBitset;  // 哪些 work 属于这个 channel
};
```

`offsetBitset`is a 64-bit mask, with each bit corresponding to a work struct. The device side uses`__popc`and`fns`(find n-th set) instructions to locate the offset of each work.`nextJump`and`nextExtends`are used to chain multiple batches together—when there is too much work to fit in one batch, an "extended batch" is created.

## Step-by-Step Walkthrough

Now consider a concrete scenario: one AllReduce is split across 4 channels, each channel has 2 work structs, for a total of 8 work items.

**Step 1:`finishPlan`determines the storage type.**

[FACT:src/enqueue/enqueue.cc:245-255]

```c
if (sizeof(ncclDevKernelArgs) + batchBytes + workBytes workArgsBytes) {
  plan->workStorageType = ncclDevWorkStorageTypeArgs;
}
plan->kernelArgsSize = sizeof(struct ncclDevKernelArgs) + batchBytes;
plan->kernelArgsSize += (plan->workStorageType == ncclDevWorkStorageTypeArgs) ? workBytes : 0;
plan->kernelArgsSize = alignUp(plan->kernelArgsSize, 16);
plan->kernelArgs =
  (struct ncclDevKernelArgs*)ncclMemoryStackAlloc(&comm->memScoped, plan->kernelArgsSize, /*align=*/16);
plan->kernelArgs->comm = comm->devComm;
plan->kernelArgs->channelMask = plan->channelMask;
plan->kernelArgs->workStorageType = plan->workStorageType;
```

The key judgment here is: if`sizeof(ncclDevKernelArgs) + batchBytes + workBytes`can fit into`comm->workArgsBytes`(usually 4KB), then put the work directly into the kernel parameters. Otherwise, the work is placed into the FIFO or a persistent buffer, and only the batch descriptor is placed in the kernel parameters.

> **[Design Inference & Architectural Trade-offs]**
> Why prefer putting it in kernel parameters? Because kernel parameters are passed through constant memory in the CUDA driver, and when the device side reads them it uses the`ld.param`instruction, which is much faster than reading the FIFO from global memory. For small messages (small total work), this can significantly reduce latency.

**Step 2: Place batches into kernel args by rotating across channels.**

[FACT:src/enqueue/enqueue.cc:257-280]

```c
uint64_t hasBatchMask = plan->channelMask;
struct ncclDevWorkBatch* batchPrev[MAXCHANNELS] = {};
struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs + 1);
int batchIx = 0;
while (hasBatchMask != 0) {
  uint64_t tmpMask = hasBatchMask;
  do {
    int c = popFirstOneBit(&tmpMask);
    if (!ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
      struct ncclWorkBatchList* batchNode = ncclIntruQueueDequeue(&wipChannels[c].workBatchQueue);
      if (batchPrev[c] != nullptr) {
        batchPrev[c]->nextJump = int(&batchZero[batchIx] - batchPrev[c]);
      }
      batchPrev[c] = &batchZero[batchIx];
      batchZero[batchIx++] = batchNode->batch;
    }
    if (ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
      hasBatchMask ^= 1ull isSymColl || plan->isCeColl || plan->isRma) return ncclSuccess;
  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches * sizeof(struct ncclDevWorkBatch);
  void* fifoBufHost;
  uint32_t fifoCursor, fifoMask;
  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeArgs:
    plan->kernelArgs->workBuf = nullptr;
    fifoBufHost = (void*)plan->kernelArgs;
    fifoCursor = sizeof(ncclDevKernelArgs) + batchBytes;
    fifoMask = ~0u;
    break;
  case ncclDevWorkStorageTypeFifo:
    fifoBufHost = comm->workFifoBuf;
    fifoCursor = comm->workFifoProduced;
    fifoMask = comm->workFifoBytes - 1;
    NCCLCHECK(waitWorkFifoAvailable(comm, fifoCursor + workBytes));
    plan->kernelArgs->workBuf = comm->workFifoBufDev;
    break;
  // ...
  }
  plan->kernelArgs->workMask = fifoMask;
  // 修正 batch 的 offsetBase
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs + 1);
  for (int b = 0; b nWorkBatches; b++) {
    batchZero[b].offsetBase += fifoCursor;
  }
  // 拷贝 work 结构体
  struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue);
  while (workNode != nullptr) {
    char* dst = (char*)fifoBufHost;
    char* src = (char*)(workNode + 1);
    for (int n = workNode->size; n != 0; n -= 16) {
      memcpy(COMPILER_ASSUME_ALIGNED(dst + (fifoCursor & fifoMask), 16), COMPILER_ASSUME_ALIGNED(src, 16), 16);
      fifoCursor += 16;
      src += 16;
    }
    workNode = workNode->next;
  }
  // ...
}
```

There are several key points here:

1. **`fifoCursor`The semantics of**: for the`Args`type, it is an offset relative to the`kernelArgs`start address; for the`Fifo`type, it is an offset relative to the FIFO base address; for the`Persistent`type, it starts from 0.

2. **`offsetBase`Correction of**：`finishPlan`The batch's`offsetBase`in`uploadWork`is relative to the work start position of the plan (starting from 0).`Args`It needs to be converted into an offset relative to the actual storage location. For the`sizeof(ncclDevKernelArgs) + batchBytes`type, add`Fifo`; for the`comm->workFifoProduced`。

3. **type, add**16-byte aligned copy`alignas(16)`: work structs are all 16-byte aligned (`COMPILER_ASSUME_ALIGNED`), so copying is done in units of 16 bytes.

4. **tells the compiler that this address is 16-byte aligned, allowing the compiler to generate more efficient vectorized instructions.**FIFO wait`Fifo`: for the`waitWorkFifoAvailable`type,`comm->abortFlag`spins waiting for the FIFO to have enough space. This wait checks

## to avoid deadlock on abort.

> **[Design Inference & Architectural Trade-offs]**
> **[Design inference and architectural trade-offs]**Why are there three storage types?

- `Args`This is a trade-off between space and latency:
- `Fifo`: fastest (constant memory), but limited capacity (4KB). Suitable for small messages and a small amount of work.
- `Persistent`: large capacity (ring buffer), but device-side reads must go through global memory. Suitable for medium messages.`cudaMemcpy`: used for CUDA Graph capture scenarios. Because graph capture cannot perform

**, a persistent buffer must be preallocated, the work copied into it, and then the kernel reads from there.**Pitfall 1: FIFO overflow causing deadlock.`waitWorkFifoAvailable`If`abortFlag`does not check[FACT:src/enqueue/enqueue.cc:1333-1349], then when the FIFO is full and the consumer (GPU kernel) stops consuming for some reason, the host will spin forever. In the source code,

```c
if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
  return ncclInternalError;
}
```

**Copy`offsetBitset`Pitfall 2:** `offsetBitset`overflow.`1ull << (offset / workSize)`is 64-bit and supports at most 64 work items in one batch. If there are more than 64,`NCCL_MAX_DEV_WORK_BATCH_BYTES`will overflow. In the source code,`ncclDevWorkColl`limits the batch size (1024 bytes), and the smallest work struct is

**(about 80 bytes), so there are at most 12 work items, and it will not overflow.**Pitfall 3: Memory leak in Persistent mode.`uploadWork`In`Persistent`'s`fifoBufHost`branch,`ncclOsAlignedAlloc`is allocated through`uploadWork_cleanup_fn`and needs to be freed in`cudaMemcpyAsync`. If`fail`fails, the`cleanup`label checks whether`fifoBufHost`is null, and if it is null, directly frees[FACT:src/enqueue/enqueue.cc:1483-1485]. This error recovery chain can be seen in

# Kernel launch: from CUlaunchConfig to cuLaunchKernelEx

## Intuitive model

`ncclLaunchKernel`role is similar to a "rocket launch console." It receives a plan already loaded with fuel (work data), calculates the rocket's flight parameters (grid/block dimensions), sets various launch options (cluster, mem sync domain, completion event), and then presses the launch button (`cuLaunchKernelEx`）。

If something goes wrong at this stage—for example, if the grid dimensions are calculated incorrectly—the wrong number of blocks will be launched on the GPU, causing the work of some channels to never be executed and communication to hang.

## Data Structures and Memory Layout

`CUlaunchConfig`is the launch configuration struct of the CUDA driver API, and NCCL constructs it on the stack:

[FACT:src/enqueue/enqueue.cc:1916-1917]

```c
CUlaunchConfig launchConfig = {0};
CUlaunchAttribute launchAttrs[6] = {};
int attrs = 0;
```

`launchAttrs`is an array of at most 6 elements, each element being a`CUlaunchAttribute`. NCCL conditionally adds different attributes based on hardware capabilities and driver version:

- `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`: CGA cluster dimension (sm90+)
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`: cluster scheduling policy
- `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`: memory sync domain (CUDA 12.0+)
- `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`: launch completion event (CUDA 12.3+)
- `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`: programmatic stream serialization (sym kernel)
- `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING`: NVLink utilization-centric scheduling (CUDA 13.0+)

## Step-by-Step Walkthrough

**Step 1: Calculate grid and block dimensions.**

[FACT:src/enqueue/enqueue.cc:1889-1893]

```c
int nChannels = countOneBits(plan->channelMask);
void* sym = plan->kernelFn;
dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
```

`nChannels`is`channelMask`the number of set bits in , that is, how many blocks this plan needs to launch. Each block is responsible for one channel.`threadPerBlock`is in`scheduleCollTasksToPlan`computed via`plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE)`, taking the maximum among all tasks`nWarps * 32`。

`smem`is the dynamic shared memory size. For a normal kernel, it is`ncclShmemDynamicSize(comm->cudaArch)`, which is a compile-time constant depending on the architecture (sm70+ is`ncclShmemScratchWarpSize * (NCCL_MAX_NTHREADS / WARP_SIZE)`). For a sym kernel, it is`plan->kernelDynSmem`, because the shared memory requirements of a sym kernel may differ.

**Step 2: Assemble kernel arguments.**

[FACT:src/enqueue/enqueue.cc:1902-1903]

```c
void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
                 CU_LAUNCH_PARAM_END};
```

This is a way of passing arguments in the CUDA driver API:`CU_LAUNCH_PARAM_BUFFER_POINTER`tells the driver "the arguments are not passed one by one, but as one contiguous memory block,"`CU_LAUNCH_PARAM_BUFFER_SIZE`tells the driver the size of this block. The advantage of doing this is that NCCL can pass`ncclDevKernelArgs`and the following batch array all at once, without needing to pack each argument individually.

**Step 3: Add launch attributes.**

[FACT:src/enqueue/enqueue.cc:1929-1936]

```c
if (clusterSize) {
  if (grid.x % clusterSize) clusterSize = 1;
  launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
  launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
  launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
}
```

CGA (Cooperative Group Array) is a hardware feature introduced in sm90 that allows multiple blocks to be grouped into a cluster. Blocks within a cluster are guaranteed to be scheduled simultaneously onto a set of SMs and can access each other's shared memory. NCCL uses this feature to implement algorithms such as NVLS that require cross-block synchronization.

Note the`if (grid.x % clusterSize) clusterSize = 1;`protection: the cluster dimension must evenly divide the grid dimension, otherwise the driver will report an error. If`grid.x`is not divisible by`clusterSize`, it degrades to not using a cluster.

**Step 4: Add launch completion event.**

[FACT:src/enqueue/enqueue.cc:1944-1964]

```c
#if CUDART_VERSION >= 12030
enum ncclImplicitOrder implicitOrder;
NCCLCHECKGOTO(getImplicitOrder(&implicitOrder, comm, plan->persistent, driverVersion), ret, do_return);
if (implicitOrder == ncclImplicitOrderLaunch) {
  launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
  launchAttrs[attrs].value.launchCompletionEvent.event = comm->sharedRes->launchEvent;
  launchAttrs[attrs].value.launchCompletionEvent.flags = 0;
  attrs++;
  if (userKernelEvent) {
    NCCLCHECKGOTO(ncclUncapturedStreamPoolAcquire(&comm->sharedRes->uncapturedStreamPool, &relayStream), ret, do_return);
    relayUserLaunchCompletionEvent = true;
    userKernelEventArmed = true;
  }
} else if (userKernelEvent && driverVersion >= 12030) {
  launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
  launchAttrs[attrs].value.launchCompletionEvent.event = plan->launchCompletionEvent;
  launchAttrs[attrs].value.launchCompletionEvent.flags = 0;
  attrs++;
  userKernelEventArmed = true;
}
#endif
```

`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`is a feature introduced in CUDA 12.3: the driver records an event when the kernel actually starts executing (rather than when the host-side call returns). This is crucial for implementing "implicit order"—NCCL needs to ensure that multiple kernels execute in order, but does not want the host side to block and wait.

`getImplicitOrder`The logic of is: if the user has set`launchOrderImplicit`, and the driver version is new enough, use`ncclImplicitOrderLaunch`(ordering via launch event); otherwise use`ncclImplicitOrderSerial`(ordering via completion event, i.e., serial execution).

**Step 5: Call`cuLaunchKernelEx`。**

[FACT:src/enqueue/enqueue.cc:1978-1996]

```c
launchConfig.gridDimX = grid.x;
launchConfig.gridDimY = grid.y;
launchConfig.gridDimZ = grid.z;
launchConfig.blockDimX = block.x;
launchConfig.blockDimY = block.y;
launchConfig.blockDimZ = block.z;
launchConfig.sharedMemBytes = smem;
launchConfig.attrs = launchAttrs;
launchConfig.numAttrs = attrs;
launchConfig.hStream = launchStream;
if (userKernelEvent && !userKernelEventArmed) {
  WARN("CUDA launch-completion events require CUDA 12.3 or newer; recording the user event before launch");
  CUDACHECKGOTO(cudaEventRecord(plan->launchCompletionEvent, launchStream), ret, do_return);
}
CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
if (relayUserLaunchCompletionEvent) {
  CUDACHECKGOTO(cudaStreamWaitEvent(relayStream, comm->sharedRes->launchEvent, 0), ret, do_return);
  CUDACHECKGOTO(cudaEventRecord(plan->launchCompletionEvent, relayStream), ret, do_return);
}
```

`cuLaunchKernelEx`is a new API introduced in CUDA 12.0 that supports launch attributes. For older drivers (< 11.8), NCCL falls back to`cuLaunchKernel`：

[FACT:src/enqueue/enqueue.cc:1998-2007]

```c
} else {
  // Standard kernel launch
  if (userKernelEvent) {
    WARN("CUDA launch-completion events require CUDA 12.3 or newer; recording the user event before launch");
    CUDACHECKGOTO(cudaEventRecord(plan->launchCompletionEvent, launchStream), ret, do_return);
  }
  CUCHECKGOTO(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr,
                             extra),
              ret, do_return);
}
```

## Concurrency Control and Hardware Interaction

**The relay mechanism of the Launch completion event.**When using`ncclImplicitOrderLaunch`and the user has provided`launchCompletionEvent`, NCCL cannot pass the user's event directly to the driver, because the driver supports only one launch completion event. NCCL's approach is:

1. Pass`comm->sharedRes->launchEvent`to the driver.

2. Wait on`relayStream`for`launchEvent`。

3. Record the user's event on`relayStream`This way, the user's event is triggered after the kernel actually starts executing, rather than when the host-side call returns.

On sm90+, NCCL sets

**Mem Sync Domain。** [FACT:src/enqueue/enqueue.cc:1938-1942]to`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`. This is the memory sync domain mechanism introduced by the Hopper architecture, used to isolate memory barriers of different kernels and reduce unnecessary synchronization overhead.`cudaLaunchMemSyncDomainRemote`Production Pitfall Guide

## Pitfall 1: Cluster dimension not evenly dividing causes launch failure.

**If**is not divisible by`grid.x`, the driver returns`clusterSize`. The source code protects against this via`CUDA_ERROR_INVALID_VALUE`, but this also means the cluster feature is silently disabled. If the user expects the performance improvement brought by clusters, they need to check`if (grid.x % clusterSize) clusterSize = 1;`and`cgaClusterSize`the relationship between`nChannels`Pitfall 2: Driver version not meeting requirements causes the kernel to be unavailable.

**踩坑点 2：驱动版本不满足导致 kernel 不可用。** `ncclInitKernelsForDevice`checks the driver requirements of each kernel during initialization:

[FACT:src/enqueue/enqueue.cc:71-76]

```c
for (int k = 0; k channelMask & (1ull channelMask & ((1ull channels[ncclShmem.channelId];
    int bytes = sizeof(ncclDevChannel);
    static_assert(sizeof(ncclDevChannel) > 32) & (1u > 32) & ((1u > 32));
    __syncwarp();
    // ...
  }
}
```

is the most complex part. Its task is to copy the work struct pointed to by the batch descriptor from global memory (or kernel parameters) into the shared memory`fnsOfBitset`.`offsetBitset`Copy`fns`The core of this code is computing`fnsOfBitset[nWorksBelow]`。

: for the nth set bit in

[FACT:src/device/common.h:209-241]

```c
if (tid = %d && __CUDA_ARCH__ >= %d\n" % (cudart ,arch))
  out("/*%4d*/ %s,\n" % (index, sym))
  if (cudart, arch) != (0, 0):
    out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
  index += 1
out("nullptr};\n")
```

## Design Thinking and Production Pitfalls

**Why use`__grid_constant__`？** [FACT:src/device/common.h:19-24]

```c
#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif
```

`__grid_constant__`Tells the compiler this parameter is read-only and can be placed in constant memory. This way, device-side reads go through`ld.param`instructions, which is faster than reading from global memory. The comment mentions it breaks cuda-gdb, so it's only enabled on sm70+.

**Pitfall 1:`workStorage`overflow.** `workStorage`The size of is`ncclMaxDevWorkBatchBytes()`, sm90+ is 16KB. If`nWorks * workSize`exceeds this value, it will write out of bounds. In the source code,`NCCL_MAX_DEV_WORK_BATCH_BYTES`limits the batch size on the host side, but there is no additional check on the device side. If the host-side constraint is bypassed (e.g., by modifying environment variables), it will cause shared memory out-of-bounds.

**Pitfall 2:`__syncthreads()`The absence of causes data races.**After`loadWorkBatchToShmem`, there must be a`__syncthreads()`for all threads to see the complete`workStorage`. In the source code, at[FACT:src/device/common.h:479]there is`__syncthreads(); // publish ncclShmem`. If this synchronization is removed, some threads may start reading before`workStorage`has finished writing, resulting in reading garbage data.

**Pitfall 3: Timing of abort checks.** `while (ncclShmem.aborted == 0)`Only checks abort at the start of each batch. If a batch takes a long time to execute, the abort signal may take a long time to take effect. This is a design trade-off: more frequent checks add overhead, but respond faster.

# Kernel Variant Selection: How generate.py Generates the Kernel List

## Intuitive Model

`generate.py`'s role is similar to a "production line planner in a car factory." It faces a huge combinatorial space (7 set operations × 5 reduction operations × 12 data types × 7 algorithms × 3 protocols) and needs to decide: which combinations need dedicated kernels generated? Which can share a generic kernel?

If a kernel is generated for every combination, compilation time and binary size will explode. If only one generic kernel is generated, runtime will be slow due to function pointer calls and branch judgments.`generate.py`'s solution is "representative kernels": generate one kernel for each equivalence class, and dispatch at runtime through a function pointer table.

## Data Structures and Memory Layout

`generate.py`Generates three key files:

1. **`device_table.cu`**: device-side`ncclDevFuncTable`, mapping funcId to specific device functions.

2. **`host_table.cc`**: host-side`ncclDevKernelList`、`ncclDevKernelForFunc`、`ncclDevFuncRowToId`and other tables.

3. **Each`<coll>_<op>_<ty>.cu`**: specific kernel implementation.

## Step-by-Step Walkthrough

**Step 1: Enumerate all function rows.**

[FACT:src/device/generate.py:186-199]

```python
def enumerate_func_rows():
  yield ("SendRecv", None, None, None, None)
  for coll in ("AllGather", "Broadcast", "AllGatherV"):
    algos = algos_of_coll[coll]
    for algo in algos:
      for proto in all_protos:
        yield (coll, None, None, algo, proto)
  for coll in ("AllReduce", "Reduce", "ReduceScatter"):
    algos = algos_of_coll[coll]
    for redop in all_redops:
      for ty in all_tys:
        for algo in algos:
          for proto in all_protos:
            yield (coll, redop, ty, algo, proto)
```

This enumeration order must match`ncclDevFuncId()`'s calculation formula:

[FACT:src/include/device.h:646-706]

```c
inline int ncclDevFuncId(int coll, int devRedOp, int type, int algo, int proto) {
  constexpr int NumTypes = ncclNumTypes;
  int row;
  do {
    row = 0; // ncclDevFuncIndex_P2p
    if (coll == ncclFuncSendRecv) break;
    row += 1;
    // ...
  } while (false);
  return ncclDevFuncRowToId[row];
}
```

`ncclDevFuncId`What is calculated is the "row number," which is then mapped through`ncclDevFuncRowToId`to the "main function ID." The reason for this mapping is: many rows may map to the same main function (e.g., all`AllReduce Sum i32`rows map to`AllReduce Sum u32`'s main function).

**Step 2: Compute main functions and kernel functions.**

[FACT:src/device/generate.py:211-225]

```python
func_rows = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}
kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))
```

`equivalent_primary`Maps signed integers to unsigned integers (because addition/multiplication are the same for both):

[FACT:src/device/generate.py:158-166]

```python
def equivalent_primary(coll, redop, ty, algo, proto):
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    if redop in ("Sum","Prod","PreMulSum","SumPostDiv") and ty[0]=="i":
      return (coll, redop, "u"+ty[1:], algo, proto)
    if redop=="MinMax" and ty[0]=="i" and ("NVLS" not in algo):
      return (coll, redop, "u"+ty[1:], algo, proto)
  return (coll, redop, ty, algo, proto)
```

`best_kernel`Maps multiple main functions to the same kernel (e.g., all`AllGather`algorithms map to`AllGather RING LL`）：

[FACT:src/device/generate.py:171-183]

```python
def best_kernel(coll, redop, ty, algo, proto):
  def best(coll, redop, ty, algo, proto):
    if coll=="Nop": return ("Generic", None, None, None, None)
    if coll=="SendRecv": return ("SendRecv", None, None, None, None)
    if exact_kernel_names: return (coll, redop, ty, algo, proto)
    if coll in ("AllGather","Broadcast","AllGatherV"): return (coll, None, None, "RING", "LL")
    return (coll, "Sum", ty, ("TREE" if algo=="TREE" else "RING"), "LL")
  kfn = equivalent_primary(*best(coll, redop, ty, algo, proto))
  if not func_filter(*kfn): return ("Generic", None, None, None, None)
  return kfn
```

**Step 3: Generate kernel definitions.**

[FACT:src/device/generate.py:458-480]

```python
(_, kfns) = name_to_kernels.get(name) or (None, [])
for kfn in kfns:
  (coll, redop, ty, algo, proto) = kfn
  sym = kernel_suffix(kfn)
  fn_id = primary_to_index[kfn]
  cudart, arch = required_cuda(*kfn)
  s = "DEFINE_ncclDevKernel({sym}, ncclFunc{coll}, {redop_cxx}, {ty_cxx}, NCCL_ALGO_{algo}, NCCL_PROTO_{proto}, {fn_id})\n"
  # ...
  out(s.format(...))
```

`DEFINE_ncclDevKernel`After macro expansion:

[FACT:src/device/common.h:507-509]

```c
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain, algo, proto>>(&args4K.args); \
  }
```

So each kernel is a`__global__`function, calling`ncclKernelMain`, with template parameters`specializedFnId`and`RunWorkBatch<coll, ty, redop<ty>, algo, proto>`。

## Design Thinking and Production Pitfalls

> **[Design Inference & Architectural Trade-offs]**
> **Why use "representative kernels" instead of one kernel per combination?**Trade-off between compilation time and binary size. The full combinatorial space is 7 × 5 × 12 × 7 × 3 ≈ 8820 kernels, each kernel takes a few seconds to compile, totaling several hours. And the binary size would reach hundreds of MB. By mapping to representative kernels, the actual number of generated kernels is reduced to a few dozen.

**Pitfall 1:`NCCL_EXACT_KERNEL_NAMES`causes compilation explosion.**If this environment variable is set,`best_kernel`returns the original function, and a kernel is generated for every combination. This is useful during development (you can precisely control which kernel is compiled), but in production it causes excessively long compilation times.

**Pitfall 2:`required_cuda`'s version check.**Some kernels require a specific CUDA version or architecture:

[FACT:src/device/generate.py:130-154]

At this point, the kernel has been launched on the GPU, and the device side has obtained the work descriptor. But what really determines performance is how data is moved inside the device. The next chapter will dive into the three protocol primitives under src/device: LL, LL128, and Simple, to see why the same AllReduce logic requires three sets of transport primitives, and their differences in synchronization methods, buffer layouts, and flag semantics.
