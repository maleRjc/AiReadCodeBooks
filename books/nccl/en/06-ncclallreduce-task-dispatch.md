# Chapter 6: Operator dispatch panorama: How ncclAllReduce becomes an executable kernel task

In the previous chapter, we walked through the tuning module and learned that NCCL selects an (algorithm, protocol, channel, warp) combination for a collective communication within microseconds. But the selection result itself is just a bunch of numbers—it needs to be "translated" into a task description object that the GPU kernel can understand before it can actually be executed. This chapter enters the main body of src/enqueue/enqueue.cc and answers a core question: when the user calls ncclAllReduce, what exactly happens on the host side? From ncclAllReduce to ncclEnqueueCheck, it goes through parameter validation, algorithm/protocol determination, and channel partitioning, ultimately generating the ncclInfo and ncclTaskColl structures. This is the key chapter where the book switches from the "user perspective" to the "engine perspective." If NCCL is compared to a restaurant, then the enqueue module is the "front desk ordering system": the user (application layer) says "I want an AllReduce," and the front desk translates it into a work order that the kitchen (GPU kernel) can execute—which stove, what pan to use, and how many batches to make it in. Without this translation layer, the kitchen wouldn't know what dish to make at all.

# I. Entry Point: How ncclAllReduce Constructs ncclInfo

## Intuitive Model

`ncclAllReduce`It is the API function directly called by the user. Its responsibility is extremely singular:**Package the raw parameters passed in by the user into a`ncclInfo`structure, then hand it off to`ncclEnqueueCheck`**. This is like going to a bank counter to handle business—the teller first fills your request into a standard form, then forwards it to the backend system.

Without this layer, every collective communication API would have to handle parameter validation, group semantics, and profiler instrumentation on its own—the code would become so repetitive it would be unmaintainable.

## Data Structure: Memory Layout of ncclInfo

`ncclInfo`It is the core carrier that runs through the entire enqueue process. Its definition is in`src/include/info.h`：

[FACT:src/include/info.h:17-44]

This structure has 20+ fields, which we can divide into four groups by function:

| Field Group | Field | Purpose |
| --- | --- | --- |
| Collective Communication Parameters | `coll`, `sendbuff`, `recvbuff`, `count`, `datatype`, `op`, `root` | Describes "what to do" |
| Communication Domain and Stream | `comm`, `stream` | Describes "where to do it" |
| Algorithm Details | `chunkSteps`, `sliceSteps` | Describes "how to partition" |
| One-sided Operations | `peerWinOffset`, `peerWin`, `sigIdx`, `ctx`, `flags`, `nDesc`, `signalDescs` | RMA-specific |
| User Configuration | `collConfig` | A private copy copied from the user config |

Note the`collConfig`comment:**"A config copied from config passed by user so older user config can be safely accessed during synchronous host scheduling (never at launch/replay)"** [FACT:src/include/info.h:41-43]. This is a key design—the config pointer passed in by the user may be destroyed before`ncclGroupEnd`, so NCCL makes a copy in`ncclInfo`.

## Step-by-Step: The Call Chain of ncclAllReduce

We take`ncclAllReduce`as an example, tracing the complete path from the user call to the construction of`ncclInfo`.

**Step 1: The user calls ncclAllReduce.**The entry point is in`src/collectives.cc`：

[FACT:src/collectives.cc:206-211]

Three things are done here:

1. `NVTX3_FUNC_WITH_PARAMS`Add an NVTX marker (for visualization in tools like Nsight)

2. Call`ncclAllReduceConfigImpl`, passing in`config = nullptr`

3. Return the result

**Step 2: ncclAllReduceConfigImpl constructs ncclInfo.**This is the key step:

[FACT:src/collectives.cc:192-202]

Note that C-style aggregate initialization is used here:

```c
struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                        sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
                        ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
```

The fields correspond one-to-one in the declaration order of`ncclInfo`.`ALLREDUCE_CHUNKSTEPS`and`ALLREDUCE_SLICESTEPS`are defined in`src/include/collectives.h`：

[FACT:src/include/collectives.h:19-20]

`NCCL_STEPS`is the number of steps in the ring buffer (usually 8 or 16), so AllReduce's chunkSteps is`NCCL_STEPS/2`, and sliceSteps is`NCCL_STEPS/4`. This means one chunk contains 2 slices.

**Step 3: Parse the user config.** `ncclParseCollConfig`Parses the user-passed`ncclCollConfig_t*`into`info.collConfig`. If`config == nullptr`, this field remains zero-initialized.

**Step 4: Hand off to ncclEnqueueCheck.**This is the true entry point of the enqueue module.

## Design Thinking: Why Use Aggregate Initialization Instead of Field-by-Field Assignment?

> **[Design Inference & Architectural Trade-offs]**
> Aggregate initialization has two benefits: first, the compiler checks whether the number of fields matches (a missing field triggers a warning); second, the code is more compact. But the drawback is that**the field order must strictly match the struct declaration**—if someone inserts a field in the middle of`ncclInfo`, all aggregate initialization sites will silently misalign. This is an implicit maintenance risk in the NCCL codebase.

## Production Pitfall: Config Lifetime

A real pitfall scenario: the user writes code like this:

```c
ncclCollConfig_t config = {...};
ncclAllReduceConfig(..., &config);
// config 在这里被销毁（比如是栈变量，函数返回了）
```

If NCCL did not copy the config in`ncclInfo`, then accessing`ncclGroupEnd`at`info.collConfig`would read already-freed memory.`src/include/info.h:41-43`The comment in**is precisely to explain this design—**。

---

# the config is parsed and copied during the task append phase, and afterward no longer depends on the user pointer

## II. ncclEnqueueCheck: Parameter Validation and Group Semantics

`ncclEnqueueCheck`Intuitive Model**It is the "main gate" of the enqueue module. All collective communication APIs ultimately converge here. Its responsibilities are:**Validate parameter legality, handle group semantics, and call taskAppend to generate tasks`ncclEnqueueCheck`。

. If compared to airport security, then each API function is a check-in counter—check-in only takes luggage; the real security check is at

## Step-by-Step: The execution flow of ncclEnqueueCheck

[FACT:src/enqueue/enqueue.cc:3478-3527]

Let's break it down step by step:

**Step 1: CommCheck validates the communicator.** `CommCheck(info->comm, info->opName, "comm")`Check whether the comm pointer is non-null and whether it has been initialized. If the comm has been revoked (for example, if some rank encounters an error), return an error directly:

[FACT:src/enqueue/enqueue.cc:3480-3485]

**Step 2: Handle profiler depth.**If already inside a group (`profilerGroupDepth > 0`), increment the depth counter. This is to correctly handle implicit`ncclGroupStartInternal`/`ncclGroupEndInternal`calls.

**Step 3: Enter the internal group.** `ncclGroupStartInternal()`This is NCCL's internal group mechanism.**Key point**: Even if the user does not explicitly call`ncclGroupStart`, NCCL will create an implicit group for each API call. This guarantees the atomicity of a single call.

**Step 4: Ensure comm is ready.** `ncclCommEnsureReady(info->comm)`Wait for communicator initialization to complete (for example, bootstrap completion and connection establishment).

**Step 5: ArgsCheck parameter validation.**This is the most complex validation step:

[FACT:src/enqueue/enqueue.cc:3497-3503]

Note the handling of`checkMode`: If it is`ncclCheckModeDebugGlobal`，`ArgsCheck`, info will be enqueued, and global validation will be performed at`ncclGroupEnd`(for example, checking whether the count is consistent across all ranks).

**Step 6: Call taskAppend.**This is the core conversion step:

[FACT:src/enqueue/enqueue.cc:3513]

**Step 7: Increment opCount.**After each successful enqueue,`comm->opCount++`. This counter is used to match send/recv operations and is also the basis for the profiler timeline.

**Step 8: Exit the group.** `ncclGroupEndInternal()`If depth drops to 0, the actual group operation is triggered (scheduling and kernel launch).

## Concurrency control: group semantics and thread safety

> **[Design Inference & Architectural Trade-offs]**
> `ncclGroupStartInternal`/`ncclGroupEndInternal`Thread-local storage (TLS) is used to maintain group state. This means that**multiple API calls within the same thread will be merged into one group**, but calls from different threads are independent. This is the foundation of NCCL's support for multithreaded calls.

An easy pitfall: If the user calls a non-NCCL CUDA API between`ncclGroupStart`and`ncclGroupEnd`(for example,`cudaMemcpy`), it may cause stream ordering issues. NCCL's group mechanism assumes that operations within a group are all on the same set of streams.

## Error recovery chain

`ncclEnqueueCheck`The error handling of

[FACT:src/enqueue/enqueue.cc:3524-3526]

has an ingenious design:`taskAppend`If`ncclCommSetAsyncError`fails and comm is in non-blocking mode, it will call

---

# to record the error. In this way, subsequent API calls will immediately return an error instead of continuing to try. This is the asynchronous error propagation mechanism.

## III. taskAppend: The crossroads of task dispatch

`taskAppend`Intuitive model`info->coll`It is the "transport hub" of the enqueue module. Based on the value of

, it dispatches tasks to different processing paths: P2P, RMA, CE, or ordinary collective communication. This is like a post office sorting center - based on the address on the envelope, it delivers letters to different mailboxes.

## Without this dispatch layer, all types of operations would have to be crammed into one huge if-else, making the code difficult to maintain.

[FACT:src/enqueue/enqueue.cc:3337-3476]

**Step-by-Step: The dispatch logic of taskAppend** `ncclParamEnqueueRearchEnable()`Step 1: Determine whether the new architecture is enabled.`rawTaskAppend`It is an environment variable switch (default 0). If enabled, take the

**path - this is the new task model that NCCL is developing.**Step 2: P2P dispatch.`p2pTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3343-3345]

**If it is Send/Recv, call**Step 3: RMA dispatch.`rmaTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3346-3347]

**If it is PutSignal/Signal/WaitSignal, call** `if (info->count == 0) return ncclSuccess;`Step 4: Early return for empty collective communication.

**- Collective communication with count 0 is discarded directly.** `ncclCollConfigGetAlgMask`Step 5: Algorithm selection validation.

[FACT:src/enqueue/enqueue.cc:3357-3358]

**Validate whether the algorithm selection passed in by the user is legal:**Step 6: FP8 type check.

[FACT:src/enqueue/enqueue.cc:3360-3366]

**FP8 reduction requires sm90+:** `hostToDevRedOp`Step 7: Reduction operation conversion.`ncclRedOp_t`Convert the host-side`ncclDevRedOpFull`：

[FACT:src/enqueue/enqueue.cc:3370-3371]

**to the device-side**Step 8: Early return for single rank.`comm->nRanks == 1`If`ncclLaunchOneRank`, directly call

[FACT:src/enqueue/enqueue.cc:3373-3377]

**to perform local reduction without generating a task:**Step 9: Multi-rank path.

[FACT:src/enqueue/enqueue.cc:3378-3470]

## This is the most complex branch, including CE routing, AllToAll/Gather/Scatter fallback, and ordinary collective communication:

`collTaskAppend`Data structure: fields of ncclTaskColl`ncclTaskColl`This is where

[FACT:src/enqueue/enqueue.cc:2757-2851]

is generated. Let's look at its core logic:

| Key field assignments: | Field | Source |
| --- | --- | --- |
| `func` | `info->coll` | Meaning |
| `sendbuff`/`recvbuff` | `info->sendbuff`/`recvbuff` | Collective communication type |
| `count` | `info->count` | Buffer pointer |
| `datatype` | `info->datatype` | Element count |
| `trafficBytes` | `count * elementSize * ncclFuncTrafficPerByte` | Data type |
| `opHost`/`opDev` | `info->op`/`opDev` | Traffic estimation |
| `chunkSteps`/`sliceSteps` | `info->chunkSteps`/`sliceSteps` | Reduction operation |
| `minCTAs`/`maxCTAs`/`nvlsCTAs` | Number of split steps | Configuration parsing |
| `algMask` | `ncclCollConfigGetAlgMask` | Resource limit |

Algorithm selection mask`trafficBytes`Note the calculation of

[FACT:src/enqueue/enqueue.cc:2813]

`ncclFuncTrafficPerByte`:

[FACT:src/enqueue/enqueue.cc:123-134]

It returns the traffic multiplier for each collective communication type:

## AllReduce returns 2 (because it needs reduce + broadcast), AllGather/ReduceScatter returns nRanks, and others return 1.

[FACT:src/enqueue/enqueue.cc:2808-2812]

Design consideration: Why should AllGather/Broadcast be converted to int8?`ncclInt8`. This is an optimization:**These two operations do not involve reduction, so there is no need to care about data types. Handling them uniformly as bytes can simplify the kernel logic.**。

## Production pitfall: the parsing order of CTAPolicy

[FACT:src/enqueue/enqueue.cc:3390-3397]

CTAPolicy parsing has a subtle priority:**env > per-call > comm**. And`NCCL_CTA_POLICY_ZERO`takes precedence over`NCCL_CTA_POLICY_EFFICIENCY`. If the user sets both flags at the same time, ZERO will take effect.

A real pitfall scenario: the user set`NCCL_CTA_POLICY=EFFICIENCY`, but found that the CE path was not used. The reason is that CE routing requires`CTAPolicy & NCCL_CTA_POLICY_ZERO`to be true, and EFFICIENCY does not satisfy this condition.

---

# 4. ncclPrepareTasks: from task list to scheduling queue

## Intuitive model

`ncclPrepareTasks`It is the "preprocessor" of the enqueue module. It buckets the scattered task list by (func, op, datatype), and then computes the algorithm and protocol for each bucket. This is like a librarian - first sorting returned books by category, then deciding which shelf each category of books goes on.

Without this step, the subsequent`scheduleCollTasksToPlan`would have to compute the algorithm separately for each task, which is extremely inefficient.

## Step-by-Step: the bucketing logic of ncclPrepareTasks

[FACT:src/enqueue/enqueue.cc:423-642]

**Step 1: Broadcast task conversion.**If there is only one broadcast peer, convert the broadcast task into a coll task:

[FACT:src/enqueue/enqueue.cc:430-461]

Note that here the fields of`bcastTask`are copied to the new`ncclTaskColl`, and`trafficBytes`is computed. Then the original task is released from`memPool_ncclTaskBcast`.

**Step 2: Bucket by (func, op, datatype).**Tasks come out of the sorter in descending order of size, and are then assigned to the`tasksByFnOpTy`array:

[FACT:src/enqueue/enqueue.cc:464-487]

Index calculation:`((int)task->func * ncclNumDevRedOps + (int)task->opDev.op) * ncclNumTypes + (int)task->datatype`. This is the linearization of a three-dimensional array.

**Step 3: Aggregation and algorithm selection.**For each bucket, aggregate tasks with similar sizes (within 4x), and then call`ncclGetAlgoInfo`：

[FACT:src/enqueue/enqueue.cc:503-547]

**Step 4: Bucket by (collnet, nvls).**According to the algorithm type, assign tasks to`collBins[2][2]`：

[FACT:src/enqueue/enqueue.cc:517-544]

**Step 5: Concatenate the final queue.**Concatenate the four buckets into`planner->collTaskQueue`：

[FACT:src/enqueue/enqueue.cc:553-557]

## Data structure: ncclTaskCollSorter

`ncclTaskCollSorter`is an insertion sorter ordered by`trafficBytes`.`ncclTaskCollSorterInsert`inserts the task into the correct position,`ncclTaskCollSorterDequeueAll`retrieves all tasks in order.

> **[Design Inference & Architectural Trade-offs]**
> The design motivation of this sorter is:**Large tasks are scheduled first**. Because large tasks have long transfer times, starting them first allows better overlap of computation and communication.

## Concurrency control: runtimeConn and connection establishment

[FACT:src/enqueue/enqueue.cc:572-583]

If`comm->runtimeConn`is true (runtime connection mode), and the channel of some algorithm has not yet been initialized, mark`algoNeedConnect`. This will trigger connection establishment later.

## Production pitfall: boundary conditions of aggregation

[FACT:src/enqueue/enqueue.cc:507-508]

The aggregation condition is`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes`, and neither task sets`aggIsolate`. If the user sets a per-call config (for example,`maxCTAs`），`aggIsolate`will be set to true, this task will not be aggregated.

A real pitfall scenario: the user set`maxCTAs=4`for a certain AllReduce, expecting it to use only 4 CTAs. However, due to the aggregation logic, this task may be merged with adjacent tasks, causing the actual number of CTAs used to not match expectations. The solution is to set`aggIsolate`- NCCL has already handled this in`collTaskAppend`:

[FACT:src/enqueue/enqueue.cc:2821-2822]

---

# 5. scheduleCollTasksToPlan: channel splitting and budget control

## Intuitive model

`scheduleCollTasksToPlan`It is the "scheduler" of the enqueue module. It assigns tasks to specific channels and computes the data split for each channel. This is like a factory's production scheduling system - deciding what each production line does and how much it does.

Without this step, the GPU kernel would not know which part of the data it needs to process.

## Step-by-Step: channel splitting algorithm

[FACT:src/enqueue/enqueue.cc:644-947]

**Step 1: Budget estimation.**First estimate the number of tasks that can fit into this plan:

[FACT:src/enqueue/enqueue.cc:648-689]

`ncclTestBudget`Check whether the work byte count exceeds the budget:

[FACT:src/enqueue/enqueue.cc:343-349]

**Step 2: Compute the traffic for each channel.**According to kind (collnet/nvls), compute`trafficPerChannel`：

[FACT:src/enqueue/enqueue.cc:701-707]

**Step 3: Collnet path.**If it is a collnet algorithm, channel assignment is relatively simple:

[FACT:src/enqueue/enqueue.cc:709-739]

**Step 4: Cell splitting for the normal path.**This is the most complex part. NCCL splits data into "cells", and each cell is a minimum transfer unit:

[FACT:src/enqueue/enqueue.cc:740-845]

Key variables:

- `cellSize`: the number of bytes per cell, at least`MinTrafficPerChannel`（32KB）
- `cells`: total number of cells
- `cellsPerChannel`: number of cells processed by each channel
- `cellsLo`/`cellsHi`: number of cells for the first and last channels (may be less than full)

**Step 5: Compute chunkGrains.**Call for each channel segment`calcCollChunking`：

[FACT:src/enqueue/enqueue.cc:811-825]

**Step 6: Generate proxyOp.**Generate a proxy operation for each channel:

[FACT:src/enqueue/enqueue.cc:844-894]

## Data structure: ncclDevWorkColl

`ncclDevWorkColl`is the device-side work descriptor. Its key fields:

| Field | Meaning |
| --- | --- |
| `sendbuff`/`recvbuff` | Buffer pointer |
| `channelLo`/`channelHi` | Channel range |
| `cbd.countLo`/`countMid`/`countHi` | Number of elements in each segment |
| `cbd.chunkGrainsLo`/`Mid`/`Hi` | Chunk granularity of each segment |
| `direct` | Direct flag |

## Concurrency control: bit operations of channelMask

[FACT:src/enqueue/enqueue.cc:897]

This line of code uses bit operations to set channelMask:`(2ull << channelHi) - (1ull << channelLo)`For example, channelLo=2, channelHi=5, the result is`(2<<5) - (1<<2) = 64 - 4 = 60 = 0b111100`, meaning bits 2-5 are set.

## Production pitfall: budget overflow

[FACT:src/enqueue/enqueue.cc:792-794]

If the budget is insufficient, directly return`ncclSuccess`, letting the outer loop create a new plan. This is an elegant degradation strategy—**no error, just batch processing**。

A real pitfall scenario: if`NCCL_WORK_FIFO_BYTES`is set too small, each plan can only hold very few tasks, increasing the number of kernel launches and reducing performance.

---

# 6. finishPlan: From tasks to kernel parameters

## Intuitive model

`finishPlan`is the "packer" of the enqueue module. It packs tasks, batches, and proxyOps into a parameter structure that the kernel can directly read. This is like express packaging—putting loose items into boxes, attaching waybills, and waiting for shipment.

## Step-by-Step: The packing logic of finishPlan

[FACT:src/enqueue/enqueue.cc:236-330]

**Step 1: Decide the storage type.**If all work can fit into kernel args, use`ncclDevWorkStorageTypeArgs`：

[FACT:src/enqueue/enqueue.cc:244-250]

**Step 2: Allocate kernelArgs.**Allocate from the memory stack:

[FACT:src/enqueue/enqueue.cc:251-255]

**Step 3: Round-robin placement of batches.**The first batch of each channel must be placed at`batchZero[blockIdx.x]`：

[FACT:src/enqueue/enqueue.cc:257-280]

**Step 4: Merge proxyOp queues.**Merge sort by opCount:

[FACT:src/enqueue/enqueue.cc:282-329]

## Data structure: ncclDevKernelArgs

`ncclDevKernelArgs`is the parameter structure passed to the kernel. It contains:

- `comm`: device-side communicator
- `channelMask`: channel bitmask
- `workStorageType`: work storage type
- `workBuf`: work buffer pointer
- `workMask`: work buffer mask

## Production pitfall: batch order

[FACT:src/enqueue/enqueue.cc:257-259]

The comment states it clearly: "The first batch for each channel must be located at batchZero[blockIdx.x]". If this order is wrong, the kernel will read the wrong batch, causing data corruption.

---

# Chapter summary

In this chapter, we traced the complete path from`ncclAllReduce`to`ncclTaskColl`:

1. **ncclAllReduce**constructs`ncclInfo`, packing user parameters

2. **ncclEnqueueCheck**validates parameters, handles group semantics

3. **taskAppend**dispatches to different paths based on operation type

4. **collTaskAppend**generates`ncclTaskColl`, parses configuration

5. **ncclPrepareTasks**buckets by (func, op, datatype), computes algorithms

6. **scheduleCollTasksToPlan**splits channels, generates`ncclDevWorkColl`

7. **finishPlan**packs into kernel parameters

Key design principles:

- **Layered decoupling**: each function does only one thing, passing state through`ncclInfo`and`ncclTaskColl`
- **Budget control**: controlling the size of each plan through`ncclTestBudget`
- **Aggregation optimization**: tasks of similar size are aggregated, reducing the number of kernel launches
- **Configuration priority**：env > per-call > comm

In the next chapter, we will enter`task_sched`, to see how NCCL orchestrates the execution order of multiple channels and multiple kernels.

# Chapter review and self-test

Q1: If the`collTaskAppend`in`aggIsolate`is removed (i.e.,`src/enqueue/enqueue.cc:2821-2822`always returns false), in what scenarios would the user-set`maxCTAs`become ineffective? Why?

**Reference analysis**：`aggIsolate`'s purpose is to mark "this task cannot be aggregated". If this check is removed, tasks with per-call config set will be merged with adjacent tasks. In`ncclPrepareTasks`'s aggregation loop (`src/enqueue/enqueue.cc:507-508`), the aggregation condition is`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes && !aggBeg->aggIsolate && !aggEnd->aggIsolate`. If`aggIsolate`always returns false, then even if a task has`maxCTAs=4`set, it may be merged with a task that has`maxCTAs=32`. The merged`agg`will take some combination of the two (depending on the implementation of`ncclGetAlgoInfo`), causing the actual number of CTAs used to not match user expectations.

More seriously, in`scheduleCollTasksToPlan`(`src/enqueue/enqueue.cc:665-666`），`taskAggIsolate`is used to ensure that tasks with per-call resources configured occupy a separate plan. If this check fails, multiple tasks will share the plan's channel budget, causing resource allocation to not match expectations.

Q2: In`ncclEnqueueCheck`, if`ncclGroupEndInternal()`returns an error (e.g., ArgsCheck fails for some rank), but`taskAppend`has already executed successfully, what happens? How does NCCL ensure state consistency?

**Reference analysis**: Look at`src/enqueue/enqueue.cc:3513-3519`'s control flow:

```c
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
info->comm->opCount++;
exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
```

If`taskAppend`succeeds but`ncclGroupEndInternal`fails,`opCount`has already been incremented. This will cause the opCount of subsequent operations to mismatch with the peer, potentially triggering a hang.

NCCL's approach is:`ncclGroupErrCheck(ret)`will check for errors, and if there are any, will set the comm's error state. Subsequent API calls will detect this error through`ncclCommGetAsyncError`and return immediately. This is a "fail-fast" strategy—once an error occurs, the entire comm enters an error state and no longer attempts recovery.

In a production environment, this means that once a group error occurs, the user needs to destroy and rebuild the communicator.

Q3: `scheduleCollTasksToPlan`The cell splitting algorithm in`src/enqueue/enqueue.cc:740-845`) has a boundary condition: when`cellsLo == 0`, it skips the minimum number of channels. If this skip logic has a bug (e.g.,`channelId`is not correctly incremented), what consequences would it cause?

**Reference analysis**: Look at`src/enqueue/enqueue.cc:770-780`：

```c
if (cellsLo == 0) {
  // Least channel skipped. Make the next channel the new least.
  channelId += 1;
  if (nMidChannels == 0) {
    cellsLo = cellsHi;
    cellsHi = 0;
  } else {
    cellsLo = cellsPerChannel;
    nMidChannels -= 1;
  }
}
```

If`channelId`is not correctly incremented, then the next task will start allocating from the wrong channel. This will cause:

1. **Channel overlap**: two tasks may be allocated to the same segment of data in the same channel

2. **Data corruption**: the kernel will repeatedly process or miss data

3. **Performance degradation**: channel load imbalance

More insidiously, this kind of bug may only be triggered under specific message sizes (when`cellsLo == 0`), making it hard to reproduce. NCCL uses`plan->channelMask |= (2ull << devWork->channelHi) - (1ull << devWork->channelLo)`to track the channels already in use, but this is only a record; it cannot prevent overlap.

At this point, we have clearly seen how ncclAllReduce goes from a user call to a series of executable kernel tasks: parameter validation, algorithm/protocol determination, channel partitioning, and finally generating ncclInfo and ncclTaskColl. But creating the tasks is only the first step—they still need to be scheduled onto multiple channels, generate kernel launch parameters, and handle batch submission and dependency ordering under group semantics. The next chapter will dive into src/enqueue/task_sched and src/enqueue/task_prep to answer "why a single AllReduce launches multiple kernels, and how their order and dependencies are guaranteed," while also revealing how ncclGroupStart/ncclGroupEnd in src/group.cc merge multiple API calls into a single submission.
