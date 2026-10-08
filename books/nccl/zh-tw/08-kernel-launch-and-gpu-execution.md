# 第 8 章：Kernel 啟動與裝置端執行：從 host 側呼叫到 GPU 執行緒塊起跑

上一章我們拆解了任務如何被切分到多個 channel、如何生成 kernel 啟動參數，以及 group 語意下批次提交與依賴排序的機制。現在，啟動計畫已經就緒，但它還只是 host 側的資料結構。本章要回答的核心問題是：`ncclKernelPlan`如何變成 GPU 上一個真正在跑的 grid？我們將沿著`ncclLaunchKernel`的呼叫鏈，看參數如何被塞進 kernel args、kernel 變體如何被選中、`cuLaunchKernelEx`如何被呼叫，以及裝置側`ncclKernelMain`如何從共享記憶體裡把工作描述讀出來並分發到具體實現。

# 從 Plan 到 Grid：啟動路徑的全景

在深入細節之前，先建立一個整體心智模型。把`ncclKernelPlan`想像成一張「施工圖紙」：它記錄了這次要啟動幾個 channel（幾個 block）、每個 block 多少執行緒、要執行哪些 work、用哪個 kernel 函式。而`ncclLaunchKernel`就是「施工隊進場」的動作——它把圖紙上的資訊翻譯成 CUDA 驅動能理解的`CUlaunchConfig`，然後呼叫`cuLaunchKernelEx`把 grid 真正發射到 GPU 上。

如果沒有這一層，host 側的所有調度（上一章的 channel 切分、batch 組織、proxy op 排序）都只是紙上談兵，GPU 上不會有任何 kernel 運行，通訊永遠不會發生。這是端到端主幹的最後一環，也是 host 與 device 的分界線。

整個啟動路徑可以概括為三個階段：

1. **參數準備**（`finishPlan` + `uploadWork`）：把 work 結構體、batch 描述符、kernel args 組織到一塊連續記憶體裡，決定是放在 kernel 參數裡、FIFO 裡還是持久化緩衝區裡。

2. **kernel 發射**（`ncclLaunchKernel`）：計算 grid/block 維度，組裝 launch attributes（CGA cluster、mem sync domain、launch completion event），呼叫`cuLaunchKernelEx`。

3. **裝置側入口**（`ncclKernelMain`）：每個 block 根據`blockIdx.x`確定自己的 channelId，從 args 或 FIFO 裡載入 work batch 到共享記憶體，然後透過`ncclDevFuncTable`分發到具體的演算法/協定實現。

下面這張圖展示了從 plan 到 grid 的完整控制流，包含關鍵的分支判斷：

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

這張圖錨定了本章的三個核心函式：`finishPlan`、`uploadWork`、`ncclLaunchKernel`。接下來我們逐個拆解。

# 參數準備：work 結構體如何找到自己的位置

## 直覺模型

`finishPlan`的角色類似於快遞分揀中心的「裝箱員」。它面對一堆零散的 work 結構體（每個 collective 或 p2p 操作對應一個），需要決定：這些 work 是塞進 kernel 參數這個「隨身背包」裡，還是放進 FIFO 這個「傳送帶」上，還是放進持久化緩衝區這個「倉庫」裡？

如果這個決策做錯了——比如 work 太大塞不進 kernel 參數卻硬塞——kernel 啟動會直接失敗。如果 work 放錯了位置，裝置側讀到的就是垃圾資料，通訊結果完全錯誤。

## 資料結構與記憶體佈局

先看`ncclDevKernelArgs`的結構，它是 host 和 device 之間的「信封」：

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

這個結構體只有 5 個欄位，但每個欄位都承載著關鍵資訊。`channelMask`是一個 64 位元遮罩，每一位對應一個 channel，裝置側透過`__popcll`計算`blockIdx.x`對應的 channelId。`workStorageType`決定了裝置側從哪裡讀 work：`Args`表示 work 就在 kernel 參數裡，`Fifo`表示在環形緩衝區裡，`Persistent`表示在持久化緩衝區裡。

`ncclDevWorkBatch`是 batch 描述符，它告訴裝置側「這個 channel 的 work 在哪裡、有多少個」：

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

`offsetBitset`是一個 64 位元遮罩，每一個位元對應一個 work 結構體。裝置側透過`__popc`和`fns`（find n-th set）指令來定位每個 work 的偏移。`nextJump`和`nextExtends`用於把多個 batch 串聯起來——當 work 太多裝不下一個 batch 時，會建立「擴展 batch」。

## Step-by-Step Walkthrough

現在代入一個具體場景：一次 AllReduce 被切分到 4 個 channel，每個 channel 有 2 個 work 結構體，總共 8 個 work。

**第一步：`finishPlan`決定儲存類型。**

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

這裡的關鍵判斷是：如果`sizeof(ncclDevKernelArgs) + batchBytes + workBytes`能裝進`comm->workArgsBytes`（通常是 4KB），就把 work 直接放進 kernel 參數裡。否則，work 會被放到 FIFO 或持久化緩衝區，kernel 參數裡只放 batch 描述符。

> **[Design Inference & Architectural Trade-offs]**
> 為什麼優先放 kernel 參數？ 因為 kernel 參數在 CUDA 驅動裡是透過常量記憶體（constant memory）傳遞的，裝置側讀取時走的是`ld.param`指令，比從全域記憶體讀取 FIFO 要快得多。對於小訊息（work 總量小），這能顯著降低延遲。

**第二步：把 batch 按 channel 輪流放入 kernel args。**

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

這裡有幾個關鍵點：

1. **`fifoCursor`的語意**：對於`Args`類型，它是相對於`kernelArgs`起始位址的偏移；對於`Fifo`類型，它是相對於 FIFO 基底位址的偏移；對於`Persistent`類型，它從 0 開始。

2. **`offsetBase`的修正**：`finishPlan`裡 batch 的`offsetBase`是相對於 plan 的 work 起始位置的（從 0 開始）。`uploadWork`需要把它轉換成相對於實際儲存位置的偏移。對於`Args`類型，加上`sizeof(ncclDevKernelArgs) + batchBytes`；對於`Fifo`類型，加上`comm->workFifoProduced`。

3. **16 位元組對齊拷貝**：work 結構體都是 16 位元組對齊的（`alignas(16)`），所以拷貝時按 16 位元組為單位。`COMPILER_ASSUME_ALIGNED`告訴編譯器這個位址是 16 位元組對齊的，讓編譯器生成更高效的向量化指令。

4. **FIFO 等待**：對於`Fifo`類型，`waitWorkFifoAvailable`會自旋等待 FIFO 有足夠空間。這個等待會檢查`comm->abortFlag`，避免在 abort 時死鎖。

## 設計思考與生產踩坑

> **[Design Inference & Architectural Trade-offs]**
> **為什麼要有三種儲存類型？**這是空間和延遲的權衡：

- `Args`：最快（常量記憶體），但容量有限（4KB）。適合小訊息、少量 work。
- `Fifo`：容量大（環形緩衝區），但裝置側讀取要走全域記憶體。適合中等訊息。
- `Persistent`：用於 CUDA Graph 捕獲場景。因為 graph 捕獲時不能做`cudaMemcpy`，所以需要預先分配持久化緩衝區，把 work 拷貝進去，然後讓 kernel 從那裡讀。

**踩坑點 1：FIFO 溢出導致死鎖。**如果`waitWorkFifoAvailable`沒有檢查`abortFlag`，當 FIFO 滿且消費者（GPU kernel）因為某種原因停止消費時，host 會永遠自旋。原始碼裡[FACT:src/enqueue/enqueue.cc:1333-1349]明確檢查了 abort flag：

```c
if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
  return ncclInternalError;
}
```

**踩坑點 2：`offsetBitset`溢出。** `offsetBitset`是 64 位元的，最多支援 64 個 work 在一個 batch 裡。如果超過 64 個，`1ull << (offset / workSize)`會溢出。原始碼裡透過`NCCL_MAX_DEV_WORK_BATCH_BYTES`限制了 batch 的大小（1024 位元組），而最小的 work 結構體是`ncclDevWorkColl`（約 80 位元組），所以最多 12 個 work，不會溢出。

**踩坑點 3：Persistent 模式下的記憶體洩漏。**在`uploadWork`的`Persistent`分支裡，`fifoBufHost`是透過`ncclOsAlignedAlloc`分配的，需要在`uploadWork_cleanup_fn`裡釋放。如果`cudaMemcpyAsync`失敗，`fail`標籤會檢查`cleanup`是否為 null，如果為 null 就直接釋放`fifoBufHost`。這個錯誤恢復鏈在[FACT:src/enqueue/enqueue.cc:1483-1485]可以看到。

# Kernel 發射：從 CUlaunchConfig 到 cuLaunchKernelEx

## 直覺模型

`ncclLaunchKernel`的角色類似於「火箭發射控制台」。它接收一個已經裝好燃料（work 資料）的 plan，計算出火箭的飛行參數（grid/block 維度），設定好各種發射選項（cluster、mem sync domain、completion event），然後按下發射按鈕（`cuLaunchKernelEx`）。

如果這個環節出錯——比如 grid 維度算錯了——GPU 上會啟動錯誤數量的 block，導致部分 channel 的工作永遠不會被執行，通訊掛起。

## 資料結構與記憶體佈局

`CUlaunchConfig`是 CUDA 驅動 API 的啟動配置結構體，NCCL 在堆疊上建構它：

[FACT:src/enqueue/enqueue.cc:1916-1917]

```c
CUlaunchConfig launchConfig = {0};
CUlaunchAttribute launchAttrs[6] = {};
int attrs = 0;
```

`launchAttrs`是一個最多 6 個元素的陣列，每個元素是一個`CUlaunchAttribute`。NCCL 根據硬體能力和驅動版本，有條件地加入不同的屬性：

- `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`：CGA cluster 維度（sm90+）
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`：cluster 排程策略
- `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`：記憶體同步域（CUDA 12.0+）
- `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`：啟動完成事件（CUDA 12.3+）
- `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`：程式化流序列化（sym kernel）
- `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING`：NVLink 利用率中心排程（CUDA 13.0+）

## Step-by-Step Walkthrough

**第一步：計算 grid 和 block 維度。**

[FACT:src/enqueue/enqueue.cc:1889-1893]

```c
int nChannels = countOneBits(plan->channelMask);
void* sym = plan->kernelFn;
dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
```

`nChannels`是`channelMask`中置位的個數，也就是這個 plan 要啟動多少個 block。每個 block 負責一個 channel。`threadPerBlock`是在`scheduleCollTasksToPlan`裡透過`plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE)`計算出來的，取所有 task 中最大的`nWarps * 32`。

`smem`是動態共享記憶體大小。對於普通 kernel，它是`ncclShmemDynamicSize(comm->cudaArch)`，這是一個編譯期常數，取決於架構（sm70+ 是`ncclShmemScratchWarpSize * (NCCL_MAX_NTHREADS / WARP_SIZE)`）。對於 sym kernel，它是`plan->kernelDynSmem`，因為 sym kernel 的共享記憶體需求可能不同。

**第二步：組裝 kernel 參數。**

[FACT:src/enqueue/enqueue.cc:1902-1903]

```c
void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
                 CU_LAUNCH_PARAM_END};
```

這是 CUDA 驅動 API 的一種參數傳遞方式：`CU_LAUNCH_PARAM_BUFFER_POINTER`告訴驅動「參數不是一個個傳的，而是一個連續的記憶體塊」，`CU_LAUNCH_PARAM_BUFFER_SIZE`告訴驅動這個塊的大小。這樣做的好處是 NCCL 可以把`ncclDevKernelArgs`和後面的 batch 陣列一次性傳進去，不需要逐個參數打包。

**第三步：加入 launch attributes。**

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

CGA（Cooperative Group Array）是 sm90 引入的硬體特性，允許把多個 block 組成一個 cluster，cluster 內的 block 可以保證同時排程到一組 SM 上，並且可以互相存取共享記憶體。NCCL 用這個特性來實現 NVLS 等需要跨 block 同步的演算法。

注意`if (grid.x % clusterSize) clusterSize = 1;`這個保護：cluster 維度必須能整除 grid 維度，否則驅動會報錯。如果`grid.x`不能被`clusterSize`整除，就退化為不使用 cluster。

**第四步：加入 launch completion event。**

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

`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`是 CUDA 12.3 引入的特性：驅動會在 kernel 真正開始執行時（而不是在 host 側呼叫返回時）記錄一個事件。這對於實現「隱式順序」（implicit order）至關重要——NCCL 需要保證多個 kernel 按順序執行，但又不希望 host 側阻塞等待。

`getImplicitOrder`的邏輯是：如果使用者設定了`launchOrderImplicit`，並且驅動版本足夠新，就使用`ncclImplicitOrderLaunch`（用 launch event 排序）；否則使用`ncclImplicitOrderSerial`（用 completion event 排序，即串行執行）。

**第五步：呼叫`cuLaunchKernelEx`。**

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

`cuLaunchKernelEx`是 CUDA 12.0 引入的新 API，支援 launch attributes。對於舊驅動（< 11.8），NCCL 會退回到`cuLaunchKernel`：

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

## 並發控制與硬體互動

**Launch completion event 的 relay 機制。**當使用`ncclImplicitOrderLaunch`且使用者提供了`launchCompletionEvent`時，NCCL 不能直接把使用者的 event 傳給驅動，因為驅動只支援一個 launch completion event。NCCL 的做法是：

1. 把`comm->sharedRes->launchEvent`傳給驅動。

2. 在`relayStream`上等待`launchEvent`。

3. 在`relayStream`上記錄使用者的 event。

這樣使用者的 event 會在 kernel 真正開始執行後觸發，而不是在 host 側呼叫返回時觸發。

**Mem Sync Domain。** [FACT:src/enqueue/enqueue.cc:1938-1942]在 sm90+ 上，NCCL 設定`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`為`cudaLaunchMemSyncDomainRemote`。這是 Hopper 架構引入的記憶體同步域機制，用於隔離不同 kernel 的記憶體屏障，減少不必要的同步開銷。

## 生產踩坑指南

**踩坑點 1：cluster 維度不整除導致啟動失敗。**如果`grid.x`不能被`clusterSize`整除，驅動會返回`CUDA_ERROR_INVALID_VALUE`。原始碼裡透過`if (grid.x % clusterSize) clusterSize = 1;`做了保護，但這也意味著 cluster 特性被靜默停用了。如果使用者期望 cluster 帶來的效能提升，需要檢查`cgaClusterSize`和`nChannels`的關係。

**踩坑點 2：驅動版本不滿足導致 kernel 不可用。** `ncclInitKernelsForDevice`會在初始化時檢查每個 kernel 的驅動要求：

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

這段程式碼的核心是計算`fnsOfBitset`：對於`offsetBitset`中第 n 個置位，它的位索引是多少。PTX 有`fns`指令可以做這個，但它展開成很多 SASS 指令。NCCL 的做法是用共享記憶體：每個 lane 檢查自己的位是否置位，如果是，計算它前面有多少個置位，然後把自己的 lane 編號寫到`fnsOfBitset[nWorksBelow]`。

接下來是實際的拷貝：

[FACT:src/device/common.h:209-241]

```c
if (tid = %d && __CUDA_ARCH__ >= %d\n" % (cudart ,arch))
  out("/*%4d*/ %s,\n" % (index, sym))
  if (cudart, arch) != (0, 0):
    out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
  index += 1
out("nullptr};\n")
```

## 設計思考與生產踩坑

**為什麼用`__grid_constant__`？** [FACT:src/device/common.h:19-24]

```c
#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif
```

`__grid_constant__`告訴編譯器這個參數是唯讀的，可以放在常量記憶體裡。這樣裝置側讀取時走`ld.param`指令，比從全域記憶體讀取快。註解裡提到它會破壞 cuda-gdb，所以只在 sm70+ 上啟用。

**踩坑點 1：`workStorage`溢出。** `workStorage`的大小是`ncclMaxDevWorkBatchBytes()`，sm90+ 是 16KB。如果`nWorks * workSize`超過這個值，會寫越界。原始碼裡透過`NCCL_MAX_DEV_WORK_BATCH_BYTES`在 host 側限制了 batch 的大小，但裝置側沒有額外的檢查。如果 host 側的約束被繞過（比如透過修改環境變數），會導致共享記憶體越界。

**踩坑點 2：`__syncthreads()`的缺失導致資料競爭。**在`loadWorkBatchToShmem`之後，必須有一個`__syncthreads()`才能讓所有執行緒看到完整的`workStorage`。原始碼裡在[FACT:src/device/common.h:479]有`__syncthreads(); // publish ncclShmem`。如果這個同步被去掉，某些執行緒可能會在`workStorage`還沒寫完時就開始讀取，導致讀到垃圾資料。

**踩坑點 3：abort 檢查的時機。** `while (ncclShmem.aborted == 0)`只在每個 batch 開始時檢查 abort。如果某個 batch 執行時間很長，abort 信號可能要等很久才能生效。這是設計上的權衡：更頻繁的檢查會增加開銷，但回應更快。

# Kernel 變體選擇：generate.py 如何生成 kernel 列表

## 直覺模型

`generate.py`的角色類似於「汽車工廠的生產線規劃師」。它面對一個巨大的組合空間（7 種集合操作 × 5 種歸約操作 × 12 種資料類型 × 7 種演算法 × 3 種協定），需要決定：哪些組合需要生成專門的 kernel？哪些可以共用一個通用 kernel？

如果每個組合都生成一個 kernel，編譯時間和二進位大小會爆炸。如果只生成一個通用 kernel，執行時會因為函式指標呼叫和分支判斷而變慢。`generate.py`的解決方案是「代表性 kernel」：為每個等價類生成一個 kernel，執行時透過函式指標表分發。

## 資料結構與記憶體佈局

`generate.py`生成三個關鍵檔案：

1. **`device_table.cu`**：裝置側的`ncclDevFuncTable`，把 funcId 映射到具體的裝置函式。

2. **`host_table.cc`**：host 側的`ncclDevKernelList`、`ncclDevKernelForFunc`、`ncclDevFuncRowToId`等表。

3. **各個`<coll>_<op>_<ty>.cu`**：具體的 kernel 實作。

## Step-by-Step Walkthrough

**第一步：列舉所有函式行。**

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

這個列舉順序必須和`ncclDevFuncId()`的計算公式匹配：

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

`ncclDevFuncId`計算出的是「行號」，然後透過`ncclDevFuncRowToId`映射到「主函式 ID」。這個映射的原因是：很多行可能映射到同一個主函式（比如所有`AllReduce Sum i32`的行都映射到`AllReduce Sum u32`的主函式）。

**第二步：計算主函式和 kernel 函式。**

[FACT:src/device/generate.py:211-225]

```python
func_rows = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}
kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))
```

`equivalent_primary`把有符號整數映射到無符號整數（因為加法/乘法對兩者是一樣的）：

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

`best_kernel`把多個主函式映射到同一個 kernel（比如所有`AllGather`的演算法都映射到`AllGather RING LL`）：

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

**第三步：生成 kernel 定義。**

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

`DEFINE_ncclDevKernel`巨集展開後是：

[FACT:src/device/common.h:507-509]

```c
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain, algo, proto>>(&args4K.args); \
  }
```

所以每個 kernel 都是一個`__global__`函式，呼叫`ncclKernelMain`，模板參數是`specializedFnId`和`RunWorkBatch<coll, ty, redop<ty>, algo, proto>`。

## 設計思考與生產踩坑

> **[Design Inference & Architectural Trade-offs]**
> **為什麼用「代表性 kernel」而不是每個組合一個 kernel？**編譯時間和二進位大小的權衡。完整的組合空間是 7 × 5 × 12 × 7 × 3 ≈ 8820 個 kernel，每個 kernel 編譯需要幾秒鐘，總共需要幾個小時。而且二進位大小會達到幾百 MB。透過映射到代表性 kernel，實際生成的 kernel 數量減少到幾十個。

**踩坑點 1：`NCCL_EXACT_KERNEL_NAMES`導致編譯爆炸。**如果設置了這個環境變數，`best_kernel`會回傳原始函式，每個組合都會生成一個 kernel。這在開發時有用（可以精確控制哪個 kernel 被編譯），但在生產環境會導致編譯時間過長。

**踩坑點 2：`required_cuda`的版本檢查。**某些 kernel 需要特定的 CUDA 版本或架構：

[FACT:src/device/generate.py:130-154]

至此，kernel 已經在 GPU 上啟動，裝置側也拿到了工作描述。但真正決定效能的，是裝置內部如何搬運資料。下一章將深入 src/device 下的三種協定原語：LL、LL128 和 Simple，看看同一份 AllReduce 邏輯為什麼需要三套搬運原語，以及它們在同步方式、緩衝區佈局和 flag 語意上的差異。
