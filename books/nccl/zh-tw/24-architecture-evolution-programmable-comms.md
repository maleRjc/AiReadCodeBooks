# 第 24 章：架構演進與未來方向：從靜態通訊到可程式化通訊

上一章我們看到社群如何圍繞 NCCL 核心構建周邊生態：Python 綁定、Rust 綁定、專家並行通訊、超頻寬原語、通訊檢查點。這些專案都在復用 NCCL 的穩定 API，但它們的訴求已經超出了傳統集合通訊的範疇——專家並行需要細粒度的點對點收發，檢查點需要暫停/恢復通訊狀態，超頻寬原語需要繞過標準集合操作直接操作網路。這些訴求指向同一個問題：NCCL 的固定集合操作模型，正在被更靈活的通訊需求撐破。本章我們不再看某個單一模組，而是從原始碼中已經出現的演進痕跡出發，討論 NCCL 正在走向何方。具體來說，我們將剖析三股交織的演進力量：通訊原語從固定集合走向可程式化——src/rma/rma.cc 中的 RMA 任務調度，讓上層可以組合 Put/Signal/WaitSignal 原語，而不是只能呼叫 AllReduce；網路發起從 host proxy 走向 GPU 直發——src/gin/gin_host.cc 中的 GIN 後端管理，讓 GPU kernel 直接驅動網卡；記憶體模型從註冊緩衝區走向對稱記憶體——src/sym_kernels.cc 中的對稱記憶體 kernel 選擇，讓所有 rank 用同一套虛擬位址存取彼此的緩衝區。這三股力量不是孤立的，它們共享同一個基礎設施：src/nccl_device/core.cc 中的 team 抽象和 src/devcomm/devcomm_v23100.cc 中的版本化 DevComm。理解它們如何咬合，就理解了 NCCL 從「集合通訊庫」到「可程式化通訊引擎」的演進邏輯。

# 一、可程式化通訊原語：RMA 如何把「固定菜譜」變成「自助餐」

## 直覺模型

傳統 NCCL 的集合通訊像一份固定套餐：你點 AllReduce，廚房就按 AllReduce 的流程做完。但專家平行（MoE）場景下，每個 token 要發給不同的專家，發送模式在編譯期根本不知道——這就像自助餐，你得自己決定拿什麼、拿多少、什麼時候拿。

RMA 就是 NCCL 給上層提供的「自助餐台」：Put（把資料寫到對端記憶體）、Signal（通知對端）、WaitSignal（等待對端信號）。上層框架可以自由組合這三個原語，實現任意通訊模式。

如果沒有 RMA，MoE 的 all-to-all 只能靠多次小規模集合操作模擬，每次都要走完整的 kernel 啟動和同步流程，延遲高得無法接受。

## 資料結構與記憶體佈局

RMA 的核心資料結構是`ncclTaskRma`（任務描述）和`ncclRmaArgs`（計劃參數）。我們先看`ncclRmaArgs`的欄位，它在`scheduleRmaTasksToPlan`中被初始化。

[FACT:src/rma/rma.cc:166-171]

```cpp
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
plan->rmaArgs->nRmaTasks = 0;
plan->rmaArgs->nRmaTasksProxy = 0;
plan->rmaArgs->nRmaTasksCe = 0;
```

這裡的關鍵欄位是`nRmaTasksProxy`和`nRmaTasksCe`。它們把 RMA 任務分成兩條執行路徑：

- **CE 路徑**（Copy Engine，拷貝引擎）：目標 rank 在 LSA（Local Symmetric Access，本地對稱存取）範圍內，可以用 GPU 的拷貝引擎直接完成，不需要網路。
- **Proxy 路徑**：目標 rank 不在 LSA 範圍內，必須走 host proxy 執行緒驅動網路。

> **[Design Inference & Architectural Trade-offs]**
> 這種二分法的設計動機很直接：LSA 範圍內的通訊走 NVLink 或 PCIe，頻寬高、延遲低，用 CE 非同步拷貝最划算；跨機通訊必須走網卡，只能由 proxy 執行緒驅動。把兩類任務分開排程，才能讓 CE 和 proxy 平行執行，而不是串列等待。

`ncclTaskRma`本身包含`peers`、`nsignals`、`signalIdxs`三個陣列指標，分別記錄對端 rank、信號數量、信號索引。對於 WaitSignal 任務，一個任務可以等待多個 peer；對於 Put/Signal 任務，一個任務只針對一個 peer。

## Step-by-Step Walkthrough：一次 WaitSignal 的排程

我們代入一個具體場景：rank 0 呼叫`ncclWaitSignal`，等待 rank 1 和 rank 3 的信號。假設 rank 1 在 LSA 範圍內，rank 3 不在。

**第一步：找到第一個非空上下文佇列。**

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

RMA 任務按 context 分佇列，每個 context 是一個獨立的 RMA 通道。這裡找到第一個有任務的 context，取出它的佇列。

**第二步：取出第一個任務，判斷類型。**

[FACT:src/rma/rma.cc:163-168]

```cpp
struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
```

`firstTask->func`是`ncclFuncWaitSignal`，進入 WaitSignal 分支。

**第三步：按 LSA 可達性拆分 peer。**

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

`isLsaAccessible`遍歷`comm->devrState.lsaRankList`，判斷 peer 是否在 LSA 團隊內。rank 1 在 LSA 內，進 CE 列表；rank 3 不在，進 Proxy 列表。

**第四步：為 CE 和 Proxy 各建立一個新任務。**

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

原來的一個 WaitSignal 任務被拆成兩個：CE 任務等 rank 1，Proxy 任務等 rank 3。兩個任務可以平行執行——CE 路徑在 GPU 上等，Proxy 路徑在 host 執行緒上等。

**第五步：釋放原任務。**

[FACT:src/rma/rma.cc:249-251]

```cpp
planner->nTasksRma -= 1;
ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
```

原任務已經拆成兩個新任務，釋放回記憶體池。

## 並行控制與硬體互動

RMA 的平行執行體現在`ncclRmaWaitSignal`中。

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

這段程式碼用 CUDA event 做流間同步：先在輸入流上記錄 event，讓 CE 流等待這個 event，然後在兩個流上分別啟動 proxy 和 CE 任務，最後讓輸入流等待 CE 流的 event。這樣兩條路徑平行推進，但對外表現為一個同步操作。

> **[Design Inference & Architectural Trade-offs]**
> 這裡的設計權衡是：平行執行能降低延遲，但引入了額外的 event 記錄和流同步開銷。對於小訊息，這個開銷可能超過平行收益；對於大訊息，平行收益顯著。NCCL 沒有在這裡做自適應判斷，而是統一走平行路徑——因為 RMA 的典型場景就是大訊息的細粒度通訊。

## 生產避坑指南

**坑 1：LSA 可達性判斷錯誤導致任務走錯路徑。** `isLsaAccessible`遍歷`lsaRankList`，如果`lsaSize`為 0（比如單 rank 通訊域），所有 peer 都會被判為不可達，全部走 Proxy 路徑。這在小規模測試時不會暴露，但在大規模部署時會導致效能驟降。排查方法是看`scheduleRmaTasksToPlan`的 INFO 日誌中`nRmaTasksProxy`和`nRmaTasksCe`的比例。

**坑 2：WaitSignal 任務拆分後 peer 陣列的生命週期。**CE 路徑的`peersCe`用`ncclMemoryStackAlloc`分配，生命週期跟隨`comm->memScoped`；Proxy 路徑的`peersProxy`用`ncclCalloc`分配，在任務執行完後需要手動`free`。如果 Proxy 任務建立失敗，`fail`分支會釋放這些陣列。

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

**坑 3：Put/Signal 任務的跨 context 批次。**在 Put/Signal 分支中，NCCL 會把所有 context 的 put/signal 任務拉進同一個 plan，但遇到 WaitSignal 就停止。

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

這個設計的意圖是：一次 kernel 啟動涵蓋所有 context 的 put/signal，減少啟動開銷。但每個 context 的佇列只消費到第一個 WaitSignal 為止，保證 per-context FIFO 順序。如果上層在同一個 context 裡交替呼叫 put 和 waitSignal，批次效果會大打折扣——這是使用 RMA 時需要注意的模式。

---

# 二、GPU 直發網路：GIN 如何讓 kernel 繞過 host proxy

## 直覺模型

傳統 NCCL 的網路通訊像寄信：GPU kernel 把資料放到緩衝區，host proxy 執行緒把資料交給網卡，網卡發出去。GIN 則是讓 GPU kernel 直接把信投進對方信箱——kernel 直接寫網卡的發送佇列，網卡直接讀 GPU 顯存。

如果沒有 GIN，每次網路通訊都要經過 host 記憶體中轉，延遲至少多一個 PCIe 往返。對於 MoE 這種細粒度通訊，這個延遲是致命的。

## 資料結構與記憶體佈局

GIN 的核心狀態是`ncclGinState`，它管理多個後端（backend）和多個 DevComm。我們先看後端版本相容表。

[FACT:src/gin/gin_host.cc:27-33]

```cpp
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5), NCCL_VERSION(2, 32, 0)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
constexpr int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0), NCCL_VERSION(2, 32, 0)};
```

這些陣列的索引是後端版本號，值是相容的最低 NCCL 版本。比如`proxyBackendMinVersions[3]`對應後端版本 3，要求 NCCL 至少 2.32.0。這個設計讓 NCCL 可以在執行時根據裝置程式碼版本選擇合適後端版本，而不是編譯期綁定。

> **[Design Inference & Architectural Trade-offs]**
> 這種版本相容表的設計動機是：GIN 後端（網卡驅動、韌體）和 NCCL 函式庫的版本演進節奏不同。如果硬編碼版本要求，任何一方升級都會導致不相容。用陣列做版本映射，可以在執行時動態選擇，向後相容舊後端。

`ncclGinStateDevComm`是每個 DevComm 的 GIN 狀態，包含`contextCount`、`backendIndex`、`ginCtx[]`、`devHandles[]`等欄位。它被串成鏈結串列掛在`ginState->devComms`上。

## Step-by-Step Walkthrough：一次 GIN 連線建立

我們代入一個場景：rank 0 初始化通訊域，需要建立 GIN 連線。

**第一步：檢查 GIN 是否啟用和支援。**

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

`ncclParamGinEnable()`讀取環境變數`NCCL_GIN_ENABLE`，預設 1。如果使用者顯式停用，直接回傳錯誤。

**第二步：檢查對稱記憶體支援。**

[FACT:src/gin/gin_host.cc:111-114]

```cpp
if (!comm->symmetricSupport) {
  WARN("Communicator does not support symmetric memory!");
  return ncclInternalError;
}
```

GIN 依賴對稱記憶體——因為 GPU kernel 需要知道對端緩衝區的虛擬位址，只有對稱記憶體才能保證位址一致。

**第三步：取得本地 GIN 裝置列表。**

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

`ncclTopoGetLocalGinDevs`從拓撲圖中找出所有支援 GIN 的網卡。如果超過`NCCL_GIN_MAX_CONNECTIONS`，只取前幾個並列印警告。

**第四步：計算 GIN 團隊。**

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

每個後端先呼叫`devices`取得裝置數量，然後對每個連線執行 listen→getProperties→allGather→connect→closeListen 的流程。`bootstrapAllGather`在所有 rank 之間交換 handle，這樣每個 rank 都知道對端的連線資訊。

## 並發控制與硬體互動

GIN 的進度執行緒是核心並發機制。

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

這裡有幾個關鍵設計：

1. **CPU 親和性**：`ncclOsSetAffinity`把進度執行緒綁定到指定 CPU 核，避免執行緒遷移帶來的快取失效。

2. **寫鎖退避**：`writePending`是一個原子標誌，主執行緒要修改`devComms`鏈結串列時先置位，進度執行緒看到後主動 yield，避免鎖競爭。

3. **讀寫鎖**：`devCommRwMutex`是`shared_timed_mutex`，進度執行緒持讀鎖遍歷鏈結串列，主執行緒持寫鎖修改鏈結串列。

4. **執行緒分工**：執行緒 t 負責連接 t, t+proxyNthreads, t+2*proxyNthreads, ...，透過 stride 迴圈實現負載均衡。

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

這個寫鎖的實現假設只有一個寫者（主執行緒），所以不需要額外的互斥。`writePending`先置位再拿鎖，確保進度執行緒在拿鎖前就能看到寫意圖，主動退避。

## 生產避坑指南

**坑 1：GIN 連線數不匹配導致 AllGather 死鎖。**每個 rank 的`ginCommCount`可能不同（取決於本地網卡數量），NCCL 透過`bootstrapAllGather`取所有 rank 的最小值。

[FACT:src/gin/gin_host.cc:176-180]

```cpp
ginCommCountHandles[comm->rank] = backend->ginCommCount;
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
for (int r = 0; r nRanks; r++) {
  backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
}
```

如果某個 rank 的網卡數量少於其他 rank，所有 rank 都會降到最小值。這保證了連接對稱，但會浪費網卡資源。

**坑 2：proxyNthreads 超過 ginCommCount 導致執行緒空轉。**如果使用者設定了`NCCL_GIN_PROXY_NTHREADS`大於`ginCommCount`，多餘的執行緒會在 stride 迴圈中空轉。

[FACT:src/gin/gin_host.cc:181-183]

```cpp
// After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
// on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
// Extra threads simply idle in the stride loop; no correctness issue.
```

這不是正確性問題，但會浪費 CPU 資源。排查方法是看`NCCL_GIN_PROXY_NTHREADS`是否大於實際網卡數。

**坑 3：DevComm 釋放時的競態。** `ncclGinDevCommFree`先從鏈結串列摘除 DevComm，再銷毀 context。

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

摘除後，進度執行緒再也看不到這個 DevComm，所以銷毀 context 是安全的。但如果銷毀過程中有 in-flight 的網路操作，可能會導致未定義行為——這是使用 GIN 時需要確保的：釋放 DevComm 前必須確保所有操作已完成。

---

# 三、對稱記憶體 kernel：從「註冊緩衝區」到「統一地址空間」

## 直覺模型

傳統 NCCL 的緩衝區是「註冊制」：每個 rank 註冊自己的緩衝區，通訊時透過 handle 交換地址。對稱記憶體則是「統一地址空間」：所有 rank 約定同一套虛擬地址，rank 0 的地址 A 和 rank 1 的地址 A 指向各自的實體記憶體，但程式碼裡用同一個地址就能存取。

這就像大家約定「第 3 排第 5 座」在每個人家里都指同一個位置，找東西時不用先問「你家第 3 排第 5 座在哪」。

如果沒有對稱記憶體，每個 kernel 都要先解析對端地址，增加了指令開銷和暫存器壓力。

## 資料結構與記憶體佈局

對稱記憶體 kernel 的核心是 kernel mask——一個位元圖，標記哪些 kernel 在當前通訊域中可用。

[FACT:src/sym_kernels.cc:17-63]

```cpp
constexpr uint32_t kernelMask_STMC =
  1  **[Design Inference & Architectural Trade-offs]**
> 這種位元圖設計的好處是：可以用位元運算快速篩選可用 kernel。比如`kmask &= ~kernelMask_STMC`一行就能停用所有 STMC kernel，不需要遍歷列表。

## Step-by-Step Walkthrough：一次 kernel mask 計算

我們代入一個場景：rank 0 要執行 AllReduce，資料型別是 float16，訊息大小 1MB，通訊域有 8 個 rank，全部 NVLink 互聯。

**第一步：取得操作對應的基礎 mask。**

[FACT:src/sym_kernels.cc:304-306]

```cpp
uint32_t kmask = kernelMask_coll(coll);
```

`kernelMask_coll(ncclFuncAllReduce)`回傳`kernelMask_AR`，包含 5 個 AllReduce kernel。

**第二步：檢查 STMC 和 LDMC 可用性。**

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

`hasLsaMultimem`在`ncclSymkInitOnce`中計算，要求 NVLS 對稱多播可用且 LSA 團隊大於 2 個 rank。float16 支援 LDMC，所以如果`hasLsaMultimem`為真，LDMC kernel 保留。

**第三步：檢查訊息大小限制。**

[FACT:src/sym_kernels.cc:336-342]

```cpp
size_t nBytes = alignUp(nElts * ncclTypeSize(ty), NCCL_SYM_KERNEL_CELL_SIZE);
size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks) * nBytes;
if (nBusBytes >= (size_t(2) = 32 * (size_t(2) nRanks;
kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
```

如果 LSA 團隊覆蓋所有 rank，不需要 GIN；否則只保留 GIN kernel。

## 並行控制與硬體互動

對稱記憶體 kernel 的初始化涉及 DevComm 建立和資源分配。

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

這裡的關鍵是`ncclDevrCommCreateInternal`，它建立一個內部 DevComm，包含 LSA 多播、GIN inbox/outbox、訊號等資源。`reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL`指定 GIN 用 rail 連接模式。

[FACT:src/sym_kernels.cc:257-261]

```cpp
symk->kcomm.workStarted = comm->profiler.symWorkStarted;
symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
symk->kcomm.workPhases = comm->profiler.symWorkPhases;
```

對稱記憶體 kernel 使用獨立的 profiler 緩衝區，避免與常規 kernel 的 workCounter 交錯。

## 生產避坑指南

**坑 1：TMA kernel 的 SMEM 需求。**TMA 需要每個 warp 約 8KB 的 SMEM scratch，16 個 warp 就是 128KB。

[FACT:src/sym_kernels.cc:135-142]

```cpp
bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  if (comm->maxSharedMemOptin minCompCap >= 100 && ncclParamSymTmaEnable();
}
```

如果 GPU 的 SMEM 容量不足（比如 MIG 實例），TMA kernel 會被停用。排查方法是看`maxSharedMemOptin`是否小於`ncclTmaShmemScratchWarpSize() * 16`。

**坑 2：GIN chunk size 的邊界。**ReduceScatter GIN kernel 的 chunk size 有上下限。

[FACT:src/sym_kernels.cc:148-153]

```cpp
static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128  0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}
```

如果使用者設定的`NCCL_SYM_RS_GIN_CHUNK_SIZE`超過 1GB，會被截斷到 1GB；如果小於 128 位元組，會被提升到 128 位元組。最終值還會被向下取整到 2 的冪。

**坑 3：對稱記憶體註冊類型不匹配。** `ncclGetSymRegType`根據 sendWin 和 recvWin 的`NCCL_WIN_COLL_SYMMETRIC`標誌判斷註冊類型。

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

如果 send 和 recv 的註冊類型不一致，kernel 需要走不同的程式碼路徑。這會影響效能，但不會導致錯誤。

---

# 四、Team 抽象與版本化 DevComm：演進的基礎設施

## 直覺模型

Team 抽象就像「分組」：世界團隊是全班，LSA 團隊是同桌，Rail 團隊是同一列的座位。不同的通訊模式需要不同的分組視角。

版本化 DevComm 就像「翻譯官」：不同版本的裝置程式碼說不同的「方言」，DevComm 相容層負責翻譯，讓新舊程式碼能互相理解。

如果沒有 Team 抽象，每個 kernel 都要自己計算 rank 映射；如果沒有版本化 DevComm，任何 ABI 變化都會導致所有裝置程式碼重新編譯。

## 資料結構與記憶體佈局

Team 是一個簡單的三元組：`nRanks`、`rank`、`stride`。

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

世界團隊的 stride 是 1，因為所有 rank 連續排列。

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

Rail 團隊的 stride 是`lsaSize`，因為每個 rail 上的 rank 間隔一個 LSA 團隊的大小。

版本化 DevComm 的核心是`ncclDevCommCompat`結構。

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

這個結構定義了版本 2.31.0 的相容性規則。`minVersion`和`maxVersion`定義了適用版本範圍，後面四個函式指標定義了屬性過濾和結構轉換邏輯。如果都是 nullptr，表示這個版本沒有特殊相容需求。

## Step-by-Step Walkthrough：一次 Team 轉換

我們代入一個場景：rank 5 在 8 rank 通訊域中，LSA 團隊大小是 4。要計算 rank 5 在 Rail 團隊中的 rank。

**第一步：初始化 DevR 狀態。**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
```

`ncclDevrInitOnce`計算 LSA 團隊、CFT 團隊等衍生資訊。如果失敗，返回空團隊。

**第二步：計算 Rail 團隊參數。**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
ncclTeam_t ans;
ans.nRanks = comm->nRanks / comm->devrState.lsaSize;  // 8 / 4 = 2
ans.rank = comm->rank / comm->devrState.lsaSize;       // 5 / 4 = 1
ans.stride = comm->devrState.lsaSize;                  // 4
```

rank 5 在 Rail 團隊中的 rank 是 1，團隊有 2 個 rank，stride 是 4。

**第三步：轉換回世界 rank。**

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

如果要把 Rail rank 0 轉成世界 rank：`5 + (0 - 1) * 4 = 1`。驗證：rank 1 和 rank 5 在同一個 rail 上（間隔 4）。

## 並行控制與硬體互動

Team 抽象本身是無狀態的，不需要並行控制。但`ncclDevrInitOnce`是懶載入的，第一次呼叫時會計算所有衍生資訊。

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

註解說「Ignoring errors since if it fails ncclDevrInitOnce will try again」——如果初始化失敗，返回空團隊，下次呼叫會重試。

## 生產避坑指南

**坑 1：Team 轉換的 stride 假設。** `ncclTeamRankToWorld`假設團隊內 rank 是等差數列。

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

如果團隊不是等差數列（比如自訂的任意分組），這個函式會算錯。NCCL 目前只支援規則團隊。

**坑 2：版本化 DevComm 的空指標。** `ncclDevCommCompat_v23100`的所有函式指標都是 nullptr，表示沒有特殊相容邏輯。如果未來版本需要轉換，必須實作這些函式，否則新舊程式碼無法互操作。

**坑 3：CFT 團隊的層級模式。** `ncclTeamCft`支援三種模式：FLAT、HIER_MULTIMEM、HIER_LSA。

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

如果傳入無效模式，返回空團隊。使用 CFT 團隊時需要確保模式正確。

---

# 設計思考

**為什麼 NCCL 要同時支援 RMA、GIN、對稱記憶體三條演進路徑？**

> **[Design Inference & Architectural Trade-offs]**
> 這三條路徑解決的是不同層次的問題：

- **RMA**解決「通訊模式固定」的問題——讓上層可以組合原語，實作任意通訊模式。
- **GIN**解決「網路延遲高」的問題——讓 GPU 直接驅動網卡，繞過 host proxy。
- **對稱記憶體**解決「位址解析開銷」的問題——讓 kernel 直接用統一地址存取對端記憶體。

它們不是替代關係，而是互補關係。RMA 可以用 GIN 作為底層傳輸，GIN 依賴對稱記憶體提供位址一致性。三者共同構成了「可程式化通訊引擎」的基礎設施。

**版本化 DevComm 的設計哲學是什麼？**

> **[Design Inference & Architectural Trade-offs]**
> 版本化 DevComm 的核心思想是「ABI 穩定，API 演進」。裝置程式碼（kernel）編譯後嵌入二進位，不能隨 NCCL 函式庫升級而重新編譯。所以 NCCL 必須保證舊裝置程式碼能在新函式庫上運行。`ncclDevCommCompat`結構就是相容層的入口：新函式庫根據裝置程式碼版本選擇合適的相容規則，必要時做結構轉換。

---

# 本章小結

本章我們從原始碼中的演進痕跡出發，剖析了 NCCL 從集合通訊函式庫走向可程式化通訊引擎的三股力量：

1. **RMA**（`src/rma/rma.cc`）：透過 Put/Signal/WaitSignal 原語組合，讓上層實現任意通訊模式。核心設計是按 LSA 可達性把任務拆成 CE 和 Proxy 兩條路徑並行執行。

2. **GIN**（`src/gin/gin_host.cc`）：透過 GPU 直發網路，繞過 host proxy。核心設計是多後端管理、版本相容表、進度執行緒池。

3. **對稱記憶體 kernel**（`src/sym_kernels.cc`）：透過統一地址空間，消除地址解析開銷。核心設計是 kernel mask 位圖和 TMA/GIN 硬體加速。

4. **Team 抽象與版本化 DevComm**（`src/nccl_device/core.cc`、`src/devcomm/devcomm_v23100.cc`）：為演進提供基礎設施。Team 提供分組視角，版本化 DevComm 提供 ABI 相容。

這些變化對上層框架的影響是深遠的：PyTorch 的 ProcessGroup 可以直接呼叫 RMA 原語實現自訂通訊模式；Megatron 的專家並行可以利用 GIN 降低 all-to-all 延遲；對稱記憶體讓 kernel 程式碼更簡潔。

# 本章思考與自測

Q1：如果把`scheduleRmaTasksToPlan`中 WaitSignal 分支的 LSA 可達性判斷去掉，所有 peer 都走 Proxy 路徑，會有什麼後果？在什麼場景下會觸發效能災難？

**參考解析**：

LSA 可達性判斷在[FACT:src/rma/rma.cc:187-204]，它把 peer 分成 CE 和 Proxy 兩組。如果去掉這個判斷，所有 peer 都走 Proxy 路徑，`nRmaTasksCe`始終為 0。

後果是：CE 路徑完全不被使用，所有 WaitSignal 都透過 host proxy 執行緒輪詢網路。對於 LSA 範圍內的 peer（同機 NVLink 互聯），本來可以用 GPU 拷貝引擎非同步等待，現在變成 host 執行緒輪詢，延遲從微秒級升到毫秒級。

效能災難場景：MoE 訓練中，每個 token 要等待多個專家的信號。如果所有信號都走 Proxy，host 執行緒成為瓶頸，GPU 大量時間在等 host 輪詢。在 8 卡全 NVLink 的機器上，這個退化尤其明顯——本來所有通訊都可以走 CE，現在全部擠到 host。

排查方法：看`scheduleRmaTasksToPlan`的 INFO 日誌，如果`nRmaTasksCe`始終為 0 而`nRmaTasksProxy`很大，說明 LSA 判斷有問題。

Q2：`ncclGinProgress`中`writePending`標誌和`devCommRwMutex`讀寫鎖的配合，如果去掉`writePending`檢查，只保留讀寫鎖，會有什麼問題？

**參考解析**：

`writePending`檢查在[FACT:src/gin/gin_host.cc:63-66]，它讓進度執行緒在主執行緒要寫時主動 yield。如果去掉這個檢查，進度執行緒會直接嘗試拿讀鎖。

問題在於：`std::shared_timed_mutex`的讀鎖是共享的，多個進度執行緒可以同時持有。如果主執行緒要拿寫鎖，必須等所有讀鎖釋放。在高負載下，進度執行緒頻繁拿讀鎖，主執行緒可能長時間拿不到寫鎖，導致`ncclGinDevCommSetup`或`ncclGinDevCommFree`阻塞。

更嚴重的是：如果主執行緒在`ginProgressWriteLock`中先置位`writePending`再拿鎖，而進度執行緒不檢查`writePending`，那麼進度執行緒可能在主執行緒置位後仍然拿讀鎖，導致主執行緒等待時間不可預測。

`writePending`的作用是「軟性通知」：告訴進度執行緒「我要寫了，你們先讓讓」。這比單純依賴鎖的公平性更高效，因為進度執行緒可以主動 yield 而不是阻塞在鎖上。

Q3：`ncclSymkMask`中，如果`nBusBytes >= 32 * (size_t(2) << 30)`時把所有 kernel 都禁用（`kmask = 0`），此時`ncclSymkAvailable`返回 false，NCCL 會回退到什麼路徑？這個回退路徑有什麼效能影響？

**參考解析**：

`kmask = 0`在[FACT:src/sym_kernels.cc:342]，此時`ncclSymkAvailable`返回 false（[FACT:src/sym_kernels.cc:354-361]）。

回退路徑是：NCCL 會使用傳統的集合通訊 kernel（非對稱記憶體 kernel）。這些 kernel 透過註冊緩衝區的方式存取對端記憶體，需要先解析地址，指令開銷更大。

效能影響：對於超大訊息（超過 64GB 總線位元組），傳統 kernel 的地址解析開銷佔比很小，因為資料傳輸本身佔主導。但在邊界情況下（剛好超過 64GB），傳統 kernel 可能比對稱記憶體 kernel 慢 10-20%。

這個限制的根本原因是：對稱記憶體 kernel 用 32 位元整數追蹤 unrolled loop chunk，每個 chunk 至少 32 位元組，所以最大可定址範圍是 32 * 2^31 = 64GB。超過這個範圍會整數溢位。

實際生產中，單次集合通訊超過 64GB 的場景很少（通常是梯度累積後的 all-reduce），但並非不可能。如果遇到這種場景，可以考慮分片通訊或使用傳統 kernel。

---

# 章末過渡

本章我們看到 NCCL 正在從「固定集合操作」走向「可程式化通訊引擎」：RMA 提供原語組合，GIN 提供 GPU 直發，對稱記憶體提供統一地址空間，Team 和版本化 DevComm 提供基礎設施。

這些演進不是孤立的，它們共同指向一個目標：**讓上層框架能夠以更低的延遲、更高的靈活性實現自訂通訊模式**。對於 PyTorch、Megatron 這樣的框架，這意味著它們可以直接在 NCCL 之上構建 MoE all-to-all、流水線並行、專家並行等複雜通訊模式，而不需要繞過 NCCL 自己實現網路層。

下一章是全書最後一章。我們將把一次 AllReduce 的完整鏈路重新走一遍——從`ncclAllReduce`呼叫開始，經過任務入隊、演算法選擇、kernel 啟動、proxy 推進、網路傳輸，直到結果返回。這次回顧會把前面 24 章的知識點串聯起來，形成一個完整的認知地圖。

至此，我們看清了 NCCL 從固定集合操作向可編程通訊引擎演進的三條主線：RMA 原語組合、GPU 直發網路、對稱記憶體模型，以及支撐它們的 team 抽象與版本化 DevComm。這些機制共同指向一個更靈活、更貼近硬體能力的通訊未來。然而，無論架構如何演進，一次 AllReduce 的完整鏈路始終是理解 NCCL 的基石。下一章我們將不引入新程式碼，而是把第 3 章到第 10 章的端到端流程重新串講一遍——從 ncclAllReduce 呼叫，到通訊域建立、拓撲搜尋、演算法選型、任務入隊、kernel 啟動、裝置側原語執行、結果回寫。你將把分散在各章的機制重新組裝成一個完整心智模型，並得到一份「遇到問題該查哪一章」的索引。
