# 第 6 章：算子下發全景：ncclAllReduce 如何變成一個可執行的 kernel 任務

上一章我們走完了 tuning 模組，知道 NCCL 會在微秒級內為一次集合通訊選定 (演算法, 協定, channel, warp) 組合。但選型結果本身只是一堆數字——它需要被「翻譯」成 GPU kernel 能讀懂的任務描述物件，才能被真正執行。本章進入 src/enqueue/enqueue.cc 的主幹，回答一個核心問題：當使用者呼叫 ncclAllReduce 時，host 側到底發生了什麼？從 ncclAllReduce 到 ncclEnqueueCheck，經過參數校驗、演算法/協定確定、channel 切分，最終生成 ncclInfo 與 ncclTaskColl 結構。這是全書從「使用者視角」切換到「引擎視角」的關鍵一章。如果把 NCCL 比作一家餐廳，那麼 enqueue 模組就是「前台點單系統」：使用者（應用層）說「我要一份 AllReduce」，前台把它翻譯成廚房（GPU kernel）能執行的工單——幾號灶台、用什麼鍋、分幾批做。沒有這個翻譯層，廚房根本不知道要做什麼菜。

# 一、入口：ncclAllReduce 如何建構 ncclInfo

## 直覺模型

`ncclAllReduce`是使用者直接呼叫的 API 函式。它的職責極其單一：**把使用者傳入的裸參數打包成一個`ncclInfo`結構體，然後交給`ncclEnqueueCheck`**。這就像你去銀行櫃檯辦業務，櫃員先把你的需求填進一張標準表單，再轉交給後台系統。

如果沒有這一層，每個集合通訊 API 都要自己處理參數校驗、group 語意、profiler 埋點——程式碼會重複到無法維護。

## 資料結構：ncclInfo 的記憶體佈局

`ncclInfo`是貫穿整個 enqueue 流程的核心載體。它的定義在`src/include/info.h`：

[FACT:src/include/info.h:17-44]

這個結構體有 20+ 個欄位，我們可以按功能分成四組：

| 欄位組 | 欄位 | 作用 |
| --- | --- | --- |
| 集合通訊參數 | `coll`, `sendbuff`, `recvbuff`, `count`, `datatype`, `op`, `root` | 描述「做什麼」 |
| 通訊域與流 | `comm`, `stream` | 描述「在哪做」 |
| 演算法細節 | `chunkSteps`, `sliceSteps` | 描述「怎麼切分」 |
| 單邊操作 | `peerWinOffset`, `peerWin`, `sigIdx`, `ctx`, `flags`, `nDesc`, `signalDescs` | RMA 專用 |
| 使用者配置 | `collConfig` | 從使用者 config 拷貝的私有副本 |

注意`collConfig`的註解：**"A config copied from config passed by user so older user config can be safely accessed during synchronous host scheduling (never at launch/replay)"** [FACT:src/include/info.h:41-43]。這是一個關鍵設計——使用者傳入的 config 指標可能在`ncclGroupEnd`之前就被銷毀，所以 NCCL 在`ncclInfo`裡做了一份拷貝。

## Step-by-Step：ncclAllReduce 的呼叫鏈

我們以`ncclAllReduce`為例，追蹤從使用者呼叫到`ncclInfo`建構的完整路徑。

**第 1 步：使用者呼叫 ncclAllReduce。**入口在`src/collectives.cc`：

[FACT:src/collectives.cc:206-211]

這裡做了三件事：

1. `NVTX3_FUNC_WITH_PARAMS`打 NVTX 標記（用於 Nsight 等工具視覺化）

2. 呼叫`ncclAllReduceConfigImpl`，傳入`config = nullptr`

3. 回傳結果

**第 2 步：ncclAllReduceConfigImpl 建構 ncclInfo。**這是關鍵的一步：

[FACT:src/collectives.cc:192-202]

注意這裡用了 C 風格的聚合初始化：

```c
struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                        sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
                        ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
```

欄位按`ncclInfo`的宣告順序一一對應。`ALLREDUCE_CHUNKSTEPS`和`ALLREDUCE_SLICESTEPS`定義在`src/include/collectives.h`：

[FACT:src/include/collectives.h:19-20]

`NCCL_STEPS`是環形緩衝區裡的步數（通常為 8 或 16），所以 AllReduce 的 chunkSteps 是`NCCL_STEPS/2`，sliceSteps 是`NCCL_STEPS/4`。這意味著一個 chunk 包含 2 個 slice。

**第 3 步：解析使用者 config。** `ncclParseCollConfig`把使用者傳入的`ncclCollConfig_t*`解析進`info.collConfig`。如果`config == nullptr`，這個欄位保持零初始化。

**第 4 步：交給 ncclEnqueueCheck。**這是 enqueue 模組的真正入口。

## 設計思考：為什麼用聚合初始化而不是逐欄位賦值？

> **[Design Inference & Architectural Trade-offs]**
> 聚合初始化有兩個好處：一是編譯器會檢查欄位數量是否匹配（少一個欄位會警告），二是程式碼更緊湊。但缺點是**欄位順序必須與結構體宣告嚴格一致**——如果有人在`ncclInfo`中間插入一個欄位，所有聚合初始化點都會靜默錯位。這是 NCCL 程式碼裡一個隱含的維護風險。

## 生產踩坑：config 生命週期

一個真實的踩坑場景：使用者這樣寫程式碼：

```c
ncclCollConfig_t config = {...};
ncclAllReduceConfig(..., &config);
// config 在这里被销毁（比如是栈变量，函数返回了）
```

如果 NCCL 沒有在`ncclInfo`裡拷貝 config，那麼`ncclGroupEnd`時存取`info.collConfig`就會讀到已釋放的記憶體。`src/include/info.h:41-43`的註解正是為了說明這個設計——**config 在 task append 階段就被解析並拷貝，之後不再依賴使用者指標**。

---

# 二、ncclEnqueueCheck：參數校驗與 group 語意

## 直覺模型

`ncclEnqueueCheck`是 enqueue 模組的「總閘門」。所有集合通訊 API 最終都匯聚到這裡。它的職責是：**校驗參數合法性、處理 group 語意、呼叫 taskAppend 生成任務**。如果把它比作機場安檢，那麼每個 API 函式就是值機櫃檯——值機只是收行李，真正的安檢在`ncclEnqueueCheck`。

如果沒有這一層，每個 API 都要自己寫一遍參數校驗和 group 處理，程式碼會膨脹數倍，而且容易漏掉某個校驗。

## Step-by-Step：ncclEnqueueCheck 的執行流程

[FACT:src/enqueue/enqueue.cc:3478-3527]

我們逐步拆解：

**第 1 步：CommCheck 校驗通信域。** `CommCheck(info->comm, info->opName, "comm")`檢查 comm 指標是否非空、是否已初始化。如果 comm 被 revoke（比如某個 rank 出錯），直接返回錯誤：

[FACT:src/enqueue/enqueue.cc:3480-3485]

**第 2 步：處理 profiler 深度。**如果已經在 group 內部（`profilerGroupDepth > 0`），遞增深度計數。這是為了正確處理隱式的`ncclGroupStartInternal`/`ncclGroupEndInternal`調用。

**第 3 步：進入內部 group。** `ncclGroupStartInternal()`是 NCCL 內部的 group 機制。**關鍵點**：即使用戶沒有顯式調用`ncclGroupStart`，NCCL 也會為每次 API 調用創建一個隱式 group。這保證了單次調用的原子性。

**第 4 步：確保 comm 就緒。** `ncclCommEnsureReady(info->comm)`等待通信域初始化完成（比如 bootstrap 完成、連接建立）。

**第 5 步：ArgsCheck 參數校驗。**這是最複雜的校驗步驟：

[FACT:src/enqueue/enqueue.cc:3497-3503]

注意`checkMode`的處理：如果是`ncclCheckModeDebugGlobal`，`ArgsCheck`會把 info 入隊，等`ncclGroupEnd`時做全局校驗（比如檢查所有 rank 的 count 是否一致）。

**第 6 步：調用 taskAppend。**這是核心轉換步驟：

[FACT:src/enqueue/enqueue.cc:3513]

**第 7 步：遞增 opCount。**每次成功入隊後，`comm->opCount++`。這個計數器用於匹配 send/recv 操作，也是 profiler 的時間線依據。

**第 8 步：退出 group。** `ncclGroupEndInternal()`如果 depth 降到 0，會觸發真正的 group 操作（調度、啟動 kernel）。

## 並發控制：group 語義與線程安全

> **[Design Inference & Architectural Trade-offs]**
> `ncclGroupStartInternal`/`ncclGroupEndInternal`使用線程局部存儲（TLS）來維護 group 狀態。這意味著**同一個線程內的多個 API 調用會被合併成一個 group**，但不同線程的調用是獨立的。這是 NCCL 支持多線程調用的基礎。

一個容易踩的坑：如果用戶在`ncclGroupStart`和`ncclGroupEnd`之間調用了非 NCCL 的 CUDA API（比如`cudaMemcpy`），可能會導致 stream 順序問題。NCCL 的 group 機制假設 group 內的操作都在同一組 stream 上。

## 錯誤恢復鏈

`ncclEnqueueCheck`的錯誤處理有一個精巧的設計：

[FACT:src/enqueue/enqueue.cc:3524-3526]

如果`taskAppend`失敗，且 comm 是非阻塞模式，會調用`ncclCommSetAsyncError`記錄錯誤。這樣後續的 API 調用會立即返回錯誤，而不是繼續嘗試。這是異步錯誤傳播機制。

---

# 三、taskAppend：任務分發的十字路口

## 直覺模型

`taskAppend`是 enqueue 模組的「交通樞紐」。它根據`info->coll`的值，把任務分發到不同的處理路徑：P2P、RMA、CE、或者普通集合通信。這就像一個郵局分揀中心——根據信封上的地址，把信件投到不同的郵筒。

如果沒有這個分發層，所有類型的操作都要擠在一個巨大的 if-else 裡，代碼會難以維護。

## Step-by-Step：taskAppend 的分發邏輯

[FACT:src/enqueue/enqueue.cc:3337-3476]

**第 1 步：判斷是否啟用新架構。** `ncclParamEnqueueRearchEnable()`是一個環境變量開關（默認 0）。如果啟用，走`rawTaskAppend`路徑——這是 NCCL 正在開發的新任務模型。

**第 2 步：P2P 分發。**如果是 Send/Recv，調用`p2pTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3343-3345]

**第 3 步：RMA 分發。**如果是 PutSignal/Signal/WaitSignal，調用`rmaTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3346-3347]

**第 4 步：空集合通信提前返回。** `if (info->count == 0) return ncclSuccess;`——count 為 0 的集合通信直接丟棄。

**第 5 步：算法選擇校驗。** `ncclCollConfigGetAlgMask`校驗用戶傳入的算法選擇是否合法：

[FACT:src/enqueue/enqueue.cc:3357-3358]

**第 6 步：FP8 類型檢查。**FP8 歸約需要 sm90+：

[FACT:src/enqueue/enqueue.cc:3360-3366]

**第 7 步：歸約操作轉換。** `hostToDevRedOp`把 host 側的`ncclRedOp_t`轉換成設備側的`ncclDevRedOpFull`：

[FACT:src/enqueue/enqueue.cc:3370-3371]

**第 8 步：單 rank 提前返回。**如果`comm->nRanks == 1`，直接調用`ncclLaunchOneRank`執行本地歸約，不需要生成任務：

[FACT:src/enqueue/enqueue.cc:3373-3377]

**第 9 步：多 rank 路徑。**這是最複雜的分支，包含 CE 路由、AllToAll/Gather/Scatter 降級、以及普通集合通信：

[FACT:src/enqueue/enqueue.cc:3378-3470]

## 數據結構：ncclTaskColl 的字段

`collTaskAppend`是生成`ncclTaskColl`的地方。我們看它的核心邏輯：

[FACT:src/enqueue/enqueue.cc:2757-2851]

關鍵字段賦值：

| 字段 | 來源 | 含義 |
| --- | --- | --- |
| `func` | `info->coll` | 集合通信類型 |
| `sendbuff`/`recvbuff` | `info->sendbuff`/`recvbuff` | 緩衝區指標 |
| `count` | `info->count` | 元素數量 |
| `datatype` | `info->datatype` | 數據類型 |
| `trafficBytes` | `count * elementSize * ncclFuncTrafficPerByte` | 流量估算 |
| `opHost`/`opDev` | `info->op`/`opDev` | 歸約操作 |
| `chunkSteps`/`sliceSteps` | `info->chunkSteps`/`sliceSteps` | 切分步數 |
| `minCTAs`/`maxCTAs`/`nvlsCTAs` | 配置解析 | 資源上限 |
| `algMask` | `ncclCollConfigGetAlgMask` | 算法選擇掩碼 |

注意`trafficBytes`的計算：

[FACT:src/enqueue/enqueue.cc:2813]

`ncclFuncTrafficPerByte`返回每種集合通信的流量倍數：

[FACT:src/enqueue/enqueue.cc:123-134]

AllReduce 返回 2（因為要 reduce + broadcast），AllGather/ReduceScatter 返回 nRanks，其他返回 1。

## 設計思考：為什麼 AllGather/Broadcast 要轉成 int8？

[FACT:src/enqueue/enqueue.cc:2808-2812]

AllGather 和 Broadcast 把 count 乘以 elementSize，然後把 datatype 改成`ncclInt8`。這是一個優化：**這兩種操作不涉及歸約，所以不需要關心資料類型，統一按位元組處理可以簡化 kernel 邏輯**。

## 生產踩坑：CTAPolicy 的解析順序

[FACT:src/enqueue/enqueue.cc:3390-3397]

CTAPolicy 的解析有一個微妙的優先級：**env > per-call > comm**。而且`NCCL_CTA_POLICY_ZERO`優先於`NCCL_CTA_POLICY_EFFICIENCY`。如果使用者同時設定了這兩個標誌，ZERO 會生效。

一個真實的踩坑場景：使用者設定了`NCCL_CTA_POLICY=EFFICIENCY`，但發現 CE 路徑沒有被使用。原因是 CE 路由要求`CTAPolicy & NCCL_CTA_POLICY_ZERO`為真，而 EFFICIENCY 不滿足這個條件。

---

# 四、ncclPrepareTasks：從任務列表到調度佇列

## 直覺模型

`ncclPrepareTasks`是 enqueue 模組的「預處理器」。它把散亂的任務列表按 (func, op, datatype) 分桶，然後為每個桶計算演算法和協定。這就像一個圖書館管理員——先把還回來的書按類別分好，再決定每類書放在哪個書架。

如果沒有這一步，後續的`scheduleCollTasksToPlan`就要為每個任務單獨計算演算法，效率極低。

## Step-by-Step：ncclPrepareTasks 的分桶邏輯

[FACT:src/enqueue/enqueue.cc:423-642]

**第 1 步：Broadcast 任務轉換。**如果只有一個 broadcast peer，把 broadcast 任務轉成 coll 任務：

[FACT:src/enqueue/enqueue.cc:430-461]

注意這裡把`bcastTask`的欄位拷貝到新的`ncclTaskColl`，並計算`trafficBytes`。然後從`memPool_ncclTaskBcast`釋放原任務。

**第 2 步：按 (func, op, datatype) 分桶。**任務從 sorter 出來是按 size 降序的，然後被分到`tasksByFnOpTy`陣列：

[FACT:src/enqueue/enqueue.cc:464-487]

索引計算：`((int)task->func * ncclNumDevRedOps + (int)task->opDev.op) * ncclNumTypes + (int)task->datatype`。這是一個三維陣列的線性化。

**第 3 步：聚合與演算法選擇。**對每個桶，聚合大小相近的任務（4 倍以內），然後呼叫`ncclGetAlgoInfo`：

[FACT:src/enqueue/enqueue.cc:503-547]

**第 4 步：按 (collnet, nvls) 分桶。**根據演算法類型，把任務分到`collBins[2][2]`：

[FACT:src/enqueue/enqueue.cc:517-544]

**第 5 步：拼接最終佇列。**把四個桶拼接成`planner->collTaskQueue`：

[FACT:src/enqueue/enqueue.cc:553-557]

## 資料結構：ncclTaskCollSorter

`ncclTaskCollSorter`是一個按`trafficBytes`排序的插入式排序器。`ncclTaskCollSorterInsert`把任務插入到正確位置，`ncclTaskCollSorterDequeueAll`按順序取出所有任務。

> **[Design Inference & Architectural Trade-offs]**
> 這個排序器的設計動機是：**大任務優先調度**。因為大任務的傳輸時間長，先啟動它們可以更好地重疊計算和通訊。

## 並發控制：runtimeConn 與連線建立

[FACT:src/enqueue/enqueue.cc:572-583]

如果`comm->runtimeConn`為真（執行時連線模式），且某個演算法的 channel 還沒初始化，就標記`algoNeedConnect`。這會在後續觸發連線建立。

## 生產踩坑：聚合的邊界條件

[FACT:src/enqueue/enqueue.cc:507-508]

聚合條件是`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes`，且兩個任務都不設定`aggIsolate`。如果使用者設定了 per-call config（比如`maxCTAs`），`aggIsolate`會被設為 true，這個任務就不會被聚合。

一個真實的踩坑場景：使用者為某個 AllReduce 設定了`maxCTAs=4`，期望它只用 4 個 CTA。但由於聚合邏輯，這個任務可能和相鄰任務合併，導致實際使用的 CTA 數量不符合預期。解決方案是設定`aggIsolate`——NCCL 在`collTaskAppend`裡已經處理了這一點：

[FACT:src/enqueue/enqueue.cc:2821-2822]

---

# 五、scheduleCollTasksToPlan：channel 切分與預算控制

## 直覺模型

`scheduleCollTasksToPlan`是 enqueue 模組的「調度器」。它把任務分配到具體的 channel，並計算每個 channel 的資料切分。這就像一個工廠的排產系統——決定每條生產線做什麼、做多少。

如果沒有這一步，GPU kernel 就不知道自己要處理哪部分資料。

## Step-by-Step：channel 切分演算法

[FACT:src/enqueue/enqueue.cc:644-947]

**第 1 步：預算估算。**先估算能放進這個 plan 的任務數量：

[FACT:src/enqueue/enqueue.cc:648-689]

`ncclTestBudget`檢查工作位元組數是否超出預算：

[FACT:src/enqueue/enqueue.cc:343-349]

**第 2 步：計算每個 channel 的流量。**根據 kind（collnet/nvls）計算`trafficPerChannel`：

[FACT:src/enqueue/enqueue.cc:701-707]

**第 3 步：Collnet 路徑。**如果是 collnet 演算法，channel 分配比較簡單：

[FACT:src/enqueue/enqueue.cc:709-739]

**第 4 步：普通路徑的 cell 切分。**這是最複雜的部分。NCCL 把資料切成 "cell"，每個 cell 是一個最小傳輸單元：

[FACT:src/enqueue/enqueue.cc:740-845]

關鍵變數：

- `cellSize`：每個 cell 的位元組數，至少`MinTrafficPerChannel`（32KB）
- `cells`：總 cell 數
- `cellsPerChannel`：每個 channel 處理的 cell 數
- `cellsLo`/`cellsHi`：首尾 channel 的 cell 數（可能不滿）

**第 5 步：計算 chunkGrains。**對每個 channel 段呼叫`calcCollChunking`：

[FACT:src/enqueue/enqueue.cc:811-825]

**第 6 步：生成 proxyOp。**為每個 channel 生成 proxy 操作：

[FACT:src/enqueue/enqueue.cc:844-894]

## 資料結構：ncclDevWorkColl

`ncclDevWorkColl`是裝置側的工作描述符。它的關鍵欄位：

| 欄位 | 含義 |
| --- | --- |
| `sendbuff`/`recvbuff` | 緩衝區指標 |
| `channelLo`/`channelHi` | channel 範圍 |
| `cbd.countLo`/`countMid`/`countHi` | 各段元素數 |
| `cbd.chunkGrainsLo`/`Mid`/`Hi` | 各段 chunk 粒度 |
| `direct` | 直接標誌 |

## 並發控制：channelMask 的位元運算

[FACT:src/enqueue/enqueue.cc:897]

這行程式碼用位元運算設定 channelMask：`(2ull << channelHi) - (1ull << channelLo)`。比如 channelLo=2, channelHi=5，結果是`(2<<5) - (1<<2) = 64 - 4 = 60 = 0b111100`，即 bit 2-5 被設定。

## 生產踩坑：預算溢出

[FACT:src/enqueue/enqueue.cc:792-794]

如果預算不夠，直接返回`ncclSuccess`，讓外層迴圈建立新的 plan。這是一個優雅的降級策略——**不報錯，只是分批處理**。

一個真實的踩坑場景：如果`NCCL_WORK_FIFO_BYTES`設定得太小，會導致每個 plan 只能容納很少的任務，增加 kernel 啟動次數，降低效能。

---

# 六、finishPlan：從任務到 kernel 參數

## 直覺模型

`finishPlan`是 enqueue 模組的「打包器」。它把任務、batch、proxyOp 打包成 kernel 能直接讀取的參數結構。這就像快遞打包——把散件裝進箱子，貼上運單，等待發貨。

## Step-by-Step：finishPlan 的打包邏輯

[FACT:src/enqueue/enqueue.cc:236-330]

**第 1 步：決定儲存類型。**如果所有工作都能放進 kernel args，用`ncclDevWorkStorageTypeArgs`：

[FACT:src/enqueue/enqueue.cc:244-250]

**第 2 步：分配 kernelArgs。**從記憶體堆疊分配：

[FACT:src/enqueue/enqueue.cc:251-255]

**第 3 步：Round-robin 放置 batch。**每個 channel 的第一個 batch 必須放在`batchZero[blockIdx.x]`：

[FACT:src/enqueue/enqueue.cc:257-280]

**第 4 步：合併 proxyOp 佇列。**按 opCount 合併排序：

[FACT:src/enqueue/enqueue.cc:282-329]

## 資料結構：ncclDevKernelArgs

`ncclDevKernelArgs`是傳給 kernel 的參數結構。它包含：

- `comm`：裝置側通訊器
- `channelMask`：channel 位元遮罩
- `workStorageType`：工作儲存類型
- `workBuf`：工作緩衝區指標
- `workMask`：工作緩衝區遮罩

## 生產踩坑：batch 順序

[FACT:src/enqueue/enqueue.cc:257-259]

註解說得很清楚："The first batch for each channel must be located at batchZero[blockIdx.x]"。如果這個順序錯了，kernel 會讀到錯誤的 batch，導致資料損壞。

---

# 本章小結

本章我們追蹤了從`ncclAllReduce`到`ncclTaskColl`的完整路徑：

1. **ncclAllReduce**建構`ncclInfo`，打包使用者參數

2. **ncclEnqueueCheck**校驗參數、處理 group 語義

3. **taskAppend**根據操作類型分發到不同路徑

4. **collTaskAppend**生成`ncclTaskColl`，解析配置

5. **ncclPrepareTasks**按 (func, op, datatype) 分桶，計算演算法

6. **scheduleCollTasksToPlan**切分 channel，生成`ncclDevWorkColl`

7. **finishPlan**打包成 kernel 參數

關鍵設計思想：

- **分層解耦**：每個函式只做一件事，透過`ncclInfo`和`ncclTaskColl`傳遞狀態
- **預算控制**：透過`ncclTestBudget`控制每個 plan 的大小
- **聚合最佳化**：大小相近的任務會被聚合，減少 kernel 啟動次數
- **配置優先級**：env > per-call > comm

下一章我們將進入`task_sched`，看 NCCL 如何編排多 channel 多 kernel 的執行順序。

# 本章思考與自測

Q1: 如果把`collTaskAppend`中的`aggIsolate`判斷去掉（即`src/enqueue/enqueue.cc:2821-2822`永遠返回 false），在什麼場景下會導致使用者設定的`maxCTAs`失效？為什麼？

**參考解析**：`aggIsolate`的作用是標記「這個任務不能被聚合」。如果去掉這個判斷，設定了 per-call config 的任務會和相鄰任務合併。在`ncclPrepareTasks`的聚合迴圈中（`src/enqueue/enqueue.cc:507-508`），聚合條件是`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes && !aggBeg->aggIsolate && !aggEnd->aggIsolate`。如果`aggIsolate`永遠為 false，那麼即使任務設定了`maxCTAs=4`，它也可能和一個`maxCTAs=32`的任務合併。合併後的`agg`會取兩者的某種組合（具體取決於`ncclGetAlgoInfo`的實作），導致實際使用的 CTA 數量不符合使用者預期。

更嚴重的是，在`scheduleCollTasksToPlan`中（`src/enqueue/enqueue.cc:665-666`），`taskAggIsolate`用於確保配置了 per-call 資源的任務單獨佔一個 plan。如果這個判斷失效，多個任務會共享 plan 的 channel 預算，導致資源分配不符合預期。

Q2: 在`ncclEnqueueCheck`中，如果`ncclGroupEndInternal()`返回錯誤（比如某個 rank 的 ArgsCheck 失敗），但`taskAppend`已經成功執行了，會發生什麼？NCCL 如何保證狀態一致性？

**參考解析**：看`src/enqueue/enqueue.cc:3513-3519`的控制流：

```c
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
info->comm->opCount++;
exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
```

如果`taskAppend`成功但`ncclGroupEndInternal`失敗，`opCount`已經遞增了。這會導致後續操作的 opCount 與對端不匹配，可能觸發 hang。

NCCL 的處理方式是：`ncclGroupErrCheck(ret)`會檢查是否有錯誤，如果有，會設定 comm 的錯誤狀態。後續的 API 呼叫會透過`ncclCommGetAsyncError`檢測到這個錯誤並立即返回。這是一種「快速失敗」策略——一旦出錯，整個 comm 進入錯誤狀態，不再嘗試恢復。

在生產環境中，這意味著一旦出現 group 錯誤，使用者需要銷毀並重建 communicator。

Q3: `scheduleCollTasksToPlan`中的 cell 切分演算法（`src/enqueue/enqueue.cc:740-845`）有一個邊界條件：當`cellsLo == 0`時，會跳過最少的 channel。如果這個跳過邏輯有 bug（比如`channelId`沒有正確遞增），會導致什麼後果？

**參考解析**：看`src/enqueue/enqueue.cc:770-780`：

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

如果`channelId`沒有正確遞增，那麼下一個任務會從錯誤的 channel 開始分配。這會導致：

1. **channel 重疊**：兩個任務可能分配到同一個 channel 的同一段資料

2. **資料損壞**：kernel 會重複處理或遺漏資料

3. **效能下降**：channel 負載不均衡

更隱蔽的是，這種 bug 可能只在特定訊息大小下觸發（當`cellsLo == 0`時），難以復現。NCCL 透過`plan->channelMask |= (2ull << devWork->channelHi) - (1ull << devWork->channelLo)`來追蹤已使用的 channel，但這只是記錄，不能防止重疊。

至此，我們已經看清 ncclAllReduce 如何從使用者呼叫變成一串可執行的 kernel 任務：參數校驗、演算法/協定確定、channel 切分，最終生成 ncclInfo 與 ncclTaskColl。但任務被建立出來只是第一步——它們還需要被排程到多個 channel 上，生成 kernel 啟動參數，並在 group 語義下處理批次提交與依賴排序。下一章將深入 src/enqueue/task_sched 與 src/enqueue/task_prep，回答「為什麼一次 AllReduce 會啟動多個 kernel，它們之間的順序和依賴是怎麼保證的」，同時揭示 src/group.cc 中 ncclGroupStart/ncclGroupEnd 如何把多次 API 呼叫合併成一次提交。
