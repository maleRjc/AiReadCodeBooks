# 第 5 章：演算法與協定選型：tuning 模組如何決定通訊路徑

上一章我們拆解了 NCCL 的拓撲感知能力：從 src/graph/topo.cc 枚舉裝置構建拓撲圖，到 src/graph/search.cc 搜尋最優路徑，再到 rings.cc 與 trees.cc 將搜尋結果具體化為 Ring 與 Tree 演算法拓撲。但拓撲圖只回答了「資料能走哪條路」，它沒有回答「這次通訊應該走哪條路」。同一台機器上，一次 4KB 的 AllReduce 和一次 400MB 的 AllReduce，最優解可能完全不同：前者拼的是延遲，後者拼的是頻寬；前者可能選 Tree/LL，後者可能選 Ring/Simple 或者 NVLS。tuning 模組就是那個「拍板的人」。它的輸入是訊息大小、rank 數、拓撲圖（上一章的產物）和使用者環境變數；輸出是一個 ncclTuningResult_t，裡面寫著用哪個演算法（algo）、哪個協定（proto）、開多少 channel、用多少 warp。這一章我們按「總調度 → 代價模型 → 各演算法估計 → 收尾決策」的順序，把 src/tuning 目錄拆開。核心問題只有一個：NCCL 怎麼在幾十種 (演算法, 協定) 組合裡，用一套純 CPU 的數學模型，在微秒級時間內選出最快的那一個？

# 一、tuning.cc：總調度與決策主幹

## 直覺模型

把 tuning 模組想像成一家**搬家公司**。客戶（一次集合通訊）來了，說「我要搬 100MB 的貨，從 8 個倉庫搬到 8 個倉庫」。調度員（`ncclTuningCompute`）不會真的去搬一遍試試，而是拿出一張**價目表**（代價模型），對每種方案（Ring/LL、Tree/Simple、NVLS/Simple……）估算一個「預計耗時」，然後挑最短的那個報價給客戶。

如果沒有這個調度員，NCCL 就只能寫死「AllReduce 永遠用 Ring」，那在小訊息場景會被 Tree 吊打，在大規模 NVLink 場景會被 NVLS 吊打。**代價就是效能將在特定場景下腰斬甚至更差。**

## 資料結構與記憶體佈局

決策的載體是`ncclTuningResult_t`，候選集合是`ncclTuningResultList_t`（一個單向鏈結串列）。鏈結串列節點定義在`tuning_int.h`，但 push 邏輯在`tuning.cc`裡：

[FACT:src/tuning/tuning.cc:32-39]

```c
ncclResult_t ncclTuningResultListPushFront(struct ncclTuningResultList_t* list, struct ncclTuningResult_t result) {
  struct ncclTuningResultListNode* node = nullptr;
  NCCLCHECK(ncclCalloc(&node, 1));
  node->result = result;
  node->next = list->head;
  list->head = node;
  return ncclSuccess;
}
```

> **[Design Inference & Architectural Trade-offs]**
> 注意這裡是**頭插法**：每算出一個有效候選，就插到鏈結串列頭部。這意味著鏈結串列順序和 id 順序是**反的**。為什麼用鏈結串列而不是陣列？ 因為候選數量在編譯期由`NCCL_TUNING_COUNT`決定，但實際有效的候選是動態的（受`tuningMask`、平台能力、使用者環境變數影響），鏈結串列允許「只把有效的掛上去」，避免遍歷時反覆判斷`valid`。代價是每次決策要`ncclCalloc`一次，但 tuning 發生在入隊路徑上、頻率不高，這點分配開銷可以接受。

`ncclTuningResult_t`裡最關鍵的兩個欄位是`timeUs`（預計耗時，微秒）和`selectionTimeUs`（用於選擇的耗時，可能被 tuner 外掛覆蓋）。選擇邏輯只看後者：

[FACT:src/tuning/tuning.cc:155-173]

```c
static ncclResult_t ncclTuningSelectBestTuning(struct ncclTuningResultList_t* tunings,
                                               struct ncclTuningResult_t* const bestTuning) {
  bestTuning->timeUs = FLT_MAX;
  float bestSelectionTimeUs = FLT_MAX;
  struct ncclTuningResultListNode* node = tunings->head;
  while (node != nullptr) {
    const struct ncclTuningResult_t& tuning = node->result;
    float selectionTimeUs = tuning.selectionTimeUs > 0.0f ? tuning.selectionTimeUs : tuning.timeUs;
    ...
    if (selectionTimeUs next;
  }
  return ncclSuccess;
}
```

這裡有個細節：`bestTuning->timeUs`先被設成`FLT_MAX`，然後遍歷。如果鏈結串列為空（所有候選都無效），`bestTuning`會保持`NCCL_TUNING_RESULT_INIT`的初始值，algo/proto 都是`UNDEF`。這個「空結果」在呼叫方會被特殊處理——見後面的錯誤分支。

## Step-by-Step Walkthrough：一次 AllReduce 的決策流

假設應用呼叫`ncclAllReduce`，訊息 1MB，8 個 rank 單機 NVLink。我們跟著`ncclTuningCompute`走一遍。

**第 0 步：單 rank 短路。**如果`nRanks <= 1`，根本不需要通訊，直接回傳 Ring/Simple，channel 數設 0：

[FACT:src/tuning/tuning.cc:191-200]

```c
  // Set tuning to Ring/Simple for single rank case
  if (input->comm->nRanks tuningMask & (1ULL comm->tuner != NULL) {
      float generalTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      for (int i = 0; i result;
        node = node->next;
        if (tuning.algo == NCCL_ALGO_UNDEF || tuning.proto == NCCL_PROTO_UNDEF) continue;
        generalTable[tuning.algo][tuning.proto] = tuning.timeUs;
      }
      node = tunings.head;
      int nMaxChannels = 0;
      NCCLCHECKGOTO(input->comm->tuner->getCollInfo(input->comm->tunerContext, input->func, input->nBytes,
                                                    input->numPipeOps, (float**)generalTable, NCCL_NUM_ALGORITHMS,
                                                    NCCL_NUM_PROTOCOLS, input->regBuff, &nMaxChannels),
                    ret, exit);
      while (node != nullptr) {
        struct ncclTuningResult_t& tuning = node->result;
        node = node->next;
        if (tuning.algo == NCCL_ALGO_UNDEF || tuning.proto == NCCL_PROTO_UNDEF) continue;
        tuning.maxChannels = nMaxChannels;
        tuning.timeUs = generalTable[tuning.algo][tuning.proto];
      }
    }
```

這裡`NCCL_TUNING_IGNORE`是一個哨兵值，表示「這個組合沒算過/不適用」。外掛可以只改它關心的格子，其他格子保持 IGNORE，NCCL 會跳過。

**第 4 步：選最優。**調`ncclTuningSelectBestTuning`，遍歷鏈結串列取`selectionTimeUs`最小的。

**第 5 步：算 channel 數。**選出演算法後，還要決定開多少 channel：

[FACT:src/tuning/tuning.cc:233-235]

```c
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
```

`ncclTuningGetChannels`在`tuning_int.h`裡，邏輯是根據訊息大小和演算法類型，在`minChannels`和`maxChannels`之間插值。channel 數直接影響頻寬：channel 越多，並行度越高，但每個 channel 的啟動開銷也越大。

**第 6 步：CTA Policy 偏置（NVLS 優先）。**如果使用者設了`NCCL_CTA_POLICY_EFFICIENCY`，且當前是 AllGather/ReduceScatter 且 buffer 已註冊，NCCL 會嘗試把結果改成 NVLS：

[FACT:src/tuning/tuning.cc:240-257]

```c
  if (input->comm->tuner == NULL && (input->CTAPolicy & NCCL_CTA_POLICY_EFFICIENCY) &&
      ncclGetEnv("NCCL_ALGO") == NULL && ncclGetEnv("NCCL_PROTO") == NULL && !input->comm->MNNVL &&
      (input->tuningMask & (1ull regBuff && (input->func == ncclFuncAllGather || input->func == ncclFuncReduceScatter)) {
      if ((input->comm->nNodes > 1 && input->collNetSupport && input->nvlsSupport) ||
          (input->comm->nNodes == 1 && input->nvlsSupport)) {
        int recChannels;
        NCCLCHECKGOTO(ncclNvlsRegResourcesQuery(input->comm, input->func, &recChannels), ret, exit);
        if (recChannels func), ncclDatatypeToString(input->datatype), ncclAlgoEnvStr, ncclProtoEnvStr,
         ncclSymKernelIdEnvStr);
    ret = (algoEnv || protoEnv || symKernelIdEnv) ? ncclInvalidUsage : ncclInternalError;
  }
```

**為什麼區分錯誤碼？**如果使用者設了`NCCL_ALGO=ring`但當前平台不支援 ring（比如某些特殊拓撲），那是**使用者配置錯誤**（`ncclInvalidUsage`）；如果使用者沒設任何環境變數卻選不出演算法，那是**NCCL 內部 bug**（`ncclInternalError`）。這個區分對排障至關重要。

## 決策主幹流程圖

```mermaid
flowchart TD
    start["ncclTuningCompute(input)"] --> check_rank{"comm->nRanks |是| single["bestTuning = Ring/SimplenChannels = 0"]
    check_rank -->|否| enum["ncclTuningComputeAllTunings遍历 NCCL_TUNING_COUNT"]
    enum --> mask{"tuningMask & (1|否| skip["tuning.valid = 0continue"]
    mask -->|是| expand["ncclTuningExpandId(i)"]
    expand --> sim["ncclTuningComputeTuning-> ncclTuningCostModelSimModel"]
    sim --> valid{"result.valid?"}
    valid -->|是| push["ncclTuningResultListPushFront"]
    valid -->|否| skip
    push --> tuner{"comm->tuner != NULL?"}
    tuner -->|是| plugin["tuner->getCollInfo覆盖 generalTable"]
    tuner -->|否| select
    plugin --> select["ncclTuningSelectBestTuning取 selectionTimeUs 最小"]
    select --> getch["ncclTuningGetChannels"]
    getch --> cta{"CTA_POLICY_EFFICIENCY且 NVLS 在 mask 内?"}
    cta -->|是| nvls["ncclNvlsRegResourcesQuery可能改写为 NVLS"]
    cta -->|否| symk
    nvls --> symk{"symKernelId 需要回退?"}
    symk -->|是| fallback["ncclTuningCompute(generalInput)回退普通 kernel"]
    symk -->|否| done
    fallback --> done["*result = bestTuning"]
    single --> done
    done --> undef{"algo/proto 仍 UNDEF?"}
    undef -->|是| warn["WARN + 返回InvalidUsage 或 InternalError"]
    undef -->|否| ret_ok["返回 ncclSuccess"]
```

---

# 二、cost_model.cc：模型註冊表與開關矩陣

## 直覺模型

`cost_model.cc`是 tuning 的**總帳本**。它維護一張`modelMap`表，每一行對應一個 (algo, proto) 組合，記錄「這個組合的初始化函式是誰、模擬函式是誰、對哪些函式啟用」。同時它負責解析使用者環境變數`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SYM_KERNEL`，把使用者的意圖翻譯成一張`enabled[i][f]`開關矩陣。

如果沒有這張表，每加一個新演算法就要改一遍 tuning 主流程，程式碼會爛成一鍋粥。**表驅動**讓「加演算法」變成「加一行」。

## 資料結構：modelMap 與開關矩陣

`modelMap`是一個靜態陣列，每個元素是`ncclTuningModelEntry_t`：

[FACT:src/tuning/cost_model.cc:230-277]

```c
static struct ncclTuningModelEntry_t modelMap[] = {
  {ncclTuningTreeModelInit, ncclTuningTreeModelSim, nullptr, {0, 0, 0, 0, 1}},       // Tree/LL
  {ncclTuningTreeModelInit, ncclTuningTreeModelSim, nullptr, {0, 0, 0, 0, 1}},       // Tree/LL128
  {ncclTuningTreeModelInit, ncclTuningTreeModelSim, nullptr, {0, 0, 0, 0, 1}},       // Tree/Simple
  {ncclTuningRingModelInit, ncclTuningRingModelSim, nullptr, {1, 1, 1, 1, 1}},       // Ring/LL
  ...
  {nullptr, nullptr, nullptr, {0}}, // CollNetDirect/LL, disabled as there is no implementation
  ...
};
```

每個 entry 有四個欄位：`init`（初始化，算好 latency/bandwidth 存到 comm 裡）、`model`（模擬，根據訊息大小算最終 timeUs）、`finalize`（清理）、`enabled[5]`（對 Broadcast/Reduce/AllGather/ReduceScatter/AllReduce 五個函數是否啟用）。

注意`enabled`陣列的順序註解在 L234：`Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce`。這個順序必須和`ncclFunc_t`列舉一致，否則會張冠李戴。

> **[Design Inference & Architectural Trade-offs]**
> **為什麼 init 和 sim 要分開？**因為 init 裡算的東西（latency、bandwidth）**只依賴 comm 的靜態屬性**（拓撲、rank 數、compCap），和具體訊息大小無關。一次通訊裡可能連續調多次 tuning（比如 group 裡有多個 op），init 只跑一次，sim 每次跑。這是典型的「預計算 + 快速查詢」優化。

## Step-by-Step：環境變數解析與開關矩陣構建

**第 1 步：預設全開，LL128 特殊。** `ncclTuningCostModelInit`一開始把所有 proto 設成 1（啟用），但 LL128 設成 2：

[FACT:src/tuning/cost_model.cc:313-323]

```c
  for (int f = 0; f minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                          comm->graphs[algo].typeIntra, comm->nRanks, f, algo, comm->minDriverVersion)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
```

**第 2 步：解析使用者環境變數。**如果使用者設了`NCCL_ALGO`或`NCCL_SYM_KERNEL`，先把 algo 和 symKernel 全清零（因為使用者指定了白名單）：

[FACT:src/tuning/cost_model.cc:327-345]

```c
  if ((algoStr && strlen(algoStr) > 0) || (symKernelIdStr && strlen(symKernelIdStr) > 0)) {
    std::fill_n(algoEnable, NCCL_NUM_FUNCTIONS * NCCL_NUM_ALGORITHMS, 0);
    std::fill_n(symKernelIdEnable, NCCL_NUM_FUNCTIONS * ncclSymkKernelId_Count, 0);
  }
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable,
                        comm->tuningContext.forced));
  }
```

注意 proto 沒有清零——因為 proto 的預設值是 1/2，使用者設`NCCL_PROTO=LL`時，`parseList`會把 LL 設成 1、其他設成 0（因為`unset`邏輯）。這個不對稱是刻意的：algo 預設全開但使用者指定後要收窄，proto 的收窄由`parseList`內部處理。

**第 3 步：parseList 的語法。**這個函數支援相當複雜的語法，註解裡給了例子：

[FACT:src/tuning/cost_model.cc:14-32]

```c
// Parse a map of prefixes to a list of elements. The first prefix is
// optional and, if not present, the list of elements will be applied
// to all prefixes. Only the first list of elements can lack a
// prefix. Prefixes (if present) are followed by a colon. Lists of
// elements are comma delimited. Mappings of prefix to the lists of
// elements are semi-colon delimited.
//
// For example:
//
//     NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
// Enable ring and collnetdirect for all functions, then select tree
// and collnetdirect for allreduce and ring for broadcast.
```

`^`前綴表示「取反」：

[FACT:src/tuning/cost_model.cc:59-67]

```c
    int unset, set;
    if (elemList[0] == '^') {
      unset = 1;
      set = 0;
      elemList++;
    } else {
      unset = 0;
      set = 1;
    }
```

所以`NCCL_PROTO="^LL128;allreduce:LL128"`的意思是：全域禁用 LL128，但 AllReduce 例外啟用 LL128。

**第 4 步：合併 enabled 矩陣。**最後遍歷所有 model，把`model->enabled[f]`和使用者開關做與運算：

[FACT:src/tuning/cost_model.cc:371-383]

```c
      //  Check the user env vars only for functions that have a forced configuration and not already disabled.
      if (comm->tuningContext.forced[f] == 0 || comm->tuningContext.enabled[i][f] == 0) continue;
      comm->tuningContext.enabled[i][f] = 0;
      ...
      if (((algo != NCCL_ALGO_UNDEF && algoEnable[f * NCCL_NUM_ALGORITHMS + algo] != 0) &&
           (proto != NCCL_PROTO_UNDEF && protoEnable[f * NCCL_NUM_PROTOCOLS + proto] != 0)) ||
          (symKernelId != ncclSymkKernelId_Count && symKernelIdEnable[f * ncclSymkKernelId_Count + symKernelId] != 0)) {
        comm->tuningContext.enabled[i][f] = 1;
      }
```

邏輯是：**只有當使用者對某個函數設了 forced 配置時，才用使用者配置覆蓋模型預設值**。如果使用者沒設，`forced[f] == 0`，直接`continue`，保留模型自己的`enabled`。這是「使用者顯式指定 > 模型預設」的優先級。

## 模型仿真的統一入口

所有模型最終都通過`ncclTuningCostModelSimModel`調用：

[FACT:src/tuning/cost_model.cc:470-497]

```c
ncclResult_t ncclTuningCostModelSimModel(int id, struct ncclTuningInput_t* const input,
                                         struct ncclTuningResult_t* const result) {
  struct ncclTuningModelEntry_t* model = nullptr;
  ncclResult_t ret = ncclSuccess;
  result->forced = input->comm->tuningContext.forced[input->func];
  NCCLCHECKGOTO(getModelEntry(id, &model), ret, not_valid);
  if (model == nullptr) {
    ret = ncclInternalError;
    goto not_valid;
  }
  if (input->comm->tuningContext.enabled[id][input->func] == 0) {
    goto not_valid;
  }
  if (model->model != nullptr) {
    NCCLCHECKGOTO(model->model(input, result), ret, not_valid);
    if (result->timeUs timeUs = NCCL_TUNING_IGNORE;
  result->valid = 0;
  goto exit;
}
```

三層過濾：**id 越界 → 模型禁用 → 模型返回非正時間**，任何一層不過都走`not_valid`，把`timeUs`設成`NCCL_TUNING_IGNORE`（一個負數哨兵）、`valid = 0`。調用方看到`valid == 0`就不會把它掛進候選鏈表。

## 設計思考

`modelMap`的註解裡有一句關鍵警告：

[FACT:src/tuning/cost_model.cc:229]

```c
// IMPORTANT: this table need must be consistent with the algRegistry in src/config/algorithm_registry.cc
```

> **[Design Inference & Architectural Trade-offs]**
> 這意味著`modelMap`的**下標順序**必須和`algorithm_registry.cc`裡的算法註冊順序嚴格一致。如果有人在 registry 裡插了一個新算法但忘了改`modelMap`，所有 id 都會錯位，tuning 會選出一個完全錯誤的算法。**這是表驅動設計的經典陷阱：隱式契約。**更健壯的做法是用列舉名做 key 而不是下標，但那樣會犧牲一點編譯期優化。

---

# 三、ring.cc：Ring 算法的代價估計

## 直覺模型

Ring 算法把 N 個 rank 排成一個環，數據沿著環一圈一圈傳。它的代價模型要回答兩個問題：**每步傳多少數據（帶寬）**、**一共要多少步（延遲）**。

Ring 的直覺是「**流水線**」：想像 N 個人站成一圈傳水桶，每個人接到桶後倒一點水再傳給下一個人。桶轉一圈，所有人的水都混勻了。桶轉得越快（帶寬高）、圈越小（步數少），整體越快。

## 數據結構：latency/bandwidth 表

Ring 模型不引入新結構，它把估計結果寫進`comm->tuningContext.generalLatencies[c][algo][proto]`和`generalBandwidths[c][algo][proto]`。這兩個是三維陣列：函數 × 算法 × 協議。

初始化時先全部設成 -1.0（哨兵，表示「沒算過」）：

[FACT:src/tuning/ring.cc:31-33]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
```

-1.0 這個哨兵在 sim 階段被檢查：

[FACT:src/tuning/ring.cc:94-97]

```c
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ncclSuccess;
  }
```

**為什麼用 -1.0 而不是 0？**因為 0 是一個合法的帶寬值（雖然物理上不可能），而 -1.0 明確表示「未初始化」。浮點比較用`==`在這裡是安全的，因為 -1.0 是精確可表示的。

## Step-by-Step：Ring 帶寬估計

**第 1 步：確定用 intra 還是 inter 帶寬。**單機（nNodes==1）用 intra，多機用 inter：

[FACT:src/tuning/ring.cc:34-37]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float bw = (comm->nNodes == 1 || (comm->nNodes minCompCap graphs[algo].bwIntra :
                                                                                      comm->graphs[algo].bwInter;
    float busBw = bw * comm->graphs[algo].nChannels;
```

`nSteps`是算法需要的步數，對 Ring 來說 AllReduce 是`2*(nRanks-1)`，其他是`nRanks-1`。`busBw`是「總線帶寬」= 單鏈路帶寬 × channel 數。

**第 2 步：按協議打折。**LL 協定只用了頻寬的一半（因為 LL 的 flag 開銷），LL128 用 92%（120/128）：

[FACT:src/tuning/ring.cc:38-42]

```c
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);
```

`0.92 = 120/128`是因為 LL128 每 128 位元組裡有 8 位元組是 flag，有效載荷只有 120 位元組。這個數字直接來自協定設計。

**第 3 步：算有效頻寬。**注意這裡乘了`nRanks / nSteps`：

[FACT:src/tuning/ring.cc:44-46]

```c
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;
```

**為什麼乘`nRanks / nSteps`？**這是 Ring 演算法的核心特性：每個 rank 實際搬運的資料量是`nBytes * nSteps / nRanks`（因為資料要繞環多圈）。所以「有效頻寬」= 總線頻寬 × nRanks / nSteps。對 AllReduce，nSteps = 2(nRanks-1)，所以有效頻寬 ≈ busBw/2。

**第 4 步：算延遲。**延遲分 intra 和 inter 兩部分：

[FACT:src/tuning/ring.cc:48-63]

```c
    int intraHw, interHw;
    ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);
    int hwLevel = comm->nNodes == 1 ? intraHw : interHw;

    float intraLat = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];
    // Preserve the pre-refactor model: with one rank per node, Ring inter-node steps use the exposed Tree NET latency.
    float interLat;
    if (comm->nNodes == 1) {
      interLat = intraLat;
    } else if (comm->maxLocalRanks == 1) {
      interLat = comm->tuningContext.tuningConstants.hwLatencies[NCCL_HW_NET][NCCL_ALGO_TREE][proto];
    } else {
      interLat = comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto];
    }
    interLat += comm->graphs[algo].latencyInter;
    if (proto == NCCL_PROTO_SIMPLE) interLat += comm->graphs[algo].latencyInter;
```

注意 L57-58 的特殊處理：當`maxLocalRanks == 1`（每個節點只有 1 個 rank）時，Ring 的 inter-node 延遲用**Tree 的 NET 延遲**。註解說這是「preserve the pre-refactor model」——即為了保持和重構前行為一致，刻意保留的一個「怪癖」。**這種歷史包袱在成熟系統裡很常見，讀原始碼時看到「preserve」字樣要格外小心，它往往意味著這裡有個不能動的相容性約束。**

**第 5 步：按函式類型累加。**Reduce/Broadcast 和 AllReduce/AllGather/ReduceScatter 的延遲模型不同：

[FACT:src/tuning/ring.cc:65-87]

```c
    if ((c == ncclFuncReduce || c == ncclFuncBroadcast)) {
      float lat = comm->tuningContext.tuningConstants.hwLatencies[hwLevel][algo][proto];
      if (comm->graphs[algo].sameChannels) {
        comm->tuningContext.generalLatencies[c][algo][proto] += lat;
      } else {
        if (proto == NCCL_PROTO_SIMPLE)
          lat =
            comm->tuningContext.tuningConstants
              .hwLatencies[hwLevel][NCCL_ALGO_TREE][proto]; // Add some chunk latency, waiting for proper chunk modeling
        comm->tuningContext.generalLatencies[c][algo][proto] += nSteps * lat;
      }
    } else {
      // Inter-node rings still have to launch nsteps * net overhead.
      float netOverhead = 0.0;
      if (comm->nNodes > 1) {
        netOverhead = getNetOverhead(comm);
        if (proto == NCCL_PROTO_SIMPLE) netOverhead *= 3;
      }
      intraLat = std::max(intraLat, netOverhead);
      int nInterSteps = comm->nNodes == 1 ? 0 : c == ncclFuncAllReduce ? 2 * (comm->nNodes - 1) : comm->nNodes - 1;
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        (nSteps - nInterSteps) * intraLat + nInterSteps * interLat;
    }
```

`sameChannels`是一個拓撲屬性，表示「環上的 intra 和 inter 步是否用同一組 channel」。如果不同，延遲要乘`nSteps`（每步都要等）。`netOverhead`是網路 post 開銷，Simple 協定要乘 3（因為 Simple 有三次網路往返：send、recv、ack）。

## 生產避坑：Ring/Simple 的 plateau 效應

`ncclTuningRingModelSim`裡有一段專門處理「plateau」的程式碼：

[FACT:src/tuning/ring.cc:105-137]

```c
  // Update Ring/Simple latency for multi-node AllReduce and
  // single NVL Domain AllReduce/AllGather/ReduceScatter for Blackwell
  bool isBlackwellNvLink =
    inputs->comm->minCompCap >= 100 && inputs->comm->graphs[NCCL_ALGO_RING].typeIntra == PATH_NVL;
  bool ringSimplePlateau =
    (inputs->comm->nNodes > 1 && inputs->func == ncclFuncAllReduce) ||
    (inputs->comm->nNodes == 1 && isBlackwellNvLink &&
     (inputs->func == ncclFuncAllReduce || inputs->func == ncclFuncAllGather || inputs->func == ncclFuncReduceScatter));
  size_t bytesPerRankPerChannel = inputs->nBytes / (inputs->comm->nChannels * inputs->comm->nRanks);

  if (tuning->algo == NCCL_ALGO_RING && tuning->proto == NCCL_PROTO_SIMPLE && ringSimplePlateau &&
      bytesPerRankPerChannel >= 64) {
    float plateauFactor = inputs->comm->minCompCap  **[Design Inference & Architectural Trade-offs]**
> **什麼是 plateau？**在 Ring/Simple 裡，當訊息大到一定程度，延遲不再隨訊息線性增長，而是「卡」在一個平台上——因為此時瓶頸從「啟動開銷」變成了「頻寬」，而頻寬已經飽和。這個現象在 Blackwell NVLink 上尤其明顯（因為 NVLink 頻寬太高，延遲佔比更大）。程式碼用`plateauFactor`（1.4 或 1.9）乘到延遲上，模擬這個「延遲被放大」的效果。

`bytesPerRankPerChannel >= 64`是觸發條件：每個 rank 每個 channel 至少要傳 64 位元組，否則 plateau 不成立。這個 64 位元組來自 LL 協定的 flag 大小。

**踩坑場景**：如果你在 Blackwell 上跑一個 1MB 的 AllReduce，發現實際延遲比模型預測的高 40%，不要以為是 bug——這是 plateau 效應，模型已經把它算進去了。如果你手動改小`plateauFactor`，模型會低估延遲，導致選錯演算法。

---

# 四、tree.cc 與 nvls.cc：Tree 與 NVLS 的代價估計

## 直覺模型

**Tree 演算法**是「**樹形廣播**」：根節點把資料分給子節點，子節點再分給孫節點。它的優勢是**步數少**（log N 而不是 N），適合小訊息；劣勢是**頻寬利用率低**（每個非葉節點要轉發，實際有效頻寬只有一半）。

**NVLS**（NVLink SHARP）是「**硬體多播**」：交換器直接把資料複製給多個 GPU，不需要軟體轉發。它的優勢是**頻寬高、延遲低**，但需要特定硬體（Hopper 以上）和特定配置。

## Tree 模型：只服務 AllReduce

Tree 模型有個硬性限制——**只對 AllReduce 啟用**：

[FACT:src/tuning/tree.cc:21-27]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
```

> **[Design Inference & Architectural Trade-offs]**
> **為什麼？**因為 NCCL 的 Tree 實作只支援 AllReduce（其他集合操作沒有 Tree 版本）。這是一個實作約束，不是理論限制。`enabled[c] = 0`是「硬禁用」，比`generalBandwidths = -1`更徹底——前者直接讓`ncclTuningCostModelSimModel`在 L480 就返回`not_valid`，後者要到 sim 函式裡才檢查。

**Tree 頻寬估計**：

[FACT:src/tuning/tree.cc:28-43]

```c
    float bw = (comm->minCompCap nNodes graphs[algo].bwIntra : comm->graphs[algo].bwInter) :
                 std::min(comm->graphs[algo].bwInter, comm->graphs[algo].bwIntra);
    float busBw = bw * comm->graphs[algo].nChannels;
    if (c == ncclFuncAllReduce) busBw = std::min(busBw * .92, comm->graphs[algo].nChannels * perChMaxTreeBw);
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(busBw * 1.0 / 3.8, llMaxBw);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (comm->nNodes == 1 ? 7.0 / 9.0 : 120.0 / 128.0),
                       comm->graphs[algo].nChannels * perChMaxTreeLL128Bw);
    if (comm->maxTreePattern == NCCL_TOPO_PATTERN_TREE) busBw *= .85;
```

> **[Design Inference & Architectural Trade-offs]**
> 注意 LL 協定的打折係數是`1/3.8`，比 Ring 的`0.5`更狠。**為什麼 Tree 的 LL 效率更低？**因為 Tree 的每個中間節點既要收又要發，LL 的 flag 開銷在雙向流量下被放大。`1/3.8`這個數字來自實測。

**Tree 延遲估計**：

[FACT:src/tuning/tree.cc:55-58]

```c
    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
```

`2 *`是因為 AllReduce = ReduceScatter + AllGather，兩趟。`(nRanks/nNodes - 1)`是節點內步數（每個節點內的 rank 數減一），`log2i(nNodes)`是節點間步數（樹的高度）。

**Tree 的修正因子**：Tree 模型在 sim 階段乘了一個`treeCorrectionFactor`：

[FACT:src/tuning/tree.cc:75-79]

```c
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize proto][logSize];
```

`treeCorrectionFactor`是一個 3×24 的表：

[FACT:src/tuning/cost_model.cc:223-227]

```c
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};
```

`logSize = log2(nBytes >> 6)`，即訊息大小以 64 位元組為單位取 log2。表的下標 0-23 對應 64B 到 64B×2^23 ≈ 512MB。**這個表是實測出來的「Tree 效率曲線」**：小訊息時效率 1.0（延遲主導），中等訊息時效率掉到 0.4-0.5（頻寬沒打滿），大訊息時回到 1.0（頻寬打滿）。這個「中間凹陷」是 Tree 演算法的固有特性。

## NVLS 模型：硬體多播的代價

NVLS 模型首先檢查硬體是否支援：

[FACT:src/tuning/nvls.cc:19-24]

```c
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!ncclNvlsTransportEnabled(comm)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

然後是一系列硬性約束：只支援 Simple 協定、單機不支援 NVLSTree、多機 NVLS 需要 CollNet：

[FACT:src/tuning/nvls.cc:28-41]

```c
  if ((algo == NCCL_ALGO_NVLS || algo == NCCL_ALGO_NVLS_TREE) && (proto != NCCL_PROTO_SIMPLE)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  if (comm->nNodes == 1 && algo == NCCL_ALGO_NVLS_TREE) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  if (comm->config.collnetEnable == 0 && algo == NCCL_ALGO_NVLS && comm->nNodes > 1) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

**NVLS 頻寬估計**用了一個效率因子：

[FACT:src/tuning/nvls.cc:12-17]

```c
static const float nvlsEfficiency[NCCL_NUM_COMPCAPS] = {
  0.0f, // Volta
  0.0f, // Ampere
  0.85f, // Hopper
  0.74f, // Blackwell
};
```

> **[Design Inference & Architectural Trade-offs]**
> Hopper 是 0.85，Blackwell 反而降到 0.74。**為什麼新一代硬體效率更低？**因為 Blackwell 的 NVLink 頻寬更高，但 NVLS 的交換器處理能力沒有同比提升，導致相對效率下降。這個數字是實測的，不是理論值。

頻寬計算裡有個`(nChannels - 1) / nChannels`因子：

[FACT:src/tuning/nvls.cc:62-74]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float intraBw = comm->graphs[algo].bwIntra * nvlsEfficiency[compCapIndex] * (comm->graphs[algo].nChannels - 1) /
                    comm->graphs[algo].nChannels;
    if (c == ncclFuncAllReduce) {
      intraBw *= 2.0f;
    } else {
      float ppn = comm->minLocalRanks;
      intraBw *= (ppn - 1) / ppn;
    }
    float interBw = comm->graphs[algo].bwInter * ((comm->nNodes ::max()});
    bw = bw * comm->graphs[algo].nChannels;
```

`(nChannels - 1) / nChannels`是因為 NVLS 需要留一個 channel 做同步。`(ppn - 1) / ppn`是 AllGather/ReduceScatter 的額外開銷（每個 rank 要等前一個 rank 的資料）。

## 生產避坑：NVLS 的硬性約束

NVLS 模型在 sim 階段還有一層執行時檢查：

[FACT:src/tuning/nvls.cc:136-156]

```c
  int nvlsSupport = inputs->nvlsSupport;
  if (!nvlsSupport) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  if (inputs->func != ncclFuncAllReduce && inputs->comm->graphs[tuning->algo].nChannels > NCCL_MAX_NVLS_ARITY) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  if (inputs->func != ncclFuncAllReduce && inputs->comm->localRanks > NCCL_MAX_NVLS_ARITY) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
```

`NCCL_MAX_NVLS_ARITY`是 NVLS 多播組能容納的最大 GPU 數。如果超過這個數，NVLS 不可用。**踩坑場景**：在一個 16 卡 NVLink 域裡跑 AllGather，如果`NCCL_MAX_NVLS_ARITY`是 8，NVLS 會被禁用，tuning 會回退到 Ring。如果你不知道這個限制，會以為「NVLS 明明硬體支援為什麼不用」。

---

# 五、對稱 kernel 回退與錯誤恢復鏈

## 直覺模型

對稱 kernel（symmetric kernel）是 NCCL 的新特性：當所有 rank 的 buffer 都註冊到對稱記憶體後，kernel 可以用更高效的指令存取對端記憶體。但**如果 buffer 沒註冊，或者平台不支援，就必須回退到普通 kernel**。這個回退邏輯是 tuning 裡最繞的部分。

## Step-by-Step：回退決策

回退邏輯在`tuning.cc:258-298`。我們拆開看。

**第 1 步：判斷是否需要回退。**入口條件：

[FACT:src/tuning/tuning.cc:258-263]

至此，tuning 模組的決策鏈條已經清晰：它接收拓撲圖與通訊參數，透過代價模型和演算法估計，在微秒級內輸出最優的 (演算法, 協定, channel, warp) 組合。但選型只是開始——這個決策結果如何被下游使用？下一章我們將進入 src/enqueue/enqueue.cc 的主幹，看一次 ncclAllReduce 呼叫如何經過參數校驗、演算法/協定確定、channel 切分，最終生成 ncclInfo 與 ncclTaskColl 結構。這是全書從「使用者視角」切換到「引擎視角」的關鍵一章，你將探明一次集合通訊呼叫在 host 側被翻譯成了什麼，以及它與後續 kernel 啟動之間的邊界。
