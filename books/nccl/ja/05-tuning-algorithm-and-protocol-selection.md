# 第 5 章：アルゴリズムとプロトコルの選定：tuning モジュールが通信パスをどのように決定するか

# 第5章：アルゴリズムとプロトコルの選定：tuning モジュールが通信パスをどのように決定するか

前の章では、NCCL のトポロジ認識能力を分解した：src/graph/topo.cc でデバイスを列挙してトポロジグラフを構築し、src/graph/search.cc で最適パスを検索し、rings.cc と trees.cc で検索結果を Ring と Tree アルゴリズムのトポロジとして具体化する。しかし、トポロジグラフは「データがどのパスを通れるか」に答えるだけで、「今回の通信がどのパスを通るべきか」には答えていない。同じマシン上でも、4KB の AllReduce と 400MB の AllReduce では最適解が全く異なる可能性がある：前者はレイテンシを競い、後者は帯域幅を競う；前者は Tree/LL を選び、後者は Ring/Simple や NVLS を選ぶかもしれない。tuning モジュールこそがその「決定を下す者」である。その入力はメッセージサイズ、ランク数、トポロジグラフ（前章の産物）、ユーザー環境変数；出力は ncclTuningResult_t で、どのアルゴリズム（algo）、どのプロトコル（proto）、いくつのチャネルを開くか、いくつの warp を使うかが書かれている。この章では「全体スケジューリング → コストモデル → 各アルゴリズムの推定 → 最終決定」の順に、src/tuning ディレクトリを分解する。核心的な問題はただ一つ：NCCL はどのように数十種類の (アルゴリズム, プロトコル) の組み合わせの中から、純粋に CPU の数学モデルを用いて、マイクロ秒単位の時間で最速のものを選び出すのか？

# 一、tuning.cc：全体スケジューリングと決定の幹

## 直感的モデル

tuning モジュールを一家の**引越し業者**と想像してほしい。顧客（1回の集合通信）が来て、「100MB の荷物を8つの倉庫から8つの倉庫へ運びたい」と言う。ディスパッチャー（`ncclTuningCompute`）は実際に一度運んで試すのではなく、**価格表**（コストモデル）を取り出し、各方案（Ring/LL、Tree/Simple、NVLS/Simple……）に対して「予想所要時間」を見積もり、最も短いものを見積もりを顧客に提示する。

もしこのディスパッチャーがなければ、NCCL は「AllReduce は常に Ring を使う」とハードコードするしかなく、小メッセージのシナリオでは Tree に圧倒され、大規模 NVLink のシナリオでは NVLS に圧倒されるだろう。**その代償は、特定のシナリオで性能が半減、あるいはそれ以上に悪化することである。**

## データ構造とメモリレイアウト

決定の担体は`ncclTuningResult_t`であり、候補集合は`ncclTuningResultList_t`（単方向連結リスト）である。リストノードは`tuning_int.h`で定義されているが、push ロジックは`tuning.cc`にある：

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
> ここで注意すべきは**先頭挿入法**である：有効な候補が計算されるたびに、リストの先頭に挿入される。これは、リストの順序と id の順序が**逆**であることを意味する。なぜ配列ではなく連結リストを使うのか？ 候補数はコンパイル時に`NCCL_TUNING_COUNT`で決定されるが、実際に有効な候補は動的であり（`tuningMask`、プラットフォーム能力、ユーザー環境変数に影響される）、連結リストは「有効なものだけを繋ぐ」ことを可能にし、走査時に繰り返し`valid`を判断するのを避ける。代償は、各決定で`ncclCalloc`一度だけだが、tuning はエンキュー経路で発生し、頻度は高くないため、この程度の割り当てオーバーヘッドは許容できる。

`ncclTuningResult_t`で最も重要な2つのフィールドは`timeUs`（推定所要時間、マイクロ秒）と`selectionTimeUs`（選択に使用される所要時間、tuner プラグインによって上書きされる可能性がある）。選択ロジックは後者のみを見る：

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

ここに細かい点がある：`bestTuning->timeUs`は最初に`FLT_MAX`に設定され、その後走査する。リンクリストが空の場合（すべての候補が無効）、`bestTuning`は`NCCL_TUNING_RESULT_INIT`の初期値を保持し、algo/proto はともに`UNDEF`となる。この「空結果」は呼び出し側で特別に処理される——後述のエラー分岐を参照。

## Step-by-Step Walkthrough：1回の AllReduce の意思決定フロー

アプリケーションが`ncclAllReduce`を呼び出し、メッセージ 1MB、8 ランクの単一マシン NVLink と仮定する。我々は`ncclTuningCompute`を追って進む。

**第0步：単一ランクのショートカット。**もし`nRanks <= 1`なら、通信は全く不要で、直接 Ring/Simple を返し、channel 数を 0 に設定する：

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

ここで`NCCL_TUNING_IGNORE`はセンチネル値であり、「この組み合わせは計算されていない/適用外」を意味する。プラグインは関心のあるセルのみを変更でき、他のセルは IGNORE のままにしておくと、NCCL はスキップする。

**第4步：最適なものを選択。**は`ncclTuningSelectBestTuning`を呼び出し、リンクリストを走査して`selectionTimeUs`が最小のものを取る。

**第5步：channel 数を計算。**アルゴリズムを選択した後、いくつの channel を開くかを決定する必要がある：

[FACT:src/tuning/tuning.cc:233-235]

```c
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
```

`ncclTuningGetChannels`は`tuning_int.h`内で、メッセージサイズとアルゴリズムタイプに基づいて、`minChannels`と`maxChannels`の間で補間する。channel 数は帯域幅に直接影響する：channel が多いほど並列度は高いが、各 channel の起動オーバーヘッドも大きくなる。

**第6步：CTA Policy バイアス（NVLS 優先）。**ユーザーが`NCCL_CTA_POLICY_EFFICIENCY`を設定し、かつ現在が AllGather/ReduceScatter で buffer が登録済みの場合、NCCL は結果を NVLS に変更しようとする：

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

**なぜエラーコードを区別するのか？**ユーザーが`NCCL_ALGO=ring`を設定したが現在のプラットフォームが ring をサポートしていない場合（例えば一部の特殊なトポロジ）、それは**ユーザー設定エラー**（`ncclInvalidUsage`）である；ユーザーが環境変数を何も設定していないのにアルゴリズムを選択できない場合、それは**NCCL 内部バグ**（`ncclInternalError`）である。この区別はトラブルシューティングにとって極めて重要である。

## 意思決定の主干フローチャート

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

# 二、cost_model.cc：モデルレジストリとスイッチマトリックス

## 直感モデル

`cost_model.cc`は tuning の**総勘定元帳**である。それは`modelMap`テーブルを維持し、各行は1つの (algo, proto) の組み合わせに対応し、「この組み合わせの初期化関数は誰か、シミュレーション関数は誰か、どの関数に対して有効か」を記録する。同時に、ユーザー環境変数`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SYM_KERNEL`を解析し、ユーザーの意図を`enabled[i][f]`スイッチマトリックスに変換する。

このテーブルがなければ、新しいアルゴリズムを追加するたびに tuning のメインフローを修正する必要があり、コードはぐちゃぐちゃになる。**テーブル駆動**により、「アルゴリズムの追加」が「1行の追加」になる。

## データ構造：modelMap とスイッチマトリックス

`modelMap`は静的配列であり、各要素は`ncclTuningModelEntry_t`：

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

各 entry には4つのフィールドがある：`init`（初期化、latency/bandwidth を計算して comm に保存）、`model`（シミュレーション、メッセージサイズに基づいて最終的な timeUs を計算）、`finalize`（クリーンアップ）、`enabled[5]`（Broadcast/Reduce/AllGather/ReduceScatter/AllReduce の5つの関数が有効かどうかに対して）。

注意`enabled`配列の順序コメントは L234 にあります：`Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce`。この順序は`ncclFunc_t`列挙と一致していなければなりません。そうでなければ取り違えが発生します。

> **[Design Inference & Architectural Trade-offs]**
> **なぜ init と sim を分けるのか？**なぜなら init で計算するもの（latency、bandwidth）は**comm の静的な属性にのみ依存し**（トポロジ、rank 数、compCap）、具体的なメッセージサイズとは無関係だからです。1回の通信で連続して複数回 tuning が呼ばれる可能性があります（例えば group 内に複数の op がある場合）が、init は1回だけ実行され、sim は毎回実行されます。これは典型的な「事前計算 + 高速クエリ」の最適化です。

## Step-by-Step：環境変数の解析とスイッチマトリクスの構築

**ステップ1：デフォルトは全て有効、LL128 は特殊。** `ncclTuningCostModelInit`最初に全ての proto を 1（有効）に設定しますが、LL128 は 2 に設定します：

[FACT:src/tuning/cost_model.cc:313-323]

```c
  for (int f = 0; f minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                          comm->graphs[algo].typeIntra, comm->nRanks, f, algo, comm->minDriverVersion)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
```

**ステップ2：ユーザー環境変数の解析。**ユーザーが`NCCL_ALGO`または`NCCL_SYM_KERNEL`を設定した場合、まず algo と symKernel を全てクリアします（ユーザーがホワイトリストを指定したため）：

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

proto はクリアされていないことに注意——proto のデフォルト値は 1/2 であり、ユーザーが`NCCL_PROTO=LL`を設定した場合、`parseList`は LL を 1 に、その他を 0 に設定します（`unset`ロジックのため）。この非対称性は意図的なものです：algo はデフォルトで全て有効ですがユーザー指定後は絞り込む必要があり、proto の絞り込みは`parseList`が内部で処理します。

**ステップ3：parseList の構文。**この関数はかなり複雑な構文をサポートしており、コメントに例が示されています：

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

`^`プレフィックスは「否定」を意味します：

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

したがって`NCCL_PROTO="^LL128;allreduce:LL128"`の意味は：グローバルに LL128 を無効化するが、AllReduce では例外的に LL128 を有効化する、ということです。

**ステップ4：enabled マトリクスのマージ。**最後に全ての model を走査し、`model->enabled[f]`とユーザースイッチの論理積を取ります：

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

ロジックは：**ユーザーが特定の関数に対して forced 設定を行った場合にのみ、ユーザー設定でモデルのデフォルト値を上書きする**。ユーザーが設定していない場合、`forced[f] == 0`、直接`continue`、モデル自身の`enabled`を保持します。これは「ユーザーの明示的指定 > モデルのデフォルト」という優先順位です。

## モデルシミュレーションの統一エントリポイント

全てのモデルは最終的に`ncclTuningCostModelSimModel`を通じて呼び出されます：

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

三層フィルタ：**id が範囲外 → モデル無効 → モデルが非正の時間を返す**、いずれかの層を通過しなければ`not_valid`に進み、`timeUs`を`NCCL_TUNING_IGNORE`（負のセンチネル値）に設定し、`valid = 0`。呼び出し側は`valid == 0`を見ると候補リンクリストに追加しません。

## 設計上の考察

`modelMap`のコメントに重要な警告があります：

[FACT:src/tuning/cost_model.cc:229]

```c
// IMPORTANT: this table need must be consistent with the algRegistry in src/config/algorithm_registry.cc
```

> **[Design Inference & Architectural Trade-offs]**
> これは`modelMap`の**添字順序**が`algorithm_registry.cc`内のアルゴリズム登録順序と厳密に一致しなければならないことを意味します。もし誰かが registry に新しいアルゴリズムを挿入して`modelMap`の変更を忘れた場合、全ての id がずれ、tuning は完全に誤ったアルゴリズムを選択します。**これはテーブル駆動設計の古典的な罠です：暗黙の契約。**より堅牢な方法は添字ではなく列挙名を key として使うことですが、そうするとコンパイル時の最適化が少し犠牲になります。

---

# 三、ring.cc：Ring アルゴリズムのコスト推定

## 直感的モデル

Ring アルゴリズムは N 個の rank を環状に並べ、データが環に沿って一周ずつ伝わります。そのコストモデルは2つの問いに答える必要があります：**各ステップでどれだけのデータを伝送するか（帯域幅）**、**合計で何ステップ必要か（遅延）**。

Ring の直感は「**パイプライン**」です：N 人が円になってバケツを回すと想像してください。各人はバケツを受け取ったら少し水を入れて次の人に渡します。バケツが一周すると、全員の水が混ざり合います。バケツが回るのが速いほど（帯域幅が高い）、円が小さいほど（ステップ数が少ない）、全体が速くなります。

## データ構造：latency/bandwidth テーブル

Ring モデルは新しい構造を導入せず、推定結果を`comm->tuningContext.generalLatencies[c][algo][proto]`と`generalBandwidths[c][algo][proto]`に書き込みます。これらは三次元配列です：関数 × アルゴリズム × プロトコル。

初期化時にまず全てを -1.0（センチネル値、「未計算」を意味する）に設定します：

[FACT:src/tuning/ring.cc:31-33]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
```

-1.0 というセンチネル値は sim 段階でチェックされます：

[FACT:src/tuning/ring.cc:94-97]

```c
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ncclSuccess;
  }
```

**なぜ 0 ではなく -1.0 を使うのか？**なぜなら 0 は合法的な帯域幅値（物理的には不可能ですが）であり、-1.0 は明確に「未初期化」を意味するからです。浮動小数点比較に`==`を使うことはここでは安全です。なぜなら -1.0 は正確に表現可能だからです。

## Step-by-Step：Ring 帯域幅推定

**ステップ1：intra と inter のどちらの帯域幅を使うか決定。**単一ノード（nNodes==1）は intra、マルチノードは inter を使用：

[FACT:src/tuning/ring.cc:34-37]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float bw = (comm->nNodes == 1 || (comm->nNodes minCompCap graphs[algo].bwIntra :
                                                                                      comm->graphs[algo].bwInter;
    float busBw = bw * comm->graphs[algo].nChannels;
```

`nSteps`はアルゴリズムに必要なステップ数で、Ring の場合 AllReduce は`2*(nRanks-1)`、その他は`nRanks-1`。`busBw`は「バス帯域幅」= 単一リンク帯域幅 × channel 数。

**ステップ2：プロトコルによる割引。**LL プロトコルは帯域の半分しか使っていない（LL の flag オーバーヘッドのため）、LL128 は 92%（120/128）を使用する：

[FACT:src/tuning/ring.cc:38-42]

```c
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);
```

`0.92 = 120/128`これは LL128 では 128 バイトごとに 8 バイトが flag で、有効ペイロードは 120 バイトしかないためである。この数字はプロトコル設計から直接来ている。

**ステップ 3：有効帯域を計算する。**ここで掛けていることに注意`nRanks / nSteps`：

[FACT:src/tuning/ring.cc:44-46]

```c
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;
```

**なぜ掛けるのか`nRanks / nSteps`？**これは Ring アルゴリズムの核心的な特性である：各 rank が実際に運ぶデータ量は`nBytes * nSteps / nRanks`（データがリングを何周もするため）。したがって「有効帯域」= バス帯域 × nRanks / nSteps。AllReduce では nSteps = 2(nRanks-1) なので、有効帯域 ≈ busBw/2。

**ステップ 4：遅延を計算する。**遅延は intra と inter の二部分に分かれる：

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

L57-58 の特殊処理に注意：`maxLocalRanks == 1`（各ノードに rank が 1 つだけ）の場合、Ring の inter-node 遅延は**Tree の NET 遅延**を使用する。コメントには「preserve the pre-refactor model」とあり、つまりリファクタリング前の動作との一貫性を保つために意図的に残された「癖」である。**このような歴史的負債は成熟したシステムではよく見られる。ソースコードを読むときに「preserve」という語を見かけたら特に注意が必要で、それは多くの場合、動かせない互換性制約があることを意味する。**

**ステップ 5：関数タイプごとに累積する。**Reduce/Broadcast と AllReduce/AllGather/ReduceScatter では遅延モデルが異なる：

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

`sameChannels`はトポロジ属性で、「リング上の intra と inter のステップが同じチャネル群を使うかどうか」を表す。異なる場合、遅延に`nSteps`（各ステップで待つ必要がある）を掛ける。`netOverhead`はネットワーク post オーバーヘッドで、Simple プロトコルでは 3 を掛ける（Simple には send、recv、ack の 3 回のネットワーク往復があるため）。

## 本番での落とし穴回避：Ring/Simple の plateau 効果

`ncclTuningRingModelSim`には「plateau」を専門に扱うコードがある：

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
> **plateau とは何か？**Ring/Simple では、メッセージがある程度大きくなると、遅延はもはやメッセージに線形に増加せず、「プラトー」に張り付く——このときボトルネックが「起動オーバーヘッド」から「帯域」に変わり、帯域はすでに飽和しているためである。この現象は Blackwell NVLink で特に顕著である（NVLink の帯域が高すぎて、遅延の割合が大きくなるため）。コードでは`plateauFactor`（1.4 または 1.9）を遅延に掛けて、この「遅延が増幅される」効果をシミュレートしている。

`bytesPerRankPerChannel >= 64`はトリガ条件である：各 rank の各チャネルが少なくとも 64 バイトを転送する必要があり、そうでなければ plateau は成立しない。この 64 バイトは LL プロトコルの flag サイズに由来する。

**落とし穴シナリオ**：Blackwell 上で 1MB の AllReduce を実行し、実際の遅延がモデル予測より 40% 高いと気づいても、バグだと思ってはいけない——これは plateau 効果であり、モデルはすでにそれを計算に含めている。もし手動で`plateauFactor`を小さくすると、モデルは遅延を過小評価し、アルゴリズムの選択を誤る。

---

# 四、tree.cc と nvls.cc：Tree と NVLS のコスト推定

## 直感的モデル

**Tree アルゴリズム**は「**木構造ブロードキャスト**」である：ルートノードがデータを子ノードに配り、子ノードがさらに孫ノードに配る。その利点は**ステップ数が少ない**こと（N ではなく log N）で、小メッセージに適している；欠点は**帯域利用率が低い**こと（各非葉ノードが転送する必要があり、実際の有効帯域は半分しかない）。

**NVLS**（NVLink SHARP）は「**ハードウェアマルチキャスト**」である：スイッチが直接データを複数の GPU にコピーし、ソフトウェア転送を必要としない。その利点は**帯域が高く、遅延が低い**ことだが、特定のハードウェア（Hopper 以上）と特定の設定が必要である。

## Tree モデル：AllReduce のみを対象とする

Tree モデルには厳しい制限がある——**AllReduce にのみ有効化される**：

[FACT:src/tuning/tree.cc:21-27]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
```

> **[Design Inference & Architectural Trade-offs]**
> **なぜか？**なぜなら NCCL の Tree 実装は AllReduce のみをサポートしているためである（他の集合操作には Tree バージョンがない）。これは実装上の制約であり、理論上の制限ではない。`enabled[c] = 0`は「ハード無効化」で、`generalBandwidths = -1`よりも徹底している——前者は`ncclTuningCostModelSimModel`を L480 で即座に返すが、後者は sim 関数内でようやくチェックする。`not_valid`Tree 帯域推定

**コピー**：

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
> で、Ring の`1/3.8`よりも厳しい。`0.5`なぜ Tree の LL 効率はより低いのか？**なぜなら Tree の各中間ノードは受信と送信の両方を行い、LL の flag オーバーヘッドが双方向トラフィックで増幅されるためである。**この数字は実測から来ている。`1/3.8`Tree 遅延推定

**コピー**：

[FACT:src/tuning/tree.cc:55-58]

```c
    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
```

`2 *`はノード内ステップ数（各ノード内の rank 数から 1 を引いたもの）、`(nRanks/nNodes - 1)`はノード間ステップ数（木の高さ）である。`log2i(nNodes)`Tree の補正因子

**Tree 的修正因子**：Tree モデルは sim 段階で`treeCorrectionFactor`：

[FACT:src/tuning/tree.cc:75-79]

```c
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize proto][logSize];
```

`treeCorrectionFactor`は 3×24 のテーブルです：

[FACT:src/tuning/cost_model.cc:223-227]

```c
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};
```

`logSize = log2(nBytes >> 6)`、つまりメッセージサイズを 64 バイト単位で log2 を取ります。テーブルの添字 0-23 は 64B から 64B×2^23 ≈ 512MB に対応します。**このテーブルは実測で得られた「Tree 効率曲線」です**：小メッセージ時は効率 1.0（レイテンシ支配）、中メッセージ時は効率が 0.4-0.5 に落ち（帯域が使い切れていない）、大メッセージ時は 1.0 に戻ります（帯域を使い切る）。この「中間の窪み」は Tree アルゴリズム固有の特性です。

## NVLS モデル：ハードウェアマルチキャストのコスト

NVLS モデルはまずハードウェアがサポートしているか確認します：

[FACT:src/tuning/nvls.cc:19-24]

```c
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!ncclNvlsTransportEnabled(comm)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

次に一連のハード制約があります：Simple プロトコルのみサポート、単機では NVLSTree をサポートしない、マルチ機の NVLS には CollNet が必要：

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

**NVLS 帯域推定**効率因子を使用しています：

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
> Hopper は 0.85、Blackwell は逆に 0.74 に下がります。**なぜ新世代ハードウェアの効率が低いのか？**Blackwell の NVLink 帯域はより高いですが、NVLS のスイッチ処理能力が同比率で向上していないため、相対効率が低下します。この数値は実測値であり、理論値ではありません。

帯域計算には`(nChannels - 1) / nChannels`因子があります：

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

`(nChannels - 1) / nChannels`は NVLS が同期用に 1 チャネルを確保する必要があるためです。`(ppn - 1) / ppn`は AllGather/ReduceScatter の追加オーバーヘッドです（各 rank が前の rank のデータを待つ必要があります）。

## 本番環境の落とし穴回避：NVLS のハード制約

NVLS モデルは sim 段階でさらにランタイムチェックがあります：

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

`NCCL_MAX_NVLS_ARITY`は NVLS マルチキャストグループが収容できる最大 GPU 数です。この数を超えると、NVLS は使用できません。**落とし穴シナリオ**：16 カードの NVLink ドメインで AllGather を実行する場合、`NCCL_MAX_NVLS_ARITY`が 8 であれば、NVLS は無効化され、tuning は Ring にフォールバックします。この制限を知らないと、「NVLS はハードウェアがサポートしているのになぜ使わないのか」と誤解するでしょう。

---

# 五、対称 kernel フォールバックとエラー回復チェーン

## 直感モデル

対称 kernel（symmetric kernel）は NCCL の新機能です：すべての rank のバッファが対称メモリに登録されると、kernel はより効率的な命令でピアメモリにアクセスできます。しかし**バッファが登録されていない場合、またはプラットフォームがサポートしていない場合は、通常の kernel にフォールバックする必要があります**。このフォールバックロジックは tuning の中で最も複雑な部分です。

## Step-by-Step：フォールバック決定

フォールバックロジックは`tuning.cc:258-298`にあります。分解して見ていきましょう。

**ステップ 1：フォールバックが必要か判断する。**の入口条件：

[FACT:src/tuning/tuning.cc:258-263]

ここまでで、tuning モジュールの決定チェーンは明確になりました：トポロジグラフと通信パラメータを受け取り、コストモデルとアルゴリズム推定を通じて、マイクロ秒レベルで最適な (アルゴリズム, プロトコル, channel, warp) の組み合わせを出力します。しかし選定は始まりに過ぎません——この決定結果は下流でどのように使用されるのか？次の章では src/enqueue/enqueue.cc の幹に入り、一度の ncclAllReduce 呼び出しがパラメータ検証、アルゴリズム/プロトコル決定、channel 分割を経て、最終的に ncclInfo と ncclTaskColl 構造を生成する様子を見ていきます。これは本書が「ユーザー視点」から「エンジン視点」に切り替わる重要な章であり、一度の集合通信呼び出しが host 側で何に翻訳されるのか、そしてそれが後続の kernel 起動との境界を明らかにします。
