# 第 6 章：オペレータ発行全景：ncclAllReduce がいかにして実行可能な kernel タスクになるか

前章でtuningモジュールを完了し、NCCLがマイクロ秒レベルで1回の集合通信に対して(アルゴリズム、プロトコル、channel、warp)の組み合わせを選択することを理解しました。しかし選型結果自体は単なる数値の集まりであり、GPU kernelが理解できるタスク記述オブジェクトに「翻訳」されて初めて実際に実行できます。本章ではsrc/enqueue/enqueue.ccの主干に入り、核心的な問いに答えます：ユーザーがncclAllReduceを呼び出したとき、host側で一体何が起こるのか？ncclAllReduceからncclEnqueueCheckまで、パラメータ検証、アルゴリズム/プロトコル決定、channel分割を経て、最終的にncclInfoとncclTaskColl構造体を生成します。これは本書全体が「ユーザー視点」から「エンジン視点」へ切り替わる重要な章です。NCCLをレストランに例えるなら、enqueueモジュールは「フロントの注文システム」です：ユーザー(アプリケーション層)が「AllReduceを1つください」と言うと、フロントがそれを厨房(GPU kernel)が実行できる作業指示書に翻訳します——何番のコンロ、どの鍋を使うか、何バッチに分けるか。この翻訳層がなければ、厨房は何を作るべきか全く分かりません。

# 一、入口：ncclAllReduceがncclInfoをどのように構築するか

## 直感的モデル

`ncclAllReduce`はユーザーが直接呼び出すAPI関数です。その責務は極めて単一です：**ユーザーが渡した生のパラメータを1つの`ncclInfo`構造体にパッケージ化し、それを`ncclEnqueueCheck`**に渡します。これは銀行の窓口で手続きをするようなもので、窓口係がまずあなたの要件を標準的な申請書に記入し、その後バックエンドシステムに転送します。

この層がなければ、各集合通信APIが自分でパラメータ検証、groupセマンティクス、profiler埋め込みポイントを処理しなければならず——コードは保守不可能なほど重複します。

## データ構造：ncclInfoのメモリレイアウト

`ncclInfo`はenqueueフロー全体を貫く核心的なキャリアです。その定義は`src/include/info.h`：

[FACT:src/include/info.h:17-44]

にあります。この構造体には20以上のフィールドがあり、機能別に4つのグループに分けられます：

| フィールドグループ | フィールド | 役割 |
| --- | --- | --- |
| 集合通信パラメータ | `coll`, `sendbuff`, `recvbuff`, `count`, `datatype`, `op`, `root` | 「何をするか」を記述 |
| 通信ドメインとストリーム | `comm`, `stream` | 「どこで行うか」を記述 |
| アルゴリズム詳細 | `chunkSteps`, `sliceSteps` | 「どのように分割するか」を記述 |
| 単辺操作 | `peerWinOffset`, `peerWin`, `sigIdx`, `ctx`, `flags`, `nDesc`, `signalDescs` | RMA専用 |
| ユーザー設定 | `collConfig` | ユーザーconfigからコピーされたプライベートコピー |

注意`collConfig`のコメント：**"A config copied from config passed by user so older user config can be safely accessed during synchronous host scheduling (never at launch/replay)"** [FACT:src/include/info.h:41-43]。これは重要な設計です——ユーザーが渡したconfigポインタは`ncclGroupEnd`の前に破棄される可能性があるため、NCCLは`ncclInfo`内でコピーを作成します。

## Step-by-Step：ncclAllReduceの呼び出しチェーン

私たちは`ncclAllReduce`を例に、ユーザーの呼び出しから`ncclInfo`の構築までの完全なパスを追跡します。

**第1ステップ：ユーザーがncclAllReduceを呼び出す。**入口は`src/collectives.cc`：

[FACT:src/collectives.cc:206-211]

にあります。ここで3つのことを行います：

1. `NVTX3_FUNC_WITH_PARAMS`NVTXマーカーを打つ(Nsightなどのツールでの可視化用)

2.`ncclAllReduceConfigImpl`を呼び出し、`config = nullptr`

を渡す

**3. 結果を返す**第2ステップ：ncclAllReduceConfigImplがncclInfoを構築。

[FACT:src/collectives.cc:192-202]

これが重要なステップです：

```c
struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                        sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
                        ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
```

コピー`ncclInfo`フィールドは`ALLREDUCE_CHUNKSTEPS`の宣言順序に一対一で対応します。`ALLREDUCE_SLICESTEPS`と`src/include/collectives.h`：

[FACT:src/include/collectives.h:19-20]

`NCCL_STEPS`は`NCCL_STEPS/2`で定義されています。`NCCL_STEPS/4`はリングバッファ内のステップ数(通常8または16)なので、AllReduceのchunkStepsは

**、sliceStepsは** `ncclParseCollConfig`です。これは1つのchunkが2つのsliceを含むことを意味します。`ncclCollConfig_t*`第3ステップ：ユーザーconfigを解析。`info.collConfig`ユーザーが渡した`config == nullptr`を

**に解析します。もし**なら、このフィールドはゼロ初期化のままです。

## 第4ステップ：ncclEnqueueCheckに渡す。

> **[Design Inference & Architectural Trade-offs]**
> 設計上の考察：なぜフィールドごとの代入ではなく集約初期化を使うのか？**〔設計推論とアーキテクチャトレードオフ〕**集約初期化には2つの利点があります：1つはコンパイラがフィールド数が一致するかチェックする(1つ少ないと警告が出る)、もう1つはコードがよりコンパクトになることです。しかし欠点は`ncclInfo`フィールド順序が構造体宣言と厳密に一致しなければならない

## ことです——もし誰かが

の途中にフィールドを挿入すると、すべての集約初期化ポイントが静かにずれます。これはNCCLコード内の暗黙的な保守リスクです。

```c
ncclCollConfig_t config = {...};
ncclAllReduceConfig(..., &config);
// config 在这里被销毁（比如是栈变量，函数返回了）
```

実際の落とし穴シナリオ：ユーザーがこのようにコードを書く場合：`ncclInfo`コピー`ncclGroupEnd`もしNCCLが`info.collConfig`内でconfigをコピーしていなければ、`src/include/info.h:41-43`時に**にアクセスすると解放済みメモリを読むことになります。**。

---

# のコメントはまさにこの設計を説明するためのものです——

## configはtask append段階で解析・コピーされ、その後はユーザーポインタに依存しない

`ncclEnqueueCheck`二、ncclEnqueueCheck：パラメータ検証とgroupセマンティクス**直感的モデル**はenqueueモジュールの「総ゲート」です。すべての集合通信APIは最終的にここに集まります。その責務は：`ncclEnqueueCheck`。

パラメータの正当性検証、groupセマンティクスの処理、taskAppendの呼び出しによるタスク生成

## Step-by-Step：ncclEnqueueCheck の実行フロー

[FACT:src/enqueue/enqueue.cc:3478-3527]

段階的に分解していきます：

**ステップ1：CommCheck で通信ドメインを検証する。** `CommCheck(info->comm, info->opName, "comm")`comm ポインタが非NULLか、初期化済みかを確認する。もし comm が revoke されている場合（例えばある rank でエラーが発生した場合）、直ちにエラーを返す：

[FACT:src/enqueue/enqueue.cc:3480-3485]

**ステップ2：profiler の深さを処理する。**すでに group 内部にある場合（`profilerGroupDepth > 0`）、深さカウンタをインクリメントする。これは暗黙的な`ncclGroupStartInternal`/`ncclGroupEndInternal`呼び出しを正しく処理するためである。

**ステップ3：内部 group に入る。** `ncclGroupStartInternal()`は NCCL 内部の group メカニズムである。**重要なポイント**：ユーザーが明示的に`ncclGroupStart`を呼び出さなくても、NCCL は各 API 呼び出しに対して暗黙的な group を作成する。これにより単一呼び出しの原子性が保証される。

**ステップ4：comm が準備完了であることを確認する。** `ncclCommEnsureReady(info->comm)`通信ドメインの初期化完了（例えば bootstrap の完了、接続の確立）を待つ。

**ステップ5：ArgsCheck によるパラメータ検証。**これが最も複雑な検証ステップである：

[FACT:src/enqueue/enqueue.cc:3497-3503]

注意`checkMode`の処理：もし`ncclCheckModeDebugGlobal`，`ArgsCheck`の場合は info をキューに入れ、`ncclGroupEnd`時にグローバル検証（例えば全 rank の count が一致するかの確認）を行う。

**ステップ6：taskAppend を呼び出す。**これが核心的な変換ステップである：

[FACT:src/enqueue/enqueue.cc:3513]

**ステップ7：opCount をインクリメントする。**キューへの追加が成功するたびに、`comm->opCount++`。このカウンタは send/recv 操作のマッチングに使用され、profiler のタイムラインの根拠にもなる。

**ステップ8：group を退出する。** `ncclGroupEndInternal()`depth が 0 に下がると、実際の group 操作（スケジューリング、カーネル起動）がトリガーされる。

## 並行制御：group セマンティクスとスレッドセーフティ

> **[Design Inference & Architectural Trade-offs]**
> `ncclGroupStartInternal`/`ncclGroupEndInternal`スレッドローカルストレージ（TLS）を使用して group 状態を維持する。これはつまり**同一スレッド内の複数の API 呼び出しは1つの group に統合される**が、異なるスレッドの呼び出しは独立している。これが NCCL がマルチスレッド呼び出しをサポートする基盤である。

陥りやすい罠：ユーザーが`ncclGroupStart`と`ncclGroupEnd`の間に NCCL 以外の CUDA API（例えば`cudaMemcpy`）を呼び出すと、stream の順序問題が発生する可能性がある。NCCL の group メカニズムは group 内の操作がすべて同じ stream 上にあることを前提としている。

## エラー回復チェーン

`ncclEnqueueCheck`のエラー処理には精巧な設計がある：

[FACT:src/enqueue/enqueue.cc:3524-3526]

もし`taskAppend`が失敗し、かつ comm が非ブロッキングモードの場合、`ncclCommSetAsyncError`を呼び出してエラーを記録する。これにより後続の API 呼び出しは再試行せずに直ちにエラーを返す。これは非同期エラー伝播メカニズムである。

---

# 三、taskAppend：タスク分配の十字路

## 直感的モデル

`taskAppend`は enqueue モジュールの「交通ハブ」である。それは`info->coll`の値に基づいて、タスクを異なる処理パスに分配する：P2P、RMA、CE、または通常の集合通信。これは郵便局の仕分けセンターのようなもので——封筒の住所に基づいて、手紙を異なるポストに投函する。

もしこの分配層がなければ、すべてのタイプの操作が1つの巨大な if-else に詰め込まれ、コードの保守が困難になる。

## Step-by-Step：taskAppend の分配ロジック

[FACT:src/enqueue/enqueue.cc:3337-3476]

**ステップ1：新アーキテクチャが有効かどうかを判定する。** `ncclParamEnqueueRearchEnable()`は環境変数スイッチである（デフォルト 0）。有効な場合、`rawTaskAppend`パスを通る——これは NCCL が開発中の新しいタスクモデルである。

**ステップ2：P2P 分配。**Send/Recv の場合、`p2pTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3343-3345]

**ステップ3：RMA 分配。**PutSignal/Signal/WaitSignal の場合、`rmaTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3346-3347]

**ステップ4：空の集合通信は早期リターン。** `if (info->count == 0) return ncclSuccess;`——count が 0 の集合通信は直接破棄される。

**ステップ5：アルゴリズム選択の検証。** `ncclCollConfigGetAlgMask`ユーザーが渡したアルゴリズム選択が正当かどうかを検証する：

[FACT:src/enqueue/enqueue.cc:3357-3358]

**ステップ6：FP8 型チェック。**FP8 リダクションには sm90+ が必要：

[FACT:src/enqueue/enqueue.cc:3360-3366]

**ステップ7：リダクション操作の変換。** `hostToDevRedOp`host 側の`ncclRedOp_t`をデバイス側の`ncclDevRedOpFull`：

[FACT:src/enqueue/enqueue.cc:3370-3371]

**ステップ8：単一 rank の早期リターン。**もし`comm->nRanks == 1`の場合、直接`ncclLaunchOneRank`を呼び出してローカルリダクションを実行し、タスクを生成する必要はない：

[FACT:src/enqueue/enqueue.cc:3373-3377]

**ステップ9：マルチ rank パス。**これが最も複雑な分岐であり、CE ルーティング、AllToAll/Gather/Scatter の降格、および通常の集合通信を含む：

[FACT:src/enqueue/enqueue.cc:3378-3470]

## データ構造：ncclTaskColl のフィールド

`collTaskAppend`は`ncclTaskColl`を生成する場所である。その核心的なロジックを見てみよう：

[FACT:src/enqueue/enqueue.cc:2757-2851]

主要フィールドの代入：

| フィールド | ソース | 意味 |
| --- | --- | --- |
| `func` | `info->coll` | 集合通信タイプ |
| `sendbuff`/`recvbuff` | `info->sendbuff`/`recvbuff` | バッファポインタ |
| `count` | `info->count` | 要素数 |
| `datatype` | `info->datatype` | データ型 |
| `trafficBytes` | `count * elementSize * ncclFuncTrafficPerByte` | トラフィック推定 |
| `opHost`/`opDev` | `info->op`/`opDev` | リダクション操作 |
| `chunkSteps`/`sliceSteps` | `info->chunkSteps`/`sliceSteps` | 分割ステップ数 |
| `minCTAs`/`maxCTAs`/`nvlsCTAs` | 設定解析 | リソース上限 |
| `algMask` | `ncclCollConfigGetAlgMask` | アルゴリズム選択マスク |

注意`trafficBytes`の計算：

[FACT:src/enqueue/enqueue.cc:2813]

`ncclFuncTrafficPerByte`は各集合通信のトラフィック倍数を返す：

[FACT:src/enqueue/enqueue.cc:123-134]

AllReduce は 2 を返し（reduce + broadcast が必要なため）、AllGather/ReduceScatter は nRanks を返し、その他は 1 を返す。

## 設計思考：なぜ AllGather/Broadcast は int8 に変換するのか？

[FACT:src/enqueue/enqueue.cc:2808-2812]

AllGather と Broadcast は count に elementSize を掛け、そして datatype を`ncclInt8`。これは最適化です：**これらの2つの操作はリダクションを伴わないため、データ型を気にする必要がなく、一律にバイト単位で処理することでカーネルロジックを簡素化できます**。

## 本番での落とし穴：CTAPolicy の解析順序

[FACT:src/enqueue/enqueue.cc:3390-3397]

CTAPolicy の解析には微妙な優先順位があります：**env > per-call > comm**。そして`NCCL_CTA_POLICY_ZERO`が`NCCL_CTA_POLICY_EFFICIENCY`より優先されます。ユーザーが両方のフラグを同時に設定した場合、ZERO が有効になります。

実際の落とし穴シナリオ：ユーザーが`NCCL_CTA_POLICY=EFFICIENCY`を設定したが、CE パスが使用されていないことに気づきました。原因は CE ルーティングには`CTAPolicy & NCCL_CTA_POLICY_ZERO`が真であることが必要ですが、EFFICIENCY はこの条件を満たしません。

---

# 四、ncclPrepareTasks：タスクリストからスケジューリングキューへ

## 直感的モデル

`ncclPrepareTasks`は enqueue モジュールの「プリプロセッサ」です。散在するタスクリストを (func, op, datatype) でバケット分けし、各バケットに対してアルゴリズムとプロトコルを計算します。これは図書館の司書のようなものです——返却された本をまずカテゴリ別に分類し、次に各カテゴリの本をどの棚に置くかを決定します。

このステップがなければ、後続の`scheduleCollTasksToPlan`は各タスクごとに個別にアルゴリズムを計算する必要があり、効率が極めて低くなります。

## Step-by-Step：ncclPrepareTasks のバケット分けロジック

[FACT:src/enqueue/enqueue.cc:423-642]

**ステップ1：Broadcast タスクの変換。**broadcast ピアが1つだけの場合、broadcast タスクを coll タスクに変換します：

[FACT:src/enqueue/enqueue.cc:430-461]

ここで`bcastTask`のフィールドを新しい`ncclTaskColl`にコピーし、`trafficBytes`を計算します。その後`memPool_ncclTaskBcast`から元のタスクを解放します。

**ステップ2：(func, op, datatype) によるバケット分け。**タスクは sorter から size の降順で出てきて、その後`tasksByFnOpTy`配列に振り分けられます：

[FACT:src/enqueue/enqueue.cc:464-487]

インデックス計算：`((int)task->func * ncclNumDevRedOps + (int)task->opDev.op) * ncclNumTypes + (int)task->datatype`。これは3次元配列の線形化です。

**ステップ3：集約とアルゴリズム選択。**各バケットについて、サイズが近いタスク（4倍以内）を集約し、その後`ncclGetAlgoInfo`：

[FACT:src/enqueue/enqueue.cc:503-547]

**を呼び出します**ステップ4：(collnet, nvls) によるバケット分け。`collBins[2][2]`：

[FACT:src/enqueue/enqueue.cc:517-544]

**アルゴリズムタイプに基づいて、タスクを**に振り分けます`planner->collTaskQueue`：

[FACT:src/enqueue/enqueue.cc:553-557]

## ステップ5：最終キューの結合。

`ncclTaskCollSorter`4つのバケットを`trafficBytes`に結合します`ncclTaskCollSorterInsert`データ構造：ncclTaskCollSorter`ncclTaskCollSorterDequeueAll`は

> **[Design Inference & Architectural Trade-offs]**
> タスクを正しい位置に挿入し、**すべてのタスクを順番に取り出します。**〔設計推論とアーキテクチャのトレードオフ〕

## このソーターの設計動機は：

[FACT:src/enqueue/enqueue.cc:572-583]

大きなタスクを優先的にスケジューリングする`comm->runtimeConn`。大きなタスクは転送時間が長いため、先に起動することで計算と通信をより良くオーバーラップできます。`algoNeedConnect`並行制御：runtimeConn と接続確立

## もし

[FACT:src/enqueue/enqueue.cc:507-508]

が真（ランタイム接続モード）で、あるアルゴリズムの channel がまだ初期化されていない場合、`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes`をマークします。これにより後続で接続確立がトリガーされます。`aggIsolate`本番での落とし穴：集約の境界条件`maxCTAs`），`aggIsolate`集約条件は

であり、かつ両方のタスクが`maxCTAs=4`を設定していないことです。ユーザーが per-call config を設定した場合（例えば`aggIsolate`が true に設定される）、このタスクは集約されません。`collTaskAppend`実際の落とし穴シナリオ：ユーザーがある AllReduce に対して

[FACT:src/enqueue/enqueue.cc:2821-2822]

---

# を設定し、4つの CTA のみを使用することを期待しました。しかし集約ロジックにより、このタスクは隣接するタスクと統合される可能性があり、実際に使用される CTA 数が期待に合わなくなります。解決策は

## を設定することです——NCCL は

`scheduleCollTasksToPlan`内で既にこれを処理しています：

五、scheduleCollTasksToPlan：channel 分割と予算制御

## 直感的モデル

[FACT:src/enqueue/enqueue.cc:644-947]

**は enqueue モジュールの「スケジューラ」です。タスクを具体的な channel に割り当て、各 channel のデータ分割を計算します。これは工場の生産計画システムのようなものです——各生産ラインが何を作り、どれだけ作るかを決定します。**このステップがなければ、GPU kernel は自分がどの部分のデータを処理すべきかわかりません。

[FACT:src/enqueue/enqueue.cc:648-689]

`ncclTestBudget`Step-by-Step：channel 分割アルゴリズム

[FACT:src/enqueue/enqueue.cc:343-349]

**ステップ1：予算見積もり。**まずこの plan に収められるタスク数を推定します：`trafficPerChannel`：

[FACT:src/enqueue/enqueue.cc:701-707]

**作業バイト数が予算を超えているかチェックします：**ステップ2：各 channel のトラフィックを計算。

[FACT:src/enqueue/enqueue.cc:709-739]

**kind（collnet/nvls）に基づいて**を計算します

[FACT:src/enqueue/enqueue.cc:740-845]

ステップ3：Collnet パス。

- `cellSize`collnet アルゴリズムの場合、channel 割り当ては比較的簡単です：`MinTrafficPerChannel`（32KB）
- `cells`ステップ4：通常パスの cell 分割。
- `cellsPerChannel`これが最も複雑な部分です。NCCL はデータを「cell」に分割し、各 cell は最小転送単位です：
- `cellsLo`/`cellsHi`主要変数：

**：各 cell のバイト数、最低**：総 cell 数`calcCollChunking`：

[FACT:src/enqueue/enqueue.cc:811-825]

**：各 channel が処理する cell 数**：先頭と末尾の channel の cell 数（満たない可能性がある）

[FACT:src/enqueue/enqueue.cc:844-894]

## ステップ5：chunkGrains の計算。

`ncclDevWorkColl`各 channel セグメントに対して

| を呼び出します | ステップ6：proxyOp の生成。 |
| --- | --- |
| `sendbuff`/`recvbuff` | 各 channel に対して proxy 操作を生成します： |
| `channelLo`/`channelHi` | データ構造：ncclDevWorkColl |
| `cbd.countLo`/`countMid`/`countHi` | はデバイス側の作業記述子です。その主要フィールド： |
| `cbd.chunkGrainsLo`/`Mid`/`Hi` | フィールド |
| `direct` | 意味 |

## バッファポインタ

[FACT:src/enqueue/enqueue.cc:897]

channel 範囲`(2ull << channelHi) - (1ull << channelLo)`。例えば channelLo=2, channelHi=5 の場合、結果は`(2<<5) - (1<<2) = 64 - 4 = 60 = 0b111100`、つまり bit 2-5 が設定されます。

## 本番での落とし穴：予算オーバーフロー

[FACT:src/enqueue/enqueue.cc:792-794]

予算が足りない場合、直接`ncclSuccess`を返し、外側のループに新しい plan を作成させます。これはエレガントなデグレード戦略です——**エラーにはせず、バッチ処理するだけ**。

実際の落とし穴シナリオ：もし`NCCL_WORK_FIFO_BYTES`を小さく設定しすぎると、各 plan にわずかなタスクしか収容できなくなり、kernel の起動回数が増えて性能が低下します。

---

# 六、finishPlan：タスクから kernel パラメータへ

## 直感モデル

`finishPlan`は enqueue モジュールの「パッケージャー」です。タスク、batch、proxyOp を kernel が直接読み取れるパラメータ構造にパッケージします。これは宅配便の梱包のようなものです——バラバラの荷物を箱に詰め、送り状を貼り、発送を待ちます。

## Step-by-Step：finishPlan のパッケージングロジック

[FACT:src/enqueue/enqueue.cc:236-330]

**ステップ 1：ストレージタイプを決定する。**すべての作業が kernel args に収まる場合、`ncclDevWorkStorageTypeArgs`：

[FACT:src/enqueue/enqueue.cc:244-250]

**ステップ 2：kernelArgs を割り当てる。**メモリスタックから割り当て：

[FACT:src/enqueue/enqueue.cc:251-255]

**ステップ 3：Round-robin で batch を配置する。**各 channel の最初の batch は`batchZero[blockIdx.x]`：

[FACT:src/enqueue/enqueue.cc:257-280]

**ステップ 4：proxyOp キューをマージする。**opCount でマージソート：

[FACT:src/enqueue/enqueue.cc:282-329]

## データ構造：ncclDevKernelArgs

`ncclDevKernelArgs`は kernel に渡されるパラメータ構造です。以下を含みます：

- `comm`：デバイス側コミュニケータ
- `channelMask`：channel ビットマスク
- `workStorageType`：ワークストレージタイプ
- `workBuf`：ワークバッファポインタ
- `workMask`：ワークバッファマスク

## 本番での落とし穴：batch の順序

[FACT:src/enqueue/enqueue.cc:257-259]

コメントには明確に書かれています："The first batch for each channel must be located at batchZero[blockIdx.x]"。この順序が間違っていると、kernel が誤った batch を読み取り、データ破損を引き起こします。

---

# 本章のまとめ

本章では、`ncclAllReduce`から`ncclTaskColl`までの完全なパスを追跡しました：

1. **ncclAllReduce**を構築し`ncclInfo`、ユーザーパラメータをパッケージ

2. **ncclEnqueueCheck**パラメータを検証し、group セマンティクスを処理

3. **taskAppend**操作タイプに応じて異なるパスにディスパッチ

4. **collTaskAppend**を生成し`ncclTaskColl`、設定を解析

5. **ncclPrepareTasks**(func, op, datatype) でバケット化し、アルゴリズムを計算

6. **scheduleCollTasksToPlan**channel を分割し、`ncclDevWorkColl`

7. **finishPlan**を生成して kernel パラメータにパッケージ

重要な設計思想：

- **階層的疎結合**：各関数は一つのことだけを行い、`ncclInfo`と`ncclTaskColl`を通じて状態を伝達
- **予算制御**：`ncclTestBudget`を通じて各 plan のサイズを制御
- **集約最適化**：サイズが近いタスクが集約され、kernel の起動回数を削減
- **設定の優先順位**：env > per-call > comm

次章では`task_sched`に入り、NCCL がマルチ channel・マルチ kernel の実行順序をどのように編成するかを見ていきます。

# 本章の考察とセルフチェック

Q1: もし`collTaskAppend`の`aggIsolate`判定を削除した場合（つまり`src/enqueue/enqueue.cc:2821-2822`が常に false を返す場合）、どのようなシナリオでユーザーが設定した`maxCTAs`が無効になりますか？なぜですか？

**参考解説**：`aggIsolate`の役割は「このタスクは集約できない」とマークすることです。この判定を削除すると、per-call config を設定したタスクが隣接タスクとマージされます。`ncclPrepareTasks`の集約ループ（`src/enqueue/enqueue.cc:507-508`）では、集約条件は`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes && !aggBeg->aggIsolate && !aggEnd->aggIsolate`です。もし`aggIsolate`が常に false なら、`maxCTAs=4`を設定したタスクでも、`maxCTAs=32`のタスクとマージされる可能性があります。マージ後の`agg`は両者のある種の組み合わせを取ります（`ncclGetAlgoInfo`の実装に依存）、その結果、実際に使用される CTA 数がユーザーの期待に合わなくなります。

さらに深刻なのは、`scheduleCollTasksToPlan`において（`src/enqueue/enqueue.cc:665-666`），`taskAggIsolate`は per-call リソースが設定されたタスクが単独で一つの plan を占めることを保証するために使用されます。この判定が無効になると、複数のタスクが plan の channel 予算を共有し、リソース配分が期待に合わなくなります。

Q2:`ncclEnqueueCheck`において、もし`ncclGroupEndInternal()`がエラーを返した場合（例えばある rank の ArgsCheck が失敗）、しかし`taskAppend`がすでに正常に実行された場合、何が起こりますか？NCCL はどのように状態の一貫性を保証しますか？

**参考解説**：`src/enqueue/enqueue.cc:3513-3519`の制御フローを見てみましょう：

```c
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
info->comm->opCount++;
exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
```

もし`taskAppend`が成功したが`ncclGroupEndInternal`が失敗した場合、`opCount`はすでにインクリメントされています。これにより後続操作の opCount が対端と一致しなくなり、hang を引き起こす可能性があります。

NCCL の処理方法は：`ncclGroupErrCheck(ret)`はエラーがあるかチェックし、あれば comm のエラー状態を設定します。後続の API 呼び出しは`ncclCommGetAsyncError`を通じてこのエラーを検出し、即座にリターンします。これは「フェイルファスト」戦略です——一度エラーが発生すると、comm 全体がエラー状態に入り、回復を試みなくなります。

本番環境では、group エラーが一度発生すると、ユーザーは communicator を破棄して再構築する必要があります。

Q3: `scheduleCollTasksToPlan`の cell 分割アルゴリズム（`src/enqueue/enqueue.cc:740-845`）には境界条件があります：`cellsLo == 0`のとき、最小の channel をスキップします。もしこのスキップロジックにバグがある場合（例えば`channelId`が正しくインクリメントされない）、どのような結果を引き起こしますか？

**参考解説**：`src/enqueue/enqueue.cc:770-780`：

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

もし`channelId`が正しくインクリメントされないと、次のタスクが誤った channel から割り当てを開始します。これにより：

1. **channel の重複**：二つのタスクが同じ channel の同じデータ区間を割り当てられる可能性

2. **データ破損**：kernel がデータを重複処理または欠落させる

3. **性能低下**：channel の負荷不均衡

さらに隠蔽性が高いのは、この種のバグが特定のメッセージサイズでのみ発生する可能性があることだ（`cellsLo == 0`のとき）。再現が難しい。NCCL は`plan->channelMask |= (2ull << devWork->channelHi) - (1ull << devWork->channelLo)`によって使用済みの channel を追跡するが、これは単なる記録であり、重複を防ぐことはできない。

ここまでで、ncclAllReduce がユーザーの呼び出しから一連の実行可能な kernel タスクへとどのように変化するかを明らかにした：パラメータ検証、アルゴリズム/プロトコルの決定、channel 分割、最終的に ncclInfo と ncclTaskColl を生成する。しかしタスクが作成されただけでは最初の一歩に過ぎない——それらは複数の channel にスケジュールされ、kernel 起動パラメータを生成し、group セマンティクスの下でバッチ送信と依存関係の順序付けを処理する必要がある。次の章では src/enqueue/task_sched と src/enqueue/task_prep を深掘りし、「なぜ1回の AllReduce で複数の kernel が起動されるのか、それらの間の順序と依存関係はどのように保証されるのか」に答え、同時に src/group.cc の ncclGroupStart/ncclGroupEnd がどのように複数の API 呼び出しを1回の送信に統合するかを明らかにする。
