# 第 24 章：アーキテクチャの進化と将来の方向性：静的通信からプログラマブル通信へ

前章では、コミュニティが NCCL コアを中心にどのように周辺エコシステムを構築しているかを見た：Python バインディング、Rust バインディング、エキスパート並列通信、超帯域幅プリミティブ、通信チェックポイント。これらのプロジェクトはすべて NCCL の安定した API を再利用しているが、その要求は従来の集合通信の範疇を超えている——エキスパート並列は細粒度のポイントツーポイント送受信を必要とし、チェックポイントは通信状態の一時停止/再開を必要とし、超帯域幅プリミティブは標準の集合操作をバイパスして直接ネットワークを操作する必要がある。これらの要求は同じ問題を指し示している：NCCL の固定された集合操作モデルは、より柔軟な通信ニーズによって押し広げられつつある。本章ではもはや単一のモジュールを見るのではなく、ソースコードにすでに現れている進化の痕跡から、NCCL がどこへ向かおうとしているかを議論する。具体的には、絡み合う三つの進化の力が分析される：通信プリミティブが固定集合からプログラマブルへ——src/rma/rma.cc の RMA タスクスケジューリングにより、上位層は AllReduce を呼び出すだけでなく、Put/Signal/WaitSignal プリミティブを組み合わせることができる；ネットワーク起動が host proxy から GPU 直発へ——src/gin/gin_host.cc の GIN バックエンド管理により、GPU kernel が直接ネットワークカードを駆動する；メモリモデルが登録バッファから対称メモリへ——src/sym_kernels.cc の対称メモリ kernel 選択により、すべての rank が同一の仮想アドレスセットで互いのバッファにアクセスする。これら三つの力は孤立しておらず、同じインフラストラクチャを共有している：src/nccl_device/core.cc の team 抽象と src/devcomm/devcomm_v23100.cc のバージョン化された DevComm。それらがどのように噛み合っているかを理解すれば、NCCL が「集合通信ライブラリ」から「プログラマブル通信エンジン」への進化する論理を理解できる。

# 一、プログラマブル通信プリミティブ：RMA が「固定レシピ」を「ビュッフェ」に変える方法

## 直感的モデル

従来の NCCL の集合通信は固定セットメニューのようなものだ。AllReduce を注文すれば、キッチンは AllReduce の手順通りに調理を完了する。しかし、エキスパート並列（MoE）のシナリオでは、各トークンを異なるエキスパートに送信する必要があり、その送信パターンはコンパイル時には全く分からない——これはビュッフェのようなもので、何を取るか、どれだけ取るか、いつ取るかを自分で決めなければならない。

RMA とは、NCCL が上位層に提供する「ビュッフェ台」である：Put（データを相手側メモリに書き込む）、Signal（相手側に通知する）、WaitSignal（相手側の信号を待つ）。上位フレームワークはこれら3つのプリミティブを自由に組み合わせて、任意の通信パターンを実現できる。

RMA がなければ、MoE の all-to-all は複数回の小規模な集合操作でしか模擬できず、毎回完全なカーネル起動と同期フローを経る必要があり、遅延が許容できないほど高くなる。

## データ構造とメモリレイアウト

RMA の核心的なデータ構造は`ncclTaskRma`（タスク記述）と`ncclRmaArgs`（計画パラメータ）である。まず`ncclRmaArgs`のフィールドを見てみよう。これは`scheduleRmaTasksToPlan`で初期化される。

[FACT:src/rma/rma.cc:166-171]

```cpp
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
plan->rmaArgs->nRmaTasks = 0;
plan->rmaArgs->nRmaTasksProxy = 0;
plan->rmaArgs->nRmaTasksCe = 0;
```

ここでの重要なフィールドは`nRmaTasksProxy`と`nRmaTasksCe`である。これらは RMA タスクを2つの実行パスに分ける：

- **CE パス**（Copy Engine、コピーエンジン）：対象 rank が LSA（Local Symmetric Access、ローカル対称アクセス）範囲内にあり、GPU のコピーエンジンで直接完了でき、ネットワークは不要である。
- **Proxy パス**：対象 rank が LSA 範囲内にない場合、host proxy スレッドがネットワークを駆動する必要がある。

> **[Design Inference & Architectural Trade-offs]**
> この二分法の設計動機は非常に直接的である：LSA 範囲内の通信は NVLink または PCIe を通り、帯域幅が高く遅延が低いため、CE 非同期コピーが最も効率的である。マシン間通信は必ずネットワークカードを通るため、proxy スレッドが駆動するしかない。2種類のタスクを分けてスケジューリングすることで、CE と proxy を直列に待つのではなく並列に実行できる。

`ncclTaskRma`自体は`peers`、`nsignals`、`signalIdxs`という3つの配列ポインタを含み、それぞれ対象 rank、信号数、信号インデックスを記録する。WaitSignal タスクでは、1つのタスクが複数の peer を待つことができる。Put/Signal タスクでは、1つのタスクは1つの peer のみを対象とする。

## Step-by-Step Walkthrough：ある WaitSignal のスケジューリング

具体的なシナリオを代入しよう：rank 0 が`ncclWaitSignal`を呼び出し、rank 1 と rank 3 の信号を待つ。rank 1 は LSA 範囲内にあり、rank 3 はそうでないと仮定する。

**第一步：最初の非空コンテキストキューを見つける。**

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

RMA タスクは context ごとにキューに分けられ、各 context は独立した RMA チャネルである。ここではタスクがある最初の context を見つけ、そのキューを取り出す。

**第二步：最初のタスクを取り出し、タイプを判定する。**

[FACT:src/rma/rma.cc:163-168]

```cpp
struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
```

`firstTask->func`が`ncclFuncWaitSignal`であり、WaitSignal 分岐に入る。

**第三步：LSA 到達可能性に従って peer を分割する。**

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

`isLsaAccessible`が`comm->devrState.lsaRankList`を走査し、peer が LSA チーム内にあるか判定する。rank 1 は LSA 内にあり、CE リストに入る。rank 3 はそうでなく、Proxy リストに入る。

**第四步：CE と Proxy それぞれに新しいタスクを作成する。**

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

元の1つの WaitSignal タスクが2つに分割される：CE タスクは rank 1 を待ち、Proxy タスクは rank 3 を待つ。2つのタスクは並列に実行できる——CE パスは GPU 上で待ち、Proxy パスは host スレッド上で待つ。

**第五步：元のタスクを解放する。**

[FACT:src/rma/rma.cc:249-251]

```cpp
planner->nTasksRma -= 1;
ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
```

元のタスクはすでに2つの新しいタスクに分割されており、メモリプールに解放される。

## 並行制御とハードウェア相互作用

RMA の並列実行は`ncclRmaWaitSignal`に現れる。

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

このコードは CUDA event でストリーム間同期を行う：まず入力ストリームで event を記録し、CE ストリームにこの event を待たせ、次に2つのストリームでそれぞれ proxy と CE タスクを起動し、最後に入力ストリームに CE ストリームの event を待たせる。これにより2つのパスが並行して進むが、外部からは1つの同期操作として見える。

> **[Design Inference & Architectural Trade-offs]**
> ここでの設計トレードオフは：並列実行は遅延を低減できるが、追加の event 記録とストリーム同期のオーバーヘッドを導入する。小さいメッセージでは、このオーバーヘッドが並列の利益を上回る可能性がある。大きいメッセージでは、並列の利益が顕著である。NCCL はここで適応的な判断を行わず、一律に並列パスを通る——RMA の典型的なシナリオは大きいメッセージの細粒度通信だからである。

## 本番環境の落とし穴回避ガイド

**落とし穴 1：LSA 到達可能性の判定ミスによりタスクが誤ったパスを通る。** `isLsaAccessible`が`lsaRankList`を走査し、もし`lsaSize`が 0 の場合（例えば単一 rank の通信ドメイン）、すべての peer が到達不可能と判定され、すべて Proxy パスを通る。これは小規模テストでは露見しないが、大規模デプロイでは性能が急落する。調査方法は`scheduleRmaTasksToPlan`の INFO ログで`nRmaTasksProxy`と`nRmaTasksCe`の比率を見ることである。

**落とし穴 2：WaitSignal タスク分割後の peer 配列のライフサイクル。**CE パスの`peersCe`は`ncclMemoryStackAlloc`で割り当てられ、ライフサイクルは`comm->memScoped`に従う。Proxy パスの`peersProxy`は`ncclCalloc`で割り当てられ、タスク実行完了後に手動で`free`。もし Proxy タスクの作成に失敗した場合、`fail`ブランチはこれらの配列を解放します。

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

**落とし穴 3：Put/Signal タスクのクロス context バッチ処理。**Put/Signal ブランチでは、NCCL はすべての context の put/signal タスクを同じ plan にまとめますが、WaitSignal に遭遇すると停止します。

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

この設計の意図は、1 回の kernel 起動ですべての context の put/signal をカバーし、起動オーバーヘッドを削減することです。ただし、各 context のキューは最初の WaitSignal までしか消費せず、per-context FIFO 順序を保証します。上位層が同じ context 内で put と waitSignal を交互に呼び出すと、バッチ効果は大幅に低下します——これは RMA 使用時に注意が必要なパターンです。

---

# 二、GPU 直送ネットワーク：GIN が kernel を host proxy から迂回させる仕組み

## 直感的モデル

従来の NCCL のネットワーク通信は手紙を送るようなものです：GPU kernel がデータをバッファに置き、host proxy スレッドがデータを NIC に渡し、NIC が送信します。GIN は GPU kernel が直接相手のメールボックスに手紙を投函するようなものです——kernel が NIC の送信キューに直接書き込み、NIC が GPU メモリを直接読み取ります。

GIN がなければ、ネットワーク通信のたびに host メモリを経由するため、レイテンシは少なくとも PCIe の往復 1 回分増加します。MoE のような細粒度通信では、このレイテンシは致命的です。

## データ構造とメモリレイアウト

GIN の核心状態は`ncclGinState`であり、複数のバックエンド（backend）と複数の DevComm を管理します。まずバックエンドバージョン互換テーブルを見てみましょう。

[FACT:src/gin/gin_host.cc:27-33]

```cpp
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5), NCCL_VERSION(2, 32, 0)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
constexpr int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0), NCCL_VERSION(2, 32, 0)};
```

これらの配列のインデックスはバックエンドバージョン番号で、値は互換性のある最低 NCCL バージョンです。例えば`proxyBackendMinVersions[3]`はバックエンドバージョン 3 に対応し、NCCL 2.32.0 以上を要求します。この設計により、NCCL はコンパイル時にバインドするのではなく、実行時にデバイスコードバージョンに応じて適切なバックエンドバージョンを選択できます。

> **[Design Inference & Architectural Trade-offs]**
> このバージョン互換テーブルの設計動機は、GIN バックエンド（NIC ドライバ、ファームウェア）と NCCL ライブラリのバージョン進化のペースが異なることです。バージョン要件をハードコードすると、どちらか一方のアップグレードで非互換が発生します。配列でバージョンマッピングを行うことで、実行時に動的に選択でき、古いバックエンドとの後方互換性を保てます。

`ncclGinStateDevComm`は各 DevComm の GIN 状態であり、`contextCount`、`backendIndex`、`ginCtx[]`、`devHandles[]`などのフィールドを含みます。これはリンクリストとして`ginState->devComms`に接続されます。

## Step-by-Step Walkthrough：1 回の GIN 接続確立

シナリオを想定しましょう：rank 0 が通信ドメインを初期化し、GIN 接続を確立する必要があります。

**第一步：GIN が有効かつサポートされているか確認。**

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

`ncclParamGinEnable()`は環境変数`NCCL_GIN_ENABLE`を読み取り、デフォルトは 1 です。ユーザーが明示的に無効化した場合、直接エラーを返します。

**第二步：対称メモリのサポートを確認。**

[FACT:src/gin/gin_host.cc:111-114]

```cpp
if (!comm->symmetricSupport) {
  WARN("Communicator does not support symmetric memory!");
  return ncclInternalError;
}
```

GIN は対称メモリに依存します——GPU kernel が相手側バッファの仮想アドレスを知る必要があり、対称メモリでのみアドレスの一致が保証されるからです。

**第三步：ローカル GIN デバイスリストを取得。**

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

`ncclTopoGetLocalGinDevs`はトポロジグラフから GIN をサポートするすべての NIC を見つけます。もし`NCCL_GIN_MAX_CONNECTIONS`を超える場合、先頭のいくつかのみを取り、警告を出力します。

**第四步：GIN チームを計算。**

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

各バックエンドはまず`devices`を呼び出してデバイス数を取得し、その後各接続に対して listen→getProperties→allGather→connect→closeListen のフローを実行します。`bootstrapAllGather`はすべての rank 間で handle を交換し、各 rank が相手側の接続情報を知ることができるようにします。

## 並行制御とハードウェア相互作用

GIN の進捗スレッドは核心的な並行メカニズムです。

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

ここにはいくつかの重要な設計があります：

1. **CPU アフィニティ**：`ncclOsSetAffinity`は進捗スレッドを指定された CPU コアにバインドし、スレッドマイグレーションによるキャッシュ無効化を回避します。

2. **書き込みロックバックオフ**：`writePending`はアトミックフラグで、メインスレッドが`devComms`リンクリストを変更する際に先にセットし、進捗スレッドがそれを見ると自発的に yield し、ロック競合を回避します。

3. **読み書きロック**：`devCommRwMutex`は`shared_timed_mutex`であり、進捗スレッドは読み取りロックを保持してリンクリストを走査し、メインスレッドは書き込みロックを保持してリンクリストを変更します。

4. **スレッド分担**：スレッド t は接続 t, t+proxyNthreads, t+2*proxyNthreads, ... を担当し、stride ループで負荷分散を実現します。

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

この書き込みロックの実装は書き手が 1 つ（メインスレッド）のみであることを前提としているため、追加のミューテックスは不要です。`writePending`は先にセットしてからロックを取得し、進捗スレッドがロック取得前に書き込み意図を確認して自発的に退避できるようにします。

## 本番環境の落とし穴回避ガイド

**落とし穴 1：GIN 接続数の不一致による AllGather デッドロック。**各 rank の`ginCommCount`は異なる可能性があり（ローカル NIC 数に依存）、NCCL は`bootstrapAllGather`で全 rank の最小値を取ります。

[FACT:src/gin/gin_host.cc:176-180]

```cpp
ginCommCountHandles[comm->rank] = backend->ginCommCount;
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
for (int r = 0; r nRanks; r++) {
  backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
}
```

ある rank の NIC 数が他の rank より少ない場合、すべての rank が最小値に揃えられます。これにより接続の対称性は保証されますが、NIC リソースが無駄になります。

**落とし穴 2：proxyNthreads が ginCommCount を超えるとスレッドが空回りする。**ユーザーが設定した場合`NCCL_GIN_PROXY_NTHREADS`より大きい`ginCommCount`、余分なスレッドは stride ループ内で空回りします。

[FACT:src/gin/gin_host.cc:181-183]

```cpp
// After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
// on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
// Extra threads simply idle in the stride loop; no correctness issue.
```

これは正確性の問題ではありませんが、CPU リソースを無駄にします。調査方法は`NCCL_GIN_PROXY_NTHREADS`が実際の NIC 数より大きいかどうかを確認することです。

**落とし穴 3：DevComm 解放時の競合状態。** `ncclGinDevCommFree`まず DevComm をリンクリストから取り外し、その後 context を破棄します。

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

取り外した後、進捗スレッドはこの DevComm を認識できなくなるため、context の破棄は安全です。ただし、破棄中に in-flight のネットワーク操作があると未定義動作を引き起こす可能性があります。これは GIN を使用する際に確保すべき事項です：DevComm を解放する前に、すべての操作が完了していることを確認する必要があります。

---

# 三、対称メモリ kernel：「登録バッファ」から「統一アドレス空間」へ

## 直感的モデル

従来の NCCL のバッファは「登録制」です：各 rank が自身のバッファを登録し、通信時に handle を介してアドレスを交換します。対称メモリは「統一アドレス空間」です：すべての rank が同じ仮想アドレスを約束し、rank 0 のアドレス A と rank 1 のアドレス A はそれぞれの物理メモリを指しますが、コード内では同じアドレスでアクセスできます。

これは、皆が「3 列 5 番」と約束すれば、各人の家で同じ位置を指し、物を探すときに「君の家の 3 列 5 番はどこ？」と先に聞く必要がないのと同じです。

対称メモリがなければ、各 kernel はまず相手のアドレスを解析する必要があり、命令オーバーヘッドとレジスタプレッシャーが増加します。

## データ構造とメモリレイアウト

対称メモリ kernel の核心は kernel mask です——現在の通信ドメインでどの kernel が利用可能かをマークするビットマップです。

[FACT:src/sym_kernels.cc:17-63]

```cpp
constexpr uint32_t kernelMask_STMC =
  1  **[Design Inference & Architectural Trade-offs]**
> このビットマップ設計の利点は、ビット演算で利用可能な kernel を高速にフィルタリングできることです。例えば`kmask &= ~kernelMask_STMC`の一行で全ての STMC kernel を無効化でき、リストを走査する必要がありません。

## Step-by-Step Walkthrough：kernel mask 計算の一例

シナリオを代入します：rank 0 が AllReduce を実行、データ型は float16、メッセージサイズ 1MB、通信ドメインは 8 つの rank、全て NVLink 接続。

**第一步：操作に対応する基本 mask を取得。**

[FACT:src/sym_kernels.cc:304-306]

```cpp
uint32_t kmask = kernelMask_coll(coll);
```

`kernelMask_coll(ncclFuncAllReduce)`が返す`kernelMask_AR`、5 つの AllReduce kernel を含みます。

**第二步：STMC と LDMC の可用性を確認。**

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

`hasLsaMultimem`は`ncclSymkInitOnce`で計算され、NVLS 対称マルチキャストが利用可能で LSA チームが 2 つの rank より大きいことを要求します。float16 は LDMC をサポートするため、`hasLsaMultimem`が真なら LDMC kernel は保持されます。

**第三步：メッセージサイズ制限を確認。**

[FACT:src/sym_kernels.cc:336-342]

```cpp
size_t nBytes = alignUp(nElts * ncclTypeSize(ty), NCCL_SYM_KERNEL_CELL_SIZE);
size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks) * nBytes;
if (nBusBytes >= (size_t(2) = 32 * (size_t(2) nRanks;
kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
```

LSA チームが全ての rank をカバーする場合、GIN は不要です；そうでなければ GIN kernel のみを保持します。

## 並行制御とハードウェア相互作用

対称メモリ kernel の初期化は DevComm 作成とリソース割り当てを含みます。

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

ここでの鍵は`ncclDevrCommCreateInternal`で、これは LSA マルチキャスト、GIN inbox/outbox、シグナルなどのリソースを含む内部 DevComm を作成します。`reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL`は GIN が rail 接続モードを使用することを指定します。

[FACT:src/sym_kernels.cc:257-261]

```cpp
symk->kcomm.workStarted = comm->profiler.symWorkStarted;
symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
symk->kcomm.workPhases = comm->profiler.symWorkPhases;
```

対称メモリ kernel は独立した profiler バッファを使用し、通常の kernel の workCounter と交錯するのを避けます。

## 本番環境の落とし穴ガイド

**落とし穴 1：TMA kernel の SMEM 要件。**TMA は各 warp に約 8KB の SMEM スクラッチが必要で、16 warp なら 128KB です。

[FACT:src/sym_kernels.cc:135-142]

```cpp
bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  if (comm->maxSharedMemOptin minCompCap >= 100 && ncclParamSymTmaEnable();
}
```

GPU の SMEM 容量が不足している場合（例：MIG インスタンス）、TMA kernel は無効化されます。調査方法は`maxSharedMemOptin`が`ncclTmaShmemScratchWarpSize() * 16`。

**より小さいかどうかを確認することです。**落とし穴 2：GIN chunk size の境界。

[FACT:src/sym_kernels.cc:148-153]

```cpp
static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128  0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}
```

コピー`NCCL_SYM_RS_GIN_CHUNK_SIZE`ユーザーが設定した

**落とし穴3：対称メモリ登録タイプの不一致。** `ncclGetSymRegType`sendWin と recvWin の`NCCL_WIN_COLL_SYMMETRIC`フラグに基づいて登録タイプを判断する。

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

send と recv の登録タイプが一致しない場合、kernel は異なるコードパスを通る必要がある。これはパフォーマンスに影響するが、エラーにはならない。

---

# 四、Team 抽象とバージョン化 DevComm：進化の基盤インフラ

## 直感的モデル

Team 抽象は「グループ分け」のようなものだ：ワールドチームはクラス全体、LSA チームは隣の席、Rail チームは同じ列の席。異なる通信パターンには異なるグループ分けの視点が必要だ。

バージョン化 DevComm は「翻訳者」のようなものだ：異なるバージョンのデバイスコードは異なる「方言」を話し、DevComm 互換レイヤーが翻訳を担当し、新旧のコードが互いを理解できるようにする。

Team 抽象がなければ、各 kernel が自分で rank マッピングを計算しなければならない。バージョン化 DevComm がなければ、ABI の変更がすべてのデバイスコードの再コンパイルを引き起こす。

## データ構造とメモリレイアウト

Team はシンプルな三つ組である：`nRanks`、`rank`、`stride`。

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

ワールドチームの stride は 1 である。なぜならすべての rank が連続して配置されているからだ。

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

Rail チームの stride は`lsaSize`である。なぜなら各 rail 上の rank は LSA チームのサイズ分だけ間隔が空いているからだ。

バージョン化 DevComm の核心は`ncclDevCommCompat`構造体である。

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

この構造体はバージョン 2.31.0 の互換性ルールを定義している。`minVersion`と`maxVersion`が適用バージョン範囲を定義し、後ろの4つの関数ポインタが属性フィルタリングと構造変換ロジックを定義する。すべてが nullptr の場合、このバージョンには特別な互換性要件がないことを示す。

## Step-by-Step Walkthrough：1回の Team 変換

あるシナリオを想定しよう：rank 5 が 8 rank の通信ドメインにあり、LSA チームサイズが 4 である。rank 5 の Rail チームにおける rank を計算する。

**第一步：DevR 状態を初期化する。**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
```

`ncclDevrInitOnce`LSA チーム、CFT チームなどの派生情報を計算する。失敗した場合、空のチームを返す。

**第二步：Rail チームのパラメータを計算する。**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
ncclTeam_t ans;
ans.nRanks = comm->nRanks / comm->devrState.lsaSize;  // 8 / 4 = 2
ans.rank = comm->rank / comm->devrState.lsaSize;       // 5 / 4 = 1
ans.stride = comm->devrState.lsaSize;                  // 4
```

rank 5 の Rail チームにおける rank は 1 で、チームには 2 つの rank があり、stride は 4 である。

**第三步：ワールド rank に変換し戻す。**

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Rail rank 0 をワールド rank に変換する場合：`5 + (0 - 1) * 4 = 1`。検証：rank 1 と rank 5 は同じ rail 上にある（間隔 4）。

## 並行制御とハードウェア相互作用

Team 抽象自体はステートレスであり、並行制御は不要である。しかし`ncclDevrInitOnce`は遅延ロードであり、最初の呼び出し時にすべての派生情報を計算する。

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

コメントには「Ignoring errors since if it fails ncclDevrInitOnce will try again」とある——初期化に失敗した場合、空のチームを返し、次回の呼び出しで再試行する。

## 本番環境の落とし穴ガイド

**落とし穴1：Team 変換の stride 仮定。** `ncclTeamRankToWorld`はチーム内の rank が等差数列であると仮定している。

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

チームが等差数列でない場合（例えばカスタムの任意のグループ分け）、この関数は誤った計算をする。NCCL は現在、規則的なチームのみをサポートしている。

**落とし穴2：バージョン化 DevComm のヌルポインタ。** `ncclDevCommCompat_v23100`のすべての関数ポインタは nullptr であり、特別な互換性ロジックがないことを示す。将来のバージョンで変換が必要な場合、これらの関数を実装しなければならず、そうでなければ新旧のコードが相互運用できない。

**落とし穴3：CFT チームの階層モード。** `ncclTeamCft`は3つのモードをサポートする：FLAT、HIER_MULTIMEM、HIER_LSA。

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

無効なモードを渡した場合、空のチームを返す。CFT チームを使用する際はモードが正しいことを確認する必要がある。

---

# 設計上の考察

**なぜ NCCL は RMA、GIN、対称メモリの3つの進化パスを同時にサポートするのか？**

> **[Design Inference & Architectural Trade-offs]**
> これら3つのパスは異なるレベルの問題を解決している：

- **RMA**「通信パターンが固定」という問題を解決する——上位層がプリミティブを組み合わせて、任意の通信パターンを実現できるようにする。
- **GIN**「ネットワーク遅延が高い」という問題を解決する——GPU が直接ネットワークカードを駆動し、host proxy をバイパスする。
- **対称メモリ**「アドレス解決のオーバーヘッド」という問題を解決する——kernel が統一アドレスで直接ピアメモリにアクセスできるようにする。

これらは代替関係ではなく、補完関係にある。RMA は GIN を下位伝送として使用でき、GIN は対称メモリに依存してアドレス一貫性を提供する。三者が共同で「プログラマブル通信エンジン」の基盤インフラを構成している。

**バージョン化 DevComm の設計哲学とは何か？**

> **[Design Inference & Architectural Trade-offs]**
> バージョン化 DevComm の核心思想は「ABI 安定、API 進化」である。デバイスコード（kernel）はコンパイル後にバイナリに埋め込まれ、NCCL ライブラリのアップグレードに伴って再コンパイルできない。したがって NCCL は古いデバイスコードが新しいライブラリ上で動作することを保証しなければならない。`ncclDevCommCompat`構造体が互換レイヤーの入口である：新しいライブラリはデバイスコードのバージョンに基づいて適切な互換ルールを選択し、必要に応じて構造変換を行う。

---

# 本章のまとめ

本章では、ソースコード中の進化の痕跡から出発し、NCCL が集合通信ライブラリからプログラマブル通信エンジンへと向かう3つの力

1. **RMA**（`src/rma/rma.cc`）：Put/Signal/WaitSignal プリミティブの組み合わせにより、上位層があらゆる通信パターンを実装できるようにする。中核設計は、LSA 到達可能性に基づいてタスクを CE と Proxy の二経路に分割し並列実行することである。

2. **GIN**（`src/gin/gin_host.cc`）：GPU から直接ネットワークへ送信し、host proxy を迂回する。中核設計はマルチバックエンド管理、バージョン互換表、進捗スレッドプールである。

3. **対称メモリ kernel**（`src/sym_kernels.cc`）：統一アドレス空間により、アドレス解決のオーバーヘッドを排除する。中核設計は kernel mask ビットマップと TMA/GIN ハードウェアアクセラレーションである。

4. **Team 抽象化とバージョン化 DevComm**（`src/nccl_device/core.cc`、`src/devcomm/devcomm_v23100.cc`）：進化のための基盤を提供する。Team はグループ視点を提供し、バージョン化 DevComm は ABI 互換性を提供する。

これらの変化が上位フレームワークに与える影響は深遠である：PyTorch の ProcessGroup は RMA プリミティブを直接呼び出してカスタム通信パターンを実装できる；Megatron のエキスパート並列は GIN を利用して all-to-all の遅延を低減できる；対称メモリは kernel コードをより簡潔にする。

# 本章の考察とセルフチェック

Q1：もし`scheduleRmaTasksToPlan`における WaitSignal 分岐の LSA 到達可能性判定を削除し、すべての peer が Proxy 経路を通ると、どのような結果になるか？どのようなシナリオで性能災害が引き起こされるか？

**参考解説**：

LSA 到達可能性判定は[FACT:src/rma/rma.cc:187-204]にあり、peer を CE と Proxy の二組に分ける。もしこの判定を削除すると、すべての peer が Proxy 経路を通り、`nRmaTasksCe`は常に 0 となる。

結果は：CE 経路が全く使用されず、すべての WaitSignal が host proxy スレッドを介してネットワークをポーリングする。LSA 範囲内の peer（同一マシンの NVLink 相互接続）では、本来 GPU コピーエンジンで非同期に待機できるものが、host スレッドのポーリングになり、遅延がマイクロ秒級からミリ秒級に上昇する。

性能災害シナリオ：MoE 訓練では、各 token が複数のエキスパートの信号を待つ必要がある。もしすべての信号が Proxy を通ると、host スレッドがボトルネックとなり、GPU は多くの時間を host のポーリング待ちに費やす。8 基の GPU がすべて NVLink で接続されたマシンでは、この劣化は特に顕著である——本来すべての通信が CE を通れたものが、すべて host に集中する。

調査方法：`scheduleRmaTasksToPlan`の INFO ログを見て、もし`nRmaTasksCe`が常に 0 で`nRmaTasksProxy`が大きい場合、LSA 判定に問題があることを示す。

Q2：`ncclGinProgress`における`writePending`フラグと`devCommRwMutex`読み書きロックの連携で、もし`writePending`チェックを削除し、読み書きロックのみを残すと、どのような問題が生じるか？

**参考解説**：

`writePending`チェックは[FACT:src/gin/gin_host.cc:63-66]にあり、メインスレッドが書き込もうとするときに進捗スレッドが自発的に yield するようにする。もしこのチェックを削除すると、進捗スレッドは直接読みロックを取得しようとする。

問題は：`std::shared_timed_mutex`の読みロックは共有であり、複数の進捗スレッドが同時に保持できる。メインスレッドが書きロックを取得するには、すべての読みロックが解放されるのを待たなければならない。高負荷時には、進捗スレッドが頻繁に読みロックを取得し、メインスレッドが長時間書きロックを取得できず、`ncclGinDevCommSetup`や`ncclGinDevCommFree`がブロックされる可能性がある。

さらに深刻なのは：もしメインスレッドが`ginProgressWriteLock`で先に`writePending`をセットしてからロックを取得し、進捗スレッドが`writePending`をチェックしない場合、進捗スレッドはメインスレッドのセット後も読みロックを取得し続け、メインスレッドの待ち時間が予測不能になる。

`writePending`の役割は「ソフト通知」である：進捗スレッドに「これから書くので、少し譲ってくれ」と伝える。これは単にロックの公平性に依存するよりも効率的である。なぜなら進捗スレッドはロックでブロックするのではなく、自発的に yield できるからである。

Q3：`ncclSymkMask`において、もし`nBusBytes >= 32 * (size_t(2) << 30)`時にすべての kernel を無効化（`kmask = 0`）すると、このとき`ncclSymkAvailable`は false を返し、NCCL はどの経路にフォールバックするか？このフォールバック経路にはどのような性能影響があるか？

**参考解説**：

`kmask = 0`は[FACT:src/sym_kernels.cc:342]にあり、このとき`ncclSymkAvailable`は false を返す（[FACT:src/sym_kernels.cc:354-361]）。

フォールバック経路は：NCCL は従来の集合通信 kernel（非対称メモリ kernel）を使用する。これらの kernel は登録バッファ方式で対向メモリにアクセスし、まずアドレスを解決する必要があり、命令オーバーヘッドが大きい。

性能影響：超大メッセージ（64GB バスバイト超）では、従来 kernel のアドレス解決オーバーヘッドの割合は非常に小さい。データ転送自体が支配的だからである。しかし境界ケース（ちょうど 64GB を超える場合）では、従来 kernel は対称メモリ kernel より 10-20% 遅くなる可能性がある。

この制限の根本原因は：対称メモリ kernel は 32 ビット整数で unrolled loop chunk を追跡し、各 chunk は少なくとも 32 バイトであるため、最大アドレス可能範囲は 32 * 2^31 = 64GB となる。この範囲を超えると整数オーバーフローが発生する。

実際の運用では、単一の集合通信が 64GB を超えるシナリオは稀である（通常は勾配蓄積後の all-reduce）が、不可能ではない。このようなシナリオに遭遇した場合、分割通信や従来 kernel の使用を検討できる。

---

# 章末の橋渡し

本章では、NCCL が「固定集合操作」から「プログラマブル通信エンジン」へと進化していることを見た：RMA はプリミティブの組み合わせを提供し、GIN は GPU 直接送信を提供し、対称メモリは統一アドレス空間を提供し、Team とバージョン化 DevComm は基盤を提供する。

これらの進化は孤立したものではなく、共通の目標を指し示している：**上位フレームワークがより低い遅延と高い柔軟性でカスタム通信パターンを実装できるようにする**。PyTorchやMegatronのようなフレームワークにとって、これはNCCLの上に直接MoE all-to-all、パイプライン並列、エキスパート並列などの複雑な通信パターンを構築でき、NCCLを迂回して独自にネットワーク層を実装する必要がないことを意味する。

次の章は本書の最終章である。我々は一度のAllReduceの完全な経路をもう一度辿る——`ncclAllReduce`呼び出しから始まり、タスクのエンキュー、アルゴリズム選択、kernel起動、proxy推進、ネットワーク転送を経て、結果が返るまで。この振り返りでは、前の24章の知識点を繋ぎ合わせ、完全な認知マップを形成する。

ここまでで、我々はNCCLが固定集合操作からプログラマブル通信エンジンへと進化する三条の主線を明らかにした：RMAプリミティブの組み合わせ、GPU直送ネットワーク、対称メモリモデル、そしてそれらを支えるteam抽象とバージョン化されたDevCommである。これらのメカニズムは共により柔軟でハードウェア能力に近い通信の未来を指し示している。しかし、アーキテクチャがどのように進化しようとも、一度のAllReduceの完全な経路は常にNCCLを理解する基盤である。次の章では新しいコードを導入せず、第3章から第10章のエンドツーエンドフローを再度通しで解説する——ncclAllReduce呼び出しから、通信ドメインの確立、トポロジ探索、アルゴリズム選定、タスクエンキュー、kernel起動、デバイス側プリミティブ実行、結果の書き戻しまで。あなたは各章に散らばったメカニズムを再び完全なメンタルモデルに組み立て、そして「問題に遭遇したらどの章を調べるべきか」の索引を得るだろう。
