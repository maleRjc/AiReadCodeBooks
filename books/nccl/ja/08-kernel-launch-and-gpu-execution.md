# 第 8 章：カーネル起動とデバイス側実行：host 側呼び出しから GPU スレッドブロックの起動まで

前の章では、タスクがどのように複数の channel に分割され、カーネル起動パラメータがどのように生成され、group セマンティクスの下でバッチコミットと依存関係の順序付けがどのように行われるかを分解した。今、起動計画は準備完了だが、それはまだ host 側のデータ構造にすぎない。本章で答える核心的な問いは：`ncclKernelPlan`はどのようにして GPU 上で実際に動作する grid になるのか？我々は`ncclLaunchKernel`の呼び出しチェーンをたどり、パラメータがどのように kernel args に詰め込まれ、カーネルバリアントがどのように選択され、`cuLaunchKernelEx`がどのように呼び出され、デバイス側の`ncclKernelMain`がどのように共有メモリから作業記述を読み出し、具体的な実装へと分配するかを見ていく。

# Plan から Grid へ：起動パスの全体像

詳細に入る前に、まず全体的なメンタルモデルを確立しよう。`ncclKernelPlan`を「施工図面」と想像してほしい：これは今回起動する channel の数（ブロック数）、各ブロックのスレッド数、実行する work、使用するカーネル関数を記録している。そして`ncclLaunchKernel`は「施工隊の現場入り」の動作である——図面上の情報を CUDA ドライバが理解できる`CUlaunchConfig`に翻訳し、次に`cuLaunchKernelEx`を呼び出して grid を実際に GPU 上へ発射する。

もしこの層がなければ、host 側のすべてのスケジューリング（前章の channel 分割、バッチ編成、proxy op の順序付け）は机上の空論にすぎず、GPU 上ではどのカーネルも動作せず、通信は永遠に発生しない。これはエンドツーエンドの主干の最後の一环であり、host と device の境界線でもある。

起動パス全体は三つの段階に要約できる：

1. **パラメータ準備**（`finishPlan` + `uploadWork`）：work 構造体、バッチ記述子、kernel args を連続したメモリ領域に組織し、カーネルパラメータ内に置くか、FIFO 内に置くか、永続化バッファ内に置くかを決定する。

2. **カーネル発射**（`ncclLaunchKernel`）：grid/block 次元を計算し、起動属性（CGA cluster、mem sync domain、launch completion event）を組み立て、`cuLaunchKernelEx`。

3. **デバイス側エントリ**（`ncclKernelMain`）：各ブロックは`blockIdx.x`に基づいて自身の channelId を決定し、args または FIFO から work batch を共有メモリにロードし、次に`ncclDevFuncTable`を通じて具体的なアルゴリズム/プロトコル実装へ分配する。

以下の図は plan から grid への完全な制御フローを示し、重要な分岐判断を含んでいる：

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

この図は本章の三つの核心関数を固定している：`finishPlan`、`uploadWork`、`ncclLaunchKernel`。次にこれらを一つずつ分解していく。

# パラメータ準備：work 構造体がどのように自分の位置を見つけるか

## 直感的モデル

`finishPlan`の役割は、宅配仕分けセンターの「梱包係」に似ている。それは散在する work 構造体の山（各 collective または p2p 操作に対応する一つ）に直面し、これらの work をカーネルパラメータという「携帯バックパック」に詰め込むか、FIFO という「コンベアベルト」に載せるか、永続化バッファという「倉庫」に入れるかを決定する必要がある。

もしこの決定を誤ると——例えば work が大きすぎてカーネルパラメータに収まらないのに無理に詰め込むと——カーネル起動は直接失敗する。もし work を間違った場所に置くと、デバイス側が読み取るのはゴミデータであり、通信結果は完全に誤りとなる。

## データ構造とメモリレイアウト

まず`ncclDevKernelArgs`の構造を見てみよう。これは host と device の間の「封筒」である：

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

この構造体はわずか 5 つのフィールドしか持たないが、各フィールドが重要な情報を担っている。`channelMask`は 64 ビットマスクで、各ビットが一つの channel に対応し、デバイス側は`__popcll`を通じて`blockIdx.x`に対応する channelId を計算する。`workStorageType`はデバイス側がどこから work を読むかを決定する：`Args`は work がカーネルパラメータ内にあることを示し、`Fifo`はリングバッファ内にあることを示し、`Persistent`は永続化バッファ内にあることを示す。

`ncclDevWorkBatch`これは batch 記述子であり、デバイス側に「この channel の work がどこにあり、いくつあるか」を伝えます：

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

`offsetBitset`は 64 ビットマスクで、各ビットが 1 つの work 構造体に対応します。デバイス側は`__popc`と`fns`（find n-th set）命令を使って各 work のオフセットを特定します。`nextJump`と`nextExtends`は複数の batch を連結するために使われます——work が多すぎて 1 つの batch に収まらない場合、「拡張 batch」が作成されます。

## Step-by-Step Walkthrough

ここで具体的なシナリオを考えます：1 回の AllReduce が 4 つの channel に分割され、各 channel に 2 つの work 構造体があり、合計 8 つの work があります。

**ステップ 1：`finishPlan`がストレージタイプを決定します。**

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

ここでの重要な判断は：もし`sizeof(ncclDevKernelArgs) + batchBytes + workBytes`が`comm->workArgsBytes`（通常 4KB）に収まるなら、work を直接 kernel パラメータに入れます。そうでなければ、work は FIFO または永続化バッファに置かれ、kernel パラメータには batch 記述子だけが入ります。

> **[Design Inference & Architectural Trade-offs]**
> なぜ kernel パラメータを優先するのか？ kernel パラメータは CUDA ドライバ内で定数メモリ（constant memory）を通じて渡されるため、デバイス側の読み取りは`ld.param`命令を使い、グローバルメモリから FIFO を読むよりもはるかに高速です。小さいメッセージ（work 総量が少ない）では、これによりレイテンシを大幅に削減できます。

**ステップ 2：batch を channel ごとに順番に kernel args に入れます。**

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

ここにはいくつかの重要なポイントがあります：

1. **`fifoCursor`のセマンティクス**：`Args`タイプでは、これは`kernelArgs`の開始アドレスに対するオフセットです；`Fifo`タイプでは、これは FIFO ベースアドレスに対するオフセットです；`Persistent`タイプでは、0 から始まります。

2. **`offsetBase`の修正**：`finishPlan`内の batch の`offsetBase`は plan の work 開始位置に対する相対値です（0 から始まる）。`uploadWork`これを実際のストレージ位置に対するオフセットに変換する必要があります。`Args`タイプでは、`sizeof(ncclDevKernelArgs) + batchBytes`を加算します；`Fifo`タイプでは、`comm->workFifoProduced`。

3. **16 バイトアライメントコピー**：work 構造体はすべて 16 バイトアライメントされており（`alignas(16)`）、コピー時は 16 バイト単位で行います。`COMPILER_ASSUME_ALIGNED`はコンパイラにこのアドレスが 16 バイトアライメントであることを伝え、コンパイラにより効率的なベクトル化命令を生成させます。

4. **FIFO 待機**：`Fifo`タイプでは、`waitWorkFifoAvailable`が FIFO に十分な空きができるまでスピン待機します。この待機は`comm->abortFlag`をチェックし、abort 時のデッドロックを回避します。

## 設計上の考察と本番環境での落とし穴

> **[Design Inference & Architectural Trade-offs]**
> **なぜ 3 つのストレージタイプが必要なのか？**これは空間とレイテンシのトレードオフです：

- `Args`：最速（定数メモリ）だが容量が限られる（4KB）。小さいメッセージ、少量の work に適しています。
- `Fifo`：容量が大きい（リングバッファ）が、デバイス側の読み取りはグローバルメモリ経由になります。中程度のメッセージに適しています。
- `Persistent`：CUDA Graph キャプチャシナリオで使用されます。graph キャプチャ時には`cudaMemcpy`を実行できないため、永続化バッファを事前に確保し、work をコピーしてから kernel にそこから読ませる必要があります。

**落とし穴 1：FIFO オーバーフローによるデッドロック。**もし`waitWorkFifoAvailable`が`abortFlag`をチェックしていなければ、FIFO が満杯でコンシューマ（GPU kernel）が何らかの理由で消費を停止した場合、host は永遠にスピンし続けます。ソースコードでは[FACT:src/enqueue/enqueue.cc:1333-1349]が abort flag を明確にチェックしています：

```c
if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
  return ncclInternalError;
}
```

**落とし穴 2：`offsetBitset`のオーバーフロー。** `offsetBitset`は 64 ビットで、1 つの batch 内で最大 64 個の work をサポートします。64 個を超えると、`1ull << (offset / workSize)`がオーバーフローします。ソースコードでは`NCCL_MAX_DEV_WORK_BATCH_BYTES`によって batch のサイズ（1024 バイト）が制限されており、最小の work 構造体は`ncclDevWorkColl`（約 80 バイト）なので、最大 12 個の work となり、オーバーフローしません。

**落とし穴 3：Persistent モードでのメモリリーク。**の`uploadWork`の`Persistent`分岐では、`fifoBufHost`は`ncclOsAlignedAlloc`によって割り当てられ、`uploadWork_cleanup_fn`で解放する必要があります。もし`cudaMemcpyAsync`が失敗すると、`fail`ラベルが`cleanup`が null かどうかをチェックし、null なら`fifoBufHost`を直接解放します。このエラー回復チェーンは[FACT:src/enqueue/enqueue.cc:1483-1485]で確認できます。

# Kernel 発射：CUlaunchConfig から cuLaunchKernelEx まで

## 直感的モデル

`ncclLaunchKernel`の役割は「ロケット打ち上げコントロールコンソール」に似ています。燃料（work データ）が既に積まれた plan を受け取り、ロケットの飛行パラメータ（grid/block 次元）を計算し、各種打ち上げオプション（cluster、mem sync domain、completion event）を設定してから、打ち上げボタンを押します（`cuLaunchKernelEx`）。

この段階でエラーが発生した場合——例えば grid 次元の計算を間違えた場合——GPU 上で誤った数の block が起動され、一部の channel の作業が永遠に実行されず、通信がハングします。

## データ構造とメモリレイアウト

`CUlaunchConfig`は CUDA ドライバ API の起動設定構造体であり、NCCL はスタック上でこれを構築します：

[FACT:src/enqueue/enqueue.cc:1916-1917]

```c
CUlaunchConfig launchConfig = {0};
CUlaunchAttribute launchAttrs[6] = {};
int attrs = 0;
```

`launchAttrs`は最大 6 要素の配列で、各要素は`CUlaunchAttribute`です。NCCL はハードウェア能力とドライババージョンに応じて、条件付きで異なる属性を追加します：

- `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`：CGA cluster 次元（sm90+）
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`：cluster スケジューリングポリシー
- `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`：メモリ同期ドメイン（CUDA 12.0+）
- `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`：起動完了イベント（CUDA 12.3+）
- `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`：プログラム的ストリーム直列化（sym kernel）
- `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING`：NVLink 利用率中心スケジューリング（CUDA 13.0+）

## Step-by-Step Walkthrough

**ステップ 1：grid と block の次元を計算する。**

[FACT:src/enqueue/enqueue.cc:1889-1893]

```c
int nChannels = countOneBits(plan->channelMask);
void* sym = plan->kernelFn;
dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
```

`nChannels`は`channelMask`でセットされているビットの数、つまりこの plan が起動する block の数です。各 block は 1 つの channel を担当します。`threadPerBlock`は`scheduleCollTasksToPlan`内で`plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE)`によって計算され、すべての task の中で最大の`nWarps * 32`。

`smem`は動的共有メモリサイズです。通常の kernel の場合、それは`ncclShmemDynamicSize(comm->cudaArch)`であり、これはコンパイル時定数で、アーキテクチャに依存します（sm70+ では`ncclShmemScratchWarpSize * (NCCL_MAX_NTHREADS / WARP_SIZE)`）。sym kernel の場合、それは`plan->kernelDynSmem`です。なぜなら sym kernel の共有メモリ要件は異なる可能性があるからです。

**ステップ 2：kernel パラメータを組み立てる。**

[FACT:src/enqueue/enqueue.cc:1902-1903]

```c
void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
                 CU_LAUNCH_PARAM_END};
```

これは CUDA ドライバ API のパラメータ受け渡し方式の 1 つです：`CU_LAUNCH_PARAM_BUFFER_POINTER`はドライバに「パラメータは 1 つずつ渡すのではなく、連続したメモリブロックとして渡す」ことを伝え、`CU_LAUNCH_PARAM_BUFFER_SIZE`はドライバにこのブロックのサイズを伝えます。この方法の利点は、NCCL が`ncclDevKernelArgs`と後続の batch 配列を一度に渡すことができ、パラメータを 1 つずつパッケージする必要がないことです。

**ステップ 3：launch attributes を追加する。**

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

CGA（Cooperative Group Array）は sm90 で導入されたハードウェア機能で、複数の block を 1 つの cluster にまとめることを可能にします。cluster 内の block は同じ SM 群に同時にスケジュールされることが保証され、互いの共有メモリにアクセスできます。NCCL はこの機能を使って、NVLS など block 間同期が必要なアルゴリズムを実装しています。

`if (grid.x % clusterSize) clusterSize = 1;`という保護に注意してください：cluster 次元は grid 次元を割り切れる必要があり、そうでなければドライバがエラーを返します。もし`grid.x`が`clusterSize`で割り切れない場合、cluster を使用しない形に退化します。

**ステップ 4：launch completion event を追加する。**

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

`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`は CUDA 12.3 で導入された機能です：ドライバは kernel が実際に実行を開始したとき（host 側の呼び出しが戻ったときではなく）にイベントを記録します。これは「暗黙的順序」（implicit order）を実装するために極めて重要です——NCCL は複数の kernel が順番に実行されることを保証する必要がありますが、host 側でブロッキング待機はしたくありません。

`getImplicitOrder`のロジックは：ユーザーが`launchOrderImplicit`を設定しており、かつドライババージョンが十分に新しければ、`ncclImplicitOrderLaunch`を使用し（launch event で順序付け）；そうでなければ`ncclImplicitOrderSerial`を使用します（completion event で順序付け、つまり直列実行）。

**ステップ 5：`cuLaunchKernelEx`。**

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

`cuLaunchKernelEx`は CUDA 12.0 で導入された新しい API で、launch attributes をサポートします。古いドライバ（< 11.8）の場合、NCCL は`cuLaunchKernel`：

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

## 並行制御とハードウェア相互作用

**Launch completion event の relay メカニズム。**`ncclImplicitOrderLaunch`を使用し、かつユーザーが`launchCompletionEvent`を提供した場合、NCCL はユーザーの event を直接ドライバに渡すことができません。なぜならドライバは 1 つの launch completion event しかサポートしていないからです。NCCL の方法は：

1.`comm->sharedRes->launchEvent`をドライバに渡す。

2.`relayStream`上で`launchEvent`。

を待つ。`relayStream`3.

上でユーザーの event を記録する。

**Mem Sync Domain。** [FACT:src/enqueue/enqueue.cc:1938-1942]これによりユーザーの event は、host 側の呼び出しが戻ったときではなく、kernel が実際に実行を開始した後にトリガーされます。`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`sm90+ では、NCCL は`cudaLaunchMemSyncDomainRemote`を

## に設定します。これは Hopper アーキテクチャで導入されたメモリ同期ドメイン機構で、異なる kernel のメモリバリアを分離し、不要な同期オーバーヘッドを削減するために使用されます。

**本番環境の落とし穴ガイド**落とし穴 1：cluster 次元が割り切れないことによる起動失敗。`grid.x`もし`clusterSize`が`CUDA_ERROR_INVALID_VALUE`で割り切れない場合、ドライバは`if (grid.x % clusterSize) clusterSize = 1;`を返します。ソースコードでは`cgaClusterSize`によって保護されていますが、これは cluster 機能が黙って無効化されることも意味します。ユーザーが cluster による性能向上を期待している場合は、`nChannels`と

**の関係を確認する必要があります。** `ncclInitKernelsForDevice`初期化時に各カーネルのドライバ要件をチェックします：

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

このコードの核心は`fnsOfBitset`の計算です：`offsetBitset`内の n 番目のセットビットについて、そのビットインデックスは何か。PTX には`fns`命令がありこれを行えますが、多くの SASS 命令に展開されます。NCCL の方法は共有メモリを使用します：各 lane が自分のビットがセットされているかチェックし、セットされていればその前にいくつのセットビットがあるかを計算し、自分の lane 番号を`fnsOfBitset[nWorksBelow]`。

に書き込みます。次に実際のコピーです：

[FACT:src/device/common.h:209-241]

```c
if (tid = %d && __CUDA_ARCH__ >= %d\n" % (cudart ,arch))
  out("/*%4d*/ %s,\n" % (index, sym))
  if (cudart, arch) != (0, 0):
    out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
  index += 1
out("nullptr};\n")
```

## 設計思考と本番環境での落とし穴

**なぜ`__grid_constant__`？** [FACT:src/device/common.h:19-24]

```c
#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif
```

`__grid_constant__`このパラメータが読み取り専用であることをコンパイラに伝え、定数メモリに配置できるようにします。これにより、デバイス側での読み取り時に`ld.param`命令を使用し、グローバルメモリからの読み取りよりも高速になります。コメントには cuda-gdb を破壊すると記載されているため、sm70+ でのみ有効化されています。

**落とし穴 1：`workStorage`オーバーフロー。** `workStorage`のサイズは`ncclMaxDevWorkBatchBytes()`、sm90+ は 16KB です。もし`nWorks * workSize`がこの値を超えると、書き込みが範囲外になります。ソースコードでは`NCCL_MAX_DEV_WORK_BATCH_BYTES`によってホスト側でバッチサイズを制限していますが、デバイス側には追加のチェックがありません。ホスト側の制約が回避された場合（例えば環境変数の変更など）、共有メモリの範囲外アクセスが発生します。

**落とし穴 2：`__syncthreads()`の欠如によるデータ競合。**の後には、すべてのスレッドが完全な`loadWorkBatchToShmem`を参照できるようにするために`__syncthreads()`が必要です。ソースコードでは`workStorage`に[FACT:src/device/common.h:479]があります。この同期が削除されると、一部のスレッドが`__syncthreads(); // publish ncclShmem`の書き込みが完了する前に読み取りを開始し、ゴミデータを読み取る可能性があります。`workStorage`落とし穴 3：abort チェックのタイミング。

**は各バッチの開始時にのみ abort をチェックします。あるバッチの実行時間が長い場合、abort シグナルが有効になるまでに時間がかかる可能性があります。これは設計上のトレードオフです：より頻繁なチェックはオーバーヘッドを増やしますが、応答が速くなります。** `while (ncclShmem.aborted == 0)`Kernel バリアントの選択：generate.py が kernel リストを生成する方法

# 直感的モデル

## の役割は「自動車工場の生産ライン設計者」に似ています。それは巨大な組み合わせ空間（7 種の集合操作 × 5 種のリダクション操作 × 12 種のデータ型 × 7 種のアルゴリズム × 3 種のプロトコル）に直面し、決定する必要があります：どの組み合わせに専用の kernel を生成する必要があるか？どの組み合わせが汎用 kernel を共有できるか？

`generate.py`すべての組み合わせに kernel を生成すると、コンパイル時間とバイナリサイズが爆発します。汎用 kernel を 1 つだけ生成すると、実行時に関数ポインタ呼び出しと分岐判定により遅くなります。

の解決策は「代表 kernel」です：各等価クラスに 1 つの kernel を生成し、実行時に関数ポインタテーブルを介してディスパッチします。`generate.py`データ構造とメモリレイアウト

## は 3 つの重要なファイルを生成します：

`generate.py`：デバイス側の

1. **`device_table.cu`**、funcId を具体的なデバイス関数にマッピングします。`ncclDevFuncTable`：ホスト側の

2. **`host_table.cc`**などのテーブル。`ncclDevKernelList`、`ncclDevKernelForFunc`、`ncclDevFuncRowToId`各

3. **：具体的な kernel 実装。`<coll>_<op>_<ty>.cu`**ステップ 1：すべての関数行を列挙する。

## Step-by-Step Walkthrough

**コピー**

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

の計算式と一致する必要があります：`ncclDevFuncId()`コピー

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

`ncclDevFuncId`を介して「メイン関数 ID」にマッピングされます。このマッピングの理由は：多くの行が同じメイン関数にマッピングされる可能性があるためです（例えばすべての`ncclDevFuncRowToId`の行は`AllReduce Sum i32`のメイン関数にマッピングされます）。`AllReduce Sum u32`ステップ 2：メイン関数と kernel 関数を計算する。

**コピー**

[FACT:src/device/generate.py:211-225]

```python
func_rows = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}
kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))
```

`equivalent_primary`コピー

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

`best_kernel`のアルゴリズムは`AllGather`にマッピングされます）`AllGather RING LL`）：

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

**ステップ 3：kernel 定義を生成する。**

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

`DEFINE_ncclDevKernel`マクロ展開後は：

[FACT:src/device/common.h:507-509]

```c
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain, algo, proto>>(&args4K.args); \
  }
```

したがって、各 kernel は`__global__`関数であり、`ncclKernelMain`を呼び出し、テンプレートパラメータは`specializedFnId`と`RunWorkBatch<coll, ty, redop<ty>, algo, proto>`。

## 設計思考と本番環境での落とし穴

> **[Design Inference & Architectural Trade-offs]**
> **なぜ「代表 kernel」を使用し、各組み合わせに 1 つの kernel ではないのか？**コンパイル時間とバイナリサイズのトレードオフ。完全な組み合わせ空間は 7 × 5 × 12 × 7 × 3 ≈ 8820 個の kernel であり、各 kernel のコンパイルには数秒かかり、合計で数時間必要です。また、バイナリサイズは数百 MB に達します。代表 kernel にマッピングすることで、実際に生成される kernel 数は数十個に削減されます。

**落とし穴 1：`NCCL_EXACT_KERNEL_NAMES`によるコンパイル爆発。**この環境変数が設定されている場合、`best_kernel`は元の関数を返し、すべての組み合わせで kernel が生成されます。これは開発時には有用です（どの kernel がコンパイルされるかを正確に制御できる）が、本番環境ではコンパイル時間が長くなりすぎます。

**落とし穴 2：`required_cuda`のバージョンチェック。**一部の kernel は特定の CUDA バージョンやアーキテクチャを必要とします：

[FACT:src/device/generate.py:130-154]

ここまでで、kernel は GPU 上で起動され、デバイス側もワーク記述を取得しました。しかし真のパフォーマンスを決定するのは、デバイス内部でどのようにデータを転送するかです。次の章では src/device 配下の 3 つのプロトコルプリミティブ：LL、LL128、Simple を深掘りし、同じ AllReduce ロジックに为什么 3 セットの転送プリミティブが必要なのか、そしてそれらの同期方式、バッファレイアウト、flag セマンティクスの違いを見ていきます。
