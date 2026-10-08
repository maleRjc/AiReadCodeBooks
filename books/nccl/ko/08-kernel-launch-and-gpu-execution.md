# 제 8 장: Kernel 시작 및 디바이스 측 실행: host 측 호출에서 GPU 스레드 블록 출발까지

# 제8장: Kernel 시작 및 디바이스 측 실행: host 측 호출에서 GPU 스레드 블록 출발까지

이전 장에서 우리는 작업이 어떻게 여러 channel로 분할되고, kernel 시작 파라미터가 어떻게 생성되며, group 시맨틱 하에서 일괄 제출과 의존성 정렬 메커니즘이 어떻게 작동하는지 분석했다. 이제 시작 계획은 준비되었지만, 아직 host 측의 데이터 구조일 뿐이다. 이 장에서 답하고자 하는 핵심 질문은:`ncclKernelPlan`어떻게 GPU 상에서 실제로 실행되는 grid로 변하는가? 우리는`ncclLaunchKernel`의 호출 체인을 따라가며 파라미터가 어떻게 kernel args에 삽입되고, kernel 변형이 어떻게 선택되며,`cuLaunchKernelEx`가 어떻게 호출되고, 디바이스 측`ncclKernelMain`가 어떻게 공유 메모리에서 작업 설명을 읽어 구체적인 구현으로 분배하는지 살펴볼 것이다.

# Plan에서 Grid로: 시작 경로의 전경

세부 사항을 파고들기 전에, 먼저 전체적인 멘탈 모델을 세워보자.`ncclKernelPlan`를 "시공 도면"이라고 상상해보자: 이번에 몇 개의 channel(몇 개의 block)을 시작할지, 각 block에 몇 개의 스레드가 있는지, 어떤 work를 실행할지, 어떤 kernel 함수를 사용할지를 기록한다. 그리고`ncclLaunchKernel`는 "시공팀이 현장에 들어가는" 동작이다—도면의 정보를 CUDA 드라이버가 이해할 수 있는`CUlaunchConfig`로 번역한 다음,`cuLaunchKernelEx`를 호출하여 grid를 실제로 GPU에 발사한다.

이 계층이 없다면, host 측의 모든 스케줄링(이전 장의 channel 분할, batch 구성, proxy op 정렬)은 종이 위의 계획에 불과하며, GPU 상에서 어떤 kernel도 실행되지 않고 통신은 결코 일어나지 않을 것이다. 이것은 엔드투엔드 메인라인의 마지막 고리이자, host와 device의 경계선이다.

전체 시작 경로는 세 단계로 요약할 수 있다:

1. **파라미터 준비**（`finishPlan` + `uploadWork`): work 구조체, batch 디스크립터, kernel args를 하나의 연속 메모리에 구성하고, kernel 파라미터에 넣을지, FIFO에 넣을지, 아니면 영구 버퍼에 넣을지를 결정한다.

2. **kernel 발사**（`ncclLaunchKernel`): grid/block 차원을 계산하고, launch attributes(CGA cluster, mem sync domain, launch completion event)를 조립한 후,`cuLaunchKernelEx`。

3. **디바이스 측 진입점**（`ncclKernelMain`): 각 block은`blockIdx.x`에 따라 자신의 channelId를 결정하고, args 또는 FIFO에서 work batch를 공유 메모리에 로드한 다음,`ncclDevFuncTable`를 통해 구체적인 알고리즘/프로토콜 구현으로 분배한다.

아래 그림은 plan에서 grid까지의 완전한 제어 흐름을 보여주며, 핵심 분기 판단을 포함한다:

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

이 그림은 이 장의 세 가지 핵심 함수를 고정한다:`finishPlan`、`uploadWork`、`ncclLaunchKernel`. 다음으로 하나씩 분석해보자.

# 파라미터 준비: work 구조체가 자신의 위치를 찾는 방법

## 직관적 모델

`finishPlan`의 역할은 택배 분류 센터의 "포장 담당자"와 유사하다. 여러 개의 흩어진 work 구조체(각 collective 또는 p2p 작업에 하나씩 대응)를 마주하고, 다음과 같이 결정해야 한다: 이 work들을 kernel 파라미터라는 "휴대용 배낭"에 넣을지, FIFO라는 "컨베이어 벨트"에 넣을지, 아니면 영구 버퍼라는 "창고"에 넣을지?

만약 이 결정이 잘못되면—예를 들어 work가 너무 커서 kernel 파라미터에 들어가지 않는데 억지로 넣으면—kernel 시작이 바로 실패한다. work를 잘못된 위치에 넣으면, 디바이스 측에서 읽는 것이 쓰레기 데이터가 되어 통신 결과가 완전히 잘못된다.

## 데이터 구조와 메모리 레이아웃

먼저`ncclDevKernelArgs`의 구조를 보자. 이것은 host와 device 사이의 "봉투"이다:

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

이 구조체는 단 5개의 필드만 있지만, 각 필드는 핵심 정보를 담고 있다.`channelMask`는 64비트 마스크로, 각 비트가 하나의 channel에 대응하며, 디바이스 측은`__popcll`를 통해`blockIdx.x`에 대응하는 channelId를 계산한다.`workStorageType`는 디바이스 측이 어디서 work를 읽을지 결정한다:`Args`는 work가 kernel 파라미터에 있음을 의미하고,`Fifo`는 링 버퍼에 있음을 의미하며,`Persistent`는 영구 버퍼에 있음을 의미한다.

`ncclDevWorkBatch`batch 디스크립터로, 디바이스 측에 "이 channel의 work가 어디에 있고 몇 개인지"를 알려줍니다:

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

`offsetBitset`64비트 마스크로, 각 비트가 하나의 work 구조체에 대응합니다. 디바이스 측은`__popc`과`fns`(find n-th set) 명령어로 각 work의 오프셋을 찾습니다.`nextJump`과`nextExtends`은 여러 batch를 연결하는 데 사용됩니다——work가 너무 많아 하나의 batch에 담을 수 없을 때 "확장 batch"를 생성합니다.

## Step-by-Step Walkthrough

이제 구체적인 시나리오를 대입해 봅시다: 하나의 AllReduce가 4개의 channel로 분할되고, 각 channel에 2개의 work 구조체가 있어 총 8개의 work가 있습니다.

**첫 번째 단계:`finishPlan`이 저장 유형을 결정합니다.**

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

여기서 핵심 판단은: 만약`sizeof(ncclDevKernelArgs) + batchBytes + workBytes`이`comm->workArgsBytes`(보통 4KB)에 들어갈 수 있으면 work를 커널 파라미터에 직접 넣습니다. 그렇지 않으면 work는 FIFO 또는 영구 버퍼에 배치되고, 커널 파라미터에는 batch 디스크립터만 넣습니다.

> **[Design Inference & Architectural Trade-offs]**
> 왜 커널 파라미터에 우선 배치하는가? 커널 파라미터는 CUDA 드라이버에서 상수 메모리(constant memory)를 통해 전달되며, 디바이스 측에서 읽을 때`ld.param`명령어를 사용하므로 전역 메모리에서 FIFO를 읽는 것보다 훨씬 빠릅니다. 작은 메시지(work 총량이 적음)의 경우 이는 지연을 크게 줄일 수 있습니다.

**두 번째 단계: batch를 channel별로 번갈아 가며 kernel args에 넣습니다.**

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

여기에는 몇 가지 핵심 사항이 있습니다:

1. **`fifoCursor`의 의미**:`Args`유형의 경우`kernelArgs`시작 주소에 대한 오프셋이고,`Fifo`유형의 경우 FIFO 기본 주소에 대한 오프셋이며,`Persistent`유형의 경우 0부터 시작합니다.

2. **`offsetBase`의 보정**：`finishPlan`에서 batch의`offsetBase`은 plan의 work 시작 위치에 대한 것입니다(0부터 시작).`uploadWork`이를 실제 저장 위치에 대한 오프셋으로 변환해야 합니다.`Args`유형의 경우`sizeof(ncclDevKernelArgs) + batchBytes`을 더하고,`Fifo`유형의 경우`comm->workFifoProduced`。

3. **16바이트 정렬 복사**: work 구조체는 모두 16바이트 정렬(`alignas(16)`)이므로 복사할 때 16바이트 단위로 합니다.`COMPILER_ASSUME_ALIGNED`은 컴파일러에게 이 주소가 16바이트 정렬임을 알려주어 컴파일러가 더 효율적인 벡터화 명령어를 생성하도록 합니다.

4. **FIFO 대기**:`Fifo`유형의 경우,`waitWorkFifoAvailable`은 FIFO에 충분한 공간이 생길 때까지 스핀 대기합니다. 이 대기는`comm->abortFlag`을 확인하여 abort 시 데드락을 방지합니다.

## 설계 고찰 및 프로덕션 함정

> **[Design Inference & Architectural Trade-offs]**
> **왜 세 가지 저장 유형이 있어야 하는가?**이는 공간과 지연의 트레이드오프입니다:

- `Args`: 가장 빠르지만(상수 메모리) 용량이 제한적(4KB)입니다. 작은 메시지, 적은 work에 적합합니다.
- `Fifo`: 용량이 크지만(링 버퍼) 디바이스 측 읽기가 전역 메모리를 거쳐야 합니다. 중간 크기 메시지에 적합합니다.
- `Persistent`: CUDA Graph 캡처 시나리오에 사용됩니다. graph 캡처 시`cudaMemcpy`을 할 수 없으므로 영구 버퍼를 미리 할당하고 work를 복사한 다음 커널이 거기서 읽도록 해야 합니다.

**함정 1: FIFO 오버플로로 인한 데드락.**만약`waitWorkFifoAvailable`이`abortFlag`을 확인하지 않으면, FIFO가 가득 차고 소비자(GPU kernel)가 어떤 이유로 소비를 중단할 때 host가 영원히 스핀합니다. 소스 코드에서[FACT:src/enqueue/enqueue.cc:1333-1349]은 abort flag를 명확히 확인합니다:

```c
if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
  return ncclInternalError;
}
```

**함정 2:`offsetBitset`오버플로.** `offsetBitset`은 64비트로, 하나의 batch에 최대 64개의 work를 지원합니다. 64개를 초과하면`1ull << (offset / workSize)`이 오버플로합니다. 소스 코드에서`NCCL_MAX_DEV_WORK_BATCH_BYTES`을 통해 batch 크기를 제한하며(1024바이트), 가장 작은 work 구조체가`ncclDevWorkColl`(약 80바이트)이므로 최대 12개의 work로 오버플로하지 않습니다.

**함정 3: Persistent 모드에서의 메모리 누수.**`uploadWork`의`Persistent`분기에서,`fifoBufHost`은`ncclOsAlignedAlloc`을 통해 할당되며`uploadWork_cleanup_fn`에서 해제해야 합니다. 만약`cudaMemcpyAsync`이 실패하면,`fail`레이블이`cleanup`이 null인지 확인하고, null이면`fifoBufHost`을 직접 해제합니다. 이 오류 복구 체인은[FACT:src/enqueue/enqueue.cc:1483-1485]에서 볼 수 있습니다.

# Kernel 발사: CUlaunchConfig에서 cuLaunchKernelEx까지

## 직관적 모델

`ncclLaunchKernel`의 역할은 "로켓 발사 제어 콘솔"과 유사합니다. 이미 연료(work 데이터)가 장전된 plan을 받아 로켓의 비행 파라미터(grid/block 차원)를 계산하고, 다양한 발사 옵션(cluster, mem sync domain, completion event)을 설정한 뒤 발사 버튼을 누릅니다(`cuLaunchKernelEx`）。

이 단계에서 오류가 발생하면——예를 들어 grid 차원을 잘못 계산하면——GPU에서 잘못된 수의 block이 시작되어 일부 channel의 작업이 영원히 실행되지 않고 통신이 중단됩니다.

## 데이터 구조와 메모리 레이아웃

`CUlaunchConfig`는 CUDA 드라이버 API의 런치 구성 구조체이며, NCCL은 스택에 이를 생성합니다:

[FACT:src/enqueue/enqueue.cc:1916-1917]

```c
CUlaunchConfig launchConfig = {0};
CUlaunchAttribute launchAttrs[6] = {};
int attrs = 0;
```

`launchAttrs`는 최대 6개 요소의 배열이며, 각 요소는`CUlaunchAttribute`입니다. NCCL은 하드웨어 능력과 드라이버 버전에 따라 조건부로 다양한 속성을 추가합니다:

- `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`: CGA cluster 차원(sm90+)
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`: cluster 스케줄링 정책
- `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`: 메모리 동기화 도메인(CUDA 12.0+)
- `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`: 런치 완료 이벤트(CUDA 12.3+)
- `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`: 프로그램적 스트림 직렬화(sym kernel)
- `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING`: NVLink 활용도 중심 스케줄링(CUDA 13.0+)

## Step-by-Step Walkthrough

**첫 번째 단계: grid와 block 차원 계산.**

[FACT:src/enqueue/enqueue.cc:1889-1893]

```c
int nChannels = countOneBits(plan->channelMask);
void* sym = plan->kernelFn;
dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
```

`nChannels`는`channelMask`에서 설정된 비트의 개수, 즉 이 plan이 시작할 block의 수입니다. 각 block은 하나의 channel을 담당합니다.`threadPerBlock`는`scheduleCollTasksToPlan`에서`plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE)`를 통해 계산되며, 모든 task 중 가장 큰`nWarps * 32`。

`smem`는 동적 공유 메모리 크기입니다. 일반 kernel의 경우`ncclShmemDynamicSize(comm->cudaArch)`이며, 이는 아키텍처에 따라 달라지는 컴파일 타임 상수입니다(sm70+는`ncclShmemScratchWarpSize * (NCCL_MAX_NTHREADS / WARP_SIZE)`). sym kernel의 경우`plan->kernelDynSmem`인데, sym kernel의 공유 메모리 요구 사항이 다를 수 있기 때문입니다.

**두 번째 단계: kernel 파라미터 조립.**

[FACT:src/enqueue/enqueue.cc:1902-1903]

```c
void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
                 CU_LAUNCH_PARAM_END};
```

이는 CUDA 드라이버 API의 파라미터 전달 방식 중 하나입니다:`CU_LAUNCH_PARAM_BUFFER_POINTER`는 드라이버에게 "파라미터가 하나씩 전달되는 것이 아니라 연속된 메모리 블록이다"라고 알려주고,`CU_LAUNCH_PARAM_BUFFER_SIZE`는 드라이버에게 이 블록의 크기를 알려줍니다. 이 방식의 장점은 NCCL이`ncclDevKernelArgs`와 뒤따르는 batch 배열을 한 번에 전달할 수 있어 파라미터를 하나씩 패킹할 필요가 없다는 것입니다.

**세 번째 단계: launch attributes 추가.**

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

CGA(Cooperative Group Array)는 sm90에서 도입된 하드웨어 기능으로, 여러 block을 하나의 cluster로 구성할 수 있게 해줍니다. cluster 내의 block은 동시에 한 그룹의 SM에 스케줄링되는 것이 보장되며, 서로의 공유 메모리에 접근할 수 있습니다. NCCL은 이 기능을 사용하여 NVLS 등 block 간 동기화가 필요한 알고리즘을 구현합니다.

주의:`if (grid.x % clusterSize) clusterSize = 1;`이 보호 조건이 있습니다: cluster 차원은 grid 차원을 나눌 수 있어야 하며, 그렇지 않으면 드라이버가 오류를 반환합니다. 만약`grid.x`이`clusterSize`로 나누어지지 않으면 cluster를 사용하지 않는 것으로 퇴화합니다.

**네 번째 단계: launch completion event 추가.**

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

`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`는 CUDA 12.3에서 도입된 기능입니다: 드라이버가 kernel이 실제로 실행을 시작할 때(host 측 호출이 반환될 때가 아니라) 이벤트를 기록합니다. 이는 "암시적 순서"(implicit order)를 구현하는 데 매우 중요합니다——NCCL은 여러 kernel이 순서대로 실행되는 것을 보장해야 하지만, host 측이 블로킹 대기하는 것은 원하지 않습니다.

`getImplicitOrder`의 로직은 다음과 같습니다: 사용자가`launchOrderImplicit`를 설정했고 드라이버 버전이 충분히 새로우면`ncclImplicitOrderLaunch`를 사용하고(launch event로 정렬), 그렇지 않으면`ncclImplicitOrderSerial`를 사용합니다(completion event로 정렬, 즉 직렬 실행).

**다섯 번째 단계:`cuLaunchKernelEx`。**

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

`cuLaunchKernelEx`호출.`cuLaunchKernel`：

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

## 복사

**동시성 제어와 하드웨어 상호작용**Launch completion event의 relay 메커니즘.`ncclImplicitOrderLaunch``launchCompletionEvent`를 사용하고 사용자가

를 제공한 경우, NCCL은 사용자의 event를 드라이버에 직접 전달할 수 없습니다. 드라이버는 하나의 launch completion event만 지원하기 때문입니다. NCCL의 방식은:`comm->sharedRes->launchEvent`1.

를 드라이버에 전달합니다.`relayStream`2.`launchEvent`。

에서`relayStream`를 기다립니다.

3.

**Mem Sync Domain。** [FACT:src/enqueue/enqueue.cc:1938-1942]에 사용자의 event를 기록합니다.`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`이렇게 하면 사용자의 event는 host 측 호출이 반환될 때가 아니라 kernel이 실제로 실행을 시작한 후에 트리거됩니다.`cudaLaunchMemSyncDomainRemote`

## sm90+에서 NCCL은

**를**로 설정합니다. 이는 Hopper 아키텍처에서 도입된 메모리 동기화 도메인 메커니즘으로, 서로 다른 kernel의 메모리 배리어를 격리하여 불필요한 동기화 오버헤드를 줄입니다.`grid.x`프로덕션 함정 가이드`clusterSize`함정 1: cluster 차원이 나누어지지 않아 런치 실패.`CUDA_ERROR_INVALID_VALUE`만약`if (grid.x % clusterSize) clusterSize = 1;`이`cgaClusterSize`로 나누어지지 않으면 드라이버는`nChannels`를 반환합니다. 소스 코드에서는

**로 보호하고 있지만, 이는 cluster 기능이 조용히 비활성화되었음을 의미합니다. 사용자가 cluster로 인한 성능 향상을 기대한다면** `ncclInitKernelsForDevice`초기화 시 각 kernel의 드라이버 요구 사항을 확인합니다:

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

이 코드의 핵심은`fnsOfBitset`을 계산하는 것입니다:`offsetBitset`에서 n번째 설정된 비트의 비트 인덱스가 무엇인지. PTX에는`fns`명령어가 있어 이를 수행할 수 있지만, 많은 SASS 명령어로 전개됩니다. NCCL의 방식은 공유 메모리를 사용하는 것입니다: 각 lane이 자신의 비트가 설정되었는지 확인하고, 설정되었다면 앞에 몇 개의 설정이 있는지 계산한 후 자신의 lane 번호를`fnsOfBitset[nWorksBelow]`。

에 기록합니다. 다음은 실제 복사입니다:

[FACT:src/device/common.h:209-241]

```c
if (tid = %d && __CUDA_ARCH__ >= %d\n" % (cudart ,arch))
  out("/*%4d*/ %s,\n" % (index, sym))
  if (cudart, arch) != (0, 0):
    out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
  index += 1
out("nullptr};\n")
```

## 설계 사고와 프로덕션 함정

**왜 사용하는가`__grid_constant__`？** [FACT:src/device/common.h:19-24]

```c
#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif
```

`__grid_constant__`컴파일러에게 이 파라미터가 읽기 전용이며 상수 메모리에 배치될 수 있음을 알려줍니다. 이렇게 하면 디바이스 측에서 읽을 때`ld.param`명령어를 사용하여 전역 메모리에서 읽는 것보다 빠릅니다. 주석에 cuda-gdb를 손상시킨다고 언급되어 있어 sm70+에서만 활성화됩니다.

**함정 포인트 1:`workStorage`오버플로.** `workStorage`의 크기는`ncclMaxDevWorkBatchBytes()`, sm90+는 16KB입니다. 만약`nWorks * workSize`이 값을 초과하면 범위를 벗어나 쓰게 됩니다. 소스 코드에서는`NCCL_MAX_DEV_WORK_BATCH_BYTES`을 통해 host 측에서 batch 크기를 제한하지만, 디바이스 측에는 추가 검사가 없습니다. 만약 host 측 제약이 우회되면(예: 환경 변수 수정을 통해) 공유 메모리 범위 초과가 발생합니다.

**함정 포인트 2:`__syncthreads()`의 부재로 인한 데이터 경쟁.**이`loadWorkBatchToShmem`이후에, 모든 스레드가 완전한`__syncthreads()`을 볼 수 있도록 반드시`workStorage`이 있어야 합니다. 소스 코드에서[FACT:src/device/common.h:479]에`__syncthreads(); // publish ncclShmem`이 있습니다. 만약 이 동기화가 제거되면, 일부 스레드가`workStorage`이 아직 쓰여지기 전에 읽기를 시작하여 쓰레기 데이터를 읽을 수 있습니다.

**함정 포인트 3: abort 검사의 타이밍.** `while (ncclShmem.aborted == 0)`은 각 batch 시작 시에만 abort를 검사합니다. 만약 특정 batch의 실행 시간이 매우 길면, abort 신호가 적용되기까지 오래 기다려야 할 수 있습니다. 이는 설계상의 트레이드오프입니다: 더 빈번한 검사는 오버헤드를 증가시키지만 응답이 더 빠릅니다.

# Kernel 변형 선택: generate.py가 kernel 목록을 생성하는 방법

## 직관적 모델

`generate.py`의 역할은 "자동차 공장의 생산 라인 설계자"와 유사합니다. 그것은 거대한 조합 공간(7가지 집합 연산 × 5가지 리덕션 연산 × 12가지 데이터 타입 × 7가지 알고리즘 × 3가지 프로토콜)에 직면하여 결정해야 합니다: 어떤 조합에 전용 kernel을 생성해야 하는가? 어떤 것이 범용 kernel을 공유할 수 있는가?

만약 모든 조합에 대해 kernel을 생성하면, 컴파일 시간과 바이너리 크기가 폭발합니다. 만약 하나의 범용 kernel만 생성하면, 런타임에 함수 포인터 호출과 분기 판단으로 인해 느려집니다.`generate.py`의 해결책은 "대표적 kernel"입니다: 각 동등 클래스에 대해 하나의 kernel을 생성하고, 런타임에 함수 포인터 테이블을 통해 디스패치합니다.

## 데이터 구조와 메모리 레이아웃

`generate.py`은 세 가지 핵심 파일을 생성합니다:

1. **`device_table.cu`**: 디바이스 측의`ncclDevFuncTable`, funcId를 구체적인 디바이스 함수에 매핑합니다.

2. **`host_table.cc`**: host 측의`ncclDevKernelList`、`ncclDevKernelForFunc`、`ncclDevFuncRowToId`등의 테이블.

3. **각`<coll>_<op>_<ty>.cu`**: 구체적인 kernel 구현.

## Step-by-Step Walkthrough

**첫 번째 단계: 모든 함수 행을 열거합니다.**

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

이 열거 순서는`ncclDevFuncId()`의 계산 공식과 일치해야 합니다:

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

`ncclDevFuncId`이 계산하는 것은 "행 번호"이며, 그런 다음`ncclDevFuncRowToId`을 통해 "주 함수 ID"로 매핑됩니다. 이 매핑의 이유는: 많은 행이 동일한 주 함수에 매핑될 수 있기 때문입니다(예: 모든`AllReduce Sum i32`의 행이`AllReduce Sum u32`의 주 함수에 매핑됨).

**두 번째 단계: 주 함수와 kernel 함수를 계산합니다.**

[FACT:src/device/generate.py:211-225]

```python
func_rows = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}
kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))
```

`equivalent_primary`은 부호 있는 정수를 부호 없는 정수로 매핑합니다(덧셈/곱셈이 둘 다에 대해 동일하기 때문):

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

`best_kernel`은 여러 주 함수를 동일한 kernel에 매핑합니다(예: 모든`AllGather`의 알고리즘이`AllGather RING LL`）：

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

**세 번째 단계: kernel 정의를 생성합니다.**

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

`DEFINE_ncclDevKernel`매크로 확장 후:

[FACT:src/device/common.h:507-509]

```c
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain, algo, proto>>(&args4K.args); \
  }
```

따라서 각 kernel은`__global__`함수이며,`ncclKernelMain`을 호출하고, 템플릿 파라미터는`specializedFnId`과`RunWorkBatch<coll, ty, redop<ty>, algo, proto>`。

## 설계 사고와 프로덕션 함정

> **[Design Inference & Architectural Trade-offs]**
> **왜 "대표적 kernel"을 사용하고 각 조합마다 하나의 kernel을 사용하지 않는가?**컴파일 시간과 바이너리 크기의 트레이드오프. 완전한 조합 공간은 7 × 5 × 12 × 7 × 3 ≈ 8820개의 kernel이며, 각 kernel 컴파일에는 몇 초가 걸려 총 몇 시간이 필요합니다. 또한 바이너리 크기가 수백 MB에 달합니다. 대표적 kernel로 매핑함으로써 실제 생성되는 kernel 수가 수십 개로 줄어듭니다.

**함정 포인트 1:`NCCL_EXACT_KERNEL_NAMES`로 인한 컴파일 폭발.**만약 이 환경 변수가 설정되면,`best_kernel`은 원시 함수를 반환하여 모든 조합에 대해 kernel이 생성됩니다. 이는 개발 시 유용하지만(어떤 kernel이 컴파일되는지 정확히 제어 가능), 프로덕션 환경에서는 컴파일 시간이 너무 길어집니다.

**함정 포인트 2:`required_cuda`의 버전 검사.**일부 kernel은 특정 CUDA 버전이나 아키텍처가 필요합니다:

[FACT:src/device/generate.py:130-154]

여기까지, kernel이 GPU에서 시작되었고 디바이스 측도 작업 설명을 받았습니다. 하지만 성능을 진정으로 결정하는 것은 디바이스 내부에서 데이터를 어떻게 운반하는가입니다. 다음 장에서는 src/device 아래의 세 가지 프로토콜 원시 요소인 LL, LL128, Simple을 깊이 살펴보며, 동일한 AllReduce 로직에 왜 세 가지 운반 원시 요소가 필요한지, 그리고 동기화 방식, 버퍼 레이아웃, flag 의미에서의 차이점을 알아봅니다.
