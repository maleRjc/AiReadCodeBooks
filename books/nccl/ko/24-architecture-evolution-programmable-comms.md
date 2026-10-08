# 제 24 장: 아키텍처 진화와 미래 방향: 정적 통신에서 프로그래밍 가능한 통신으로

# 제24장: 아키텍처 진화와 미래 방향: 정적 통신에서 프로그래밍 가능한 통신으로

이전 장에서 우리는 커뮤니티가 NCCL 핵심을 중심으로 주변 생태계를 어떻게 구축하는지 보았다: Python 바인딩, Rust 바인딩, 전문가 병렬 통신, 초대역폭 프리미티브, 통신 체크포인트. 이러한 프로젝트들은 모두 NCCL의 안정적인 API를 재사용하지만, 그들의 요구는 이미 전통적인 집합 통신의 범주를 넘어섰다——전문가 병렬은 세밀한 점대점 송수신이 필요하고, 체크포인트는 통신 상태의 일시정지/재개가 필요하며, 초대역폭 프리미티브는 표준 집합 연산을 우회하여 네트워크를 직접 조작해야 한다. 이러한 요구는 동일한 문제를 가리킨다: NCCL의 고정 집합 연산 모델이 더 유연한 통신 요구에 의해 팽창되고 있다. 이 장에서는 더 이상 단일 모듈을 보지 않고, 소스 코드에 이미 나타난 진화의 흔적에서 출발하여 NCCL이 어디로 향하고 있는지 논의한다. 구체적으로, 우리는 세 가지 얽힌 진화의 힘을 분석할 것이다: 통신 프리미티브가 고정 집합에서 프로그래밍 가능으로——src/rma/rma.cc의 RMA 작업 스케줄링은 상위 계층이 AllReduce만 호출하는 것이 아니라 Put/Signal/WaitSignal 프리미티브를 조합할 수 있게 한다; 네트워크 발신이 host proxy에서 GPU 직접 발송으로——src/gin/gin_host.cc의 GIN 백엔드 관리는 GPU kernel이 직접 네트워크 카드를 구동하게 한다; 메모리 모델이 등록 버퍼에서 대칭 메모리로——src/sym_kernels.cc의 대칭 메모리 kernel 선택은 모든 rank가 동일한 가상 주소 세트로 서로의 버퍼에 접근하게 한다. 이 세 가지 힘은 고립되어 있지 않으며, 동일한 인프라를 공유한다: src/nccl_device/core.cc의 team 추상화와 src/devcomm/devcomm_v23100.cc의 버전화된 DevComm. 이들이 어떻게 맞물리는지 이해하면, NCCL이 「집합 통신 라이브러리」에서 「프로그래밍 가능한 통신 엔진」으로 진화하는 논리를 이해하게 된다.

# 一、프로그래밍 가능한 통신 프리미티브: RMA가 「고정 레시피」를 「뷔페」로 바꾸는 방법

## 직관적 모델

전통적인 NCCL의 집합 통신은 고정 세트 메뉴와 같습니다: AllReduce를 주문하면 주방에서 AllReduce 절차대로 다 만들어 줍니다. 하지만 전문가 병렬(MoE) 시나리오에서는 각 토큰을 서로 다른 전문가에게 보내야 하는데, 전송 패턴을 컴파일 시점에는 전혀 알 수 없습니다—이것은 뷔페와 같아서 무엇을, 얼마나, 언제 가져갈지 스스로 결정해야 합니다.

RMA는 NCCL이 상위 계층에 제공하는 「뷔페 테이블」입니다: Put(데이터를 상대방 메모리에 쓰기), Signal(상대방에게 알림), WaitSignal(상대방 신호 대기). 상위 프레임워크는 이 세 가지 원시 연산을 자유롭게 조합하여 임의의 통신 패턴을 구현할 수 있습니다.

RMA가 없다면 MoE의 all-to-all은 여러 번의 소규모 집합 연산으로만 시뮬레이션할 수 있으며, 매번 전체 kernel 시작과 동기화 과정을 거쳐야 하므로 지연이 허용할 수 없을 정도로 높아집니다.

## 데이터 구조와 메모리 레이아웃

RMA의 핵심 데이터 구조는`ncclTaskRma`(작업 설명)과`ncclRmaArgs`(계획 매개변수)입니다. 먼저`ncclRmaArgs`의 필드를 살펴보겠습니다. 이것은`scheduleRmaTasksToPlan`에서 초기화됩니다.

[FACT:src/rma/rma.cc:166-171]

```cpp
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
plan->rmaArgs->nRmaTasks = 0;
plan->rmaArgs->nRmaTasksProxy = 0;
plan->rmaArgs->nRmaTasksCe = 0;
```

여기서 핵심 필드는`nRmaTasksProxy`과`nRmaTasksCe`입니다. 이들은 RMA 작업을 두 가지 실행 경로로 나눕니다:

- **CE 경로**(Copy Engine, 복사 엔진): 대상 rank가 LSA(Local Symmetric Access, 로컬 대칭 접근) 범위 내에 있으면 GPU의 복사 엔진으로 직접 완료할 수 있으며 네트워크가 필요하지 않습니다.
- **Proxy 경로**: 대상 rank가 LSA 범위 내에 없으면 반드시 host proxy 스레드가 네트워크를 구동해야 합니다.

> **[Design Inference & Architectural Trade-offs]**
> 이러한 이분법적 설계 동기는 매우 직접적입니다: LSA 범위 내 통신은 NVLink 또는 PCIe를 통해 대역폭이 높고 지연이 낮아 CE 비동기 복사가 가장 효율적이며, 크로스 머신 통신은 반드시 네트워크 카드를 거쳐야 하므로 proxy 스레드만이 구동할 수 있습니다. 두 유형의 작업을 분리하여 스케줄링해야 CE와 proxy가 직렬 대기하지 않고 병렬로 실행될 수 있습니다.

`ncclTaskRma`자체에는`peers`、`nsignals`、`signalIdxs`세 개의 배열 포인터가 포함되어 있으며, 각각 상대방 rank, 신호 수, 신호 인덱스를 기록합니다. WaitSignal 작업의 경우 하나의 작업이 여러 peer를 기다릴 수 있고, Put/Signal 작업의 경우 하나의 작업이 하나의 peer만 대상으로 합니다.

## 단계별 살펴보기: WaitSignal 한 번의 스케줄링

구체적인 시나리오를 대입해 보겠습니다: rank 0이`ncclWaitSignal`을 호출하여 rank 1과 rank 3의 신호를 기다립니다. rank 1은 LSA 범위 내에 있고, rank 3은 그렇지 않다고 가정합니다.

**첫 번째 단계: 첫 번째 비어 있지 않은 컨텍스트 큐를 찾습니다.**

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

RMA 작업은 context별로 큐를 나누며, 각 context는 독립적인 RMA 채널입니다. 여기서 작업이 있는 첫 번째 context를 찾아 그 큐를 가져옵니다.

**두 번째 단계: 첫 번째 작업을 꺼내 유형을 판단합니다.**

[FACT:src/rma/rma.cc:163-168]

```cpp
struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
```

`firstTask->func`은`ncclFuncWaitSignal`이므로 WaitSignal 분기로 들어갑니다.

**세 번째 단계: LSA 도달 가능성에 따라 peer를 분할합니다.**

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

`isLsaAccessible`이`comm->devrState.lsaRankList`을 순회하며 peer가 LSA 팀 내에 있는지 판단합니다. rank 1은 LSA 내에 있으므로 CE 목록에 들어가고, rank 3은 그렇지 않으므로 Proxy 목록에 들어갑니다.

**네 번째 단계: CE와 Proxy 각각에 대해 새 작업을 하나씩 생성합니다.**

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

원래의 WaitSignal 작업 하나가 둘로 나뉩니다: CE 작업은 rank 1을 기다리고, Proxy 작업은 rank 3을 기다립니다. 두 작업은 병렬로 실행될 수 있습니다—CE 경로는 GPU에서 기다리고, Proxy 경로는 host 스레드에서 기다립니다.

**다섯 번째 단계: 원래 작업을 해제합니다.**

[FACT:src/rma/rma.cc:249-251]

```cpp
planner->nTasksRma -= 1;
ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
```

원래 작업은 이미 두 개의 새 작업으로 나뉘었으므로 메모리 풀에 반환하여 해제합니다.

## 동시성 제어와 하드웨어 상호작용

RMA의 병렬 실행은`ncclRmaWaitSignal`에서 나타납니다.

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

이 코드는 CUDA event로 스트림 간 동기화를 수행합니다: 먼저 입력 스트림에 event를 기록하고, CE 스트림이 이 event를 기다리게 한 다음, 두 스트림에서 각각 proxy와 CE 작업을 시작하고, 마지막으로 입력 스트림이 CE 스트림의 event를 기다리게 합니다. 이렇게 두 경로가 병렬로 진행되지만 외부적으로는 하나의 동기 작업으로 나타납니다.

> **[Design Inference & Architectural Trade-offs]**
> 여기서의 설계 트레이드오프는: 병렬 실행이 지연을 줄일 수 있지만 추가적인 event 기록과 스트림 동기화 오버헤드를 도입한다는 것입니다. 작은 메시지의 경우 이 오버헤드가 병렬 이득을 초과할 수 있고, 큰 메시지의 경우 병렬 이득이 두드러집니다. NCCL은 여기서 적응형 판단을 하지 않고 일괄적으로 병렬 경로를 사용합니다—RMA의 전형적인 시나리오가 바로 큰 메시지의 세밀한 통신이기 때문입니다.

## 프로덕션 함정 회피 가이드

**함정 1: LSA 도달 가능성 판단 오류로 인해 작업이 잘못된 경로로 갑니다.** `isLsaAccessible`이`lsaRankList`을 순회하는데, 만약`lsaSize`이 0이면(예: 단일 rank 통신 도메인), 모든 peer가 도달 불가능으로 판정되어 전부 Proxy 경로로 갑니다. 이는 소규모 테스트에서는 드러나지 않지만 대규모 배포에서는 성능이 급락할 수 있습니다.排查 방법은`scheduleRmaTasksToPlan`의 INFO 로그에서`nRmaTasksProxy`과`nRmaTasksCe`의 비율을 보는 것입니다.

**함정 2: WaitSignal 작업 분할 후 peer 배열의 수명 주기.**CE 경로의`peersCe`은`ncclMemoryStackAlloc`으로 할당되며 수명 주기는`comm->memScoped`을 따릅니다; Proxy 경로의`peersProxy`은`ncclCalloc`으로 할당되며 작업 실행 완료 후 수동으로`free`. Proxy 작업 생성에 실패하면,`fail`분기가 이 배열들을 해제합니다.

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

**함정 3: Put/Signal 작업의 크로스 context 배치.**Put/Signal 분기에서 NCCL은 모든 context의 put/signal 작업을 동일한 plan으로 가져오지만, WaitSignal을 만나면 중단합니다.

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

이 설계의 의도는 하나의 kernel 실행으로 모든 context의 put/signal을 처리하여 실행 오버헤드를 줄이는 것입니다. 그러나 각 context의 큐는 첫 번째 WaitSignal까지만 소비하여 per-context FIFO 순서를 보장합니다. 상위 계층에서 동일한 context 내에서 put과 waitSignal을 번갈아 호출하면 배치 효과가 크게 떨어집니다 — 이는 RMA 사용 시 주의해야 할 패턴입니다.

---

# 2. GPU 직접 네트워크 전송: GIN이 kernel이 host proxy를 우회하는 방법

## 직관적 모델

전통적인 NCCL 네트워크 통신은 편지 보내기와 같습니다: GPU kernel이 데이터를 버퍼에 넣으면, host proxy 스레드가 데이터를 NIC에 전달하고, NIC가 전송합니다. GIN은 GPU kernel이 직접 상대방의 우편함에 편지를 넣는 것입니다 — kernel이 NIC의 전송 큐에 직접 쓰고, NIC가 GPU 메모리를 직접 읽습니다.

GIN이 없다면 매번 네트워크 통신이 host 메모리를 경유해야 하므로 지연이 최소 한 번의 PCIe 왕복만큼 추가됩니다. MoE와 같은 세밀한 통신에서는 이 지연이 치명적입니다.

## 데이터 구조와 메모리 레이아웃

GIN의 핵심 상태는`ncclGinState`이며, 여러 백엔드(backend)와 여러 DevComm을 관리합니다. 먼저 백엔드 버전 호환성 테이블을 살펴보겠습니다.

[FACT:src/gin/gin_host.cc:27-33]

```cpp
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5), NCCL_VERSION(2, 32, 0)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
constexpr int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0), NCCL_VERSION(2, 32, 0)};
```

이 배열들의 인덱스는 백엔드 버전 번호이고, 값은 호환되는 최소 NCCL 버전입니다. 예를 들어`proxyBackendMinVersions[3]`은 백엔드 버전 3에 대응하며 NCCL 2.32.0 이상이 필요합니다. 이 설계를 통해 NCCL은 컴파일 타임에 바인딩하는 대신 런타임에 디바이스 코드 버전에 따라 적절한 백엔드 버전을 선택할 수 있습니다.

> **[Design Inference & Architectural Trade-offs]**
> 이러한 버전 호환성 테이블의 설계 동기는: GIN 백엔드(NIC 드라이버, 펌웨어)와 NCCL 라이브러리의 버전 진화 속도가 다르다는 것입니다. 버전 요구사항을 하드코딩하면 어느 한쪽이 업그레이드될 때마다 호환성이 깨집니다. 배열로 버전 매핑을 하면 런타임에 동적으로 선택하여 구 버전 백엔드와의 하위 호환성을 유지할 수 있습니다.

`ncclGinStateDevComm`은 각 DevComm의 GIN 상태이며,`contextCount`、`backendIndex`、`ginCtx[]`、`devHandles[]`등의 필드를 포함합니다. 이는 연결 리스트로 연결되어`ginState->devComms`에 매달립니다.

## 단계별 워크스루: GIN 연결 설정 과정

시나리오를 가정합니다: rank 0이 통신 도메인을 초기화하고 GIN 연결을 설정해야 합니다.

**1단계: GIN 활성화 및 지원 여부 확인.**

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

`ncclParamGinEnable()`은 환경 변수`NCCL_GIN_ENABLE`를 읽으며, 기본값은 1입니다. 사용자가 명시적으로 비활성화하면 바로 오류를 반환합니다.

**2단계: 대칭 메모리 지원 확인.**

[FACT:src/gin/gin_host.cc:111-114]

```cpp
if (!comm->symmetricSupport) {
  WARN("Communicator does not support symmetric memory!");
  return ncclInternalError;
}
```

GIN은 대칭 메모리에 의존합니다 — GPU kernel이 상대방 버퍼의 가상 주소를 알아야 하므로, 대칭 메모리만이 주소 일관성을 보장할 수 있습니다.

**3단계: 로컬 GIN 디바이스 목록 가져오기.**

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

`ncclTopoGetLocalGinDevs`은 토폴로지 그래프에서 GIN을 지원하는 모든 NIC를 찾습니다.`NCCL_GIN_MAX_CONNECTIONS`를 초과하면 앞의 몇 개만 가져오고 경고를 출력합니다.

**4단계: GIN 팀 계산.**

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

각 백엔드는 먼저`devices`를 호출하여 디바이스 수를 가져온 다음, 각 연결에 대해 listen→getProperties→allGather→connect→closeListen 과정을 수행합니다.`bootstrapAllGather`은 모든 rank 간에 handle을 교환하여 각 rank가 상대방의 연결 정보를 알 수 있게 합니다.

## 동시성 제어와 하드웨어 상호작용

GIN의 진행 스레드는 핵심 동시성 메커니즘입니다.

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

여기에는 몇 가지 핵심 설계가 있습니다:

1. **CPU 친화성**：`ncclOsSetAffinity`은 진행 스레드를 지정된 CPU 코어에 바인딩하여 스레드 마이그레이션으로 인한 캐시 무효화를 방지합니다.

2. **쓰기 잠금 백오프**：`writePending`은 원자적 플래그로, 메인 스레드가`devComms`연결 리스트를 수정하려 할 때 먼저 설정하면 진행 스레드가 이를 보고 자발적으로 yield하여 잠금 경쟁을 방지합니다.

3. **읽기-쓰기 잠금**：`devCommRwMutex`은`shared_timed_mutex`이며, 진행 스레드는 읽기 잠금을 보유하고 연결 리스트를 순회하며, 메인 스레드는 쓰기 잠금을 보유하고 연결 리스트를 수정합니다.

4. **스레드 분담**: 스레드 t는 연결 t, t+proxyNthreads, t+2*proxyNthreads, ...를 담당하며, stride 루프를 통해 부하 분산을 구현합니다.

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

이 쓰기 잠금 구현은 작성자가 하나(메인 스레드)만 있다고 가정하므로 추가 뮤텍스가 필요하지 않습니다.`writePending`은 먼저 플래그를 설정한 후 잠금을 획득하여, 진행 스레드가 잠금을 획득하기 전에 쓰기 의도를 볼 수 있고 자발적으로 물러날 수 있게 합니다.

## 프로덕션 함정 회피 가이드

**함정 1: GIN 연결 수 불일치로 인한 AllGather 교착.**각 rank의`ginCommCount`은 다를 수 있으며(로컬 NIC 수에 따라), NCCL은`bootstrapAllGather`을 통해 모든 rank의 최솟값을 취합니다.

[FACT:src/gin/gin_host.cc:176-180]

```cpp
ginCommCountHandles[comm->rank] = backend->ginCommCount;
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
for (int r = 0; r nRanks; r++) {
  backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
}
```

특정 rank의 NIC 수가 다른 rank보다 적으면 모든 rank가 최소값으로 떨어진다. 이는 연결 대칭성을 보장하지만 NIC 자원을 낭비한다.

**함정 2: proxyNthreads가 ginCommCount를 초과하여 스레드가 공회전함.**사용자가 설정한 경우`NCCL_GIN_PROXY_NTHREADS`보다 큰`ginCommCount`, 초과 스레드는 stride 루프에서 공회전한다.

[FACT:src/gin/gin_host.cc:181-183]

```cpp
// After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
// on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
// Extra threads simply idle in the stride loop; no correctness issue.
```

이는 정확성 문제는 아니지만 CPU 자원을 낭비한다.排查 방법은`NCCL_GIN_PROXY_NTHREADS`이 실제 NIC 수보다 큰지 확인하는 것이다.

**함정 3: DevComm 해제 시의 경쟁 상태.** `ncclGinDevCommFree`먼저 연결 리스트에서 DevComm을 제거한 후 context를 파괴한다.

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

제거 후 진행 스레드는 이 DevComm을 더 이상 볼 수 없으므로 context 파괴는 안전하다. 그러나 파괴 과정에서 in-flight 네트워크 작업이 있으면 정의되지 않은 동작이 발생할 수 있다 — 이는 GIN 사용 시 보장해야 할 사항이다: DevComm 해제 전에 모든 작업이 완료되었는지 확인해야 한다.

---

# 3. 대칭 메모리 kernel: 「등록 버퍼」에서 「통합 주소 공간」으로

## 직관적 모델

전통적인 NCCL의 버퍼는 「등록제」이다: 각 rank가 자신의 버퍼를 등록하고, 통신 시 handle을 통해 주소를 교환한다. 대칭 메모리는 「통합 주소 공간」이다: 모든 rank가 동일한 가상 주소 집합을 약속하고, rank 0의 주소 A와 rank 1의 주소 A는 각자의 물리 메모리를 가리키지만 코드에서는 동일한 주소로 접근할 수 있다.

이는 마치 모두가 「3열 5번 좌석」이 각자 집에서 같은 위치를 가리키기로 약속하여, 물건을 찾을 때 「너희 집 3열 5번 좌석이 어디야」라고 먼저 묻지 않아도 되는 것과 같다.

대칭 메모리가 없으면 각 kernel이 먼저 상대 주소를解析해야 하므로 명령 오버헤드와 레지스터 압력이 증가한다.

## 데이터 구조와 메모리 레이아웃

대칭 메모리 kernel의 핵심은 kernel mask — 현재 통신 도메인에서 어떤 kernel을 사용할 수 있는지 표시하는 비트맵이다.

[FACT:src/sym_kernels.cc:17-63]

```cpp
constexpr uint32_t kernelMask_STMC =
  1  **[Design Inference & Architectural Trade-offs]**
> 이 비트맵 설계의 장점은 비트 연산으로 사용 가능한 kernel을 빠르게 필터링할 수 있다는 것이다. 예를 들어`kmask &= ~kernelMask_STMC`한 줄로 모든 STMC kernel을 비활성화할 수 있으며, 리스트를 순회할 필요가 없다.

## Step-by-Step Walkthrough: kernel mask 계산 한 번

시나리오를 대입해보자: rank 0이 AllReduce를 실행하고, 데이터 타입은 float16, 메시지 크기는 1MB, 통신 도메인은 8개 rank, 모두 NVLink로 연결됨.

**첫 번째 단계: 연산에 해당하는 기본 mask를 가져온다.**

[FACT:src/sym_kernels.cc:304-306]

```cpp
uint32_t kmask = kernelMask_coll(coll);
```

`kernelMask_coll(ncclFuncAllReduce)`가 반환된다`kernelMask_AR`, 5개의 AllReduce kernel을 포함한다.

**두 번째 단계: STMC와 LDMC 사용 가능성을 확인한다.**

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

`hasLsaMultimem`는`ncclSymkInitOnce`에서 계산되며, NVLS 대칭 멀티캐스트가 사용 가능하고 LSA 팀이 2개 rank보다 커야 한다. float16은 LDMC를 지원하므로`hasLsaMultimem`가 참이면 LDMC kernel이 유지된다.

**세 번째 단계: 메시지 크기 제한을 확인한다.**

[FACT:src/sym_kernels.cc:336-342]

```cpp
size_t nBytes = alignUp(nElts * ncclTypeSize(ty), NCCL_SYM_KERNEL_CELL_SIZE);
size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks) * nBytes;
if (nBusBytes >= (size_t(2) = 32 * (size_t(2) nRanks;
kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
```

LSA 팀이 모든 rank를 커버하면 GIN이 필요 없다; 그렇지 않으면 GIN kernel만 유지된다.

## 동시성 제어와 하드웨어 상호작용

대칭 메모리 kernel의 초기화는 DevComm 생성과 자원 할당을 포함한다.

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

여기서 핵심은`ncclDevrCommCreateInternal`이며, 이는 LSA 멀티캐스트, GIN inbox/outbox, 신호 등의 자원을 포함하는 내부 DevComm을 생성한다.`reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL`는 GIN이 rail 연결 모드를 사용하도록 지정한다.

[FACT:src/sym_kernels.cc:257-261]

```cpp
symk->kcomm.workStarted = comm->profiler.symWorkStarted;
symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
symk->kcomm.workPhases = comm->profiler.symWorkPhases;
```

대칭 메모리 kernel은 독립적인 profiler 버퍼를 사용하여 일반 kernel의 workCounter와 교차하지 않도록 한다.

## 프로덕션 함정 회피 가이드

**함정 1: TMA kernel의 SMEM 요구사항.**TMA는 warp당 약 8KB의 SMEM scratch가 필요하며, 16개 warp면 128KB이다.

[FACT:src/sym_kernels.cc:135-142]

```cpp
bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  if (comm->maxSharedMemOptin minCompCap >= 100 && ncclParamSymTmaEnable();
}
```

GPU의 SMEM 용량이 부족하면(예: MIG 인스턴스), TMA kernel이 비활성화된다.排查 방법은`maxSharedMemOptin`이`ncclTmaShmemScratchWarpSize() * 16`。

**보다 작은지 확인하는 것이다.**함정 2: GIN chunk size의 경계.

[FACT:src/sym_kernels.cc:148-153]

```cpp
static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128  0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}
```

복사`NCCL_SYM_RS_GIN_CHUNK_SIZE`사용자가 설정한

**함정 3: 대칭 메모리 등록 유형 불일치.** `ncclGetSymRegType`sendWin과 recvWin의`NCCL_WIN_COLL_SYMMETRIC`플래그에 따라 등록 유형을 판단합니다.

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

send와 recv의 등록 유형이 일치하지 않으면 kernel은 다른 코드 경로를 거쳐야 합니다. 이는 성능에 영향을 주지만 오류를 발생시키지는 않습니다.

---

# 4. Team 추상화와 버전화된 DevComm: 진화의 기반 인프라

## 직관적 모델

Team 추상화는 '그룹화'와 같습니다: 월드 팀은 전체 학급, LSA 팀은 짝꿍, Rail 팀은 같은 열의 좌석입니다. 서로 다른 통신 모드에는 서로 다른 그룹화 관점이 필요합니다.

버전화된 DevComm은 '번역가'와 같습니다: 서로 다른 버전의 디바이스 코드는 서로 다른 '방언'을 사용하며, DevComm 호환 레이어가 번역을 담당하여 신구 코드가 서로를 이해할 수 있게 합니다.

Team 추상화가 없다면 모든 kernel이 자체적으로 rank 매핑을 계산해야 합니다; 버전화된 DevComm이 없다면 ABI 변경 시 모든 디바이스 코드를 재컴파일해야 합니다.

## 데이터 구조와 메모리 레이아웃

Team은 간단한 삼중항입니다:`nRanks`、`rank`、`stride`。

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

월드 팀의 stride는 1입니다. 모든 rank가 연속적으로 배열되기 때문입니다.

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

Rail 팀의 stride는`lsaSize`입니다. 각 rail의 rank가 LSA 팀 크기만큼 떨어져 있기 때문입니다.

버전화된 DevComm의 핵심은`ncclDevCommCompat`구조입니다.

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

이 구조는 버전 2.31.0의 호환성 규칙을 정의합니다.`minVersion`와`maxVersion`는 적용 가능한 버전 범위를 정의하고, 뒤의 네 함수 포인터는 속성 필터링과 구조 변환 로직을 정의합니다. 모두 nullptr이면 이 버전에 특별한 호환 요구사항이 없음을 의미합니다.

## 단계별 살펴보기: 한 번의 Team 변환

시나리오를 가정합니다: rank 5가 8개 rank 통신 도메인에 있고, LSA 팀 크기는 4입니다. rank 5의 Rail 팀 내 rank를 계산해야 합니다.

**1단계: DevR 상태 초기화.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
```

`ncclDevrInitOnce`LSA 팀, CFT 팀 등의 파생 정보를 계산합니다. 실패하면 빈 팀을 반환합니다.

**2단계: Rail 팀 매개변수 계산.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
ncclTeam_t ans;
ans.nRanks = comm->nRanks / comm->devrState.lsaSize;  // 8 / 4 = 2
ans.rank = comm->rank / comm->devrState.lsaSize;       // 5 / 4 = 1
ans.stride = comm->devrState.lsaSize;                  // 4
```

rank 5의 Rail 팀 내 rank는 1이고, 팀에는 2개의 rank가 있으며, stride는 4입니다.

**3단계: 월드 rank로 변환.**

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Rail rank 0을 월드 rank로 변환하려면:`5 + (0 - 1) * 4 = 1`. 검증: rank 1과 rank 5는 같은 rail에 있습니다 (간격 4).

## 동시성 제어와 하드웨어 상호작용

Team 추상화 자체는 무상태이므로 동시성 제어가 필요하지 않습니다. 하지만`ncclDevrInitOnce`는 지연 로딩되며, 첫 호출 시 모든 파생 정보를 계산합니다.

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

주석에 "Ignoring errors since if it fails ncclDevrInitOnce will try again"이라고 쓰여 있습니다 — 초기화가 실패하면 빈 팀을 반환하고, 다음 호출 시 재시도합니다.

## 프로덕션 함정 회피 가이드

**함정 1: Team 변환의 stride 가정.** `ncclTeamRankToWorld`팀 내 rank가 등차수열이라고 가정합니다.

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

팀이 등차수열이 아니면 (예: 사용자 정의 임의 그룹화), 이 함수는 잘못 계산합니다. NCCL은 현재 규칙적 팀만 지원합니다.

**함정 2: 버전화된 DevComm의 널 포인터.** `ncclDevCommCompat_v23100`의 모든 함수 포인터가 nullptr이면 특별한 호환 로직이 없음을 의미합니다. 향후 버전에서 변환이 필요하면 이 함수들을 반드시 구현해야 하며, 그렇지 않으면 신구 코드가 상호 운용될 수 없습니다.

**함정 3: CFT 팀의 계층 모드.** `ncclTeamCft`는 세 가지 모드를 지원합니다: FLAT, HIER_MULTIMEM, HIER_LSA.

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

유효하지 않은 모드를 전달하면 빈 팀을 반환합니다. CFT 팀을 사용할 때는 모드가 올바른지 확인해야 합니다.

---

# 설계 고찰

**NCCL이 왜 RMA, GIN, 대칭 메모리 세 가지 진화 경로를 동시에 지원하는가?**

> **[Design Inference & Architectural Trade-offs]**
> 이 세 경로는 서로 다른 계층의 문제를 해결합니다:

- **RMA**'통신 모드 고정' 문제를 해결합니다 — 상위 계층이 프리미티브를 조합하여 임의의 통신 모드를 구현할 수 있게 합니다.
- **GIN**'네트워크 지연 높음' 문제를 해결합니다 — GPU가 직접 네트워크 카드를 구동하여 host proxy를 우회합니다.
- **대칭 메모리**'주소 해석 오버헤드' 문제를 해결합니다 — kernel이 통합 주소로 직접 상대방 메모리에 접근할 수 있게 합니다.

이들은 대체 관계가 아니라 상호 보완 관계입니다. RMA는 GIN을 하위 전송으로 사용할 수 있고, GIN은 대칭 메모리에 의존하여 주소 일관성을 제공합니다. 세 가지가 함께 '프로그래밍 가능한 통신 엔진'의 기반 인프라를 구성합니다.

**버전화된 DevComm의 설계 철학은 무엇인가?**

> **[Design Inference & Architectural Trade-offs]**
> 버전화된 DevComm의 핵심 사상은 'ABI 안정, API 진화'입니다. 디바이스 코드(kernel)는 컴파일 후 바이너리에 내장되어 NCCL 라이브러리 업그레이드에 따라 재컴파일될 수 없습니다. 따라서 NCCL은 구 디바이스 코드가 신 라이브러리에서 실행될 수 있음을 보장해야 합니다.`ncclDevCommCompat`구조가 바로 호환 레이어의 진입점입니다: 신 라이브러리가 디바이스 코드 버전에 따라 적절한 호환 규칙을 선택하고, 필요 시 구조 변환을 수행합니다.

---

# 이 장 요약

이 장에서는 소스 코드의 진화 흔적에서 출발하여, NCCL이 집합 통신 라이브러리에서 프로그래밍 가능한 통신 엔진으로 나아가는 세 가지 동력을 분석했습니다:

1. **RMA**（`src/rma/rma.cc`): Put/Signal/WaitSignal 프리미티브 조합을 통해 상위 계층이 임의의 통신 모드를 구현할 수 있게 한다. 핵심 설계는 LSA 도달 가능성에 따라 작업을 CE와 Proxy 두 경로로 나누어 병렬 실행하는 것이다.

2. **GIN**（`src/gin/gin_host.cc`): GPU 직접 네트워크 전송을 통해 host proxy를 우회한다. 핵심 설계는 다중 백엔드 관리, 버전 호환성 테이블, 진행 스레드 풀이다.

3. **대칭 메모리 kernel**（`src/sym_kernels.cc`): 통합 주소 공간을 통해 주소 해석 오버헤드를 제거한다. 핵심 설계는 kernel mask 비트맵과 TMA/GIN 하드웨어 가속이다.

4. **Team 추상화와 버전화된 DevComm**（`src/nccl_device/core.cc`、`src/devcomm/devcomm_v23100.cc`): 진화를 위한 인프라를 제공한다. Team은 그룹 관점을 제공하고, 버전화된 DevComm은 ABI 호환성을 제공한다.

이러한 변화가 상위 프레임워크에 미치는 영향은 심대하다: PyTorch의 ProcessGroup은 RMA 프리미티브를 직접 호출하여 사용자 정의 통신 모드를 구현할 수 있고; Megatron의 전문가 병렬화는 GIN을 활용하여 all-to-all 지연을 줄일 수 있으며; 대칭 메모리는 kernel 코드를 더 간결하게 만든다.

# 이 장의 고찰과 자가 점검

Q1: 만약`scheduleRmaTasksToPlan`에서 WaitSignal 분기의 LSA 도달 가능성 판단을 제거하고, 모든 peer가 Proxy 경로를 타게 하면 어떤 결과가 발생하는가? 어떤 시나리오에서 성능 재앙이 촉발되는가?

**참고 해석**：

LSA 도달 가능성 판단은[FACT:src/rma/rma.cc:187-204]에 있으며, peer를 CE와 Proxy 두 그룹으로 나눈다. 이 판단을 제거하면 모든 peer가 Proxy 경로를 타게 되어,`nRmaTasksCe`은 항상 0이 된다.

결과는: CE 경로가 전혀 사용되지 않고, 모든 WaitSignal이 host proxy 스레드를 통해 네트워크를 폴링한다. LSA 범위 내의 peer(동일 머신 NVLink 연결)의 경우, 원래 GPU 복사 엔진으로 비동기 대기할 수 있었지만 이제 host 스레드 폴링으로 바뀌어 지연이 마이크로초 수준에서 밀리초 수준으로 상승한다.

성능 재앙 시나리오: MoE 훈련에서 각 token은 여러 전문가의 신호를 기다려야 한다. 모든 신호가 Proxy를 타면 host 스레드가 병목이 되어 GPU가 대량의 시간을 host 폴링 대기에 소비한다. 8카드 전 NVLink 머신에서 이 퇴화는 특히 두드러진다 — 원래 모든 통신이 CE를 탈 수 있었지만 이제 전부 host로 몰린다.

진단 방법:`scheduleRmaTasksToPlan`의 INFO 로그를 보면, 만약`nRmaTasksCe`이 항상 0이고`nRmaTasksProxy`이 매우 크면 LSA 판단에 문제가 있음을 의미한다.

Q2：`ncclGinProgress`에서`writePending`플래그와`devCommRwMutex`읽기-쓰기 락의 협력에서, 만약`writePending`검사를 제거하고 읽기-쓰기 락만 유지하면 어떤 문제가 발생하는가?

**참고 해석**：

`writePending`검사는[FACT:src/gin/gin_host.cc:63-66]에 있으며, 진행 스레드가 메인 스레드가 쓰려고 할 때 능동적으로 yield하게 한다. 이 검사를 제거하면 진행 스레드는 직접 읽기 락을 획득하려고 시도한다.

문제는:`std::shared_timed_mutex`의 읽기 락은 공유되므로 여러 진행 스레드가 동시에 보유할 수 있다. 메인 스레드가 쓰기 락을 얻으려면 모든 읽기 락이 해제될 때까지 기다려야 한다. 높은 부하에서 진행 스레드가 빈번히 읽기 락을 획득하면 메인 스레드가 장시간 쓰기 락을 얻지 못해`ncclGinDevCommSetup`또는`ncclGinDevCommFree`이 블록될 수 있다.

더 심각한 것은: 만약 메인 스레드가`ginProgressWriteLock`에서 먼저`writePending`을 설정한 후 락을 획득하는데, 진행 스레드가`writePending`을 검사하지 않으면, 진행 스레드가 메인 스레드의 설정 후에도 읽기 락을 획득할 수 있어 메인 스레드의 대기 시간이 예측 불가능해진다.

`writePending`의 역할은 "소프트 알림"이다: 진행 스레드에게 "내가 쓰려고 하니 잠시 양보하라"고 알린다. 이는 단순히 락의 공정성에 의존하는 것보다 더 효율적인데, 진행 스레드가 락에서 블록되는 대신 능동적으로 yield할 수 있기 때문이다.

Q3：`ncclSymkMask`에서, 만약`nBusBytes >= 32 * (size_t(2) << 30)`시 모든 kernel을 비활성화하면(`kmask = 0`), 이때`ncclSymkAvailable`이 false를 반환하고, NCCL은 어떤 경로로 폴백하는가? 이 폴백 경로는 어떤 성능 영향을 미치는가?

**참고 해석**：

`kmask = 0`은[FACT:src/sym_kernels.cc:342]에 있으며, 이때`ncclSymkAvailable`이 false를 반환한다([FACT:src/sym_kernels.cc:354-361]）。

폴백 경로는: NCCL이 전통적인 집합 통신 kernel(비대칭 메모리 kernel)을 사용한다. 이 kernel들은 등록 버퍼 방식으로 상대방 메모리에 접근하며, 먼저 주소를 해석해야 하므로 명령 오버헤드가 더 크다.

성능 영향: 초대형 메시지(64GB 버스 바이트 초과)의 경우, 전통 kernel의 주소 해석 오버헤드 비중이 매우 작은데, 데이터 전송 자체가 지배적이기 때문이다. 그러나 경계 상황(막 64GB를 초과)에서는 전통 kernel이 대칭 메모리 kernel보다 10-20% 느릴 수 있다.

이 제한의 근본 원인은: 대칭 메모리 kernel이 32비트 정수로 unrolled loop chunk를 추적하고, 각 chunk가 최소 32바이트이므로 최대 주소 지정 범위가 32 * 2^31 = 64GB이다. 이 범위를 초과하면 정수 오버플로가 발생한다.

실제 생산에서 단일 집합 통신이 64GB를 초과하는 시나리오는 드물지만(보통 그래디언트 누적 후의 all-reduce), 불가능하지는 않다. 이런 시나리오를 만나면 분할 통신이나 전통 kernel 사용을 고려할 수 있다.

---

# 장말 전환

이 장에서 우리는 NCCL이 "고정 집합 연산"에서 "프로그래밍 가능한 통신 엔진"으로 나아가고 있음을 보았다: RMA는 프리미티브 조합을 제공하고, GIN은 GPU 직접 전송을 제공하며, 대칭 메모리는 통합 주소 공간을 제공하고, Team과 버전화된 DevComm은 인프라를 제공한다.

이러한 진화는 고립된 것이 아니라, 공동으로 하나의 목표를 향한다:**상위 프레임워크가 더 낮은 지연과 더 높은 유연성으로 사용자 정의 통신 패턴을 구현할 수 있게 한다**. PyTorch, Megatron 같은 프레임워크에게 이는 NCCL 위에서 직접 MoE all-to-all, 파이프라인 병렬화, 전문가 병렬화 등 복잡한 통신 패턴을 구축할 수 있으며, NCCL을 우회해 자체적으로 네트워크 계층을 구현할 필요가 없다는 것을 의미한다.

다음 장은 전书的 마지막 장이다. 우리는 AllReduce 한 번의 전체 경로를 다시 한 번 따라가 볼 것이다——`ncclAllReduce`호출부터 시작해 작업 큐잉, 알고리즘 선택, kernel 시작, proxy 진행, 네트워크 전송을 거쳐 결과가 반환될 때까지. 이번 회고는 앞선 24개 장의 지식 포인트를 연결해 완전한 인지 지도를 형성할 것이다.

여기까지 우리는 NCCL이 고정 집합 연산에서 프로그래밍 가능한 통신 엔진으로 진화하는 세 가지 주된 흐름을 보았다: RMA 프리미티브 조합, GPU 직접 네트워크 전송, 대칭 메모리 모델, 그리고 이를 뒷받침하는 team 추상화와 버전화된 DevComm. 이러한 메커니즘은 함께 더 유연하고 하드웨어 능력에 더 가까운 통신의 미래를 가리킨다. 그러나 아키텍처가 어떻게 진화하든, AllReduce 한 번의 전체 경로는 항상 NCCL을 이해하는 초석이다. 다음 장에서는 새로운 코드를 도입하지 않고, 제3장부터 제10장까지의 엔드투엔드 흐름을 다시 한 번 연결해 설명할 것이다—— ncclAllReduce 호출부터 통신 도메인 구축, 토폴로지 탐색, 알고리즘 선택, 작업 큐잉, kernel 시작, 디바이스 측 프리미티브 실행, 결과 기록까지. 당신은 각 장에 흩어져 있던 메커니즘을 다시 조립해 완전한 멘탈 모델로 만들고, "문제가 생기면 어느 장을 봐야 하는가"에 대한 색인을 얻게 될 것이다.
