# 제 6 장: 연산자 하달 전경: ncclAllReduce가 어떻게 실행 가능한 kernel 작업이 되는가

지난 장에서 우리는 tuning 모듈을 마치며, NCCL이 마이크로초 단위로 한 번의 집합 통신을 위해 (알고리즘, 프로토콜, channel, warp) 조합을 선택한다는 것을 알았다. 하지만 선택 결과 자체는 단지 숫자 더미일 뿐이다. 이것이 실제로 실행되려면 GPU kernel이 읽을 수 있는 작업 설명 객체로 "번역"되어야 한다. 이번 장에서는 src/enqueue/enqueue.cc의 본체로 들어가, 핵심 질문 하나에 답한다. 사용자가 ncclAllReduce를 호출할 때 host 측에서는 대체 무슨 일이 일어나는가? ncclAllReduce에서 ncclEnqueueCheck까지, 파라미터 검증, 알고리즘/프로토콜 결정, channel 분할을 거쳐 최종적으로 ncclInfo와 ncclTaskColl 구조체가 생성된다. 이는 이 책 전체에서 "사용자 관점"에서 "엔진 관점"으로 전환되는 핵심 장이다. NCCL을 식당에 비유하면, enqueue 모듈은 "프런트 주문 시스템"이다. 사용자(애플리케이션 계층)가 "AllReduce 하나 주세요"라고 말하면, 프런트는 이를 주방(GPU kernel)이 실행할 수 있는 작업 지시서로 번역한다. 몇 번 화구, 어떤 팬을 쓸지, 몇 배치로 나눌지까지. 이 번역 계층이 없으면 주방은 무슨 요리를 해야 할지 전혀 알 수 없다.

# 一、入口：ncclAllReduce 如何构造 ncclInfo

## 직관적 모델

`ncclAllReduce`는 사용자가 직접 호출하는 API 함수이다. 그 역할은 극도로 단일하다:**사용자가 전달한 날것의 파라미터를 하나의`ncclInfo`구조체로 패키징한 뒤`ncclEnqueueCheck`**에 넘긴다. 이는 마치 은행 창구에서 업무를 보는 것과 같다. 창구 직원이 먼저 당신의 요구를 표준 양식에 기입한 뒤 백엔드 시스템으로 전달한다.

이 계층이 없다면, 모든 집합 통신 API가 각자 파라미터 검증, group 시맨틱, profiler 계측을 처리해야 한다. 코드는 유지보수 불가능할 정도로 중복될 것이다.

## 데이터 구조: ncclInfo의 메모리 레이아웃

`ncclInfo`는 enqueue 전체 흐름을 관통하는 핵심 매개체이다. 그 정의는`src/include/info.h`：

[FACT:src/include/info.h:17-44]

이 구조체에는 20개 이상의 필드가 있으며, 기능별로 네 그룹으로 나눌 수 있다:

| 필드 그룹 | 필드 | 역할 |
| --- | --- | --- |
| 집합 통신 파라미터 | `coll`, `sendbuff`, `recvbuff`, `count`, `datatype`, `op`, `root` | "무엇을 하는지" 설명 |
| 통신 도메인과 스트림 | `comm`, `stream` | "어디서 하는지" 설명 |
| 알고리즘 세부사항 | `chunkSteps`, `sliceSteps` | "어떻게 분할하는지" 설명 |
| 단방향 연산 | `peerWinOffset`, `peerWin`, `sigIdx`, `ctx`, `flags`, `nDesc`, `signalDescs` | RMA 전용 |
| 사용자 구성 | `collConfig` | 사용자 config에서 복사한 사설 복사본 |

주의`collConfig`의 주석:**"A config copied from config passed by user so older user config can be safely accessed during synchronous host scheduling (never at launch/replay)"** [FACT:src/include/info.h:41-43]. 이것은 핵심 설계이다. 사용자가 전달한 config 포인터는`ncclGroupEnd`이전에 파괴될 수 있으므로, NCCL은`ncclInfo`에서 복사본을 만들어 둔다.

## Step-by-Step: ncclAllReduce의 호출 체인

우리는`ncclAllReduce`을 예로 들어, 사용자 호출부터`ncclInfo`생성까지의 전체 경로를 추적한다.

**1단계: 사용자가 ncclAllReduce를 호출한다.**진입점은`src/collectives.cc`：

[FACT:src/collectives.cc:206-211]

여기서 세 가지 일을 한다:

1. `NVTX3_FUNC_WITH_PARAMS`NVTX 마커를 찍는다(Nsight 등 도구 시각화용)

2.`ncclAllReduceConfigImpl`을 호출하며,`config = nullptr`

을 전달한다

**3. 결과를 반환한다**2단계: ncclAllReduceConfigImpl이 ncclInfo를 생성한다.

[FACT:src/collectives.cc:192-202]

이것이 핵심 단계이다:

```c
struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                        sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
                        ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
```

복사`ncclInfo`필드는`ALLREDUCE_CHUNKSTEPS`의 선언 순서와 일대일로 대응한다.`ALLREDUCE_SLICESTEPS`과`src/include/collectives.h`：

[FACT:src/include/collectives.h:19-20]

`NCCL_STEPS`는`NCCL_STEPS/2`에 정의되어 있다.`NCCL_STEPS/4`는 링 버퍼의 스텝 수(보통 8 또는 16)이므로, AllReduce의 chunkSteps는

**, sliceSteps는** `ncclParseCollConfig`이다. 이는 하나의 chunk가 2개의 slice를 포함한다는 뜻이다.`ncclCollConfig_t*`3단계: 사용자 config를 파싱한다.`info.collConfig`사용자가 전달한`config == nullptr`을

**로 파싱한다. 만약**이면, 이 필드는 0으로 초기화된 상태를 유지한다.

## 4단계: ncclEnqueueCheck에 넘긴다.

> **[Design Inference & Architectural Trade-offs]**
> 설계 고찰: 왜 필드별 할당 대신 집합 초기화를 사용하는가?**〔설계 추론과 아키텍처 트레이드오프〕**집합 초기화에는 두 가지 장점이 있다. 첫째, 컴파일러가 필드 수가 맞는지 검사한다(필드가 하나 부족하면 경고). 둘째, 코드가 더 간결하다. 하지만 단점은`ncclInfo`필드 순서가 구조체 선언과 엄격히 일치해야 한다

## 는 점이다. 만약 누군가

중간에 필드를 삽입하면, 모든 집합 초기화 지점이 조용히 어긋난다. 이는 NCCL 코드에 내재된 유지보수 위험이다.

```c
ncclCollConfig_t config = {...};
ncclAllReduceConfig(..., &config);
// config 在这里被销毁（比如是栈变量，函数返回了）
```

실제 함정 시나리오: 사용자가 이렇게 코드를 작성한다:`ncclInfo`복사`ncclGroupEnd`만약 NCCL이`info.collConfig`에서 config를 복사하지 않았다면,`src/include/info.h:41-43`시**에 접근할 때 이미 해제된 메모리를 읽게 된다.**。

---

# 의 주석은 바로 이 설계를 설명하기 위한 것이다.

## config는 task append 단계에서 파싱되고 복사되며, 이후에는 사용자 포인터에 의존하지 않는다

`ncclEnqueueCheck`二、ncclEnqueueCheck：参数校验与 group 语义**직관적 모델**는 enqueue 모듈의 "메인 게이트"이다. 모든 집합 통신 API가 최종적으로 여기로 모인다. 그 역할은:`ncclEnqueueCheck`。

파라미터 유효성 검증, group 시맨틱 처리, taskAppend 호출로 작업 생성

## Step-by-Step: ncclEnqueueCheck의 실행 흐름

[FACT:src/enqueue/enqueue.cc:3478-3527]

단계별로 분석해 보겠습니다:

**1단계: CommCheck로 통신 도메인 검증.** `CommCheck(info->comm, info->opName, "comm")`comm 포인터가 null이 아닌지, 초기화되었는지 확인합니다. 만약 comm이 revoke되었다면(예: 특정 rank 오류), 바로 오류를 반환합니다:

[FACT:src/enqueue/enqueue.cc:3480-3485]

**2단계: profiler 깊이 처리.**이미 group 내부에 있다면(`profilerGroupDepth > 0`), 깊이 카운트를 증가시킵니다. 이는 암시적`ncclGroupStartInternal`/`ncclGroupEndInternal`호출을 올바르게 처리하기 위함입니다.

**3단계: 내부 group 진입.** `ncclGroupStartInternal()`는 NCCL 내부의 group 메커니즘입니다.**핵심 포인트**: 사용자가 명시적으로`ncclGroupStart`를 호출하지 않아도, NCCL은 각 API 호출마다 암시적 group을 생성합니다. 이는 단일 호출의 원자성을 보장합니다.

**4단계: comm 준비 확인.** `ncclCommEnsureReady(info->comm)`통신 도메인 초기화 완료(예: bootstrap 완료, 연결 설정)를 기다립니다.

**5단계: ArgsCheck 매개변수 검증.**이것이 가장 복잡한 검증 단계입니다:

[FACT:src/enqueue/enqueue.cc:3497-3503]

주의`checkMode`처리: 만약`ncclCheckModeDebugGlobal`，`ArgsCheck`이면 info를 큐에 넣고,`ncclGroupEnd`시점에 전역 검증(예: 모든 rank의 count 일치 여부 확인)을 수행합니다.

**6단계: taskAppend 호출.**이것이 핵심 변환 단계입니다:

[FACT:src/enqueue/enqueue.cc:3513]

**7단계: opCount 증가.**매번 성공적으로 큐에 넣은 후,`comm->opCount++`. 이 카운터는 send/recv 작업 매칭에 사용되며, profiler 타임라인의 근거이기도 합니다.

**8단계: group 종료.** `ncclGroupEndInternal()`depth가 0으로 떨어지면, 실제 group 작업(스케줄링, kernel 시작)이 트리거됩니다.

## 동시성 제어: group 의미론과 스레드 안전성

> **[Design Inference & Architectural Trade-offs]**
> `ncclGroupStartInternal`/`ncclGroupEndInternal`스레드 로컬 저장소(TLS)를 사용하여 group 상태를 유지합니다. 이는**동일 스레드 내의 여러 API 호출이 하나의 group으로 병합됨**을 의미하지만, 다른 스레드의 호출은 독립적입니다. 이것이 NCCL이 멀티스레드 호출을 지원하는 기반입니다.

쉽게 빠질 수 있는 함정: 사용자가`ncclGroupStart`와`ncclGroupEnd`사이에 비 NCCL CUDA API(예:`cudaMemcpy`)를 호출하면, stream 순서 문제가 발생할 수 있습니다. NCCL의 group 메커니즘은 group 내 작업이 모두 동일한 stream 그룹에 있다고 가정합니다.

## 오류 복구 체인

`ncclEnqueueCheck`의 오류 처리에는 정교한 설계가 있습니다:

[FACT:src/enqueue/enqueue.cc:3524-3526]

만약`taskAppend`이 실패하고 comm이 비차단 모드이면,`ncclCommSetAsyncError`을 호출하여 오류를 기록합니다. 이렇게 하면 이후 API 호출이 계속 시도하는 대신 즉시 오류를 반환합니다. 이것이 비동기 오류 전파 메커니즘입니다.

---

# 3. taskAppend: 작업 분배의 교차로

## 직관적 모델

`taskAppend`는 enqueue 모듈의 "교통 허브"입니다.`info->coll`값에 따라 작업을 P2P, RMA, CE, 또는 일반 집합 통신 등 다양한 처리 경로로 분배합니다. 이는 마치 우체국 분류 센터처럼——봉투에 적힌 주소에 따라 편지를 다른 우체통에 넣는 것과 같습니다.

만약 이 분배 계층이 없다면, 모든 유형의 작업이 하나의 거대한 if-else에 몰려 코드 유지보수가 어려워질 것입니다.

## Step-by-Step: taskAppend의 분배 로직

[FACT:src/enqueue/enqueue.cc:3337-3476]

**1단계: 새 아키텍처 활성화 여부 판단.** `ncclParamEnqueueRearchEnable()`는 환경 변수 스위치(기본값 0)입니다. 활성화되면`rawTaskAppend`경로로 진행합니다——이는 NCCL이 개발 중인 새 작업 모델입니다.

**2단계: P2P 분배.**Send/Recv인 경우,`p2pTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3343-3345]

**호출**3단계: RMA 분배.`rmaTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3346-3347]

**PutSignal/Signal/WaitSignal인 경우,** `if (info->count == 0) return ncclSuccess;`호출

**4단계: 빈 집합 통신 조기 반환.** `ncclCollConfigGetAlgMask`——count가 0인 집합 통신은 바로 폐기합니다.

[FACT:src/enqueue/enqueue.cc:3357-3358]

**5단계: 알고리즘 선택 검증.**사용자가 전달한 알고리즘 선택이 유효한지 검증합니다:

[FACT:src/enqueue/enqueue.cc:3360-3366]

**6단계: FP8 타입 검사.** `hostToDevRedOp`FP8 리덕션은 sm90+가 필요합니다:`ncclRedOp_t`7단계: 리덕션 작업 변환.`ncclDevRedOpFull`：

[FACT:src/enqueue/enqueue.cc:3370-3371]

**host 측**을 디바이스 측`comm->nRanks == 1`로 변환`ncclLaunchOneRank`8단계: 단일 rank 조기 반환.

[FACT:src/enqueue/enqueue.cc:3373-3377]

**만약**이면,

[FACT:src/enqueue/enqueue.cc:3378-3470]

## 을 직접 호출하여 로컬 리덕션을 수행하고, 작업을 생성할 필요가 없습니다:

`collTaskAppend`9단계: 다중 rank 경로.`ncclTaskColl`이것이 가장 복잡한 분기로, CE 라우팅, AllToAll/Gather/Scatter 강등, 그리고 일반 집합 통신을 포함합니다:

[FACT:src/enqueue/enqueue.cc:2757-2851]

데이터 구조: ncclTaskColl의 필드

| 는 | 을 생성하는 곳입니다. 핵심 로직을 살펴보겠습니다: | 주요 필드 할당: |
| --- | --- | --- |
| `func` | `info->coll` | 필드 |
| `sendbuff`/`recvbuff` | `info->sendbuff`/`recvbuff` | 출처 |
| `count` | `info->count` | 의미 |
| `datatype` | `info->datatype` | 집합 통신 유형 |
| `trafficBytes` | `count * elementSize * ncclFuncTrafficPerByte` | 버퍼 포인터 |
| `opHost`/`opDev` | `info->op`/`opDev` | 요소 수 |
| `chunkSteps`/`sliceSteps` | `info->chunkSteps`/`sliceSteps` | 데이터 타입 |
| `minCTAs`/`maxCTAs`/`nvlsCTAs` | 트래픽 추정 | 리덕션 작업 |
| `algMask` | `ncclCollConfigGetAlgMask` | 분할 단계 수 |

설정 파싱`trafficBytes`리소스 상한

[FACT:src/enqueue/enqueue.cc:2813]

`ncclFuncTrafficPerByte`알고리즘 선택 마스크

[FACT:src/enqueue/enqueue.cc:123-134]

주의

## 계산:

[FACT:src/enqueue/enqueue.cc:2808-2812]

은 각 집합 통신의 트래픽 배수를 반환합니다:`ncclInt8`. 이것은 최적화입니다:**이 두 작업은 리덕션을 포함하지 않으므로 데이터 타입을 신경 쓸 필요가 없고, 통일적으로 바이트 단위로 처리하면 kernel 로직을 단순화할 수 있습니다**。

## 프로덕션 함정: CTAPolicy의 파싱 순서

[FACT:src/enqueue/enqueue.cc:3390-3397]

CTAPolicy의 파싱에는 미묘한 우선순위가 있습니다:**env > per-call > comm**. 그리고`NCCL_CTA_POLICY_ZERO`이(가)`NCCL_CTA_POLICY_EFFICIENCY`보다 우선합니다. 사용자가 이 두 플래그를 동시에 설정하면 ZERO가 적용됩니다.

실제 함정 시나리오: 사용자가`NCCL_CTA_POLICY=EFFICIENCY`를 설정했지만 CE 경로가 사용되지 않는 것을 발견했습니다. 원인은 CE 라우팅이`CTAPolicy & NCCL_CTA_POLICY_ZERO`가 참이어야 하는데 EFFICIENCY는 이 조건을 충족하지 않기 때문입니다.

---

# 4. ncclPrepareTasks: 작업 목록에서 스케줄링 큐까지

## 직관적 모델

`ncclPrepareTasks`은 enqueue 모듈의 "전처리기"입니다. 흩어진 작업 목록을 (func, op, datatype)별로 버킷팅한 다음, 각 버킷에 대해 알고리즘과 프로토콜을 계산합니다. 이는 마치 도서관 사서와 같습니다 — 반납된 책을 먼저 분류별로 정리한 다음, 각 분류의 책을 어느 서가에 놓을지 결정합니다.

이 단계가 없으면 이후의`scheduleCollTasksToPlan`이 각 작업에 대해 개별적으로 알고리즘을 계산해야 하므로 효율이 매우 낮습니다.

## 단계별: ncclPrepareTasks의 버킷팅 로직

[FACT:src/enqueue/enqueue.cc:423-642]

**1단계: Broadcast 작업 변환.**broadcast peer가 하나뿐이면 broadcast 작업을 coll 작업으로 변환합니다:

[FACT:src/enqueue/enqueue.cc:430-461]

여기서`bcastTask`의 필드를 새로운`ncclTaskColl`에 복사하고`trafficBytes`을 계산합니다. 그런 다음`memPool_ncclTaskBcast`에서 원래 작업을 해제합니다.

**2단계: (func, op, datatype)별 버킷팅.**작업은 sorter에서 size 내림차순으로 나온 다음`tasksByFnOpTy`배열에 할당됩니다:

[FACT:src/enqueue/enqueue.cc:464-487]

인덱스 계산:`((int)task->func * ncclNumDevRedOps + (int)task->opDev.op) * ncclNumTypes + (int)task->datatype`. 이것은 3차원 배열의 선형화입니다.

**3단계: 집계 및 알고리즘 선택.**각 버킷에 대해 크기가 비슷한 작업(4배 이내)을 집계한 다음`ncclGetAlgoInfo`：

[FACT:src/enqueue/enqueue.cc:503-547]

**을 호출합니다**4단계: (collnet, nvls)별 버킷팅.`collBins[2][2]`：

[FACT:src/enqueue/enqueue.cc:517-544]

**알고리즘 유형에 따라 작업을**에 할당합니다`planner->collTaskQueue`：

[FACT:src/enqueue/enqueue.cc:553-557]

## 5단계: 최종 큐 연결.

`ncclTaskCollSorter`네 개의 버킷을`trafficBytes`로 연결합니다`ncclTaskCollSorterInsert`데이터 구조: ncclTaskCollSorter`ncclTaskCollSorterDequeueAll`은

> **[Design Inference & Architectural Trade-offs]**
> 작업을 올바른 위치에 삽입하고,**순서대로 모든 작업을 꺼냅니다.**〔설계 추론 및 아키텍처 트레이드오프〕

## 이 정렬기의 설계 동기는:

[FACT:src/enqueue/enqueue.cc:572-583]

대형 작업 우선 스케줄링`comm->runtimeConn`. 대형 작업은 전송 시간이 길기 때문에 먼저 시작하면 계산과 통신을 더 잘 중첩할 수 있습니다.`algoNeedConnect`동시성 제어: runtimeConn과 연결 설정

## 만약

[FACT:src/enqueue/enqueue.cc:507-508]

이 참이면(런타임 연결 모드), 그리고 특정 알고리즘의 channel이 아직 초기화되지 않았다면`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes`을 표시합니다. 이는 이후에 연결 설정을 트리거합니다.`aggIsolate`프로덕션 함정: 집계의 경계 조건`maxCTAs`），`aggIsolate`집계 조건은

이고, 두 작업 모두`maxCTAs=4`을 설정하지 않아야 합니다. 사용자가 per-call config를 설정하면(예:`aggIsolate`가 true로 설정됨), 이 작업은 집계되지 않습니다.`collTaskAppend`실제 함정 시나리오: 사용자가 특정 AllReduce에

[FACT:src/enqueue/enqueue.cc:2821-2822]

---

# 를 설정하여 4개의 CTA만 사용하기를 기대했습니다. 그러나 집계 로직으로 인해 이 작업이 인접 작업과 병합되어 실제 사용되는 CTA 수가 예상과 다를 수 있습니다. 해결책은

## 을 설정하는 것입니다 — NCCL은

`scheduleCollTasksToPlan`에서 이미 이를 처리했습니다:

5. scheduleCollTasksToPlan: channel 분할과 예산 제어

## 직관적 모델

[FACT:src/enqueue/enqueue.cc:644-947]

**은 enqueue 모듈의 "스케줄러"입니다. 작업을 구체적인 channel에 할당하고 각 channel의 데이터 분할을 계산합니다. 이는 공장의 생산 계획 시스템과 같습니다 — 각 생산 라인이 무엇을, 얼마나 만들지 결정합니다.**이 단계가 없으면 GPU kernel은 자신이 어느 부분의 데이터를 처리해야 하는지 알 수 없습니다.

[FACT:src/enqueue/enqueue.cc:648-689]

`ncclTestBudget`단계별: channel 분할 알고리즘

[FACT:src/enqueue/enqueue.cc:343-349]

**1단계: 예산 추정.**먼저 이 plan에 넣을 수 있는 작업 수를 추정합니다:`trafficPerChannel`：

[FACT:src/enqueue/enqueue.cc:701-707]

**작업 바이트 수가 예산을 초과하는지 확인합니다:**2단계: 각 channel의 트래픽 계산.

[FACT:src/enqueue/enqueue.cc:709-739]

**kind(collnet/nvls)에 따라**을 계산합니다

[FACT:src/enqueue/enqueue.cc:740-845]

3단계: Collnet 경로.

- `cellSize`collnet 알고리즘이면 channel 할당이 비교적 간단합니다:`MinTrafficPerChannel`（32KB）
- `cells`4단계: 일반 경로의 cell 분할.
- `cellsPerChannel`이것이 가장 복잡한 부분입니다. NCCL은 데이터를 "cell"로 나누며, 각 cell은 최소 전송 단위입니다:
- `cellsLo`/`cellsHi`핵심 변수:

**: 각 cell의 바이트 수, 최소**: 총 cell 수`calcCollChunking`：

[FACT:src/enqueue/enqueue.cc:811-825]

**: 각 channel이 처리하는 cell 수**: 처음과 끝 channel의 cell 수(불완전할 수 있음)

[FACT:src/enqueue/enqueue.cc:844-894]

## 5단계: chunkGrains 계산.

`ncclDevWorkColl`각 channel 세그먼트에 대해

| 을 호출합니다 | 6단계: proxyOp 생성. |
| --- | --- |
| `sendbuff`/`recvbuff` | 각 channel에 대해 proxy 작업을 생성합니다: |
| `channelLo`/`channelHi` | 데이터 구조: ncclDevWorkColl |
| `cbd.countLo`/`countMid`/`countHi` | 은 디바이스 측 작업 설명자입니다. 핵심 필드: |
| `cbd.chunkGrainsLo`/`Mid`/`Hi` | 필드 |
| `direct` | 의미 |

## 버퍼 포인터

[FACT:src/enqueue/enqueue.cc:897]

channel 범위`(2ull << channelHi) - (1ull << channelLo)`。예를 들어 channelLo=2, channelHi=5이면 결과는`(2<<5) - (1<<2) = 64 - 4 = 60 = 0b111100`, 즉 bit 2-5가 설정됩니다.

## 프로덕션 함정: 예산 오버플로

[FACT:src/enqueue/enqueue.cc:792-794]

예산이 부족하면 바로 반환하여`ncclSuccess`, 외부 루프가 새로운 plan을 생성하도록 합니다. 이것은 우아한 성능 저하 전략입니다——**오류를 내지 않고, 단지 배치로 처리할 뿐입니다**。

실제 함정 시나리오: 만약`NCCL_WORK_FIFO_BYTES`이 너무 작게 설정되면, 각 plan이 매우 적은 작업만 수용할 수 있어 kernel 시작 횟수가 증가하고 성능이 저하됩니다.

---

# 6. finishPlan: 작업에서 kernel 파라미터로

## 직관적 모델

`finishPlan`은 enqueue 모듈의 "패커"입니다. 작업, batch, proxyOp를 kernel이 직접 읽을 수 있는 파라미터 구조로 패킹합니다. 이는 택배 포장과 같습니다——낱개 물품을 상자에 넣고, 운송장을 붙여, 발송을 기다립니다.

## 단계별: finishPlan의 패킹 로직

[FACT:src/enqueue/enqueue.cc:236-330]

**1단계: 저장 유형 결정.**모든 작업이 kernel args에 들어갈 수 있으면,`ncclDevWorkStorageTypeArgs`：

[FACT:src/enqueue/enqueue.cc:244-250]

**2단계: kernelArgs 할당.**메모리 스택에서 할당:

[FACT:src/enqueue/enqueue.cc:251-255]

**3단계: Round-robin으로 batch 배치.**각 channel의 첫 번째 batch는 반드시`batchZero[blockIdx.x]`：

[FACT:src/enqueue/enqueue.cc:257-280]

**4단계: proxyOp 큐 병합.**opCount 기준 병합 정렬:

[FACT:src/enqueue/enqueue.cc:282-329]

## 데이터 구조: ncclDevKernelArgs

`ncclDevKernelArgs`은 kernel에 전달되는 파라미터 구조입니다. 다음을 포함합니다:

- `comm`: 디바이스 측 communicator
- `channelMask`: channel 비트 마스크
- `workStorageType`: 작업 저장 유형
- `workBuf`: 작업 버퍼 포인터
- `workMask`: 작업 버퍼 마스크

## 프로덕션 함정: batch 순서

[FACT:src/enqueue/enqueue.cc:257-259]

주석에 명확히 나와 있습니다: "The first batch for each channel must be located at batchZero[blockIdx.x]". 이 순서가 틀리면 kernel이 잘못된 batch를 읽어 데이터 손상이 발생합니다.

---

# 이 장 요약

이 장에서 우리는`ncclAllReduce`에서`ncclTaskColl`까지의 전체 경로를 추적했습니다:

1. **ncclAllReduce**이`ncclInfo`을 구성하고, 사용자 파라미터를 패킹

2. **ncclEnqueueCheck**파라미터 검증, group 시맨틱 처리

3. **taskAppend**작업 유형에 따라 다른 경로로 분배

4. **collTaskAppend**을 생성하고, 설정 파싱`ncclTaskColl`(func, op, datatype) 기준 버킷팅, 알고리즘 계산

5. **ncclPrepareTasks**channel 분할,

6. **scheduleCollTasksToPlan**을 생성하여 kernel 파라미터로 패킹`ncclDevWorkColl`

7. **finishPlan**핵심 설계 사상:

계층적 디커플링

- **: 각 함수는 한 가지 일만 하며,**과`ncclInfo`을 통해 상태 전달`ncclTaskColl`예산 제어
- **:**을 통해 각 plan의 크기 제어`ncclTestBudget`집계 최적화
- **: 크기가 비슷한 작업이 집계되어 kernel 시작 횟수 감소**설정 우선순위
- **다음 장에서는**：env > per-call > comm

으로 들어가, NCCL이 다중 channel 다중 kernel의 실행 순서를 어떻게 편성하는지 살펴봅니다.`task_sched`이 장 생각해보기와 자가 점검

# Q1: 만약

에서`collTaskAppend`판단을 제거하면(즉,`aggIsolate`이 항상 false를 반환하면), 어떤 시나리오에서 사용자가 설정한`src/enqueue/enqueue.cc:2821-2822`이 무효화됩니까? 왜 그럴까요?`maxCTAs`참고 해석

**의 역할은 "이 작업은 집계될 수 없다"를 표시하는 것입니다. 이 판단을 제거하면, per-call config를 설정한 작업이 인접 작업과 병합됩니다.**：`aggIsolate`의 집계 루프에서(`ncclPrepareTasks`), 집계 조건은`src/enqueue/enqueue.cc:507-508`입니다. 만약`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes && !aggBeg->aggIsolate && !aggEnd->aggIsolate`이 항상 false이면, 작업이`aggIsolate`을 설정했더라도`maxCTAs=4`인 작업과 병합될 수 있습니다. 병합된`maxCTAs=32`은两者的某种组合을 취하게 되어(구체적으로는`agg`의 구현에 따라 다름), 실제 사용되는 CTA 수가 사용자 기대와 맞지 않게 됩니다.`ncclGetAlgoInfo`더 심각한 것은,

에서(`scheduleCollTasksToPlan`는 per-call 리소스가 설정된 작업이 단독으로 하나의 plan을 차지하도록 보장하는 데 사용됨) 이 판단이 무효화되면, 여러 작업이 plan의 channel 예산을 공유하게 되어 리소스 할당이 기대와 맞지 않게 됩니다.`src/enqueue/enqueue.cc:665-666`），`taskAggIsolate`Q2:

에서 만약`ncclEnqueueCheck`이 오류를 반환하면(예: 특정 rank의 ArgsCheck 실패), 하지만`ncclGroupEndInternal()`이 이미 성공적으로 실행되었다면, 무슨 일이 발생합니까? NCCL은 어떻게 상태 일관성을 보장합니까?`taskAppend`참고 해석

**:**의 제어 흐름을 보면:`src/enqueue/enqueue.cc:3513-3519`복사

```c
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
info->comm->opCount++;
exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
```

이 성공했지만`taskAppend`이 실패하면,`ncclGroupEndInternal`이 이미 증가했습니다. 이로 인해 후속 작업의 opCount가 상대방과 맞지 않아 hang이 발생할 수 있습니다.`opCount`NCCL의 처리 방식은:

이 오류가 있는지 확인하고, 있으면 comm의 오류 상태를 설정합니다. 후속 API 호출은`ncclGroupErrCheck(ret)`을 통해 이 오류를 감지하고 즉시 반환합니다. 이것은 "빠른 실패" 전략입니다——일단 오류가 나면 전체 comm이 오류 상태로 들어가고, 더 이상 복구를 시도하지 않습니다.`ncclCommGetAsyncError`프로덕션 환경에서 이는 group 오류가 발생하면 사용자가 communicator를 파괴하고 재생성해야 함을 의미합니다.

의 cell 분할 알고리즘(

Q3: `scheduleCollTasksToPlan`)에는 경계 조건이 있습니다: 만약`src/enqueue/enqueue.cc:740-845`이면, 최소 channel을 건너뜁니다. 만약 이 건너뛰기 로직에 버그가 있으면(예:`cellsLo == 0`이 올바르게 증가하지 않으면), 어떤 결과가 발생합니까?`channelId`참고 해석

**:**복사`src/enqueue/enqueue.cc:770-780`：

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

이 올바르게 증가하지 않으면, 다음 작업이 잘못된 channel에서 할당을 시작합니다. 이로 인해:`channelId`channel 중복

1. **: 두 작업이 같은 channel의 같은 데이터 구간에 할당될 수 있음**데이터 손상

2. **: kernel이 데이터를 중복 처리하거나 누락함**성능 저하

3. **性能下降**: channel 부하 불균형

더 은밀한 것은, 이런 버그가 특정 메시지 크기에서만 발생할 수 있다는 점이다(`cellsLo == 0`일 때), 재현하기 어렵다. NCCL은`plan->channelMask |= (2ull << devWork->channelHi) - (1ull << devWork->channelLo)`를 통해 사용된 channel을 추적하지만, 이는 단지 기록일 뿐 중복을 방지하지는 못한다.

여기까지 우리는 ncclAllReduce가 사용자 호출에서 어떻게 실행 가능한 kernel 작업들의 연속으로 변하는지 살펴보았다: 파라미터 검증, 알고리즘/프로토콜 결정, channel 분할, 최종적으로 ncclInfo와 ncclTaskColl 생성. 하지만 작업이 생성되는 것은 첫 단계일 뿐이다——이들은 여러 channel에 스케줄링되고, kernel 시작 파라미터를 생성하며, group 시맨틱 하에서 배치 제출과 의존성 정렬을 처리해야 한다. 다음 장에서는 src/enqueue/task_sched와 src/enqueue/task_prep를 깊이 파고들어 "왜 한 번의 AllReduce가 여러 kernel을 시작하는가, 이들 간의 순서와 의존성은 어떻게 보장되는가"를 답하고, 동시에 src/group.cc에서 ncclGroupStart/ncclGroupEnd가 어떻게 여러 API 호출을 하나의 제출로 병합하는지 밝힌다.
