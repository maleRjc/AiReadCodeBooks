# 제 5 장: 알고리즘과 프로토콜 선택: tuning 모듈이 통신 경로를 결정하는 방법

# 제5장: 알고리즘과 프로토콜 선택: tuning 모듈이 통신 경로를 결정하는 방법

지난 장에서 우리는 NCCL의 토폴로지 인식 능력을 분석했다: src/graph/topo.cc에서 디바이스를 열거하여 토폴로지 그래프를 구축하고, src/graph/search.cc에서 최적 경로를 검색하며, rings.cc와 trees.cc에서 검색 결과를 Ring과 Tree 알고리즘 토폴로지로 구체화한다. 그러나 토폴로지 그래프는 "데이터가 어느 길로 갈 수 있는가"만 답할 뿐, "이번 통신이 어느 길로 가야 하는가"는 답하지 않는다. 같은 머신에서 4KB AllReduce와 400MB AllReduce의 최적해는 완전히 다를 수 있다: 전자는 지연 시간을 겨루고, 후자는 대역폭을 겨룬다. 전자는 Tree/LL을 선택할 수 있고, 후자는 Ring/Simple 또는 NVLS를 선택할 수 있다. tuning 모듈이 바로 그 "결정권자"다. 그 입력은 메시지 크기, rank 수, 토폴로지 그래프(지난 장의 산물), 사용자 환경 변수이며, 출력은 ncclTuningResult_t로, 여기에 어떤 알고리즘(algo)을, 어떤 프로토콜(proto)을, 채널을 몇 개 열고, warp를 몇 개 사용할지가 담겨 있다. 이 장에서는 "총调度 → 비용 모델 → 각 알고리즘 추정 → 마무리 결정"의 순서로 src/tuning 디렉터리를 해부한다. 핵심 질문은 단 하나다: NCCL은 어떻게 수십 가지 (알고리즘, 프로토콜) 조합 중에서 순수 CPU 수학 모델로 마이크로초 단위 시간 안에 가장 빠른 하나를 선택하는가?

# 一、tuning.cc: 총调度와 결정의 주 간선

## 직관적 모델

tuning 모듈을 한**이사 회사**라고 상상해 보자. 고객(한 번의 집합 통신)이 와서 "100MB의 화물을 8개 창고에서 8개 창고로 옮기고 싶다"고 말한다. 배차 담당자(`ncclTuningCompute`)는 실제로 한 번 옮겨보지 않고,**가격표**(비용 모델)를 꺼내서 각 방안(Ring/LL, Tree/Simple, NVLS/Simple……)에 대해 "예상 소요 시간"을 추정한 뒤, 가장 짧은 견적을 골라 고객에게 제시한다.

이 배차 담당자가 없다면, NCCL은 "AllReduce는 항상 Ring을 쓴다"고 하드코딩할 수밖에 없고, 그러면 작은 메시지 시나리오에서는 Tree에게, 대규모 NVLink 시나리오에서는 NVLS에게 완패할 것이다.**그 대가는 특정 시나리오에서 성능이 반토막 나거나 더 나빠지는 것이다.**

## 데이터 구조와 메모리 레이아웃

결정의 매개체는`ncclTuningResult_t`이고, 후보 집합은`ncclTuningResultList_t`(단일 연결 리스트)이다. 연결 리스트 노드는`tuning_int.h`에 정의되어 있지만, push 로직은`tuning.cc`에 있다:

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
> 여기서 주목할 점은**헤드 삽입법**이라는 것이다: 유효한 후보가 하나 계산될 때마다 연결 리스트의 머리에 삽입한다. 이는 연결 리스트 순서와 id 순서가**반대**라는 뜻이다. 왜 배열이 아니라 연결 리스트를 쓰는가? 후보 수는 컴파일 시점에`NCCL_TUNING_COUNT`에 의해 결정되지만, 실제 유효한 후보는 동적이며(`tuningMask`, 플랫폼 능력, 사용자 환경 변수의 영향을 받음), 연결 리스트는 "유효한 것만 매달아 두기"를 허용하여 순회 시 반복적으로`valid`를 판단하는 것을 피할 수 있기 때문이다. 그 대가는 매 결정마다`ncclCalloc`한 번이지만, 튜닝이 인큐 경로에서 발생하고 빈도가 높지 않으므로 이 정도 할당 오버헤드는 허용 가능하다.

`ncclTuningResult_t`에서 가장 핵심적인 두 필드는`timeUs`(예상 소요 시간, 마이크로초)와`selectionTimeUs`(선택에 사용되는 소요 시간, tuner 플러그인에 의해 덮어쓰여질 수 있음)이다. 선택 로직은 후자만 본다:

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

여기에는 디테일이 하나 있다:`bestTuning->timeUs`이 먼저`FLT_MAX`로 설정된 후, 순회한다. 만약 연결 리스트가 비어 있다면 (모든 후보가 무효),`bestTuning`은`NCCL_TUNING_RESULT_INIT`의 초기값을 유지하며, algo/proto 모두`UNDEF`이다. 이 "빈 결과"는 호출 측에서 특별 처리된다 — 뒤의 에러 분기를 참조.

## Step-by-Step Walkthrough: 한 번의 AllReduce 의사결정 흐름

애플리케이션이`ncclAllReduce`를 호출한다고 가정하자. 메시지 1MB, 8개 rank 단일 머신 NVLink. 우리는`ncclTuningCompute`을 따라가 본다.

**0단계: 단일 rank 단락.**만약`nRanks <= 1`이면, 통신이 전혀 필요 없으므로 바로 Ring/Simple을 반환하고, channel 수를 0으로 설정한다:

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

여기서`NCCL_TUNING_IGNORE`은 센티넬 값으로, "이 조합은 계산되지 않았음/적용 불가"를 나타낸다. 플러그인은 관심 있는 셀만 수정할 수 있고, 다른 셀은 IGNORE로 유지하면 NCCL이 건너뛴다.

**4단계: 최적 선택.**이`ncclTuningSelectBestTuning`을 호출하여, 연결 리스트를 순회하며`selectionTimeUs`이 가장 작은 것을 선택한다.

**5단계: channel 수 계산.**알고리즘을 선택한 후, channel을 몇 개 열지 결정해야 한다:

[FACT:src/tuning/tuning.cc:233-235]

```c
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
```

`ncclTuningGetChannels`은`tuning_int.h`에서, 메시지 크기와 알고리즘 타입에 따라`minChannels`와`maxChannels`사이를 보간한다. channel 수는 대역폭에 직접 영향을 미친다: channel이 많을수록 병렬성이 높아지지만, 각 channel의 시작 오버헤드도 커진다.

**6단계: CTA Policy 편향 (NVLS 우선).**만약 사용자가`NCCL_CTA_POLICY_EFFICIENCY`을 설정했고, 현재 AllGather/ReduceScatter이며 buffer가 등록되어 있다면, NCCL은 결과를 NVLS로 변경하려고 시도한다:

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

**왜 오류 코드를 구분하는가?**만약 사용자가`NCCL_ALGO=ring`을 설정했지만 현재 플랫폼이 ring을 지원하지 않는다면(예: 일부 특수 토폴로지), 그것은**사용자 구성 오류**（`ncclInvalidUsage`)이다. 만약 사용자가 어떤 환경 변수도 설정하지 않았는데 알고리즘을 선택할 수 없다면, 그것은**NCCL 내부 버그**（`ncclInternalError`)이다. 이 구분은 트러블슈팅에 매우 중요하다.

## 의사결정 주간 흐름도

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

# 2. cost_model.cc: 모델 레지스트리와 스위치 매트릭스

## 직관적 모델

`cost_model.cc`은 tuning의**총 장부**이다. 그것은`modelMap`테이블을 유지하며, 각 행은 (algo, proto) 조합에 대응하고 "이 조합의 초기화 함수는 누구인지, 시뮬레이션 함수는 누구인지, 어떤 함수에 대해 활성화되는지"를 기록한다. 동시에 사용자 환경 변수`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SYM_KERNEL`를 파싱하여, 사용자의 의도를`enabled[i][f]`스위치 매트릭스로 변환한다.

만약 이 테이블이 없다면, 새로운 알고리즘을 추가할 때마다 tuning 메인 흐름을 수정해야 하며, 코드는 엉망이 될 것이다.**테이블 기반**은 "알고리즘 추가"를 "한 줄 추가"로 만든다.

## 데이터 구조: modelMap과 스위치 매트릭스

`modelMap`은 정적 배열이며, 각 요소는`ncclTuningModelEntry_t`：

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

각 entry는 네 개의 필드를 가진다:`init`(초기화, latency/bandwidth를 계산하여 comm에 저장),`model`(시뮬레이션, 메시지 크기에 따라 최종 timeUs 계산),`finalize`(정리),`enabled[5]`(Broadcast/Reduce/AllGather/ReduceScatter/AllReduce 다섯 함수의 활성화 여부에 대해).

주의`enabled`배열의 순서 주석은 L234에 있음:`Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce`. 이 순서는 반드시`ncclFunc_t`열거형과 일치해야 하며, 그렇지 않으면 잘못 매칭됨.

> **[Design Inference & Architectural Trade-offs]**
> **왜 init과 sim을 분리해야 하는가?**init에서 계산하는 것들(latency, bandwidth)은**comm의 정적 속성에만 의존하기 때문**(토폴로지, rank 수, compCap), 구체적인 메시지 크기와는 무관. 한 번의 통신에서 연속으로 여러 번 tuning을 호출할 수 있음(예: group에 여러 op가 있을 때), init은 한 번만 실행되고 sim은 매번 실행됨. 이는 전형적인 「사전 계산 + 빠른 조회」 최적화.

## Step-by-Step: 환경 변수 파싱과 스위치 매트릭스 구축

**1단계: 기본값은 전부 활성화, LL128은 특별.** `ncclTuningCostModelInit`처음에 모든 proto를 1(활성화)로 설정하지만, LL128은 2로 설정:

[FACT:src/tuning/cost_model.cc:313-323]

```c
  for (int f = 0; f minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                          comm->graphs[algo].typeIntra, comm->nRanks, f, algo, comm->minDriverVersion)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
```

**2단계: 사용자 환경 변수 파싱.**사용자가`NCCL_ALGO`또는`NCCL_SYM_KERNEL`를 설정했다면, 먼저 algo와 symKernel을 전부 0으로 초기화(사용자가 화이트리스트를 지정했기 때문):

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

proto는 초기화하지 않음에 주의 — proto의 기본값은 1/2이고, 사용자가`NCCL_PROTO=LL`를 설정하면,`parseList`LL을 1로, 나머지를 0으로 설정하기 때문(`unset`로직). 이 비대칭은 의도적: algo는 기본적으로 전부 활성화지만 사용자가 지정하면 좁혀야 하고, proto의 축소는`parseList`내부에서 처리.

**3단계: parseList의 문법.**이 함수는 상당히 복잡한 문법을 지원하며, 주석에 예시가 있음:

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

`^`접두사는 「부정」을 나타냄:

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

따라서`NCCL_PROTO="^LL128;allreduce:LL128"`의 의미는: 전역적으로 LL128을 비활성화하지만, AllReduce는 예외적으로 LL128을 활성화.

**4단계: enabled 매트릭스 병합.**마지막으로 모든 model을 순회하며,`model->enabled[f]`과 사용자 스위치를 AND 연산:

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

로직은:**사용자가 특정 함수에 forced 설정을 했을 때만, 사용자 설정으로 모델 기본값을 덮어씀**. 사용자가 설정하지 않았다면,`forced[f] == 0`, 바로`continue`, 모델 자체의`enabled`을 유지. 이는 「사용자 명시 지정 > 모델 기본값」의 우선순위.

## 모델 시뮬레이션의 통합 진입점

모든 모델은 최종적으로`ncclTuningCostModelSimModel`을 통해 호출됨:

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

삼중 필터:**id 범위 초과 → 모델 비활성화 → 모델이 비양수 시간 반환**, 어느 한 계층이라도 통과하지 못하면`not_valid`로 가서,`timeUs`을`NCCL_TUNING_IGNORE`(음수 센티넬)로 설정,`valid = 0`. 호출자가`valid == 0`을 보면 후보 연결 리스트에 넣지 않음.

## 설계 사고

`modelMap`의 주석에 핵심 경고가 한 줄 있음:

[FACT:src/tuning/cost_model.cc:229]

```c
// IMPORTANT: this table need must be consistent with the algRegistry in src/config/algorithm_registry.cc
```

> **[Design Inference & Architectural Trade-offs]**
> 이는`modelMap`의**인덱스 순서**가`algorithm_registry.cc`의 알고리즘 등록 순서와 엄격히 일치해야 함을 의미. 만약 누군가 registry에 새 알고리즘을 삽입하고`modelMap`를 수정하는 것을 잊으면, 모든 id가 어긋나서 tuning이 완전히 잘못된 알고리즘을 선택하게 됨.**이는 테이블 기반 설계의 전형적인 함정: 암묵적 계약.**더 견고한 방법은 인덱스 대신 열거형 이름을 key로 사용하는 것이지만, 그렇게 하면 컴파일 타임 최적화를 약간 희생함.

---

# 3. ring.cc: Ring 알고리즘의 비용 추정

## 직관적 모델

Ring 알고리즘은 N개의 rank를 원형으로 배열하고, 데이터가 원을 따라 한 바퀴씩 전달됨. 그 비용 모델은 두 가지 질문에 답해야 함:**각 단계에서 얼마나 많은 데이터를 전송하는가 (대역폭)**、**총 몇 단계가 필요한가 (지연)**。

Ring의 직관은 「**파이프라인**」: N명이 원형으로 서서 물통을 전달하고, 각자 통을 받으면 물을 조금 붓고 다음 사람에게 전달한다고 상상. 통이 한 바퀴 돌면 모든 사람의 물이 섞임. 통이 빨리 돌수록(대역폭 높음), 원이 작을수록(단계 수 적음), 전체가 빨라짐.

## 데이터 구조: latency/bandwidth 테이블

Ring 모델은 새로운 구조를 도입하지 않고, 추정 결과를`comm->tuningContext.generalLatencies[c][algo][proto]`과`generalBandwidths[c][algo][proto]`에 기록. 이 둘은 3차원 배열: 함수 × 알고리즘 × 프로토콜.

초기화 시 먼저 전부 -1.0으로 설정(센티넬, 「계산 안 됨」을 나타냄):

[FACT:src/tuning/ring.cc:31-33]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
```

-1.0 센티넬은 sim 단계에서 검사됨:

[FACT:src/tuning/ring.cc:94-97]

```c
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ncclSuccess;
  }
```

**왜 0이 아니라 -1.0을 사용하는가?**0은 합법적인 대역폭 값(물리적으로 불가능하지만)이고, -1.0은 「미초기화」를 명확히 나타내기 때문. 부동소수점 비교에`==`을 사용하는 것은 여기서 안전한데, -1.0은 정확히 표현 가능하기 때문.

## Step-by-Step: Ring 대역폭 추정

**1단계: intra를 사용할지 inter를 사용할지 결정.**단일 노드(nNodes==1)는 intra, 다중 노드는 inter:

[FACT:src/tuning/ring.cc:34-37]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float bw = (comm->nNodes == 1 || (comm->nNodes minCompCap graphs[algo].bwIntra :
                                                                                      comm->graphs[algo].bwInter;
    float busBw = bw * comm->graphs[algo].nChannels;
```

`nSteps`은 알고리즘에 필요한 단계 수이며, Ring의 경우 AllReduce는`2*(nRanks-1)`, 나머지는`nRanks-1`。`busBw`은 「버스 대역폭」 = 단일 링크 대역폭 × channel 수.

**2단계: 프로토콜별 할인.**LL 프로토콜은 대역폭의 절반만 사용한다 (LL의 flag 오버헤드 때문). LL128은 92%(120/128)를 사용한다:

[FACT:src/tuning/ring.cc:38-42]

```c
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);
```

`0.92 = 120/128`LL128은 128바이트마다 8바이트가 flag이고 유효 페이로드는 120바이트이기 때문이다. 이 숫자는 프로토콜 설계에서 직접 나온 것이다.

**3단계: 유효 대역폭 계산.**여기서 곱한 것에 주의`nRanks / nSteps`：

[FACT:src/tuning/ring.cc:44-46]

```c
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;
```

**왜 곱하는가`nRanks / nSteps`？**이것은 Ring 알고리즘의 핵심 특성이다: 각 rank가 실제로 운반하는 데이터량은`nBytes * nSteps / nRanks`(데이터가 링을 여러 바퀴 돌기 때문). 따라서 '유효 대역폭' = 버스 대역폭 × nRanks / nSteps. AllReduce의 경우 nSteps = 2(nRanks-1)이므로 유효 대역폭 ≈ busBw/2.

**4단계: 지연 계산.**지연은 intra와 inter 두 부분으로 나뉜다:

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

L57-58의 특수 처리에 주의:`maxLocalRanks == 1`(각 노드에 rank가 1개만 있을 때) Ring의 inter-node 지연은**Tree의 NET 지연**을 사용한다. 주석에 따르면 이것은 'preserve the pre-refactor model' — 즉 리팩터링 전 동작과 일치시키기 위해 의도적으로 남겨둔 '괴이한 특성'이다.**이런 역사적 부채는 성숙한 시스템에서 흔하다. 소스 코드를 읽을 때 'preserve'라는 단어를 보면 각별히 주의해야 한다. 그것은 종종 여기에 건드릴 수 없는 호환성 제약이 있음을 의미한다.**

**5단계: 함수 유형별로 누적.**Reduce/Broadcast와 AllReduce/AllGather/ReduceScatter의 지연 모델은 다르다:

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

`sameChannels`은 토폴로지 속성으로, '링의 intra와 inter 단계가 같은 채널 그룹을 사용하는지'를 나타낸다. 다르면 지연에`nSteps`을 곱해야 한다 (매 단계마다 대기해야 함).`netOverhead`은 네트워크 post 오버헤드이며, Simple 프로토콜은 3을 곱한다 (Simple은 send, recv, ack 세 번의 네트워크 왕복이 있기 때문).

## 프로덕션 함정 회피: Ring/Simple의 plateau 효과

`ncclTuningRingModelSim`에 'plateau'를专门 처리하는 코드가 있다:

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
> **plateau란 무엇인가?**Ring/Simple에서 메시지가 어느 정도 커지면 지연이 더 이상 메시지에 따라 선형으로 증가하지 않고 '정체'되어 플랫폼에 머문다 — 이때 병목이 '시작 오버헤드'에서 '대역폭'으로 바뀌었고 대역폭은 이미 포화 상태이기 때문이다. 이 현상은 Blackwell NVLink에서 특히 두드러진다 (NVLink 대역폭이 너무 높아 지연 비중이 더 크기 때문). 코드는`plateauFactor`(1.4 또는 1.9)을 지연에 곱하여 이 '지연이 증폭되는' 효과를 시뮬레이션한다.

`bytesPerRankPerChannel >= 64`은 트리거 조건이다: 각 rank의 각 채널이 최소 64바이트를 전송해야 하며, 그렇지 않으면 plateau가 성립하지 않는다. 이 64바이트는 LL 프로토콜의 flag 크기에서 온다.

**함정 시나리오**: Blackwell에서 1MB AllReduce를 실행했는데 실제 지연이 모델 예측보다 40% 높다면 버그라고 생각하지 마라 — 이것은 plateau 효과이며 모델이 이미それを 계산에 넣은 것이다. 만약 수동으로`plateauFactor`을 줄이면 모델이 지연을 과소평가하여 잘못된 알고리즘을 선택하게 된다.

---

# 4. tree.cc와 nvls.cc: Tree와 NVLS의 비용 추정

## 직관적 모델

**Tree 알고리즘**은 '**트리 브로드캐스트**'이다: 루트 노드가 데이터를 자식 노드에 나누고, 자식 노드가 다시 손자 노드에 나눈다. 장점은**단계 수가 적다**(N이 아닌 log N)는 것으로, 작은 메시지에 적합하다; 단점은**대역폭 활용률이 낮다**(각 비단말 노드가 포워딩해야 하므로 실제 유효 대역폭은 절반에 불과하다).

**NVLS**(NVLink SHARP)는 '**하드웨어 멀티캐스트**'이다: 스위치가 데이터를 여러 GPU에 직접 복사하며 소프트웨어 포워딩이 필요 없다. 장점은**대역폭이 높고 지연이 낮다**는 것이지만, 특정 하드웨어(Hopper 이상)와 특정 구성이 필요하다.

## Tree 모델: AllReduce만 서비스

Tree 모델에는硬性 제한이 있다 —**AllReduce에만 활성화됨**：

[FACT:src/tuning/tree.cc:21-27]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
```

> **[Design Inference & Architectural Trade-offs]**
> **왜인가?**NCCL의 Tree 구현은 AllReduce만 지원하기 때문이다 (다른 집합 연산에는 Tree 버전이 없다). 이것은 구현 제약이지 이론적 한계가 아니다.`enabled[c] = 0`은 '하드 비활성화'로,`generalBandwidths = -1`보다 더 철저하다 — 전자는`ncclTuningCostModelSimModel`이 L480에서 바로`not_valid`을 반환하게 하고, 후자는 sim 함수에 이르러서야 검사한다.

**Tree 대역폭 추정**：

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
> LL 프로토콜의 할인 계수는`1/3.8`로, Ring의`0.5`보다 더 강하다.**왜 Tree의 LL 효율이 더 낮은가?**Tree의 각 중간 노드가 받기도 하고 보내기도 해야 하므로 LL의 flag 오버헤드가 양방향 트래픽에서 증폭되기 때문이다.`1/3.8`이 숫자는 실측에서 온 것이다.

**Tree 지연 추정**：

[FACT:src/tuning/tree.cc:55-58]

```c
    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
```

`2 *`은 AllReduce = ReduceScatter + AllGather, 두 번의 패스이기 때문이다.`(nRanks/nNodes - 1)`은 노드 내 단계 수(각 노드 내 rank 수 빼기 1)이고,`log2i(nNodes)`은 노드 간 단계 수(트리의 높이)이다.

**Tree의 수정 계수**：Tree 모델은 sim 단계에서`treeCorrectionFactor`：

[FACT:src/tuning/tree.cc:75-79]

```c
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize proto][logSize];
```

`treeCorrectionFactor`을 곱하는데, 이는 3×24 테이블입니다:

[FACT:src/tuning/cost_model.cc:223-227]

```c
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};
```

`logSize = log2(nBytes >> 6)`, 즉 메시지 크기를 64바이트 단위로 log2 취한 값입니다. 테이블의 인덱스 0-23은 64B부터 64B×2^23 ≈ 512MB에 대응합니다.**이 테이블은 실측으로 얻은 「Tree 효율 곡선」입니다**：작은 메시지에서는 효율 1.0(지연 지배), 중간 메시지에서는 효율이 0.4-0.5로 떨어지고(대역폭 미포화), 큰 메시지에서는 다시 1.0으로 돌아옵니다(대역폭 포화). 이 「중간 함몰」은 Tree 알고리즘의 고유 특성입니다.

## NVLS 모델: 하드웨어 멀티캐스트의 대가

NVLS 모델은 먼저 하드웨어가 지원하는지 확인합니다:

[FACT:src/tuning/nvls.cc:19-24]

```c
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!ncclNvlsTransportEnabled(comm)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

그다음 일련의 강제 제약이 있습니다: Simple 프로토콜만 지원, 단일 노드에서는 NVLSTree 미지원, 다중 노드 NVLS는 CollNet 필요:

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

**NVLS 대역폭 추정**에는 효율 계수를 사용합니다:

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
> Hopper는 0.85, Blackwell은 오히려 0.74로 떨어집니다.**왜 신세대 하드웨어의 효율이 더 낮을까요?**Blackwell의 NVLink 대역폭은 더 높지만, NVLS의 스위치 처리 능력은 그에 비례해 향상되지 않아 상대 효율이 떨어지기 때문입니다. 이 숫자는 이론값이 아니라 실측값입니다.

대역폭 계산에는`(nChannels - 1) / nChannels`계수가 있습니다:

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

`(nChannels - 1) / nChannels`은 NVLS가 동기화를 위해 채널 하나를 남겨둬야 하기 때문입니다.`(ppn - 1) / ppn`은 AllGather/ReduceScatter의 추가 오버헤드입니다(각 rank가 이전 rank의 데이터를 기다려야 함).

## 프로덕션 함정 회피: NVLS의 강제 제약

NVLS 모델은 sim 단계에서 런타임 검사를 한 겹 더 거칩니다:

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

`NCCL_MAX_NVLS_ARITY`은 NVLS 멀티캐스트 그룹이 수용할 수 있는 최대 GPU 수입니다. 이 수를 초과하면 NVLS를 사용할 수 없습니다.**함정 시나리오**：16카드 NVLink 도메인에서 AllGather를 실행할 때, 만약`NCCL_MAX_NVLS_ARITY`이 8이라면 NVLS가 비활성화되고 tuning은 Ring으로 폴백합니다. 이 제한을 모르면 「NVLS가 분명히 하드웨어 지원이 되는데 왜 안 쓰지」라고 생각하게 됩니다.

---

# 5. 대칭 kernel 폴백과 오류 복구 체인

## 직관적 모델

대칭 kernel(symmetric kernel)은 NCCL의 새로운 기능입니다: 모든 rank의 buffer가 대칭 메모리에 등록되면, kernel이 더 효율적인 명령어로 상대방 메모리에 접근할 수 있습니다. 하지만**buffer가 등록되지 않았거나 플랫폼이 지원하지 않으면 반드시 일반 kernel로 폴백해야 합니다**. 이 폴백 로직은 tuning에서 가장 복잡한 부분입니다.

## Step-by-Step: 폴백 결정

폴백 로직은`tuning.cc:258-298`에 있습니다. 하나씩 뜯어봅시다.

**1단계: 폴백이 필요한지 판단합니다.**진입 조건:

[FACT:src/tuning/tuning.cc:258-263]

여기까지, tuning 모듈의 의사결정 체인은 명확해졌습니다: 토폴로지 그래프와 통신 파라미터를 받아, 비용 모델과 알고리즘 추정을 통해 마이크로초 단위로 최적의 (알고리즘, 프로토콜, channel, warp) 조합을 출력합니다. 하지만 선택은 시작일 뿐입니다——이 결정 결과는 하류에서 어떻게 사용될까요? 다음 장에서는 src/enqueue/enqueue.cc의 본줄기로 들어가, 한 번의 ncclAllReduce 호출이 파라미터 검증, 알고리즘/프로토콜 확정, channel 분할을 거쳐 최종적으로 ncclInfo와 ncclTaskColl 구조를 생성하는 과정을 살펴봅니다. 이는 전서에서 「사용자 관점」에서 「엔진 관점」으로 전환하는 핵심 장이며, 한 번의 집합 통신 호출이 host 측에서 무엇으로 번역되는지, 그리고 그것이 이후 kernel 시작과의 경계가 어디인지 밝혀낼 것입니다.
