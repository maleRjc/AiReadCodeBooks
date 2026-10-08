# Chapter 5: Algorithm and Protocol Selection: How the tuning Module Determines the Communication Path

In the previous chapter, we broke down NCCL's topology-awareness capability: from enumerating devices in src/graph/topo.cc to build the topology graph, to searching for the optimal path in src/graph/search.cc, and then to rings.cc and trees.cc materializing the search results into Ring and Tree algorithm topologies. But the topology graph only answers "which paths data can take"; it does not answer "which path this communication should take." On the same machine, a 4KB AllReduce and a 400MB AllReduce may have completely different optimal solutions: the former competes on latency, while the latter competes on bandwidth; the former may choose Tree/LL, while the latter may choose Ring/Simple or NVLS. The tuning module is the one that "makes the call." Its inputs are message size, number of ranks, topology graph (the product of the previous chapter), and user environment variables; its output is an ncclTuningResult_t, which specifies which algorithm (algo) to use, which protocol (proto) to use, how many channels to open, and how many warps to use. In this chapter, we will break open the src/tuning directory in the order of "overall scheduling → cost model → estimates for each algorithm → final decision." There is only one core question: how does NCCL choose the fastest one among dozens of (algorithm, protocol) combinations using a purely CPU-based mathematical model within microseconds?

# 1. tuning.cc: Overall Scheduling and Decision Backbone

## Intuitive Model

Imagine the tuning module as a**moving company**. A customer (one collective communication) arrives and says, "I want to move 100MB of goods from 8 warehouses to 8 warehouses." The dispatcher (`ncclTuningCompute`) will not actually try moving it once, but instead takes out a**price list**(cost model), estimates an "expected time" for each option (Ring/LL, Tree/Simple, NVLS/Simple, ...), and then picks the shortest quote for the customer.

Without this dispatcher, NCCL could only hard-code "AllReduce always uses Ring," which would be crushed by Tree in small-message scenarios and by NVLS in large-scale NVLink scenarios.**The cost is that performance is halved or even worse in specific scenarios.**

## Data Structures and Memory Layout

The carrier of the decision is`ncclTuningResult_t`, and the candidate set is`ncclTuningResultList_t`(a singly linked list). The linked list node is defined in`tuning_int.h`, but the push logic is in`tuning.cc`:

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
> Note that this is**head insertion**: each time a valid candidate is computed, it is inserted at the head of the linked list. This means the linked list order and the id order are**reversed**. Why use a linked list instead of an array? Because the number of candidates is determined at compile time by`NCCL_TUNING_COUNT`, but the actually valid candidates are dynamic (affected by`tuningMask`, platform capabilities, and user environment variables). A linked list allows "only attaching the valid ones," avoiding repeated checks of`valid`during traversal. The cost is that each decision requires`ncclCalloc`once, but tuning happens on the enqueue path and at low frequency, so this allocation overhead is acceptable.

`ncclTuningResult_t`The two most critical fields in`timeUs`(estimated time, microseconds) and`selectionTimeUs`(the time used for selection, which may be overridden by a tuner plugin). The selection logic only looks at the latter:

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

There is a detail here:`bestTuning->timeUs`is first set to`FLT_MAX`, then iterated. If the linked list is empty (all candidates are invalid),`bestTuning`will keep`NCCL_TUNING_RESULT_INIT`'s initial value, and both algo/proto are`UNDEF`. This "empty result" is specially handled by the caller—see the error branch later.

## Step-by-Step Walkthrough: The Decision Flow of a Single AllReduce

Suppose the application calls`ncclAllReduce`, with a 1MB message and 8 ranks on a single-node NVLink. We follow`ncclTuningCompute`through it once.

**Step 0: Single-rank short circuit.**If`nRanks <= 1`, communication is not needed at all, and it directly returns Ring/Simple with the channel count set to 0:

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

Here`NCCL_TUNING_IGNORE`is a sentinel value indicating "this combination has not been computed/is not applicable." The plugin can modify only the cells it cares about, leaving other cells as IGNORE, and NCCL will skip them.

**Step 4: Select the best.**calls`ncclTuningSelectBestTuning`, traversing the linked list to take the one with the smallest`selectionTimeUs`.

**Step 5: Compute the channel count.**After selecting the algorithm, it still needs to decide how many channels to open:

[FACT:src/tuning/tuning.cc:233-235]

```c
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
```

`ncclTuningGetChannels`In`tuning_int.h`, the logic is to interpolate between`minChannels`and`maxChannels`based on message size and algorithm type. The channel count directly affects bandwidth: more channels mean higher parallelism, but the startup overhead of each channel is also greater.

**Step 6: CTA Policy bias (NVLS priority).**If the user has set`NCCL_CTA_POLICY_EFFICIENCY`, and the current operation is AllGather/ReduceScatter and the buffer is already registered, NCCL will try to change the result to NVLS:

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

**Why distinguish the error codes?**If the user has set`NCCL_ALGO=ring`but the current platform does not support ring (for example, some special topologies), then that is**a user configuration error**（`ncclInvalidUsage`); if the user has not set any environment variables but still cannot select an algorithm, then that is**an NCCL internal bug**（`ncclInternalError`). This distinction is crucial for troubleshooting.

## Main Decision Flowchart

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

# 2. cost_model.cc: Model Registry and Switch Matrix

## Intuitive model

`cost_model.cc`is tuning's**general ledger**. It maintains a`modelMap`table, where each row corresponds to an (algo, proto) combination and records "who this combination's initialization function is, who its simulation function is, and for which functions it is enabled." At the same time, it is responsible for parsing the user environment variable`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SYM_KERNEL`, translating the user's intent into a`enabled[i][f]`switch matrix.

Without this table, every time a new algorithm is added, the main tuning flow would have to be modified again, and the code would rot into a mess.**Table-driven**makes "adding an algorithm" become "adding a row."

## Data structure: modelMap and switch matrix

`modelMap`is a static array, and each element is`ncclTuningModelEntry_t`：

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

Each entry has four fields:`init`(initialization, computing latency/bandwidth and storing them in comm),`model`(simulation, computing the final timeUs based on message size),`finalize`(cleanup),`enabled[5]`(whether the five functions Broadcast/Reduce/AllGather/ReduceScatter/AllReduce are enabled).

Note`enabled`The order of the array is annotated at L234:`Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce`. This order must match the`ncclFunc_t`enum, otherwise things will get mixed up.

> **[Design Inference & Architectural Trade-offs]**
> **Why should init and sim be separated?**Because the things computed in init (latency, bandwidth)**only depend on the static properties of comm**(topology, number of ranks, compCap), and are independent of the specific message size. Within a single communication, tuning may be called multiple times in succession (for example, when a group has multiple ops), init runs only once, while sim runs every time. This is a typical "precompute + fast lookup" optimization.

## Step-by-Step: Environment Variable Parsing and Switch Matrix Construction

**Step 1: All enabled by default, LL128 is special.** `ncclTuningCostModelInit`Initially all protos are set to 1 (enabled), but LL128 is set to 2:

[FACT:src/tuning/cost_model.cc:313-323]

```c
  for (int f = 0; f minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                          comm->graphs[algo].typeIntra, comm->nRanks, f, algo, comm->minDriverVersion)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
```

**Step 2: Parse user environment variables.**If the user sets`NCCL_ALGO`or`NCCL_SYM_KERNEL`, first clear algo and symKernel entirely (because the user has specified a whitelist):

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

Note that proto is not cleared—because proto's default value is 1/2, when the user sets`NCCL_PROTO=LL`,`parseList`will set LL to 1 and others to 0 (because of the`unset`logic). This asymmetry is intentional: algo is fully enabled by default but must be narrowed after the user specifies it, while proto's narrowing is handled internally by`parseList`.

**Step 3: The syntax of parseList.**This function supports fairly complex syntax, and the comments give examples:

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

`^`The prefix means "negation":

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

So`NCCL_PROTO="^LL128;allreduce:LL128"`means: globally disable LL128, but enable LL128 as an exception for AllReduce.

**Step 4: Merge the enabled matrix.**Finally, iterate over all models and AND`model->enabled[f]`with the user switches:

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

The logic is:**Only when the user has set a forced configuration for some function is the user configuration used to override the model default value**. If the user has not set it,`forced[f] == 0`, directly`continue`, retaining the model's own`enabled`. This is the priority of "user explicit specification > model default".

## Unified entry point for model simulation

All models are ultimately invoked through`ncclTuningCostModelSimModel`:

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

Three layers of filtering:**id out of bounds → model disabled → model returns non-positive time**, if any layer fails, go to`not_valid`, set`timeUs`to`NCCL_TUNING_IGNORE`(a negative sentinel),`valid = 0`. When the caller sees`valid == 0`, it will not attach it to the candidate linked list.

## Design considerations

`modelMap`There is a key warning in the comments of

[FACT:src/tuning/cost_model.cc:229]

```c
// IMPORTANT: this table need must be consistent with the algRegistry in src/config/algorithm_registry.cc
```

> **[Design Inference & Architectural Trade-offs]**
> This means that the`modelMap`of**index order**must strictly match the algorithm registration order in`algorithm_registry.cc`. If someone inserts a new algorithm into the registry but forgets to change`modelMap`, all ids will be misaligned, and tuning will select a completely wrong algorithm.**This is the classic trap of table-driven design: implicit contracts.**A more robust approach would be to use enum names as keys instead of indices, but that would sacrifice a bit of compile-time optimization.

---

# III. ring.cc: Cost Estimation for the Ring Algorithm

## Intuitive model

The Ring algorithm arranges N ranks into a ring, and data is passed around the ring circle by circle. Its cost model must answer two questions:**How much data is passed per step (bandwidth)**、**How many steps are needed in total (latency)**。

The intuition of Ring is "**pipeline**": imagine N people standing in a circle passing a bucket, and each person pours a little water into it before passing it to the next person. After the bucket goes around once, everyone's water is mixed evenly. The faster the bucket goes around (higher bandwidth) and the smaller the circle (fewer steps), the faster the whole thing is.

## Data structure: latency/bandwidth table

The Ring model does not introduce new structures; it writes the estimation results into`comm->tuningContext.generalLatencies[c][algo][proto]`and`generalBandwidths[c][algo][proto]`. These two are three-dimensional arrays: function × algorithm × protocol.

During initialization, all are first set to -1.0 (sentinel, meaning "not computed yet"):

[FACT:src/tuning/ring.cc:31-33]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
```

The -1.0 sentinel is checked during the sim stage:

[FACT:src/tuning/ring.cc:94-97]

```c
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ncclSuccess;
  }
```

**Why use -1.0 instead of 0?**Because 0 is a legal bandwidth value (though physically impossible), while -1.0 clearly indicates "uninitialized". Using`==`for floating-point comparison is safe here, because -1.0 is exactly representable.

## Step-by-Step: Ring Bandwidth Estimation

**Step 1: Determine whether to use intra or inter bandwidth.**Single node (nNodes==1) uses intra, multi-node uses inter:

[FACT:src/tuning/ring.cc:34-37]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float bw = (comm->nNodes == 1 || (comm->nNodes minCompCap graphs[algo].bwIntra :
                                                                                      comm->graphs[algo].bwInter;
    float busBw = bw * comm->graphs[algo].nChannels;
```

`nSteps`is the number of steps required by the algorithm; for Ring, AllReduce is`2*(nRanks-1)`, others are`nRanks-1`。`busBw`is the "bus bandwidth" = single-link bandwidth × number of channels.

**Step 2: Apply protocol discount.**The LL protocol only uses half the bandwidth (because of LL's flag overhead), while LL128 uses 92% (120/128):

[FACT:src/tuning/ring.cc:38-42]

```c
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);
```

`0.92 = 120/128`This is because in LL128, 8 out of every 128 bytes are flags, leaving only 120 bytes of payload. This number comes directly from the protocol design.

**Step 3: Calculate effective bandwidth.**Note that here it is multiplied by`nRanks / nSteps`：

[FACT:src/tuning/ring.cc:44-46]

```c
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;
```

**Why multiply by`nRanks / nSteps`？**This is the core characteristic of the Ring algorithm: the amount of data each rank actually moves is`nBytes * nSteps / nRanks`(because the data must go around the ring multiple times). So "effective bandwidth" = bus bandwidth × nRanks / nSteps. For AllReduce, nSteps = 2(nRanks-1), so effective bandwidth ≈ busBw/2.

**Step 4: Calculate latency.**Latency is split into two parts: intra and inter:

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

Note the special handling at L57-58: when`maxLocalRanks == 1`(each node has only 1 rank), Ring's inter-node latency uses**Tree's NET latency**. The comment says this is to "preserve the pre-refactor model" — that is, a deliberate "quirk" kept to maintain consistency with pre-refactor behavior.**This kind of historical baggage is very common in mature systems. When reading source code and seeing the word "preserve," be especially careful; it often means there is a compatibility constraint here that cannot be changed.**

**Step 5: Accumulate by function type.**The latency models for Reduce/Broadcast and AllReduce/AllGather/ReduceScatter are different:

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

`sameChannels`is a topology property indicating "whether the intra and inter steps on the ring use the same set of channels." If they are different, latency must be multiplied by`nSteps`(each step must wait).`netOverhead`is the network post overhead. For the Simple protocol, multiply by 3 (because Simple has three network round trips: send, recv, ack).

## Production pitfall avoidance: the plateau effect of Ring/Simple

`ncclTuningRingModelSim`There is a section of code specifically handling "plateau":

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
> **What is plateau?**In Ring/Simple, when the message becomes large enough, latency no longer grows linearly with message size but instead "gets stuck" on a plateau — because at this point the bottleneck shifts from "startup overhead" to "bandwidth," and bandwidth is already saturated. This phenomenon is especially obvious on Blackwell NVLink (because NVLink bandwidth is so high that latency accounts for a larger proportion). The code uses`plateauFactor`(1.4 or 1.9) multiplied onto the latency to simulate this "latency amplification" effect.

`bytesPerRankPerChannel >= 64`is the trigger condition: each rank must transfer at least 64 bytes per channel, otherwise the plateau does not hold. This 64 bytes comes from the LL protocol's flag size.

**Pitfall scenario**: If you run a 1MB AllReduce on Blackwell and find that the actual latency is 40% higher than the model predicts, do not assume it is a bug — this is the plateau effect, and the model has already accounted for it. If you manually reduce`plateauFactor`, the model will underestimate latency, leading to the wrong algorithm being selected.

---

# IV. tree.cc and nvls.cc: Cost estimation for Tree and NVLS

## Intuitive model

**The Tree algorithm**is "**tree broadcast**": the root node distributes data to child nodes, and child nodes then distribute it to grandchild nodes. Its advantage is**fewer steps**(log N instead of N), making it suitable for small messages; its disadvantage is**low bandwidth utilization**(each non-leaf node must forward, so the actual effective bandwidth is only half).

**NVLS**(NVLink SHARP) is "**hardware multicast**": the switch directly copies data to multiple GPUs without software forwarding. Its advantages are**high bandwidth and low latency**, but it requires specific hardware (Hopper or above) and specific configuration.

## Tree model: only serves AllReduce

The Tree model has a hard restriction —**it is only enabled for AllReduce**：

[FACT:src/tuning/tree.cc:21-27]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
```

> **[Design Inference & Architectural Trade-offs]**
> **Why?**Because NCCL's Tree implementation only supports AllReduce (other collective operations do not have Tree versions). This is an implementation constraint, not a theoretical limitation.`enabled[c] = 0`is a "hard disable," more thorough than`generalBandwidths = -1`— the former directly makes`ncclTuningCostModelSimModel`return at L480`not_valid`, while the latter only checks inside the sim function.

**Tree bandwidth estimation**：

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
> Note that the LL protocol's discount factor is`1/3.8`, which is even more aggressive than Ring's`0.5`.**Why is Tree's LL efficiency lower?**Because each intermediate node in Tree must both receive and send, and LL's flag overhead is amplified under bidirectional traffic.`1/3.8`This number comes from measurement.

**Tree latency estimation**：

[FACT:src/tuning/tree.cc:55-58]

```c
    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
```

`2 *`is because AllReduce = ReduceScatter + AllGather, two passes.`(nRanks/nNodes - 1)`is the intra-node step count (the number of ranks within each node minus one),`log2i(nNodes)`is the inter-node step count (the height of the tree).

**Tree's correction factor**: The Tree model is multiplied by a factor during the sim stage`treeCorrectionFactor`：

[FACT:src/tuning/tree.cc:75-79]

```c
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize proto][logSize];
```

`treeCorrectionFactor`is a 3×24 table:

[FACT:src/tuning/cost_model.cc:223-227]

```c
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};
```

`logSize = log2(nBytes >> 6)`, i.e., the message size is taken as log2 in units of 64 bytes. The table indices 0-23 correspond to 64B to 64B×2^23 ≈ 512MB.**This table is a measured "Tree efficiency curve"**: For small messages, efficiency is 1.0 (latency-dominated); for medium messages, efficiency drops to 0.4-0.5 (bandwidth not saturated); for large messages, it returns to 1.0 (bandwidth saturated). This "mid-range dip" is an inherent characteristic of the Tree algorithm.

## NVLS model: The cost of hardware multicast

The NVLS model first checks whether the hardware supports it:

[FACT:src/tuning/nvls.cc:19-24]

```c
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!ncclNvlsTransportEnabled(comm)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

Then there is a series of hard constraints: only the Simple protocol is supported, NVLSTree is not supported on a single node, and multi-node NVLS requires CollNet:

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

**NVLS bandwidth estimation**uses an efficiency factor:

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
> Hopper is 0.85, while Blackwell actually drops to 0.74.**Why is the efficiency lower on the newer generation of hardware?**Because Blackwell's NVLink bandwidth is higher, but the NVLS switch processing capability has not increased proportionally, resulting in a relative efficiency decrease. This number is measured, not theoretical.

In the bandwidth calculation there is a`(nChannels - 1) / nChannels`factor:

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

`(nChannels - 1) / nChannels`because NVLS needs to reserve one channel for synchronization.`(ppn - 1) / ppn`is the additional overhead of AllGather/ReduceScatter (each rank has to wait for the previous rank's data).

## Production pitfalls: NVLS hard constraints

The NVLS model also has a runtime check during the sim stage:

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

`NCCL_MAX_NVLS_ARITY`is the maximum number of GPUs that an NVLS multicast group can accommodate. If this number is exceeded, NVLS is unavailable.**Pitfall scenario**: Running AllGather in a 16-GPU NVLink domain, if`NCCL_MAX_NVLS_ARITY`is 8, NVLS will be disabled and tuning will fall back to Ring. If you don't know this limitation, you'll wonder, "NVLS is clearly supported by the hardware, why isn't it being used?"

---

# V. Symmetric kernel fallback and error recovery chain

## Intuitive model

Symmetric kernel is a new NCCL feature: when all ranks' buffers are registered to symmetric memory, the kernel can use more efficient instructions to access peer memory. But**if the buffer is not registered, or the platform does not support it, it must fall back to the normal kernel**. This fallback logic is the most convoluted part of tuning.

## Step-by-Step: Fallback decision

The fallback logic is in`tuning.cc:258-298`. Let's break it down.

**Step 1: Determine whether fallback is needed.**Entry conditions:

[FACT:src/tuning/tuning.cc:258-263]

At this point, the decision chain of the tuning module is clear: it receives the topology graph and communication parameters, and through cost models and algorithm estimation, outputs the optimal (algorithm, protocol, channel, warp) combination within microseconds. But selection is only the beginning—how is this decision result used downstream? In the next chapter, we will enter the main trunk of src/enqueue/enqueue.cc and see how a single ncclAllReduce call goes through parameter validation, algorithm/protocol determination, and channel partitioning, ultimately generating the ncclInfo and ncclTaskColl structures. This is the key chapter in the book where we switch from the "user perspective" to the "engine perspective." You will discover what a collective communication call is translated into on the host side, and the boundary between it and subsequent kernel launch.
