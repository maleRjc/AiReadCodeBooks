# Capítulo 5: Seleção de algoritmos e protocolos: como o módulo tuning decide o caminho de comunicação

No capítulo anterior, dissecamos a capacidade de percepção de topologia do NCCL: desde a enumeração de dispositivos em src/graph/topo.cc para construir o grafo de topologia, passando pela busca do caminho ótimo em src/graph/search.cc, até a concretização dos resultados de busca em topologias de algoritmos Ring e Tree em rings.cc e trees.cc. Mas o grafo de topologia só responde "por quais caminhos os dados podem passar"; ele não responde "por qual caminho esta comunicação deve passar". Em uma mesma máquina, um AllReduce de 4KB e um AllReduce de 400MB podem ter soluções ótimas completamente diferentes: o primeiro prioriza latência, o segundo prioriza largura de banda; o primeiro pode escolher Tree/LL, o segundo pode escolher Ring/Simple ou NVLS. O módulo tuning é aquele que "bate o martelo". Suas entradas são o tamanho da mensagem, o número de ranks, o grafo de topologia (produto do capítulo anterior) e as variáveis de ambiente do usuário; sua saída é um ncclTuningResult_t, que contém qual algoritmo (algo) usar, qual protocolo (proto), quantos channels abrir e quantas warps usar. Neste capítulo, seguimos a ordem "agendamento geral → modelo de custo → estimativas de cada algoritmo → decisão final" para dissecar o diretório src/tuning. A questão central é apenas uma: como o NCCL, entre dezenas de combinações de (algoritmo, protocolo), usando um conjunto de modelos matemáticos puramente em CPU, seleciona o mais rápido em tempo de microssegundos?

# I. tuning.cc: agendamento geral e espinha dorsal da decisão

## Modelo intuitivo

Imagine o módulo tuning como uma empresa de**mudanças**. O cliente (uma comunicação coletiva) chega e diz "quero mover 100MB de carga, de 8 armazéns para 8 armazéns". O despachante (`ncclTuningCompute`) não vai realmente tentar mover para testar, mas pega uma**tabela de preços**(modelo de custo), estima um "tempo previsto" para cada opção (Ring/LL, Tree/Simple, NVLS/Simple...) e escolhe a cotação mais curta para o cliente.

Sem esse despachante, o NCCL só poderia fixar "AllReduce sempre usa Ring", o que seria dominado por Tree em cenários de mensagens pequenas e por NVLS em cenários de NVLink em larga escala.**O custo é o desempenho cair pela metade ou pior em cenários específicos.**

## Estruturas de dados e layout de memória

O portador da decisão é`ncclTuningResult_t`, e o conjunto de candidatos é`ncclTuningResultList_t`(uma lista encadeada simples). O nó da lista é definido em`tuning_int.h`, mas a lógica de push está em`tuning.cc`:

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
> Note que aqui é**inserção no início**: cada candidato válido calculado é inserido no início da lista. Isso significa que a ordem da lista e a ordem dos ids são**inversas**. Por que usar lista encadeada em vez de array? Porque a quantidade de candidatos é determinada em tempo de compilação por`NCCL_TUNING_COUNT`, mas os candidatos realmente válidos são dinâmicos (afetados por`tuningMask`, capacidades da plataforma, variáveis de ambiente do usuário), e a lista encadeada permite "anexar apenas os válidos", evitando verificar repetidamente`valid`durante a iteração. O custo é que a cada decisão é preciso`ncclCalloc`uma vez, mas o tuning ocorre no caminho de enfileiramento e com baixa frequência, então esse custo de alocação é aceitável.

`ncclTuningResult_t`Os dois campos mais críticos em`timeUs`(tempo estimado, microssegundos) e`selectionTimeUs`(tempo usado para seleção, pode ser sobrescrito por plugins tuner). A lógica de seleção considera apenas o último:

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

Há um detalhe aqui:`bestTuning->timeUs`é primeiro definido como`FLT_MAX`, e então percorre. Se a lista encadeada estiver vazia (todos os candidatos inválidos),`bestTuning`manterá o valor inicial de`NCCL_TUNING_RESULT_INIT`, algo/proto serão`UNDEF`. Este "resultado vazio" será tratado especialmente pelo chamador — veja o ramo de erro mais adiante.

## Passo a Passo: Fluxo de decisão de um AllReduce

Suponha que a aplicação chame`ncclAllReduce`, mensagem de 1MB, 8 ranks em máquina única NVLink. Vamos acompanhar`ncclTuningCompute`até o fim.

**Passo 0: curto-circuito de rank único.**Se`nRanks <= 1`, não há necessidade de comunicação, retorna diretamente Ring/Simple, número de channels definido como 0:

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

Copiar`NCCL_TUNING_IGNORE`Aqui

**é um valor sentinela, indicando "esta combinação não foi calculada/não se aplica". O plugin pode alterar apenas as células que lhe interessam, mantendo as outras como IGNORE, e o NCCL as ignorará.**Passo 4: selecionar o melhor.`ncclTuningSelectBestTuning`chama`selectionTimeUs`, percorre a lista encadeada e pega o menor

**.**Passo 5: calcular número de channels.

[FACT:src/tuning/tuning.cc:233-235]

```c
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
```

`ncclTuningGetChannels`Copiar`tuning_int.h`Em`minChannels`, a lógica é interpolar entre`maxChannels`e

**com base no tamanho da mensagem e tipo de algoritmo. O número de channels afeta diretamente a largura de banda: mais channels, maior paralelismo, mas maior custo de inicialização por channel.**Passo 6: viés de CTA Policy (prioridade NVLS).`NCCL_CTA_POLICY_EFFICIENCY`Se o usuário definiu

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

**Por que distinguir códigos de erro?**Se o usuário definiu`NCCL_ALGO=ring`mas a plataforma atual não suporta ring (por exemplo, algumas topologias especiais), é**erro de configuração do usuário**（`ncclInvalidUsage`); se o usuário não definiu nenhuma variável de ambiente mas não consegue selecionar algoritmo, é**bug interno do NCCL**（`ncclInternalError`). Esta distinção é crucial para troubleshooting.

## Fluxograma do tronco de decisão

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

# II. cost_model.cc: registro de modelos e matriz de switches

## Modelo intuitivo

`cost_model.cc`é o**livro-razão geral**do tuning. Ele mantém uma tabela`modelMap`, cada linha corresponde a uma combinação (algo, proto), registrando "quem é a função de inicialização desta combinação, quem é a função de simulação, para quais funções está habilitada". Também é responsável por analisar a variável de ambiente do usuário`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SYM_KERNEL`, traduzindo a intenção do usuário em uma matriz de switches`enabled[i][f]`.

Sem esta tabela, cada novo algoritmo exigiria alterar o fluxo principal de tuning, e o código viraria uma bagunça.**Orientado a tabela**torna "adicionar algoritmo" em "adicionar uma linha".

## Estrutura de dados: modelMap e matriz de switches

`modelMap`é um array estático, cada elemento é`ncclTuningModelEntry_t`：

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

Cada entry tem quatro campos:`init`(inicialização, calcula latency/bandwidth e armazena em comm),`model`(simulação, calcula timeUs final com base no tamanho da mensagem),`finalize`(limpeza),`enabled[5]`(se habilitado para as cinco funções Broadcast/Reduce/AllGather/ReduceScatter/AllReduce).

Nota`enabled`A ordem do array está comentada na L234:`Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce`. Esta ordem deve ser consistente com`ncclFunc_t`a enumeração, caso contrário haverá confusão.

> **[Design Inference & Architectural Trade-offs]**
> **Por que init e sim devem ser separados?**Porque o que é calculado em init (latency, bandwidth)**depende apenas das propriedades estáticas de comm**(topologia, número de ranks, compCap), e não do tamanho específico da mensagem. Numa comunicação podem ocorrer múltiplas chamadas consecutivas de tuning (por exemplo, vários ops num group), init corre apenas uma vez, sim corre a cada vez. Esta é uma otimização típica de «pré-cálculo + consulta rápida».

## Step-by-Step: Análise de variáveis de ambiente e construção da matriz de switches

**Passo 1: Por defeito tudo ativado, LL128 é especial.** `ncclTuningCostModelInit`Inicialmente todos os proto são definidos como 1 (ativado), mas LL128 é definido como 2:

[FACT:src/tuning/cost_model.cc:313-323]

```c
  for (int f = 0; f minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                          comm->graphs[algo].typeIntra, comm->nRanks, f, algo, comm->minDriverVersion)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
```

**Passo 2: Analisar variáveis de ambiente do utilizador.**Se o utilizador definiu`NCCL_ALGO`ou`NCCL_SYM_KERNEL`, primeiro limpar algo e symKernel para zero (porque o utilizador especificou uma whitelist):

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

Nota: proto não é limpo — porque o valor por defeito de proto é 1/2, quando o utilizador define`NCCL_PROTO=LL`,`parseList`define LL como 1 e os outros como 0 (devido à lógica de`unset`). Esta assimetria é intencional: algo está todo ativado por defeito mas deve ser restringido após especificação do utilizador; a restrição de proto é tratada internamente por`parseList`.

**Passo 3: Sintaxe de parseList.**Esta função suporta uma sintaxe bastante complexa, com exemplos nos comentários:

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

`^`O prefixo indica «negação»:

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

Portanto`NCCL_PROTO="^LL128;allreduce:LL128"`significa: desativar LL128 globalmente, mas ativar LL128 como exceção para AllReduce.

**Passo 4: Combinar a matriz enabled.**Finalmente percorrer todos os models, fazendo a operação AND entre`model->enabled[f]`e os switches do utilizador:

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

A lógica é:**Só quando o utilizador definiu uma configuração forced para uma função, é que a configuração do utilizador substitui o valor por defeito do modelo**. Se o utilizador não definiu,`forced[f] == 0`, diretamente`continue`, mantendo o`enabled`do próprio modelo. Esta é a prioridade de «especificação explícita do utilizador > predefinição do modelo».

## Entrada unificada para simulação de modelos

Todos os modelos são finalmente invocados através de`ncclTuningCostModelSimModel`:

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

Três camadas de filtragem:**id fora dos limites → modelo desativado → modelo retorna tempo não positivo**, se qualquer camada falhar vai para`not_valid`, definindo`timeUs`como`NCCL_TUNING_IGNORE`(um sentinela negativo),`valid = 0`. O chamador ao ver`valid == 0`não o colocará na lista de candidatos.

## Reflexão sobre design

`modelMap`Nos comentários de

[FACT:src/tuning/cost_model.cc:229]

```c
// IMPORTANT: this table need must be consistent with the algRegistry in src/config/algorithm_registry.cc
```

> **[Design Inference & Architectural Trade-offs]**
> 〔Inferência de design e trade-offs arquiteturais〕`modelMap`Isto significa que**a**ordem dos índices`algorithm_registry.cc`deve ser estritamente consistente com a ordem de registo dos algoritmos em`modelMap`. Se alguém inserir um novo algoritmo no registry mas esquecer de alterar**, todos os ids ficarão desalinhados e o tuning selecionará um algoritmo completamente errado.**Esta é a armadilha clássica do design orientado a tabelas: contrato implícito.

---

# Uma abordagem mais robusta seria usar nomes de enumeração como key em vez de índices, mas isso sacrificaria um pouco de otimização em tempo de compilação.

## Três, ring.cc: Estimativa de custo do algoritmo Ring

Modelo intuitivo**O algoritmo Ring organiza N ranks num anel, e os dados são transmitidos ao longo do anel volta após volta. O seu modelo de custo deve responder a duas questões:**、**Quanto dado é transmitido em cada passo (bandwidth)**。

Quantos passos são necessários no total (latency)**A intuição do Ring é «**pipeline

## »: imagine N pessoas em círculo a passar um balde de água, cada pessoa ao receber o balde deita um pouco de água e passa ao próximo. Quando o balde dá uma volta completa, a água de todos está misturada. Quanto mais rápido o balde roda (bandwidth alta), menor o círculo (menos passos), mais rápido o todo.

Estrutura de dados: tabela latency/bandwidth`comm->tuningContext.generalLatencies[c][algo][proto]`O modelo Ring não introduz novas estruturas, escreve os resultados da estimativa em`generalBandwidths[c][algo][proto]`e

. Estes dois são arrays tridimensionais: função × algoritmo × protocolo.

[FACT:src/tuning/ring.cc:31-33]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
```

Copiar

[FACT:src/tuning/ring.cc:94-97]

```c
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ncclSuccess;
  }
```

**Copiar**Por que usar -1.0 em vez de 0?`==`Porque 0 é um valor de bandwidth legítimo (embora fisicamente impossível), enquanto -1.0 indica claramente «não inicializado». A comparação de floats com

## é segura aqui, porque -1.0 é exatamente representável.

**Step-by-Step: Estimativa de bandwidth do Ring**Passo 1: Determinar se usar bandwidth intra ou inter.

[FACT:src/tuning/ring.cc:34-37]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float bw = (comm->nNodes == 1 || (comm->nNodes minCompCap graphs[algo].bwIntra :
                                                                                      comm->graphs[algo].bwInter;
    float busBw = bw * comm->graphs[algo].nChannels;
```

`nSteps`Copiar`2*(nRanks-1)`é o número de passos necessários para o algoritmo; para Ring, AllReduce é`nRanks-1`。`busBw`, os outros são

**é a «bandwidth de barramento» = bandwidth de link único × número de channels.**O protocolo LL usa apenas metade da largura de banda (devido ao overhead do flag do LL), o LL128 usa 92% (120/128):

[FACT:src/tuning/ring.cc:38-42]

```c
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);
```

`0.92 = 120/128`Isso ocorre porque no LL128, a cada 128 bytes, 8 bytes são flag, e o payload útil é de apenas 120 bytes. Esse número vem diretamente do design do protocolo.

**Passo 3: calcular a largura de banda efetiva.**Observe que aqui foi multiplicado por`nRanks / nSteps`：

[FACT:src/tuning/ring.cc:44-46]

```c
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;
```

**Por que multiplicar`nRanks / nSteps`？**Esta é uma característica central do algoritmo Ring: a quantidade de dados que cada rank realmente transporta é`nBytes * nSteps / nRanks`(porque os dados precisam dar várias voltas no anel). Portanto, "largura de banda efetiva" = largura de banda do barramento × nRanks / nSteps. Para AllReduce, nSteps = 2(nRanks-1), então a largura de banda efetiva ≈ busBw/2.

**Passo 4: calcular a latência.**A latência é dividida em duas partes: intra e inter:

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

Observe o tratamento especial nas linhas L57-58: quando`maxLocalRanks == 1`(cada nó tem apenas 1 rank), a latência inter-node do Ring usa**a latência NET da Tree**. O comentário diz que isso é "preserve the pre-refactor model" — ou seja, uma "peculiaridade" deliberadamente mantida para preservar a consistência com o comportamento anterior à refatoração.**Esse tipo de fardo histórico é muito comum em sistemas maduros. Ao ler o código-fonte, tenha cuidado especial ao ver a palavra "preserve", pois ela geralmente indica que há uma restrição de compatibilidade que não pode ser alterada.**

**Passo 5: acumular por tipo de função.**Os modelos de latência de Reduce/Broadcast e AllReduce/AllGather/ReduceScatter são diferentes:

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

`sameChannels`É uma propriedade de topologia que indica "se os passos intra e inter no anel usam o mesmo conjunto de channels". Se forem diferentes, a latência deve ser multiplicada por`nSteps`(é preciso esperar em cada passo).`netOverhead`É o custo de post de rede; o protocolo Simple deve ser multiplicado por 3 (porque o Simple tem três idas e voltas de rede: send, recv, ack).

## Evitando armadilhas em produção: o efeito plateau do Ring/Simple

`ncclTuningRingModelSim`Há um trecho de código dedicado a lidar com o "plateau":

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
> **O que é plateau?**No Ring/Simple, quando a mensagem atinge um certo tamanho, a latência deixa de crescer linearmente com a mensagem e "trava" em um patamar — porque nesse momento o gargalo muda de "overhead de inicialização" para "largura de banda", e a largura de banda já está saturada. Esse fenômeno é especialmente evidente no Blackwell NVLink (porque a largura de banda do NVLink é muito alta, e a proporção da latência é maior). O código usa`plateauFactor`(1.4 ou 1.9) multiplicado pela latência para simular esse efeito de "latência amplificada".

`bytesPerRankPerChannel >= 64`É a condição de disparo: cada rank deve transmitir pelo menos 64 bytes por channel, caso contrário o plateau não se aplica. Esses 64 bytes vêm do tamanho do flag do protocolo LL.

**Cenário de armadilha**: Se você executar um AllReduce de 1MB no Blackwell e descobrir que a latência real é 40% maior do que a prevista pelo modelo, não pense que é um bug — isso é o efeito plateau, e o modelo já o contabilizou. Se você reduzir manualmente`plateauFactor`, o modelo subestimará a latência, levando à escolha errada de algoritmo.

---

# IV. tree.cc e nvls.cc: estimativa de custo de Tree e NVLS

## Modelo intuitivo

**O algoritmo Tree**é uma "**transmissão em árvore**": o nó raiz distribui os dados para os nós filhos, e os nós filhos os distribuem para os netos. Sua vantagem é o**baixo número de passos**(log N em vez de N), adequado para mensagens pequenas; a desvantagem é a**baixa utilização da largura de banda**(cada nó não-folha precisa encaminhar, e a largura de banda efetiva real é apenas metade).

**NVLS**(NVLink SHARP) é a "**multicast por hardware**": o switch copia diretamente os dados para múltiplas GPUs, sem necessidade de encaminhamento por software. Sua vantagem é**alta largura de banda e baixa latência**, mas requer hardware específico (Hopper ou superior) e configuração específica.

## Modelo Tree: serve apenas AllReduce

O modelo Tree tem uma restrição rígida —**só é habilitado para AllReduce**：

[FACT:src/tuning/tree.cc:21-27]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
```

> **[Design Inference & Architectural Trade-offs]**
> **Por quê?**Porque a implementação Tree do NCCL só suporta AllReduce (as outras operações coletivas não têm versão Tree). Esta é uma restrição de implementação, não uma limitação teórica.`enabled[c] = 0`É uma "desabilitação rígida", mais radical que`generalBandwidths = -1`— a primeira faz com que`ncclTuningCostModelSimModel`retorne em L480, enquanto a segunda só é verificada dentro da função sim.`not_valid`Estimativa de largura de banda do Tree

**copiar**：

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
> , mais agressivo que o`1/3.8`do Ring.`0.5`Por que a eficiência do LL no Tree é menor?**Porque cada nó intermediário da Tree precisa tanto receber quanto enviar, e o overhead do flag do LL é amplificado sob tráfego bidirecional.**Esse número vem de medições reais.`1/3.8`Estimativa de latência do Tree

**copiar**：

[FACT:src/tuning/tree.cc:55-58]

```c
    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
```

`2 *`É o número de passos intra-nó (o número de ranks em cada nó menos um),`(nRanks/nNodes - 1)`é o número de passos inter-nó (a altura da árvore).`log2i(nNodes)`Fator de correção do Tree

**Tree 的修正因子**：O modelo Tree multiplica no estágio sim por um`treeCorrectionFactor`：

[FACT:src/tuning/tree.cc:75-79]

```c
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize proto][logSize];
```

`treeCorrectionFactor`é uma tabela 3×24:

[FACT:src/tuning/cost_model.cc:223-227]

```c
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};
```

`logSize = log2(nBytes >> 6)`, ou seja, o tamanho da mensagem é tomado em log2 com unidade de 64 bytes. Os índices 0-23 da tabela correspondem a 64B até 64B×2^23 ≈ 512MB.**Esta tabela é a "curva de eficiência do Tree" medida empiricamente**：Para mensagens pequenas, a eficiência é 1.0 (dominada pela latência), para mensagens médias a eficiência cai para 0.4-0.5 (largura de banda não saturada), e para mensagens grandes volta a 1.0 (largura de banda saturada). Esta "depressão intermediária" é uma característica inerente do algoritmo Tree.

## Modelo NVLS: o custo do multicast por hardware

O modelo NVLS primeiro verifica se o hardware suporta:

[FACT:src/tuning/nvls.cc:19-24]

```c
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!ncclNvlsTransportEnabled(comm)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

Em seguida, há uma série de restrições rígidas: suporta apenas o protocolo Simple, não suporta NVLSTree em máquina única, e NVLS multi-máquina requer CollNet:

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

**Estimativa de largura de banda do NVLS**Utiliza um fator de eficiência:

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
> Hopper é 0.85, enquanto Blackwell na verdade cai para 0.74.**Por que o hardware de nova geração tem eficiência menor?**Porque a largura de banda do NVLink do Blackwell é maior, mas a capacidade de processamento do switch NVLS não aumentou proporcionalmente, resultando em queda da eficiência relativa. Este número é medido empiricamente, não é um valor teórico.

No cálculo de largura de banda há um fator`(nChannels - 1) / nChannels`:

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

`(nChannels - 1) / nChannels`porque o NVLS precisa reservar um channel para sincronização.`(ppn - 1) / ppn`é a sobrecarga adicional de AllGather/ReduceScatter (cada rank precisa esperar pelos dados do rank anterior).

## Evitando armadilhas em produção: restrições rígidas do NVLS

O modelo NVLS ainda tem uma camada de verificação em tempo de execução no estágio sim:

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

`NCCL_MAX_NVLS_ARITY`é o número máximo de GPUs que um grupo multicast NVLS pode acomodar. Se exceder este número, o NVLS fica indisponível.**Cenário de armadilha**：Executar AllGather em um domínio NVLink de 16 placas, se`NCCL_MAX_NVLS_ARITY`for 8, o NVLS será desabilitado e o tuning fará fallback para Ring. Se você não conhece essa limitação, vai pensar "por que o NVLS não é usado se o hardware claramente suporta".

---

# V. Fallback de kernel simétrico e cadeia de recuperação de erros

## Modelo intuitivo

O kernel simétrico (symmetric kernel) é uma nova funcionalidade do NCCL: quando os buffers de todos os ranks são registrados em memória simétrica, o kernel pode acessar a memória do par com instruções mais eficientes. Mas**se o buffer não estiver registrado, ou a plataforma não suportar, é obrigatório fazer fallback para o kernel normal**. Esta lógica de fallback é a parte mais complicada do tuning.

## Step-by-Step: decisão de fallback

A lógica de fallback está em`tuning.cc:258-298`. Vamos analisar por partes.

**Passo 1: determinar se é necessário fallback.**Condição de entrada:

[FACT:src/tuning/tuning.cc:258-263]

Até aqui, a cadeia de decisões do módulo tuning já está clara: ele recebe o grafo de topologia e os parâmetros de comunicação, e através de modelos de custo e estimativas de algoritmos, produz em nível de microssegundos a combinação ótima de (algoritmo, protocolo, channel, warp). Mas a seleção é apenas o começo — como este resultado de decisão é usado downstream? No próximo capítulo entraremos no tronco de src/enqueue/enqueue.cc, para ver como uma chamada ncclAllReduce passa por validação de parâmetros, determinação de algoritmo/protocolo, divisão de channel, e finalmente gera as estruturas ncclInfo e ncclTaskColl. Este é o capítulo chave do livro onde se muda da "perspectiva do usuário" para a "perspectiva da engine", você descobrirá em que uma chamada de comunicação coletiva é traduzida no lado host, e qual é a fronteira entre isso e o lançamento subsequente do kernel.
