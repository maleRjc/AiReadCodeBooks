# Capítulo 24: Evolução arquitetural e direções futuras: da comunicação estática à comunicação programável

No capítulo anterior, vimos como a comunidade constrói um ecossistema periférico em torno do núcleo do NCCL: bindings Python, bindings Rust, comunicação de paralelismo de especialistas, primitivas de ultra-banda larga, checkpoint de comunicação. Esses projetos reutilizam a API estável do NCCL, mas suas demandas já ultrapassam o escopo da comunicação coletiva tradicional — o paralelismo de especialistas requer envio/recepção ponto a ponto de granularidade fina, o checkpoint requer pausar/retomar o estado de comunicação, as primitivas de ultra-banda larga requerem contornar as operações coletivas padrão para operar diretamente a rede. Essas demandas apontam para a mesma questão: o modelo de operações coletivas fixas do NCCL está sendo rompido por necessidades de comunicação mais flexíveis. Neste capítulo, não olharemos para um único módulo, mas sim, partindo dos vestígios de evolução já presentes no código-fonte, discutiremos para onde o NCCL está caminhando. Especificamente, analisaremos três forças de evolução entrelaçadas: as primitivas de comunicação vão de coletivas fixas para programáveis — o agendamento de tarefas RMA em src/rma/rma.cc permite que a camada superior combine as primitivas Put/Signal/WaitSignal, em vez de apenas chamar AllReduce; a iniciação de rede vai de host proxy para envio direto pela GPU — o gerenciamento de backend GIN em src/gin/gin_host.cc permite que o kernel da GPU acione diretamente a placa de rede; o modelo de memória vai de buffers registrados para memória simétrica — a seleção de kernel de memória simétrica em src/sym_kernels.cc permite que todos os ranks usem o mesmo conjunto de endereços virtuais para acessar os buffers uns dos outros. Essas três forças não são isoladas; elas compartilham a mesma infraestrutura: a abstração de team em src/nccl_device/core.cc e o DevComm versionado em src/devcomm/devcomm_v23100.cc. Entender como elas se encaixam é entender a lógica de evolução do NCCL de "biblioteca de comunicação coletiva" para "motor de comunicação programável".

# I. Primitivas de comunicação programáveis: como o RMA transforma a "receita fixa" em "buffet"

## Modelo intuitivo

A comunicação coletiva do NCCL tradicional é como um pacote fixo: você pede AllReduce, e a cozinha executa todo o fluxo do AllReduce. Mas no cenário de paralelismo de especialistas (MoE), cada token precisa ser enviado para especialistas diferentes, e o padrão de envio não é conhecido em tempo de compilação — isso é como um buffet, você mesmo decide o que pegar, quanto pegar e quando pegar.

RMA é exatamente o "balcão de buffet" que o NCCL oferece para as camadas superiores: Put (escrever dados na memória do par), Signal (notificar o par), WaitSignal (aguardar sinal do par). Frameworks de camadas superiores podem combinar livremente essas três primitivas para implementar qualquer padrão de comunicação.

Sem RMA, o all-to-all do MoE só poderia ser simulado por múltiplas operações coletivas de pequena escala, cada uma passando pelo fluxo completo de inicialização de kernel e sincronização, com latência alta demais para ser aceitável.

## Estrutura de dados e layout de memória

A estrutura de dados central do RMA é`ncclTaskRma`(descrição de tarefa) e`ncclRmaArgs`(parâmetros do plano). Vamos primeiro ver os campos de`ncclRmaArgs`, que é inicializado em`scheduleRmaTasksToPlan`.

[FACT:src/rma/rma.cc:166-171]

```cpp
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
plan->rmaArgs->nRmaTasks = 0;
plan->rmaArgs->nRmaTasksProxy = 0;
plan->rmaArgs->nRmaTasksCe = 0;
```

Os campos-chave aqui são`nRmaTasksProxy`e`nRmaTasksCe`. Eles dividem as tarefas RMA em dois caminhos de execução:

- **Caminho CE**(Copy Engine, mecanismo de cópia): o rank de destino está dentro do escopo LSA (Local Symmetric Access, acesso simétrico local), pode ser concluído diretamente com o mecanismo de cópia da GPU, sem necessidade de rede.
- **Caminho Proxy**: o rank de destino não está dentro do escopo LSA, obrigatoriamente precisa passar pela thread host proxy para acionar a rede.

> **[Design Inference & Architectural Trade-offs]**
> A motivação desse design dicotômico é direta: comunicação dentro do escopo LSA usa NVLink ou PCIe, com alta largura de banda e baixa latência, sendo mais vantajoso usar cópia assíncrona via CE; comunicação entre máquinas obrigatoriamente passa pela placa de rede, só podendo ser acionada por threads proxy. Separar os dois tipos de tarefas para agendamento é o que permite que CE e proxy executem em paralelo, em vez de esperar em série.

`ncclTaskRma`contém em si`peers`、`nsignals`、`signalIdxs`três ponteiros de array, registrando respectivamente o rank do par, a quantidade de sinais e o índice do sinal. Para tarefas WaitSignal, uma tarefa pode aguardar múltiplos peers; para tarefas Put/Signal, uma tarefa é direcionada a apenas um peer.

## Step-by-Step Walkthrough: o agendamento de um WaitSignal

Vamos usar um cenário concreto: rank 0 chama`ncclWaitSignal`, aguardando sinais de rank 1 e rank 3. Suponha que rank 1 está dentro do escopo LSA e rank 3 não está.

**Primeiro passo: encontrar a primeira fila de contexto não vazia.**

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

As tarefas RMA são enfileiradas por context, cada context é um canal RMA independente. Aqui encontra-se o primeiro context com tarefas e retira-se sua fila.

**Segundo passo: retirar a primeira tarefa e determinar o tipo.**

[FACT:src/rma/rma.cc:163-168]

```cpp
struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
```

`firstTask->func`é`ncclFuncWaitSignal`, entra no branch WaitSignal.

**Terceiro passo: dividir os peers por alcançabilidade LSA.**

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

`isLsaAccessible`percorre`comm->devrState.lsaRankList`, determinando se o peer está dentro do time LSA. rank 1 está dentro do LSA, vai para a lista CE; rank 3 não está, vai para a lista Proxy.

**Quarto passo: criar uma nova tarefa para CE e para Proxy.**

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

A tarefa WaitSignal original é dividida em duas: a tarefa CE aguarda rank 1, a tarefa Proxy aguarda rank 3. As duas tarefas podem executar em paralelo — o caminho CE aguarda na GPU, o caminho Proxy aguarda na thread host.

**Quinto passo: liberar a tarefa original.**

[FACT:src/rma/rma.cc:249-251]

```cpp
planner->nTasksRma -= 1;
ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
```

A tarefa original já foi dividida em duas novas tarefas, é liberada de volta para o pool de memória.

## Controle de concorrência e interação com hardware

A execução paralela do RMA se manifesta em`ncclRmaWaitSignal`.

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

Este trecho de código usa CUDA event para sincronização entre streams: primeiro registra um event no stream de entrada, faz o stream CE aguardar esse event, depois inicia as tarefas proxy e CE nos dois streams respectivamente, e por fim faz o stream de entrada aguardar o event do stream CE. Assim os dois caminhos avançam em paralelo, mas externamente se apresentam como uma operação síncrona.

> **[Design Inference & Architectural Trade-offs]**
> O trade-off de design aqui é: execução paralela reduz a latência, mas introduz overhead adicional de registro de event e sincronização de streams. Para mensagens pequenas, esse overhead pode superar o ganho do paralelismo; para mensagens grandes, o ganho do paralelismo é significativo. O NCCL não faz julgamento adaptativo aqui, mas segue uniformemente o caminho paralelo — porque o cenário típico do RMA é comunicação de granulação fina com mensagens grandes.

## Guia de prevenção de armadilhas em produção

**Armadilha 1: erro na determinação de alcançabilidade LSA faz a tarefa seguir o caminho errado.** `isLsaAccessible`percorre`lsaRankList`, se`lsaSize`for 0 (por exemplo, domínio de comunicação de rank único), todos os peers serão considerados inalcançáveis, todos seguindo o caminho Proxy. Isso não se manifesta em testes de pequena escala, mas em implantações de grande escala causa queda abrupta de desempenho. O método de investigação é ver no log INFO de`scheduleRmaTasksToPlan`a proporção de`nRmaTasksProxy`e`nRmaTasksCe`.

**Armadilha 2: ciclo de vida do array de peers após a divisão da tarefa WaitSignal.**O`peersCe`do caminho CE usa`ncclMemoryStackAlloc`para alocação, o ciclo de vida acompanha`comm->memScoped`; o`peersProxy`do caminho Proxy usa`ncclCalloc`para alocação, e após a execução da tarefa precisa ser manualmente`free`. Se a criação da tarefa Proxy falhar,`fail`o ramo irá liberar esses arrays.

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

**Armadilha 3: Lote entre contextos de tarefas Put/Signal.**No ramo Put/Signal, o NCCL agrupa as tarefas put/signal de todos os contextos no mesmo plano, mas para ao encontrar WaitSignal.

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

A intenção deste design é: uma única inicialização de kernel cobre os put/signal de todos os contextos, reduzindo o overhead de inicialização. Mas a fila de cada contexto só é consumida até o primeiro WaitSignal, garantindo a ordem FIFO por contexto. Se a camada superior alternar chamadas de put e waitSignal no mesmo contexto, o efeito de agrupamento será bastante reduzido — este é um padrão que precisa de atenção ao usar RMA.

---

# Dois, envio direto pela rede da GPU: como o GIN permite que o kernel contorne o host proxy

## Modelo intuitivo

A comunicação de rede tradicional do NCCL é como enviar uma carta: o kernel da GPU coloca os dados no buffer, a thread do host proxy entrega os dados à placa de rede, e a placa de rede os envia. O GIN, por sua vez, permite que o kernel da GPU deposite a carta diretamente na caixa de correio do destinatário — o kernel escreve diretamente na fila de transmissão da placa de rede, e a placa de rede lê diretamente da memória da GPU.

Sem o GIN, cada comunicação de rede precisa passar pela memória do host como intermediária, adicionando pelo menos uma ida e volta de PCIe à latência. Para comunicações de granularidade fina como MoE, essa latência é fatal.

## Estruturas de dados e layout de memória

O estado central do GIN é`ncclGinState`, que gerencia múltiplos backends e múltiplos DevComm. Vamos primeiro ver a tabela de compatibilidade de versões de backend.

[FACT:src/gin/gin_host.cc:27-33]

```cpp
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5), NCCL_VERSION(2, 32, 0)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
constexpr int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0), NCCL_VERSION(2, 32, 0)};
```

O índice desses arrays é o número da versão do backend, e o valor é a versão mínima compatível do NCCL. Por exemplo,`proxyBackendMinVersions[3]`corresponde à versão de backend 3, exigindo NCCL pelo menos 2.32.0. Esse design permite que o NCCL selecione a versão de backend adequada em tempo de execução com base na versão do código do dispositivo, em vez de vinculá-la em tempo de compilação.

> **[Design Inference & Architectural Trade-offs]**
> A motivação para esse design de tabela de compatibilidade de versões é: o ritmo de evolução do backend GIN (driver da placa de rede, firmware) e da biblioteca NCCL é diferente. Se os requisitos de versão fossem codificados de forma fixa, qualquer atualização de um dos lados causaria incompatibilidade. Usar arrays para mapeamento de versões permite seleção dinâmica em tempo de execução, mantendo compatibilidade com backends antigos.

`ncclGinStateDevComm`é o estado GIN de cada DevComm, contendo campos como`contextCount`、`backendIndex`、`ginCtx[]`、`devHandles[]`. Ele é encadeado em uma lista ligada anexada a`ginState->devComms`.

## Passo a passo: o estabelecimento de uma conexão GIN

Vamos considerar um cenário: o rank 0 inicializa o domínio de comunicação e precisa estabelecer uma conexão GIN.

**Primeiro passo: verificar se o GIN está habilitado e suportado.**

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

`ncclParamGinEnable()`lê a variável de ambiente`NCCL_GIN_ENABLE`, padrão 1. Se o usuário desabilitar explicitamente, retorna erro diretamente.

**Segundo passo: verificar o suporte a memória simétrica.**

[FACT:src/gin/gin_host.cc:111-114]

```cpp
if (!comm->symmetricSupport) {
  WARN("Communicator does not support symmetric memory!");
  return ncclInternalError;
}
```

O GIN depende de memória simétrica — porque o kernel da GPU precisa conhecer o endereço virtual do buffer do par, e apenas a memória simétrica pode garantir endereços consistentes.

**Terceiro passo: obter a lista local de dispositivos GIN.**

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

`ncclTopoGetLocalGinDevs`encontra todas as placas de rede que suportam GIN a partir do grafo de topologia. Se exceder`NCCL_GIN_MAX_CONNECTIONS`, pega apenas os primeiros e imprime um aviso.

**Quarto passo: calcular a equipe GIN.**

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

Cada backend primeiro chama`devices`para obter o número de dispositivos, e então executa o fluxo listen→getProperties→allGather→connect→closeListen para cada conexão.`bootstrapAllGather`troca handles entre todos os ranks, de modo que cada rank conhece as informações de conexão do par.

## Controle de concorrência e interação com hardware

A thread de progresso do GIN é o mecanismo central de concorrência.

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

Aqui há alguns designs-chave:

1. **Afinidade de CPU**：`ncclOsSetAffinity`vincula a thread de progresso a um núcleo de CPU específico, evitando invalidação de cache causada por migração de thread.

2. **Backoff de trava de escrita**：`writePending`é um flag atômico; quando a thread principal precisa modificar a lista ligada`devComms`, ela o define primeiro, e a thread de progresso, ao vê-lo, faz yield ativamente, evitando disputa de lock.

3. **Trava de leitura/escrita**：`devCommRwMutex`é`shared_timed_mutex`, a thread de progresso mantém a trava de leitura ao percorrer a lista ligada, e a thread principal mantém a trava de escrita ao modificá-la.

4. **Divisão de threads**: a thread t é responsável pelas conexões t, t+proxyNthreads, t+2*proxyNthreads, ..., implementando balanceamento de carga por meio de um laço com stride.

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

A implementação desta trava de escrita assume que há apenas um escritor (a thread principal), portanto não precisa de mutex adicional.`writePending`define o flag primeiro e depois adquire a trava, garantindo que a thread de progresso veja a intenção de escrita antes de adquirir a trava e faça backoff ativamente.

## Guia de armadilhas em produção

**Armadilha 1: incompatibilidade no número de conexões GIN causando deadlock no AllGather.**O`ginCommCount`de cada rank pode ser diferente (dependendo do número de placas de rede locais), e o NCCL usa`bootstrapAllGather`para obter o valor mínimo entre todos os ranks.

[FACT:src/gin/gin_host.cc:176-180]

```cpp
ginCommCountHandles[comm->rank] = backend->ginCommCount;
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
for (int r = 0; r nRanks; r++) {
  backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
}
```

Se o número de placas de rede de um determinado rank for menor que o dos outros ranks, todos os ranks serão reduzidos ao valor mínimo. Isso garante simetria nas conexões, mas desperdiça recursos de placas de rede.

**Armadilha 2: proxyNthreads excede ginCommCount, causando ociosidade das threads.**Se o usuário configurou`NCCL_GIN_PROXY_NTHREADS`maior que`ginCommCount`, as threads excedentes ficarão ociosas no loop de stride.

[FACT:src/gin/gin_host.cc:181-183]

```cpp
// After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
// on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
// Extra threads simply idle in the stride loop; no correctness issue.
```

Isso não é um problema de correção, mas desperdiça recursos de CPU. O método de diagnóstico é verificar se`NCCL_GIN_PROXY_NTHREADS`é maior que o número real de placas de rede.

**Armadilha 3: condição de corrida ao liberar o DevComm.** `ncclGinDevCommFree`Primeiro remove o DevComm da lista encadeada, depois destrói o context.

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

Após a remoção, a thread de progresso não consegue mais ver esse DevComm, então destruir o context é seguro. Porém, se houver operações de rede in-flight durante a destruição, pode ocorrer comportamento indefinido — isso é o que precisa ser garantido ao usar GIN: antes de liberar o DevComm, é preciso garantir que todas as operações foram concluídas.

---

# Três, kernel de memória simétrica: de "registrar buffer" para "espaço de endereçamento unificado"

## Modelo intuitivo

O buffer do NCCL tradicional é "baseado em registro": cada rank registra seu próprio buffer e, durante a comunicação, troca endereços via handle. Já a memória simétrica é um "espaço de endereçamento unificado": todos os ranks concordam com o mesmo conjunto de endereços virtuais; o endereço A do rank 0 e o endereço A do rank 1 apontam para suas respectivas memórias físicas, mas no código basta usar o mesmo endereço para acessá-las.

É como se todos concordassem que "fileira 3, assento 5" se refere ao mesmo local na casa de cada um; ao procurar algo, não é preciso perguntar primeiro "onde fica a fileira 3, assento 5 da sua casa".

Sem memória simétrica, cada kernel precisaria primeiro resolver o endereço do par, aumentando o custo de instruções e a pressão sobre os registradores.

## Estruturas de dados e layout de memória

O núcleo do kernel de memória simétrica é o kernel mask — um bitmap que marca quais kernels estão disponíveis no domínio de comunicação atual.

[FACT:src/sym_kernels.cc:17-63]

```cpp
constexpr uint32_t kernelMask_STMC =
  1  **[Design Inference & Architectural Trade-offs]**
> A vantagem desse design de bitmap é que é possível filtrar rapidamente os kernels disponíveis com operações de bits. Por exemplo,`kmask &= ~kernelMask_STMC`uma linha já desabilita todos os kernels STMC, sem precisar percorrer a lista.

## Step-by-Step Walkthrough: um cálculo de kernel mask

Vamos usar um cenário: o rank 0 precisa executar AllReduce, o tipo de dado é float16, o tamanho da mensagem é 1MB, o domínio de comunicação tem 8 ranks, todos interconectados por NVLink.

**Primeiro passo: obter o mask base correspondente à operação.**

[FACT:src/sym_kernels.cc:304-306]

```cpp
uint32_t kmask = kernelMask_coll(coll);
```

`kernelMask_coll(ncclFuncAllReduce)`retorna`kernelMask_AR`, contendo 5 kernels AllReduce.

**Segundo passo: verificar a disponibilidade de STMC e LDMC.**

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

`hasLsaMultimem`é calculado em`ncclSymkInitOnce`, exigindo que o multicast simétrico NVLS esteja disponível e que o time LSA tenha mais de 2 ranks. float16 suporta LDMC, então se`hasLsaMultimem`for verdadeiro, o kernel LDMC é mantido.

**Terceiro passo: verificar o limite de tamanho da mensagem.**

[FACT:src/sym_kernels.cc:336-342]

```cpp
size_t nBytes = alignUp(nElts * ncclTypeSize(ty), NCCL_SYM_KERNEL_CELL_SIZE);
size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks) * nBytes;
if (nBusBytes >= (size_t(2) = 32 * (size_t(2) nRanks;
kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
```

Se o time LSA cobre todos os ranks, GIN não é necessário; caso contrário, mantém-se apenas o kernel GIN.

## Controle de concorrência e interação com hardware

A inicialização do kernel de memória simétrica envolve a criação do DevComm e a alocação de recursos.

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

O ponto-chave aqui é`ncclDevrCommCreateInternal`, que cria um DevComm interno contendo recursos como multicast LSA, inbox/outbox GIN, sinais, etc.`reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL`especifica que o GIN usa o modo de conexão rail.

[FACT:src/sym_kernels.cc:257-261]

```cpp
symk->kcomm.workStarted = comm->profiler.symWorkStarted;
symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
symk->kcomm.workPhases = comm->profiler.symWorkPhases;
```

O kernel de memória simétrica usa um buffer de profiler independente, evitando intercalação com o workCounter dos kernels regulares.

## Guia de prevenção de armadilhas em produção

**Armadilha 1: requisitos de SMEM do kernel TMA.**TMA requer cerca de 8KB de SMEM scratch por warp; com 16 warps, são 128KB.

[FACT:src/sym_kernels.cc:135-142]

```cpp
bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  if (comm->maxSharedMemOptin minCompCap >= 100 && ncclParamSymTmaEnable();
}
```

Se a capacidade de SMEM da GPU for insuficiente (por exemplo, em instâncias MIG), o kernel TMA será desabilitado. O método de diagnóstico é verificar se`maxSharedMemOptin`é menor que`ncclTmaShmemScratchWarpSize() * 16`。

**Armadilha 2: limites do chunk size do GIN.**O chunk size do kernel ReduceScatter GIN tem limites superior e inferior.

[FACT:src/sym_kernels.cc:148-153]

```cpp
static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128  0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}
```

Se o`NCCL_SYM_RS_GIN_CHUNK_SIZE`configurado pelo usuário exceder 1GB, será truncado para 1GB; se for menor que 128 bytes, será elevado para 128 bytes. O valor final também será arredondado para baixo para uma potência de 2.

**Armadilha 3: Incompatibilidade de tipo de registro de memória simétrica.** `ncclGetSymRegType`Com base nas flags de sendWin e recvWin,`NCCL_WIN_COLL_SYMMETRIC`determine o tipo de registro.

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

Se os tipos de registro de send e recv forem inconsistentes, o kernel precisa seguir caminhos de código diferentes. Isso afeta o desempenho, mas não causa erros.

---

# IV. Abstração de Team e DevComm versionado: infraestrutura de evolução

## Modelo intuitivo

A abstração de Team é como "agrupamento": o time mundial é a turma inteira, o time LSA são os colegas de mesa, o time Rail são os assentos da mesma coluna. Diferentes modos de comunicação exigem diferentes perspectivas de agrupamento.

O DevComm versionado é como um "tradutor": diferentes versões do código de dispositivo falam "dialetos" diferentes, e a camada de compatibilidade do DevComm é responsável por traduzir, permitindo que códigos novos e antigos se entendam.

Sem a abstração de Team, cada kernel teria que calcular seu próprio mapeamento de rank; sem o DevComm versionado, qualquer mudança de ABI faria com que todo o código de dispositivo fosse recompilado.

## Estruturas de dados e layout de memória

Team é uma tripla simples:`nRanks`、`rank`、`stride`。

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

O stride do time mundial é 1, porque todos os ranks estão dispostos consecutivamente.

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

O stride do time Rail é`lsaSize`, porque os ranks em cada rail são separados pelo tamanho de um time LSA.

O núcleo do DevComm versionado é a estrutura`ncclDevCommCompat`.

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

Esta estrutura define as regras de compatibilidade da versão 2.31.0.`minVersion`e`maxVersion`definem o intervalo de versões aplicável, e os quatro ponteiros de função seguintes definem a filtragem de atributos e a lógica de conversão de estrutura. Se todos forem nullptr, significa que esta versão não tem requisitos especiais de compatibilidade.

## Passo a passo: uma conversão de Team

Vamos considerar um cenário: rank 5 em um domínio de comunicação de 8 ranks, com tamanho do time LSA igual a 4. Queremos calcular o rank do rank 5 no time Rail.

**Primeiro passo: inicializar o estado do DevR.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
```

`ncclDevrInitOnce`Calcula informações derivadas como o time LSA, time CFT, etc. Se falhar, retorna um time vazio.

**Segundo passo: calcular os parâmetros do time Rail.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
ncclTeam_t ans;
ans.nRanks = comm->nRanks / comm->devrState.lsaSize;  // 8 / 4 = 2
ans.rank = comm->rank / comm->devrState.lsaSize;       // 5 / 4 = 1
ans.stride = comm->devrState.lsaSize;                  // 4
```

O rank do rank 5 no time Rail é 1, o time tem 2 ranks e o stride é 4.

**Terceiro passo: converter de volta para o rank mundial.**

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Se quisermos converter o Rail rank 0 para o rank mundial:`5 + (0 - 1) * 4 = 1`. Verificação: rank 1 e rank 5 estão no mesmo rail (separados por 4).

## Controle de concorrência e interação com hardware

A abstração de Team em si é sem estado e não requer controle de concorrência. Mas`ncclDevrInitOnce`é carregado de forma preguiçosa, calculando todas as informações derivadas na primeira chamada.

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

O comentário diz "Ignoring errors since if it fails ncclDevrInitOnce will try again" — se a inicialização falhar, retorna um time vazio e a próxima chamada tentará novamente.

## Guia de prevenção de armadilhas em produção

**Armadilha 1: Suposição de stride na conversão de Team.** `ncclTeamRankToWorld`Assume que os ranks dentro do time formam uma progressão aritmética.

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Se o time não for uma progressão aritmética (por exemplo, um agrupamento arbitrário personalizado), esta função calculará errado. Atualmente, o NCCL só suporta times regulares.

**Armadilha 2: Ponteiro nulo no DevComm versionado.** `ncclDevCommCompat_v23100`Todos os ponteiros de função são nullptr, indicando que não há lógica de compatibilidade especial. Se versões futuras precisarem de conversão, essas funções devem ser implementadas, caso contrário, códigos novos e antigos não poderão interoperar.

**Armadilha 3: Modo hierárquico do time CFT.** `ncclTeamCft`Suporta três modos: FLAT, HIER_MULTIMEM, HIER_LSA.

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

Se um modo inválido for passado, retorna um time vazio. Ao usar o time CFT, é preciso garantir que o modo esteja correto.

---

# Reflexões de design

**Por que o NCCL suporta simultaneamente três caminhos de evolução: RMA, GIN e memória simétrica?**

> **[Design Inference & Architectural Trade-offs]**
> Esses três caminhos resolvem problemas em diferentes níveis:

- **RMA**Resolve o problema de "modo de comunicação fixo" — permitindo que a camada superior combine primitivas para implementar qualquer modo de comunicação.
- **GIN**Resolve o problema de "alta latência de rede" — permitindo que a GPU controle diretamente a placa de rede, contornando o host proxy.
- **Memória simétrica**Resolve o problema de "sobrecarga de resolução de endereço" — permitindo que o kernel acesse diretamente a memória do par usando um endereço unificado.

Eles não são relações de substituição, mas de complementaridade. O RMA pode usar o GIN como transporte subjacente, e o GIN depende da memória simétrica para fornecer consistência de endereço. Juntos, os três formam a infraestrutura do "motor de comunicação programável".

**Qual é a filosofia de design do DevComm versionado?**

> **[Design Inference & Architectural Trade-offs]**
> A ideia central do DevComm versionado é "ABI estável, API em evolução". O código de dispositivo (kernel) é compilado e embutido no binário, e não pode ser recompilado a cada atualização da biblioteca NCCL. Portanto, o NCCL deve garantir que códigos de dispositivo antigos possam ser executados na nova biblioteca.`ncclDevCommCompat`A estrutura é a entrada da camada de compatibilidade: a nova biblioteca seleciona as regras de compatibilidade apropriadas com base na versão do código de dispositivo e, se necessário, realiza a conversão de estrutura.

---

# Resumo do capítulo

Neste capítulo, partindo dos vestígios de evolução no código-fonte, analisamos as três forças que levaram o NCCL de uma biblioteca de comunicação coletiva a um motor de comunicação programável:

1. **RMA**（`src/rma/rma.cc`): Através da combinação das primitivas Put/Signal/WaitSignal, permite que as camadas superiores implementem qualquer padrão de comunicação. O design central é dividir as tarefas em dois caminhos paralelos, CE e Proxy, com base na acessibilidade LSA.

2. **GIN**（`src/gin/gin_host.cc`): Através do envio direto da GPU para a rede, contornando o host proxy. O design central é o gerenciamento multi-backend, tabela de compatibilidade de versões, pool de threads de progresso.

3. **kernel de memória simétrica**（`src/sym_kernels.cc`): Através do espaço de endereçamento unificado, elimina a sobrecarga de resolução de endereços. O design central é o bitmap de máscara do kernel e a aceleração de hardware TMA/GIN.

4. **Abstração Team e DevComm versionado**（`src/nccl_device/core.cc`、`src/devcomm/devcomm_v23100.cc`): Fornece infraestrutura para evolução. Team fornece uma visão de agrupamento, DevComm versionado fornece compatibilidade ABI.

Essas mudanças têm um impacto profundo nas camadas superiores dos frameworks: o ProcessGroup do PyTorch pode chamar diretamente as primitivas RMA para implementar padrões de comunicação personalizados; o paralelismo de especialistas do Megatron pode utilizar GIN para reduzir a latência do all-to-all; a memória simétrica torna o código do kernel mais conciso.

# Reflexões e autoavaliação deste capítulo

Q1: Se removermos`scheduleRmaTasksToPlan`a verificação de acessibilidade LSA do ramo WaitSignal em , fazendo com que todos os peers sigam o caminho Proxy, quais seriam as consequências? Em quais cenários isso desencadearia um desastre de desempenho?

**Análise de referência**：

A verificação de acessibilidade LSA está em[FACT:src/rma/rma.cc:187-204], ela divide os peers em dois grupos: CE e Proxy. Se removermos essa verificação, todos os peers seguirão o caminho Proxy,`nRmaTasksCe`será sempre 0.

As consequências são: o caminho CE não será utilizado de forma alguma, todos os WaitSignal farão polling na rede através de threads do host proxy. Para peers dentro do alcance LSA (interconectados por NVLink na mesma máquina), que poderiam usar o mecanismo de cópia assíncrona da GPU, agora passam a fazer polling por threads do host, com latência subindo de microssegundos para milissegundos.

Cenário de desastre de desempenho: No treinamento MoE, cada token precisa esperar pelos sinais de múltiplos especialistas. Se todos os sinais passarem pelo Proxy, as threads do host se tornam o gargalo, e a GPU passa a maior parte do tempo esperando o polling do host. Em máquinas com 8 GPUs totalmente interconectadas por NVLink, essa degradação é especialmente evidente — toda a comunicação que poderia usar CE agora fica congestionada no host.

Método de diagnóstico: Verifique os logs INFO de`scheduleRmaTasksToPlan`, se`nRmaTasksCe`for sempre 0 enquanto`nRmaTasksProxy`for muito grande, isso indica um problema na verificação LSA.

Q2：`ncclGinProgress`Em`writePending`, a combinação do flag`devCommRwMutex`com o lock de leitura/escrita`writePending`, se removermos a verificação de

**e mantivermos apenas o lock de leitura/escrita, quais seriam os problemas?**：

`writePending`Análise de referência[FACT:src/gin/gin_host.cc:63-66]A verificação de

está em`std::shared_timed_mutex`, ela faz com que a thread de progresso ceda ativamente quando a thread principal precisa escrever. Se removermos essa verificação, a thread de progresso tentará diretamente adquirir o lock de leitura.`ncclGinDevCommSetup`O problema é que:`ncclGinDevCommFree`o lock de leitura de

é compartilhado, múltiplas threads de progresso podem mantê-lo simultaneamente. Se a thread principal precisar adquirir o lock de escrita, terá que esperar que todos os locks de leitura sejam liberados. Sob alta carga, as threads de progresso adquirem frequentemente o lock de leitura, e a thread principal pode não conseguir adquirir o lock de escrita por um longo período, causando bloqueio em`ginProgressWriteLock`ou`writePending`.`writePending`Mais grave ainda: se a thread principal definir

`writePending`antes de adquirir o lock em

Q3：`ncclSymkMask`, e a thread de progresso não verificar`nBusBytes >= 32 * (size_t(2) << 30)`, então a thread de progresso pode continuar adquirindo o lock de leitura após a thread principal definir o flag, tornando o tempo de espera da thread principal imprevisível.`kmask = 0`A função de`ncclSymkAvailable`é uma "notificação suave": informar às threads de progresso "vou escrever, cedam a vez". Isso é mais eficiente do que depender apenas da justiça do lock, porque as threads de progresso podem ceder ativamente em vez de bloquear no lock.

**Em**：

`kmask = 0`, se[FACT:src/sym_kernels.cc:342]desabilitar todos os kernels (`ncclSymkAvailable`), nesse momento[FACT:src/sym_kernels.cc:354-361]）。

retorna false, para qual caminho o NCCL fará fallback? Qual é o impacto de desempenho desse caminho de fallback?

Análise de referência

Em

, nesse momento

---

# retorna false (

O caminho de fallback é: o NCCL usará os kernels tradicionais de comunicação coletiva (kernels de memória não simétrica). Esses kernels acessam a memória do peer através de buffers registrados, precisando primeiro resolver o endereço, com maior sobrecarga de instruções.

Impacto de desempenho: Para mensagens muito grandes (acima de 64GB de bytes no barramento), a sobrecarga de resolução de endereços dos kernels tradicionais é proporcionalmente pequena, pois a transferência de dados em si domina. Mas em casos limítrofes (logo acima de 64GB), os kernels tradicionais podem ser 10-20% mais lentos que os kernels de memória simétrica.**Permite que as frameworks de nível superior implementem padrões de comunicação personalizados com menor latência e maior flexibilidade**. Para frameworks como PyTorch e Megatron, isso significa que eles podem construir diretamente sobre o NCCL padrões de comunicação complexos como MoE all-to-all, paralelismo de pipeline e paralelismo de especialistas, sem precisar contornar o NCCL e implementar sua própria camada de rede.

O próximo capítulo é o último capítulo do livro. Vamos percorrer novamente toda a cadeia completa de um AllReduce — começando pela chamada`ncclAllReduce`, passando pelo enfileiramento de tarefas, seleção de algoritmo, lançamento de kernel, avanço do proxy, transmissão de rede, até o retorno do resultado. Esta revisão conectará os pontos de conhecimento dos 24 capítulos anteriores, formando um mapa cognitivo completo.

Até aqui, vimos claramente as três linhas principais da evolução do NCCL de operações de conjunto fixas para um mecanismo de comunicação programável: composição de primitivas RMA, envio direto da GPU para a rede, modelo de memória simétrica, e a abstração de team e o DevComm versionado que os sustentam. Esses mecanismos apontam juntos para um futuro de comunicação mais flexível e mais próximo das capacidades do hardware. No entanto, independentemente de como a arquitetura evolua, a cadeia completa de um AllReduce é sempre a base para entender o NCCL. No próximo capítulo não introduziremos novo código, mas reconectaremos o fluxo ponta a ponta do Capítulo 3 ao Capítulo 10 — desde a chamada ncclAllReduce, até o estabelecimento do domínio de comunicação, busca de topologia, seleção de algoritmo, enfileiramento de tarefas, lançamento de kernel, execução de primitivas no lado do dispositivo e escrita de volta do resultado. Você remontará os mecanismos dispersos pelos capítulos em um modelo mental completo e obterá um índice de "qual capítulo consultar ao encontrar problemas".
