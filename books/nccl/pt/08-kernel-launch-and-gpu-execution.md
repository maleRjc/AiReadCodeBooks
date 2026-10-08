# Capítulo 8: Lançamento de Kernel e Execução no Lado do Device: da chamada no lado host à partida dos blocos de threads da GPU

No capítulo anterior, decompusemos como a tarefa é dividida entre múltiplos channels, como os parâmetros de lançamento de kernel são gerados e o mecanismo de submissão em lote e ordenação de dependências sob a semântica de group. Agora, o plano de lançamento está pronto, mas ainda é apenas uma estrutura de dados no lado host. A questão central que este capítulo responde é:`ncclKernelPlan`Como isso se transforma em um grid realmente em execução na GPU? Vamos seguir a cadeia de chamadas de`ncclLaunchKernel`para ver como os parâmetros são inseridos nos kernel args, como a variante de kernel é selecionada,`cuLaunchKernelEx`como é chamado, e como no lado do device`ncclKernelMain`lê a descrição do trabalho da memória compartilhada e a distribui para a implementação concreta.

# Do Plan ao Grid: panorama do caminho de lançamento

Antes de entrar em detalhes, vamos estabelecer um modelo mental geral. Imagine`ncclKernelPlan`como uma "planta de construção": ela registra quantos channels serão lançados (quantos blocks), quantas threads por block, quais works serão executados e qual função de kernel será usada. E`ncclLaunchKernel`é a ação da "equipe de construção entrando no canteiro" — ela traduz as informações da planta no que o driver CUDA consegue entender,`CUlaunchConfig`e então chama`cuLaunchKernelEx`para realmente disparar o grid na GPU.

Sem essa camada, todo o escalonamento no lado host (a divisão de channels do capítulo anterior, a organização de batches, a ordenação de proxy ops) seria apenas teoria no papel, nenhum kernel seria executado na GPU e a comunicação nunca aconteceria. Esta é a última peça do backbone ponta a ponta e também a linha divisória entre host e device.

Todo o caminho de lançamento pode ser resumido em três fases:

1. **Preparação de parâmetros**（`finishPlan` + `uploadWork`): organizar as structs de work, os descritores de batch e os kernel args em um bloco contíguo de memória, decidindo se ficam nos parâmetros do kernel, na FIFO ou em um buffer persistente.

2. **Lançamento do kernel**（`ncclLaunchKernel`): calcular as dimensões de grid/block, montar os launch attributes (CGA cluster, mem sync domain, launch completion event), chamar`cuLaunchKernelEx`。

3. **Entrada no lado do device**（`ncclKernelMain`): cada block determina seu channelId com base em`blockIdx.x`carrega o work batch dos args ou da FIFO para a memória compartilhada e então, por meio de`ncclDevFuncTable`distribui para a implementação concreta de algoritmo/protocolo.

A figura abaixo mostra o fluxo de controle completo do plan ao grid, incluindo os pontos de decisão críticos:

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

Esta figura ancora as três funções centrais deste capítulo:`finishPlan`、`uploadWork`、`ncclLaunchKernel`A seguir, vamos decompô-las uma a uma.

# Preparação de parâmetros: como a struct de work encontra seu lugar

## Modelo intuitivo

`finishPlan`O papel de é semelhante ao de um "empacotador" em um centro de triagem de encomendas. Ele lida com um monte de structs de work dispersas (cada operação collective ou p2p corresponde a uma) e precisa decidir: esses works vão para a "mochila de mão" dos parâmetros do kernel, para a "esteira transportadora" da FIFO, ou para o "armazém" do buffer persistente?

Se essa decisão for tomada errado — por exemplo, um work grande demais para caber nos parâmetros do kernel for forçado a entrar — o lançamento do kernel falhará diretamente. Se o work for colocado no lugar errado, o lado do device lerá dados inválidos e o resultado da comunicação estará completamente errado.

## Estruturas de dados e layout de memória

Primeiro vejamos`ncclDevKernelArgs`a estrutura de , que é o "envelope" entre host e device:

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

Essa struct tem apenas 5 campos, mas cada campo carrega informações críticas.`channelMask`é uma máscara de 64 bits, cada bit corresponde a um channel; o lado do device, por meio de`__popcll`calcula`blockIdx.x`o channelId correspondente.`workStorageType`determina de onde o lado do device lê o work:`Args`indica que o work está nos parâmetros do kernel,`Fifo`indica que está no buffer circular,`Persistent`indica que está no buffer persistente.

`ncclDevWorkBatch`é o descritor de batch, que informa ao lado do dispositivo "onde está o work deste channel e quantos são":

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

`offsetBitset`é uma máscara de 64 bits, cada bit corresponde a uma estrutura work. O lado do dispositivo, através das instruções`__popc`e`fns`(find n-th set), localiza o offset de cada work.`nextJump`e`nextExtends`são usados para encadear múltiplos batches — quando há work demais para caber em um batch, cria-se um "batch estendido".

## Step-by-Step Walkthrough

Agora vamos usar um cenário concreto: um AllReduce é dividido em 4 channels, cada channel tem 2 estruturas work, totalizando 8 works.

**Primeiro passo:`finishPlan`decide o tipo de armazenamento.**

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

A decisão-chave aqui é: se`sizeof(ncclDevKernelArgs) + batchBytes + workBytes`couber em`comm->workArgsBytes`(normalmente 4KB), coloca-se o work diretamente nos parâmetros do kernel. Caso contrário, o work é colocado no FIFO ou em buffer persistente, e nos parâmetros do kernel fica apenas o descritor de batch.

> **[Design Inference & Architectural Trade-offs]**
> Por que priorizar colocar nos parâmetros do kernel? Porque os parâmetros do kernel são passados através de constant memory no driver CUDA, e a leitura pelo lado do dispositivo usa a instrução`ld.param`, que é muito mais rápida do que ler o FIFO da memória global. Para mensagens pequenas (volume total de work pequeno), isso reduz significativamente a latência.

**Segundo passo: colocar os batches nos kernel args alternando por channel.**

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

Aqui há alguns pontos-chave:

1. **`fifoCursor`A semântica de**: para o tipo`Args`, é o offset em relação ao endereço inicial de`kernelArgs`; para o tipo`Fifo`, é o offset em relação ao endereço base do FIFO; para o tipo`Persistent`, começa em 0.

2. **`offsetBase`A correção de**：`finishPlan`O`offsetBase`do batch em é relativo à posição inicial do work no plan (começando em 0).`uploadWork`É preciso convertê-lo para um offset relativo à posição real de armazenamento. Para o tipo`Args`, soma-se`sizeof(ncclDevKernelArgs) + batchBytes`; para o tipo`Fifo`, soma-se`comm->workFifoProduced`。

3. **Cópia alinhada a 16 bytes**: as estruturas work são todas alinhadas a 16 bytes (`alignas(16)`), então a cópia é feita em unidades de 16 bytes.`COMPILER_ASSUME_ALIGNED`informa ao compilador que este endereço está alinhado a 16 bytes, fazendo o compilador gerar instruções vetorizadas mais eficientes.

4. **Espera no FIFO**: para o tipo`Fifo`,`waitWorkFifoAvailable`faz spin-wait até o FIFO ter espaço suficiente. Essa espera verifica`comm->abortFlag`, evitando deadlock em caso de abort.

## Reflexões de design e armadilhas em produção

> **[Design Inference & Architectural Trade-offs]**
> **Por que existem três tipos de armazenamento?**Isto é um trade-off entre espaço e latência:

- `Args`: o mais rápido (constant memory), mas com capacidade limitada (4KB). Adequado para mensagens pequenas e pouco work.
- `Fifo`: capacidade grande (ring buffer), mas a leitura pelo lado do dispositivo passa pela memória global. Adequado para mensagens médias.
- `Persistent`: usado em cenários de captura de CUDA Graph. Como durante a captura de graph não se pode fazer`cudaMemcpy`, é necessário pré-alocar um buffer persistente, copiar o work para lá, e então fazer o kernel ler de lá.

**Armadilha 1: overflow do FIFO causando deadlock.**Se`waitWorkFifoAvailable`não verificar`abortFlag`, quando o FIFO estiver cheio e o consumidor (GPU kernel) parar de consumir por algum motivo, o host ficará em spin para sempre. No código-fonte,[FACT:src/enqueue/enqueue.cc:1333-1349]verifica explicitamente a abort flag:

```c
if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
  return ncclInternalError;
}
```

**Armadilha 2: overflow de`offsetBitset`.** `offsetBitset`é de 64 bits, suportando no máximo 64 works em um batch. Se passar de 64,`1ull << (offset / workSize)`sofre overflow. No código-fonte,`NCCL_MAX_DEV_WORK_BATCH_BYTES`limita o tamanho do batch (1024 bytes), e a menor estrutura work é`ncclDevWorkColl`(cerca de 80 bytes), então no máximo 12 works, sem overflow.

**Armadilha 3: vazamento de memória no modo Persistent.**Em`uploadWork`, no branch`Persistent`,`fifoBufHost`é alocado através de`ncclOsAlignedAlloc`, e precisa ser liberado em`uploadWork_cleanup_fn`. Se`cudaMemcpyAsync`falhar, o label`fail`verifica se`cleanup`é null, e se for null libera diretamente`fifoBufHost`. Essa cadeia de recuperação de erro pode ser vista em[FACT:src/enqueue/enqueue.cc:1483-1485].

# Lançamento de Kernel: de CUlaunchConfig a cuLaunchKernelEx

## Modelo intuitivo

`ncclLaunchKernel`O papel é semelhante a um "console de controle de lançamento de foguete". Ele recebe um plan já abastecido (dados de work), calcula os parâmetros de voo do foguete (dimensões de grid/block), configura várias opções de lançamento (cluster, mem sync domain, completion event) e então pressiona o botão de lançamento (`cuLaunchKernelEx`）。

Se esta etapa falhar — por exemplo, se a dimensão do grid for calculada incorretamente — um número errado de blocks será iniciado na GPU, fazendo com que o trabalho de alguns channels nunca seja executado, e a comunicação ficará suspensa.

## Estrutura de dados e layout de memória

`CUlaunchConfig`É a estrutura de configuração de lançamento da API do driver CUDA, e o NCCL a constrói na stack:

[FACT:src/enqueue/enqueue.cc:1916-1917]

```c
CUlaunchConfig launchConfig = {0};
CUlaunchAttribute launchAttrs[6] = {};
int attrs = 0;
```

`launchAttrs`É um array de no máximo 6 elementos, cada elemento sendo um`CUlaunchAttribute`. O NCCL adiciona condicionalmente diferentes atributos com base na capacidade de hardware e na versão do driver:

- `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`: dimensão do CGA cluster (sm90+)
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`: política de agendamento de cluster
- `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`: domínio de sincronização de memória (CUDA 12.0+)
- `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`: evento de conclusão de lançamento (CUDA 12.3+)
- `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`: serialização de stream programática (sym kernel)
- `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING`: agendamento centralizado de utilização de NVLink (CUDA 13.0+)

## Step-by-Step Walkthrough

**Primeiro passo: calcular as dimensões de grid e block.**

[FACT:src/enqueue/enqueue.cc:1889-1893]

```c
int nChannels = countOneBits(plan->channelMask);
void* sym = plan->kernelFn;
dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
```

`nChannels`É`channelMask`o número de bits definidos em , ou seja, quantos blocks este plan deve iniciar. Cada block é responsável por um channel.`threadPerBlock`É calculado em`scheduleCollTasksToPlan`através de`plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE)`, tomando o maior`nWarps * 32`。

`smem`entre todas as tasks. É o tamanho da memória compartilhada dinâmica. Para kernels normais, é`ncclShmemDynamicSize(comm->cudaArch)`, que é uma constante de tempo de compilação dependente da arquitetura (sm70+ é`ncclShmemScratchWarpSize * (NCCL_MAX_NTHREADS / WARP_SIZE)`). Para sym kernels, é`plan->kernelDynSmem`, porque os requisitos de memória compartilhada do sym kernel podem ser diferentes.

**Segundo passo: montar os parâmetros do kernel.**

[FACT:src/enqueue/enqueue.cc:1902-1903]

```c
void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
                 CU_LAUNCH_PARAM_END};
```

Esta é uma forma de passagem de parâmetros da API do driver CUDA:`CU_LAUNCH_PARAM_BUFFER_POINTER`informa ao driver que "os parâmetros não são passados um a um, mas sim como um bloco contíguo de memória",`CU_LAUNCH_PARAM_BUFFER_SIZE`informa ao driver o tamanho desse bloco. A vantagem disso é que o NCCL pode passar`ncclDevKernelArgs`e o array batch seguinte de uma só vez, sem precisar empacotar parâmetro por parâmetro.

**Terceiro passo: adicionar launch attributes.**

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

CGA (Cooperative Group Array) é um recurso de hardware introduzido no sm90, que permite agrupar múltiplos blocks em um cluster; os blocks dentro do cluster têm garantia de serem agendados simultaneamente em um conjunto de SMs e podem acessar a memória compartilhada uns dos outros. O NCCL usa esse recurso para implementar algoritmos como NVLS que exigem sincronização entre blocks.

Observe a proteção`if (grid.x % clusterSize) clusterSize = 1;`: a dimensão do cluster deve dividir exatamente a dimensão do grid, caso contrário o driver retornará erro. Se`grid.x`não for divisível por`clusterSize`, ele degrada para não usar cluster.

**Quarto passo: adicionar launch completion event.**

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

`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`É um recurso introduzido no CUDA 12.3: o driver registra um evento quando o kernel realmente começa a executar (e não quando a chamada no lado do host retorna). Isso é crucial para implementar a "ordem implícita" (implicit order) — o NCCL precisa garantir que múltiplos kernels sejam executados em ordem, mas não quer que o lado do host bloqueie esperando.

`getImplicitOrder`A lógica de é: se o usuário definiu`launchOrderImplicit`, e a versão do driver for suficientemente nova, usar`ncclImplicitOrderLaunch`(ordenar com launch event); caso contrário, usar`ncclImplicitOrderSerial`(ordenar com completion event, ou seja, execução serial).

**Quinto passo: chamar`cuLaunchKernelEx`。**

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

`cuLaunchKernelEx`É uma nova API introduzida no CUDA 12.0, que suporta launch attributes. Para drivers antigos (< 11.8), o NCCL recorre a`cuLaunchKernel`：

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

## Controle de concorrência e interação com hardware

**Mecanismo de relay do Launch completion event.**Quando se usa`ncclImplicitOrderLaunch`e o usuário fornece`launchCompletionEvent`, o NCCL não pode passar diretamente o event do usuário para o driver, porque o driver suporta apenas um launch completion event. A abordagem do NCCL é:

1. Passar`comm->sharedRes->launchEvent`para o driver.

2. Em`relayStream`, esperar por`launchEvent`。

3. Em`relayStream`, registrar o event do usuário.

Assim, o event do usuário será disparado após o kernel realmente começar a executar, e não quando a chamada no lado do host retornar.

**Mem Sync Domain。** [FACT:src/enqueue/enqueue.cc:1938-1942]Em sm90+, o NCCL define`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`como`cudaLaunchMemSyncDomainRemote`. Este é o mecanismo de domínio de sincronização de memória introduzido pela arquitetura Hopper, usado para isolar as barreiras de memória de diferentes kernels e reduzir a sobrecarga de sincronização desnecessária.

## Guia de armadilhas em produção

**Armadilha 1: dimensão de cluster não divisível causa falha no lançamento.**Se`grid.x`não for divisível por`clusterSize`, o driver retornará`CUDA_ERROR_INVALID_VALUE`. No código-fonte, há proteção via`if (grid.x % clusterSize) clusterSize = 1;`, mas isso também significa que o recurso de cluster é silenciosamente desabilitado. Se o usuário espera o ganho de desempenho trazido pelo cluster, é preciso verificar`cgaClusterSize`e`nChannels`a relação entre .

**Armadilha 2: versão do driver não atendida torna o kernel indisponível.** `ncclInitKernelsForDevice`verifica os requisitos de driver de cada kernel durante a inicialização:

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

O núcleo desse trecho é calcular`fnsOfBitset`: para`offsetBitset`o n-ésimo bit ativo, qual é o seu índice de bit. O PTX tem a instrução`fns`para fazer isso, mas ela se expande em muitas instruções SASS. A abordagem do NCCL é usar memória compartilhada: cada lane verifica se seu bit está ativo; se estiver, calcula quantos bits ativos existem antes dele e então escreve seu número de lane em`fnsOfBitset[nWorksBelow]`。

Em seguida vem a cópia propriamente dita:

[FACT:src/device/common.h:209-241]

```c
if (tid = %d && __CUDA_ARCH__ >= %d\n" % (cudart ,arch))
  out("/*%4d*/ %s,\n" % (index, sym))
  if (cudart, arch) != (0, 0):
    out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
  index += 1
out("nullptr};\n")
```

## Reflexões de design e armadilhas em produção

**Por que usar`__grid_constant__`？** [FACT:src/device/common.h:19-24]

```c
#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif
```

`__grid_constant__`informa ao compilador que este parâmetro é somente leitura e pode ser colocado na memória constante. Assim, quando o lado do dispositivo faz a leitura, ele usa a instrução`ld.param`o que é mais rápido do que ler da memória global. O comentário menciona que isso quebra o cuda-gdb, então só é habilitado em sm70+.

**Armadilha 1:`workStorage`overflow.** `workStorage`O tamanho de`ncclMaxDevWorkBatchBytes()`é`nWorks * workSize`, sm90+ é 16KB. Se`NCCL_MAX_DEV_WORK_BATCH_BYTES`exceder esse valor, haverá escrita fora dos limites. No código-fonte, o tamanho do batch é limitado no lado host por meio de

**, mas não há verificação adicional no lado do dispositivo. Se a restrição do lado host for contornada (por exemplo, modificando variáveis de ambiente), isso causará acesso fora dos limites na memória compartilhada.`__syncthreads()`Armadilha 2:**A ausência de`loadWorkBatchToShmem`causa corrida de dados.`__syncthreads()`Depois de`workStorage`, deve haver um[FACT:src/device/common.h:479]para que todas as threads vejam o`__syncthreads(); // publish ncclShmem`completo. No código-fonte, em`workStorage`há

**. Se essa sincronização for removida, algumas threads podem começar a ler antes de** `while (ncclShmem.aborted == 0)`terminar de escrever, resultando na leitura de dados inválidos.

# Armadilha 3: o momento da verificação de abort.

## O abort só é verificado no início de cada batch. Se um batch demorar muito para executar, o sinal de abort pode levar muito tempo para entrar em vigor. Isso é um trade-off de design: verificações mais frequentes aumentam o overhead, mas respondem mais rápido.

`generate.py`Seleção de variantes de kernel: como generate.py gera a lista de kernels

Modelo intuitivo`generate.py`O papel de

## é semelhante ao de um "planejador de linha de produção de uma fábrica de automóveis". Ele enfrenta um enorme espaço combinatório (7 tipos de operações de conjunto × 5 tipos de operações de redução × 12 tipos de dados × 7 algoritmos × 3 protocolos) e precisa decidir: quais combinações precisam gerar kernels dedicados? Quais podem compartilhar um kernel genérico?

`generate.py`Se cada combinação gerar um kernel, o tempo de compilação e o tamanho do binário explodirão. Se apenas um kernel genérico for gerado, a execução ficará mais lenta devido a chamadas por ponteiro de função e avaliação de branches.

1. **`device_table.cu`**A solução de`ncclDevFuncTable`é o "kernel representativo": gerar um kernel para cada classe de equivalência e distribuir em tempo de execução por meio de uma tabela de ponteiros de função.

2. **`host_table.cc`**Estruturas de dados e layout de memória`ncclDevKernelList`、`ncclDevKernelForFunc`、`ncclDevFuncRowToId`gera três arquivos principais:

3. **: o`<coll>_<op>_<ty>.cu`**do lado do dispositivo, que mapeia funcId para funções específicas do dispositivo.

## Step-by-Step Walkthrough

**: as tabelas**

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

Cada`ncclDevFuncId()`: implementações concretas de kernel.

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

`ncclDevFuncId`Copiar`ncclDevFuncRowToId`Essa ordem de enumeração deve corresponder à fórmula de cálculo de`AllReduce Sum i32`:`AllReduce Sum u32`Copiar

**O que**

[FACT:src/device/generate.py:211-225]

```python
func_rows = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}
kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))
```

`equivalent_primary`. A razão desse mapeamento é que muitas linhas podem mapear para a mesma função principal (por exemplo, todas as linhas de

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

`best_kernel`).`AllGather`Segundo passo: calcular as funções principal e de kernel.`AllGather RING LL`）：

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

**mapeia inteiros com sinal para inteiros sem sinal (porque adição/multiplicação são iguais para ambos):**

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

`DEFINE_ncclDevKernel`mapeia várias funções principais para o mesmo kernel (por exemplo, todos os algoritmos de

[FACT:src/device/common.h:507-509]

```c
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain, algo, proto>>(&args4K.args); \
  }
```

Copiar`__global__`Terceiro passo: gerar a definição do kernel.`ncclKernelMain`Copiar`specializedFnId`Após a expansão da macro`RunWorkBatch<coll, ty, redop<ty>, algo, proto>`。

## , fica:

> **[Design Inference & Architectural Trade-offs]**
> **Portanto, cada kernel é uma função**, chamando

**, com parâmetros de template`NCCL_EXACT_KERNEL_NAMES`e**Reflexões de design e armadilhas em produção`best_kernel`〔Inferência de design e trade-offs arquiteturais〕

**Por que usar "kernel representativo" em vez de um kernel para cada combinação?`required_cuda`Trade-off entre tempo de compilação e tamanho do binário. O espaço combinatório completo é 7 × 5 × 12 × 7 × 3 ≈ 8820 kernels, cada kernel leva alguns segundos para compilar, totalizando várias horas. Além disso, o tamanho do binário chegaria a centenas de MB. Ao mapear para kernels representativos, o número real de kernels gerados é reduzido para algumas dezenas.**Armadilha 1:

[FACT:src/device/generate.py:130-154]

causa explosão de compilação.
