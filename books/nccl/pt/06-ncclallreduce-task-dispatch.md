# Capítulo 6: Panorama do despacho de operadores: como ncclAllReduce se torna uma tarefa de kernel executável

No capítulo anterior percorremos o módulo de tuning e sabemos que o NCCL seleciona, em nível de microssegundos, a combinação (algoritmo, protocolo, channel, warp) para uma comunicação coletiva. Mas o resultado da seleção em si é apenas um monte de números — ele precisa ser "traduzido" em objetos de descrição de tarefas que os kernels da GPU possam entender, para só então ser realmente executado. Este capítulo entra no corpo principal de src/enqueue/enqueue.cc e responde a uma pergunta central: quando o usuário chama ncclAllReduce, o que exatamente acontece no lado do host? De ncclAllReduce até ncclEnqueueCheck, passando por validação de parâmetros, determinação de algoritmo/protocolo, divisão de channels, e finalmente gerando as estruturas ncclInfo e ncclTaskColl. Este é o capítulo-chave do livro para mudar da "perspectiva do usuário" para a "perspectiva da engine". Se compararmos o NCCL a um restaurante, então o módulo enqueue é o "sistema de pedidos do balcão": o usuário (camada de aplicação) diz "quero um AllReduce", e o balcão o traduz em uma ordem de serviço que a cozinha (kernel da GPU) pode executar — qual fogão, qual panela usar, em quantos lotes fazer. Sem essa camada de tradução, a cozinha não saberia qual prato preparar.

# I. Entrada: como ncclAllReduce constrói ncclInfo

## Modelo intuitivo

`ncclAllReduce`É a função de API chamada diretamente pelo usuário. Sua responsabilidade é extremamente única:**Empacotar os parâmetros brutos passados pelo usuário em uma`ncclInfo`estrutura, e então entregá-la a`ncclEnqueueCheck`**. Isso é como ir ao balcão do banco para tratar um assunto: o atendente primeiro preenche sua demanda em um formulário padrão e depois o encaminha ao sistema de retaguarda.

Sem essa camada, cada API de comunicação coletiva teria que lidar por conta própria com validação de parâmetros, semântica de group e instrumentação de profiler — o código se repetiria a ponto de ser impossível de manter.

## Estrutura de dados: layout de memória de ncclInfo

`ncclInfo`É o veículo central que percorre todo o fluxo de enqueue. Sua definição está em`src/include/info.h`：

[FACT:src/include/info.h:17-44]

Esta estrutura tem mais de 20 campos, que podemos dividir em quatro grupos por função:

| Grupo de campos | Campo | Função |
| --- | --- | --- |
| Parâmetros de comunicação coletiva | `coll`, `sendbuff`, `recvbuff`, `count`, `datatype`, `op`, `root` | Descreve "o que fazer" |
| Domínio de comunicação e stream | `comm`, `stream` | Descreve "onde fazer" |
| Detalhes do algoritmo | `chunkSteps`, `sliceSteps` | Descreve "como dividir" |
| Operações unilaterais | `peerWinOffset`, `peerWin`, `sigIdx`, `ctx`, `flags`, `nDesc`, `signalDescs` | Exclusivo para RMA |
| Configuração do usuário | `collConfig` | Cópia privada copiada da config do usuário |

Observe o comentário de`collConfig`:**"A config copied from config passed by user so older user config can be safely accessed during synchronous host scheduling (never at launch/replay)"** [FACT:src/include/info.h:41-43]. Este é um design crucial — o ponteiro de config passado pelo usuário pode ser destruído antes de`ncclGroupEnd`, então o NCCL faz uma cópia em`ncclInfo`.

## Step-by-Step: a cadeia de chamadas de ncclAllReduce

Tomemos`ncclAllReduce`como exemplo e rastreemos o caminho completo da chamada do usuário até a construção de`ncclInfo`.

**Passo 1: o usuário chama ncclAllReduce.**A entrada está em`src/collectives.cc`：

[FACT:src/collectives.cc:206-211]

Aqui três coisas são feitas:

1. `NVTX3_FUNC_WITH_PARAMS`Marcar NVTX (para visualização em ferramentas como Nsight)

2. Chamar`ncclAllReduceConfigImpl`, passando`config = nullptr`

3. Retornar o resultado

**Passo 2: ncclAllReduceConfigImpl constrói ncclInfo.**Este é o passo crucial:

[FACT:src/collectives.cc:192-202]

Observe que aqui é usada inicialização agregada no estilo C:

```c
struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                        sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
                        ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
```

Os campos correspondem um a um à ordem de declaração de`ncclInfo`.`ALLREDUCE_CHUNKSTEPS`e`ALLREDUCE_SLICESTEPS`definidos em`src/include/collectives.h`：

[FACT:src/include/collectives.h:19-20]

`NCCL_STEPS`é o número de passos no buffer circular (geralmente 8 ou 16), então o chunkSteps do AllReduce é`NCCL_STEPS/2`, e o sliceSteps é`NCCL_STEPS/4`. Isso significa que um chunk contém 2 slices.

**Passo 3: analisar a config do usuário.** `ncclParseCollConfig`Analisa o`ncclCollConfig_t*`passado pelo usuário em`info.collConfig`. Se`config == nullptr`, este campo permanece inicializado com zero.

**Passo 4: entregar a ncclEnqueueCheck.**Esta é a verdadeira entrada do módulo enqueue.

## Reflexão de design: por que usar inicialização agregada em vez de atribuição campo a campo?

> **[Design Inference & Architectural Trade-offs]**
> A inicialização agregada tem duas vantagens: primeiro, o compilador verifica se o número de campos corresponde (faltar um campo gera aviso); segundo, o código é mais compacto. Mas a desvantagem é que**a ordem dos campos deve ser estritamente consistente com a declaração da estrutura**— se alguém inserir um campo no meio de`ncclInfo`, todos os pontos de inicialização agregada ficarão silenciosamente desalinhados. Este é um risco implícito de manutenção no código do NCCL.

## Armadilha em produção: ciclo de vida da config

Um cenário real de armadilha: o usuário escreve o código assim:

```c
ncclCollConfig_t config = {...};
ncclAllReduceConfig(..., &config);
// config 在这里被销毁（比如是栈变量，函数返回了）
```

Se o NCCL não copiasse a config em`ncclInfo`, então ao acessar`ncclGroupEnd`em`info.collConfig`seria lida memória já liberada.`src/include/info.h:41-43`O comentário de**serve justamente para explicar este design —**。

---

# a config é analisada e copiada já na fase de task append, e depois não depende mais do ponteiro do usuário

## II. ncclEnqueueCheck: validação de parâmetros e semântica de group

`ncclEnqueueCheck`Modelo intuitivo**É o "portão principal" do módulo enqueue. Todas as APIs de comunicação coletiva acabam convergindo para cá. Sua responsabilidade é:**validar a legalidade dos parâmetros, tratar a semântica de group e chamar taskAppend para gerar tarefas`ncclEnqueueCheck`。

. Se compararmos com a segurança de um aeroporto, então cada função de API é o balcão de check-in — o check-in apenas recebe a bagagem, a segurança de verdade está em

## Passo a passo: o fluxo de execução do ncclEnqueueCheck

[FACT:src/enqueue/enqueue.cc:3478-3527]

Vamos decompor passo a passo:

**Passo 1: CommCheck valida o domínio de comunicação.** `CommCheck(info->comm, info->opName, "comm")`Verifica se o ponteiro comm é não nulo e se foi inicializado. Se comm foi revogado (por exemplo, algum rank com erro), retorna erro diretamente:

[FACT:src/enqueue/enqueue.cc:3480-3485]

**Passo 2: Trata a profundidade do profiler.**Se já estiver dentro de um group (`profilerGroupDepth > 0`), incrementa o contador de profundidade. Isso serve para tratar corretamente chamadas implícitas de`ncclGroupStartInternal`/`ncclGroupEndInternal`.

**Passo 3: Entra no group interno.** `ncclGroupStartInternal()`É o mecanismo interno de group do NCCL.**Ponto-chave**: mesmo que o usuário não chame explicitamente`ncclGroupStart`, o NCCL cria um group implícito para cada chamada de API. Isso garante a atomicidade de uma única chamada.

**Passo 4: Garante que comm esteja pronto.** `ncclCommEnsureReady(info->comm)`Aguarda a conclusão da inicialização do domínio de comunicação (por exemplo, conclusão do bootstrap, estabelecimento de conexões).

**Passo 5: ArgsCheck valida parâmetros.**Esta é a etapa de validação mais complexa:

[FACT:src/enqueue/enqueue.cc:3497-3503]

Atenção ao tratamento de`checkMode`: se for`ncclCheckModeDebugGlobal`，`ArgsCheck`, enfileira info e faz a validação global em`ncclGroupEnd`(por exemplo, verificar se o count de todos os ranks é consistente).

**Passo 6: Chama taskAppend.**Esta é a etapa central de conversão:

[FACT:src/enqueue/enqueue.cc:3513]

**Passo 7: Incrementa opCount.**Após cada enfileiramento bem-sucedido,`comm->opCount++`. Esse contador é usado para casar operações send/recv e também é a base da linha do tempo do profiler.

**Passo 8: Sai do group.** `ncclGroupEndInternal()`Se depth cair para 0, dispara a operação real de group (escalonamento, lançamento de kernel).

## Controle de concorrência: semântica de group e thread safety

> **[Design Inference & Architectural Trade-offs]**
> `ncclGroupStartInternal`/`ncclGroupEndInternal`Usa thread-local storage (TLS) para manter o estado do group. Isso significa que**múltiplas chamadas de API na mesma thread serão combinadas em um único group**, mas chamadas de threads diferentes são independentes. Essa é a base do suporte do NCCL a múltiplas threads.

Uma armadilha fácil de encontrar: se o usuário chamar uma API CUDA que não é do NCCL entre`ncclGroupStart`e`ncclGroupEnd`(por exemplo,`cudaMemcpy`), pode causar problemas de ordem de stream. O mecanismo de group do NCCL assume que as operações dentro do group estão no mesmo conjunto de streams.

## Cadeia de recuperação de erros

`ncclEnqueueCheck`O tratamento de erros de

[FACT:src/enqueue/enqueue.cc:3524-3526]

tem um design engenhoso:`taskAppend`Se`ncclCommSetAsyncError`falhar e comm estiver em modo não bloqueante, chama

---

# para registrar o erro. Assim, chamadas de API subsequentes retornarão erro imediatamente, em vez de continuar tentando. Esse é o mecanismo de propagação assíncrona de erros.

## Três, taskAppend: a encruzilhada da distribuição de tarefas

`taskAppend`Modelo intuitivo`info->coll`É o "hub de tráfego" do módulo enqueue. Com base no valor de

, distribui tarefas para diferentes caminhos de processamento: P2P, RMA, CE ou comunicação coletiva comum. É como um centro de triagem dos correios — de acordo com o endereço no envelope, entrega a carta em diferentes caixas postais.

## Sem essa camada de distribuição, todos os tipos de operação teriam que se espremer em um enorme if-else, e o código seria difícil de manter.

[FACT:src/enqueue/enqueue.cc:3337-3476]

**Passo a passo: a lógica de distribuição do taskAppend** `ncclParamEnqueueRearchEnable()`Passo 1: Determina se a nova arquitetura está habilitada.`rawTaskAppend`É uma flag de variável de ambiente (padrão 0). Se habilitada, segue o caminho

**— este é o novo modelo de tarefas que o NCCL está desenvolvendo.**Passo 2: Distribuição P2P.`p2pTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3343-3345]

**Se for Send/Recv, chama**Passo 3: Distribuição RMA.`rmaTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3346-3347]

**Se for PutSignal/Signal/WaitSignal, chama** `if (info->count == 0) return ncclSuccess;`Passo 4: Retorno antecipado para comunicação coletiva vazia.

**— comunicação coletiva com count 0 é descartada diretamente.** `ncclCollConfigGetAlgMask`Passo 5: Validação da seleção de algoritmo.

[FACT:src/enqueue/enqueue.cc:3357-3358]

**Valida se a seleção de algoritmo passada pelo usuário é legal:**Passo 6: Verificação de tipo FP8.

[FACT:src/enqueue/enqueue.cc:3360-3366]

**Redução FP8 requer sm90+:** `hostToDevRedOp`Passo 7: Conversão da operação de redução.`ncclRedOp_t`Converte o`ncclDevRedOpFull`：

[FACT:src/enqueue/enqueue.cc:3370-3371]

**do lado host para o**do lado dispositivo`comm->nRanks == 1`Passo 8: Retorno antecipado para rank único.`ncclLaunchOneRank`Se

[FACT:src/enqueue/enqueue.cc:3373-3377]

**, chama diretamente**para executar a redução local, sem necessidade de gerar tarefa:

[FACT:src/enqueue/enqueue.cc:3378-3470]

## Passo 9: Caminho multi-rank.

`collTaskAppend`Este é o ramo mais complexo, incluindo roteamento CE, degradação de AllToAll/Gather/Scatter e comunicação coletiva comum:`ncclTaskColl`Estrutura de dados: campos de ncclTaskColl

[FACT:src/enqueue/enqueue.cc:2757-2851]

É onde

| é gerado. Vejamos sua lógica central: | Atribuição de campos-chave: | Campo |
| --- | --- | --- |
| `func` | `info->coll` | Origem |
| `sendbuff`/`recvbuff` | `info->sendbuff`/`recvbuff` | Significado |
| `count` | `info->count` | Tipo de comunicação coletiva |
| `datatype` | `info->datatype` | Ponteiro de buffer |
| `trafficBytes` | `count * elementSize * ncclFuncTrafficPerByte` | Número de elementos |
| `opHost`/`opDev` | `info->op`/`opDev` | Tipo de dados |
| `chunkSteps`/`sliceSteps` | `info->chunkSteps`/`sliceSteps` | Estimativa de tráfego |
| `minCTAs`/`maxCTAs`/`nvlsCTAs` | Operação de redução | Número de passos de divisão |
| `algMask` | `ncclCollConfigGetAlgMask` | Análise de configuração |

Limite de recursos`trafficBytes`Máscara de seleção de algoritmo

[FACT:src/enqueue/enqueue.cc:2813]

`ncclFuncTrafficPerByte`Atenção ao cálculo de

[FACT:src/enqueue/enqueue.cc:123-134]

:

## retorna o multiplicador de tráfego de cada tipo de comunicação coletiva:

[FACT:src/enqueue/enqueue.cc:2808-2812]

AllReduce retorna 2 (porque precisa de reduce + broadcast), AllGather/ReduceScatter retorna nRanks, os outros retornam 1.`ncclInt8`. Esta é uma otimização:**Essas duas operações não envolvem redução, então não é necessário se preocupar com o tipo de dado; processar uniformemente por bytes pode simplificar a lógica do kernel**。

## Armadilha em produção: a ordem de parsing do CTAPolicy

[FACT:src/enqueue/enqueue.cc:3390-3397]

O parsing do CTAPolicy tem uma prioridade sutil:**env > per-call > comm**. E além disso`NCCL_CTA_POLICY_ZERO`tem prioridade sobre`NCCL_CTA_POLICY_EFFICIENCY`. Se o usuário definir ambos os flags ao mesmo tempo, ZERO entrará em vigor.

Um cenário real de armadilha: o usuário definiu`NCCL_CTA_POLICY=EFFICIENCY`, mas descobriu que o caminho CE não estava sendo usado. A razão é que o roteamento CE exige que`CTAPolicy & NCCL_CTA_POLICY_ZERO`seja verdadeiro, e EFFICIENCY não satisfaz essa condição.

---

# Quatro, ncclPrepareTasks: da lista de tarefas à fila de agendamento

## Modelo intuitivo

`ncclPrepareTasks`é o "pré-processador" do módulo enqueue. Ele agrupa a lista dispersa de tarefas por (func, op, datatype) em buckets e, em seguida, calcula o algoritmo e o protocolo para cada bucket. Isso é como um bibliotecário — primeiro organiza os livros devolvidos por categoria e depois decide em qual estante cada categoria de livro será colocada.

Sem esta etapa, o`scheduleCollTasksToPlan`subsequente teria que calcular o algoritmo individualmente para cada tarefa, com eficiência extremamente baixa.

## Passo a passo: a lógica de bucketing do ncclPrepareTasks

[FACT:src/enqueue/enqueue.cc:423-642]

**Etapa 1: Conversão de tarefas Broadcast.**Se houver apenas um broadcast peer, converta a tarefa broadcast em tarefa coll:

[FACT:src/enqueue/enqueue.cc:430-461]

Observe que aqui os campos de`bcastTask`são copiados para o novo`ncclTaskColl`, e calcula-se`trafficBytes`. Em seguida, a partir de`memPool_ncclTaskBcast`libera-se a tarefa original.

**Etapa 2: Bucketing por (func, op, datatype).**As tarefas saem do sorter em ordem decrescente de size e então são distribuídas no`tasksByFnOpTy`array:

[FACT:src/enqueue/enqueue.cc:464-487]

Cálculo do índice:`((int)task->func * ncclNumDevRedOps + (int)task->opDev.op) * ncclNumTypes + (int)task->datatype`. Esta é a linearização de um array tridimensional.

**Etapa 3: Agregação e seleção de algoritmo.**Para cada bucket, agregam-se tarefas de tamanho semelhante (dentro de 4 vezes) e então chama-se`ncclGetAlgoInfo`：

[FACT:src/enqueue/enqueue.cc:503-547]

**Etapa 4: Bucketing por (collnet, nvls).**De acordo com o tipo de algoritmo, distribuem-se as tarefas em`collBins[2][2]`：

[FACT:src/enqueue/enqueue.cc:517-544]

**Etapa 5: Concatenação da fila final.**Concatenam-se os quatro buckets em`planner->collTaskQueue`：

[FACT:src/enqueue/enqueue.cc:553-557]

## Estrutura de dados: ncclTaskCollSorter

`ncclTaskCollSorter`é um ordenador por inserção que ordena por`trafficBytes`.`ncclTaskCollSorterInsert`insere a tarefa na posição correta,`ncclTaskCollSorterDequeueAll`retira todas as tarefas em ordem.

> **[Design Inference & Architectural Trade-offs]**
> A motivação de design deste ordenador é:**Tarefas grandes são agendadas primeiro**. Como tarefas grandes têm tempo de transmissão longo, iniciá-las primeiro permite sobrepor melhor computação e comunicação.

## Controle de concorrência: runtimeConn e estabelecimento de conexão

[FACT:src/enqueue/enqueue.cc:572-583]

Se`comm->runtimeConn`for verdadeiro (modo de conexão em runtime), e o channel de algum algoritmo ainda não tiver sido inicializado, marca-se`algoNeedConnect`. Isso disparará o estabelecimento de conexão posteriormente.

## Armadilha em produção: condições de contorno da agregação

[FACT:src/enqueue/enqueue.cc:507-508]

A condição de agregação é`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes`, e ambas as tarefas não definem`aggIsolate`. Se o usuário definir per-call config (por exemplo,`maxCTAs`），`aggIsolate`será definido como true, essa tarefa não será agregada.

Um cenário real de armadilha: o usuário definiu para um certo AllReduce`maxCTAs=4`, esperando que ele usasse apenas 4 CTAs. Mas, devido à lógica de agregação, essa tarefa pode ser mesclada com tarefas adjacentes, fazendo com que o número real de CTAs usados não corresponda ao esperado. A solução é definir`aggIsolate`— o NCCL já tratou disso em`collTaskAppend`:

[FACT:src/enqueue/enqueue.cc:2821-2822]

---

# Cinco, scheduleCollTasksToPlan: divisão de channel e controle de orçamento

## Modelo intuitivo

`scheduleCollTasksToPlan`é o "agendador" do módulo enqueue. Ele distribui as tarefas para channels específicos e calcula a divisão de dados de cada channel. Isso é como o sistema de programação de produção de uma fábrica — decide o que cada linha de produção fará e quanto fará.

Sem esta etapa, o kernel da GPU não saberia qual parte dos dados deve processar.

## Passo a passo: algoritmo de divisão de channel

[FACT:src/enqueue/enqueue.cc:644-947]

**Etapa 1: Estimativa de orçamento.**Primeiro estima-se a quantidade de tarefas que podem caber neste plan:

[FACT:src/enqueue/enqueue.cc:648-689]

`ncclTestBudget`Verifica-se se o número de bytes de trabalho excede o orçamento:

[FACT:src/enqueue/enqueue.cc:343-349]

**Etapa 2: Calcular o tráfego de cada channel.**De acordo com o kind (collnet/nvls), calcula-se`trafficPerChannel`：

[FACT:src/enqueue/enqueue.cc:701-707]

**Etapa 3: Caminho Collnet.**Se for um algoritmo collnet, a alocação de channel é relativamente simples:

[FACT:src/enqueue/enqueue.cc:709-739]

**Etapa 4: Divisão em cells do caminho comum.**Esta é a parte mais complexa. O NCCL divide os dados em "cells", e cada cell é uma unidade mínima de transmissão:

[FACT:src/enqueue/enqueue.cc:740-845]

Variáveis-chave:

- `cellSize`: número de bytes por cell, no mínimo`MinTrafficPerChannel`（32KB）
- `cells`: número total de cells
- `cellsPerChannel`: número de cells processadas por channel
- `cellsLo`/`cellsHi`: número de cells dos channels inicial e final (pode não estar cheio)

**Etapa 5: Calcular chunkGrains.**Para cada segmento de channel, chama-se`calcCollChunking`：

[FACT:src/enqueue/enqueue.cc:811-825]

**Etapa 6: Gerar proxyOp.**Gera-se uma operação proxy para cada channel:

[FACT:src/enqueue/enqueue.cc:844-894]

## Estrutura de dados: ncclDevWorkColl

`ncclDevWorkColl`é o descritor de trabalho do lado do dispositivo. Seus campos-chave:

| Campo | Significado |
| --- | --- |
| `sendbuff`/`recvbuff` | Ponteiro de buffer |
| `channelLo`/`channelHi` | Intervalo de channel |
| `cbd.countLo`/`countMid`/`countHi` | Número de elementos de cada segmento |
| `cbd.chunkGrainsLo`/`Mid`/`Hi` | Granularidade de chunk de cada segmento |
| `direct` | Flag direto |

## Controle de concorrência: operação bit a bit de channelMask

[FACT:src/enqueue/enqueue.cc:897]

Esta linha de código define channelMask com operação bit a bit:`(2ull << channelHi) - (1ull << channelLo)`. Por exemplo, channelLo=2, channelHi=5, o resultado é`(2<<5) - (1<<2) = 64 - 4 = 60 = 0b111100`, ou seja, os bits 2-5 são definidos.

## Armadilha em produção: estouro de orçamento

[FACT:src/enqueue/enqueue.cc:792-794]

Se o orçamento não for suficiente, retorna diretamente`ncclSuccess`, deixando o loop externo criar um novo plan. Esta é uma estratégia de degradação elegante——**não gera erro, apenas processa em lotes**。

Um cenário real de armadilha: se`NCCL_WORK_FIFO_BYTES`for definido muito pequeno, cada plan só conseguirá acomodar poucas tarefas, aumentando o número de inicializações de kernel e reduzindo o desempenho.

---

# Seis, finishPlan: das tarefas aos parâmetros do kernel

## Modelo intuitivo

`finishPlan`é o "empacotador" do módulo enqueue. Ele empacota tarefas, batch e proxyOp em uma estrutura de parâmetros que o kernel pode ler diretamente. Isso é como empacotar uma encomenda——colocar itens soltos em uma caixa, colar a etiqueta de envio e aguardar o despacho.

## Passo a Passo: a lógica de empacotamento do finishPlan

[FACT:src/enqueue/enqueue.cc:236-330]

**Passo 1: decidir o tipo de armazenamento.**Se todo o trabalho puder caber em kernel args, use`ncclDevWorkStorageTypeArgs`：

[FACT:src/enqueue/enqueue.cc:244-250]

**Passo 2: alocar kernelArgs.**Alocar da pilha de memória:

[FACT:src/enqueue/enqueue.cc:251-255]

**Passo 3: posicionar batches em round-robin.**O primeiro batch de cada channel deve ser colocado em`batchZero[blockIdx.x]`：

[FACT:src/enqueue/enqueue.cc:257-280]

**Passo 4: mesclar as filas de proxyOp.**Ordenar por merge com base em opCount:

[FACT:src/enqueue/enqueue.cc:282-329]

## Estrutura de dados: ncclDevKernelArgs

`ncclDevKernelArgs`é a estrutura de parâmetros passada ao kernel. Ela contém:

- `comm`: comunicador do lado do dispositivo
- `channelMask`: máscara de bits de channel
- `workStorageType`: tipo de armazenamento de trabalho
- `workBuf`: ponteiro do buffer de trabalho
- `workMask`: máscara do buffer de trabalho

## Armadilha em produção: ordem dos batches

[FACT:src/enqueue/enqueue.cc:257-259]

O comentário deixa claro: "The first batch for each channel must be located at batchZero[blockIdx.x]". Se essa ordem estiver errada, o kernel lerá o batch errado, causando corrupção de dados.

---

# Resumo do capítulo

Neste capítulo, rastreamos o caminho completo de`ncclAllReduce`até`ncclTaskColl`:

1. **ncclAllReduce**constrói`ncclInfo`, empacota os parâmetros do usuário

2. **ncclEnqueueCheck**valida parâmetros, trata a semântica de group

3. **taskAppend**distribui para caminhos diferentes de acordo com o tipo de operação

4. **collTaskAppend**gera`ncclTaskColl`, analisa a configuração

5. **ncclPrepareTasks**agrupa por (func, op, datatype), calcula o algoritmo

6. **scheduleCollTasksToPlan**divide channels, gera`ncclDevWorkColl`

7. **finishPlan**empacota em parâmetros de kernel

Ideias-chave de design:

- **Desacoplamento em camadas**: cada função faz apenas uma coisa, passando estado por meio de`ncclInfo`e`ncclTaskColl`
- **Controle de orçamento**: controla o tamanho de cada plan por meio de`ncclTestBudget`
- **Otimização por agregação**: tarefas de tamanho semelhante são agregadas, reduzindo o número de inicializações de kernel
- **Prioridade de configuração**：env > per-call > comm

No próximo capítulo entraremos em`task_sched`, para ver como a NCCL orquestra a ordem de execução de múltiplos channels e múltiplos kernels.

# Reflexões e autoavaliação deste capítulo

Q1: Se removermos a verificação de`collTaskAppend`em`aggIsolate`(ou seja,`src/enqueue/enqueue.cc:2821-2822`sempre retorna false), em qual cenário a configuração definida pelo usuário`maxCTAs`deixaria de funcionar? Por quê?

**Análise de referência**：`aggIsolate`serve para marcar "esta tarefa não pode ser agregada". Se removermos essa verificação, tarefas com per-call config definido serão mescladas com tarefas adjacentes. No`ncclPrepareTasks`loop de agregação de`src/enqueue/enqueue.cc:507-508`, a condição de agregação é`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes && !aggBeg->aggIsolate && !aggEnd->aggIsolate`. Se`aggIsolate`sempre for false, então mesmo que uma tarefa defina`maxCTAs=4`, ela ainda poderá ser mesclada com uma tarefa`maxCTAs=32`. O`agg`resultante da mesclagem assumirá alguma combinação dos dois (dependendo da implementação de`ncclGetAlgoInfo`), fazendo com que o número real de CTAs usados não corresponda à expectativa do usuário.

Mais grave ainda, em`scheduleCollTasksToPlan`(`src/enqueue/enqueue.cc:665-666`），`taskAggIsolate`é usado para garantir que tarefas com recursos per-call configurados ocupem sozinhas um plan. Se essa verificação falhar, várias tarefas compartilharão o orçamento de channel do plan, fazendo com que a alocação de recursos não corresponda à expectativa.

Q2: Em`ncclEnqueueCheck`, se`ncclGroupEndInternal()`retornar erro (por exemplo, o ArgsCheck de algum rank falhar), mas`taskAppend`já tiver sido executado com sucesso, o que acontece? Como a NCCL garante a consistência de estado?

**Análise de referência**: veja o fluxo de controle de`src/enqueue/enqueue.cc:3513-3519`:

```c
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
info->comm->opCount++;
exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
```

Se`taskAppend`tiver sucesso mas`ncclGroupEndInternal`falhar,`opCount`já foi incrementado. Isso fará com que o opCount das operações subsequentes não corresponda ao do par, podendo causar hang.

A forma como a NCCL trata isso é:`ncclGroupErrCheck(ret)`verificará se há erro e, se houver, definirá o estado de erro da comm. Chamadas de API subsequentes detectarão esse erro por meio de`ncclCommGetAsyncError`e retornarão imediatamente. Esta é uma estratégia de "falha rápida"——uma vez que ocorre um erro, toda a comm entra em estado de erro e não tenta mais se recuperar.

Em ambiente de produção, isso significa que, uma vez ocorrido um erro de group, o usuário precisa destruir e recriar o communicator.

Q3: `scheduleCollTasksToPlan`O algoritmo de divisão de cells em`src/enqueue/enqueue.cc:740-845`) tem uma condição de borda: quando`cellsLo == 0`, ele pula o menor número de channels. Se essa lógica de salto tiver bug (por exemplo,`channelId`não for incrementado corretamente), quais consequências isso causaria?

**Análise de referência**: veja`src/enqueue/enqueue.cc:770-780`：

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

Se`channelId`não for incrementado corretamente, a próxima tarefa começará a alocação a partir do channel errado. Isso causará:

1. **Sobreposição de channels**: duas tarefas podem ser alocadas para o mesmo trecho de dados do mesmo channel

2. **Corrupção de dados**: o kernel processará dados repetidamente ou os omitirá

3. **Queda de desempenho**：desequilíbrio de carga do channel

De forma mais sutil, esse bug pode ser acionado apenas com tamanhos de mensagem específicos (quando`cellsLo == 0`), tornando difícil de reproduzir. O NCCL rastreia os channels já utilizados por meio de`plan->channelMask |= (2ull << devWork->channelHi) - (1ull << devWork->channelLo)`, mas isso é apenas um registro, não impede sobreposição.

Até aqui, vimos como ncclAllReduce se transforma de uma chamada do usuário em uma sequência de tarefas de kernel executáveis: validação de parâmetros, determinação de algoritmo/protocolo, divisão de channels, e finalmente a geração de ncclInfo e ncclTaskColl. Mas criar as tarefas é apenas o primeiro passo — elas ainda precisam ser escalonadas em múltiplos channels, gerar parâmetros de lançamento de kernel, e lidar com submissão em lote e ordenação de dependências sob a semântica de group. O próximo capítulo mergulhará em src/enqueue/task_sched e src/enqueue/task_prep, respondendo "por que um único AllReduce inicia múltiplos kernels, e como a ordem e as dependências entre eles são garantidas", enquanto revela como ncclGroupStart/ncclGroupEnd em src/group.cc combinam múltiplas chamadas de API em uma única submissão.
