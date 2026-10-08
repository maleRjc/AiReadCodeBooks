# 第 24 章：第 24 章：架构演进与未来方向：从静态通信到可编程通信

# 第 24 章：架构演进与未来方向：从静态通信到可编程通信

В предыдущей главе мы увидели, как сообщество строит экосистему вокруг ядра NCCL: привязки Python, привязки Rust, экспертную параллельную коммуникацию, сверхширокополосные примитивы, контрольные точки коммуникации. Все эти проекты используют стабильный API NCCL, но их требования выходят за рамки традиционной коллективной коммуникации — экспертному параллелизму нужен мелкозернистый обмен точка-точка, контрольным точкам нужно приостанавливать/возобновлять состояние коммуникации, сверхширокополосным примитивам нужно обходить стандартные коллективные операции и напрямую работать с сетью. Эти требования указывают на одну проблему: фиксированная модель коллективных операций NCCL разрывается более гибкими потребностями коммуникации. В этой главе мы не будем рассматривать отдельный модуль, а, отталкиваясь от уже появившихся в исходном коде следов эволюции, обсудим, куда движется NCCL. Конкретно, мы разберём три переплетающиеся эволюционные силы: коммуникационные примитивы переходят от фиксированных коллективов к программируемым — планирование задач RMA в src/rma/rma.cc позволяет верхнему уровню комбинировать примитивы Put/Signal/WaitSignal, а не только вызывать AllReduce; инициирование сети переходит от host proxy к прямой отправке с GPU — управление бэкендом GIN в src/gin/gin_host.cc позволяет GPU-ядру напрямую управлять сетевой картой; модель памяти переходит от зарегистрированных буферов к симметричной памяти — выбор ядра симметричной памяти в src/sym_kernels.cc позволяет всем рангам использовать один и тот же набор виртуальных адресов для доступа к буферам друг друга. Эти три силы не изолированы, они используют одну и ту же инфраструктуру: абстракцию team в src/nccl_device/core.cc и версионированный DevComm в src/devcomm/devcomm_v23100.cc. Поняв, как они сцепляются, вы поймёте логику эволюции NCCL от «библиотеки коллективной коммуникации» к «программируемому коммуникационному движку».

# 一、可编程通信原语：RMA 如何把「固定菜谱」变成「自助餐」

## Интуитивная модель

Традиционная коллективная связь NCCL похожа на фиксированный набор: вы заказываете AllReduce, и кухня выполняет весь процесс AllReduce. Но в сценарии экспертного параллелизма (MoE) каждый токен должен быть отправлен разным экспертам, и шаблон отправки вообще неизвестен на этапе компиляции — это как шведский стол, где вы сами решаете, что взять, сколько взять и когда взять.

RMA — это тот самый «шведский стол», который NCCL предоставляет верхнему уровню: Put (записать данные в память удалённой стороны), Signal (уведомить удалённую сторону), WaitSignal (ожидать сигнал от удалённой стороны). Верхнеуровневый фреймворк может свободно комбинировать эти три примитива для реализации произвольных шаблонов связи.

Без RMA all-to-all в MoE можно было бы реализовать только через многократные мелкомасштабные коллективные операции, каждая из которых требует полного запуска ядра и процедуры синхронизации, что даёт неприемлемо высокую задержку.

## Структуры данных и разметка памяти

Ключевая структура данных RMA — это`ncclTaskRma`(описание задачи) и`ncclRmaArgs`(параметры плана). Сначала рассмотрим`ncclRmaArgs`поля, которые инициализируются в`scheduleRmaTasksToPlan`.

[FACT:src/rma/rma.cc:166-171]

```cpp
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
plan->rmaArgs->nRmaTasks = 0;
plan->rmaArgs->nRmaTasksProxy = 0;
plan->rmaArgs->nRmaTasksCe = 0;
```

Здесь ключевые поля —`nRmaTasksProxy`и`nRmaTasksCe`. Они разделяют задачи RMA на два пути выполнения:

- **Путь CE**(Copy Engine, движок копирования): целевой rank находится в пределах LSA (Local Symmetric Access, локальный симметричный доступ), и задача может быть выполнена напрямую с помощью движка копирования GPU без использования сети.
- **Путь Proxy**: целевой rank находится вне зоны LSA, и задача должна управляться сетевым взаимодействием через поток host proxy.

> **[Design Inference & Architectural Trade-offs]**
> Мотивация такого дихотомического подхода очевидна: связь в пределах LSA идёт через NVLink или PCIe с высокой пропускной способностью и низкой задержкой, и асинхронное копирование через CE здесь наиболее выгодно; межмашинная связь обязательно должна идти через сетевой адаптер и может управляться только потоком proxy. Раздельное планирование двух типов задач позволяет CE и proxy выполняться параллельно, а не последовательно ждать друг друга.

`ncclTaskRma`Сам`peers`、`nsignals`、`signalIdxs`содержит три указателя на массивы

## , которые соответственно хранят удалённый rank, количество сигналов и индекс сигнала. Для задач WaitSignal одна задача может ожидать несколько peer; для задач Put/Signal одна задача нацелена только на один peer.

Пошаговый разбор: планирование одного WaitSignal`ncclWaitSignal`Рассмотрим конкретный сценарий: rank 0 вызывает

**, ожидая сигналы от rank 1 и rank 3. Предположим, rank 1 находится в пределах LSA, а rank 3 — нет.**

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

Копировать

**Задачи RMA распределяются по очередям в зависимости от context, каждый context — это независимый канал RMA. Здесь находится первый context с задачами и извлекается его очередь.**

[FACT:src/rma/rma.cc:163-168]

```cpp
struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
```

`firstTask->func`Копировать`ncclFuncWaitSignal`Если

**равно**

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

`isLsaAccessible`Шаг третий: разделить peer по доступности через LSA.`comm->devrState.lsaRankList`Копировать

**Происходит обход**

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

Шаг четвёртый: создать по одной новой задаче для CE и Proxy.

**Копировать**

[FACT:src/rma/rma.cc:249-251]

```cpp
planner->nTasksRma -= 1;
ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
```

Шаг пятый: освободить исходную задачу.

## Копировать

Исходная задача уже разделена на две новые задачи и возвращается в пул памяти.`ncclRmaWaitSignal`Управление параллелизмом и взаимодействие с оборудованием

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

.

> **[Design Inference & Architectural Trade-offs]**
> Этот код использует CUDA event для синхронизации между потоками: сначала event записывается во входном потоке, поток CE ожидает этот event, затем в двух потоках分别 запускаются задачи proxy и CE, и наконец входной поток ожидает event потока CE. Таким образом, оба пути продвигаются параллельно, но внешне это выглядит как одна синхронная операция.

## 〔Проектные соображения и архитектурные компромиссы〕

**Здесь компромисс таков: параллельное выполнение снижает задержку, но добавляет накладные расходы на запись event и синхронизацию потоков. Для малых сообщений эти накладные расходы могут превысить выгоду от параллелизма; для больших сообщений выгода от параллелизма значительна. NCCL не делает здесь адаптивного выбора, а всегда идёт по параллельному пути — потому что типичный сценарий RMA — это мелкогранулярная связь с большими сообщениями.** `isLsaAccessible`Руководство по избежанию проблем в production`lsaRankList`Проблема 1: ошибка в определении доступности LSA приводит к выбору неправильного пути для задачи.`lsaSize`Происходит обход`scheduleRmaTasksToPlan`, и если`nRmaTasksProxy`равно 0 (например, домен связи с одним rank), все peer будут признаны недоступными и все пойдут по пути Proxy. Это не проявится при мелкомасштабном тестировании, но при крупномасштабном развёртывании приведёт к резкому падению производительности. Метод диагностики — посмотреть в INFO-логах`nRmaTasksCe`соотношение

**и**.`peersCe`Проблема 2: жизненный цикл массива peer после разделения задачи WaitSignal.`ncclMemoryStackAlloc`В пути CE`comm->memScoped`использует`peersProxy`для выделения, и жизненный цикл следует за`ncclCalloc`; в пути Proxy`free`. Если создание задачи Proxy завершается неудачей,`fail`ветка освобождает эти массивы.

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

**Ловушка 3: пакетная обработка задач Put/Signal между контекстами.**В ветке Put/Signal NCCL объединяет задачи put/signal всех контекстов в один план, но останавливается при обнаружении WaitSignal.

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

Замысел этого дизайна: один запуск ядра охватывает put/signal всех контекстов, снижая накладные расходы на запуск. Но очередь каждого контекста потребляется только до первого WaitSignal, что гарантирует порядок FIFO для каждого контекста. Если верхний уровень чередует вызовы put и waitSignal в одном контексте, эффект пакетной обработки сильно снижается — это паттерн, который нужно учитывать при использовании RMA.

---

# II. Прямая отправка в сеть с GPU: как GIN позволяет ядру обойти host proxy

## Интуитивная модель

Традиционная сетевая коммуникация NCCL похожа на отправку письма: ядро GPU помещает данные в буфер, поток host proxy передаёт данные сетевой карте, карта отправляет их. GIN же позволяет ядру GPU напрямую опустить письмо в почтовый ящик получателя — ядро напрямую пишет в очередь отправки сетевой карты, а карта напрямую читает память GPU.

Без GIN каждая сетевая коммуникация должна проходить через промежуточную память host, что добавляет как минимум один цикл PCIe к задержке. Для такой мелкозернистой коммуникации, как MoE, эта задержка критична.

## Структуры данных и разметка памяти

Основное состояние GIN — это`ncclGinState`, оно управляет несколькими бэкендами (backend) и несколькими DevComm. Сначала рассмотрим таблицу совместимости версий бэкендов.

[FACT:src/gin/gin_host.cc:27-33]

```cpp
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5), NCCL_VERSION(2, 32, 0)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
constexpr int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0), NCCL_VERSION(2, 32, 0)};
```

Индексом этих массивов является номер версии бэкенда, а значением — минимальная совместимая версия NCCL. Например,`proxyBackendMinVersions[3]`соответствует версии бэкенда 3 и требует NCCL не ниже 2.32.0. Этот дизайн позволяет NCCL во время выполнения выбирать подходящую версию бэкенда в зависимости от версии кода устройства, а не привязываться на этапе компиляции.

> **[Design Inference & Architectural Trade-offs]**
> Мотивация такого дизайна таблицы совместимости версий: темпы эволюции версий бэкенда GIN (драйвер сетевой карты, прошивка) и библиотеки NCCL различаются. Если жёстко закодировать требования к версиям, любое обновление одной из сторон приведёт к несовместимости. Использование массивов для сопоставления версий позволяет динамически выбирать во время выполнения, обеспечивая обратную совместимость со старыми бэкендами.

`ncclGinStateDevComm`— это состояние GIN каждого DevComm, содержащее`contextCount`、`backendIndex`、`ginCtx[]`、`devHandles[]`и другие поля. Оно связывается в связный список и прикрепляется к`ginState->devComms`.

## Пошаговый разбор: установка одного GIN-соединения

Представим сценарий: rank 0 инициализирует коммуникационный домен и должен установить GIN-соединение.

**Шаг первый: проверка, включён ли и поддерживается ли GIN.**

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

`ncclParamGinEnable()`читает переменную окружения`NCCL_GIN_ENABLE`, по умолчанию 1. Если пользователь явно отключил, сразу возвращается ошибка.

**Шаг второй: проверка поддержки симметричной памяти.**

[FACT:src/gin/gin_host.cc:111-114]

```cpp
if (!comm->symmetricSupport) {
  WARN("Communicator does not support symmetric memory!");
  return ncclInternalError;
}
```

GIN зависит от симметричной памяти — поскольку ядру GPU нужно знать виртуальный адрес буфера удалённой стороны, только симметричная память гарантирует совпадение адресов.

**Шаг третий: получение списка локальных устройств GIN.**

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

`ncclTopoGetLocalGinDevs`находит в топологической схеме все сетевые карты, поддерживающие GIN. Если их больше`NCCL_GIN_MAX_CONNECTIONS`, берутся только первые несколько с выводом предупреждения.

**Шаг четвёртый: вычисление команды GIN.**

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

Каждый бэкенд сначала вызывает`devices`для получения количества устройств, затем для каждого соединения выполняет流程 listen→getProperties→allGather→connect→closeListen.`bootstrapAllGather`обменивается handle между всеми rank, так что каждый rank знает информацию о соединении удалённой стороны.

## Управление конкурентностью и взаимодействие с оборудованием

Поток прогресса GIN — это основной механизм конкурентности.

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

Здесь есть несколько ключевых решений:

1. **Привязка к CPU**：`ncclOsSetAffinity`привязывает поток прогресса к указанному ядру CPU, избегая инвалидации кэша из-за миграции потока.

2. **Отступление при блокировке записи**：`writePending`— это атомарный флаг; главный поток перед изменением`devComms`связного списка сначала устанавливает его, а поток прогресса, увидев это, добровольно уступает, избегая конкуренции за блокировку.

3. **Блокировка чтения-записи**：`devCommRwMutex`— это`shared_timed_mutex`, поток прогресса удерживает блокировку чтения при обходе списка, главный поток удерживает блокировку записи при изменении списка.

4. **Разделение труда потоков**: поток t отвечает за соединения t, t+proxyNthreads, t+2*proxyNthreads, ..., балансировка нагрузки достигается через stride-цикл.

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

Эта реализация блокировки записи предполагает наличие только одного писателя (главного потока), поэтому дополнительная взаимная блокировка не нужна.`writePending`сначала устанавливает флаг, затем берёт блокировку, гарантируя, что поток прогресса увидит намерение записи до взятия блокировки и добровольно уступит.

## Руководство по избеганию проблем в продакшене

**Ловушка 1: несовпадение числа GIN-соединений приводит к взаимоблокировке AllGather.**у каждого rank`ginCommCount`может различаться (зависит от количества локальных сетевых карт), NCCL через`bootstrapAllGather`берёт минимум по всем rank.

[FACT:src/gin/gin_host.cc:176-180]

```cpp
ginCommCountHandles[comm->rank] = backend->ginCommCount;
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
for (int r = 0; r nRanks; r++) {
  backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
}
```

Если количество сетевых карт у какого-либо ранга меньше, чем у других рангов, все ранги понижаются до минимального значения. Это гарантирует симметрию соединений, но приводит к неэффективному использованию ресурсов сетевых карт.

**Проблема 2: proxyNthreads превышает ginCommCount, что приводит к холостому вращению потоков.**Если пользователь установил`NCCL_GIN_PROXY_NTHREADS`больше`ginCommCount`, лишние потоки будут холостым образом вращаться в цикле stride.

[FACT:src/gin/gin_host.cc:181-183]

```cpp
// After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
// on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
// Extra threads simply idle in the stride loop; no correctness issue.
```

Это не проблема корректности, но приводит к неэффективному использованию ресурсов CPU. Метод диагностики — проверить,`NCCL_GIN_PROXY_NTHREADS`больше ли фактического количества сетевых карт.

**Проблема 3: состояние гонки при освобождении DevComm.** `ncclGinDevCommFree`Сначала DevComm удаляется из связного списка, затем уничтожается context.

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

После удаления из списка поток прогресса больше не видит этот DevComm, поэтому уничтожение context безопасно. Однако если во время уничтожения есть незавершённые сетевые операции, это может привести к неопределённому поведению — это то, что необходимо обеспечить при использовании GIN: перед освобождением DevComm нужно убедиться, что все операции завершены.

---

# Часть третья. Ядро симметричной памяти: от «регистрируемых буферов» к «единому адресному пространству»

## Интуитивная модель

Буферы традиционного NCCL основаны на «регистрации»: каждый ранг регистрирует свой буфер, а при обмене данными адреса передаются через handle. Симметричная память — это «единое адресное пространство»: все ранги договариваются об одном и том же наборе виртуальных адресов; адрес A ранга 0 и адрес A ранга 1 указывают на их собственную физическую память, но в коде можно обращаться к ним по одному и тому же адресу.

Это похоже на договорённость «3-й ряд, 5-е место» — у каждого дома оно указывает на одно и то же место, и при поиске вещей не нужно сначала спрашивать «где у тебя 3-й ряд, 5-е место».

Без симметричной памяти каждому ядру пришлось бы сначала разрешать адрес удалённой стороны, что увеличивает накладные расходы на инструкции и давление на регистры.

## Структуры данных и раскладка памяти

Ядро симметричной памяти основано на kernel mask — битовой карте, которая отмечает, какие ядра доступны в текущем домене связи.

[FACT:src/sym_kernels.cc:17-63]

```cpp
constexpr uint32_t kernelMask_STMC =
  1  **[Design Inference & Architectural Trade-offs]**
> Преимущество такого битового дизайна в том, что можно быстро фильтровать доступные ядра с помощью битовых операций. Например,`kmask &= ~kernelMask_STMC`одной строкой можно отключить все ядра STMC, не перебирая список.

## Пошаговый разбор: одно вычисление kernel mask

Возьмём сценарий: ранг 0 должен выполнить AllReduce, тип данных float16, размер сообщения 1MB, домен связи содержит 8 рангов, все соединены через NVLink.

**Шаг первый: получить базовую маску, соответствующую операции.**

[FACT:src/sym_kernels.cc:304-306]

```cpp
uint32_t kmask = kernelMask_coll(coll);
```

`kernelMask_coll(ncclFuncAllReduce)`возвращает`kernelMask_AR`, включающую 5 ядер AllReduce.

**Шаг второй: проверить доступность STMC и LDMC.**

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

`hasLsaMultimem`вычисляется в`ncclSymkInitOnce`, требует доступности симметричного мультикаста NVLS и группы LSA размером более 2 рангов. float16 поддерживает LDMC, поэтому если`hasLsaMultimem`истинно, ядро LDMC сохраняется.

**Шаг третий: проверить ограничения по размеру сообщения.**

[FACT:src/sym_kernels.cc:336-342]

```cpp
size_t nBytes = alignUp(nElts * ncclTypeSize(ty), NCCL_SYM_KERNEL_CELL_SIZE);
size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks) * nBytes;
if (nBusBytes >= (size_t(2) = 32 * (size_t(2) nRanks;
kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
```

Если группа LSA охватывает все ранги, GIN не нужен; иначе сохраняются только ядра GIN.

## Управление параллелизмом и взаимодействие с аппаратным обеспечением

Инициализация ядра симметричной памяти включает создание DevComm и распределение ресурсов.

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

Ключевым здесь является`ncclDevrCommCreateInternal`, который создаёт внутренний DevComm, содержащий ресурсы LSA-мультикаста, GIN inbox/outbox, сигналы и т. д.`reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL`задаёт режим соединения GIN по rail.

[FACT:src/sym_kernels.cc:257-261]

```cpp
symk->kcomm.workStarted = comm->profiler.symWorkStarted;
symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
symk->kcomm.workPhases = comm->profiler.symWorkPhases;
```

Ядро симметричной памяти использует отдельный буфер profiler, чтобы избежать чередования с workCounter обычных ядер.

## Руководство по избежанию проблем в production

**Проблема 1: требования TMA-ядра к SMEM.**TMA требует примерно 8KB SMEM scratch на каждый warp; для 16 warp это 128KB.

[FACT:src/sym_kernels.cc:135-142]

```cpp
bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  if (comm->maxSharedMemOptin minCompCap >= 100 && ncclParamSymTmaEnable();
}
```

Если ёмкости SMEM на GPU недостаточно (например, в экземпляре MIG), TMA-ядро будет отключено. Метод диагностики — проверить,`maxSharedMemOptin`меньше ли`ncclTmaShmemScratchWarpSize() * 16`。

**Проблема 2: границы GIN chunk size.**У chunk size ядра ReduceScatter GIN есть верхняя и нижняя границы.

[FACT:src/sym_kernels.cc:148-153]

```cpp
static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128  0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}
```

Если пользователь установил`NCCL_SYM_RS_GIN_CHUNK_SIZE`больше 1GB, значение будет усечено до 1GB; если меньше 128 байт, оно будет повышено до 128 байт. Итоговое значение также будет округлено вниз до степени двойки.

**Проблема 3: несоответствие типов регистрации симметричной памяти.** `ncclGetSymRegType`На основе флагов sendWin и recvWin`NCCL_WIN_COLL_SYMMETRIC`определяется тип регистрации.

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

Если типы регистрации send и recv не совпадают, ядро должно использовать разные пути кода. Это влияет на производительность, но не приводит к ошибкам.

---

# IV. Абстракция Team и версионированный DevComm: инфраструктура для эволюции

## Интуитивная модель

Абстракция Team похожа на «группировку»: мировая команда — это весь класс, команда LSA — это соседи по парте, команда Rail — это места в одном столбце. Разные режимы коммуникации требуют разных перспектив группировки.

Версионированный DevComm похож на «переводчика»: разные версии кода устройства говорят на разных «диалектах», а слой совместимости DevComm отвечает за перевод, позволяя старому и новому коду понимать друг друга.

Без абстракции Team каждое ядро должно само вычислять отображение рангов; без версионированного DevComm любое изменение ABI приведёт к перекомпиляции всего кода устройства.

## Структуры данных и размещение в памяти

Team — это простой кортеж из трёх элементов:`nRanks`、`rank`、`stride`。

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

Шаг (stride) мировой команды равен 1, так как все ранги расположены последовательно.

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

Шаг команды Rail равен`lsaSize`, так как ранги на каждом rail разделены размером команды LSA.

Ядром версионированного DevComm является структура`ncclDevCommCompat`.

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

Эта структура определяет правила совместимости для версии 2.31.0.`minVersion`и`maxVersion`определяют диапазон применимых версий, а следующие четыре указателя на функции определяют логику фильтрации свойств и преобразования структур. Если все они равны nullptr, это означает, что данная версия не имеет особых требований к совместимости.

## Пошаговое руководство: одно преобразование Team

Рассмотрим сценарий: ранг 5 в домене коммуникации из 8 рангов, размер команды LSA равен 4. Нужно вычислить ранг ранга 5 в команде Rail.

**Шаг первый: инициализация состояния DevR.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
```

`ncclDevrInitOnce`вычисляет производную информацию, такую как команда LSA, команда CFT и т.д. В случае неудачи возвращает пустую команду.

**Шаг второй: вычисление параметров команды Rail.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
ncclTeam_t ans;
ans.nRanks = comm->nRanks / comm->devrState.lsaSize;  // 8 / 4 = 2
ans.rank = comm->rank / comm->devrState.lsaSize;       // 5 / 4 = 1
ans.stride = comm->devrState.lsaSize;                  // 4
```

Ранг ранга 5 в команде Rail равен 1, команда содержит 2 ранга, шаг равен 4.

**Шаг третий: преобразование обратно в мировой ранг.**

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Если нужно преобразовать Rail rank 0 в мировой ранг:`5 + (0 - 1) * 4 = 1`. Проверка: ранг 1 и ранг 5 находятся на одном rail (интервал 4).

## Управление параллелизмом и взаимодействие с оборудованием

Сама абстракция Team не имеет состояния и не требует управления параллелизмом. Но`ncclDevrInitOnce`загружается лениво, при первом вызове вычисляется вся производная информация.

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

Комментарий гласит: «Ignoring errors since if it fails ncclDevrInitOnce will try again» — если инициализация не удалась, возвращается пустая команда, при следующем вызове будет повторная попытка.

## Руководство по избеганию проблем в production

**Проблема 1: предположение о шаге при преобразовании Team.** `ncclTeamRankToWorld`предполагает, что ранги внутри команды образуют арифметическую прогрессию.

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Если команда не является арифметической прогрессией (например, произвольная пользовательская группировка), эта функция вычислит неверно. NCCL в настоящее время поддерживает только регулярные команды.

**Проблема 2: нулевые указатели в версионированном DevComm.** `ncclDevCommCompat_v23100`все указатели на функции равны nullptr, что означает отсутствие специальной логики совместимости. Если в будущих версиях потребуется преобразование, эти функции должны быть реализованы, иначе старый и новый код не смогут взаимодействовать.

**Проблема 3: иерархические режимы команды CFT.** `ncclTeamCft`поддерживает три режима: FLAT, HIER_MULTIMEM, HIER_LSA.

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

При передаче недопустимого режима возвращается пустая команда. При использовании команды CFT необходимо убедиться в правильности режима.

---

# Размышления о дизайне

**Почему NCCL одновременно поддерживает три пути эволюции: RMA, GIN и симметричную память?**

> **[Design Inference & Architectural Trade-offs]**
> Эти три пути решают проблемы разных уровней:

- **RMA**решает проблему «фиксированного режима коммуникации» — позволяет верхнему уровню комбинировать примитивы для реализации произвольных режимов коммуникации.
- **GIN**решает проблему «высокой сетевой задержки» — позволяет GPU напрямую управлять сетевой картой, минуя host proxy.
- **Симметричная память**решает проблему «накладных расходов на разрешение адресов» — позволяет ядру напрямую обращаться к памяти удалённой стороны по унифицированному адресу.

Они не являются взаимоисключающими, а дополняют друг друга. RMA может использовать GIN в качестве нижележащего транспорта, GIN зависит от симметричной памяти для обеспечения согласованности адресов. Вместе эти три компонента образуют инфраструктуру «программируемого коммуникационного движка».

**В чём заключается философия дизайна версионированного DevComm?**

> **[Design Inference & Architectural Trade-offs]**
> Основная идея версионированного DevComm — «стабильный ABI, эволюционирующий API». Код устройства (ядро) после компиляции встраивается в бинарный файл и не может быть перекомпилирован при обновлении библиотеки NCCL. Поэтому NCCL должен гарантировать, что старый код устройства может работать с новой библиотекой.`ncclDevCommCompat`Структура является точкой входа слоя совместимости: новая библиотека выбирает подходящие правила совместимости в зависимости от версии кода устройства и при необходимости выполняет преобразование структур.

---

# Резюме главы

В этой главе, отталкиваясь от следов эволюции в исходном коде, мы проанализировали три силы, движущие NCCL от библиотеки коллективных коммуникаций к программируемому коммуникационному движку:

1. **RMA**（`src/rma/rma.cc`): Через комбинацию примитивов Put/Signal/WaitSignal верхний уровень может реализовать любой режим связи. Ключевая идея дизайна — разделить задачи на два пути, CE и Proxy, которые выполняются параллельно в зависимости от достижимости LSA.

2. **GIN**（`src/gin/gin_host.cc`): Через прямой выход GPU в сеть, минуя host proxy. Ключевая идея дизайна — управление несколькими бэкендами, таблица совместимости версий, пул потоков прогресса.

3. **Ядро симметричной памяти**（`src/sym_kernels.cc`): Через единое адресное пространство устраняется накладные расходы на разрешение адресов. Ключевая идея дизайна — битовая карта маски ядра и аппаратное ускорение TMA/GIN.

4. **Абстракция Team и версионированный DevComm**（`src/nccl_device/core.cc`、`src/devcomm/devcomm_v23100.cc`): Предоставляет инфраструктуру для эволюции. Team даёт групповое представление, версионированный DevComm обеспечивает совместимость ABI.

Влияние этих изменений на верхнеуровневые фреймворки глубоко: ProcessGroup в PyTorch может напрямую вызывать примитивы RMA для реализации пользовательских режимов связи; экспертный параллелизм в Megatron может использовать GIN для снижения задержки all-to-all; симметричная память делает код ядер более лаконичным.

# Вопросы для размышления и самопроверки в этой главе

Q1: Если убрать`scheduleRmaTasksToPlan`проверку достижимости LSA в ветке WaitSignal, и все peer пойдут по пути Proxy, какие будут последствия? В каких сценариях это вызовет катастрофу производительности?

**Справочный анализ**：

Проверка достижимости LSA в[FACT:src/rma/rma.cc:187-204], она делит peer на две группы: CE и Proxy. Если убрать эту проверку, все peer пойдут по пути Proxy,`nRmaTasksCe`всегда будет равно 0.

Последствия: путь CE полностью не используется, все WaitSignal опрашивают сеть через потоки host proxy. Для peer в пределах LSA (соединённых по NVLink на одной машине), которые могли бы асинхронно ожидать через копировальный движок GPU, теперь используется опрос потока host, задержка возрастает с микросекунд до миллисекунд.

Сценарий катастрофы производительности: при обучении MoE каждый token должен ждать сигналы от нескольких экспертов. Если все сигналы идут через Proxy, поток host становится узким местом, и GPU значительное время ждёт опроса host. На машине с 8 GPU, полностью соединённых NVLink, эта деградация особенно заметна — вся связь, которая могла бы идти через CE, теперь идёт через host.

Метод диагностики: смотреть`scheduleRmaTasksToPlan`INFO-логи, если`nRmaTasksCe`всегда равно 0, а`nRmaTasksProxy`очень велико, значит, с проверкой LSA проблема.

Q2：`ncclGinProgress`В`writePending`флаг`devCommRwMutex`и взаимодействие с блокировкой чтения-записи, если убрать`writePending`проверку, оставив только блокировку чтения-записи, какие будут проблемы?

**Справочный анализ**：

`writePending`Проверка в[FACT:src/gin/gin_host.cc:63-66], она заставляет поток прогресса активно уступать, когда главный поток собирается писать. Если убрать эту проверку, поток прогресса сразу попытается взять блокировку чтения.

Проблема в том, что:`std::shared_timed_mutex`блокировка чтения является разделяемой, несколько потоков прогресса могут держать её одновременно. Если главный поток хочет взять блокировку записи, он должен дождаться освобождения всех блокировок чтения. При высокой нагрузке потоки прогресса часто берут блокировку чтения, и главный поток может долго не получить блокировку записи, что приводит к`ncclGinDevCommSetup`или`ncclGinDevCommFree`блокировке.

Что ещё серьёзнее: если главный поток в`ginProgressWriteLock`сначала устанавливает`writePending`, а затем берёт блокировку, а поток прогресса не проверяет`writePending`, то поток прогресса может продолжать брать блокировку чтения после того, как главный поток установил флаг, что делает время ожидания главного потока непредсказуемым.

`writePending`Роль

Q3：`ncclSymkMask`В`nBusBytes >= 32 * (size_t(2) << 30)`, если при`kmask = 0`все ядра отключены (`ncclSymkAvailable`), в этот момент

**возвращает false, к какому пути откатится NCCL? Какое влияние на производительность оказывает этот путь отката?**：

`kmask = 0`Справочный анализ[FACT:src/sym_kernels.cc:342]В`ncclSymkAvailable`, в этот момент[FACT:src/sym_kernels.cc:354-361]）。

возвращает false (

Путь отката: NCCL будет использовать традиционные ядра коллективных операций (не ядра симметричной памяти). Эти ядра обращаются к памяти peer через зарегистрированные буферы, требуя предварительного разрешения адресов, что увеличивает накладные расходы на инструкции.

Влияние на производительность: для очень больших сообщений (более 64 ГБ шинных байт) накладные расходы на разрешение адресов в традиционных ядрах составляют малую долю, поскольку сама передача данных доминирует. Но в пограничных случаях (чуть больше 64 ГБ) традиционные ядра могут быть на 10-20% медленнее, чем ядра симметричной памяти.

Корневая причина этого ограничения: ядра симметричной памяти отслеживают chunk развёрнутого цикла с помощью 32-битных целых чисел, каждый chunk не менее 32 байт, поэтому максимальный адресуемый диапазон составляет 32 * 2^31 = 64 ГБ. Превышение этого диапазона вызывает целочисленное переполнение.

---

# В реальном производстве сценарии, где одна коллективная операция превышает 64 ГБ, редки (обычно это all-reduce после накопления градиентов), но не невозможны. Если такой сценарий встретится, можно рассмотреть фрагментированную связь или использование традиционных ядер.

Переход к концу главы

В этой главе мы увидели, что NCCL движется от «фиксированных коллективных операций» к «программируемому движку связи»: RMA предоставляет комбинацию примитивов, GIN обеспечивает прямой выход GPU, симметричная память предоставляет единое адресное пространство, Team и версионированный DevComm предоставляют инфраструктуру.**Позволяет вышестоящим фреймворкам реализовывать пользовательские схемы коммуникации с меньшей задержкой и большей гибкостью**. Для таких фреймворков, как PyTorch и Megatron, это означает, что они могут напрямую строить поверх NCCL сложные схемы коммуникации, такие как MoE all-to-all, конвейерный параллелизм, экспертный параллелизм, без необходимости обходить NCCL и реализовывать собственный сетевой уровень.

Следующая глава — последняя в книге. Мы ещё раз пройдём весь путь одного AllReduce — от вызова`ncclAllReduce`, через постановку задачи в очередь, выбор алгоритма, запуск ядра, продвижение прокси, передачу по сети, вплоть до возврата результата. Этот обзор свяжет знания из предыдущих 24 глав в единую карту понимания.

Итак, мы увидели три основные линии эволюции NCCL от фиксированных коллективных операций к программируемому коммуникационному движку: композиция примитивов RMA, прямая отправка в сеть с GPU, модель симметричной памяти, а также поддерживающие их абстракция team и версионированный DevComm. Эти механизмы вместе указывают на более гибкое будущее коммуникаций, более близкое к аппаратным возможностям. Однако, как бы ни развивалась архитектура, полный путь одного AllReduce всегда остаётся краеугольным камнем понимания NCCL. В следующей главе мы не будем вводить новый код, а заново пройдём сквозной поток от главы 3 до главы 10 — от вызова ncclAllReduce до установления коммуникационного домена, поиска топологии, выбора алгоритма, постановки задачи в очередь, запуска ядра, выполнения примитивов на устройстве, записи результата. Вы заново соберёте механизмы, разбросанные по главам, в целостную ментальную модель и получите индекс «при какой проблеме какую главу смотреть».
