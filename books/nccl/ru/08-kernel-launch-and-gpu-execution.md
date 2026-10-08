# Глава 8: Запуск kernel и выполнение на устройстве: от вызова на стороне host до старта блоков потоков GPU

В предыдущей главе мы разобрали, как задачи разбиваются на несколько channel, как генерируются параметры запуска kernel и как работает механизм пакетной отправки и упорядочивания зависимостей в семантике group. Теперь план запуска готов, но это всё ещё только структура данных на стороне host. Ключевой вопрос этой главы:`ncclKernelPlan`Как это превращается в реально работающий grid на GPU? Мы пройдём по цепочке вызовов`ncclLaunchKernel`, посмотрим, как параметры помещаются в kernel args, как выбирается вариант kernel,`cuLaunchKernelEx`как вызывается`ncclKernelMain`, а также как на стороне устройства

# считывает описание работы из разделяемой памяти и распределяет его по конкретным реализациям.

От плана к grid: полная картина пути запуска`ncclKernelPlan`Прежде чем углубляться в детали, построим общую ментальную модель. Представим`ncclLaunchKernel`как «чертёж строительства»: он фиксирует, сколько channel (сколько block) нужно запустить, сколько потоков в каждом block, какие work выполнять и какую функцию kernel использовать. А`CUlaunchConfig`— это действие «входа строительной бригады»: он переводит информацию с чертежа в понятный драйверу CUDA`cuLaunchKernelEx`, а затем вызывает

, чтобы действительно отправить grid на GPU.

Без этого слоя вся диспетчеризация на стороне host (разбиение по channel, организация batch, упорядочивание proxy op из предыдущей главы) осталась бы лишь теорией, на GPU не запустился бы ни один kernel, и коммуникация никогда бы не произошла. Это последнее звено сквозного магистрального пути и граница между host и device.

1. **Весь путь запуска можно обобщить тремя этапами:**（`finishPlan` + `uploadWork`Подготовка параметров

2. **): организация структур work, дескрипторов batch и kernel args в один непрерывный блок памяти, решение о том, размещать ли их в параметрах kernel, в FIFO или в персистентном буфере.**（`ncclLaunchKernel`Запуск kernel`cuLaunchKernelEx`。

3. **): вычисление размерностей grid/block, сборка атрибутов запуска (CGA cluster, mem sync domain, launch completion event), вызов**（`ncclKernelMain`Точка входа на стороне устройства`blockIdx.x`): каждый block на основе`ncclDevFuncTable`определяет свой channelId, загружает work batch из args или FIFO в разделяемую память, а затем через

распределяет его по конкретной реализации алгоритма/протокола.

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

Копировать`finishPlan`、`uploadWork`、`ncclLaunchKernel`Эта диаграмма привязывает три ключевые функции этой главы:

# . Далее мы разберём их по очереди.

## Подготовка параметров: как структура work находит своё место

`finishPlan`Интуитивная модель

Роль

## похожа на «упаковщика» в сортировочном центре доставки. Он имеет дело с кучей разрозненных структур work (по одной на каждую коллективную или p2p-операцию) и должен решить: поместить эти work в «рюкзак» параметров kernel, на «конвейер» FIFO или на «склад» персистентного буфера?

Если это решение принято неверно — например, work слишком велик, чтобы поместиться в параметры kernel, но его всё равно туда запихивают — запуск kernel сразу завершится ошибкой. Если work размещён не в том месте, устройство прочитает мусорные данные, и результат коммуникации будет полностью неверным.`ncclDevKernelArgs`Структуры данных и разметка памяти

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

структуру`channelMask`— это «конверт» между host и device:`__popcll`Копировать`blockIdx.x`В этой структуре всего 5 полей, но каждое несёт ключевую информацию.`workStorageType`— это 64-битная маска, каждый бит которой соответствует одному channel; на стороне устройства через`Args`вычисляется`Fifo`соответствующий channelId.`Persistent`определяет, откуда на стороне устройства читается work:

`ncclDevWorkBatch`— это дескриптор batch, который сообщает устройству, «где находится работа этого channel и сколько её»:

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

`offsetBitset`— это 64-битная маска, каждый бит которой соответствует одной структуре work. Устройство с помощью`__popc`и`fns`(find n-th set) инструкций определяет смещение каждого work.`nextJump`и`nextExtends`используются для связывания нескольких batch — когда work слишком много и они не помещаются в один batch, создаётся «расширенный batch».

## Step-by-Step Walkthrough

Теперь рассмотрим конкретный сценарий: один AllReduce разбивается на 4 channel, в каждом channel по 2 структуры work, всего 8 work.

**Первый шаг:`finishPlan`определяет тип хранения.**

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

Ключевое решение здесь такое: если`sizeof(ncclDevKernelArgs) + batchBytes + workBytes`помещается в`comm->workArgsBytes`(обычно 4KB), то work кладётся напрямую в параметры kernel. Иначе work помещается в FIFO или persistent-буфер, а в параметрах kernel остаётся только дескриптор batch.

> **[Design Inference & Architectural Trade-offs]**
> Почему предпочтительно размещать в параметрах kernel? Потому что параметры kernel в драйвере CUDA передаются через constant memory, и при чтении на стороне устройства используется инструкция`ld.param`, что намного быстрее, чем чтение FIFO из глобальной памяти. Для небольших сообщений (малый общий объём work) это заметно снижает задержку.

**Второй шаг: batch по channel поочерёдно помещаются в kernel args.**

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

копирование

1. **`fifoCursor`Здесь есть несколько ключевых моментов:**Семантика`Args`: для типа`kernelArgs`это смещение относительно начального адреса`Fifo`; для типа`Persistent`это смещение относительно базового адреса FIFO; для типа

2. **`offsetBase`оно начинается с 0.**：`finishPlan`Коррекция`offsetBase`:`uploadWork`в batch`Args`задаётся относительно начала work в plan (с 0).`sizeof(ncclDevKernelArgs) + batchBytes`нужно преобразовать в смещение относительно фактического места хранения. Для типа`Fifo`добавляется`comm->workFifoProduced`。

3. **; для типа**добавляется`alignas(16)`16-байтовое выравнивание при копировании`COMPILER_ASSUME_ALIGNED`: структуры work выровнены по 16 байт (

4. **), поэтому копирование выполняется блоками по 16 байт.**сообщает компилятору, что этот адрес выровнен по 16 байт, чтобы компилятор генерировал более эффективные векторизованные инструкции.`Fifo`Ожидание FIFO`waitWorkFifoAvailable`: для типа`comm->abortFlag`,

## будет в цикле ожидать, пока в FIFO появится достаточно места. Это ожидание проверяет

> **[Design Inference & Architectural Trade-offs]**
> **Проектные соображения и подводные камни в production**〔Проектные предположения и архитектурные компромиссы〕

- `Args`Почему существуют три типа хранения?
- `Fifo`Это компромисс между объёмом и задержкой:
- `Persistent`: самый быстрый (constant memory), но ограниченный по объёму (4KB). Подходит для небольших сообщений и малого числа work.`cudaMemcpy`: большой объём (кольцевой буфер), но чтение на стороне устройства идёт через глобальную память. Подходит для сообщений среднего размера.

**: используется в сценариях захвата CUDA Graph. Поскольку при захвате graph нельзя выполнять**, нужно заранее выделить persistent-буфер, скопировать туда work, а затем заставить kernel читать оттуда.`waitWorkFifoAvailable`Подводный камень 1: переполнение FIFO приводит к взаимоблокировке.`abortFlag`Если[FACT:src/enqueue/enqueue.cc:1333-1349]не проверяет

```c
if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
  return ncclInternalError;
}
```

**явно проверяет abort flag:`offsetBitset`копирование** `offsetBitset`Подводный камень 2:`1ull << (offset / workSize)`переполнение.`NCCL_MAX_DEV_WORK_BATCH_BYTES`является 64-битным и поддерживает максимум 64 work в одном batch. Если их больше 64,`ncclDevWorkColl`произойдёт переполнение. В исходном коде через

**ограничивается размер batch (1024 байта), а минимальная структура work —**(около 80 байт), поэтому максимум 12 work, переполнения не будет.`uploadWork`Подводный камень 3: утечка памяти в persistent-режиме.`Persistent`В ветке`fifoBufHost`для`ncclOsAlignedAlloc`,`uploadWork_cleanup_fn`выделяется через`cudaMemcpyAsync`и должен освобождаться в`fail`. Если`cleanup`завершается неудачно, метка`fifoBufHost`проверяет, равен ли[FACT:src/enqueue/enqueue.cc:1483-1485]null, и если null, сразу освобождает

# . Эту цепочку восстановления после ошибок можно увидеть в

## Запуск kernel: от CUlaunchConfig до cuLaunchKernelEx

`ncclLaunchKernel`的角色类似于"火箭发射控制台"。Он принимает plan с уже загруженным топливом (данными work), вычисляет параметры полёта ракеты (размерности grid/block), настраивает различные опции запуска (cluster, mem sync domain, completion event), а затем нажимает кнопку запуска (`cuLaunchKernelEx`）。

Если на этом этапе происходит ошибка — например, неверно вычислена размерность grid — на GPU будет запущено неправильное количество блоков, что приведёт к тому, что работа части каналов никогда не будет выполнена, и通信 зависнет.

## Структуры данных и разметка памяти

`CUlaunchConfig`— это структура конфигурации запуска CUDA Driver API, NCCL создаёт её на стеке:

[FACT:src/enqueue/enqueue.cc:1916-1917]

```c
CUlaunchConfig launchConfig = {0};
CUlaunchAttribute launchAttrs[6] = {};
int attrs = 0;
```

`launchAttrs`— это массив максимум из 6 элементов, каждый элемент — это`CUlaunchAttribute`. NCCL в зависимости от возможностей аппаратуры и версии драйвера условно добавляет различные атрибуты:

- `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`: размерность CGA cluster (sm90+)
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`: стратегия планирования cluster
- `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`: домен синхронизации памяти (CUDA 12.0+)
- `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`: событие завершения запуска (CUDA 12.3+)
- `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`: программная сериализация потоков (sym kernel)
- `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING`: централизованное планирование с использованием NVLink (CUDA 13.0+)

## Step-by-Step Walkthrough

**Шаг первый: вычисление размерностей grid и block.**

[FACT:src/enqueue/enqueue.cc:1889-1893]

```c
int nChannels = countOneBits(plan->channelMask);
void* sym = plan->kernelFn;
dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
```

`nChannels`— это`channelMask`— количество установленных битов, то есть сколько блоков должен запустить этот plan. Каждый блок отвечает за один channel.`threadPerBlock`вычисляется в`scheduleCollTasksToPlan`через`plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE)`, берётся максимальное значение`nWarps * 32`。

`smem`среди всех task — это размер динамической разделяемой памяти. Для обычного kernel это`ncclShmemDynamicSize(comm->cudaArch)`, это константа времени компиляции, зависящая от архитектуры (для sm70+ это`ncclShmemScratchWarpSize * (NCCL_MAX_NTHREADS / WARP_SIZE)`). Для sym kernel это`plan->kernelDynSmem`, поскольку требования к разделяемой памяти у sym kernel могут отличаться.

**Шаг второй: сборка параметров kernel.**

[FACT:src/enqueue/enqueue.cc:1902-1903]

```c
void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
                 CU_LAUNCH_PARAM_END};
```

Это один из способов передачи параметров в CUDA Driver API:`CU_LAUNCH_PARAM_BUFFER_POINTER`сообщает драйверу, что "параметры передаются не по отдельности, а одним непрерывным блоком памяти",`CU_LAUNCH_PARAM_BUFFER_SIZE`сообщает драйверу размер этого блока. Преимущество такого подхода в том, что NCCL может передать`ncclDevKernelArgs`и следующий за ним массив batch за один раз, без необходимости упаковывать параметры по отдельности.

**Шаг третий: добавление launch attributes.**

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

CGA (Cooperative Group Array) — это аппаратная возможность, появившаяся в sm90, позволяющая объединить несколько блоков в один cluster; блоки внутри cluster гарантированно одновременно планируются на одну группу SM и могут обращаться к разделяемой памяти друг друга. NCCL использует эту возможность для реализации алгоритмов, требующих межблочной синхронизации, таких как NVLS.

Обратите внимание на защиту`if (grid.x % clusterSize) clusterSize = 1;`: размерность cluster должна нацело делить размерность grid, иначе драйвер выдаст ошибку. Если`grid.x`не делится нацело на`clusterSize`, происходит откат к использованию без cluster.

**Шаг четвёртый: добавление launch completion event.**

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

`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`— это возможность, появившаяся в CUDA 12.3: драйвер записывает событие в момент, когда kernel действительно начинает выполняться (а не когда host-сторона возвращается из вызова). Это критически важно для реализации "неявного порядка" (implicit order) — NCCL должен гарантировать последовательное выполнение нескольких kernel, но при этом не хочет блокировать ожидание на host-стороне.

`getImplicitOrder`логика такова: если пользователь установил`launchOrderImplicit`, и версия драйвера достаточно новая, используется`ncclImplicitOrderLaunch`(упорядочивание через launch event); иначе используется`ncclImplicitOrderSerial`(упорядочивание через completion event, то есть последовательное выполнение).

**Шаг пятый: вызов`cuLaunchKernelEx`。**

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

`cuLaunchKernelEx`— это новый API, появившийся в CUDA 12.0, поддерживающий launch attributes. Для старых драйверов (< 11.8) NCCL откатывается к`cuLaunchKernel`：

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

## Управление конкурентностью и взаимодействие с аппаратурой

**Механизм relay для Launch completion event.**Когда используется`ncclImplicitOrderLaunch`и пользователь предоставил`launchCompletionEvent`, NCCL не может напрямую передать пользовательское event драйверу, поскольку драйвер поддерживает только один launch completion event. NCCL поступает следующим образом:

1. Передаёт`comm->sharedRes->launchEvent`драйверу.

2. Ожидает на`relayStream`события`launchEvent`。

3. Записывает пользовательское event на`relayStream`.

Таким образом, пользовательское event сработает после того, как kernel действительно начнёт выполняться, а не когда host-сторона вернётся из вызова.

**Mem Sync Domain。** [FACT:src/enqueue/enqueue.cc:1938-1942]На sm90+ NCCL устанавливает`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`в`cudaLaunchMemSyncDomainRemote`. Это механизм доменов синхронизации памяти, введённый в архитектуре Hopper, предназначенный для изоляции барьеров памяти разных kernel и снижения избыточных накладных расходов на синхронизацию.

## Руководство по подводным камням в продакшене

**Подводный камень 1: размерность cluster не делится нацело, что приводит к сбою запуска.**Если`grid.x`не делится нацело на`clusterSize`, драйвер вернёт`CUDA_ERROR_INVALID_VALUE`. В исходном коде предусмотрена защита через`if (grid.x % clusterSize) clusterSize = 1;`, но это также означает, что возможность cluster молча отключается. Если пользователь ожидает повышения производительности от cluster, необходимо проверить соотношение`cgaClusterSize`и`nChannels`.

**Подводный камень 2: несоответствие версии драйвера приводит к недоступности kernel.** `ncclInitKernelsForDevice`проверяет требования к драйверу для каждого ядра при инициализации:

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

Ядро этого кода — вычисление`fnsOfBitset`: для n-го установленного бита в`offsetBitset`каков его битовый индекс. В PTX есть инструкция`fns`для этого, но она разворачивается в множество инструкций SASS. Подход NCCL — использовать разделяемую память: каждая lane проверяет, установлен ли её бит, и если да, вычисляет, сколько установленных битов идёт перед ней, а затем записывает номер своей lane в`fnsOfBitset[nWorksBelow]`。

Далее идёт собственно копирование:

[FACT:src/device/common.h:209-241]

```c
if (tid = %d && __CUDA_ARCH__ >= %d\n" % (cudart ,arch))
  out("/*%4d*/ %s,\n" % (index, sym))
  if (cudart, arch) != (0, 0):
    out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
  index += 1
out("nullptr};\n")
```

## Проектные размышления и подводные камни в продакшене

**Почему используется`__grid_constant__`？** [FACT:src/device/common.h:19-24]

```c
#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif
```

`__grid_constant__`сообщает компилятору, что этот параметр доступен только для чтения и может быть размещён в константной памяти. Таким образом, при чтении на стороне устройства используется`ld.param`инструкция, что быстрее, чем чтение из глобальной памяти. В комментарии упоминается, что это ломает cuda-gdb, поэтому включается только на sm70+.

**Подводный камень 1:`workStorage`переполнение.** `workStorage`Размер`ncclMaxDevWorkBatchBytes()`составляет`nWorks * workSize`, для sm90+ — 16KB. Если`NCCL_MAX_DEV_WORK_BATCH_BYTES`Подводный камень 2:

**на стороне host ограничивается размер batch, но на стороне устройства дополнительной проверки нет. Если ограничение на стороне host будет обойдено (например, путём изменения переменной окружения), это приведёт к выходу за границы разделяемой памяти.`__syncthreads()`отсутствие**приводит к гонке данных.`loadWorkBatchToShmem`После`__syncthreads()`должен быть`workStorage`чтобы все потоки увидели полный[FACT:src/device/common.h:479]. В исходном коде в`__syncthreads(); // publish ncclShmem`есть`workStorage`. Если эту синхронизацию убрать, некоторые потоки могут начать чтение до того, как

**Подводный камень 3: момент проверки abort.** `while (ncclShmem.aborted == 0)`проверяет abort только в начале каждого batch. Если какой-то batch выполняется долго, сигнал abort может вступить в силу нескоро. Это компромисс в дизайне: более частые проверки увеличивают накладные расходы, но обеспечивают более быстрый отклик.

# Выбор варианта kernel: как generate.py генерирует список kernel

## Интуитивная модель

`generate.py`Роль

Если для каждой комбинации генерировать отдельный kernel, время компиляции и размер бинарного файла взорвутся. Если генерировать только один универсальный kernel, во время выполнения всё замедлится из-за вызовов через указатели на функции и ветвлений.`generate.py`Решение

## Структуры данных и layout памяти

`generate.py`генерирует три ключевых файла:

1. **`device_table.cu`**: на стороне устройства`ncclDevFuncTable`, отображающий funcId на конкретную функцию устройства.

2. **`host_table.cc`**: на стороне host`ncclDevKernelList`、`ncclDevKernelForFunc`、`ncclDevFuncRowToId`и другие таблицы.

3. **Различные`<coll>_<op>_<ty>.cu`**: конкретные реализации kernel.

## Step-by-Step Walkthrough

**Шаг первый: перечисление всех строк функций.**

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

Этот порядок перечисления должен совпадать с`ncclDevFuncId()`формулой вычисления:

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

`ncclDevFuncId`вычисляет «номер строки», затем через`ncclDevFuncRowToId`отображает его на «ID основной функции». Причина этого отображения: многие строки могут отображаться на одну и ту же основную функцию (например, все`AllReduce Sum i32`строки отображаются на`AllReduce Sum u32`основную функцию).

**Шаг второй: вычисление основной функции и kernel-функции.**

[FACT:src/device/generate.py:211-225]

```python
func_rows = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}
kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))
```

`equivalent_primary`отображает знаковые целые в беззнаковые (поскольку сложение/умножение для них одинаково):

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

`best_kernel`отображает несколько основных функций на один kernel (например, все`AllGather`алгоритмы отображаются на`AllGather RING LL`）：

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

**Шаг третий: генерация определения kernel.**

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

`DEFINE_ncclDevKernel`После раскрытия макроса получается:

[FACT:src/device/common.h:507-509]

```c
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain, algo, proto>>(&args4K.args); \
  }
```

Таким образом, каждый kernel — это`__global__`функция, вызывающая`ncclKernelMain`, с шаблонными параметрами`specializedFnId`и`RunWorkBatch<coll, ty, redop<ty>, algo, proto>`。

## Проектные размышления и подводные камни в продакшене

> **[Design Inference & Architectural Trade-offs]**
> **Почему используются «представительные kernel», а не отдельный kernel для каждой комбинации?**Компромисс между временем компиляции и размером бинарного файла. Полное комбинаторное пространство — 7 × 5 × 12 × 7 × 3 ≈ 8820 kernel, компиляция каждого занимает несколько секунд, в сумме — несколько часов. К тому же размер бинарного файла достигнет нескольких сотен MB. Благодаря отображению на представительные kernel фактическое количество генерируемых kernel сокращается до нескольких десятков.

**Подводный камень 1:`NCCL_EXACT_KERNEL_NAMES`приводит к взрыву компиляции.**Если установлена эта переменная окружения,`best_kernel`возвращает исходные функции, и для каждой комбинации генерируется отдельный kernel. Это полезно при разработке (можно точно контролировать, какой kernel компилируется), но в продакшене приводит к чрезмерно долгой компиляции.

**Подводный камень 2:`required_cuda`проверка версии.**Некоторые kernel требуют определённой версии CUDA или архитектуры:

[FACT:src/device/generate.py:130-154]

На этом этапе kernel уже запущен на GPU, и сторона устройства также получила описание работы. Но реально производительность определяет то, как данные перемещаются внутри устройства. В следующей главе мы углубимся в три протокольных примитива в src/device: LL, LL128 и Simple, и посмотрим, почему для одной и той же логики AllReduce нужны три набора примитивов перемещения данных, а также в чём их различия в способах синхронизации, layout буферов и семантике flag.
