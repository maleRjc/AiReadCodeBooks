# Глава 5: Тюнинг и выбор протокола: как модуль tuning определяет оптимальные пути и каналы

# Глава 5: Выбор алгоритма и протокола: как модуль tuning определяет путь коммуникации

В предыдущей главе мы разобрали возможности NCCL по учёту топологии: от перечисления устройств и построения графа топологии в src/graph/topo.cc, до поиска оптимальных путей в src/graph/search.cc, и далее до конкретизации результатов поиска в топологии алгоритмов Ring и Tree в rings.cc и trees.cc. Но граф топологии отвечает лишь на вопрос «по какому пути могут идти данные», а не на вопрос «по какому пути должно идти данное взаимодействие». На одной и той же машине оптимальное решение для AllReduce размером 4KB и AllReduce размером 400MB может быть совершенно различным: в первом случае важна задержка, во втором — пропускная способность; в первом случае может быть выбран Tree/LL, во втором — Ring/Simple или NVLS. Модуль tuning — это тот, кто «принимает решение». Его входные данные — размер сообщения, количество рангов, граф топологии (результат предыдущей главы) и переменные окружения пользователя; выходные — структура ncclTuningResult_t, содержащая информацию о том, какой алгоритм (algo) использовать, какой протокол (proto), сколько каналов задействовать и сколько warp'ов. В этой главе мы разберём каталог src/tuning в порядке «общее управление → модель стоимости → оценка для каждого алгоритма → финальное решение». Ключевой вопрос только один: как NCCL среди десятков комбинаций (алгоритм, протокол) с помощью чисто CPU-математической модели за микросекунды выбирает самую быструю?

# I. tuning.cc: общее управление и основная логика принятия решений

## Интуитивная модель

Представьте модуль tuning как**компанию по переездам**. Приходит клиент (одна коллективная операция) и говорит: «Мне нужно перевезти 100MB груза из 8 складов в 8 складов». Диспетчер (`ncclTuningCompute`) не станет реально перевозить груз, чтобы попробовать, а достанет**прайс-лист**(модель стоимости), оценит для каждого варианта (Ring/LL, Tree/Simple, NVLS/Simple……) «ожидаемое время выполнения» и выберет самое короткое предложение для клиента.

Без этого диспетчера NCCL мог бы только жёстко прописать «AllReduce всегда использует Ring», и тогда в сценариях с малыми сообщениями он бы проигрывал Tree, а в крупномасштабных сценариях с NVLink — NVLS.**Цена этого — падение производительности в определённых сценариях вдвое или даже хуже.**

## Структуры данных и компоновка памяти

Носителем решения является`ncclTuningResult_t`, множество кандидатов — это`ncclTuningResultList_t`(односвязный список). Узлы списка определены в`tuning_int.h`, но логика push находится в`tuning.cc`:

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
> Обратите внимание, здесь используется**вставка в голову**: каждый раз, когда вычисляется допустимый кандидат, он вставляется в голову списка. Это означает, что порядок списка и порядок id**обратны**. Почему используется список, а не массив? Потому что количество кандидатов на этапе компиляции определяется`NCCL_TUNING_COUNT`, но фактически допустимые кандидаты динамичны (зависят от`tuningMask`, возможностей платформы, переменных окружения пользователя), список позволяет «прикреплять только допустимые», избегая повторных проверок при обходе`valid`. Цена — при каждом принятии решения необходимо`ncclCalloc`один раз, но tuning происходит на пути постановки в очередь и нечасто, так что эти накладные расходы на выделение приемлемы.

`ncclTuningResult_t`два наиболее важных поля —`timeUs`(ожидаемое время, микросекунды) и`selectionTimeUs`(время, используемое для выбора, может быть переопределено плагином tuner). Логика выбора смотрит только на последнее:

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

Здесь есть деталь:`bestTuning->timeUs`сначала устанавливается в`FLT_MAX`, затем выполняется обход. Если список пуст (все кандидаты недействительны),`bestTuning`сохранит`NCCL_TUNING_RESULT_INIT`начальное значение, algo/proto оба равны`UNDEF`. Этот «пустой результат» особым образом обрабатывается вызывающей стороной — см. ветку ошибки ниже.

## Step-by-Step Walkthrough: поток принятия решений одного AllReduce

Предположим, приложение вызывает`ncclAllReduce`, сообщение 1MB, 8 рангов, одна машина NVLink. Мы пройдём вместе с`ncclTuningCompute`весь путь.

**Шаг 0: короткое замыкание для одного ранга.**Если`nRanks <= 1`, коммуникация вообще не нужна, сразу возвращается Ring/Simple, число channel устанавливается в 0:

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

копировать`NCCL_TUNING_IGNORE`Здесь

**— это сигнальное значение, означающее «эта комбинация не вычислялась/неприменима». Плагин может изменить только интересующие его ячейки, остальные остаются IGNORE, и NCCL их пропустит.**Шаг 4: выбор оптимального.`ncclTuningSelectBestTuning`вызывает`selectionTimeUs`, обходит список и берёт

**с минимальным значением.**Шаг 5: вычисление числа channel.

[FACT:src/tuning/tuning.cc:233-235]

```c
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
```

`ncclTuningGetChannels`копировать`tuning_int.h`В`minChannels`логика основана на размере сообщения и типе алгоритма, интерполируя между`maxChannels`и

**. Число channel напрямую влияет на пропускную способность: чем больше channel, тем выше параллелизм, но и накладные расходы на запуск каждого channel тоже больше.**Шаг 6: смещение CTA Policy (приоритет NVLS).`NCCL_CTA_POLICY_EFFICIENCY`Если пользователь установил

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

**Почему коды ошибок различаются?**Если пользователь установил`NCCL_ALGO=ring`, но текущая платформа не поддерживает ring (например, некоторые особые топологии), то это**ошибка конфигурации пользователя**（`ncclInvalidUsage`); если пользователь не устанавливал никаких переменных окружения, но алгоритм выбрать не удалось, то это**внутренний баг NCCL**（`ncclInternalError`). Это различие критически важно для отладки.

## Блок-схема основного потока принятия решений

```mermaid
flowchart TD
    start["ncclTuningCompute(input)"] --> check_rank{"comm->nRanks |да| single["bestTuning = Ring/SimplenChannels = 0"]
    check_rank -->|нет| enum["ncclTuningComputeAllTuningsперебор NCCL_TUNING_COUNT"]
    enum --> mask{"tuningMask & (1|нет| skip["tuning.valid = 0continue"]
    mask -->|да| expand["ncclTuningExpandId(i)"]
    expand --> sim["ncclTuningComputeTuning-> ncclTuningCostModelSimModel"]
    sim --> valid{"result.valid?"}
    valid -->|да| push["ncclTuningResultListPushFront"]
    valid -->|нет| skip
    push --> tuner{"comm->tuner != NULL?"}
    tuner -->|да| plugin["tuner->getCollInfoперезапись generalTable"]
    tuner -->|нет| select
    plugin --> select["ncclTuningSelectBestTuningвыбор с минимальным selectionTimeUs"]
    select --> getch["ncclTuningGetChannels"]
    getch --> cta{"CTA_POLICY_EFFICIENCYи NVLS в mask?"}
    cta -->|да| nvls["ncclNvlsRegResourcesQueryвозможна замена на NVLS"]
    cta -->|нет| symk
    nvls --> symk{"symKernelId требует отката?"}
    symk -->|да| fallback["ncclTuningCompute(generalInput)откат к обычному kernel"]
    symk -->|нет| done
    fallback --> done["*result = bestTuning"]
    single --> done
    done --> undef{"algo/proto всё ещё UNDEF?"}
    undef -->|да| warn["WARN + возвратInvalidUsage или InternalError"]
    undef -->|нет| ret_ok["возврат ncclSuccess"]
```

---

# II. cost_model.cc: реестр моделей и матрица переключателей

## Интуитивная модель

`cost_model.cc`— это**главная книга**tuning. Она поддерживает таблицу`modelMap`, каждая строка которой соответствует комбинации (algo, proto) и хранит «кто функция инициализации этой комбинации, кто функция симуляции, для каких функций она включена». Одновременно она отвечает за разбор пользовательской переменной окружения`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SYM_KERNEL`, переводя намерения пользователя в матрицу переключателей`enabled[i][f]`.

Без этой таблицы при добавлении каждого нового алгоритма пришлось бы менять основной поток tuning, и код превратился бы в кашу.**Табличный подход**превращает «добавление алгоритма» в «добавление одной строки».

## Структуры данных: modelMap и матрица переключателей

`modelMap`— это статический массив, каждый элемент которого —`ncclTuningModelEntry_t`：

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

Каждый entry имеет четыре поля:`init`(инициализация, вычисляет latency/bandwidth и сохраняет в comm),`model`(симуляция, вычисляет итоговое timeUs по размеру сообщения),`finalize`(очистка),`enabled[5]`(включены ли пять функций Broadcast/Reduce/AllGather/ReduceScatter/AllReduce).

Примечание`enabled`Порядок массива закомментирован в L234:`Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce`. Этот порядок должен совпадать с`ncclFunc_t`перечислением, иначе произойдёт путаница.

> **[Design Inference & Architectural Trade-offs]**
> **Почему init и sim разделены?**Потому что то, что вычисляется в init (latency, bandwidth),**зависит только от статических свойств comm**(топология, число rank'ов, compCap) и не связано с конкретным размером сообщения. В одной коммуникации может последовательно вызываться tuning несколько раз (например, в группе несколько op), init выполняется один раз, sim — каждый раз. Это типичная оптимизация «предвычисление + быстрый запрос».

## Пошагово: разбор переменных окружения и построение матрицы переключателей

**Шаг 1: по умолчанию всё включено, LL128 — особый случай.** `ncclTuningCostModelInit`Вначале все proto устанавливаются в 1 (включено), но LL128 устанавливается в 2:

[FACT:src/tuning/cost_model.cc:313-323]

```c
  for (int f = 0; f minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                          comm->graphs[algo].typeIntra, comm->nRanks, f, algo, comm->minDriverVersion)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
```

**Шаг 2: разбор пользовательских переменных окружения.**Если пользователь задал`NCCL_ALGO`или`NCCL_SYM_KERNEL`, сначала обнуляются algo и symKernel (поскольку пользователь указал белый список):

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

Обратите внимание, что proto не обнуляется — потому что значения по умолчанию для proto равны 1/2, и когда пользователь задаёт`NCCL_PROTO=LL`,`parseList`устанавливает LL в 1, а остальные в 0 (из-за`unset`логики). Эта асимметрия намеренна: algo по умолчанию полностью включён, но после указания пользователя сужается; сужение proto обрабатывается внутри`parseList`.

**Шаг 3: синтаксис parseList.**Эта функция поддерживает довольно сложный синтаксис, в комментариях приведены примеры:

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

`^`Префикс означает «отрицание»:

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

Таким образом,`NCCL_PROTO="^LL128;allreduce:LL128"`означает: глобально отключить LL128, но для AllReduce сделать исключение и включить LL128.

**Шаг 4: объединение матрицы enabled.**В конце выполняется обход всех model и логическое И между`model->enabled[f]`и пользовательскими переключателями:

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

Логика такова:**Только когда пользователь задал forced-конфигурацию для некоторой функции, пользовательская конфигурация перекрывает значение по умолчанию модели**. Если пользователь не задал,`forced[f] == 0`, напрямую`continue`, сохраняется собственное`enabled`модели. Это приоритет «явное указание пользователя > значение по умолчанию модели».

## Единая точка входа для моделирования

Все модели в конечном итоге вызываются через`ncclTuningCostModelSimModel`:

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

Три уровня фильтрации:**id вне диапазона → модель отключена → модель возвращает неположительное время**, если любой уровень не проходит, идёт`not_valid`, устанавливая`timeUs`в`NCCL_TUNING_IGNORE`(отрицательный sentinel),`valid = 0`. Вызывающая сторона, увидев`valid == 0`, не добавит это в список кандидатов.

## Размышления о дизайне

`modelMap`В комментариях к

[FACT:src/tuning/cost_model.cc:229]

```c
// IMPORTANT: this table need must be consistent with the algRegistry in src/config/algorithm_registry.cc
```

> **[Design Inference & Architectural Trade-offs]**
> 〔Проектные выводы и архитектурные компромиссы〕`modelMap`Это означает, что**порядок индексов**должен строго совпадать с порядком регистрации алгоритмов в`algorithm_registry.cc`. Если кто-то вставит новый алгоритм в registry, но забудет изменить`modelMap`, все id сместятся, и tuning выберет совершенно неправильный алгоритм.**Это классическая ловушка таблично-управляемого дизайна: неявный контракт.**Более надёжный подход — использовать в качестве ключа имя перечисления, а не индекс, но это пожертвует небольшой частью оптимизации на этапе компиляции.

---

# Три, ring.cc: оценка стоимости алгоритма Ring

## Интуитивная модель

Алгоритм Ring выстраивает N rank'ов в кольцо, данные передаются по кольцу круг за кругом. Его модель стоимости должна ответить на два вопроса:**Сколько данных передаётся на каждом шаге (пропускная способность)**、**Сколько всего шагов (задержка)**。

Интуиция Ring — это «**конвейер**»: представьте N человек, стоящих в кругу и передающих ведро с водой; каждый, получив ведро, выливает немного воды и передаёт следующему. Ведро делает круг, и вода у всех перемешивается. Чем быстрее вращается ведро (выше пропускная способность), чем меньше круг (меньше шагов), тем быстрее всё в целом.

## Структуры данных: таблицы latency/bandwidth

Модель Ring не вводит новых структур, она записывает результаты оценки в`comm->tuningContext.generalLatencies[c][algo][proto]`и`generalBandwidths[c][algo][proto]`. Это трёхмерные массивы: функция × алгоритм × протокол.

При инициализации всё сначала устанавливается в -1.0 (sentinel, означающий «не вычислено»):

[FACT:src/tuning/ring.cc:31-33]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
```

Этот sentinel -1.0 проверяется на этапе sim:

[FACT:src/tuning/ring.cc:94-97]

```c
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ncclSuccess;
  }
```

**Почему используется -1.0, а не 0?**Потому что 0 — это допустимое значение пропускной способности (хотя физически невозможно), а -1.0 явно означает «не инициализировано». Сравнение с плавающей точкой через`==`здесь безопасно, потому что -1.0 точно представимо.

## Пошагово: оценка пропускной способности Ring

**Шаг 1: определить, использовать intra или inter пропускную способность.**Одиночная машина (nNodes==1) использует intra, много машин — inter:

[FACT:src/tuning/ring.cc:34-37]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float bw = (comm->nNodes == 1 || (comm->nNodes minCompCap graphs[algo].bwIntra :
                                                                                      comm->graphs[algo].bwInter;
    float busBw = bw * comm->graphs[algo].nChannels;
```

`nSteps`— это число шагов, необходимых алгоритму; для Ring AllReduce равно`2*(nRanks-1)`, остальные —`nRanks-1`。`busBw`— это «шинная пропускная способность» = пропускная способность одного канала × число channel'ов.

**Шаг 2: скидка в зависимости от протокола.**Протокол LL использует только половину пропускной способности (из-за накладных расходов на флаги LL), LL128 использует 92% (120/128):

[FACT:src/tuning/ring.cc:38-42]

```c
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);
```

`0.92 = 120/128`Это потому, что в LL128 каждые 128 байт содержат 8 байт флагов, а полезная нагрузка составляет всего 120 байт. Это число напрямую следует из дизайна протокола.

**Шаг 3: вычисление эффективной пропускной способности.**Обратите внимание, здесь умножено на`nRanks / nSteps`：

[FACT:src/tuning/ring.cc:44-46]

```c
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;
```

**Почему умножено на`nRanks / nSteps`？**Это ключевая особенность алгоритма Ring: объём данных, фактически передаваемых каждым рангом, равен`nBytes * nSteps / nRanks`(поскольку данные должны пройти по кольцу несколько кругов). Поэтому «эффективная пропускная способность» = пропускная способность шины × nRanks / nSteps. Для AllReduce nSteps = 2(nRanks-1), поэтому эффективная пропускная способность ≈ busBw/2.

**Шаг 4: вычисление задержки.**Задержка делится на две части: intra и inter:

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

Обратите внимание на особую обработку в L57-58: когда`maxLocalRanks == 1`(в каждом узле только 1 ранг), для inter-node задержки Ring используется**NET-задержка Tree**. В комментарии сказано, что это «preserve the pre-refactor model» — то есть намеренно сохранённая «странность» для поддержания совместимости с поведением до рефакторинга.**Такой исторический багаж очень распространён в зрелых системах. При чтении исходного кода, увидев слово «preserve», будьте особенно внимательны — оно часто означает наличие неизменяемого ограничения совместимости.**

**Шаг 5: накопление по типам функций.**Модели задержки для Reduce/Broadcast и AllReduce/AllGather/ReduceScatter различаются:

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

`sameChannels`— это топологическое свойство, означающее «используют ли intra- и inter-шаги на кольце одну и ту же группу каналов». Если нет, задержку нужно умножить на`nSteps`(ожидание на каждом шаге).`netOverhead`— это накладные расходы на сетевой post. Для протокола Simple нужно умножить на 3 (поскольку Simple имеет три сетевых обмена: send, recv, ack).

## Производственные подводные камни: эффект plateau в Ring/Simple

`ncclTuningRingModelSim`В

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
> **Что такое plateau?**В Ring/Simple, когда сообщение достигает определённого размера, задержка перестаёт линейно расти с размером сообщения и «застревает» на плато — потому что в этот момент узкое место смещается от «затрат на запуск» к «пропускной способности», а пропускная способность уже насыщена. Это явление особенно заметно на Blackwell NVLink (поскольку пропускная способность NVLink очень высока, доля задержки больше). В коде`plateauFactor`(1.4 или 1.9) умножается на задержку, имитируя этот эффект «усиления задержки».

`bytesPerRankPerChannel >= 64`— это условие срабатывания: каждый ранг должен передать как минимум 64 байта на канал, иначе plateau не наступает. Эти 64 байта происходят из размера флага протокола LL.

**Сценарий подводного камня**: если вы запускаете AllReduce размером 1MB на Blackwell и обнаруживаете, что фактическая задержка на 40% выше предсказанной моделью, не думайте, что это баг — это эффект plateau, и модель уже его учла. Если вы вручную уменьшите`plateauFactor`, модель будет недооценивать задержку, что приведёт к выбору неправильного алгоритма.

---

# IV. tree.cc и nvls.cc: оценка стоимости Tree и NVLS

## Интуитивная модель

**Алгоритм Tree**— это «**древовидная широковещательная рассылка**»: корневой узел распределяет данные дочерним узлам, дочерние — внучатым. Его преимущество —**малое число шагов**(log N вместо N), подходит для маленьких сообщений; недостаток —**низкая утилизация пропускной способности**(каждый нелистовой узел должен пересылать данные, фактическая эффективная пропускная способность составляет лишь половину).

**NVLS**(NVLink SHARP) — это «**аппаратный мультикаст**»: коммутатор напрямую копирует данные нескольким GPU, без программной пересылки. Его преимущества —**высокая пропускная способность, низкая задержка**, но требуется определённое оборудование (Hopper и новее) и определённая конфигурация.

## Модель Tree: обслуживает только AllReduce

У модели Tree есть жёсткое ограничение —**включается только для AllReduce**：

[FACT:src/tuning/tree.cc:21-27]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
```

> **[Design Inference & Architectural Trade-offs]**
> **Почему?**Потому что реализация Tree в NCCL поддерживает только AllReduce (для других коллективных операций нет версии Tree). Это ограничение реализации, а не теоретическое ограничение.`enabled[c] = 0`— это «жёсткое отключение», более радикальное, чем`generalBandwidths = -1`— первое заставляет`ncclTuningCostModelSimModel`вернуться на L480, второе проверяется только внутри sim-функции.`not_valid`, второе проверяется только внутри sim-функции.

**Оценка пропускной способности Tree**：

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
> Обратите внимание, что коэффициент дисконтирования протокола LL равен`1/3.8`, что жёстче, чем`0.5`у Ring.**Почему эффективность LL у Tree ниже?**Потому что каждый промежуточный узел Tree должен и принимать, и отправлять, а накладные расходы на флаги LL усиливаются при двунаправленном трафике.`1/3.8`Это число получено из измерений.

**Оценка задержки Tree**：

[FACT:src/tuning/tree.cc:55-58]

```c
    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
```

`2 *`— потому что AllReduce = ReduceScatter + AllGather, два прохода.`(nRanks/nNodes - 1)`— число внутриузловых шагов (число рангов в узле минус один),`log2i(nNodes)`— число межузловых шагов (высота дерева).

**Поправочный коэффициент Tree**：Модель Tree на этапе sim умножается на`treeCorrectionFactor`：

[FACT:src/tuning/tree.cc:75-79]

```c
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize proto][logSize];
```

`treeCorrectionFactor`— это таблица 3×24:

[FACT:src/tuning/cost_model.cc:223-227]

```c
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};
```

`logSize = log2(nBytes >> 6)`, то есть размер сообщения берётся log2 с единицей измерения 64 байта. Индексы таблицы 0-23 соответствуют от 64B до 64B×2^23 ≈ 512MB.**Эта таблица — измеренная на практике «кривая эффективности Tree»**: при малых сообщениях эффективность 1.0 (доминирует задержка), при средних сообщениях эффективность падает до 0.4-0.5 (пропускная способность не насыщена), при больших сообщениях возвращается к 1.0 (пропускная способность насыщена). Эта «впадина в середине» — врождённая особенность алгоритма Tree.

## Модель NVLS: цена аппаратной многоадресной рассылки

Модель NVLS сначала проверяет, поддерживается ли аппаратно:

[FACT:src/tuning/nvls.cc:19-24]

```c
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!ncclNvlsTransportEnabled(comm)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

Затем идёт ряд жёстких ограничений: поддерживается только протокол Simple, NVLSTree не поддерживается на одной машине, для NVLS на нескольких машинах требуется CollNet:

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

**Оценка пропускной способности NVLS**Используется коэффициент эффективности:

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
> Для Hopper — 0.85, для Blackwell наоборот снижается до 0.74.**Почему эффективность нового поколения оборудования ниже?**Потому что пропускная способность NVLink у Blackwell выше, но вычислительная способность коммутатора NVLS не выросла пропорционально, что привело к относительному снижению эффективности. Это число измерено на практике, а не теоретическое значение.

В расчёте пропускной способности есть фактор`(nChannels - 1) / nChannels`:

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

`(nChannels - 1) / nChannels`потому что NVLS нужно оставить один channel для синхронизации.`(ppn - 1) / ppn`— это дополнительные накладные расходы AllGather/ReduceScatter (каждый rank должен ждать данные предыдущего rank).

## Производственные подводные камни: жёсткие ограничения NVLS

Модель NVLS на этапе sim имеет ещё один уровень проверки во время выполнения:

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

`NCCL_MAX_NVLS_ARITY`— максимальное число GPU, которое может вместить группа многоадресной рассылки NVLS. Если это число превышено, NVLS недоступен.**Сценарий подводного камня**: запуск AllGather в домене NVLink на 16 карт, если`NCCL_MAX_NVLS_ARITY`равно 8, NVLS будет отключён, и tuning откатится к Ring. Если вы не знаете этого ограничения, будете думать: «NVLS ведь аппаратно поддерживается, почему не используется».

---

# V. Откат симметричного kernel и цепочка восстановления после ошибок

## Интуитивная модель

Симметричный kernel (symmetric kernel) — новая возможность NCCL: когда буферы всех rank зарегистрированы в симметричной памяти, kernel может обращаться к памяти партнёра более эффективными инструкциями. Но**если буфер не зарегистрирован или платформа не поддерживается, необходимо откатиться к обычному kernel**. Эта логика отката — самая запутанная часть в tuning.

## Step-by-Step: решение об откате

Логика отката находится в`tuning.cc:258-298`. Разберём по частям.

**Шаг 1: определить, нужен ли откат.**Входное условие:

[FACT:src/tuning/tuning.cc:258-263]

На этом цепочка решений модуля tuning уже ясна: он получает граф топологии и параметры коммуникации, через модель стоимости и оценку алгоритмов за микросекунды выдаёт оптимальную комбинацию (алгоритм, протокол, channel, warp). Но выбор — это только начало — как этот результат решения используется ниже по потоку? В следующей главе мы войдём в основную часть src/enqueue/enqueue.cc и посмотрим, как вызов ncclAllReduce проходит проверку параметров, определение алгоритма/протокола, разбиение на channel и в итоге порождает структуры ncclInfo и ncclTaskColl. Это ключевая глава книги, где происходит переключение с «точки зрения пользователя» на «точку зрения движка»; вы выясните, во что на стороне host транслируется один вызов коллективной коммуникации и какова граница между ним и последующим запуском kernel.
