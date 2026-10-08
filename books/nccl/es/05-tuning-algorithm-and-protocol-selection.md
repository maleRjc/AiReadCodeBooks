# Capítulo 5: Selección de algoritmo y protocolo: cómo el módulo tuning decide la ruta de comunicación

En el capítulo anterior desglosamos la capacidad de reconocimiento topológico de NCCL: desde la enumeración de dispositivos en src/graph/topo.cc para construir el grafo topológico, pasando por la búsqueda de rutas óptimas en src/graph/search.cc, hasta la materialización de los resultados de búsqueda en topologías de algoritmos Ring y Tree en rings.cc y trees.cc. Pero el grafo topológico solo responde a «por dónde pueden ir los datos», no responde a «por dónde debería ir esta comunicación». En una misma máquina, un AllReduce de 4KB y uno de 400MB pueden tener soluciones óptimas completamente distintas: el primero compite por latencia, el segundo por ancho de banda; el primero podría elegir Tree/LL, el segundo Ring/Simple o NVLS. El módulo tuning es ese «árbitro». Sus entradas son el tamaño del mensaje, el número de ranks, el grafo topológico (producto del capítulo anterior) y las variables de entorno del usuario; su salida es un ncclTuningResult_t, que indica qué algoritmo (algo), qué protocolo (proto), cuántos channels y cuántos warps usar. En este capítulo desglosamos el directorio src/tuning en el orden «planificación general → modelo de costos → estimación por algoritmo → decisión final». La pregunta central es una sola: ¿cómo selecciona NCCL, entre docenas de combinaciones (algoritmo, protocolo), la más rápida en tiempo de microsegundos usando un modelo matemático puramente de CPU?

# I. tuning.cc: planificación general y tronco de decisión

## Modelo intuitivo

Imagina el módulo tuning como una**empresa de mudanzas**. Llega un cliente (una comunicación colectiva) y dice «quiero mover 100MB de mercancía, de 8 almacenes a 8 almacenes». El despachador (`ncclTuningCompute`) no va a moverla de verdad para probar, sino que saca una**tabla de precios**(modelo de costos), estima un «tiempo estimado» para cada opción (Ring/LL, Tree/Simple, NVLS/Simple…) y elige la cotización más corta para el cliente.

Sin este despachador, NCCL solo podría codificar «AllReduce siempre usa Ring», lo que en escenarios de mensajes pequeños sería aplastado por Tree, y en escenarios NVLink a gran escala sería aplastado por NVLS.**El costo es que el rendimiento se reduce a la mitad o incluso peor en escenarios específicos.**

## Estructuras de datos y diseño de memoria

El portador de la decisión es`ncclTuningResult_t`, y el conjunto de candidatos es`ncclTuningResultList_t`(una lista simplemente enlazada). Los nodos de la lista se definen en`tuning_int.h`, pero la lógica de push está en`tuning.cc`:

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
> Nota que aquí se usa**inserción en la cabeza**: cada vez que se calcula un candidato válido, se inserta al inicio de la lista. Esto significa que el orden de la lista y el orden de los id son**inversos**. ¿Por qué usar una lista enlazada en lugar de un arreglo? Porque la cantidad de candidatos está determinada en tiempo de compilación por`NCCL_TUNING_COUNT`, pero los candidatos realmente válidos son dinámicos (afectados por`tuningMask`, capacidades de la plataforma, variables de entorno del usuario), y la lista enlazada permite «colgar solo los válidos», evitando evaluar repetidamente`valid`durante el recorrido. El costo es que cada decisión requiere`ncclCalloc`una vez, pero el tuning ocurre en la ruta de encolado y con poca frecuencia, por lo que este coste de asignación es aceptable.

`ncclTuningResult_t`Los dos campos más críticos son`timeUs`(tiempo estimado, microsegundos) y`selectionTimeUs`(tiempo usado para la selección, puede ser sobrescrito por el plugin tuner). La lógica de selección solo mira el segundo:

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

Aquí hay un detalle:`bestTuning->timeUs`primero se establece en`FLT_MAX`, luego se recorre. Si la lista enlazada está vacía (todos los candidatos son inválidos),`bestTuning`mantendrá`NCCL_TUNING_RESULT_INIT`el valor inicial de, algo/proto son`UNDEF`. Este «resultado vacío» se maneja de forma especial en el llamador — véase la rama de error más adelante.

## Step-by-Step Walkthrough: el flujo de decisión de un AllReduce

Supongamos que la aplicación llama a`ncclAllReduce`, mensaje de 1MB, 8 ranks en una sola máquina con NVLink. Seguimos`ncclTuningCompute`paso a paso.

**Paso 0: cortocircuito de un solo rank.**Si`nRanks <= 1`, no se necesita comunicación en absoluto, se devuelve directamente Ring/Simple, con el número de channels establecido en 0:

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

Aquí`NCCL_TUNING_IGNORE`es un valor centinela que indica «esta combinación no se ha calculado / no aplica». El plugin puede modificar solo las celdas que le interesan, dejando las demás en IGNORE, y NCCL las omitirá.

**Paso 4: elegir el óptimo.**llama a`ncclTuningSelectBestTuning`, recorre la lista enlazada y toma el`selectionTimeUs`mínimo.

**Paso 5: calcular el número de channels.**Tras elegir el algoritmo, aún hay que decidir cuántos channels abrir:

[FACT:src/tuning/tuning.cc:233-235]

```c
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
```

`ncclTuningGetChannels`En`tuning_int.h`, la lógica consiste en interpolar entre`minChannels`y`maxChannels`según el tamaño del mensaje y el tipo de algoritmo. El número de channels afecta directamente al ancho de banda: cuantos más channels, mayor paralelismo, pero también mayor coste de arranque por channel.

**Paso 6: sesgo de CTA Policy (prioridad NVLS).**Si el usuario ha establecido`NCCL_CTA_POLICY_EFFICIENCY`, y actualmente es AllGather/ReduceScatter y el buffer está registrado, NCCL intentará cambiar el resultado a NVLS:

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

**¿Por qué distinguir los códigos de error?**Si el usuario ha establecido`NCCL_ALGO=ring`pero la plataforma actual no soporta ring (por ejemplo, ciertas topologías especiales), eso es**error de configuración del usuario**（`ncclInvalidUsage`); si el usuario no ha establecido ninguna variable de entorno y aun así no se puede elegir un algoritmo, eso es**un bug interno de NCCL**（`ncclInternalError`). Esta distinción es crucial para la resolución de problemas.

## Diagrama de flujo del tronco de decisión

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

# II. cost_model.cc: registro de modelos y matriz de interruptores

## Modelo intuitivo

`cost_model.cc`es el**libro mayor**del tuning. Mantiene una tabla`modelMap`, donde cada fila corresponde a una combinación (algo, proto) y registra «quién es la función de inicialización de esta combinación, quién es la función de simulación y para qué funciones está habilitada». Además, se encarga de parsear la variable de entorno del usuario`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SYM_KERNEL`, traduciendo la intención del usuario en una matriz de interruptores`enabled[i][f]`.

Sin esta tabla, cada vez que se añadiera un nuevo algoritmo habría que modificar el flujo principal de tuning, y el código se convertiría en un desastre.**La tabla como driver**convierte «añadir un algoritmo» en «añadir una fila».

## Estructuras de datos: modelMap y matriz de interruptores

`modelMap`es un array estático, cada elemento es`ncclTuningModelEntry_t`：

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

Cada entry tiene cuatro campos:`init`(inicialización, calcula latency/bandwidth y lo guarda en comm),`model`(simulación, calcula el timeUs final según el tamaño del mensaje),`finalize`(limpieza),`enabled[5]`(sobre si se habilitan las cinco funciones Broadcast/Reduce/AllGather/ReduceScatter/AllReduce).

Nota`enabled`El comentario sobre el orden del array está en L234:`Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce`. Este orden debe coincidir con`ncclFunc_t`el enum, de lo contrario se confundirán los elementos.

> **[Design Inference & Architectural Trade-offs]**
> **¿Por qué init y sim deben estar separados?**Porque lo que se calcula en init (latency, bandwidth)**solo depende de las propiedades estáticas de comm**(topología, número de ranks, compCap), y no tiene relación con el tamaño concreto del mensaje. En una comunicación pueden llamarse consecutivamente múltiples tuning (por ejemplo, si el group tiene varios op), init se ejecuta solo una vez, sim se ejecuta cada vez. Esta es la típica optimización de «precálculo + consulta rápida».

## Paso a paso: análisis de variables de entorno y construcción de la matriz de interruptores

**Paso 1: por defecto todo activado, LL128 es especial.** `ncclTuningCostModelInit`Al principio se ponen todos los proto a 1 (habilitado), pero LL128 se pone a 2:

[FACT:src/tuning/cost_model.cc:313-323]

```c
  for (int f = 0; f minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                          comm->graphs[algo].typeIntra, comm->nRanks, f, algo, comm->minDriverVersion)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
```

**Paso 2: analizar las variables de entorno del usuario.**Si el usuario configuró`NCCL_ALGO`o`NCCL_SYM_KERNEL`, primero se ponen a cero todos los algo y symKernel (porque el usuario especificó una lista blanca):

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

Nota: proto no se pone a cero — porque el valor por defecto de proto es 1/2, cuando el usuario configura`NCCL_PROTO=LL`,`parseList`pondrá LL a 1 y los demás a 0 (debido a la lógica de`unset`). Esta asimetría es intencional: algo está todo activado por defecto pero tras la especificación del usuario debe reducirse, la reducción de proto la maneja internamente`parseList`.

**Paso 3: sintaxis de parseList.**Esta función soporta una sintaxis bastante compleja, en los comentarios se dan ejemplos:

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

`^`El prefijo indica «negación»:

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

Por lo tanto`NCCL_PROTO="^LL128;allreduce:LL128"`significa: deshabilitar LL128 globalmente, pero habilitar LL128 como excepción para AllReduce.

**Paso 4: fusionar la matriz enabled.**Finalmente se recorren todos los model, y se hace una operación AND entre`model->enabled[f]`y los interruptores del usuario:

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

La lógica es:**Solo cuando el usuario ha configurado forced para alguna función, se sobrescribe el valor por defecto del modelo con la configuración del usuario**. Si el usuario no lo configuró,`forced[f] == 0`, directamente`continue`, conservando el`enabled`del propio modelo. Esta es la prioridad de «especificación explícita del usuario > valor por defecto del modelo».

## Entrada unificada de la simulación de modelos

Todos los modelos finalmente se invocan a través de`ncclTuningCostModelSimModel`:

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

Tres capas de filtrado:**id fuera de rango → modelo deshabilitado → el modelo devuelve un tiempo no positivo**, si alguna capa no pasa se va a`not_valid`, poniendo`timeUs`a`NCCL_TUNING_IGNORE`(un centinela negativo),`valid = 0`. Cuando el llamador ve`valid == 0`no lo insertará en la lista enlazada de candidatos.

## Reflexión de diseño

`modelMap`En los comentarios de

[FACT:src/tuning/cost_model.cc:229]

```c
// IMPORTANT: this table need must be consistent with the algRegistry in src/config/algorithm_registry.cc
```

> **[Design Inference & Architectural Trade-offs]**
> [Inferencia de diseño y compensaciones arquitectónicas]`modelMap`Esto significa que**el**orden de índices`algorithm_registry.cc`debe coincidir estrictamente con el orden de registro de algoritmos en`modelMap`. Si alguien inserta un nuevo algoritmo en el registry pero olvida modificar**, todos los id se desalinearán y tuning elegirá un algoritmo completamente erróneo.**Esta es la trampa clásica del diseño basado en tablas: el contrato implícito.

---

# Una práctica más robusta sería usar el nombre del enum como key en lugar del índice, pero eso sacrificaría un poco de optimización en tiempo de compilación.

## III. ring.cc: estimación del coste del algoritmo Ring

Modelo intuitivo**El algoritmo Ring coloca N ranks en un anillo, y los datos se transmiten vuelta tras vuelta a lo largo del anillo. Su modelo de coste debe responder dos preguntas:**、**Cuántos datos se transmiten en cada paso (bandwidth)**。

Cuántos pasos se necesitan en total (latency)**La intuición de Ring es «**pipeline

## »: imagina N personas en círculo pasándose un cubo, cada persona al recibir el cubo vierte un poco de agua y lo pasa a la siguiente. Cuando el cubo da una vuelta, el agua de todos se ha mezclado uniformemente. Cuanto más rápido gire el cubo (mayor bandwidth), cuanto más pequeño sea el círculo (menos pasos), más rápido será en conjunto.

Estructura de datos: tabla latency/bandwidth`comm->tuningContext.generalLatencies[c][algo][proto]`El modelo Ring no introduce nuevas estructuras, escribe los resultados de la estimación en`generalBandwidths[c][algo][proto]`y

. Estos dos son arrays tridimensionales: función × algoritmo × protocolo.

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

**Copiar**¿Por qué usar -1.0 en lugar de 0?`==`Porque 0 es un valor de bandwidth legal (aunque físicamente imposible), mientras que -1.0 indica claramente «no inicializado». La comparación de punto flotante con

## aquí es segura, porque -1.0 es exactamente representable.

**Paso a paso: estimación de bandwidth de Ring**Paso 1: determinar si usar bandwidth intra o inter.

[FACT:src/tuning/ring.cc:34-37]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float bw = (comm->nNodes == 1 || (comm->nNodes minCompCap graphs[algo].bwIntra :
                                                                                      comm->graphs[algo].bwInter;
    float busBw = bw * comm->graphs[algo].nChannels;
```

`nSteps`Copiar`2*(nRanks-1)`es el número de pasos que necesita el algoritmo, para Ring AllReduce es`nRanks-1`。`busBw`, los demás son

**es el «bandwidth de bus» = bandwidth de enlace único × número de channels.**El protocolo LL solo usa la mitad del ancho de banda (debido al overhead de flag de LL), LL128 usa el 92% (120/128):

[FACT:src/tuning/ring.cc:38-42]

```c
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);
```

`0.92 = 120/128`Esto se debe a que en LL128, de cada 128 bytes, 8 bytes son flag, y la carga útil es solo de 120 bytes. Este número proviene directamente del diseño del protocolo.

**Paso 3: Calcular el ancho de banda efectivo.**Nota: aquí se multiplica por`nRanks / nSteps`：

[FACT:src/tuning/ring.cc:44-46]

```c
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;
```

**¿Por qué multiplicar por`nRanks / nSteps`？**Esta es la característica central del algoritmo Ring: la cantidad de datos que cada rank realmente transporta es`nBytes * nSteps / nRanks`(porque los datos deben dar varias vueltas alrededor del anillo). Por lo tanto, el "ancho de banda efectivo" = ancho de banda del bus × nRanks / nSteps. Para AllReduce, nSteps = 2(nRanks-1), así que el ancho de banda efectivo ≈ busBw/2.

**Paso 4: Calcular la latencia.**La latencia se divide en dos partes: intra e inter:

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

Nota el manejo especial en L57-58: cuando`maxLocalRanks == 1`(cada nodo tiene solo 1 rank), la latencia inter-node de Ring usa**la latencia NET de Tree**. El comentario dice que esto es "preserve the pre-refactor model" — es decir, una "rareza" conservada deliberadamente para mantener la consistencia con el comportamiento previo a la refactorización.**Este tipo de lastre histórico es muy común en sistemas maduros. Al leer el código fuente, hay que tener especial cuidado cuando se ve la palabra "preserve", ya que a menudo implica que hay una restricción de compatibilidad que no se puede modificar.**

**Paso 5: Acumular según el tipo de función.**Los modelos de latencia de Reduce/Broadcast y AllReduce/AllGather/ReduceScatter son diferentes:

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

`sameChannels`Es una propiedad topológica que indica "si los pasos intra e inter en el anillo usan el mismo conjunto de channels". Si son diferentes, la latencia debe multiplicarse por`nSteps`(hay que esperar en cada paso).`netOverhead`Es el overhead de post de red; el protocolo Simple debe multiplicarse por 3 (porque Simple tiene tres idas y vueltas de red: send, recv, ack).

## Evitar trampas en producción: el efecto plateau de Ring/Simple

`ncclTuningRingModelSim`Hay una sección de código dedicada a manejar el "plateau":

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
> **¿Qué es el plateau?**En Ring/Simple, cuando el mensaje alcanza cierto tamaño, la latencia deja de crecer linealmente con el mensaje y se "estanca" en una plataforma — porque en ese momento el cuello de botella pasa de ser el "overhead de inicio" a ser el "ancho de banda", y el ancho de banda ya está saturado. Este fenómeno es especialmente evidente en Blackwell NVLink (porque el ancho de banda de NVLink es muy alto y la latencia ocupa una proporción mayor). El código multiplica`plateauFactor`(1.4 o 1.9) por la latencia para simular este efecto de "latencia amplificada".

`bytesPerRankPerChannel >= 64`Es la condición de activación: cada rank debe transmitir al menos 64 bytes por channel, de lo contrario el plateau no se cumple. Estos 64 bytes provienen del tamaño del flag del protocolo LL.

**Escenario de trampa**: Si ejecutas un AllReduce de 1MB en Blackwell y descubres que la latencia real es un 40% mayor que la predicha por el modelo, no pienses que es un bug — esto es el efecto plateau, y el modelo ya lo ha tenido en cuenta. Si modificas manualmente`plateauFactor`a un valor menor, el modelo subestimará la latencia, lo que llevará a elegir el algoritmo incorrecto.

---

# IV. tree.cc y nvls.cc: Estimación de costos de Tree y NVLS

## Modelo intuitivo

**El algoritmo Tree**es una "**difusión en árbol**": el nodo raíz distribuye los datos a los nodos hijos, y estos a su vez a los nodos nietos. Su ventaja es que tiene**pocos pasos**(log N en lugar de N), adecuado para mensajes pequeños; su desventaja es que tiene**baja utilización del ancho de banda**(cada nodo no hoja debe reenviar, por lo que el ancho de banda efectivo real es solo la mitad).

**NVLS**(NVLink SHARP) es "**multicast por hardware**": el switch copia directamente los datos a múltiples GPUs, sin necesidad de reenvío por software. Su ventaja es que tiene**alto ancho de banda y baja latencia**, pero requiere hardware específico (Hopper o superior) y una configuración específica.

## Modelo Tree: solo sirve para AllReduce

El modelo Tree tiene una restricción estricta —**solo se habilita para AllReduce**：

[FACT:src/tuning/tree.cc:21-27]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
```

> **[Design Inference & Architectural Trade-offs]**
> **¿Por qué?**Porque la implementación de Tree en NCCL solo soporta AllReduce (las demás operaciones colectivas no tienen versión Tree). Esta es una restricción de implementación, no una limitación teórica.`enabled[c] = 0`Es una "deshabilitación estricta", más contundente que`generalBandwidths = -1`— la primera hace que`ncclTuningCostModelSimModel`retorne en L480, mientras que la segunda solo se verifica dentro de la función sim.`not_valid`Estimación de ancho de banda de Tree

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
> , más agresivo que el`1/3.8`de Ring.`0.5`¿Por qué la eficiencia de LL en Tree es menor?**Porque cada nodo intermedio en Tree debe tanto recibir como enviar, y el overhead de flag de LL se amplifica con el tráfico bidireccional.**Este número proviene de mediciones reales.`1/3.8`Estimación de latencia de Tree

**copiar**：

[FACT:src/tuning/tree.cc:55-58]

```c
    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
```

`2 *`Es el número de pasos intra-nodo (el número de ranks dentro de cada nodo menos uno),`(nRanks/nNodes - 1)`es el número de pasos inter-nodo (la altura del árbol).`log2i(nNodes)`El factor de corrección de Tree

**Tree 的修正因子**: El modelo Tree multiplica en la fase de sim un`treeCorrectionFactor`：

[FACT:src/tuning/tree.cc:75-79]

```c
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize proto][logSize];
```

`treeCorrectionFactor`es una tabla de 3×24:

[FACT:src/tuning/cost_model.cc:223-227]

```c
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};
```

`logSize = log2(nBytes >> 6)`, es decir, el tamaño del mensaje se toma en log2 con unidades de 64 bytes. Los índices 0-23 de la tabla corresponden desde 64B hasta 64B×2^23 ≈ 512MB.**Esta tabla es la «curva de eficiencia de Tree» medida empíricamente**: con mensajes pequeños la eficiencia es 1.0 (dominada por la latencia), con mensajes medianos la eficiencia cae a 0.4-0.5 (el ancho de banda no se satura), y con mensajes grandes vuelve a 1.0 (ancho de banda saturado). Esta «depresión intermedia» es una característica inherente del algoritmo Tree.

## Modelo NVLS: el costo de la multidifusión por hardware

El modelo NVLS primero verifica si el hardware lo soporta:

[FACT:src/tuning/nvls.cc:19-24]

```c
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!ncclNvlsTransportEnabled(comm)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

Luego hay una serie de restricciones estrictas: solo soporta el protocolo Simple, no soporta NVLSTree en una sola máquina, y NVLS multi-máquina requiere CollNet:

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

**Estimación de ancho de banda de NVLS**Se utiliza un factor de eficiencia:

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
> Hopper es 0.85, mientras que Blackwell baja a 0.74.**¿Por qué el hardware de nueva generación tiene menor eficiencia?**Porque el ancho de banda NVLink de Blackwell es mayor, pero la capacidad de procesamiento del switch NVLS no aumentó proporcionalmente, lo que provoca una caída en la eficiencia relativa. Este número es medido empíricamente, no es un valor teórico.

En el cálculo del ancho de banda hay un`(nChannels - 1) / nChannels`factor:

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

`(nChannels - 1) / nChannels`es porque NVLS necesita reservar un canal para sincronización.`(ppn - 1) / ppn`es el costo adicional de AllGather/ReduceScatter (cada rank debe esperar los datos del rank anterior).

## Evitando trampas en producción: las restricciones estrictas de NVLS

El modelo NVLS en la fase de sim tiene además una capa de verificación en tiempo de ejecución:

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

`NCCL_MAX_NVLS_ARITY`es el número máximo de GPUs que puede contener un grupo de multidifusión NVLS. Si se supera este número, NVLS no está disponible.**Escenario problemático**: al ejecutar AllGather en un dominio NVLink de 16 tarjetas, si`NCCL_MAX_NVLS_ARITY`es 8, NVLS se deshabilitará y el tuning recurrirá a Ring. Si no conoces esta limitación, pensarás «si NVLS tiene soporte de hardware, ¿por qué no se usa?».

---

# V. Retroceso a kernel simétrico y cadena de recuperación de errores

## Modelo intuitivo

El kernel simétrico (symmetric kernel) es una nueva característica de NCCL: cuando los buffers de todos los ranks están registrados en memoria simétrica, el kernel puede acceder a la memoria del par con instrucciones más eficientes. Pero**si el buffer no está registrado, o la plataforma no lo soporta, se debe recurrir al kernel normal**. Esta lógica de retroceso es la parte más enrevesada del tuning.

## Paso a paso: decisión de retroceso

La lógica de retroceso está en`tuning.cc:258-298`. Vamos a desglosarla.

**Paso 1: determinar si se necesita retroceso.**Condición de entrada:

[FACT:src/tuning/tuning.cc:258-263]

Hasta aquí, la cadena de decisiones del módulo de tuning ya está clara: recibe el grafo de topología y los parámetros de comunicación, y mediante modelos de costo y estimaciones de algoritmos, produce en microsegundos la combinación óptima de (algoritmo, protocolo, channel, warp). Pero la selección es solo el comienzo—¿cómo se utiliza este resultado de decisión aguas abajo? En el próximo capítulo entraremos en el tronco de src/enqueue/enqueue.cc, para ver cómo una llamada a ncclAllReduce pasa por validación de parámetros, determinación de algoritmo/protocolo, división de channels, y finalmente genera las estructuras ncclInfo y ncclTaskColl. Este es el capítulo clave del libro donde se cambia de la «perspectiva del usuario» a la «perspectiva del motor»; descubrirás en qué se traduce una llamada de comunicación colectiva en el lado del host, y cuál es su frontera con el lanzamiento posterior del kernel.
