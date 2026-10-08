# Capítulo 24: Evolución de la arquitectura y direcciones futuras: de la comunicación estática a la comunicación programable

En el capítulo anterior vimos cómo la comunidad construye un ecosistema periférico alrededor del núcleo de NCCL: bindings de Python, bindings de Rust, comunicación de paralelismo experto, primitivas de ultraancho de banda, checkpoints de comunicación. Estos proyectos reutilizan la API estable de NCCL, pero sus demandas ya superan el ámbito de la comunicación colectiva tradicional — el paralelismo experto necesita envío/recepción punto a punto de grano fino, los checkpoints necesitan pausar/reanudar el estado de comunicación, y las primitivas de ultraancho de banda necesitan eludir las operaciones colectivas estándar para operar directamente sobre la red. Estas demandas apuntan al mismo problema: el modelo de operaciones colectivas fijas de NCCL está siendo desbordado por necesidades de comunicación más flexibles. En este capítulo ya no examinaremos un único módulo, sino que, partiendo de las huellas de evolución que ya aparecen en el código fuente, discutiremos hacia dónde se dirige NCCL. En concreto, analizaremos tres fuerzas de evolución entrelazadas: las primitivas de comunicación pasan de colectivas fijas a programables — la planificación de tareas RMA en src/rma/rma.cc permite a las capas superiores combinar las primitivas Put/Signal/WaitSignal, en lugar de poder llamar solo a AllReduce; el inicio de red pasa de host proxy a envío directo desde GPU — la gestión del backend GIN en src/gin/gin_host.cc permite que el kernel de GPU impulse directamente la tarjeta de red; el modelo de memoria pasa de búferes registrados a memoria simétrica — la selección de kernel de memoria simétrica en src/sym_kernels.cc permite que todos los ranks usen el mismo conjunto de direcciones virtuales para acceder a los búferes de los demás. Estas tres fuerzas no están aisladas, comparten la misma infraestructura: la abstracción de team en src/nccl_device/core.cc y el DevComm versionado en src/devcomm/devcomm_v23100.cc. Entender cómo encajan entre sí es entender la lógica de evolución de NCCL desde "biblioteca de comunicación colectiva" hasta "motor de comunicación programable".

# I. Primitivas de comunicación programables: cómo RMA convierte la "receta fija" en un "buffet"

## Modelo intuitivo

La comunicación colectiva de NCCL tradicional es como un menú fijo: pides AllReduce y la cocina lo prepara siguiendo el flujo de AllReduce. Pero en escenarios de paralelismo de expertos (MoE), cada token debe enviarse a expertos diferentes, y el patrón de envío no se conoce en tiempo de compilación — esto es como un buffet, donde tú decides qué tomar, cuánto tomar y cuándo tomarlo.

RMA es precisamente el «mostrador de buffet» que NCCL ofrece a las capas superiores: Put (escribir datos en la memoria del par), Signal (notificar al par), WaitSignal (esperar la señal del par). El framework de capas superiores puede combinar libremente estas tres primitivas para implementar cualquier patrón de comunicación.

Sin RMA, el all-to-all de MoE solo podría simularse mediante múltiples operaciones colectivas de pequeña escala, cada una requiriendo el flujo completo de lanzamiento de kernel y sincronización, con una latencia inaceptablemente alta.

## Estructuras de datos y diseño de memoria

La estructura de datos central de RMA es`ncclTaskRma`(descripción de tarea) y`ncclRmaArgs`(parámetros del plan). Primero veamos`ncclRmaArgs`los campos de, que se inicializa en`scheduleRmaTasksToPlan`.

[FACT:src/rma/rma.cc:166-171]

```cpp
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
plan->rmaArgs->nRmaTasks = 0;
plan->rmaArgs->nRmaTasksProxy = 0;
plan->rmaArgs->nRmaTasksCe = 0;
```

Los campos clave aquí son`nRmaTasksProxy`y`nRmaTasksCe`. Estos dividen las tareas RMA en dos rutas de ejecución:

- **Ruta CE**(Copy Engine, motor de copia): el rank destino está dentro del rango LSA (Local Symmetric Access, acceso simétrico local), y puede completarse directamente con el motor de copia de la GPU, sin necesidad de red.
- **Ruta Proxy**: el rank destino no está dentro del rango LSA y debe pasar obligatoriamente por el hilo proxy del host para impulsar la red.

> **[Design Inference & Architectural Trade-offs]**
> La motivación de este diseño dicotómico es directa: la comunicación dentro del rango LSA va por NVLink o PCIe, con alto ancho de banda y baja latencia, por lo que usar copia asíncrona con CE es lo más rentable; la comunicación entre máquinas debe pasar por la tarjeta de red y solo puede ser impulsada por el hilo proxy. Separar la programación de ambos tipos de tareas es lo que permite que CE y proxy se ejecuten en paralelo, en lugar de esperar en serie.

`ncclTaskRma`contiene en sí mismo`peers`、`nsignals`、`signalIdxs`tres punteros a arrays, que registran respectivamente el rank par, la cantidad de señales y el índice de señal. Para tareas WaitSignal, una tarea puede esperar a múltiples peers; para tareas Put/Signal, una tarea apunta a un solo peer.

## Step-by-Step Walkthrough: la programación de un WaitSignal

Tomemos un escenario concreto: el rank 0 llama a`ncclWaitSignal`, esperando señales del rank 1 y del rank 3. Supongamos que el rank 1 está dentro del rango LSA y el rank 3 no.

**Primer paso: encontrar la primera cola de contexto no vacía.**

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

Las tareas RMA se organizan en colas por contexto, y cada contexto es un canal RMA independiente. Aquí se encuentra el primer contexto con tareas y se extrae su cola.

**Segundo paso: extraer la primera tarea y determinar su tipo.**

[FACT:src/rma/rma.cc:163-168]

```cpp
struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
```

`firstTask->func`es`ncclFuncWaitSignal`, se entra en la rama WaitSignal.

**Tercer paso: dividir los peers según la accesibilidad LSA.**

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

`isLsaAccessible`recorre`comm->devrState.lsaRankList`, determinando si el peer está dentro del equipo LSA. El rank 1 está dentro del LSA y va a la lista CE; el rank 3 no lo está y va a la lista Proxy.

**Cuarto paso: crear una nueva tarea para CE y otra para Proxy.**

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

La tarea WaitSignal original se divide en dos: la tarea CE espera al rank 1 y la tarea Proxy espera al rank 3. Ambas tareas pueden ejecutarse en paralelo — la ruta CE espera en la GPU y la ruta Proxy espera en el hilo del host.

**Quinto paso: liberar la tarea original.**

[FACT:src/rma/rma.cc:249-251]

```cpp
planner->nTasksRma -= 1;
ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
```

La tarea original ya se ha dividido en dos nuevas tareas, se libera de vuelta al pool de memoria.

## Control de concurrencia e interacción con el hardware

La ejecución paralela de RMA se refleja en`ncclRmaWaitSignal`.

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

Este código usa CUDA event para sincronizar entre streams: primero registra un event en el stream de entrada, hace que el stream CE espere este event, luego lanza las tareas proxy y CE en sus respectivos streams, y finalmente hace que el stream de entrada espere el event del stream CE. Así ambas rutas avanzan en paralelo, pero externamente se comportan como una operación síncrona.

> **[Design Inference & Architectural Trade-offs]**
> La compensación de diseño aquí es: la ejecución paralela reduce la latencia, pero introduce sobrecarga adicional de registro de events y sincronización de streams. Para mensajes pequeños, esta sobrecarga puede superar el beneficio del paralelismo; para mensajes grandes, el beneficio del paralelismo es significativo. NCCL no hace aquí un juicio adaptativo, sino que siempre toma la ruta paralela — porque el escenario típico de RMA es la comunicación de grano fino con mensajes grandes.

## Guía de evitación de errores en producción

**Error 1: un juicio incorrecto de accesibilidad LSA hace que la tarea tome la ruta equivocada.** `isLsaAccessible`recorre`lsaRankList`, si`lsaSize`es 0 (por ejemplo, un dominio de comunicación de un solo rank), todos los peers se considerarán inalcanzables y todos tomarán la ruta Proxy. Esto no se manifiesta en pruebas a pequeña escala, pero en despliegues a gran escala provoca una caída drástica del rendimiento. El método de diagnóstico es revisar en los logs INFO de`scheduleRmaTasksToPlan`la proporción de`nRmaTasksProxy`y`nRmaTasksCe`.

**Error 2: el ciclo de vida del array de peers tras la división de una tarea WaitSignal.**El`peersCe`de la ruta CE se asigna con`ncclMemoryStackAlloc`, y su ciclo de vida sigue a`comm->memScoped`; el`peersProxy`de la ruta Proxy se asigna con`ncclCalloc`, y tras ejecutarse la tarea debe liberarse manualmente`free`. Si la creación de la tarea Proxy falla,`fail`la rama liberará estos arrays.

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

**Trampa 3: Procesamiento por lotes entre contextos de tareas Put/Signal.**En la rama Put/Signal, NCCL agrupa las tareas put/signal de todos los contextos en un mismo plan, pero se detiene al encontrar un WaitSignal.

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

La intención de este diseño es: un solo lanzamiento de kernel cubre los put/signal de todos los contextos, reduciendo la sobrecarga de lanzamiento. Pero la cola de cada contexto solo se consume hasta el primer WaitSignal, garantizando el orden FIFO por contexto. Si la capa superior alterna llamadas a put y waitSignal dentro del mismo contexto, el efecto de procesamiento por lotes se reduce drásticamente—este es un patrón a tener en cuenta al usar RMA.

---

# Dos, envío directo a red desde GPU: cómo GIN permite al kernel eludir el host proxy

## Modelo intuitivo

La comunicación de red tradicional de NCCL es como enviar una carta: el kernel de GPU coloca los datos en un búfer, el hilo host proxy entrega los datos a la tarjeta de red, y la tarjeta de red los envía. GIN, en cambio, permite que el kernel de GPU deposite la carta directamente en el buzón del destinatario—el kernel escribe directamente en la cola de envío de la tarjeta de red, y la tarjeta de red lee directamente de la memoria de la GPU.

Sin GIN, cada comunicación de red tendría que pasar por la memoria del host como intermediario, añadiendo al menos un viaje de ida y vuelta por PCIe de latencia. Para comunicaciones de grano fino como MoE, esta latencia es fatal.

## Estructuras de datos y diseño de memoria

El estado central de GIN es`ncclGinState`, que gestiona múltiples backends y múltiples DevComm. Primero veamos la tabla de compatibilidad de versiones de backend.

[FACT:src/gin/gin_host.cc:27-33]

```cpp
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5), NCCL_VERSION(2, 32, 0)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
constexpr int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0), NCCL_VERSION(2, 32, 0)};
```

El índice de estos arrays es el número de versión del backend, y el valor es la versión mínima compatible de NCCL. Por ejemplo,`proxyBackendMinVersions[3]`corresponde al backend versión 3, que requiere NCCL al menos 2.32.0. Este diseño permite que NCCL seleccione la versión de backend adecuada en tiempo de ejecución según la versión del código del dispositivo, en lugar de vincularla en tiempo de compilación.

> **[Design Inference & Architectural Trade-offs]**
> La motivación de este diseño de tabla de compatibilidad de versiones es: el backend de GIN (controlador de tarjeta de red, firmware) y la biblioteca NCCL evolucionan a ritmos diferentes. Si se codificaran rígidamente los requisitos de versión, cualquier actualización de una de las partes causaría incompatibilidad. Usar arrays para el mapeo de versiones permite la selección dinámica en tiempo de ejecución, manteniendo compatibilidad con backends antiguos.

`ncclGinStateDevComm`es el estado GIN de cada DevComm, que contiene campos como`contextCount`、`backendIndex`、`ginCtx[]`、`devHandles[]`. Está encadenado en una lista enlazada colgada de`ginState->devComms`.

## Recorrido paso a paso: el establecimiento de una conexión GIN

Nos situamos en un escenario: el rank 0 inicializa el dominio de comunicación y necesita establecer una conexión GIN.

**Primer paso: verificar si GIN está habilitado y soportado.**

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

`ncclParamGinEnable()`lee la variable de entorno`NCCL_GIN_ENABLE`, por defecto 1. Si el usuario lo deshabilita explícitamente, devuelve error directamente.

**Segundo paso: verificar el soporte de memoria simétrica.**

[FACT:src/gin/gin_host.cc:111-114]

```cpp
if (!comm->symmetricSupport) {
  WARN("Communicator does not support symmetric memory!");
  return ncclInternalError;
}
```

GIN depende de la memoria simétrica—porque el kernel de GPU necesita conocer la dirección virtual del búfer del par, y solo la memoria simétrica puede garantizar la coherencia de direcciones.

**Tercer paso: obtener la lista local de dispositivos GIN.**

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

`ncclTopoGetLocalGinDevs`encuentra en el grafo de topología todas las tarjetas de red que soportan GIN. Si superan`NCCL_GIN_MAX_CONNECTIONS`, solo toma las primeras y muestra una advertencia.

**Cuarto paso: calcular el equipo GIN.**

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

Cada backend primero llama a`devices`para obtener el número de dispositivos, luego ejecuta el flujo listen→getProperties→allGather→connect→closeListen para cada conexión.`bootstrapAllGather`intercambia handles entre todos los ranks, de modo que cada rank conoce la información de conexión del par.

## Control de concurrencia e interacción con hardware

El hilo de progreso de GIN es el mecanismo central de concurrencia.

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

Aquí hay varios diseños clave:

1. **Afinidad de CPU**：`ncclOsSetAffinity`vincula el hilo de progreso a un núcleo de CPU específico, evitando la invalidación de caché causada por la migración de hilos.

2. **Retroceso de bloqueo de escritura**：`writePending`es una bandera atómica; el hilo principal, al modificar la lista enlazada de`devComms`, la activa primero, y el hilo de progreso, al verla, cede proactivamente, evitando la contención de bloqueos.

3. **Bloqueo de lectura-escritura**：`devCommRwMutex`es`shared_timed_mutex`, el hilo de progreso mantiene el bloqueo de lectura para recorrer la lista enlazada, y el hilo principal mantiene el bloqueo de escritura para modificarla.

4. **División de trabajo entre hilos**: el hilo t se encarga de las conexiones t, t+proxyNthreads, t+2*proxyNthreads, ..., logrando el equilibrio de carga mediante un bucle con stride.

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

Esta implementación del bloqueo de escritura asume que solo hay un escritor (el hilo principal), por lo que no necesita exclusión mutua adicional.`writePending`primero activa la bandera y luego adquiere el bloqueo, asegurando que el hilo de progreso pueda ver la intención de escritura antes de adquirir el bloqueo y ceda proactivamente.

## Guía de trampas en producción

**Trampa 1: el desajuste en el número de conexiones GIN provoca un interbloqueo en AllGather.**El`ginCommCount`de cada rank puede ser diferente (dependiendo del número de tarjetas de red locales), NCCL toma el mínimo entre todos los ranks mediante`bootstrapAllGather`.

[FACT:src/gin/gin_host.cc:176-180]

```cpp
ginCommCountHandles[comm->rank] = backend->ginCommCount;
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
for (int r = 0; r nRanks; r++) {
  backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
}
```

Si el número de tarjetas de red de un rank es menor que el de otros ranks, todos los ranks se reducen al mínimo. Esto garantiza la simetría de las conexiones, pero desperdicia recursos de tarjetas de red.

**Trampa 2: proxyNthreads supera ginCommCount y provoca que los hilos giren en vacío.**Si el usuario configura`NCCL_GIN_PROXY_NTHREADS`mayor que`ginCommCount`, los hilos sobrantes girarán en vacío en el bucle stride.

[FACT:src/gin/gin_host.cc:181-183]

```cpp
// After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
// on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
// Extra threads simply idle in the stride loop; no correctness issue.
```

Esto no es un problema de corrección, pero desperdicia recursos de CPU. El método de diagnóstico es ver si`NCCL_GIN_PROXY_NTHREADS`es mayor que el número real de tarjetas de red.

**Trampa 3: condición de carrera al liberar DevComm.** `ncclGinDevCommFree`Primero se extrae DevComm de la lista enlazada y luego se destruye el context.

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

Tras la extracción, el hilo de progreso ya no puede ver este DevComm, por lo que destruir el context es seguro. Pero si durante la destrucción hay operaciones de red in-flight, puede provocar comportamiento indefinido; esto es lo que hay que garantizar al usar GIN: antes de liberar DevComm se debe asegurar que todas las operaciones hayan finalizado.

---

# Tres, kernel de memoria simétrica: de «registrar búfer» a «espacio de direcciones unificado»

## Modelo intuitivo

El búfer de NCCL tradicional es de «registro»: cada rank registra su propio búfer y, durante la comunicación, intercambia direcciones mediante un handle. La memoria simétrica, en cambio, es un «espacio de direcciones unificado»: todos los ranks acuerdan el mismo conjunto de direcciones virtuales; la dirección A del rank 0 y la dirección A del rank 1 apuntan a su propia memoria física, pero en el código basta con usar la misma dirección para acceder.

Esto es como si todos acordaran que «fila 3, asiento 5» señala la misma posición en la casa de cada uno; al buscar algo no hace falta preguntar primero «¿dónde está tu fila 3, asiento 5?».

Sin memoria simétrica, cada kernel tendría que resolver primero la dirección del par, lo que aumenta el costo de instrucciones y la presión sobre los registros.

## Estructuras de datos y diseño de memoria

El núcleo del kernel de memoria simétrica es el kernel mask: un bitmap que marca qué kernels están disponibles en el dominio de comunicación actual.

[FACT:src/sym_kernels.cc:17-63]

```cpp
constexpr uint32_t kernelMask_STMC =
  1  **[Design Inference & Architectural Trade-offs]**
> La ventaja de este diseño de bitmap es que permite filtrar rápidamente los kernels disponibles mediante operaciones de bits. Por ejemplo,`kmask &= ~kernelMask_STMC`con una sola línea se pueden deshabilitar todos los kernels STMC, sin necesidad de recorrer la lista.

## Step-by-Step Walkthrough: un cálculo de kernel mask

Tomemos un escenario: el rank 0 debe ejecutar AllReduce, el tipo de dato es float16, el tamaño del mensaje es 1MB, el dominio de comunicación tiene 8 ranks y todos están interconectados por NVLink.

**Primer paso: obtener el mask base correspondiente a la operación.**

[FACT:src/sym_kernels.cc:304-306]

```cpp
uint32_t kmask = kernelMask_coll(coll);
```

`kernelMask_coll(ncclFuncAllReduce)`devuelve`kernelMask_AR`, que contiene 5 kernels de AllReduce.

**Segundo paso: comprobar la disponibilidad de STMC y LDMC.**

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

`hasLsaMultimem`se calcula en`ncclSymkInitOnce`, y requiere que el multicast simétrico NVLS esté disponible y que el equipo LSA tenga más de 2 ranks. float16 admite LDMC, así que si`hasLsaMultimem`es verdadero, se conserva el kernel LDMC.

**Tercer paso: comprobar el límite de tamaño de mensaje.**

[FACT:src/sym_kernels.cc:336-342]

```cpp
size_t nBytes = alignUp(nElts * ncclTypeSize(ty), NCCL_SYM_KERNEL_CELL_SIZE);
size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks) * nBytes;
if (nBusBytes >= (size_t(2) = 32 * (size_t(2) nRanks;
kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
```

Si el equipo LSA cubre todos los ranks, no se necesita GIN; de lo contrario, solo se conservan los kernels GIN.

## Control de concurrencia e interacción con el hardware

La inicialización del kernel de memoria simétrica implica la creación de DevComm y la asignación de recursos.

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

La clave aquí es`ncclDevrCommCreateInternal`, que crea un DevComm interno que contiene recursos como multicast LSA, inbox/outbox de GIN, señales, etc.`reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL`especifica que GIN use el modo de conexión rail.

[FACT:src/sym_kernels.cc:257-261]

```cpp
symk->kcomm.workStarted = comm->profiler.symWorkStarted;
symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
symk->kcomm.workPhases = comm->profiler.symWorkPhases;
```

El kernel de memoria simétrica usa un búfer de profiler independiente para evitar entrelazarse con el workCounter de los kernels normales.

## Guía para evitar trampas en producción

**Trampa 1: requisitos de SMEM del kernel TMA.**TMA requiere aproximadamente 8KB de SMEM scratch por warp; con 16 warps son 128KB.

[FACT:src/sym_kernels.cc:135-142]

```cpp
bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  if (comm->maxSharedMemOptin minCompCap >= 100 && ncclParamSymTmaEnable();
}
```

Si la capacidad de SMEM de la GPU es insuficiente (por ejemplo, en instancias MIG), el kernel TMA se deshabilitará. El método de diagnóstico es ver si`maxSharedMemOptin`es menor que`ncclTmaShmemScratchWarpSize() * 16`。

**Trampa 2: límites del chunk size de GIN.**El chunk size del kernel ReduceScatter GIN tiene límites superior e inferior.

[FACT:src/sym_kernels.cc:148-153]

```cpp
static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128  0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}
```

Si el usuario configura`NCCL_SYM_RS_GIN_CHUNK_SIZE`por encima de 1GB, se truncará a 1GB; si es menor que 128 bytes, se elevará a 128 bytes. El valor final también se redondeará hacia abajo a una potencia de 2.

**Trampa 3: Tipo de registro de memoria simétrica no coincidente.** `ncclGetSymRegType`Según los de sendWin y recvWin`NCCL_WIN_COLL_SYMMETRIC`indicadores para determinar el tipo de registro.

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

Si los tipos de registro de send y recv no coinciden, el kernel necesita seguir rutas de código diferentes. Esto afecta el rendimiento, pero no provoca errores.

---

# IV. Abstracción de Team y DevComm versionado: infraestructura para la evolución

## Modelo intuitivo

La abstracción de Team es como "agrupar": el team mundial es toda la clase, el team LSA son los compañeros de pupitre, el team Rail son los asientos de la misma columna. Diferentes modos de comunicación requieren diferentes perspectivas de agrupación.

El DevComm versionado es como un "traductor": diferentes versiones del código de dispositivo hablan diferentes "dialectos", y la capa de compatibilidad de DevComm se encarga de traducir, permitiendo que el código nuevo y viejo se entiendan mutuamente.

Sin la abstracción de Team, cada kernel tendría que calcular su propio mapeo de ranks; sin el DevComm versionado, cualquier cambio de ABI provocaría la recompilación de todo el código de dispositivo.

## Estructuras de datos y diseño de memoria

Team es una tupla simple de tres elementos:`nRanks`、`rank`、`stride`。

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

El stride del team mundial es 1, porque todos los ranks están dispuestos de forma contigua.

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

El stride del team Rail es`lsaSize`, porque los ranks en cada rail están separados por el tamaño de un team LSA.

El núcleo del DevComm versionado es la estructura`ncclDevCommCompat`.

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

Esta estructura define las reglas de compatibilidad de la versión 2.31.0.`minVersion`y`maxVersion`definen el rango de versiones aplicable, y los cuatro punteros a función siguientes definen la lógica de filtrado de propiedades y conversión de estructuras. Si todos son nullptr, indica que esta versión no tiene requisitos de compatibilidad especiales.

## Step-by-Step Walkthrough: una conversión de Team

Tomemos un escenario: el rank 5 en un dominio de comunicación de 8 ranks, con un tamaño de team LSA de 4. Hay que calcular el rank del rank 5 en el team Rail.

**Primer paso: inicializar el estado de DevR.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
```

`ncclDevrInitOnce`Calcula el team LSA, el team CFT y otra información derivada. Si falla, devuelve un team vacío.

**Segundo paso: calcular los parámetros del team Rail.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
ncclTeam_t ans;
ans.nRanks = comm->nRanks / comm->devrState.lsaSize;  // 8 / 4 = 2
ans.rank = comm->rank / comm->devrState.lsaSize;       // 5 / 4 = 1
ans.stride = comm->devrState.lsaSize;                  // 4
```

El rank del rank 5 en el team Rail es 1, el team tiene 2 ranks y el stride es 4.

**Tercer paso: convertir de vuelta al rank mundial.**

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Si se quiere convertir el Rail rank 0 a rank mundial:`5 + (0 - 1) * 4 = 1`. Verificación: el rank 1 y el rank 5 están en el mismo rail (separados por 4).

## Control de concurrencia e interacción con el hardware

La abstracción de Team en sí misma no tiene estado y no necesita control de concurrencia. Pero`ncclDevrInitOnce`es de carga diferida, y en la primera llamada calcula toda la información derivada.

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

El comentario dice "Ignoring errors since if it fails ncclDevrInitOnce will try again" — si la inicialización falla, devuelve un team vacío y la siguiente llamada reintentará.

## Guía de trampas en producción

**Trampa 1: la suposición de stride en la conversión de Team.** `ncclTeamRankToWorld`asume que los ranks dentro del team forman una progresión aritmética.

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Si el team no es una progresión aritmética (por ejemplo, una agrupación arbitraria personalizada), esta función calculará mal. NCCL actualmente solo admite teams regulares.

**Trampa 2: punteros nulos en el DevComm versionado.** `ncclDevCommCompat_v23100`Todos los punteros a función de son nullptr, lo que indica que no hay lógica de compatibilidad especial. Si en versiones futuras se necesita conversión, hay que implementar estas funciones; de lo contrario, el código nuevo y viejo no podrá interoperar.

**Trampa 3: el modo jerárquico del team CFT.** `ncclTeamCft`admite tres modos: FLAT, HIER_MULTIMEM, HIER_LSA.

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

Si se pasa un modo inválido, devuelve un team vacío. Al usar el team CFT hay que asegurarse de que el modo sea correcto.

---

# Reflexiones de diseño

**¿Por qué NCCL debe soportar simultáneamente las tres rutas de evolución: RMA, GIN y memoria simétrica?**

> **[Design Inference & Architectural Trade-offs]**
> Estas tres rutas resuelven problemas de diferentes niveles:

- **RMA**Resuelve el problema de "modo de comunicación fijo" — permite que las capas superiores compongan primitivas para implementar cualquier modo de comunicación.
- **GIN**Resuelve el problema de "alta latencia de red" — permite que la GPU controle directamente la tarjeta de red, evitando el host proxy.
- **Memoria simétrica**Resuelve el problema del "coste de resolución de direcciones" — permite que el kernel acceda directamente a la memoria del par usando direcciones unificadas.

No son relaciones de sustitución, sino de complementariedad. RMA puede usar GIN como transporte subyacente, y GIN depende de la memoria simétrica para proporcionar coherencia de direcciones. Los tres juntos constituyen la infraestructura del "motor de comunicación programable".

**¿Cuál es la filosofía de diseño del DevComm versionado?**

> **[Design Inference & Architectural Trade-offs]**
> La idea central del DevComm versionado es "ABI estable, API en evolución". El código de dispositivo (kernel) se compila y se incrusta en el binario, y no puede recompilarse con cada actualización de la biblioteca NCCL. Por eso NCCL debe garantizar que el código de dispositivo antiguo pueda ejecutarse sobre la nueva biblioteca.`ncclDevCommCompat`La estructura es la entrada de la capa de compatibilidad: la nueva biblioteca selecciona las reglas de compatibilidad adecuadas según la versión del código de dispositivo y, si es necesario, realiza conversiones de estructuras.

---

# Resumen del capítulo

En este capítulo, partiendo de las huellas de evolución en el código fuente, hemos analizado las tres fuerzas que llevan a NCCL desde ser una biblioteca de comunicación colectiva hacia un motor de comunicación programable:

1. **RMA**（`src/rma/rma.cc`): Mediante la combinación de las primitivas Put/Signal/WaitSignal, permite que las capas superiores implementen cualquier patrón de comunicación. El diseño central divide las tareas en dos rutas paralelas, CE y Proxy, según la alcanzabilidad LSA.

2. **GIN**（`src/gin/gin_host.cc`): Mediante el envío directo desde la GPU a la red, evitando el proxy del host. El diseño central es la gestión multi-backend, la tabla de compatibilidad de versiones y el grupo de hilos de progreso.

3. **kernel de memoria simétrica**（`src/sym_kernels.cc`): Mediante un espacio de direcciones unificado, se elimina la sobrecarga de resolución de direcciones. El diseño central es el mapa de bits de máscara del kernel y la aceleración por hardware TMA/GIN.

4. **Abstracción Team y DevComm versionado**（`src/nccl_device/core.cc`、`src/devcomm/devcomm_v23100.cc`): Proporciona infraestructura para la evolución. Team ofrece una perspectiva de agrupación, y el DevComm versionado proporciona compatibilidad ABI.

El impacto de estos cambios en los frameworks de capas superiores es profundo: el ProcessGroup de PyTorch puede invocar directamente las primitivas RMA para implementar patrones de comunicación personalizados; el paralelismo de expertos de Megatron puede aprovechar GIN para reducir la latencia de all-to-all; la memoria simétrica simplifica el código del kernel.

# Reflexiones y autoevaluación de este capítulo

Q1: Si se elimina la`scheduleRmaTasksToPlan`verificación de alcanzabilidad LSA de la rama WaitSignal en , y todos los peers toman la ruta Proxy, ¿cuáles serían las consecuencias? ¿En qué escenarios se desencadenaría un desastre de rendimiento?

**Análisis de referencia**：

La verificación de alcanzabilidad LSA está en[FACT:src/rma/rma.cc:187-204], y divide los peers en dos grupos: CE y Proxy. Si se elimina esta verificación, todos los peers toman la ruta Proxy,`nRmaTasksCe`siempre es 0.

Las consecuencias son: la ruta CE no se utiliza en absoluto, y todos los WaitSignal sondean la red a través del hilo proxy del host. Para los peers dentro del alcance LSA (interconectados por NVLink en la misma máquina), que originalmente podían esperar de forma asíncrona mediante el motor de copia de la GPU, ahora pasan a sondeo por hilo del host, y la latencia sube de nivel de microsegundos a nivel de milisegundos.

Escenario de desastre de rendimiento: en el entrenamiento MoE, cada token debe esperar las señales de múltiples expertos. Si todas las señales pasan por Proxy, el hilo del host se convierte en cuello de botella, y la GPU pasa gran parte del tiempo esperando el sondeo del host. En una máquina de 8 GPU totalmente NVLink, esta degradación es especialmente evidente: toda la comunicación que originalmente podía ir por CE ahora se concentra en el host.

Método de diagnóstico: revisar los`scheduleRmaTasksToPlan`logs INFO de , si`nRmaTasksCe`siempre es 0 mientras que`nRmaTasksProxy`es muy grande, indica que hay un problema en la verificación LSA.

Q2：`ncclGinProgress`En`writePending`la combinación del flag y el`devCommRwMutex`lock de lectura/escritura, si se elimina la`writePending`verificación y solo se conserva el lock de lectura/escritura, ¿qué problemas habría?

**Análisis de referencia**：

`writePending`La verificación está en[FACT:src/gin/gin_host.cc:63-66], y hace que el hilo de progreso ceda activamente cuando el hilo principal quiere escribir. Si se elimina esta verificación, el hilo de progreso intentará directamente adquirir el lock de lectura.

El problema radica en que:`std::shared_timed_mutex`el lock de lectura de es compartido, y múltiples hilos de progreso pueden mantenerlo simultáneamente. Si el hilo principal quiere adquirir el lock de escritura, debe esperar a que se liberen todos los locks de lectura. Bajo alta carga, los hilos de progreso adquieren frecuentemente el lock de lectura, y el hilo principal puede tardar mucho en obtener el lock de escritura, provocando el bloqueo de`ncclGinDevCommSetup`o`ncclGinDevCommFree`.

Más grave aún: si el hilo principal en`ginProgressWriteLock`primero activa`writePending`y luego adquiere el lock, mientras que el hilo de progreso no verifica`writePending`, entonces el hilo de progreso podría seguir adquiriendo el lock de lectura después de que el hilo principal lo haya activado, haciendo impredecible el tiempo de espera del hilo principal.

`writePending`La función de es una «notificación suave»: decirle al hilo de progreso «voy a escribir, cedan primero». Esto es más eficiente que depender únicamente de la equidad del lock, porque el hilo de progreso puede ceder activamente en lugar de bloquearse en el lock.

Q3：`ncclSymkMask`En , si al`nBusBytes >= 32 * (size_t(2) << 30)`se deshabilitan todos los kernels (`kmask = 0`), en ese momento`ncclSymkAvailable`devuelve false, ¿a qué ruta recurre NCCL? ¿Qué impacto de rendimiento tiene esta ruta de respaldo?

**Análisis de referencia**：

`kmask = 0`En[FACT:src/sym_kernels.cc:342], en ese momento`ncclSymkAvailable`devuelve false ([FACT:src/sym_kernels.cc:354-361]）。

La ruta de respaldo es: NCCL usará los kernels de comunicación colectiva tradicionales (kernels de memoria no simétrica). Estos kernels acceden a la memoria del peer mediante buffers registrados, requieren resolver direcciones primero, y tienen mayor sobrecarga de instrucciones.

Impacto de rendimiento: para mensajes muy grandes (más de 64GB de bytes de bus), la sobrecarga de resolución de direcciones de los kernels tradicionales es una proporción muy pequeña, porque la transferencia de datos en sí domina. Pero en casos límite (justo por encima de 64GB), los kernels tradicionales pueden ser un 10-20% más lentos que los kernels de memoria simétrica.

La razón fundamental de esta limitación es: los kernels de memoria simétrica usan enteros de 32 bits para rastrear los chunks del bucle desenrollado, y cada chunk tiene al menos 32 bytes, por lo que el rango máximo direccionable es 32 * 2^31 = 64GB. Superar este rango provoca desbordamiento de enteros.

En producción real, los escenarios con una única comunicación colectiva superior a 64GB son raros (normalmente all-reduce tras acumulación de gradientes), pero no imposibles. Si se encuentra este escenario, se puede considerar comunicación fragmentada o usar kernels tradicionales.

---

# Transición al final del capítulo

En este capítulo hemos visto que NCCL está pasando de «operaciones colectivas fijas» a «motor de comunicación programable»: RMA ofrece composición de primitivas, GIN ofrece envío directo desde GPU, la memoria simétrica ofrece un espacio de direcciones unificado, y Team y el DevComm versionado ofrecen infraestructura.

Estas evoluciones no son aisladas, y apuntan conjuntamente a un objetivo:**permitir que los frameworks de nivel superior implementen modos de comunicación personalizados con menor latencia y mayor flexibilidad**. Para frameworks como PyTorch y Megatron, esto significa que pueden construir directamente sobre NCCL modos de comunicación complejos como MoE all-to-all, paralelismo de pipeline y paralelismo de expertos, sin necesidad de eludir NCCL e implementar su propia capa de red.

El siguiente capítulo es el último del libro. Recorreremos de nuevo la cadena completa de un AllReduce — desde la`ncclAllReduce`llamada, pasando por el encolado de tareas, la selección de algoritmo, el lanzamiento del kernel, el avance del proxy, la transmisión por red, hasta el retorno del resultado. Esta revisión conectará los conocimientos de los 24 capítulos anteriores para formar un mapa cognitivo completo.

Hasta aquí, hemos vislumbrado las tres líneas principales de la evolución de NCCL desde operaciones colectivas fijas hacia un motor de comunicación programable: la composición de primitivas RMA, el envío directo desde GPU a la red, el modelo de memoria simétrica, y la abstracción de team y el DevComm versionado que los sustentan. Estos mecanismos apuntan conjuntamente hacia un futuro de comunicación más flexible y más cercano a las capacidades del hardware. Sin embargo, independientemente de cómo evolucione la arquitectura, la cadena completa de un AllReduce sigue siendo la piedra angular para entender NCCL. En el próximo capítulo no introduciremos código nuevo, sino que repasaremos de principio a fin el flujo extremo a extremo desde el capítulo 3 hasta el capítulo 10 — desde la llamada a ncclAllReduce, pasando por el establecimiento del dominio de comunicación, la búsqueda de topología, la selección de algoritmo, el encolado de tareas, el lanzamiento del kernel, la ejecución de primitivas en el lado del dispositivo, hasta la escritura de resultados. Reensamblarás los mecanismos dispersos en cada capítulo en un modelo mental completo y obtendrás un índice de «qué capítulo consultar cuando encuentres un problema».
