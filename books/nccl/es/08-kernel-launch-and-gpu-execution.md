# Capítulo 8: Lanzamiento de kernel y ejecución en el dispositivo: de la llamada en el host al arranque de los bloques de hilos en la GPU

En el capítulo anterior desglosamos cómo se divide la tarea entre múltiples channels, cómo se generan los parámetros de lanzamiento del kernel y el mecanismo de envío por lotes y ordenación de dependencias bajo la semántica de group. Ahora, el plan de lanzamiento está listo, pero sigue siendo solo una estructura de datos en el lado del host. La pregunta central que responde este capítulo es:`ncclKernelPlan`¿cómo se convierte en un grid que realmente se ejecuta en la GPU? Seguiremos la cadena de llamadas de`ncclLaunchKernel`para ver cómo se insertan los parámetros en los kernel args, cómo se selecciona la variante de kernel,`cuLaunchKernelEx`cómo se invoca, y cómo en el lado del dispositivo`ncclKernelMain`lee la descripción del trabajo desde la memoria compartida y la distribuye a la implementación concreta.

# Del Plan al Grid: panorama completo de la ruta de lanzamiento

Antes de entrar en detalles, establezcamos un modelo mental global. Imaginemos`ncclKernelPlan`como un "plano de construcción": registra cuántos channels se van a lanzar (cuántos blocks), cuántos hilos por block, qué work se va a ejecutar y qué función de kernel se usará. Y`ncclLaunchKernel`es la acción de "entrada del equipo de construcción": traduce la información del plano a lo que el driver de CUDA puede entender,`CUlaunchConfig`, y luego llama a`cuLaunchKernelEx`para lanzar realmente el grid a la GPU.

Sin esta capa, toda la programación del lado del host (la división por channel del capítulo anterior, la organización de batches, la ordenación de proxy ops) sería pura teoría: no se ejecutaría ningún kernel en la GPU y la comunicación nunca ocurriría. Este es el último eslabón del tronco de extremo a extremo y también la frontera entre host y device.

Toda la ruta de lanzamiento se puede resumir en tres fases:

1. **Preparación de parámetros**（`finishPlan` + `uploadWork`): organizar las estructuras work, los descriptores de batch y los kernel args en un bloque de memoria contigua, y decidir si se colocan en los parámetros del kernel, en la FIFO o en un búfer persistente.

2. **Lanzamiento del kernel**（`ncclLaunchKernel`): calcular las dimensiones de grid/block, ensamblar los launch attributes (CGA cluster, mem sync domain, launch completion event), llamar a`cuLaunchKernelEx`。

3. **Entrada en el lado del dispositivo**（`ncclKernelMain`): cada block determina su channelId según`blockIdx.x`, carga el work batch desde los args o la FIFO a la memoria compartida, y luego, mediante`ncclDevFuncTable`, lo distribuye a la implementación concreta de algoritmo/protocolo.

La siguiente figura muestra el flujo de control completo desde el plan hasta el grid, incluyendo las bifurcaciones clave:

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

Esta figura ancla las tres funciones centrales de este capítulo:`finishPlan`、`uploadWork`、`ncclLaunchKernel`. A continuación las desglosaremos una por una.

# Preparación de parámetros: cómo encuentra su lugar la estructura work

## Modelo intuitivo

`finishPlan`desempeña un papel similar al de un "empacador" en un centro de clasificación de paquetes. Se enfrenta a un montón de estructuras work dispersas (una por cada operación collective o p2p) y debe decidir: ¿estos work se meten en la "mochila de mano" que son los parámetros del kernel, o se colocan en la "cinta transportadora" que es la FIFO, o se guardan en el "almacén" que es el búfer persistente?

Si esta decisión se toma mal —por ejemplo, si un work es demasiado grande para caber en los parámetros del kernel pero se fuerza a meterlo— el lanzamiento del kernel fallará directamente. Si el work se coloca en el lugar equivocado, el lado del dispositivo leerá datos basura y el resultado de la comunicación será completamente erróneo.

## Estructuras de datos y diseño de memoria

Primero veamos`ncclDevKernelArgs`la estructura de , que es el "sobre" entre host y device:

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

Esta estructura solo tiene 5 campos, pero cada uno transporta información clave.`channelMask`es una máscara de 64 bits, cada bit corresponde a un channel, y el lado del dispositivo, mediante`__popcll`, calcula`blockIdx.x`el channelId correspondiente.`workStorageType`determina desde dónde lee el work el lado del dispositivo:`Args`indica que el work está en los parámetros del kernel,`Fifo`indica que está en el búfer circular,`Persistent`indica que está en el búfer persistente.

`ncclDevWorkBatch`es el descriptor de batch, que le dice al lado del dispositivo "dónde está el work de este channel y cuántos hay":

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

`offsetBitset`es una máscara de 64 bits, cada bit corresponde a una estructura work. El lado del dispositivo mediante`__popc`y`fns`(instrucción find n-th set) localiza el offset de cada work.`nextJump`y`nextExtends`se utilizan para encadenar múltiples batches: cuando hay demasiados work para caber en un batch, se crean "batches extendidos".

## Step-by-Step Walkthrough

Ahora introduzcamos un escenario concreto: un AllReduce se divide en 4 channels, cada channel tiene 2 estructuras work, en total 8 work.

**Primer paso:`finishPlan`determina el tipo de almacenamiento.**

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

La decisión clave aquí es: si`sizeof(ncclDevKernelArgs) + batchBytes + workBytes`cabe en`comm->workArgsBytes`(normalmente 4KB), se coloca el work directamente en los parámetros del kernel. De lo contrario, el work se coloca en el FIFO o en un búfer persistente, y en los parámetros del kernel solo se coloca el descriptor de batch.

> **[Design Inference & Architectural Trade-offs]**
> ¿Por qué se prioriza colocarlo en los parámetros del kernel? Porque los parámetros del kernel se pasan a través de memoria constante (constant memory) en el driver de CUDA, y cuando el lado del dispositivo los lee utiliza la instrucción`ld.param`, que es mucho más rápida que leer el FIFO desde memoria global. Para mensajes pequeños (poca cantidad total de work), esto reduce significativamente la latencia.

**Segundo paso: colocar los batches por channel de forma alterna en los kernel args.**

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

copiar

1. **`fifoCursor`Aquí hay varios puntos clave:**La semántica de`Args`: para el tipo`kernelArgs`, es un offset relativo a la dirección inicial de`Fifo`; para el tipo`Persistent`, es un offset relativo a la dirección base del FIFO; para el tipo

2. **`offsetBase`, comienza desde 0.**：`finishPlan`La corrección de`offsetBase`: el`uploadWork`del batch en`Args`es relativo a la posición inicial del work del plan (comenzando desde 0).`sizeof(ncclDevKernelArgs) + batchBytes`necesita convertirlo a un offset relativo a la ubicación de almacenamiento real. Para el tipo`Fifo`, se suma`comm->workFifoProduced`。

3. **; para el tipo**, se suma`alignas(16)`Copia alineada a 16 bytes`COMPILER_ASSUME_ALIGNED`: las estructuras work están alineadas a 16 bytes (

4. **), por lo que la copia se realiza en unidades de 16 bytes.**le indica al compilador que esta dirección está alineada a 16 bytes, haciendo que el compilador genere instrucciones vectorizadas más eficientes.`Fifo`Espera del FIFO`waitWorkFifoAvailable`: para el tipo`comm->abortFlag`,

## espera en spin hasta que el FIFO tenga suficiente espacio. Esta espera verifica

> **[Design Inference & Architectural Trade-offs]**
> **Reflexiones de diseño y trampas en producción**〔Inferencia de diseño y compensaciones arquitectónicas〕

- `Args`¿Por qué debe haber tres tipos de almacenamiento?
- `Fifo`Esta es una compensación entre espacio y latencia:
- `Persistent`: el más rápido (memoria constante), pero de capacidad limitada (4KB). Adecuado para mensajes pequeños y poco work.`cudaMemcpy`: gran capacidad (búfer circular), pero la lectura en el lado del dispositivo pasa por memoria global. Adecuado para mensajes medianos.

**: se utiliza en escenarios de captura de CUDA Graph. Como durante la captura de graph no se puede hacer**, es necesario preasignar un búfer persistente, copiar el work allí y luego hacer que el kernel lea desde ahí.`waitWorkFifoAvailable`Trampa 1: desbordamiento del FIFO que provoca deadlock.`abortFlag`Si[FACT:src/enqueue/enqueue.cc:1333-1349]no verifica

```c
if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
  return ncclInternalError;
}
```

**verifica explícitamente el abort flag:`offsetBitset`copiar** `offsetBitset`Trampa 2: desbordamiento de`1ull << (offset / workSize)`.`NCCL_MAX_DEV_WORK_BATCH_BYTES`es de 64 bits, soporta como máximo 64 work en un batch. Si se superan los 64,`ncclDevWorkColl`se desbordará. En el código fuente, mediante

**se limita el tamaño del batch (1024 bytes), y la estructura work más pequeña es**(aproximadamente 80 bytes), por lo que como máximo hay 12 work, sin desbordamiento.`uploadWork`Trampa 3: fuga de memoria en modo Persistent.`Persistent`En la rama`fifoBufHost`de`ncclOsAlignedAlloc`,`uploadWork_cleanup_fn`se asigna mediante`cudaMemcpyAsync`y debe liberarse en`fail`. Si`cleanup`falla, la etiqueta`fifoBufHost`verificará si[FACT:src/enqueue/enqueue.cc:1483-1485]es null, y si es null liberará directamente

# . Esta cadena de recuperación de errores se puede ver en

## Lanzamiento del kernel: de CUlaunchConfig a cuLaunchKernelEx

`ncclLaunchKernel`El rol de es similar a una "consola de control de lanzamiento de cohetes". Recibe un plan que ya tiene el combustible cargado (datos de work), calcula los parámetros de vuelo del cohete (dimensiones de grid/block), configura diversas opciones de lanzamiento (cluster, mem sync domain, completion event) y luego presiona el botón de lanzamiento (`cuLaunchKernelEx`）。

Si este paso falla —por ejemplo, si las dimensiones del grid se calculan mal— se lanzará un número incorrecto de blocks en la GPU, lo que provocará que el trabajo de algunos channels nunca se ejecute y la comunicación quede colgada.

## Estructuras de datos y diseño de memoria

`CUlaunchConfig`Es la estructura de configuración de lanzamiento de la API del driver de CUDA, y NCCL la construye en la pila:

[FACT:src/enqueue/enqueue.cc:1916-1917]

```c
CUlaunchConfig launchConfig = {0};
CUlaunchAttribute launchAttrs[6] = {};
int attrs = 0;
```

`launchAttrs`Es un array de como máximo 6 elementos, cada uno de los cuales es un`CUlaunchAttribute`. NCCL añade condicionalmente distintos atributos según las capacidades del hardware y la versión del driver:

- `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`: dimensión del CGA cluster (sm90+)
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`: política de programación del cluster
- `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`: dominio de sincronización de memoria (CUDA 12.0+)
- `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`: evento de finalización de lanzamiento (CUDA 12.3+)
- `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`: serialización de flujo programática (sym kernel)
- `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING`: programación centrada en la utilización de NVLink (CUDA 13.0+)

## Step-by-Step Walkthrough

**Primer paso: calcular las dimensiones de grid y block.**

[FACT:src/enqueue/enqueue.cc:1889-1893]

```c
int nChannels = countOneBits(plan->channelMask);
void* sym = plan->kernelFn;
dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
```

`nChannels`Es el número de bits establecidos en`channelMask`, es decir, cuántos blocks debe lanzar este plan. Cada block se encarga de un channel.`threadPerBlock`Se calcula en`scheduleCollTasksToPlan`mediante`plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE)`, tomando el mayor de todos los tasks`nWarps * 32`。

`smem`Es el tamaño de memoria compartida dinámica. Para un kernel normal, es`ncclShmemDynamicSize(comm->cudaArch)`, que es una constante en tiempo de compilación que depende de la arquitectura (sm70+ es`ncclShmemScratchWarpSize * (NCCL_MAX_NTHREADS / WARP_SIZE)`). Para un sym kernel, es`plan->kernelDynSmem`, porque los requisitos de memoria compartida del sym kernel pueden ser distintos.

**Segundo paso: ensamblar los parámetros del kernel.**

[FACT:src/enqueue/enqueue.cc:1902-1903]

```c
void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
                 CU_LAUNCH_PARAM_END};
```

Esta es una forma de pasar parámetros de la API del driver de CUDA:`CU_LAUNCH_PARAM_BUFFER_POINTER`le indica al driver que "los parámetros no se pasan uno a uno, sino como un bloque de memoria contiguo",`CU_LAUNCH_PARAM_BUFFER_SIZE`le indica al driver el tamaño de ese bloque. La ventaja de esto es que NCCL puede pasar`ncclDevKernelArgs`y el array batch posterior de una sola vez, sin necesidad de empaquetar los parámetros uno por uno.

**Tercer paso: añadir launch attributes.**

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

CGA (Cooperative Group Array) es una característica de hardware introducida en sm90 que permite agrupar varios blocks en un cluster; los blocks dentro del cluster pueden garantizar su programación simultánea en un conjunto de SMs y pueden acceder a la memoria compartida entre sí. NCCL utiliza esta característica para implementar algoritmos como NVLS que requieren sincronización entre blocks.

Nótese la protección`if (grid.x % clusterSize) clusterSize = 1;`: la dimensión del cluster debe dividir exactamente la dimensión del grid, de lo contrario el driver dará error. Si`grid.x`no es divisible por`clusterSize`, se degrada a no usar cluster.

**Cuarto paso: añadir launch completion event.**

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

`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`Es una característica introducida en CUDA 12.3: el driver registra un evento cuando el kernel realmente comienza a ejecutarse (no cuando la llamada del lado del host retorna). Esto es crucial para implementar el "orden implícito" (implicit order): NCCL necesita garantizar que varios kernels se ejecuten en orden, pero sin que el lado del host bloquee la espera.

`getImplicitOrder`La lógica de es: si el usuario ha configurado`launchOrderImplicit`, y la versión del driver es lo suficientemente reciente, se usa`ncclImplicitOrderLaunch`(ordenar con launch event); de lo contrario, se usa`ncclImplicitOrderSerial`(ordenar con completion event, es decir, ejecución en serie).

**Quinto paso: llamar a`cuLaunchKernelEx`。**

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

`cuLaunchKernelEx`Es una nueva API introducida en CUDA 12.0 que admite launch attributes. Para drivers antiguos (< 11.8), NCCL recurre a`cuLaunchKernel`：

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

## Control de concurrencia e interacción con el hardware

**Mecanismo de relay del Launch completion event.**Cuando se usa`ncclImplicitOrderLaunch`y el usuario proporciona`launchCompletionEvent`, NCCL no puede pasar directamente el event del usuario al driver, porque el driver solo admite un launch completion event. La estrategia de NCCL es:

1. Pasar`comm->sharedRes->launchEvent`al driver.

2. En`relayStream`, esperar a`launchEvent`。

3. En`relayStream`, registrar el event del usuario.

De esta forma, el event del usuario se disparará después de que el kernel realmente comience a ejecutarse, y no cuando la llamada del lado del host retorne.

**Mem Sync Domain。** [FACT:src/enqueue/enqueue.cc:1938-1942]En sm90+, NCCL configura`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`como`cudaLaunchMemSyncDomainRemote`. Este es el mecanismo de dominios de sincronización de memoria introducido por la arquitectura Hopper, que sirve para aislar las barreras de memoria de distintos kernels y reducir la sobrecarga de sincronización innecesaria.

## Guía de trampas en producción

**Trampa 1: la dimensión del cluster no divide exactamente y provoca un fallo de lanzamiento.**Si`grid.x`no es divisible por`clusterSize`, el driver devolverá`CUDA_ERROR_INVALID_VALUE`. En el código fuente se protege mediante`if (grid.x % clusterSize) clusterSize = 1;`, pero esto también implica que la característica de cluster queda deshabilitada silenciosamente. Si el usuario espera la mejora de rendimiento que aporta el cluster, debe comprobar`cgaClusterSize`y`nChannels`la relación entre ellos.

**Trampa 2: la versión del driver no cumple los requisitos y el kernel no está disponible.** `ncclInitKernelsForDevice`verifica los requisitos de controlador de cada kernel durante la inicialización:

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

El núcleo de este código es calcular`fnsOfBitset`: para`offsetBitset`el n-ésimo bit activado, cuál es su índice de bit. PTX tiene la instrucción`fns`para hacer esto, pero se expande en muchas instrucciones SASS. La estrategia de NCCL es usar memoria compartida: cada lane verifica si su bit está activado, si lo está, calcula cuántos bits activados hay antes de él, y luego escribe su número de lane en`fnsOfBitset[nWorksBelow]`。

A continuación viene la copia real:

[FACT:src/device/common.h:209-241]

```c
if (tid = %d && __CUDA_ARCH__ >= %d\n" % (cudart ,arch))
  out("/*%4d*/ %s,\n" % (index, sym))
  if (cudart, arch) != (0, 0):
    out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
  index += 1
out("nullptr};\n")
```

## Reflexiones de diseño y problemas en producción

**Por qué usar`__grid_constant__`？** [FACT:src/device/common.h:19-24]

```c
#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif
```

`__grid_constant__`indica al compilador que este parámetro es de solo lectura y puede colocarse en memoria constante. Así, cuando el lado del dispositivo lee, usa la instrucción`ld.param`, que es más rápida que leer desde memoria global. El comentario menciona que rompe cuda-gdb, por lo que solo se habilita en sm70+.

**Punto problemático 1:`workStorage`desbordamiento.** `workStorage`El tamaño de`ncclMaxDevWorkBatchBytes()`es`nWorks * workSize`, en sm90+ es 16KB. Si`NCCL_MAX_DEV_WORK_BATCH_BYTES`supera este valor, se escribirá fuera de los límites. En el código fuente, mediante

**se limita el tamaño del batch en el lado host, pero no hay verificación adicional en el lado del dispositivo. Si se elude la restricción del lado host (por ejemplo, modificando variables de entorno), se producirá un desbordamiento de memoria compartida.`__syncthreads()`Punto problemático 2:**La ausencia de`loadWorkBatchToShmem`provoca condiciones de carrera.`__syncthreads()`Después de`workStorage`, debe haber un[FACT:src/device/common.h:479]para que todos los hilos vean el`__syncthreads(); // publish ncclShmem`completo. En el código fuente, en`workStorage`hay

**. Si se elimina esta sincronización, algunos hilos podrían comenzar a leer antes de que** `while (ncclShmem.aborted == 0)`termine de escribirse, leyendo datos basura.

# Punto problemático 3: el momento de la verificación de abort.

## Solo verifica abort al comienzo de cada batch. Si un batch tarda mucho en ejecutarse, la señal de abort podría tardar mucho en surtir efecto. Esta es una compensación de diseño: verificaciones más frecuentes aumentan la sobrecarga, pero responden más rápido.

`generate.py`Selección de variantes de kernel: cómo generate.py genera la lista de kernels

Modelo intuitivo`generate.py`El rol de

## es similar al de un "planificador de líneas de producción de una fábrica de automóviles". Se enfrenta a un enorme espacio combinatorio (7 tipos de operaciones de conjunto × 5 tipos de operaciones de reducción × 12 tipos de datos × 7 algoritmos × 3 protocolos) y necesita decidir: ¿qué combinaciones requieren generar un kernel dedicado? ¿Cuáles pueden compartir un kernel genérico?

`generate.py`Si se genera un kernel para cada combinación, el tiempo de compilación y el tamaño del binario explotarán. Si solo se genera un kernel genérico, en tiempo de ejecución se volverá más lento debido a llamadas a punteros de función y evaluaciones de ramas.

1. **`device_table.cu`**La solución de`ncclDevFuncTable`es el "kernel representativo": generar un kernel para cada clase de equivalencia y distribuir en tiempo de ejecución mediante una tabla de punteros de función.

2. **`host_table.cc`**Estructuras de datos y diseño de memoria`ncclDevKernelList`、`ncclDevKernelForFunc`、`ncclDevFuncRowToId`Genera tres archivos clave:

3. **: el`<coll>_<op>_<ty>.cu`**del lado del dispositivo, que mapea funcId a la función de dispositivo concreta.

## Step-by-Step Walkthrough

**: las tablas**

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

Cada`ncclDevFuncId()`: la implementación concreta del kernel.

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

`ncclDevFuncId`Copiar`ncclDevFuncRowToId`Este orden de enumeración debe coincidir con la fórmula de cálculo de`AllReduce Sum i32`:`AllReduce Sum u32`Copiar

**Lo que calcula**

[FACT:src/device/generate.py:211-225]

```python
func_rows = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}
kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))
```

`equivalent_primary`se mapea al "ID de función principal". La razón de este mapeo es que muchas filas pueden mapearse a la misma función principal (por ejemplo, todas las filas de

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

`best_kernel`).`AllGather`Segundo paso: calcular la función principal y la función kernel.`AllGather RING LL`）：

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

**Mapea enteros con signo a enteros sin signo (porque la suma/multiplicación es igual para ambos):**

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

`DEFINE_ncclDevKernel`Mapea múltiples funciones principales al mismo kernel (por ejemplo, todos los algoritmos de

[FACT:src/device/common.h:507-509]

```c
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain, algo, proto>>(&args4K.args); \
  }
```

Copiar`__global__`Tercer paso: generar la definición del kernel.`ncclKernelMain`Copiar`specializedFnId`Después de la expansión de la macro es:`RunWorkBatch<coll, ty, redop<ty>, algo, proto>`。

## Copiar

> **[Design Inference & Architectural Trade-offs]**
> **, que llama a**, con parámetros de plantilla

**y`NCCL_EXACT_KERNEL_NAMES`Reflexiones de diseño y problemas en producción**〔Inferencias de diseño y compensaciones arquitectónicas〕`best_kernel`¿Por qué usar "kernels representativos" en lugar de un kernel por combinación?

**Compensación entre tiempo de compilación y tamaño del binario. El espacio combinatorio completo es 7 × 5 × 12 × 7 × 3 ≈ 8820 kernels, cada kernel tarda unos segundos en compilarse, lo que suma varias horas. Además, el tamaño del binario alcanzaría cientos de MB. Al mapear a kernels representativos, el número real de kernels generados se reduce a unas pocas decenas.`required_cuda`Punto problemático 1:**Provoca una explosión de compilación.

[FACT:src/device/generate.py:130-154]

Si se establece esta variable de entorno,
