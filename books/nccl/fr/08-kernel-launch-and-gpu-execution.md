# Chapitre 8 : Lancement du kernel et exécution côté device : de l'appel côté host au démarrage des blocs de threads GPU

Dans le chapitre précédent, nous avons décomposé comment la tâche est répartie sur plusieurs channels, comment les paramètres de lancement du kernel sont générés, ainsi que les mécanismes de soumission par lots et d'ordonnancement des dépendances sous la sémantique de group. Maintenant, le plan de lancement est prêt, mais il ne s'agit encore que d'une structure de données côté host. La question centrale à laquelle ce chapitre répond est :`ncclKernelPlan`Comment cela devient-il une grid réellement en cours d'exécution sur le GPU ? Nous allons suivre la chaîne d'appels de`ncclLaunchKernel`pour voir comment les paramètres sont insérés dans les kernel args, comment les variantes de kernel sont sélectionnées,`cuLaunchKernelEx`comment il est appelé, et comment, côté device,`ncclKernelMain`lit la description du travail depuis la mémoire partagée et la distribue aux implémentations concrètes.

# Du Plan à la Grid : panorama du chemin de lancement

Avant d'entrer dans les détails, établissons un modèle mental global. Considérez`ncclKernelPlan`comme un « plan de construction » : il enregistre le nombre de channels à lancer (nombre de blocks), le nombre de threads par block, les work à exécuter et la fonction kernel à utiliser. Et`ncclLaunchKernel`est l'action de « l'équipe de construction qui entre en scène » — elle traduit les informations du plan en`CUlaunchConfig`compréhensibles par le pilote CUDA, puis appelle`cuLaunchKernelEx`pour réellement lancer la grid sur le GPU.

Sans cette couche, toute l'orchestration côté host (découpage en channels, organisation des batchs, ordonnancement des proxy op du chapitre précédent) ne serait que théorie : aucun kernel ne s'exécuterait sur le GPU et la communication n'aurait jamais lieu. C'est le dernier maillon du squelette de bout en bout, et aussi la frontière entre host et device.

L'ensemble du chemin de lancement peut se résumer en trois phases :

1. **Préparation des paramètres**（`finishPlan` + `uploadWork`) : organiser les structures work, les descripteurs de batch et les kernel args dans une zone de mémoire contiguë, en décidant s'ils sont placés dans les paramètres du kernel, dans la FIFO ou dans un buffer persistant.

2. **Lancement du kernel**（`ncclLaunchKernel`) : calculer les dimensions grid/block, assembler les launch attributes (CGA cluster, mem sync domain, launch completion event), appeler`cuLaunchKernelEx`。

3. **Point d'entrée côté device**（`ncclKernelMain`) : chaque block détermine son channelId en fonction de`blockIdx.x`charge le work batch depuis les args ou la FIFO vers la mémoire partagée, puis le distribue via`ncclDevFuncTable`vers l'implémentation concrète de l'algorithme/protocole.

La figure ci-dessous montre le flux de contrôle complet du plan à la grid, y compris les branchements décisifs clés :

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

Cette figure ancre les trois fonctions centrales de ce chapitre :`finishPlan`、`uploadWork`、`ncclLaunchKernel`. Nous allons maintenant les décomposer une par une.

# Préparation des paramètres : comment une structure work trouve sa place

## Modèle intuitif

`finishPlan`Le rôle de est similaire à celui d'un « emballeur » dans un centre de tri de colis. Face à un ensemble de structures work dispersées (une par opération collective ou p2p), il doit décider : ces work doivent-ils être insérés dans les paramètres du kernel, ce « sac à dos personnel », ou placés sur la FIFO, ce « tapis roulant », ou encore dans un buffer persistant, cet « entrepôt » ?

Si cette décision est mauvaise — par exemple, si un work trop volumineux est forcé dans les paramètres du kernel alors qu'il n'y rentre pas — le lancement du kernel échouera directement. Si le work est placé au mauvais endroit, le device lira des données corrompues et le résultat de la communication sera totalement erroné.

## Structures de données et disposition mémoire

Regardons d'abord`ncclDevKernelArgs`la structure de , qui est l'« enveloppe » entre host et device :

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

Cette structure ne comporte que 5 champs, mais chacun porte une information cruciale.`channelMask`est un masque de 64 bits, chaque bit correspondant à un channel ; le device calcule via`__popcll`le channelId correspondant à`blockIdx.x`détermine d'où le device lit le work :`workStorageType`signifie que le work se trouve dans les paramètres du kernel,`Args`signifie qu'il se trouve dans le buffer circulaire,`Fifo`signifie qu'il se trouve dans le buffer persistant.`Persistent` 表示在持久化缓冲区里。

`ncclDevWorkBatch`est un descripteur de batch, il indique au côté device « où se trouve le work de ce channel et combien il y en a » :

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

`offsetBitset`est un masque de 64 bits, chaque bit correspond à une structure work. Le côté device utilise`__popc`et`fns`(find n-th set) pour localiser l'offset de chaque work.`nextJump`et`nextExtends`servent à chaîner plusieurs batchs — lorsque les works sont trop nombreux pour tenir dans un seul batch, on crée des « batchs étendus ».

## Step-by-Step Walkthrough

Prenons maintenant un scénario concret : un AllReduce est découpé en 4 channels, chaque channel contient 2 structures work, soit 8 works au total.

**Première étape :`finishPlan`détermine le type de stockage.**

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

Le critère clé ici est : si`sizeof(ncclDevKernelArgs) + batchBytes + workBytes`peut tenir dans`comm->workArgsBytes`(généralement 4KB), on place directement les works dans les paramètres du kernel. Sinon, les works sont placés dans le FIFO ou dans un buffer persistant, et seuls les descripteurs de batch sont placés dans les paramètres du kernel.

> **[Design Inference & Architectural Trade-offs]**
> Pourquoi privilégier le placement dans les paramètres du kernel ? Parce que les paramètres du kernel sont transmis via la mémoire constante (constant memory) dans le driver CUDA, et le côté device les lit avec l'instruction`ld.param`, bien plus rapide qu'une lecture du FIFO depuis la mémoire globale. Pour les petits messages (faible volume de works), cela réduit significativement la latence.

**Deuxième étape : placer les batchs dans kernel args en alternant par channel.**

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

copie

1. **`fifoCursor`Il y a ici plusieurs points clés :**Sémantique de`Args`: pour le type`kernelArgs`, il s'agit d'un offset par rapport à l'adresse de début de`Fifo`; pour le type`Persistent`, il s'agit d'un offset par rapport à l'adresse de base du FIFO ; pour le type

2. **`offsetBase`, il commence à 0.**：`finishPlan`Correction de`offsetBase`: dans`uploadWork`, le`Args`du batch est relatif à la position de début des works du plan (à partir de 0).`sizeof(ncclDevKernelArgs) + batchBytes`doit le convertir en un offset relatif à l'emplacement de stockage réel. Pour le type`Fifo`, on ajoute`comm->workFifoProduced`。

3. **; pour le type**, on ajoute`alignas(16)`Copie alignée sur 16 octets`COMPILER_ASSUME_ALIGNED`: les structures work sont toutes alignées sur 16 octets (

4. **), donc la copie se fait par unités de 16 octets.**indique au compilateur que cette adresse est alignée sur 16 octets, lui permettant de générer des instructions vectorisées plus efficaces.`Fifo`Attente FIFO`waitWorkFifoAvailable`: pour le type`comm->abortFlag`,

## effectue un spin-wait jusqu'à ce que le FIFO ait suffisamment d'espace. Cette attente vérifie

> **[Design Inference & Architectural Trade-offs]**
> **Réflexions de conception et pièges en production**〔Inférence de conception et compromis architecturaux〕

- `Args`Pourquoi avoir trois types de stockage ?
- `Fifo`C'est un compromis entre espace et latence :
- `Persistent`: le plus rapide (mémoire constante), mais capacité limitée (4KB). Adapté aux petits messages et au faible nombre de works.`cudaMemcpy`: grande capacité (buffer circulaire), mais la lecture côté device passe par la mémoire globale. Adapté aux messages moyens.

**: utilisé pour les scénarios de capture CUDA Graph. Comme la capture de graph ne permet pas de faire**, il faut préallouer un buffer persistant, y copier les works, puis faire lire le kernel depuis cet emplacement.`waitWorkFifoAvailable`Piège 1 : débordement du FIFO provoquant un deadlock.`abortFlag`Si[FACT:src/enqueue/enqueue.cc:1333-1349]ne vérifie pas

```c
if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
  return ncclInternalError;
}
```

**vérifie explicitement l'abort flag :`offsetBitset`copie** `offsetBitset`Piège 2 : débordement de`1ull << (offset / workSize)`.`NCCL_MAX_DEV_WORK_BATCH_BYTES`est sur 64 bits, supportant au maximum 64 works dans un batch. Si l'on dépasse 64,`ncclDevWorkColl`déborde. Dans le code source,

**limite la taille du batch (1024 octets), et la plus petite structure work est**(environ 80 octets), donc au maximum 12 works, pas de débordement.`uploadWork`Piège 3 : fuite mémoire en mode Persistent.`Persistent`Dans la branche`fifoBufHost`de`ncclOsAlignedAlloc`,`uploadWork_cleanup_fn`est alloué via`cudaMemcpyAsync`et doit être libéré dans`fail`. Si`cleanup`échoue, le label`fifoBufHost`vérifie si[FACT:src/enqueue/enqueue.cc:1483-1485]est null, et si c'est le cas, libère directement

# . Cette chaîne de récupération d'erreur est visible dans

## Lancement de kernel : de CUlaunchConfig à cuLaunchKernelEx

`ncclLaunchKernel`Le rôle de  est similaire à une « console de contrôle de lancement de fusée ». Il reçoit un plan déjà chargé en carburant (données work), calcule les paramètres de vol de la fusée (dimensions grid/block), configure diverses options de lancement (cluster, mem sync domain, completion event), puis appuie sur le bouton de lancement (`cuLaunchKernelEx`）。

Si cette étape échoue — par exemple si la dimension grid est mal calculée — un nombre incorrect de blocks sera lancé sur le GPU, entraînant que le travail de certains channels ne sera jamais exécuté et que la communication restera bloquée.

## Structures de données et disposition mémoire

`CUlaunchConfig`est la structure de configuration de lancement de l'API CUDA driver, que NCCL construit sur la pile :

[FACT:src/enqueue/enqueue.cc:1916-1917]

```c
CUlaunchConfig launchConfig = {0};
CUlaunchAttribute launchAttrs[6] = {};
int attrs = 0;
```

`launchAttrs`est un tableau de 6 éléments maximum, chaque élément étant un`CUlaunchAttribute`. NCCL ajoute conditionnellement différentes propriétés en fonction des capacités matérielles et de la version du driver :

- `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`: dimension du CGA cluster (sm90+)
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`: politique d'ordonnancement du cluster
- `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`: domaine de synchronisation mémoire (CUDA 12.0+)
- `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`: événement de fin de lancement (CUDA 12.3+)
- `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`: sérialisation de flux programmatique (sym kernel)
- `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING`: ordonnancement centré sur l'utilisation NVLink (CUDA 13.0+)

## Step-by-Step Walkthrough

**Première étape : calculer les dimensions grid et block.**

[FACT:src/enqueue/enqueue.cc:1889-1893]

```c
int nChannels = countOneBits(plan->channelMask);
void* sym = plan->kernelFn;
dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
```

`nChannels`est le nombre de bits positionnés dans`channelMask`, c'est-à-dire le nombre de blocks que ce plan doit lancer. Chaque block est responsable d'un channel.`threadPerBlock`est calculé dans`scheduleCollTasksToPlan`via`plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE)`en prenant le maximum de`nWarps * 32`。

`smem`parmi toutes les tasks.  est la taille de la mémoire partagée dynamique. Pour un kernel ordinaire, c'est`ncclShmemDynamicSize(comm->cudaArch)`, une constante de compilation qui dépend de l'architecture (sm70+ c'est`ncclShmemScratchWarpSize * (NCCL_MAX_NTHREADS / WARP_SIZE)`). Pour un sym kernel, c'est`plan->kernelDynSmem`, car les besoins en mémoire partagée du sym kernel peuvent différer.

**Deuxième étape : assembler les paramètres du kernel.**

[FACT:src/enqueue/enqueue.cc:1902-1903]

```c
void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
                 CU_LAUNCH_PARAM_END};
```

C'est une méthode de passage de paramètres de l'API CUDA driver :`CU_LAUNCH_PARAM_BUFFER_POINTER`indique au driver que « les paramètres ne sont pas passés un par un, mais sous forme d'un bloc mémoire contigu »,`CU_LAUNCH_PARAM_BUFFER_SIZE`indique au driver la taille de ce bloc. L'avantage est que NCCL peut passer`ncclDevKernelArgs`et le tableau batch suivant en une seule fois, sans avoir à empaqueter chaque paramètre individuellement.

**Troisième étape : ajouter les launch attributes.**

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

Le CGA (Cooperative Group Array) est une fonctionnalité matérielle introduite avec sm90, permettant de regrouper plusieurs blocks en un cluster. Les blocks d'un cluster peuvent être garantis d'être ordonnancés simultanément sur un ensemble de SM et peuvent accéder mutuellement à leur mémoire partagée. NCCL utilise cette fonctionnalité pour implémenter des algorithmes nécessitant une synchronisation inter-blocks comme NVLS.

Noter la protection`if (grid.x % clusterSize) clusterSize = 1;`: la dimension du cluster doit diviser exactement la dimension du grid, sinon le driver renverra une erreur. Si`grid.x`n'est pas divisible par`clusterSize`, on dégénère en n'utilisant pas de cluster.

**Quatrième étape : ajouter le launch completion event.**

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

`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`est une fonctionnalité introduite avec CUDA 12.3 : le driver enregistre un événement lorsque le kernel commence réellement son exécution (et non lorsque l'appel côté host retourne). Ceci est crucial pour implémenter l'« ordre implicite » (implicit order) — NCCL doit garantir que plusieurs kernels s'exécutent dans l'ordre, sans pour autant bloquer le host en attente.

`getImplicitOrder`La logique est : si l'utilisateur a défini`launchOrderImplicit`, et que la version du driver est suffisamment récente, on utilise`ncclImplicitOrderLaunch`(ordonnancement par launch event) ; sinon on utilise`ncclImplicitOrderSerial`(ordonnancement par completion event, c'est-à-dire exécution séquentielle).

**Cinquième étape : appeler`cuLaunchKernelEx`。**

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

`cuLaunchKernelEx`est une nouvelle API introduite avec CUDA 12.0, supportant les launch attributes. Pour les anciens drivers (< 11.8), NCCL revient à`cuLaunchKernel`：

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

## Contrôle de concurrence et interaction matérielle

**Mécanisme de relais du Launch completion event.**Lorsqu'on utilise`ncclImplicitOrderLaunch`et que l'utilisateur a fourni`launchCompletionEvent`, NCCL ne peut pas transmettre directement l'event de l'utilisateur au driver, car le driver ne supporte qu'un seul launch completion event. L'approche de NCCL est :

1. Transmettre`comm->sharedRes->launchEvent`au driver.

2. Attendre`relayStream`sur`launchEvent`。

3. Enregistrer l'event de l'utilisateur sur`relayStream`.

Ainsi l'event de l'utilisateur sera déclenché après le début réel de l'exécution du kernel, et non au retour de l'appel côté host.

**Mem Sync Domain。** [FACT:src/enqueue/enqueue.cc:1938-1942]Sur sm90+, NCCL définit`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`à`cudaLaunchMemSyncDomainRemote`. C'est le mécanisme de domaine de synchronisation mémoire introduit par l'architecture Hopper, utilisé pour isoler les barrières mémoire de différents kernels et réduire les surcoûts de synchronisation inutiles.

## Guide des pièges en production

**Piège 1 : dimension de cluster non divisible entraînant un échec de lancement.**Si`grid.x`n'est pas divisible par`clusterSize`, le driver renverra`CUDA_ERROR_INVALID_VALUE`. Le code source protège via`if (grid.x % clusterSize) clusterSize = 1;`, mais cela signifie aussi que la fonctionnalité cluster est silencieusement désactivée. Si l'utilisateur attend le gain de performance apporté par le cluster, il faut vérifier la relation entre`cgaClusterSize`et`nChannels`.

**Piège 2 : version de driver non satisfaisante rendant le kernel indisponible.** `ncclInitKernelsForDevice`vérifie les exigences de pilote de chaque kernel lors de l'initialisation :

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

est la partie la plus complexe. Sa tâche est de copier la structure work pointée par le descripteur de batch depuis la mémoire globale (ou les paramètres du kernel) vers`fnsOfBitset`de la mémoire partagée.`offsetBitset`Copier`fns`Le cœur de ce code est de calculer`fnsOfBitset[nWorksBelow]`。

: pour le n-ième bit activé dans

[FACT:src/device/common.h:209-241]

```c
if (tid = %d && __CUDA_ARCH__ >= %d\n" % (cudart ,arch))
  out("/*%4d*/ %s,\n" % (index, sym))
  if (cudart, arch) != (0, 0):
    out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
  index += 1
out("nullptr};\n")
```

## Réflexions de conception et pièges en production

**Pourquoi utiliser`__grid_constant__`？** [FACT:src/device/common.h:19-24]

```c
#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif
```

`__grid_constant__`indique au compilateur que ce paramètre est en lecture seule et peut être placé dans la mémoire constante. Ainsi, lors de la lecture côté device, l'instruction`ld.param`est utilisée, ce qui est plus rapide que la lecture depuis la mémoire globale. Le commentaire mentionne que cela casse cuda-gdb, donc ce n'est activé que sur sm70+.

**Piège 1 :`workStorage`débordement.** `workStorage`La taille de`ncclMaxDevWorkBatchBytes()`, sm90+ est de 16KB. Si`nWorks * workSize`dépasse cette valeur, il y aura un dépassement d'écriture. Dans le code source,`NCCL_MAX_DEV_WORK_BATCH_BYTES`limite la taille du batch côté host, mais il n'y a pas de vérification supplémentaire côté device. Si la contrainte côté host est contournée (par exemple en modifiant une variable d'environnement), cela entraînera un dépassement de la mémoire partagée.

**Piège 2 :`__syncthreads()`L'absence de provoque une course de données.**Après`loadWorkBatchToShmem`, il doit y avoir un`__syncthreads()`pour que tous les threads voient le`workStorage`complet. Dans le code source,[FACT:src/device/common.h:479]il y a`__syncthreads(); // publish ncclShmem`. Si cette synchronisation est supprimée, certains threads pourraient commencer à lire avant que`workStorage`ne soit entièrement écrit, ce qui entraînerait la lecture de données corrompues.

**Piège 3 : le moment de la vérification d'abort.** `while (ncclShmem.aborted == 0)`ne vérifie l'abort qu'au début de chaque batch. Si un batch prend beaucoup de temps à s'exécuter, le signal d'abort pourrait mettre longtemps à prendre effet. C'est un compromis de conception : des vérifications plus fréquentes augmentent la surcharge, mais la réponse est plus rapide.

# Sélection des variantes de kernel : comment generate.py génère la liste des kernels

## Modèle intuitif

`generate.py`Le rôle de est similaire à celui d'un « planificateur de ligne de production d'une usine automobile ». Il fait face à un espace combinatoire gigantesque (7 types d'opérations d'ensemble × 5 types d'opérations de réduction × 12 types de données × 7 algorithmes × 3 protocoles) et doit décider : quelles combinaisons nécessitent la génération de kernels spécialisés ? Lesquelles peuvent partager un kernel générique ?

Si un kernel est généré pour chaque combinaison, le temps de compilation et la taille du binaire exploseraient. Si un seul kernel générique est généré, l'exécution serait ralentie par les appels via pointeurs de fonction et les branchements.`generate.py`La solution de est le « kernel représentatif » : générer un kernel pour chaque classe d'équivalence, et distribuer à l'exécution via une table de pointeurs de fonction.

## Structures de données et disposition mémoire

`generate.py`génère trois fichiers clés :

1. **`device_table.cu`**: côté device`ncclDevFuncTable`, qui mappe funcId vers la fonction device concrète.

2. **`host_table.cc`**: côté host`ncclDevKernelList`、`ncclDevKernelForFunc`、`ncclDevFuncRowToId`et autres tables.

3. **Les différents`<coll>_<op>_<ty>.cu`**: implémentations concrètes des kernels.

## Step-by-Step Walkthrough

**Première étape : énumérer toutes les lignes de fonctions.**

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

Cet ordre d'énumération doit correspondre à la formule de calcul de`ncclDevFuncId()`:

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

`ncclDevFuncId`calcule le « numéro de ligne », puis via`ncclDevFuncRowToId`le mappe vers l'« ID de fonction principale ». La raison de ce mappage est que : de nombreuses lignes peuvent être mappées vers la même fonction principale (par exemple toutes les lignes de`AllReduce Sum i32`sont mappées vers la fonction principale de`AllReduce Sum u32`).

**Deuxième étape : calculer les fonctions principales et les fonctions kernel.**

[FACT:src/device/generate.py:211-225]

```python
func_rows = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}
kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))
```

`equivalent_primary`mappe les entiers signés vers les entiers non signés (car l'addition/multiplication est identique pour les deux) :

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

`best_kernel`mappe plusieurs fonctions principales vers le même kernel (par exemple tous les algorithmes de`AllGather`sont mappés vers`AllGather RING LL`）：

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

**Troisième étape : générer les définitions de kernel.**

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

`DEFINE_ncclDevKernel`Après expansion de la macro :

[FACT:src/device/common.h:507-509]

```c
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain, algo, proto>>(&args4K.args); \
  }
```

Donc chaque kernel est une fonction`__global__`, appelant`ncclKernelMain`, avec comme paramètres de template`specializedFnId`et`RunWorkBatch<coll, ty, redop<ty>, algo, proto>`。

## Réflexions de conception et pièges en production

> **[Design Inference & Architectural Trade-offs]**
> **Pourquoi utiliser un « kernel représentatif » plutôt qu'un kernel par combinaison ?**Compromis entre temps de compilation et taille du binaire. L'espace combinatoire complet est de 7 × 5 × 12 × 7 × 3 ≈ 8820 kernels, chaque kernel nécessitant quelques secondes de compilation, soit plusieurs heures au total. De plus, la taille du binaire atteindrait plusieurs centaines de Mo. En mappant vers des kernels représentatifs, le nombre de kernels réellement générés est réduit à quelques dizaines.

**Piège 1 :`NCCL_EXACT_KERNEL_NAMES`provoque une explosion de la compilation.**Si cette variable d'environnement est définie,`best_kernel`renvoie la fonction originale, et un kernel est généré pour chaque combinaison. C'est utile en développement (on peut contrôler précisément quel kernel est compilé), mais en production cela entraîne des temps de compilation excessifs.

**Piège 2 :`required_cuda`vérification de version.**Certains kernels nécessitent une version CUDA ou une architecture spécifique :

[FACT:src/device/generate.py:130-154]

À ce stade, le kernel a été lancé sur le GPU et le côté device a également obtenu la description du travail. Mais ce qui détermine réellement la performance, c'est la façon dont les données sont déplacées à l'intérieur du device. Le chapitre suivant explorera en profondeur les trois primitives de protocole sous src/device : LL, LL128 et Simple, pour comprendre pourquoi la même logique AllReduce nécessite trois ensembles de primitives de transfert, et leurs différences en matière de synchronisation, de disposition des buffers et de sémantique des flags.
