# Chapitre 24 : Évolution de l'architecture et directions futures : de la communication statique à la communication programmable

Dans le chapitre précédent, nous avons vu comment la communauté construit un écosystème périphérique autour du cœur de NCCL : bindings Python, bindings Rust, communication en parallélisme expert, primitives à ultra-haute bande passante, points de contrôle de communication. Ces projets réutilisent tous l'API stable de NCCL, mais leurs exigences dépassent déjà le cadre de la communication collective traditionnelle — le parallélisme expert nécessite des échanges point à point de granularité fine, les points de contrôle nécessitent de suspendre/reprendre l'état de communication, les primitives à ultra-haute bande passante nécessitent de contourner les opérations collectives standard pour agir directement sur le réseau. Ces exigences pointent vers la même question : le modèle d'opérations collectives fixes de NCCL est en train d'être débordé par des besoins de communication plus flexibles. Dans ce chapitre, nous n'examinerons plus un module unique, mais nous partirons des traces d'évolution déjà présentes dans le code source pour discuter de la direction que prend NCCL. Concrètement, nous analyserons trois forces d'évolution entrelacées : les primitives de communication passent de collectives fixes à programmables — l'ordonnancement des tâches RMA dans src/rma/rma.cc permet aux couches supérieures de composer les primitives Put/Signal/WaitSignal, au lieu de ne pouvoir appeler qu'AllReduce ; l'initiation réseau passe du host proxy à l'envoi direct depuis le GPU — la gestion du backend GIN dans src/gin/gin_host.cc permet au kernel GPU de piloter directement la carte réseau ; le modèle mémoire passe des buffers enregistrés à la mémoire symétrique — la sélection de kernel de mémoire symétrique dans src/sym_kernels.cc permet à tous les ranks d'utiliser le même ensemble d'adresses virtuelles pour accéder aux buffers des autres. Ces trois forces ne sont pas isolées ; elles partagent la même infrastructure : l'abstraction de team dans src/nccl_device/core.cc et le DevComm versionné dans src/devcomm/devcomm_v23100.cc. Comprendre comment elles s'articulent, c'est comprendre la logique d'évolution de NCCL, de « bibliothèque de communication collective » à « moteur de communication programmable ».

# I. Primitives de communication programmables : comment RMA transforme une « recette fixe » en « buffet libre-service »

## Modèle intuitif

La communication collective de NCCL traditionnel ressemble à un menu fixe : vous commandez AllReduce, la cuisine exécute tout selon le processus AllReduce. Mais dans le scénario de parallélisme d'experts (MoE), chaque token doit être envoyé à différents experts, et le modèle d'envoi n'est pas connu du tout au moment de la compilation — c'est comme un buffet, vous devez décider vous-même quoi prendre, combien prendre et quand prendre.

RMA est le « comptoir du buffet » que NCCL fournit à la couche supérieure : Put (écrire des données dans la mémoire du pair), Signal (notifier le pair), WaitSignal (attendre le signal du pair). Le framework supérieur peut combiner librement ces trois primitives pour réaliser n'importe quel modèle de communication.

Sans RMA, l'all-to-all de MoE ne peut être simulé que par de multiples opérations collectives de petite taille, chacune devant passer par le processus complet de lancement de kernel et de synchronisation, avec une latence inacceptable.

## Structures de données et disposition mémoire

La structure de données centrale de RMA est`ncclTaskRma`(description de tâche) et`ncclRmaArgs`(paramètres de plan). Regardons d'abord les champs de`ncclRmaArgs`, il est initialisé dans`scheduleRmaTasksToPlan`.

[FACT:src/rma/rma.cc:166-171]

```cpp
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
plan->rmaArgs->nRmaTasks = 0;
plan->rmaArgs->nRmaTasksProxy = 0;
plan->rmaArgs->nRmaTasksCe = 0;
```

Les champs clés ici sont`nRmaTasksProxy`et`nRmaTasksCe`. Ils divisent les tâches RMA en deux chemins d'exécution :

- **Chemin CE**(Copy Engine, moteur de copie) : le rank cible est dans la portée LSA (Local Symmetric Access, accès symétrique local), peut être réalisé directement avec le moteur de copie du GPU, sans réseau.
- **Chemin Proxy**: le rank cible n'est pas dans la portée LSA, doit passer par un thread proxy hôte pour piloter le réseau.

> **[Design Inference & Architectural Trade-offs]**
> La motivation de cette dichotomie est directe : la communication dans la portée LSA passe par NVLink ou PCIe, avec une bande passante élevée et une faible latence, la copie asynchrone par CE est la plus avantageuse ; la communication inter-machines doit passer par la carte réseau, et ne peut être pilotée que par un thread proxy. Séparer la planification des deux types de tâches permet au CE et au proxy de s'exécuter en parallèle, plutôt que d'attendre en série.

`ncclTaskRma`contient lui-même`peers`、`nsignals`、`signalIdxs`trois pointeurs de tableaux, enregistrant respectivement le rank pair, le nombre de signaux et l'index de signal. Pour les tâches WaitSignal, une tâche peut attendre plusieurs peers ; pour les tâches Put/Signal, une tâche ne cible qu'un seul peer.

## Step-by-Step Walkthrough : la planification d'un WaitSignal

Prenons un scénario concret : le rank 0 appelle`ncclWaitSignal`, en attendant les signaux des ranks 1 et 3. Supposons que le rank 1 est dans la portée LSA, et le rank 3 ne l'est pas.

**Première étape : trouver la première file de contexte non vide.**

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

Les tâches RMA sont mises en file par contexte, chaque contexte est un canal RMA indépendant. Ici, on trouve le premier contexte ayant des tâches, et on récupère sa file.

**Deuxième étape : retirer la première tâche, déterminer le type.**

[FACT:src/rma/rma.cc:163-168]

```cpp
struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);
plan->isRma = true;
plan->rmaArgs = ncclMemoryStackAlloc(&comm->memScoped);
plan->rmaArgs->func = firstTask->func;
```

`firstTask->func`est`ncclFuncWaitSignal`, on entre dans la branche WaitSignal.

**Troisième étape : diviser les peers selon l'accessibilité LSA.**

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

`isLsaAccessible`parcourt`comm->devrState.lsaRankList`, détermine si le peer est dans l'équipe LSA. Le rank 1 est dans LSA, va dans la liste CE ; le rank 3 n'y est pas, va dans la liste Proxy.

**Quatrième étape : créer une nouvelle tâche pour CE et Proxy respectivement.**

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

La tâche WaitSignal originale est divisée en deux : la tâche CE attend le rank 1, la tâche Proxy attend le rank 3. Les deux tâches peuvent s'exécuter en parallèle — le chemin CE attend sur le GPU, le chemin Proxy attend sur le thread hôte.

**Cinquième étape : libérer la tâche originale.**

[FACT:src/rma/rma.cc:249-251]

```cpp
planner->nTasksRma -= 1;
ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
```

La tâche originale a été divisée en deux nouvelles tâches, libérée vers le pool de mémoire.

## Contrôle de concurrence et interaction matérielle

L'exécution parallèle de RMA se manifeste dans`ncclRmaWaitSignal`.

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

Ce code utilise les événements CUDA pour la synchronisation entre flux : d'abord enregistrer un événement sur le flux d'entrée, faire attendre le flux CE cet événement, puis lancer respectivement les tâches proxy et CE sur les deux flux, et enfin faire attendre le flux d'entrée l'événement du flux CE. Ainsi les deux chemins avancent en parallèle, mais se présentent extérieurement comme une opération synchrone.

> **[Design Inference & Architectural Trade-offs]**
> Le compromis de conception ici est : l'exécution parallèle peut réduire la latence, mais introduit un surcoût supplémentaire d'enregistrement d'événements et de synchronisation de flux. Pour les petits messages, ce surcoût peut dépasser le gain du parallélisme ; pour les grands messages, le gain du parallélisme est significatif. NCCL ne fait pas de jugement adaptatif ici, mais suit uniformément le chemin parallèle — car le scénario typique de RMA est la communication fine-grained de grands messages.

## Guide de production pour éviter les pièges

**Piège 1 : une erreur de jugement d'accessibilité LSA fait que la tâche prend le mauvais chemin.** `isLsaAccessible`parcourt`lsaRankList`, si`lsaSize`est 0 (par exemple un domaine de communication à rank unique), tous les peers seront jugés inaccessibles, et passeront tous par le chemin Proxy. Cela ne se manifestera pas lors de tests à petite échelle, mais entraînera une chute brutale des performances lors d'un déploiement à grande échelle. La méthode de diagnostic est de regarder dans les logs INFO de`scheduleRmaTasksToPlan`le ratio de`nRmaTasksProxy`et`nRmaTasksCe`.

**Piège 2 : le cycle de vie du tableau peer après la division de la tâche WaitSignal.**Le`peersCe`du chemin CE utilise`ncclMemoryStackAlloc`pour l'allocation, le cycle de vie suit`comm->memScoped`; le`peersProxy`du chemin Proxy utilise`ncclCalloc`pour l'allocation, et doit être manuellement [libéré] après l'exécution de la tâche`free`. Si la création de la tâche Proxy échoue,`fail`la branche libère ces tableaux.

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

**Piège 3 : traitement par lots inter-context des tâches Put/Signal.**Dans la branche Put/Signal, NCCL regroupe les tâches put/signal de tous les contextes dans un même plan, mais s'arrête à la rencontre d'un WaitSignal.

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

L'intention de cette conception est : un seul lancement de kernel couvre les put/signal de tous les contextes, réduisant ainsi les frais de lancement. Mais la file de chaque contexte n'est consommée que jusqu'au premier WaitSignal, garantissant l'ordre FIFO par contexte. Si la couche supérieure alterne les appels à put et waitSignal dans un même contexte, l'effet de traitement par lots est fortement réduit — c'est un pattern à surveiller lors de l'utilisation de RMA.

---

# II. Émission directe depuis le GPU vers le réseau : comment GIN permet au kernel de contourner le host proxy

## Modèle intuitif

La communication réseau traditionnelle de NCCL ressemble à l'envoi d'une lettre : le kernel GPU place les données dans un tampon, le thread host proxy transmet les données à la carte réseau, et la carte réseau les envoie. GIN, quant à lui, permet au kernel GPU de déposer directement la lettre dans la boîte aux lettres du destinataire — le kernel écrit directement dans la file d'envoi de la carte réseau, et la carte réseau lit directement la mémoire GPU.

Sans GIN, chaque communication réseau doit transiter par la mémoire hôte, ajoutant au moins un aller-retour PCIe de latence. Pour une communication fine-grained comme MoE, cette latence est fatale.

## Structures de données et disposition mémoire

L'état central de GIN est`ncclGinState`, qui gère plusieurs backends et plusieurs DevComm. Examinons d'abord la table de compatibilité des versions de backend.

[FACT:src/gin/gin_host.cc:27-33]

```cpp
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5), NCCL_VERSION(2, 32, 0)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
constexpr int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0), NCCL_VERSION(2, 32, 0)};
```

L'indice de ces tableaux est le numéro de version du backend, et la valeur est la version minimale de NCCL compatible. Par exemple,`proxyBackendMinVersions[3]`correspond au backend version 3, exigeant NCCL au minimum 2.32.0. Cette conception permet à NCCL de choisir la version de backend appropriée à l'exécution en fonction de la version du code du périphérique, plutôt qu'une liaison à la compilation.

> **[Design Inference & Architectural Trade-offs]**
> La motivation de cette table de compatibilité des versions est la suivante : le backend GIN (pilote de carte réseau, firmware) et la bibliothèque NCCL évoluent à des rythmes différents. Si les exigences de version étaient codées en dur, la mise à niveau de l'un ou l'autre entraînerait une incompatibilité. L'utilisation d'un tableau pour le mappage des versions permet une sélection dynamique à l'exécution, assurant la rétrocompatibilité avec les anciens backends.

`ncclGinStateDevComm`est l'état GIN de chaque DevComm, contenant`contextCount`、`backendIndex`、`ginCtx[]`、`devHandles[]`et d'autres champs. Il est chaîné en liste liée et attaché à`ginState->devComms`.

## Step-by-Step Walkthrough : établissement d'une connexion GIN

Plaçons-nous dans un scénario : le rank 0 initialise le domaine de communication et doit établir une connexion GIN.

**Première étape : vérifier si GIN est activé et pris en charge.**

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

`ncclParamGinEnable()`lit la variable d'environnement`NCCL_GIN_ENABLE`, par défaut 1. Si l'utilisateur la désactive explicitement, une erreur est renvoyée directement.

**Deuxième étape : vérifier la prise en charge de la mémoire symétrique.**

[FACT:src/gin/gin_host.cc:111-114]

```cpp
if (!comm->symmetricSupport) {
  WARN("Communicator does not support symmetric memory!");
  return ncclInternalError;
}
```

GIN dépend de la mémoire symétrique — car le kernel GPU doit connaître l'adresse virtuelle du tampon du pair, et seule la mémoire symétrique garantit la cohérence des adresses.

**Troisième étape : obtenir la liste des périphériques GIN locaux.**

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

`ncclTopoGetLocalGinDevs`identifie toutes les cartes réseau prenant en charge GIN à partir du graphe de topologie. Si le nombre dépasse`NCCL_GIN_MAX_CONNECTIONS`, seules les premières sont retenues et un avertissement est affiché.

**Quatrième étape : calculer l'équipe GIN.**

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

Chaque backend appelle d'abord`devices`pour obtenir le nombre de périphériques, puis exécute le flux listen→getProperties→allGather→connect→closeListen pour chaque connexion.`bootstrapAllGather`échange les handles entre tous les ranks, de sorte que chaque rank connaisse les informations de connexion de ses pairs.

## Contrôle de concurrence et interaction matérielle

Le thread de progression de GIN est le mécanisme de concurrence central.

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

Voici plusieurs conceptions clés :

1. **Affinité CPU**：`ncclOsSetAffinity`lie le thread de progression à un cœur CPU spécifié, évitant l'invalidation du cache due à la migration de thread.

2. **Recul du verrou d'écriture**：`writePending`est un indicateur atomique ; lorsque le thread principal veut modifier`devComms`la liste liée, il le positionne d'abord, et le thread de progression, le voyant, cède activement pour éviter la contention de verrou.

3. **Verrou lecture-écriture**：`devCommRwMutex`est`shared_timed_mutex`, le thread de progression détient le verrou de lecture pour parcourir la liste liée, et le thread principal détient le verrou d'écriture pour modifier la liste liée.

4. **Répartition des threads**: le thread t est responsable des connexions t, t+proxyNthreads, t+2*proxyNthreads, ..., réalisant l'équilibrage de charge via une boucle stride.

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

Cette implémentation du verrou d'écriture suppose qu'il n'y a qu'un seul écrivain (le thread principal), donc aucune exclusion mutuelle supplémentaire n'est nécessaire.`writePending`positionne d'abord puis acquiert le verrou, garantissant que le thread de progression puisse voir l'intention d'écriture avant d'acquérir le verrou et reculer activement.

## Guide de production pour éviter les pièges

**Piège 1 : un nombre de connexions GIN non correspondant provoque un interblocage AllGather.**Le`ginCommCount`de chaque rank peut différer (selon le nombre de cartes réseau locales), NCCL prend la valeur minimale parmi tous les ranks via`bootstrapAllGather`.

[FACT:src/gin/gin_host.cc:176-180]

```cpp
ginCommCountHandles[comm->rank] = backend->ginCommCount;
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
for (int r = 0; r nRanks; r++) {
  backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
}
```

Si le nombre de cartes réseau d'un rank est inférieur à celui des autres ranks, tous les ranks sont réduits à la valeur minimale. Cela garantit une connexion symétrique, mais gaspille les ressources des cartes réseau.

**Piège 2 : proxyNthreads dépasse ginCommCount, ce qui entraîne une rotation à vide des threads.**Si l'utilisateur a défini`NCCL_GIN_PROXY_NTHREADS`supérieur à`ginCommCount`, les threads excédentaires tournent à vide dans la boucle stride.

[FACT:src/gin/gin_host.cc:181-183]

```cpp
// After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
// on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
// Extra threads simply idle in the stride loop; no correctness issue.
```

Ce n'est pas un problème de correction, mais cela gaspille des ressources CPU. La méthode de diagnostic consiste à vérifier si`NCCL_GIN_PROXY_NTHREADS`est supérieur au nombre réel de cartes réseau.

**Piège 3 : condition de course lors de la libération de DevComm.** `ncclGinDevCommFree`On retire d'abord le DevComm de la liste chaînée, puis on détruit le context.

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

Après le retrait, le thread de progression ne voit plus ce DevComm, donc la destruction du context est sûre. Mais si des opérations réseau in-flight sont en cours pendant la destruction, cela peut entraîner un comportement indéfini — c'est ce qu'il faut garantir lors de l'utilisation de GIN : avant de libérer le DevComm, il faut s'assurer que toutes les opérations sont terminées.

---

# III. Kernel de mémoire symétrique : du « buffer enregistré » à « l'espace d'adressage unifié »

## Modèle intuitif

Le buffer du NCCL traditionnel est un « système d'enregistrement » : chaque rank enregistre son propre buffer, et lors de la communication, les adresses sont échangées via un handle. La mémoire symétrique, quant à elle, est un « espace d'adressage unifié » : tous les ranks conviennent du même ensemble d'adresses virtuelles ; l'adresse A du rank 0 et l'adresse A du rank 1 pointent vers leurs mémoires physiques respectives, mais le code peut y accéder en utilisant la même adresse.

C'est comme si tout le monde convenait que « 3e rangée, 5e siège » désigne le même emplacement chez chacun, sans avoir à demander d'abord « où se trouve le 3e rangée, 5e siège chez toi » pour trouver quelque chose.

Sans mémoire symétrique, chaque kernel devrait d'abord résoudre l'adresse du pair, ce qui augmenterait la surcharge d'instructions et la pression sur les registres.

## Structures de données et disposition mémoire

Le cœur du kernel de mémoire symétrique est le kernel mask — une bitmap qui marque quels kernels sont disponibles dans le domaine de communication actuel.

[FACT:src/sym_kernels.cc:17-63]

```cpp
constexpr uint32_t kernelMask_STMC =
  1  **[Design Inference & Architectural Trade-offs]**
> L'avantage de cette conception par bitmap est qu'elle permet de filtrer rapidement les kernels disponibles par opérations bit à bit. Par exemple,`kmask &= ~kernelMask_STMC`une seule ligne suffit pour désactiver tous les kernels STMC, sans parcourir la liste.

## Step-by-Step Walkthrough : un calcul de kernel mask

Prenons un scénario : le rank 0 doit exécuter un AllReduce, le type de données est float16, la taille du message est 1 Mo, le domaine de communication compte 8 ranks, tous interconnectés par NVLink.

**Première étape : obtenir le mask de base correspondant à l'opération.**

[FACT:src/sym_kernels.cc:304-306]

```cpp
uint32_t kmask = kernelMask_coll(coll);
```

`kernelMask_coll(ncclFuncAllReduce)`retourne`kernelMask_AR`, contenant 5 kernels AllReduce.

**Deuxième étape : vérifier la disponibilité de STMC et LDMC.**

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

`hasLsaMultimem`est calculé dans`ncclSymkInitOnce`, ce qui exige que le multicast symétrique NVLS soit disponible et que l'équipe LSA compte plus de 2 ranks. float16 prend en charge LDMC, donc si`hasLsaMultimem`est vrai, le kernel LDMC est conservé.

**Troisième étape : vérifier la limite de taille de message.**

[FACT:src/sym_kernels.cc:336-342]

```cpp
size_t nBytes = alignUp(nElts * ncclTypeSize(ty), NCCL_SYM_KERNEL_CELL_SIZE);
size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks) * nBytes;
if (nBusBytes >= (size_t(2) = 32 * (size_t(2) nRanks;
kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
```

Si l'équipe LSA couvre tous les ranks, GIN n'est pas nécessaire ; sinon, seuls les kernels GIN sont conservés.

## Contrôle de concurrence et interaction matérielle

L'initialisation du kernel de mémoire symétrique implique la création de DevComm et l'allocation de ressources.

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

Le point clé ici est`ncclDevrCommCreateInternal`, qui crée un DevComm interne contenant les ressources telles que le multicast LSA, les inbox/outbox GIN, les signaux, etc.`reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL`spécifie que GIN utilise le mode de connexion rail.

[FACT:src/sym_kernels.cc:257-261]

```cpp
symk->kcomm.workStarted = comm->profiler.symWorkStarted;
symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
symk->kcomm.workPhases = comm->profiler.symWorkPhases;
```

Le kernel de mémoire symétrique utilise un buffer de profiler indépendant pour éviter l'entrelacement avec le workCounter des kernels classiques.

## Guide de production pour éviter les pièges

**Piège 1 : besoins SMEM du kernel TMA.**TMA nécessite environ 8 Ko de SMEM scratch par warp, soit 128 Ko pour 16 warps.

[FACT:src/sym_kernels.cc:135-142]

```cpp
bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  if (comm->maxSharedMemOptin minCompCap >= 100 && ncclParamSymTmaEnable();
}
```

Si la capacité SMEM du GPU est insuffisante (par exemple une instance MIG), le kernel TMA sera désactivé. La méthode de diagnostic consiste à vérifier si`maxSharedMemOptin`est inférieur à`ncclTmaShmemScratchWarpSize() * 16`。

**Piège 2 : limites du chunk size GIN.**Le chunk size du kernel ReduceScatter GIN a des limites inférieure et supérieure.

[FACT:src/sym_kernels.cc:148-153]

```cpp
static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128  0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}
```

Si l'utilisateur a défini`NCCL_SYM_RS_GIN_CHUNK_SIZE`au-delà de 1 Go, il sera tronqué à 1 Go ; s'il est inférieur à 128 octets, il sera relevé à 128 octets. La valeur finale sera également arrondie à la puissance de 2 inférieure.

**Piège 3 : Incompatibilité du type d'enregistrement de la mémoire symétrique.** `ncclGetSymRegType`Selon les`NCCL_WIN_COLL_SYMMETRIC`indicateurs de sendWin et recvWin, déterminer le type d'enregistrement.

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

Si les types d'enregistrement de send et recv sont incohérents, le kernel doit emprunter des chemins de code différents. Cela affecte les performances, mais ne provoque pas d'erreur.

---

# IV. Abstraction Team et DevComm versionné : l'infrastructure de l'évolution

## Modèle intuitif

L'abstraction Team est comme un « regroupement » : l'équipe mondiale est la classe entière, l'équipe LSA est le voisin de table, l'équipe Rail est la même colonne de sièges. Différents modes de communication nécessitent différentes perspectives de regroupement.

Le DevComm versionné est comme un « traducteur » : différentes versions du code device parlent différents « dialectes », la couche de compatibilité DevComm se charge de traduire, permettant aux anciens et nouveaux codes de se comprendre mutuellement.

Sans l'abstraction Team, chaque kernel devrait calculer lui-même le mapping des ranks ; sans le DevComm versionné, tout changement d'ABI entraînerait la recompilation de tout le code device.

## Structures de données et disposition mémoire

Team est un simple triplet :`nRanks`、`rank`、`stride`。

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

Le stride de l'équipe mondiale est 1, car tous les ranks sont disposés consécutivement.

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

Le stride de l'équipe Rail est`lsaSize`, car les ranks sur chaque rail sont espacés de la taille d'une équipe LSA.

Le cœur du DevComm versionné est la`ncclDevCommCompat`structure.

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

Cette structure définit les règles de compatibilité de la version 2.31.0.`minVersion`et`maxVersion`définissent la plage de versions applicables, les quatre pointeurs de fonction suivants définissent la logique de filtrage des propriétés et de conversion de structure. Si tous sont nullptr, cela signifie que cette version n'a pas de besoin de compatibilité particulier.

## Step-by-Step Walkthrough : une conversion Team

Prenons un scénario : rank 5 dans un domaine de communication de 8 ranks, la taille de l'équipe LSA est 4. Il faut calculer le rank de rank 5 dans l'équipe Rail.

**Première étape : initialiser l'état DevR.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};
```

`ncclDevrInitOnce`Calcule les informations dérivées comme l'équipe LSA, l'équipe CFT, etc. En cas d'échec, retourne une équipe vide.

**Deuxième étape : calculer les paramètres de l'équipe Rail.**

[FACT:src/nccl_device/core.cc:70-79]

```cpp
ncclTeam_t ans;
ans.nRanks = comm->nRanks / comm->devrState.lsaSize;  // 8 / 4 = 2
ans.rank = comm->rank / comm->devrState.lsaSize;       // 5 / 4 = 1
ans.stride = comm->devrState.lsaSize;                  // 4
```

Le rank de rank 5 dans l'équipe Rail est 1, l'équipe a 2 ranks, le stride est 4.

**Troisième étape : reconvertir en rank mondial.**

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Pour convertir Rail rank 0 en rank mondial :`5 + (0 - 1) * 4 = 1`. Vérification : rank 1 et rank 5 sont sur le même rail (intervalle de 4).

## Contrôle de concurrence et interaction matérielle

L'abstraction Team elle-même est sans état, ne nécessite pas de contrôle de concurrence. Mais`ncclDevrInitOnce`est chargé paresseusement, toutes les informations dérivées sont calculées lors du premier appel.

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

Le commentaire dit « Ignoring errors since if it fails ncclDevrInitOnce will try again » — si l'initialisation échoue, retourne une équipe vide, le prochain appel réessaiera.

## Guide de production pour éviter les pièges

**Piège 1 : hypothèse de stride dans la conversion Team.** `ncclTeamRankToWorld`suppose que les ranks dans l'équipe forment une progression arithmétique.

[FACT:src/nccl_device/core.cc:82-84]

```cpp
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}
```

Si l'équipe n'est pas une progression arithmétique (par exemple un regroupement arbitraire personnalisé), cette fonction calculera faux. NCCL ne supporte actuellement que les équipes régulières.

**Piège 2 : pointeur nul dans le DevComm versionné.** `ncclDevCommCompat_v23100`Tous les pointeurs de fonction de sont nullptr, indiquant l'absence de logique de compatibilité particulière. Si une future version nécessite une conversion, ces fonctions doivent être implémentées, sinon les anciens et nouveaux codes ne pourront pas interopérer.

**Piège 3 : mode hiérarchique de l'équipe CFT.** `ncclTeamCft`supporte trois modes : FLAT, HIER_MULTIMEM, HIER_LSA.

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

Si un mode invalide est passé, retourne une équipe vide. Lors de l'utilisation de l'équipe CFT, il faut s'assurer que le mode est correct.

---

# Réflexions de conception

**Pourquoi NCCL doit-il supporter simultanément les trois voies d'évolution RMA, GIN et mémoire symétrique ?**

> **[Design Inference & Architectural Trade-offs]**
> Ces trois voies résolvent des problèmes à différents niveaux :

- **RMA**résout le problème du « mode de communication fixe » — permettant aux couches supérieures de combiner des primitives pour réaliser n'importe quel mode de communication.
- **GIN**résout le problème de la « latence réseau élevée » — permettant au GPU de piloter directement la carte réseau, contournant le host proxy.
- **Mémoire symétrique**résout le problème du « coût de résolution d'adresse » — permettant au kernel d'accéder directement à la mémoire distante via une adresse unifiée.

Ils ne sont pas en relation de substitution, mais de complémentarité. RMA peut utiliser GIN comme transport sous-jacent, GIN dépend de la mémoire symétrique pour fournir la cohérence d'adresse. Les trois constituent ensemble l'infrastructure du « moteur de communication programmable ».

**Quelle est la philosophie de conception du DevComm versionné ?**

> **[Design Inference & Architectural Trade-offs]**
> L'idée centrale du DevComm versionné est « ABI stable, API évolutive ». Le code device (kernel) est compilé et intégré au binaire, il ne peut pas être recompilé lors de la mise à niveau de la bibliothèque NCCL. Donc NCCL doit garantir que l'ancien code device peut fonctionner sur la nouvelle bibliothèque.`ncclDevCommCompat`La structure est le point d'entrée de la couche de compatibilité : la nouvelle bibliothèque sélectionne les règles de compatibilité appropriées selon la version du code device, et effectue des conversions de structure si nécessaire.

---

# Résumé de ce chapitre

Dans ce chapitre, en partant des traces d'évolution dans le code source, nous avons analysé les trois forces qui font passer NCCL d'une bibliothèque de communication collective à un moteur de communication programmable :

1. **RMA**（`src/rma/rma.cc`) : En combinant les primitives Put/Signal/WaitSignal, permettre aux couches supérieures d'implémenter n'importe quel modèle de communication. La conception centrale consiste à diviser les tâches en deux chemins parallèles, CE et Proxy, selon l'accessibilité LSA.

2. **GIN**（`src/gin/gin_host.cc`) : En envoyant directement depuis le GPU vers le réseau, en contournant le proxy hôte. La conception centrale comprend la gestion multi-backend, la table de compatibilité des versions et le pool de threads de progression.

3. **kernel de mémoire symétrique**（`src/sym_kernels.cc`) : En unifiant l'espace d'adressage, éliminer le surcoût de résolution d'adresses. La conception centrale repose sur le bitmap de masque de kernel et l'accélération matérielle TMA/GIN.

4. **Abstraction Team et DevComm versionné**（`src/nccl_device/core.cc`、`src/devcomm/devcomm_v23100.cc`) : Fournir l'infrastructure pour l'évolution. Team offre une vue de regroupement, le DevComm versionné assure la compatibilité ABI.

L'impact de ces changements sur les frameworks supérieurs est profond : le ProcessGroup de PyTorch peut appeler directement les primitives RMA pour implémenter des modèles de communication personnalisés ; le parallélisme d'experts de Megatron peut exploiter GIN pour réduire la latence all-to-all ; la mémoire symétrique simplifie le code des kernels.

# Réflexions et auto-évaluation de ce chapitre

Q1 : Si l'on supprime`scheduleRmaTasksToPlan`le jugement d'accessibilité LSA de la branche WaitSignal dans , et que tous les peers empruntent le chemin Proxy, quelles en seraient les conséquences ? Dans quels scénarios cela déclencherait-il une catastrophe de performance ?

**Analyse de référence**：

Le jugement d'accessibilité LSA se trouve dans[FACT:src/rma/rma.cc:187-204], il divise les peers en deux groupes : CE et Proxy. Si l'on supprime ce jugement, tous les peers empruntent le chemin Proxy,`nRmaTasksCe`reste toujours à 0.

Les conséquences sont : le chemin CE n'est plus du tout utilisé, tous les WaitSignal passent par le polling du thread proxy hôte sur le réseau. Pour les peers à portée LSA (interconnectés par NVLink sur la même machine), on pouvait initialement utiliser le moteur de copie GPU pour attendre de manière asynchrone, désormais c'est un thread hôte qui fait du polling, la latence passe de l'ordre de la microseconde à celui de la milliseconde.

Scénario de catastrophe de performance : dans l'entraînement MoE, chaque token doit attendre les signaux de plusieurs experts. Si tous les signaux passent par Proxy, le thread hôte devient le goulot d'étranglement, et le GPU passe une grande partie de son temps à attendre le polling de l'hôte. Sur une machine à 8 GPU entièrement NVLink, cette dégradation est particulièrement marquée — alors que toutes les communications pouvaient initialement passer par CE, elles sont maintenant toutes concentrées sur l'hôte.

Méthode de diagnostic : consulter`scheduleRmaTasksToPlan`les logs INFO de , si`nRmaTasksCe`reste toujours à 0 alors que`nRmaTasksProxy`est très grand, cela indique un problème dans le jugement LSA.

Q2：`ncclGinProgress`Dans , la combinaison`writePending`du flag et`devCommRwMutex`du verrou lecture-écriture, si l'on supprime`writePending`la vérification et que l'on ne conserve que le verrou lecture-écriture, quels problèmes cela poserait-il ?

**Analyse de référence**：

`writePending`La vérification se trouve dans[FACT:src/gin/gin_host.cc:63-66], elle permet au thread de progression de céder activement lorsque le thread principal veut écrire. Si l'on supprime cette vérification, le thread de progression tentera directement d'acquérir le verrou de lecture.

Le problème est le suivant :`std::shared_timed_mutex`le verrou de lecture de est partagé, plusieurs threads de progression peuvent le détenir simultanément. Si le thread principal veut acquérir le verrou d'écriture, il doit attendre que tous les verrous de lecture soient libérés. Sous forte charge, les threads de progression acquièrent fréquemment le verrou de lecture, le thread principal peut rester longtemps sans pouvoir obtenir le verrou d'écriture, ce qui provoque le blocage de`ncclGinDevCommSetup`ou`ncclGinDevCommFree`.

Plus grave encore : si le thread principal positionne d'abord`ginProgressWriteLock`dans`writePending`puis acquiert le verrou, et que les threads de progression ne vérifient pas`writePending`, alors les threads de progression peuvent encore acquérir le verrou de lecture après que le thread principal l'a positionné, rendant le temps d'attente du thread principal imprévisible.

`writePending`Le rôle de est une « notification souple » : dire aux threads de progression « je vais écrire, laissez-moi la place ». C'est plus efficace que de dépendre simplement de l'équité du verrou, car les threads de progression peuvent céder activement au lieu de se bloquer sur le verrou.

Q3：`ncclSymkMask`Dans , si`nBusBytes >= 32 * (size_t(2) << 30)`désactive tous les kernels (`kmask = 0`), alors`ncclSymkAvailable`retourne false, vers quel chemin NCCL va-t-il se replier ? Quel est l'impact sur les performances de ce chemin de repli ?

**Analyse de référence**：

`kmask = 0`Dans[FACT:src/sym_kernels.cc:342], à ce moment`ncclSymkAvailable`retourne false ([FACT:src/sym_kernels.cc:354-361]）。

Le chemin de repli est : NCCL utilisera les kernels de communication collective traditionnels (kernels de mémoire non symétrique). Ces kernels accèdent à la mémoire distante via des buffers enregistrés, nécessitant d'abord une résolution d'adresse, ce qui entraîne un surcoût d'instructions plus important.

Impact sur les performances : pour les très gros messages (dépassant 64 Go d'octets de bus), le surcoût de résolution d'adresse des kernels traditionnels est négligeable, car le transfert de données lui-même domine. Mais dans les cas limites (juste au-dessus de 64 Go), les kernels traditionnels peuvent être 10 à 20 % plus lents que les kernels de mémoire symétrique.

La raison fondamentale de cette limitation est que : les kernels de mémoire symétrique utilisent des entiers 32 bits pour suivre les chunks de boucle déroulée, chaque chunk faisant au moins 32 octets, donc la plage adressable maximale est de 32 * 2^31 = 64 Go. Au-delà de cette plage, il y a débordement d'entier.

En production réelle, les scénarios où une seule communication collective dépasse 64 Go sont rares (généralement un all-reduce après accumulation de gradients), mais pas impossibles. Si l'on rencontre ce scénario, on peut envisager une communication fragmentée ou l'utilisation de kernels traditionnels.

---

# Transition de fin de chapitre

Dans ce chapitre, nous avons vu que NCCL évolue d'« opérations collectives fixes » vers un « moteur de communication programmable » : RMA fournit la composition de primitives, GIN fournit l'envoi direct depuis le GPU, la mémoire symétrique fournit un espace d'adressage unifié, Team et le DevComm versionné fournissent l'infrastructure.

Ces évolutions ne sont pas isolées, elles pointent toutes ensemble vers un objectif :**permettent aux frameworks de niveau supérieur de mettre en œuvre des modes de communication personnalisés avec une latence plus faible et une flexibilité accrue**. Pour des frameworks comme PyTorch et Megatron, cela signifie qu'ils peuvent construire directement au-dessus de NCCL des modes de communication complexes tels que le all-to-all MoE, le parallélisme pipeline, le parallélisme d'experts, sans avoir à contourner NCCL pour implémenter leur propre couche réseau.

Le chapitre suivant est le dernier du livre. Nous allons reparcourir l'intégralité de la chaîne d'un AllReduce — depuis l'appel à`ncclAllReduce`, en passant par la mise en file des tâches, la sélection de l'algorithme, le lancement du kernel, la progression du proxy, le transfert réseau, jusqu'au retour du résultat. Cette rétrospective reliera les connaissances des 24 chapitres précédents pour former une carte cognitive complète.

À ce stade, nous avons clairement identifié les trois axes principaux de l'évolution de NCCL, passant d'opérations collectives fixes à un moteur de communication programmable : la composition de primitives RMA, l'envoi direct depuis le GPU vers le réseau, le modèle de mémoire symétrique, ainsi que l'abstraction team et le DevComm versionné qui les soutiennent. Ces mécanismes convergent vers un avenir de communication plus flexible et plus proche des capacités matérielles. Cependant, quelle que soit l'évolution de l'architecture, la chaîne complète d'un AllReduce reste la pierre angulaire de la compréhension de NCCL. Le chapitre suivant n'introduira aucun nouveau code, mais reprendra de bout en bout le flux des chapitres 3 à 10 — depuis l'appel à ncclAllReduce, jusqu'à l'établissement du domaine de communication, la recherche de topologie, la sélection d'algorithme, la mise en file des tâches, le lancement du kernel, l'exécution des primitives côté device, et l'écriture des résultats. Vous réassemblerez les mécanismes dispersés dans chaque chapitre en un modèle mental complet, et obtiendrez un index « quel chapitre consulter en cas de problème ».
