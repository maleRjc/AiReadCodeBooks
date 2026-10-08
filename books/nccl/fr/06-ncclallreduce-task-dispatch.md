# Chapitre 6 : Panorama de la soumission d'opérateurs : comment ncclAllReduce devient une tâche kernel exécutable

Dans le chapitre précédent, nous avons parcouru le module tuning et avons appris que NCCL sélectionne en quelques microsecondes la combinaison (algorithme, protocole, channel, warp) pour une communication collective. Mais le résultat de cette sélection n'est qu'un ensemble de nombres — il doit être « traduit » en un objet de description de tâche compréhensible par le kernel GPU pour pouvoir être réellement exécuté. Ce chapitre entre dans le corps principal de src/enqueue/enqueue.cc et répond à une question centrale : lorsque l'utilisateur appelle ncclAllReduce, que se passe-t-il réellement côté host ? De ncclAllReduce à ncclEnqueueCheck, en passant par la validation des paramètres, la détermination de l'algorithme/protocole, le découpage en channels, pour finalement générer les structures ncclInfo et ncclTaskColl. C'est le chapitre clé de tout l'ouvrage qui bascule du « point de vue utilisateur » au « point de vue moteur ». Si l'on compare NCCL à un restaurant, alors le module enqueue est le « système de prise de commande en salle » : l'utilisateur (couche applicative) dit « je veux un AllReduce », et la salle le traduit en un bon de travail exécutable par la cuisine (kernel GPU) — quel numéro de feu, quelle casserole utiliser, en combien de fournées. Sans cette couche de traduction, la cuisine ne saurait pas du tout quel plat préparer.

# I. Entrée : comment ncclAllReduce construit ncclInfo

## Modèle intuitif

`ncclAllReduce`est la fonction API appelée directement par l'utilisateur. Sa responsabilité est extrêmement simple :**empaqueter les paramètres bruts fournis par l'utilisateur dans une structure`ncclInfo`, puis la transmettre à`ncclEnqueueCheck`**. C'est comme lorsque vous vous rendez au guichet d'une banque : le guichetier remplit d'abord votre demande dans un formulaire standard, puis la transmet au système backend.

Sans cette couche, chaque API de communication collective devrait gérer elle-même la validation des paramètres, la sémantique de group, l'instrumentation profiler — le code deviendrait dupliqué au point d'être ingérable.

## Structure de données : disposition mémoire de ncclInfo

`ncclInfo`est le vecteur central qui traverse tout le flux enqueue. Sa définition se trouve dans`src/include/info.h`：

[FACT:src/include/info.h:17-44]

Cette structure possède plus de 20 champs, que l'on peut répartir en quatre groupes fonctionnels :

| Groupe de champs | Champ | Rôle |
| --- | --- | --- |
| Paramètres de communication collective | `coll`, `sendbuff`, `recvbuff`, `count`, `datatype`, `op`, `root` | Décrit « quoi faire » |
| Domaine de communication et stream | `comm`, `stream` | Décrit « où le faire » |
| Détails de l'algorithme | `chunkSteps`, `sliceSteps` | Décrit « comment découper » |
| Opérations unilatérales | `peerWinOffset`, `peerWin`, `sigIdx`, `ctx`, `flags`, `nDesc`, `signalDescs` | Spécifique à RMA |
| Configuration utilisateur | `collConfig` | Copie privée copiée depuis la config utilisateur |

Notez le commentaire de`collConfig`:**"A config copied from config passed by user so older user config can be safely accessed during synchronous host scheduling (never at launch/replay)"** [FACT:src/include/info.h:41-43]. C'est une conception clé — le pointeur de config passé par l'utilisateur peut être détruit avant`ncclGroupEnd`, donc NCCL en fait une copie dans`ncclInfo`.

## Step-by-Step : la chaîne d'appels de ncclAllReduce

Prenons`ncclAllReduce`comme exemple, et traçons le chemin complet de l'appel utilisateur jusqu'à la construction de`ncclInfo`.

**Étape 1 : l'utilisateur appelle ncclAllReduce.**L'entrée se trouve dans`src/collectives.cc`：

[FACT:src/collectives.cc:206-211]

Ici, trois choses sont faites :

1. `NVTX3_FUNC_WITH_PARAMS`poser un marqueur NVTX (pour la visualisation dans des outils comme Nsight)

2. appeler`ncclAllReduceConfigImpl`, en passant`config = nullptr`

3. retourner le résultat

**Étape 2 : ncclAllReduceConfigImpl construit ncclInfo.**C'est l'étape clé :

[FACT:src/collectives.cc:192-202]

Notez l'utilisation ici de l'initialisation agrégée de style C :

```c
struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                        sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
                        ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
```

Les champs correspondent un à un dans l'ordre de déclaration de`ncclInfo`.`ALLREDUCE_CHUNKSTEPS`et`ALLREDUCE_SLICESTEPS`sont définis dans`src/include/collectives.h`：

[FACT:src/include/collectives.h:19-20]

`NCCL_STEPS`est le nombre de pas dans le buffer circulaire (généralement 8 ou 16), donc le chunkSteps d'AllReduce est`NCCL_STEPS/2`, et sliceSteps est`NCCL_STEPS/4`. Cela signifie qu'un chunk contient 2 slices.

**Étape 3 : analyser la config utilisateur.** `ncclParseCollConfig`analyse le`ncclCollConfig_t*`passé par l'utilisateur dans`info.collConfig`. Si`config == nullptr`, ce champ reste initialisé à zéro.

**Étape 4 : transmettre à ncclEnqueueCheck.**C'est la véritable entrée du module enqueue.

## Réflexion de conception : pourquoi utiliser l'initialisation agrégée plutôt qu'une affectation champ par champ ?

> **[Design Inference & Architectural Trade-offs]**
> L'initialisation agrégée présente deux avantages : premièrement, le compilateur vérifie si le nombre de champs correspond (un champ manquant déclenche un avertissement), deuxièmement, le code est plus compact. Mais l'inconvénient est que**l'ordre des champs doit être strictement cohérent avec la déclaration de la structure**— si quelqu'un insère un champ au milieu de`ncclInfo`, tous les points d'initialisation agrégée seront silencieusement décalés. C'est un risque de maintenance implicite dans le code NCCL.

## Piège en production : cycle de vie de config

Un scénario de piège réel : l'utilisateur écrit ce code :

```c
ncclCollConfig_t config = {...};
ncclAllReduceConfig(..., &config);
// config 在这里被销毁（比如是栈变量，函数返回了）
```

Si NCCL ne copiait pas la config dans`ncclInfo`, alors lors de`ncclGroupEnd`, l'accès à`info.collConfig`lirait de la mémoire déjà libérée.`src/include/info.h:41-43`Le commentaire de**sert précisément à expliquer cette conception —**。

---

# la config est analysée et copiée dès la phase task append, et ne dépend plus ensuite du pointeur utilisateur

## II. ncclEnqueueCheck : validation des paramètres et sémantique de group

`ncclEnqueueCheck`Modèle intuitif**est la « vanne principale » du module enqueue. Toutes les API de communication collective convergent finalement ici. Sa responsabilité est :**valider la légalité des paramètres, gérer la sémantique de group, appeler taskAppend pour générer les tâches`ncclEnqueueCheck`。

. Si on le compare au contrôle de sécurité d'un aéroport, alors chaque fonction API est un comptoir d'enregistrement — l'enregistrement ne fait qu'accepter les bagages, le véritable contrôle de sécurité se trouve dans

## Étape par étape : le flux d'exécution de ncclEnqueueCheck

[FACT:src/enqueue/enqueue.cc:3478-3527]

Décomposons progressivement :

**Étape 1 : CommCheck valide le domaine de communication.** `CommCheck(info->comm, info->opName, "comm")`Vérifie si le pointeur comm est non nul et s'il a été initialisé. Si comm a été révoqué (par exemple, une erreur sur un rank), retourne directement une erreur :

[FACT:src/enqueue/enqueue.cc:3480-3485]

**Étape 2 : gérer la profondeur du profiler.**Si on est déjà à l'intérieur d'un group (`profilerGroupDepth > 0`), incrémente le compteur de profondeur. Cela sert à gérer correctement les appels implicites à`ncclGroupStartInternal`/`ncclGroupEndInternal`.

**Étape 3 : entrer dans le group interne.** `ncclGroupStartInternal()`est le mécanisme de group interne de NCCL.**Point clé**: même si l'utilisateur n'appelle pas explicitement`ncclGroupStart`, NCCL crée un group implicite pour chaque appel d'API. Cela garantit l'atomicité d'un appel unique.

**Étape 4 : s'assurer que comm est prêt.** `ncclCommEnsureReady(info->comm)`Attend que l'initialisation du domaine de communication soit terminée (par exemple, bootstrap terminé, connexions établies).

**Étape 5 : validation des paramètres par ArgsCheck.**C'est l'étape de validation la plus complexe :

[FACT:src/enqueue/enqueue.cc:3497-3503]

Attention au traitement de`checkMode`: s'il s'agit de`ncclCheckModeDebugGlobal`，`ArgsCheck`, info est mis en file d'attente, et la validation globale est effectuée au moment de`ncclGroupEnd`(par exemple, vérifier que les count de tous les ranks sont cohérents).

**Étape 6 : appeler taskAppend.**C'est l'étape de conversion centrale :

[FACT:src/enqueue/enqueue.cc:3513]

**Étape 7 : incrémenter opCount.**Après chaque mise en file d'attente réussie,`comm->opCount++`. Ce compteur sert à faire correspondre les opérations send/recv, et constitue aussi la base de la timeline du profiler.

**Étape 8 : quitter le group.** `ncclGroupEndInternal()`Si depth descend à 0, cela déclenche la véritable opération de group (ordonnancement, lancement du kernel).

## Contrôle de concurrence : sémantique de group et sûreté de thread

> **[Design Inference & Architectural Trade-offs]**
> `ncclGroupStartInternal`/`ncclGroupEndInternal`utilise le stockage local au thread (TLS) pour maintenir l'état du group. Cela signifie que**plusieurs appels d'API dans le même thread seront fusionnés en un seul group**, mais les appels de threads différents sont indépendants. C'est la base du support du multi-thread par NCCL.

Un piège facile à rencontrer : si l'utilisateur appelle une API CUDA non-NCCL entre`ncclGroupStart`et`ncclGroupEnd`(par exemple`cudaMemcpy`), cela peut provoquer des problèmes d'ordre des streams. Le mécanisme de group de NCCL suppose que les opérations du group se trouvent sur le même ensemble de streams.

## Chaîne de récupération d'erreur

`ncclEnqueueCheck`La gestion des erreurs de

[FACT:src/enqueue/enqueue.cc:3524-3526]

possède une conception ingénieuse :`taskAppend`Si`ncclCommSetAsyncError`échoue, et que comm est en mode non bloquant,

---

# est appelé pour enregistrer l'erreur. Ainsi, les appels d'API suivants retourneront immédiatement une erreur au lieu de continuer à essayer. C'est le mécanisme de propagation asynchrone des erreurs.

## III. taskAppend : le carrefour de la distribution des tâches

`taskAppend`Modèle intuitif`info->coll`est le « hub de trafic » du module enqueue. Selon la valeur de

, il distribue les tâches vers différents chemins de traitement : P2P, RMA, CE, ou communication collective ordinaire. C'est comme un centre de tri postal — selon l'adresse sur l'enveloppe, il dépose la lettre dans différentes boîtes aux lettres.

## Sans cette couche de distribution, tous les types d'opérations devraient s'entasser dans un énorme if-else, et le code serait difficile à maintenir.

[FACT:src/enqueue/enqueue.cc:3337-3476]

**Étape par étape : la logique de distribution de taskAppend** `ncclParamEnqueueRearchEnable()`Étape 1 : déterminer si la nouvelle architecture est activée.`rawTaskAppend`est un commutateur de variable d'environnement (0 par défaut). S'il est activé, on passe par le chemin

**— c'est le nouveau modèle de tâches en cours de développement par NCCL.**Étape 2 : distribution P2P.`p2pTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3343-3345]

**S'il s'agit de Send/Recv, appeler**Étape 3 : distribution RMA.`rmaTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3346-3347]

**S'il s'agit de PutSignal/Signal/WaitSignal, appeler** `if (info->count == 0) return ncclSuccess;`Étape 4 : retour anticipé pour communication collective vide.

**— une communication collective avec count à 0 est directement abandonnée.** `ncclCollConfigGetAlgMask`Étape 5 : validation du choix d'algorithme.

[FACT:src/enqueue/enqueue.cc:3357-3358]

**Valide si le choix d'algorithme fourni par l'utilisateur est légal :**Étape 6 : vérification du type FP8.

[FACT:src/enqueue/enqueue.cc:3360-3366]

**La réduction FP8 nécessite sm90+ :** `hostToDevRedOp`Étape 7 : conversion de l'opération de réduction.`ncclRedOp_t`Convertit le`ncclDevRedOpFull`：

[FACT:src/enqueue/enqueue.cc:3370-3371]

**côté host en**côté device.`comm->nRanks == 1`Étape 8 : retour anticipé pour un seul rank.`ncclLaunchOneRank`Si

[FACT:src/enqueue/enqueue.cc:3373-3377]

**, appeler directement**pour exécuter la réduction locale, sans générer de tâche :

[FACT:src/enqueue/enqueue.cc:3378-3470]

## Étape 9 : chemin multi-rank.

`collTaskAppend`C'est la branche la plus complexe, incluant le routage CE, la dégradation AllToAll/Gather/Scatter, ainsi que la communication collective ordinaire :`ncclTaskColl`Structure de données : les champs de ncclTaskColl

[FACT:src/enqueue/enqueue.cc:2757-2851]

est l'endroit où

| est généré. Regardons sa logique centrale : | Affectation des champs clés : | Champ |
| --- | --- | --- |
| `func` | `info->coll` | Source |
| `sendbuff`/`recvbuff` | `info->sendbuff`/`recvbuff` | Signification |
| `count` | `info->count` | Type de communication collective |
| `datatype` | `info->datatype` | Pointeur de buffer |
| `trafficBytes` | `count * elementSize * ncclFuncTrafficPerByte` | Nombre d'éléments |
| `opHost`/`opDev` | `info->op`/`opDev` | Type de données |
| `chunkSteps`/`sliceSteps` | `info->chunkSteps`/`sliceSteps` | Estimation du trafic |
| `minCTAs`/`maxCTAs`/`nvlsCTAs` | Opération de réduction | Nombre d'étapes de découpage |
| `algMask` | `ncclCollConfigGetAlgMask` | Analyse de configuration |

Limite de ressources`trafficBytes`Masque de sélection d'algorithme

[FACT:src/enqueue/enqueue.cc:2813]

`ncclFuncTrafficPerByte`Attention au calcul de

[FACT:src/enqueue/enqueue.cc:123-134]

:

## retourne le multiplicateur de trafic pour chaque type de communication collective :

[FACT:src/enqueue/enqueue.cc:2808-2812]

AllReduce retourne 2 (car il faut reduce + broadcast), AllGather/ReduceScatter retourne nRanks, les autres retournent 1.`ncclInt8`. C'est une optimisation :**Ces deux opérations n'impliquent pas de réduction, il n'est donc pas nécessaire de se soucier du type de données ; un traitement uniforme en octets permet de simplifier la logique du kernel**。

## Piège en production : l'ordre de résolution de CTAPolicy

[FACT:src/enqueue/enqueue.cc:3390-3397]

La résolution de CTAPolicy a une priorité subtile :**env > per-call > comm**. Et de plus`NCCL_CTA_POLICY_ZERO`est prioritaire sur`NCCL_CTA_POLICY_EFFICIENCY`. Si l'utilisateur définit ces deux indicateurs en même temps, ZERO prend effet.

Un scénario de piège réel : l'utilisateur a défini`NCCL_CTA_POLICY=EFFICIENCY`, mais a constaté que le chemin CE n'était pas utilisé. La raison est que le routage CE exige que`CTAPolicy & NCCL_CTA_POLICY_ZERO`soit vrai, or EFFICIENCY ne satisfait pas cette condition.

---

# IV. ncclPrepareTasks : de la liste de tâches à la file de planification

## Modèle intuitif

`ncclPrepareTasks`est le « préprocesseur » du module enqueue. Il répartit la liste de tâches en désordre dans des compartiments selon (func, op, datatype), puis calcule l'algorithme et le protocole pour chaque compartiment. C'est comme un bibliothécaire — il trie d'abord les livres rendus par catégorie, puis décide sur quelle étagère placer chaque catégorie de livres.

Sans cette étape, le`scheduleCollTasksToPlan`suivant devrait calculer l'algorithme séparément pour chaque tâche, ce qui serait extrêmement inefficace.

## Step-by-Step : la logique de compartimentage de ncclPrepareTasks

[FACT:src/enqueue/enqueue.cc:423-642]

**Étape 1 : conversion des tâches Broadcast.**S'il n'y a qu'un seul broadcast peer, convertir la tâche broadcast en tâche coll :

[FACT:src/enqueue/enqueue.cc:430-461]

Notez qu'ici on copie les champs de`bcastTask`vers le nouveau`ncclTaskColl`, et on calcule`trafficBytes`. Puis on libère la tâche originale depuis`memPool_ncclTaskBcast`.

**Étape 2 : compartimentage par (func, op, datatype).**Les tâches sortent du sorter par ordre décroissant de size, puis sont réparties dans le tableau`tasksByFnOpTy`:

[FACT:src/enqueue/enqueue.cc:464-487]

Calcul de l'indice :`((int)task->func * ncclNumDevRedOps + (int)task->opDev.op) * ncclNumTypes + (int)task->datatype`. C'est une linéarisation d'un tableau tridimensionnel.

**Étape 3 : agrégation et sélection d'algorithme.**Pour chaque compartiment, agréger les tâches de taille similaire (dans un facteur 4), puis appeler`ncclGetAlgoInfo`：

[FACT:src/enqueue/enqueue.cc:503-547]

**Étape 4 : compartimentage par (collnet, nvls).**Selon le type d'algorithme, répartir les tâches dans`collBins[2][2]`：

[FACT:src/enqueue/enqueue.cc:517-544]

**Étape 5 : concaténation de la file finale.**Concaténer les quatre compartiments en`planner->collTaskQueue`：

[FACT:src/enqueue/enqueue.cc:553-557]

## Structure de données : ncclTaskCollSorter

`ncclTaskCollSorter`est un trieur par insertion trié selon`trafficBytes`.`ncclTaskCollSorterInsert`insère la tâche à la bonne position,`ncclTaskCollSorterDequeueAll`retire toutes les tâches dans l'ordre.

> **[Design Inference & Architectural Trade-offs]**
> La motivation de conception de ce trieur est :**priorité de planification aux grosses tâches**. Comme les grosses tâches ont un temps de transfert long, les démarrer en premier permet de mieux superposer calcul et communication.

## Contrôle de concurrence : runtimeConn et établissement de connexion

[FACT:src/enqueue/enqueue.cc:572-583]

Si`comm->runtimeConn`est vrai (mode de connexion à l'exécution), et qu'un channel d'un algorithme n'est pas encore initialisé, alors marquer`algoNeedConnect`. Cela déclenchera l'établissement de la connexion par la suite.

## Piège en production : conditions aux limites de l'agrégation

[FACT:src/enqueue/enqueue.cc:507-508]

La condition d'agrégation est`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes`, et aucune des deux tâches ne définit`aggIsolate`. Si l'utilisateur définit une per-call config (par exemple`maxCTAs`），`aggIsolate`sera mis à true, cette tâche ne sera pas agrégée.

Un scénario de piège réel : l'utilisateur a défini pour un certain AllReduce`maxCTAs=4`, en s'attendant à ce qu'il n'utilise que 4 CTA. Mais à cause de la logique d'agrégation, cette tâche peut fusionner avec une tâche adjacente, entraînant un nombre de CTA réellement utilisé non conforme aux attentes. La solution est de définir`aggIsolate`— NCCL l'a déjà géré dans`collTaskAppend`:

[FACT:src/enqueue/enqueue.cc:2821-2822]

---

# V. scheduleCollTasksToPlan : découpage des channels et contrôle du budget

## Modèle intuitif

`scheduleCollTasksToPlan`est le « planificateur » du module enqueue. Il répartit les tâches sur des channels spécifiques et calcule le découpage des données pour chaque channel. C'est comme le système de planification de production d'une usine — il décide ce que fait chaque ligne de production et en quelle quantité.

Sans cette étape, le kernel GPU ne saurait pas quelle partie des données il doit traiter.

## Step-by-Step : algorithme de découpage des channels

[FACT:src/enqueue/enqueue.cc:644-947]

**Étape 1 : estimation du budget.**On estime d'abord le nombre de tâches pouvant entrer dans ce plan :

[FACT:src/enqueue/enqueue.cc:648-689]

`ncclTestBudget`Vérifier si le nombre d'octets de travail dépasse le budget :

[FACT:src/enqueue/enqueue.cc:343-349]

**Étape 2 : calcul du trafic de chaque channel.**Selon le kind (collnet/nvls), calculer`trafficPerChannel`：

[FACT:src/enqueue/enqueue.cc:701-707]

**Étape 3 : chemin Collnet.**S'il s'agit d'un algorithme collnet, l'allocation des channels est relativement simple :

[FACT:src/enqueue/enqueue.cc:709-739]

**Étape 4 : découpage en cell du chemin normal.**C'est la partie la plus complexe. NCCL découpe les données en « cell », chaque cell étant une unité de transfert minimale :

[FACT:src/enqueue/enqueue.cc:740-845]

Variables clés :

- `cellSize`: nombre d'octets par cell, au moins`MinTrafficPerChannel`（32KB）
- `cells`: nombre total de cells
- `cellsPerChannel`: nombre de cells traitées par chaque channel
- `cellsLo`/`cellsHi`: nombre de cells des channels de début et de fin (peut être incomplet)

**Étape 5 : calcul de chunkGrains.**Appeler pour chaque segment de channel`calcCollChunking`：

[FACT:src/enqueue/enqueue.cc:811-825]

**Étape 6 : génération des proxyOp.**Générer une opération proxy pour chaque channel :

[FACT:src/enqueue/enqueue.cc:844-894]

## Structure de données : ncclDevWorkColl

`ncclDevWorkColl`est le descripteur de travail côté device. Ses champs clés :

| Champ | Signification |
| --- | --- |
| `sendbuff`/`recvbuff` | Pointeur de buffer |
| `channelLo`/`channelHi` | Plage de channels |
| `cbd.countLo`/`countMid`/`countHi` | Nombre d'éléments par segment |
| `cbd.chunkGrainsLo`/`Mid`/`Hi` | Granularité de chunk par segment |
| `direct` | Indicateur direct |

## Contrôle de concurrence : opérations bit à bit sur channelMask

[FACT:src/enqueue/enqueue.cc:897]

Cette ligne de code définit channelMask par des opérations bit à bit :`(2ull << channelHi) - (1ull << channelLo)`. Par exemple channelLo=2, channelHi=5, le résultat est`(2<<5) - (1<<2) = 64 - 4 = 60 = 0b111100`, c'est-à-dire que les bits 2-5 sont définis.

## Piège en production : dépassement de budget

[FACT:src/enqueue/enqueue.cc:792-794]

Si le budget est insuffisant, retourner directement`ncclSuccess`, laisser la boucle externe créer un nouveau plan. C'est une stratégie de dégradation élégante——**pas d'erreur, juste un traitement par lots**。

Un scénario de piège réel : si`NCCL_WORK_FIFO_BYTES`est défini trop petit, chaque plan ne pourra contenir que très peu de tâches, augmentant le nombre de lancements de kernel et réduisant les performances.

---

# Six, finishPlan : des tâches aux paramètres du kernel

## Modèle intuitif

`finishPlan`est le "packageur" du module enqueue. Il empaquette les tâches, batchs et proxyOp en une structure de paramètres que le kernel peut lire directement. C'est comme l'emballage d'un colis——mettre les pièces détachées dans une boîte, coller le bordereau d'expédition, et attendre l'envoi.

## Étape par étape : la logique d'empaquetage de finishPlan

[FACT:src/enqueue/enqueue.cc:236-330]

**Étape 1 : décider du type de stockage.**Si tout le travail peut tenir dans kernel args, utiliser`ncclDevWorkStorageTypeArgs`：

[FACT:src/enqueue/enqueue.cc:244-250]

**Étape 2 : allouer kernelArgs.**Allouer depuis la pile mémoire :

[FACT:src/enqueue/enqueue.cc:251-255]

**Étape 3 : placement round-robin des batchs.**Le premier batch de chaque channel doit être placé dans`batchZero[blockIdx.x]`：

[FACT:src/enqueue/enqueue.cc:257-280]

**Étape 4 : fusionner les files proxyOp.**Tri par fusion selon opCount :

[FACT:src/enqueue/enqueue.cc:282-329]

## Structure de données : ncclDevKernelArgs

`ncclDevKernelArgs`est la structure de paramètres transmise au kernel. Elle contient :

- `comm`: communicateur côté device
- `channelMask`: masque de bits des channels
- `workStorageType`: type de stockage de travail
- `workBuf`: pointeur du buffer de travail
- `workMask`: masque du buffer de travail

## Piège en production : ordre des batchs

[FACT:src/enqueue/enqueue.cc:257-259]

Le commentaire est très clair : "The first batch for each channel must be located at batchZero[blockIdx.x]". Si cet ordre est incorrect, le kernel lira le mauvais batch, entraînant une corruption des données.

---

# Résumé du chapitre

Dans ce chapitre, nous avons suivi le chemin complet depuis`ncclAllReduce`jusqu'à`ncclTaskColl`:

1. **ncclAllReduce**construit`ncclInfo`, empaquette les paramètres utilisateur

2. **ncclEnqueueCheck**valide les paramètres, traite la sémantique de groupe

3. **taskAppend**distribue vers différents chemins selon le type d'opération

4. **collTaskAppend**génère`ncclTaskColl`, analyse la configuration

5. **ncclPrepareTasks**répartit par (func, op, datatype), calcule l'algorithme

6. **scheduleCollTasksToPlan**découpe les channels, génère`ncclDevWorkColl`

7. **finishPlan**empaquette en paramètres de kernel

Idées de conception clés :

- **Découplage par couches**: chaque fonction ne fait qu'une seule chose, transmet l'état via`ncclInfo`et`ncclTaskColl`
- **Contrôle du budget**: contrôle la taille de chaque plan via`ncclTestBudget`
- **Optimisation par agrégation**: les tâches de taille similaire sont agrégées, réduisant le nombre de lancements de kernel
- **Priorité de configuration**：env > per-call > comm

Dans le prochain chapitre, nous entrerons dans`task_sched`, pour voir comment NCCL orchestre l'ordre d'exécution multi-channel et multi-kernel.

# Réflexions et auto-évaluation de ce chapitre

Q1 : Si l'on supprime le test`collTaskAppend`dans`aggIsolate`(c'est-à-dire que`src/enqueue/enqueue.cc:2821-2822`retourne toujours false), dans quel scénario le`maxCTAs`défini par l'utilisateur deviendrait-il inopérant ? Pourquoi ?

**Analyse de référence**：`aggIsolate`sert à marquer "cette tâche ne peut pas être agrégée". Si l'on supprime ce test, les tâches avec une configuration per-call définie seraient fusionnées avec les tâches adjacentes. Dans`ncclPrepareTasks`la boucle d'agrégation de (`src/enqueue/enqueue.cc:507-508`), la condition d'agrégation est`aggEnd->trafficBytes < 4 * aggBeg->trafficBytes && !aggBeg->aggIsolate && !aggEnd->aggIsolate`. Si`aggIsolate`est toujours false, alors même si une tâche a défini`maxCTAs=4`, elle pourrait être fusionnée avec une tâche`maxCTAs=32`. Le`agg`fusionné prendra une certaine combinaison des deux (selon l'implémentation de`ncclGetAlgoInfo`), entraînant un nombre réel de CTA utilisés non conforme aux attentes de l'utilisateur.

Plus grave encore, dans`scheduleCollTasksToPlan`(`src/enqueue/enqueue.cc:665-666`），`taskAggIsolate`sert à garantir que les tâches avec des ressources per-call configurées occupent seules un plan. Si ce test devient inopérant, plusieurs tâches partageront le budget de channel du plan, entraînant une allocation de ressources non conforme aux attentes.

Q2 : Dans`ncclEnqueueCheck`, si`ncclGroupEndInternal()`retourne une erreur (par exemple l'échec de ArgsCheck d'un certain rank), mais que`taskAppend`a déjà été exécuté avec succès, que se passe-t-il ? Comment NCCL garantit-il la cohérence d'état ?

**Analyse de référence**: regarder le flux de contrôle de`src/enqueue/enqueue.cc:3513-3519`:

```c
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
info->comm->opCount++;
exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
```

Si`taskAppend`réussit mais que`ncclGroupEndInternal`échoue,`opCount`a déjà été incrémenté. Cela entraînera une inadéquation de l'opCount des opérations suivantes avec le pair, pouvant déclencher un hang.

La façon dont NCCL gère cela est :`ncclGroupErrCheck(ret)`vérifie s'il y a une erreur, et si oui, définit l'état d'erreur de comm. Les appels API suivants détecteront cette erreur via`ncclCommGetAsyncError`et retourneront immédiatement. C'est une stratégie de "échec rapide"——une fois qu'une erreur survient, tout le comm entre en état d'erreur et ne tente plus de récupération.

En environnement de production, cela signifie qu'une fois qu'une erreur de groupe survient, l'utilisateur doit détruire et reconstruire le communicateur.

Q3: `scheduleCollTasksToPlan`L'algorithme de découpage en cellules dans`src/enqueue/enqueue.cc:740-845`(`cellsLo == 0`) a une condition limite : lorsque`channelId`, il saute le moins de channels. Si cette logique de saut a un bug (par exemple

**n'est pas correctement incrémenté), quelles en seraient les conséquences ?**Analyse de référence`src/enqueue/enqueue.cc:770-780`：

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

Copier`channelId`Si

1. **n'est pas correctement incrémenté, la tâche suivante commencera son allocation à partir d'un mauvais channel. Cela entraînera :**Chevauchement de channels

2. **: deux tâches pourraient être allouées au même segment de données du même channel**Corruption de données

3. **: le kernel traitera les données en double ou en omettra**: déséquilibre de charge des channels

Plus insidieux encore, ce bug peut ne se déclencher que pour une taille de message spécifique (lorsque`cellsLo == 0`), ce qui le rend difficile à reproduire. NCCL suit les channels déjà utilisés via`plan->channelMask |= (2ull << devWork->channelHi) - (1ull << devWork->channelLo)`, mais ce n'est qu'un enregistrement, cela n'empêche pas les chevauchements.

Jusqu'ici, nous avons vu comment ncclAllReduce passe d'un appel utilisateur à une série de tâches kernel exécutables : validation des paramètres, détermination de l'algorithme/du protocole, découpage en channels, génération finale de ncclInfo et ncclTaskColl. Mais la création des tâches n'est que la première étape — elles doivent encore être ordonnancées sur plusieurs channels, générer les paramètres de lancement des kernels, et gérer la soumission par lots et le tri des dépendances dans la sémantique de groupe. Le chapitre suivant approfondira src/enqueue/task_sched et src/enqueue/task_prep, pour répondre à « pourquoi un seul AllReduce lance-t-il plusieurs kernels, et comment l'ordre et les dépendances entre eux sont-ils garantis », tout en révélant comment ncclGroupStart/ncclGroupEnd dans src/group.cc fusionnent plusieurs appels API en une seule soumission.
