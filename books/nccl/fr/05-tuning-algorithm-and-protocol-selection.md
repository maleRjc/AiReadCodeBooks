# Chapitre 5 : Sélection des algorithmes et protocoles : comment le module tuning décide du chemin de communication

Dans le chapitre précédent, nous avons décomposé la capacité de perception topologique de NCCL : de l'énumération des dispositifs dans src/graph/topo.cc pour construire le graphe topologique, à la recherche du chemin optimal dans src/graph/search.cc, puis à la concrétisation des résultats de recherche en topologies d'algorithmes Ring et Tree dans rings.cc et trees.cc. Mais le graphe topologique ne répond qu'à la question « par où les données peuvent passer » ; il ne répond pas à « par où cette communication devrait passer ». Sur une même machine, un AllReduce de 4 Ko et un AllReduce de 400 Mo peuvent avoir des solutions optimales complètement différentes : le premier privilégie la latence, le second la bande passante ; le premier peut choisir Tree/LL, le second peut choisir Ring/Simple ou NVLS. Le module tuning est celui qui « tranche ». Ses entrées sont la taille du message, le nombre de ranks, le graphe topologique (produit du chapitre précédent) et les variables d'environnement utilisateur ; sa sortie est un ncclTuningResult_t, qui indique quel algorithme (algo) utiliser, quel protocole (proto), combien de channels ouvrir et combien de warps utiliser. Dans ce chapitre, nous décomposons le répertoire src/tuning dans l'ordre suivant : « ordonnancement global → modèle de coût → estimation de chaque algorithme → décision finale ». La question centrale est unique : comment NCCL, parmi des dizaines de combinaisons (algorithme, protocole), utilise un modèle mathématique purement CPU pour sélectionner la plus rapide en quelques microsecondes ?

# I. tuning.cc : ordonnancement global et squelette décisionnel

## Modèle intuitif

Imaginez le module tuning comme une entreprise de**déménagement**. Le client (une communication collective) arrive et dit « je veux déménager 100 Mo de marchandises, de 8 entrepôts vers 8 entrepôts ». Le dispatcheur (`ncclTuningCompute`) ne va pas réellement faire le déménagement pour essayer, mais sort une**grille tarifaire**(modèle de coût), estime un « temps prévu » pour chaque option (Ring/LL, Tree/Simple, NVLS/Simple…), puis choisit le devis le plus court pour le client.

Sans ce dispatcheur, NCCL ne pourrait que coder en dur « AllReduce utilise toujours Ring », ce qui serait écrasé par Tree dans les scénarios de petits messages et par NVLS dans les scénarios NVLink à grande échelle.**Le coût serait une performance réduite de moitié, voire pire, dans des scénarios spécifiques.**

## Structures de données et disposition mémoire

Le support de la décision est`ncclTuningResult_t`, et l'ensemble des candidats est`ncclTuningResultList_t`(une liste simplement chaînée). Les nœuds de la liste sont définis dans`tuning_int.h`, mais la logique de push se trouve dans`tuning.cc`:

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
> Notez qu'il s'agit ici d'une**insertion en tête**: chaque candidat valide calculé est inséré en tête de liste. Cela signifie que l'ordre de la liste et l'ordre des id sont**inversés**. Pourquoi utiliser une liste chaînée plutôt qu'un tableau ? Parce que le nombre de candidats est déterminé à la compilation par`NCCL_TUNING_COUNT`, mais les candidats réellement valides sont dynamiques (influencés par`tuningMask`, les capacités de la plateforme, les variables d'environnement utilisateur) ; la liste chaînée permet de « n'attacher que les valides », évitant de vérifier répétitivement`valid`lors du parcours. Le coût est que chaque décision nécessite`ncclCalloc`une fois, mais le tuning se produit sur le chemin de mise en file, à une fréquence peu élevée, ce coût d'allocation est donc acceptable.

`ncclTuningResult_t`Les deux champs les plus critiques sont`timeUs`(temps estimé, en microsecondes) et`selectionTimeUs`(temps utilisé pour la sélection, pouvant être écrasé par le plugin tuner). La logique de sélection ne regarde que ce dernier :

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

Il y a un détail ici :`bestTuning->timeUs`est d'abord initialisé à`FLT_MAX`, puis on parcourt. Si la liste chaînée est vide (tous les candidats sont invalides),`bestTuning`conservera`NCCL_TUNING_RESULT_INIT`la valeur initiale, algo/proto étant tous deux`UNDEF`. Ce « résultat vide » sera traité spécialement par l'appelant — voir la branche d'erreur plus loin.

## Step-by-Step Walkthrough : le flux de décision d'un AllReduce

Supposons que l'application appelle`ncclAllReduce`, message de 1 Mo, 8 ranks sur une seule machine NVLink. Suivons`ncclTuningCompute`pas à pas.

**Étape 0 : court-circuit mono-rank.**Si`nRanks <= 1`, aucune communication n'est nécessaire, on retourne directement Ring/Simple, avec le nombre de channels fixé à 0 :

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

copie`NCCL_TUNING_IGNORE`Ici

**est une valeur sentinelle indiquant « cette combinaison n'a pas été calculée / n'est pas applicable ». Le plugin peut ne modifier que les cases qui l'intéressent, les autres restant à IGNORE, et NCCL les ignorera.**Étape 4 : choisir le meilleur.`ncclTuningSelectBestTuning`appelle`selectionTimeUs`, parcourt la liste chaînée et prend celle dont

**est minimale.**Étape 5 : calculer le nombre de channels.

[FACT:src/tuning/tuning.cc:233-235]

```c
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
```

`ncclTuningGetChannels`copie`tuning_int.h`Dans`minChannels`, la logique consiste à interpoler entre`maxChannels`et

**selon la taille du message et le type d'algorithme. Le nombre de channels influence directement la bande passante : plus il y a de channels, plus le parallélisme est élevé, mais plus le coût de démarrage de chaque channel est important.**Étape 6 : biais CTA Policy (priorité NVLS).`NCCL_CTA_POLICY_EFFICIENCY`Si l'utilisateur a défini

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

**Pourquoi distinguer les codes d'erreur ?**Si l'utilisateur a défini`NCCL_ALGO=ring`mais que la plateforme actuelle ne supporte pas ring (par exemple certaines topologies spéciales), c'est une**erreur de configuration utilisateur**（`ncclInvalidUsage`) ; si l'utilisateur n'a défini aucune variable d'environnement et qu'aucun algorithme ne peut être choisi, c'est un**bug interne de NCCL**（`ncclInternalError`). Cette distinction est essentielle pour le dépannage.

## Diagramme du flux principal de décision

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

# II. cost_model.cc : registre de modèles et matrice d'activation

## Modèle intuitif

`cost_model.cc`est le**grand livre**du tuning. Il maintient une table`modelMap`, chaque ligne correspondant à une combinaison (algo, proto), enregistrant « quelle est la fonction d'initialisation de cette combinaison, quelle est la fonction de simulation, et pour quelles fonctions elle est activée ». Il est également chargé de parser la variable d'environnement utilisateur`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SYM_KERNEL`, traduisant l'intention de l'utilisateur en une matrice d'activation`enabled[i][f]`.

Sans cette table, chaque ajout d'un nouvel algorithme obligerait à modifier tout le flux principal de tuning, et le code deviendrait un vrai gâchis.**L'approche pilotée par table**transforme « ajouter un algorithme » en « ajouter une ligne ».

## Structures de données : modelMap et matrice d'activation

`modelMap`est un tableau statique, chaque élément étant`ncclTuningModelEntry_t`：

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

Chaque entrée possède quatre champs :`init`(initialisation, calcule latency/bandwidth et les stocke dans comm),`model`(simulation, calcule le timeUs final selon la taille du message),`finalize`(nettoyage),`enabled[5]`(pour savoir si les cinq fonctions Broadcast/Reduce/AllGather/ReduceScatter/AllReduce sont activées).

Attention`enabled`L'ordre du tableau est commenté à la L234 :`Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce`. Cet ordre doit être cohérent avec`ncclFunc_t`l'énumération, sinon il y aura confusion.

> **[Design Inference & Architectural Trade-offs]**
> **Pourquoi séparer init et sim ?**Parce que ce qui est calculé dans init (latency, bandwidth)**ne dépend que des propriétés statiques de comm**(topologie, nombre de ranks, compCap), et est indépendant de la taille concrète des messages. Dans une communication, tuning peut être appelé plusieurs fois consécutivement (par exemple s'il y a plusieurs op dans un group), init ne s'exécute qu'une fois, sim s'exécute à chaque fois. C'est une optimisation typique de « précalcul + requête rapide ».

## Step-by-Step : analyse des variables d'environnement et construction de la matrice d'activation

**Étape 1 : tout activé par défaut, LL128 particulier.** `ncclTuningCostModelInit`Au début, tous les proto sont mis à 1 (activé), mais LL128 est mis à 2 :

[FACT:src/tuning/cost_model.cc:313-323]

```c
  for (int f = 0; f minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                          comm->graphs[algo].typeIntra, comm->nRanks, f, algo, comm->minDriverVersion)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
```

**Étape 2 : analyse des variables d'environnement utilisateur.**Si l'utilisateur a défini`NCCL_ALGO`ou`NCCL_SYM_KERNEL`, on remet d'abord algo et symKernel à zéro (car l'utilisateur a spécifié une liste blanche) :

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

Attention, proto n'est pas remis à zéro — car la valeur par défaut de proto est 1/2, quand l'utilisateur définit`NCCL_PROTO=LL`,`parseList`mettra LL à 1 et les autres à 0 (à cause de la logique`unset`). Cette asymétrie est intentionnelle : algo est entièrement activé par défaut mais doit être restreint après spécification par l'utilisateur, la restriction de proto est gérée en interne par`parseList`.

**Étape 3 : syntaxe de parseList.**Cette fonction supporte une syntaxe assez complexe, des exemples sont donnés dans les commentaires :

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

`^`Le préfixe indique « négation » :

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

Donc`NCCL_PROTO="^LL128;allreduce:LL128"`signifie : désactiver LL128 globalement, mais activer LL128 en exception pour AllReduce.

**Étape 4 : fusion de la matrice enabled.**Enfin, on parcourt tous les model, et on effectue un ET logique entre`model->enabled[f]`et les interrupteurs utilisateur :

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

La logique est :**Ce n'est que lorsque l'utilisateur a défini une configuration forced pour une fonction que la configuration utilisateur écrase la valeur par défaut du modèle**. Si l'utilisateur n'a rien défini,`forced[f] == 0`, directement`continue`, on conserve le`enabled`propre au modèle. C'est la priorité « spécification explicite de l'utilisateur > valeur par défaut du modèle ».

## Point d'entrée unifié de la simulation de modèle

Tous les modèles finissent par être appelés via`ncclTuningCostModelSimModel`:

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

Triple filtrage :**id hors limites → modèle désactivé → le modèle renvoie un temps non positif**, si une couche ne passe pas, on va dans`not_valid`, on met`timeUs`à`NCCL_TUNING_IGNORE`(une sentinelle négative),`valid = 0`. L'appelant, en voyant`valid == 0`, ne l'insérera pas dans la liste chaînée de candidats.

## Réflexion de conception

`modelMap`Il y a un avertissement clé dans les commentaires de

[FACT:src/tuning/cost_model.cc:229]

```c
// IMPORTANT: this table need must be consistent with the algRegistry in src/config/algorithm_registry.cc
```

> **[Design Inference & Architectural Trade-offs]**
> Cela signifie que`modelMap`l'ordre des**indices**doit être strictement cohérent avec l'ordre d'enregistrement des algorithmes dans`algorithm_registry.cc`. Si quelqu'un insère un nouvel algorithme dans le registry mais oublie de modifier`modelMap`, tous les id seront décalés, et tuning choisira un algorithme complètement erroné.**C'est le piège classique de la conception pilotée par table : le contrat implicite.**Une approche plus robuste serait d'utiliser le nom de l'énumération comme key plutôt que l'indice, mais cela sacrifierait un peu d'optimisation à la compilation.

---

# III. ring.cc : estimation du coût de l'algorithme Ring

## Modèle intuitif

L'algorithme Ring dispose N ranks en anneau, les données circulent le long de l'anneau tour après tour. Son modèle de coût doit répondre à deux questions :**Combien de données sont transmises à chaque étape (bandwidth)**、**Combien d'étapes au total (latency)**。

L'intuition de Ring est «**pipeline**» : imaginez N personnes debout en cercle qui se passent un seau d'eau, chaque personne, après avoir reçu le seau, verse un peu d'eau puis le passe à la suivante. Le seau fait un tour, et l'eau de tout le monde est bien mélangée. Plus le seau tourne vite (bandwidth élevée), plus le cercle est petit (moins d'étapes), plus l'ensemble est rapide.

## Structure de données : table latency/bandwidth

Le modèle Ring n'introduit pas de nouvelle structure, il écrit les résultats d'estimation dans`comm->tuningContext.generalLatencies[c][algo][proto]`et`generalBandwidths[c][algo][proto]`. Ce sont deux tableaux tridimensionnels : fonction × algorithme × protocole.

À l'initialisation, tout est d'abord mis à -1.0 (sentinelle, signifiant « non calculé ») :

[FACT:src/tuning/ring.cc:31-33]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
```

Cette sentinelle -1.0 est vérifiée à l'étape sim :

[FACT:src/tuning/ring.cc:94-97]

```c
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ncclSuccess;
  }
```

**Pourquoi utiliser -1.0 plutôt que 0 ?**Parce que 0 est une valeur de bandwidth légale (bien que physiquement impossible), tandis que -1.0 indique clairement « non initialisé ». La comparaison flottante avec`==`est sûre ici, car -1.0 est exactement représentable.

## Step-by-Step : estimation de la bandwidth de Ring

**Étape 1 : déterminer si l'on utilise la bandwidth intra ou inter.**Mono-machine (nNodes==1) utilise intra, multi-machine utilise inter :

[FACT:src/tuning/ring.cc:34-37]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float bw = (comm->nNodes == 1 || (comm->nNodes minCompCap graphs[algo].bwIntra :
                                                                                      comm->graphs[algo].bwInter;
    float busBw = bw * comm->graphs[algo].nChannels;
```

`nSteps`est le nombre d'étapes nécessaires à l'algorithme, pour Ring, AllReduce est`2*(nRanks-1)`, les autres sont`nRanks-1`。`busBw`est la « bandwidth de bus » = bandwidth d'un lien unique × nombre de channels.

**Étape 2 : appliquer la réduction selon le protocole.**Le protocole LL n'utilise que la moitié de la bande passante (à cause de l'overhead des flags LL), LL128 en utilise 92% (120/128) :

[FACT:src/tuning/ring.cc:38-42]

```c
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);
```

`0.92 = 120/128`C'est parce que dans LL128, 8 octets sur 128 sont des flags, la charge utile n'est que de 120 octets. Ce chiffre provient directement de la conception du protocole.

**Étape 3 : Calculer la bande passante effective.**Notez qu'ici on multiplie par`nRanks / nSteps`：

[FACT:src/tuning/ring.cc:44-46]

```c
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;
```

**Pourquoi multiplier par`nRanks / nSteps`？**C'est la caractéristique centrale de l'algorithme Ring : la quantité de données réellement transportée par chaque rank est`nBytes * nSteps / nRanks`(car les données doivent faire plusieurs tours de l'anneau). Donc « bande passante effective » = bande passante du bus × nRanks / nSteps. Pour AllReduce, nSteps = 2(nRanks-1), donc bande passante effective ≈ busBw/2.

**Étape 4 : Calculer la latence.**La latence se divise en deux parties : intra et inter :

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

Notez le traitement spécial des lignes L57-58 : lorsque`maxLocalRanks == 1`(chaque nœud n'a qu'un seul rank), la latence inter-node de Ring utilise**la latence NET de Tree**. Le commentaire dit qu'il s'agit de « preserve the pre-refactor model » — c'est-à-dire une « bizarrerie » délibérément conservée pour maintenir la cohérence avec le comportement d'avant le refactoring.**Ce genre de bagage historique est très courant dans les systèmes matures. Quand vous lisez du code source et voyez le mot « preserve », soyez particulièrement prudent : cela signifie souvent qu'il y a ici une contrainte de compatibilité qu'on ne peut pas toucher.**

**Étape 5 : Accumuler selon le type de fonction.**Les modèles de latence de Reduce/Broadcast et AllReduce/AllGather/ReduceScatter sont différents :

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

`sameChannels`est une propriété topologique, indiquant si « les étapes intra et inter sur l'anneau utilisent le même ensemble de channels ». Si ce n'est pas le cas, la latence doit être multipliée par`nSteps`(il faut attendre à chaque étape).`netOverhead`est l'overhead de post réseau ; le protocole Simple doit multiplier par 3 (car Simple a trois allers-retours réseau : send, recv, ack).

## Pièges en production : l'effet plateau de Ring/Simple

`ncclTuningRingModelSim`Il y a une section de code dédiée au traitement du « plateau » :

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
> **Qu'est-ce qu'un plateau ?**Dans Ring/Simple, lorsque le message atteint une certaine taille, la latence ne croît plus linéairement avec le message, mais se « bloque » sur un palier — car à ce moment le goulot d'étranglement passe de « l'overhead de démarrage » à « la bande passante », et la bande passante est déjà saturée. Ce phénomène est particulièrement marqué sur Blackwell NVLink (car la bande passante NVLink est très élevée, la part de latence est plus grande). Le code utilise`plateauFactor`(1.4 ou 1.9) multiplié à la latence pour simuler cet effet de « latence amplifiée ».

`bytesPerRankPerChannel >= 64`est la condition de déclenchement : chaque rank doit transmettre au moins 64 octets par channel, sinon le plateau ne se produit pas. Ces 64 octets proviennent de la taille des flags du protocole LL.

**Scénario piège**: Si vous exécutez un AllReduce de 1MB sur Blackwell et constatez que la latence réelle est 40% supérieure à la prédiction du modèle, ne croyez pas à un bug — c'est l'effet plateau, et le modèle l'a déjà pris en compte. Si vous modifiez manuellement à la baisse`plateauFactor`, le modèle sous-estimera la latence, ce qui conduira à un mauvais choix d'algorithme.

---

# IV. tree.cc et nvls.cc : estimation du coût de Tree et NVLS

## Modèle intuitif

**L'algorithme Tree**est une «**diffusion en arbre**» : le nœud racine distribue les données aux nœuds enfants, qui les distribuent à leur tour aux nœuds petits-enfants. Son avantage est le**faible nombre d'étapes**(log N au lieu de N), adapté aux petits messages ; son inconvénient est la**faible utilisation de la bande passante**(chaque nœud non-feuille doit relayer, la bande passante effective réelle n'est que de la moitié).

**NVLS**(NVLink SHARP) est la «**multidiffusion matérielle**» : le switch copie directement les données vers plusieurs GPU, sans relais logiciel. Son avantage est une**bande passante élevée et une faible latence**, mais il nécessite un matériel spécifique (Hopper ou supérieur) et une configuration spécifique.

## Modèle Tree : ne sert qu'à AllReduce

Le modèle Tree a une limitation stricte —**il n'est activé que pour AllReduce**：

[FACT:src/tuning/tree.cc:21-27]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
```

> **[Design Inference & Architectural Trade-offs]**
> **Pourquoi ?**Parce que l'implémentation Tree de NCCL ne supporte qu'AllReduce (les autres opérations collectives n'ont pas de version Tree). C'est une contrainte d'implémentation, pas une limitation théorique.`enabled[c] = 0`est un « désactivation stricte », plus radical que`generalBandwidths = -1`— le premier fait directement retourner`ncclTuningCostModelSimModel`à L480, le second ne vérifie qu'au moment de la fonction sim.`not_valid`Estimation de la bande passante de Tree

**Copier**：

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
> , plus agressif que celui de Ring`1/3.8`.`0.5`Pourquoi l'efficacité LL de Tree est-elle plus faible ?**Parce que chaque nœud intermédiaire de Tree doit à la fois recevoir et envoyer, l'overhead des flags LL est amplifié sous trafic bidirectionnel.**Ce chiffre provient de mesures réelles.`1/3.8`Estimation de la latence de Tree

**Copier**：

[FACT:src/tuning/tree.cc:55-58]

```c
    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
```

`2 *`est le nombre d'étapes intra-node (nombre de ranks par nœud moins un),`(nRanks/nNodes - 1)`est le nombre d'étapes inter-node (hauteur de l'arbre).`log2i(nNodes)`Facteur de correction de Tree

**Tree 的修正因子**: Le modèle Tree est multiplié par un facteur lors de la phase sim`treeCorrectionFactor`：

[FACT:src/tuning/tree.cc:75-79]

```c
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize proto][logSize];
```

`treeCorrectionFactor`est une table de 3×24 :

[FACT:src/tuning/cost_model.cc:223-227]

```c
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};
```

`logSize = log2(nBytes >> 6)`, c'est-à-dire que la taille du message est prise en log2 avec une unité de 64 octets. Les indices 0-23 de la table correspondent à 64B jusqu'à 64B×2^23 ≈ 512MB.**Cette table est la « courbe d'efficacité Tree » mesurée empiriquement**: pour les petits messages, l'efficacité est de 1.0 (dominée par la latence), pour les messages moyens, l'efficacité chute à 0.4-0.5 (la bande passante n'est pas saturée), et pour les grands messages, elle remonte à 1.0 (bande passante saturée). Ce « creux intermédiaire » est une caractéristique inhérente à l'algorithme Tree.

## Modèle NVLS : le coût du multicast matériel

Le modèle NVLS vérifie d'abord si le matériel supporte :

[FACT:src/tuning/nvls.cc:19-24]

```c
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!ncclNvlsTransportEnabled(comm)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

Ensuite, une série de contraintes strictes : seul le protocole Simple est supporté, NVLSTree n'est pas supporté sur une seule machine, et NVLS multi-machine nécessite CollNet :

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

**Estimation de la bande passante NVLS**utilise un facteur d'efficacité :

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
> Hopper est à 0.85, tandis que Blackwell descend à 0.74.**Pourquoi l'efficacité est-elle plus faible sur la nouvelle génération de matériel ?**Parce que la bande passante NVLink de Blackwell est plus élevée, mais la capacité de traitement des switches NVLS n'a pas augmenté proportionnellement, ce qui entraîne une baisse de l'efficacité relative. Ce chiffre est mesuré empiriquement, pas une valeur théorique.

Dans le calcul de la bande passante, il y a un facteur`(nChannels - 1) / nChannels`:

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

`(nChannels - 1) / nChannels`car NVLS doit réserver un channel pour la synchronisation.`(ppn - 1) / ppn`est le surcoût de AllGather/ReduceScatter (chaque rank doit attendre les données du rank précédent).

## Pièges en production : les contraintes strictes de NVLS

Le modèle NVLS dispose également d'une couche de vérification à l'exécution lors de la phase sim :

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

`NCCL_MAX_NVLS_ARITY`est le nombre maximum de GPU que le groupe multicast NVLS peut contenir. Si ce nombre est dépassé, NVLS n'est pas disponible.**Scénario piège**: dans un domaine NVLink de 16 cartes exécutant AllGather, si`NCCL_MAX_NVLS_ARITY`vaut 8, NVLS sera désactivé et le tuning reviendra à Ring. Si vous ne connaissez pas cette limitation, vous vous demanderez « pourquoi NVLS n'est pas utilisé alors que le matériel le supporte ».

---

# V. Repli des kernels symétriques et chaîne de récupération d'erreurs

## Modèle intuitif

Le kernel symétrique (symmetric kernel) est une nouvelle fonctionnalité de NCCL : lorsque les buffers de tous les ranks sont enregistrés dans la mémoire symétrique, le kernel peut accéder à la mémoire distante avec des instructions plus efficaces. Mais**si le buffer n'est pas enregistré, ou si la plateforme ne le supporte pas, il faut revenir à un kernel ordinaire**. Cette logique de repli est la partie la plus complexe du tuning.

## Étape par étape : décision de repli

La logique de repli se trouve dans`tuning.cc:258-298`. Décomposons-la.

**Étape 1 : déterminer si un repli est nécessaire.**Condition d'entrée :

[FACT:src/tuning/tuning.cc:258-263]

À ce stade, la chaîne de décision du module tuning est claire : il reçoit la topologie et les paramètres de communication, et via le modèle de coût et l'estimation d'algorithmes, produit en quelques microsecondes la combinaison optimale (algorithme, protocole, channel, warp). Mais la sélection n'est que le début — comment ce résultat de décision est-il utilisé en aval ? Dans le chapitre suivant, nous entrerons dans le corps de src/enqueue/enqueue.cc pour voir comment un appel ncclAllReduce passe par la validation des paramètres, la détermination algorithme/protocole, le découpage en channels, et génère finalement les structures ncclInfo et ncclTaskColl. C'est le chapitre clé où le livre passe du « point de vue utilisateur » au « point de vue moteur » ; vous découvrirez ce qu'un appel de communication collective est traduit côté host, ainsi que la frontière avec le lancement du kernel qui suit.
