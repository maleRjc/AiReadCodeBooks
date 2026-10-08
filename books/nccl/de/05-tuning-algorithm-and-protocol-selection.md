# Kapitel 5: Algorithmus- und Protokollauswahl: Wie das tuning-Modul den Kommunikationspfad bestimmt

Im vorherigen Kapitel haben wir die Topologiebewusstheit von NCCL analysiert: von der Enumeration der Geräte in src/graph/topo.cc zum Aufbau des Topologiegraphen, über die Suche nach optimalen Pfaden in src/graph/search.cc bis hin zur Konkretisierung der Suchergebnisse in Ring- und Tree-Algorithmustopologien durch rings.cc und trees.cc. Der Topologiegraph beantwortet jedoch nur die Frage „Welchen Weg können die Daten nehmen“, nicht „Welchen Weg sollte diese Kommunikation nehmen“. Auf derselben Maschine können ein 4KB-AllReduce und ein 400MB-AllReduce völlig unterschiedliche optimale Lösungen haben: Ersterer zielt auf Latenz ab, Letzterer auf Bandbreite; Ersterer könnte Tree/LL wählen, Letzterer Ring/Simple oder NVLS. Das tuning-Modul ist derjenige, der die Entscheidung trifft. Seine Eingaben sind Nachrichtengröße, Anzahl der Ranks, Topologiegraph (das Produkt des vorherigen Kapitels) und Benutzerumgebungsvariablen; seine Ausgabe ist ein ncclTuningResult_t, der angibt, welcher Algorithmus (algo), welches Protokoll (proto), wie viele Channels und wie viele Warps verwendet werden. In diesem Kapitel zerlegen wir das Verzeichnis src/tuning in der Reihenfolge „Gesamtsteuerung → Kostenmodell → Schätzung der einzelnen Algorithmen → abschließende Entscheidung“. Die Kernfrage ist nur eine: Wie wählt NCCL aus Dutzenden von (Algorithmus, Protokoll)-Kombinationen mit einem rein CPU-basierten mathematischen Modell in Mikrosekunden die schnellste aus?

# I. tuning.cc: Gesamtsteuerung und Entscheidungsrückgrat

## Intuitives Modell

Stellen Sie sich das tuning-Modul als ein**Umzugsunternehmen**vor. Ein Kunde (eine kollektive Kommunikation) kommt und sagt: „Ich möchte 100MB Fracht von 8 Lagern zu 8 Lagern transportieren“. Der Disponent (`ncclTuningCompute`) wird nicht wirklich losziehen und es ausprobieren, sondern eine**Preisliste**(Kostenmodell) herausholen, für jede Option (Ring/LL, Tree/Simple, NVLS/Simple …) eine „geschätzte Dauer“ berechnen und dann das günstigste Angebot für den Kunden auswählen.

Ohne diesen Disponenten könnte NCCL nur fest verdrahten „AllReduce verwendet immer Ring“, was bei kleinen Nachrichten von Tree und bei großen NVLink-Szenarien von NVLS übertrumpft würde.**Der Preis dafür ist, dass die Leistung in bestimmten Szenarien halbiert oder sogar schlechter wird.**

## Datenstrukturen und Speicherlayout

Der Träger der Entscheidung ist`ncclTuningResult_t`, die Kandidatenmenge ist`ncclTuningResultList_t`(eine einfach verkettete Liste). Die Listenknoten sind definiert in`tuning_int.h`, aber die push-Logik befindet sich in`tuning.cc`:

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
> Beachten Sie, dass hier**Kopf-Einfügung**verwendet wird: Jedes Mal, wenn ein gültiger Kandidat berechnet wird, wird er an den Kopf der Liste eingefügt. Das bedeutet, dass die Listenreihenfolge und die id-Reihenfolge**umgekehrt**sind. Warum eine verkettete Liste statt eines Arrays? Weil die Anzahl der Kandidaten zur Kompilierzeit durch`NCCL_TUNING_COUNT`bestimmt wird, aber die tatsächlich gültigen Kandidaten dynamisch sind (beeinflusst durch`tuningMask`, Plattformfähigkeiten, Benutzerumgebungsvariablen). Eine verkettete Liste ermöglicht es, „nur die gültigen einzuhängen“, wodurch wiederholte Überprüfungen während der Iteration vermieden werden`valid`. Der Preis dafür ist, dass bei jeder Entscheidung`ncclCalloc`einmal, aber das Tuning findet auf dem Enqueue-Pfad statt und die Frequenz ist niedrig, sodass dieser Allokationsaufwand akzeptabel ist.

`ncclTuningResult_t`Die beiden wichtigsten Felder in`timeUs`(geschätzte Dauer, Mikrosekunden) und`selectionTimeUs`(für die Auswahl verwendete Dauer, kann vom Tuner-Plugin überschrieben werden). Die Auswahllogik betrachtet nur letzteres:

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

Hier gibt es ein Detail:`bestTuning->timeUs`wird zuerst auf`FLT_MAX`gesetzt und dann durchlaufen. Wenn die verknüpfte Liste leer ist (alle Kandidaten ungültig),`bestTuning`behält`NCCL_TUNING_RESULT_INIT`den Initialwert, algo/proto sind beide`UNDEF`. Dieses „leere Ergebnis“ wird beim Aufrufer speziell behandelt – siehe den späteren Fehlerzweig.

## Step-by-Step Walkthrough: Der Entscheidungsfluss eines AllReduce

Angenommen, die Anwendung ruft`ncclAllReduce`auf, Nachricht 1MB, 8 Ranks auf einem einzelnen Knoten mit NVLink. Wir folgen`ncclTuningCompute`einmal.

**Schritt 0: Single-Rank-Kurzschluss.**Wenn`nRanks <= 1`, ist überhaupt keine Kommunikation nötig, es wird direkt Ring/Simple zurückgegeben, die Kanalanzahl auf 0 gesetzt:

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

Hier ist`NCCL_TUNING_IGNORE`ein Sentinel-Wert, der bedeutet „diese Kombination wurde nicht berechnet/nicht anwendbar“. Das Plugin kann nur die Zellen ändern, die es betrifft; andere Zellen bleiben IGNORE, und NCCL überspringt sie.

**Schritt 4: Optimum auswählen.**ruft`ncclTuningSelectBestTuning`auf und durchläuft die verknüpfte Liste, um das kleinste`selectionTimeUs`zu nehmen.

**Schritt 5: Kanalanzahl berechnen.**Nach der Algorithmusauswahl muss noch entschieden werden, wie viele Kanäle geöffnet werden:

[FACT:src/tuning/tuning.cc:233-235]

```c
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
```

`ncclTuningGetChannels`In`tuning_int.h`wird anhand der Nachrichtengröße und des Algorithmustyps zwischen`minChannels`und`maxChannels`interpoliert. Die Kanalanzahl wirkt sich direkt auf die Bandbreite aus: Je mehr Kanäle, desto höher die Parallelität, aber desto größer auch der Startaufwand pro Kanal.

**Schritt 6: CTA-Policy-Bias (NVLS bevorzugt).**Wenn der Benutzer`NCCL_CTA_POLICY_EFFICIENCY`gesetzt hat und es sich um AllGather/ReduceScatter handelt und der Buffer registriert ist, versucht NCCL, das Ergebnis auf NVLS zu ändern:

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

**Warum werden die Fehlercodes unterschieden?**Wenn der Benutzer`NCCL_ALGO=ring`gesetzt hat, aber die aktuelle Plattform Ring nicht unterstützt (z. B. bei bestimmten speziellen Topologien), dann ist das ein**Benutzerkonfigurationsfehler**（`ncclInvalidUsage`); wenn der Benutzer keine Umgebungsvariable gesetzt hat und trotzdem kein Algorithmus ausgewählt werden kann, dann ist das ein**NCCL-interner Bug**（`ncclInternalError`). Diese Unterscheidung ist für die Fehlersuche entscheidend.

## Flussdiagramm des Entscheidungs-Hauptpfads

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

# Zwei, cost_model.cc: Modellregistrierung und Schaltmatrix

## Intuitives Modell

`cost_model.cc`ist das**Hauptbuch**des Tunings. Es verwaltet eine`modelMap`-Tabelle, wobei jede Zeile einer (algo, proto)-Kombination entspricht und festhält, „wer die Initialisierungsfunktion dieser Kombination ist, wer die Simulationsfunktion ist und für welche Funktionen sie aktiviert ist“. Gleichzeitig ist es für das Parsen der Benutzerumgebungsvariable`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SYM_KERNEL`verantwortlich und übersetzt die Absicht des Benutzers in eine`enabled[i][f]`-Schaltmatrix.

Ohne diese Tabelle müsste für jeden neuen Algorithmus der Haupt-Tuning-Ablauf geändert werden, und der Code würde zu einem unentwirrbaren Brei verkommen.**Tabellengetrieben**macht „Algorithmus hinzufügen“ zu „eine Zeile hinzufügen“.

## Datenstruktur: modelMap und Schaltmatrix

`modelMap`ist ein statisches Array, jedes Element ist`ncclTuningModelEntry_t`：

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

Jeder Eintrag hat vier Felder:`init`(Initialisierung, berechnet latency/bandwidth und speichert sie in comm),`model`(Simulation, berechnet anhand der Nachrichtengröße die endgültige timeUs),`finalize`(Bereinigung),`enabled[5]`(ob Broadcast/Reduce/AllGather/ReduceScatter/AllReduce die fünf Funktionen aktiviert sind).

Achtung`enabled`Die Reihenfolge der Array-Kommentare steht in L234:`Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce`. Diese Reihenfolge muss mit`ncclFunc_t`übereinstimmen, sonst kommt es zu Verwechslungen.

> **[Design Inference & Architectural Trade-offs]**
> **Warum müssen init und sim getrennt sein?**Weil die in init berechneten Dinge (latency, bandwidth)**nur von den statischen Eigenschaften von comm abhängen**(Topologie, Rank-Anzahl, compCap) und nichts mit der konkreten Nachrichtengröße zu tun haben. In einer Kommunikation können mehrere tuning-Aufrufe hintereinander erfolgen (z. B. mehrere ops in einer group), init läuft nur einmal, sim läuft jedes Mal. Das ist eine typische „Vorberechnung + schnelle Abfrage“-Optimierung.

## Step-by-Step: Parsen der Umgebungsvariablen und Aufbau der Schaltmatrix

**Schritt 1: Standardmäßig alles an, LL128 speziell.** `ncclTuningCostModelInit`Anfangs werden alle proto auf 1 gesetzt (aktiviert), aber LL128 auf 2:

[FACT:src/tuning/cost_model.cc:313-323]

```c
  for (int f = 0; f minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                          comm->graphs[algo].typeIntra, comm->nRanks, f, algo, comm->minDriverVersion)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
```

**Schritt 2: Parsen der Benutzer-Umgebungsvariablen.**Wenn der Benutzer`NCCL_ALGO`oder`NCCL_SYM_KERNEL`gesetzt hat, werden zuerst algo und symKernel vollständig auf null gesetzt (weil der Benutzer eine Whitelist angegeben hat):

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

Achtung: proto wird nicht auf null gesetzt – weil die Standardwerte von proto 1/2 sind, wenn der Benutzer`NCCL_PROTO=LL`setzt,`parseList`wird LL auf 1 und die anderen auf 0 gesetzt (wegen der`unset`-Logik). Diese Asymmetrie ist absichtlich: algo ist standardmäßig vollständig aktiviert, muss aber nach Benutzerangabe eingeschränkt werden; die Einschränkung von proto wird intern von`parseList`behandelt.

**Schritt 3: Die Syntax von parseList.**Diese Funktion unterstützt eine recht komplexe Syntax; in den Kommentaren gibt es Beispiele:

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

`^`Das Präfix

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

Kopieren`NCCL_PROTO="^LL128;allreduce:LL128"`Also bedeutet

**: LL128 global deaktivieren, aber für AllReduce ausnahmsweise LL128 aktivieren.**Schritt 4: Zusammenführen der enabled-Matrix.`model->enabled[f]`Schließlich werden alle model durchlaufen und

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

Kopieren**Die Logik ist:**Nur wenn der Benutzer für eine Funktion eine forced-Konfiguration gesetzt hat, wird die Modell-Standardkonfiguration durch die Benutzerkonfiguration überschrieben`forced[f] == 0`. Wenn der Benutzer nichts gesetzt hat,`continue`, direkt`enabled`, und die

## des Modells bleibt erhalten. Das ist die Priorität „explizite Benutzerangabe > Modell-Standard“.

Einheitlicher Einstiegspunkt der Modellsimulation`ncclTuningCostModelSimModel`Alle Modelle werden letztlich über

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

Kopieren**Drei Filterebenen:**id außerhalb des Bereichs → Modell deaktiviert → Modell gibt eine nicht-positive Zeit zurück`not_valid`, wenn eine Ebene nicht durchläuft, geht es zu`timeUs`, wobei`NCCL_TUNING_IGNORE`auf`valid = 0`gesetzt wird (ein negativer Sentinel),`valid == 0`. Wenn der Aufrufer

## sieht, hängt er es nicht in die Kandidatenliste ein.

`modelMap`Design-Überlegung

[FACT:src/tuning/cost_model.cc:229]

```c
// IMPORTANT: this table need must be consistent with the algRegistry in src/config/algorithm_registry.cc
```

> **[Design Inference & Architectural Trade-offs]**
> Kopieren`modelMap`〔Design-Inferenz und Architektur-Abwägung〕**Das bedeutet, dass die**Indexreihenfolge`algorithm_registry.cc`von`modelMap`strikt mit der Algorithmus-Registrierungsreihenfolge in**übereinstimmen muss. Wenn jemand in der registry einen neuen Algorithmus einfügt, aber vergisst,**zu ändern, sind alle ids verschoben, und tuning wählt einen völlig falschen Algorithmus.

---

# Das ist die klassische Falle tabellengetriebenen Designs: impliziter Vertrag.

## Robuster wäre, den Enum-Namen als key statt des Index zu verwenden, aber das würde ein wenig Compile-Zeit-Optimierung opfern.

Drei, ring.cc: Kostenschätzung des Ring-Algorithmus**Intuitives Modell**、**Der Ring-Algorithmus ordnet N ranks zu einem Ring an, und die Daten werden entlang des Rings Runde für Runde übertragen. Sein Kostenmodell muss zwei Fragen beantworten:**。

Wie viele Daten pro Schritt übertragen werden (Bandbreite)**Wie viele Schritte insgesamt nötig sind (Latenz)**Die Intuition von Ring ist „

## Pipeline

“: Man stelle sich N Personen vor, die im Kreis stehen und einen Eimer weiterreichen; jeder empfängt den Eimer, gießt etwas Wasser hinein und gibt ihn an die nächste Person weiter. Wenn der Eimer eine Runde dreht, ist das Wasser aller vermischt. Je schneller der Eimer kreist (hohe Bandbreite) und je kleiner der Kreis (wenige Schritte), desto schneller geht es insgesamt.`comm->tuningContext.generalLatencies[c][algo][proto]`Datenstruktur: latency/bandwidth-Tabelle`generalBandwidths[c][algo][proto]`Das Ring-Modell führt keine neue Struktur ein; es schreibt die Schätzergebnisse in

und

[FACT:src/tuning/ring.cc:31-33]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
```

Bei der Initialisierung werden zunächst alle auf -1.0 gesetzt (Sentinel, bedeutet „noch nicht berechnet“):

[FACT:src/tuning/ring.cc:94-97]

```c
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ncclSuccess;
  }
```

**Dieser Sentinel -1.0 wird in der sim-Phase geprüft:**Kopieren`==`Warum -1.0 und nicht 0?

## Weil 0 ein legitimer Bandbreitenwert ist (obwohl physikalisch unmöglich), während -1.0 eindeutig „nicht initialisiert“ bedeutet. Der Fließkommavergleich mit

**ist hier sicher, weil -1.0 exakt darstellbar ist.**Step-by-Step: Ring-Bandbreitenschätzung

[FACT:src/tuning/ring.cc:34-37]

```c
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float bw = (comm->nNodes == 1 || (comm->nNodes minCompCap graphs[algo].bwIntra :
                                                                                      comm->graphs[algo].bwInter;
    float busBw = bw * comm->graphs[algo].nChannels;
```

`nSteps`Einzelmaschine (nNodes==1) verwendet intra, Mehrmaschinen verwenden inter:`2*(nRanks-1)`Kopieren`nRanks-1`。`busBw`ist die Anzahl der vom Algorithmus benötigten Schritte; für Ring ist AllReduce

**, die anderen sind**Das LL-Protokoll nutzt nur die Hälfte der Bandbreite (wegen des Flag-Overheads von LL), LL128 nutzt 92% (120/128):

[FACT:src/tuning/ring.cc:38-42]

```c
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);
```

`0.92 = 120/128`Das liegt daran, dass bei LL128 alle 128 Bytes 8 Bytes Flag sind und die Nutzlast nur 120 Bytes beträgt. Diese Zahl stammt direkt aus dem Protokolldesign.

**Schritt 3: Effektive Bandbreite berechnen.**Beachten Sie, dass hier multipliziert wird mit`nRanks / nSteps`：

[FACT:src/tuning/ring.cc:44-46]

```c
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;
```

**Warum multipliziert mit`nRanks / nSteps`？**Dies ist eine Kern Eigenschaft des Ring-Algorithmus: Die Datenmenge, die jeder Rank tatsächlich transportiert, ist`nBytes * nSteps / nRanks`(weil die Daten mehrere Runden um den Ring laufen müssen). Daher gilt: „Effektive Bandbreite“ = Busbandbreite × nRanks / nSteps. Für AllReduce gilt nSteps = 2(nRanks-1), also effektive Bandbreite ≈ busBw/2.

**Schritt 4: Latenz berechnen.**Die Latenz teilt sich in zwei Teile: intra und inter:

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

Beachten Sie die spezielle Behandlung in L57-58: Wenn`maxLocalRanks == 1`(jeder Knoten nur 1 Rank hat), verwendet die Inter-Node-Latenz von Ring**Die NET-Latenz von Tree**. Der Kommentar sagt, dies sei „preserve the pre-refactor model“ – also eine bewusst beibehaltene „Eigenheit“, um das Verhalten vor dem Refactoring zu bewahren.**Solche historischen Altlasten sind in ausgereiften Systemen sehr verbreitet. Wenn man beim Lesen des Quellcodes auf „preserve“ stößt, sollte man besonders vorsichtig sein, denn es bedeutet oft, dass hier eine unveränderliche Kompatibilitätsbedingung vorliegt.**

**Schritt 5: Nach Funktionstyp akkumulieren.**Reduce/Broadcast und AllReduce/AllGather/ReduceScatter haben unterschiedliche Latenzmodelle:

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

`sameChannels`Ist eine topologische Eigenschaft, die angibt, „ob die intra- und inter-Schritte auf dem Ring dieselbe Gruppe von Channels verwenden“. Wenn nicht, muss die Latenz multipliziert werden mit`nSteps`(jeder Schritt muss warten).`netOverhead`Ist der Netzwerk-Post-Overhead; beim Simple-Protokoll muss mit 3 multipliziert werden (weil Simple drei Netzwerk-Roundtrips hat: send, recv, ack).

## Produktions-Fallstricke: Der Plateau-Effekt von Ring/Simple

`ncclTuningRingModelSim`Enthält einen Codeabschnitt, der speziell den „plateau“ behandelt:

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
> **Was ist ein Plateau?**Bei Ring/Simple wächst die Latenz ab einer bestimmten Nachrichtengröße nicht mehr linear mit der Nachricht, sondern „bleibt hängen“ auf einem Plateau – weil der Engpass dann von „Startup-Overhead“ zu „Bandbreite“ wird und die Bandbreite bereits gesättigt ist. Dieses Phänomen ist auf Blackwell NVLink besonders ausgeprägt (weil die NVLink-Bandbreite so hoch ist, dass der Latenzanteil größer wird). Der Code multipliziert`plateauFactor`(1.4 oder 1.9) auf die Latenz, um diesen Effekt der „verstärkten Latenz“ zu simulieren.

`bytesPerRankPerChannel >= 64`Ist die Auslösebedingung: Jeder Rank muss pro Channel mindestens 64 Bytes übertragen, sonst gilt das Plateau nicht. Diese 64 Bytes stammen aus der Flag-Größe des LL-Protokolls.

**Fallstrick-Szenario**: Wenn Sie auf Blackwell ein 1MB AllReduce ausführen und feststellen, dass die tatsächliche Latenz 40% höher ist als vom Modell vorhergesagt, denken Sie nicht, es sei ein Bug – das ist der Plateau-Effekt, und das Modell hat ihn bereits berücksichtigt. Wenn Sie manuell`plateauFactor`verkleinern, unterschätzt das Modell die Latenz, was zur Wahl des falschen Algorithmus führt.

---

# IV. tree.cc und nvls.cc: Kostenschätzung für Tree und NVLS

## Intuitives Modell

**Der Tree-Algorithmus**Ist „**Baumförmiges Broadcast**“: Der Wurzelknoten verteilt die Daten an Kindknoten, die Kindknoten verteilen sie weiter an Enkelknoten. Sein Vorteil ist**Wenige Schritte**(log N statt N), geeignet für kleine Nachrichten; sein Nachteil ist**Geringe Bandbreiteneffizienz**(jeder Nicht-Blattknoten muss weiterleiten, die tatsächlich effektive Bandbreite beträgt nur die Hälfte).

**NVLS**(NVLink SHARP) ist „**Hardware-Multicast**“: Der Switch kopiert die Daten direkt an mehrere GPUs, ohne Software-Weiterleitung. Sein Vorteil ist**Hohe Bandbreite, niedrige Latenz**, erfordert jedoch spezielle Hardware (Hopper oder neuer) und eine spezielle Konfiguration.

## Tree-Modell: Dient nur AllReduce

Das Tree-Modell hat eine harte Einschränkung –**Nur für AllReduce aktiviert**：

[FACT:src/tuning/tree.cc:21-27]

```c
  for (int c = 0; c tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
```

> **[Design Inference & Architectural Trade-offs]**
> **Warum?**Weil die Tree-Implementierung von NCCL nur AllReduce unterstützt (andere kollektive Operationen haben keine Tree-Version). Dies ist eine Implementierungseinschränkung, keine theoretische.`enabled[c] = 0`Ist eine „harte Deaktivierung“, die`generalBandwidths = -1`Noch gründlicher als`ncclTuningCostModelSimModel`– Erstere lässt`not_valid`Bereits in L480

**zurückgeben, Letztere prüft erst in der sim-Funktion.**：

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
> 〔Design-Inferenz und Architektur-Abwägung〕`1/3.8`Beachten Sie, dass der Abschlagsfaktor des LL-Protokolls`0.5`beträgt, was härter ist als der von Ring mit**.**Warum ist die LL-Effizienz von Tree niedriger?`1/3.8`Weil jeder Zwischenknoten im Tree sowohl empfangen als auch senden muss und der Flag-Overhead von LL bei bidirektionalem Verkehr verstärkt wird.

**Diese Zahl stammt aus Messungen.**：

[FACT:src/tuning/tree.cc:55-58]

```c
    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
```

`2 *`Kopieren`(nRanks/nNodes - 1)`Weil AllReduce = ReduceScatter + AllGather, zwei Durchläufe.`log2i(nNodes)`Ist die Anzahl der Intra-Node-Schritte (Anzahl der Ranks pro Knoten minus eins),

**Ist die Anzahl der Inter-Node-Schritte (Höhe des Baums).**：Das Tree-Modell wird in der sim-Phase mit einem`treeCorrectionFactor`：

[FACT:src/tuning/tree.cc:75-79]

```c
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize proto][logSize];
```

`treeCorrectionFactor`multipliziert. Es ist eine 3×24-Tabelle:

[FACT:src/tuning/cost_model.cc:223-227]

```c
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};
```

`logSize = log2(nBytes >> 6)`, d.h. die Nachrichtengröße wird in Einheiten von 64 Byte logarithmiert (log2). Die Tabellenindizes 0-23 entsprechen 64B bis 64B×2^23 ≈ 512MB.**Diese Tabelle ist die gemessene „Tree-Effizienzkurve"**: Bei kleinen Nachrichten ist die Effizienz 1,0 (latenzdominiert), bei mittleren Nachrichten fällt sie auf 0,4-0,5 (Bandbreite nicht ausgelastet), bei großen Nachrichten steigt sie wieder auf 1,0 (Bandbreite voll ausgelastet). Diese „mittlere Senke" ist eine inhärente Eigenschaft des Tree-Algorithmus.

## NVLS-Modell: Die Kosten von Hardware-Multicast

Das NVLS-Modell prüft zunächst, ob die Hardware dies unterstützt:

[FACT:src/tuning/nvls.cc:19-24]

```c
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!ncclNvlsTransportEnabled(comm)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
```

Dann folgt eine Reihe harter Einschränkungen: Nur das Simple-Protokoll wird unterstützt, Single-Node unterstützt kein NVLSTree, Multi-Node-NVLS benötigt CollNet:

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

**NVLS-Bandbreitenschätzung**verwendet einen Effizienzfaktor:

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
> Hopper ist 0,85, Blackwell fällt jedoch auf 0,74.**Warum ist die Effizienz der neueren Hardware-Generation niedriger?**Weil Blackwells NVLink-Bandbreite höher ist, aber die Verarbeitungskapazität der NVLS-Switches nicht im gleichen Maße gestiegen ist, was zu einem relativen Effizienzrückgang führt. Diese Zahl ist gemessen, nicht theoretisch.

In der Bandbreitenberechnung gibt es einen`(nChannels - 1) / nChannels`Faktor:

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

`(nChannels - 1) / nChannels`weil NVLS einen Channel für die Synchronisation reservieren muss.`(ppn - 1) / ppn`ist der zusätzliche Overhead von AllGather/ReduceScatter (jeder Rank muss auf die Daten des vorherigen Rank warten).

## Produktions-Fallstricke: Die harten Einschränkungen von NVLS

Das NVLS-Modell hat in der sim-Phase noch eine weitere Laufzeitprüfung:

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

`NCCL_MAX_NVLS_ARITY`ist die maximale Anzahl von GPUs, die eine NVLS-Multicast-Gruppe aufnehmen kann. Wird diese Zahl überschritten, ist NVLS nicht verfügbar.**Fallstrick-Szenario**: Wenn in einer 16-Karten-NVLink-Domain AllGather ausgeführt wird und`NCCL_MAX_NVLS_ARITY`8 ist, wird NVLS deaktiviert und das Tuning fällt auf Ring zurück. Wenn man diese Einschränkung nicht kennt, fragt man sich: „Warum wird NVLS nicht genutzt, obwohl die Hardware es doch unterstützt?"

---

# Fünf: Symmetrischer Kernel-Rückfall und Fehlerwiederherstellungskette

## Intuitives Modell

Der symmetrische Kernel (symmetric kernel) ist eine neue NCCL-Funktion: Wenn die Buffer aller Ranks im symmetrischen Speicher registriert sind, kann der Kernel mit effizienteren Instruktionen auf den Speicher der Gegenseite zugreifen. Aber**wenn der Buffer nicht registriert ist oder die Plattform nicht unterstützt wird, muss auf den normalen Kernel zurückgefallen werden**. Diese Rückfalllogik ist der verworrenste Teil des Tunings.

## Step-by-Step: Die Rückfallentscheidung

Die Rückfalllogik befindet sich in`tuning.cc:258-298`. Zerlegen wir das.

**Schritt 1: Prüfen, ob ein Rückfall nötig ist.**Eintrittsbedingung:

[FACT:src/tuning/tuning.cc:258-263]

Damit ist die Entscheidungskette des Tuning-Moduls klar: Es empfängt den Topologiegraphen und Kommunikationsparameter, gibt über Kostenmodell und Algorithmusschätzung innerhalb von Mikrosekunden die optimale Kombination (Algorithmus, Protokoll, Channel, Warp) aus. Doch die Auswahl ist nur der Anfang – wie wird dieses Entscheidungsergebnis nachgelagert verwendet? Im nächsten Kapitel betreten wir das Hauptstück von src/enqueue/enqueue.cc und sehen, wie ein ncclAllReduce-Aufruf durch Parametervalidierung, Algorithmus-/Protokollbestimmung und Channel-Aufteilung schließlich die Strukturen ncclInfo und ncclTaskColl erzeugt. Dies ist das Schlüsselkapitel des Buches, in dem von der „Nutzerperspektive" zur „Engine-Perspektive" gewechselt wird. Du wirst ergründen, in was ein kollektiver Kommunikationsaufruf auf der Host-Seite übersetzt wird und wo die Grenze zum anschließenden Kernel-Start liegt.
