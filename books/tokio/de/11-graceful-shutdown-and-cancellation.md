# Kapitel 11: Stream-Ökosystem und Werkzeugschicht: Die Erweiterungsmechanismen von tokio-stream und tokio-util

Im vorherigen Kapitel haben wir die Byte-Ebene von Framed zerlegt: Decoder schneidet BytesMut in Frames, Sink schreibt Frames zurück, und damit wird die Abstraktionsgrenze der asynchronen I/O klar. Doch Frames sind nur Container für Daten. Eine echte Protokollimplementierung stößt unmittelbar danach auf drei Probleme, die weder tokio::io noch Framed lösen: asynchrone Iteration – Framed implementiert Stream, aber Stream hat nur poll_next, kein next().await, filter, take, merge; handgeschriebenes poll_fn ist nicht nur umständlich, sondern führt auch leicht zu Fehlern bei der Cancel-Sicherheit; dynamische Aufgabenmengen – ein Chat-Dienst muss gleichzeitig N Kanäle abonnieren, Kanäle treten jederzeit bei oder aus, während die Anzahl der Zweige von select! zur Compile-Zeit festgelegt ist und daher keine zur Laufzeit wachsende oder schrumpfende Stream-Menge ausdrücken kann; strukturierter Abbruch – select! kann einen einzelnen Zweig abbrechen, aber es kann nicht die Stilllegung eines gesamten Aufgabenbaums weiterreichen und auch nicht darauf warten, dass alle Aufgaben tatsächlich beendet sind. tokio-stream und tokio-util wurden genau für diese drei Dinge geschaffen. Ihr zentrales Designprinzip ist, nichts von Grund auf neu zu bauen: Jeder Kombinator von StreamExt ist nur eine Hülle um poll_next, StreamMap verwendet die Registrierungssemantik des Waker wieder, CancellationToken baut direkt auf tokio::sync::Notify auf, und TaskTracker kodiert den gesamten Zustand in einem AtomicUsize. Sie zu verstehen bedeutet im Wesentlichen zu verstehen, wie man auf den bestehenden Waker- und Scheduling-Mechanismen Zero-Cost-Abstraktionen baut. Dieses Kapitel schreitet in drei Ebenen voran: Iteration, Mengen, Abbruch. Zuerst wird gezeigt, wie StreamExt poll_next in einen komponierbaren Iterator verwandelt, dann, wie StreamMap und TaskTracker dynamische Mengen verwalten, und schließlich, wie CancellationToken mit einem Baum ein Abbruchsignal im gesamten Aufgabenbaum verbreitet.

# StreamExt: poll_next in einen komponierbaren Iterator verwandeln

## Intuitives Modell

`Stream`verhält sich zu`Future`, wie`Iterator`sich zu Werten verhält:`Future`erzeugt „einen Wert“,`Stream`erzeugt „eine Folge von Werten“. Aber`Stream`definiert nur`poll_next`als einziges Primitiv, so wie`Iterator`nur`next`definiert. Ohne`StreamExt`müsste man für jedes Filtern, Mappen und Abschneiden manuell`poll_fn`-Closures schreiben und`Pin`manuell verwalten – genau das war für frühe Nutzer des`futures`-Crates am schmerzhaftesten.`StreamExt`Die Rolle von`Stream`besteht darin,`Iterator`ein Kombinator-Ökosystem wie das von

zu geben.**Ohne es steht das System nicht vor fehlender Funktionalität, sondern vor**einem systematischen Zusammenbruch der Cancel-Sicherheit`poll_fn`: Jedes handgeschriebene`select!`kann bei einem Abbruch durch`poll`ein Element verlieren, das bereits

## wurde.

`StreamExt`Datenstruktur und Speicherlayout**ist ein**Erweiterungs-Trait

[FACT:tokio-stream/src/stream_ext.rs:106-106]

```rust
pub trait StreamExt: Stream {
```

Kopieren**Alle seine Methoden geben ein**konkretes Kombinator-Struct`Box<dyn Stream>`zurück, nicht`map`. Das ist das entscheidende Design:`Map<Self, F>`，`filter`gibt`Filter<Self, F>`，`take`zurück,`Take<Self>`gibt`poll_next`zurück,

gibt

[FACT:tokio-stream/src/stream_ext.rs:1213-1213]

```rust
impl StreamExt for St where St: Stream {}
```

-Aufrufen inline expandieren.`Stream`Beachten Sie die Blanket-Impl des Traits:`?Sized`Kopieren`dyn Stream`Jedes

erhält automatisch alle Kombinatoren, ohne manuelle Implementierung.

[FACT:tokio-stream/src/stream_ext.rs:4-59]

```rust
mod all; use all::AllFuture;
mod any; use any::AnyFuture;
mod chain; pub use chain::Chain;
pub(crate) mod collect; use collect::{Collect, FromStream};
mod filter; pub use filter::Filter;
mod filter_map; pub use filter_map::FilterMap;
mod fold; use fold::FoldFuture;
mod fuse; pub use fuse::Fuse;
mod map; pub use map::Map;
mod map_while; pub use map_while::MapWhile;
mod merge; pub use merge::Merge;
mod next; use next::Next;
mod skip; pub use skip::Skip;
mod skip_while; pub use skip_while::SkipWhile;
mod take; pub use take::Take;
mod take_while; pub use take_while::TakeWhile;
mod then; pub use then::Then;
mod try_next; use try_next::TryNext;
mod peekable; pub use peekable::Peekable;
```

die Erweiterungsmethoden nutzen kann.`next`、`try_next`、`all`、`any`、`fold`、`collect`Die Moduldeklaration der Kombinatoren offenbart die vollständige Fähigkeitsoberfläche dieses Traits:**Future**（`Next`、`TryNext`、`AllFuture`Kopieren`map`、`filter`、`take`Hier gibt es eine bemerkenswerte Unterscheidung:**Stream**gibt`next`zurück …), weil sie den gesamten Stream zu einem Wert konsumieren; während`Next<'_, Self>`usw.

[FACT:tokio-stream/src/stream_ext.rs:144-149]

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin`Der Rückgabetyp von`next`ist`Pin`, mit einem Lifetime-Parameter, weil es den Stream nur ausleiht:`!Unpin`Kopieren`Box::pin`Die`pin_mut!`-Beschränkung ist absichtlich:

[FACT:tokio-stream/src/stream_ext.rs:116-121]

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## werden. Wenn der Stream`merge`Polling von

`merge`ist das beste Beispiel dafür, wie ein Kombinator Waker wiederverwendet. Er verschränkt die Ausgabe zweier Streams und**garantiert Fairness**– wenn beide Streams gleichzeitig bereit sind, wird abwechselnd ausgegeben. Die Dokumentation warnt ausdrücklich davor, verkettete Aufrufe zu verwenden`merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321]

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge`Die Signatur von erfordert, dass beide Streams denselben`Item`Typ haben:

[FACT:tokio-stream/src/stream_ext.rs:398-404]

```rust
fn merge(self, other: U) -> Merge
where
    U: Stream,
    Self: Sized,
{
    Merge::new(self, other)
}
```

Wenn der Aufrufer`.next().await`aufruft, läuft die Ausführung wie folgt ab:

1. `Next::poll`ruft auf`Merge::poll_next`。

2. `Merge`Intern wird ein boolesches Flag „wer zuletzt an der Reihe war" verwaltet. Zuerst wird`poll`der Stream, der zuletzt nichts ausgegeben hat; wenn`Pending`, dann`poll`der andere.

3. Wenn beide`Pending`，`Merge`zurückgeben`Pending`, aber**die jeweiligen Waker beider Streams bereits registriert sind**– jede Bereitschaft weckt die aktuelle Aufgabe auf.

4. Wenn ein Stream`Ready(None)`zurückgibt (Ende),`Merge`wird vermerkt, dass dieser Stream beendet ist, und danach nur noch`poll`der andere Stream, bis auch dieser endet.

Der Schlüssel hierbei ist:`Merge`hat keine eigene Waker-Verwaltungslogik; es gibt`cx`unverändert an die internen beiden Streams weiter`poll_next`。**Die Waker-Registrierung liegt vollständig in der Verantwortung der zugrunde liegenden Streams**，`Merge`entscheidet nur, „wen diesmal zuerst gefragt wird". Genau das bedeutet „Wiederverwendung des zugrunde liegenden Waker-Mechanismus" wörtlich.

`merge_size_hints`Die Hilfsfunktion zeigt, wie Kombinatoren Kapazitätshinweise zusammenführen:

[FACT:tokio-stream/src/stream_ext.rs:1216-1226]

```rust
fn merge_size_hints(
    (left_low, left_high): (usize, Option),
    (right_low, right_high): (usize, Option),
) -> (usize, Option) {
    let low = left_low.saturating_add(right_low);
    let high = match (left_high, right_high) {
        (Some(h1), Some(h2)) => h1.checked_add(h2),
        _ => None,
    };
    (low, high)
}
```

Beachten Sie die Wahl von`saturating_add`und`checked_add`: Für die untere Schranke wird saturierende Addition verwendet (lieber unterschätzen als bei Überlauf panicen), für die obere Schranke geprüfte Addition (wenn irgendeine unbekannt ist, ist das Ganze unbekannt). Dies ist die typische Behandlung des`size_hint`-Vertrags.

## Designüberlegung: Abbruchsicherheit und`chunks_timeout`Panic-Schutz

`StreamExt`Die Dokumentation von annotiert jede Methode mit**Cancel safety**. Nehmen wir`next`als Beispiel:

[FACT:tokio-stream/src/stream_ext.rs:123-127]

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next`ist abbruchsicher, weil es den Stream nur ausleiht und keine Elemente konsumiert –`Next`wenn das Future gedroppt wird, bleibt der Zustand des Streams unverändert, beim nächsten`next`wird erneut`poll`。

Aber nicht alle Kombinatoren sind abbruchsicher.`chunks_timeout`führt bereits bei der Konstruktion eine Parameterprüfung durch:

[FACT:tokio-stream/src/stream_ext.rs:1178-1185]

```rust
#[track_caller]
fn chunks_timeout(self, max_size: usize, duration: Duration) -> ChunksTimeout
where
    Self: Sized,
{
    assert!(max_size > 0, "`max_size` must be non-zero.");
    ChunksTimeout::new(self, max_size, duration)
}
```

> **[Design Inference & Architectural Trade-offs]**
> `#[track_caller]`lässt die Panic-Position auf den Aufrufer statt auf das Bibliotheksinnere zeigen,`assert!`lehnt bereits zur Konstruktionszeit ab`max_size == 0`. Warum muss zur Konstruktionszeit geprüft werden? Wenn`max_size == 0`，`ChunksTimeout`erlaubt würde, gerät die Batch-Logik in eine Endlosschleife des „nie eine volle Charge ansammeln" oder produziert leere Chargen, und solche Bugs sind zur Laufzeit extrem schwer zu lokalisieren. Ein Panic zur Konstruktionszeit verlagert den Fehler auf den frühesten beobachtbaren Punkt.

`timeout`Der Unterschied zwischen`timeout_repeating`und`timeout`ist ebenfalls beachtenswert:**gibt nach einem Timeout einen Fehler zurück, aber**；`timeout_repeating`pollt den inneren Stream weiter`Interval`produziert gemäß

[FACT:tokio-stream/src/stream_ext.rs:985-1001]

```rust
/// Once a timeout error is received, no further events will be received
/// unless the wrapped stream yields a value (timeouts do not repeat).
```

[FACT:tokio-stream/src/stream_ext.rs:1071-1072]

```rust
/// Timeout errors will be continuously produced at the specified interval
/// until the wrapped stream yields a value.
```

---

# Kopie

## StreamMap: Dynamische Stream-Mengen und faires Polling

`select!`Intuitives Modell`StreamMap`Die Anzahl der Zweige von ist zur Kompilierzeit festgelegt. Aber die Anzahl der Kanäle, die ein Chat-Dienst abonnieren muss, oder die Anzahl der Verbindungen, die ein Crawler verfolgen muss, sind zur Laufzeit bekannt.`select!`ist ein „zur Laufzeit erweiterbares und reduzierbares`next`": Es legt beliebig viele Streams in eine Menge, und jedes`(key, value)`gibt`mpsc`zurück und teilt mit, von welchem Stream der Wert stammt. Ohne es müsste man alle Streams in einen

## -Kanal stopfen, mit zusätzlichem Weiterleitungsaufwand.

`StreamMap`Datenstruktur und Speicherlayout`Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208]

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

Kopie

[FACT:tokio-stream/src/stream_map.rs:38-44]

```rust
/// `StreamMap` is backed by a `Vec`. There is no guarantee that this
/// internal implementation detail will persist in future versions, but it is
/// important to know the runtime implications. In general, `StreamMap` works
/// best with a "smallish" number of streams as all entries are scanned on
/// insert, remove, and polling. In cases where a large number of streams need
/// to be merged, it may be advisable to use tasks sending values on a shared
/// [`mpsc`] channel.
```

> **[Design Inference & Architectural Trade-offs]**
> 〔Design-Inferenz und Architektur-Abwägungen〕`HashMap`Warum nicht`StreamMap`? Weil die Kernoperation von**darin besteht,**alle Streams zu pollen`Vec`, und nicht im Nachschlagen per Schlüssel.`swap_remove`Der lineare Scan von ist CPU-cache-freundlich, und`HashMap`ist O(1). Bei Verwendung von`poll_next`müsste jedes`insert`die Hash-Buckets durchlaufen, mit schlechterer Cache-Lokalität.`remove`Der O(n)-Scan von

`insert`und

[FACT:tokio-stream/src/stream_map.rs:446-454]

```rust
pub fn insert(&mut self, k: K, stream: V) -> Option
where
    K: Hash + Eq,
{
    let ret = self.remove(&k);
    self.entries.push((k, stream));

    ret
}
```

`remove`Die Implementierung von spiegelt die Semantik „erst löschen, dann einfügen" wider:`swap_remove`Kopie

[FACT:tokio-stream/src/stream_map.rs:471-483]

```rust
pub fn remove(&mut self, k: &Q) -> Option
where
    K: Borrow,
    Q: Hash + Eq + ?Sized,
{
    for i in 0..self.entries.len() {
        if self.entries[i].0.borrow() == k {
            return Some(self.entries.swap_remove(i).1);
        }
    }

    None
}
```

## , um das gelöschte Element mit dem letzten Element zu tauschen und dann zu entfernen, wodurch O(n)-Verschiebungen vermieden werden:

`StreamMap`Kopie`poll_next_entry`Szenario-getriebener Walkthrough: Zufälliger Startpunkt und Cursor-Korrektur von poll_next_entry**Der Kern von ist**. Es beginnt mit

[FACT:tokio-stream/src/stream_map.rs:515-550]

```rust
fn poll_next_entry(&mut self, cx: &mut Context) -> Poll> {
    let start = self::rand::thread_rng_n(self.entries.len() as u32) as usize;
    let mut idx = start;

    for _ in 0..self.entries.len() {
        let (_, stream) = &mut self.entries[idx];

        match Pin::new(stream).poll_next(cx) {
            Poll::Ready(Some(val)) => return Poll::Ready(Some((idx, val))),
            Poll::Ready(None) => {
                // Remove the entry
                self.entries.swap_remove(idx);

                // Check if this was the last entry, if so the cursor needs
                // to wrap
                if idx == self.entries.len() {
                    idx = 0;
                } else if idx  {
                idx = idx.wrapping_add(1) % self.entries.len();
            }
        }
    }

    // If the map is empty, then the stream is complete.
    if self.entries.is_empty() {
        Poll::Ready(None)
    } else {
        Poll::Pending
    }
}
```

, um Fairness zu gewährleisten – wenn immer bei Index 0 begonnen würde, würde der erste Stream die nachfolgenden Streams aushungern:

**Kopie** `thread_rng_n`Dieser Code hat drei Feinheiten, die wir einzeln aufschlüsseln:`FastRand`Erstens, der zufällige Startpunkt.`xorshift64+`verwendet thread-lokales

[FACT:tokio-stream/src/stream_map.rs:765-768]

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n`-Algorithmus:`% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792]

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**verwendet Lemires multiplikative Modulo-Operation anstelle von`swap_remove`Kopie**Zweitens, die Cursor-Korrektur nach`idx`. Wenn der Stream bei Index`None`zurückgibt`swap_remove`und entfernt wird,`idx`verschiebt das letzte Element nach**. Dieses verschobene Element könnte**bereits gepollt worden sein`start`(wenn sein ursprünglicher Index vor`idx < start && start <= self.entries.len()`lag). Der Code erkennt dies mit`idx = idx.wrapping_add(1) % len`und überspringt es gegebenenfalls (`idx == len`). Wenn das entfernte Element das letzte war (

**), wird der Cursor auf 0 zurückgesetzt.`Poll::Pending`Drittens, die Semantik von**. Wenn eine vollständige Runde keinen bereiten Stream findet und die Menge nicht leer ist, wird`Pending`zurückgegeben. Zu diesem Zeitpunkt sind die Waker aller Streams registriert, und jede Bereitschaft weckt auf.

`poll_next`ergänzt den Key auf`poll_next_entry`:

[FACT:tokio-stream/src/stream_map.rs:676-683]

```rust
fn poll_next(mut self: Pin, cx: &mut Context) -> Poll> {
    if let Some((idx, val)) = ready!(self.poll_next_entry(cx)) {
        let key = self.entries[idx].0.clone();
        Poll::Ready(Some((key, val)))
    } else {
        Poll::Ready(None)
    }
}
```

Beachten Sie das`ready!`-Makro: Wenn`poll_next_entry`zurückgibt`Pending`, gibt das gesamte`poll_next`sofort`Pending`。`K: Clone`zurück. Die Einschränkung stammt aus dem hier verwendeten`key.clone()`。

## Designüberlegung: Batch-Semantik und Abbruchsicherheit von next_many

`next_many`ist die Batch-Version von`StreamMap`und sammelt in einem Durchgang so viele bereite Elemente wie möglich:

[FACT:tokio-stream/src/stream_map.rs:581-583]

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

Seine Abbruchsicherheitsgarantie ist entscheidend:

[FACT:tokio-stream/src/stream_map.rs:573-578]

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

Warum ist`next_many`abbruchsicher? Weil es Elemente**sofort in den vom Aufrufer bereitgestellten`buffer`**pusht, statt sie intern zwischenzuspeichern. Wenn das Future gedroppt wird, bleiben die bereits gepushten Elemente im`buffer`erhalten und gehen nicht verloren. Das bedeutet aber auch: Beim Drop kann`buffer`bereits teilweise Elemente enthalten – der Aufrufer muss das wissen.

`poll_next_many`Die Schleifenstruktur von ist komplexer als die von`poll_next_entry`, weil es in einer Runde so viele Elemente wie möglich sammeln muss:

[FACT:tokio-stream/src/stream_map.rs:597-666]

```rust
pub fn poll_next_many(
    &mut self,
    cx: &mut Context,
    buffer: &mut Vec,
    limit: usize,
) -> Poll {
    if limit == 0 || self.entries.is_empty() {
        return Poll::Ready(0);
    }

    let mut added = 0;

    let start = self::rand::thread_rng_n(self.entries.len() as u32) as usize;
    let mut idx = start;

    while added  {
                    added += 1;

                    let key = self.entries[idx].0.clone();
                    buffer.push((key, val));

                    should_loop = true;

                    idx = idx.wrapping_add(1) % self.entries.len();

                    if added == limit {
                        break;
                    }
                }
                Poll::Ready(None) => {
                    // Remove the entry
                    self.entries.swap_remove(idx);

                    // Check if this was the last entry, if so the cursor needs
                    // to wrap
                    if idx == self.entries.len() {
                        idx = 0;
                    } else if idx  {
                    idx = idx.wrapping_add(1) % self.entries.len();
                }
            }
        }

        if !should_loop {
            break;
        }
    }

    if added > 0 {
        Poll::Ready(added)
    } else if self.entries.is_empty() {
        Poll::Ready(0)
    } else {
        Poll::Pending
    }
}
```

Das äußere`while added < limit`bildet zusammen mit dem inneren`for`einen „mehrrundigen Scan": Solange in der vorherigen Runde ein Stream einen Wert produziert hat (`should_loop = true`), wird eine weitere Runde gescannt, bis genügend`limit`angesammelt sind oder eine Runde keine Ausgabe liefert. Die drei Fälle des Rückgabewerts entsprechen präzise der Dokumentation:

[FACT:tokio-stream/src/stream_map.rs:588-591]

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint`Die Implementierung zeigt, wie man Kapazitätshinweise mehrerer Streams aggregiert:

[FACT:tokio-stream/src/stream_map.rs:685-701]

```rust
fn size_hint(&self) -> (usize, Option) {
    let mut ret: (usize, Option) = (0, Some(0));

    for (_, stream) in &self.entries {
        let hint = stream.size_hint();

        ret.0 = ret.0.saturating_add(hint.0);

        match (ret.1, hint.1) {
            (Some(a), Some(b)) => ret.1 = a.checked_add(b),
            (Some(_), None) => ret.1 = None,
            _ => {}
        }
    }

    ret
}
```

Dasselbe Muster wie bei`merge_size_hints`: Untere Schranke sättigend addieren, obere Schranke prüfend addieren, bei Unbekanntem bleibt das Ganze unbekannt.

Im Folgenden wird der Entscheidungspfad von`poll_next_entry`anhand eines Flussdiagramms dargestellt:

```mermaid
flowchart TD
    start["poll_next_entry(cx)"] --> rand["start = thread_rng_n(len)"]
    rand --> loop{"遍历 len 次?"}
    loop -->|"未完成"| poll["Pin::new(stream).poll_next(cx)"]
    poll -->|"Ready(Some(val))"| ret_val["返回 Ready(Some((idx, val)))"]
    poll -->|"Ready(None)"| remove["entries.swap_remove(idx)"]
    remove --> wrap{"idx == entries.len()?"}
    wrap -->|"是"| set_zero["idx = 0"]
    wrap -->|"否"| check_swap{"idx |"是"| skip["idx = idx.wrapping_add(1) % len"]
    check_swap -->|"否"| loop
    set_zero --> loop
    skip --> loop
    poll -->|"Pending"| advance["idx = idx.wrapping_add(1) % len"]
    advance --> loop
    loop -->|"遍历完成"| empty{"entries.is_empty()?"}
    empty -->|"是"| ret_none["返回 Ready(None)"]
    empty -->|"否"| ret_pending["返回 Pending"]
```

---

# TaskTracker: Alle Zustände in einem einzigen AtomicUsize kodieren

## Intuitives Modell

Graceful Shutdown erfordert zwei Dinge:**Aufgaben zum Stoppen benachrichtigen**（`CancellationToken`ist dafür zuständig), sowie**warten, bis Aufgaben tatsächlich beendet sind**（`TaskTracker`ist dafür zuständig).`TaskTracker`ist wie eine Kombination aus „Aufgabenzähler + Shutdown-Schalter": Solange noch Aufgaben laufen oder`close`，`wait()`nicht aufgerufen wurde, kehrt es nicht zurück. Ohne es könnte man nur`JoinSet`verwenden, aber`JoinSet`akkumuliert die Rückgabewerte jeder Aufgabe, und lang laufende Dienste würden OOM bekommen.

## Datenstruktur und Speicherlayout

`TaskTracker`ist ein`Arc`-Wrapper:

[FACT:tokio-util/src/task/task_tracker.rs:158-178]

```rust
pub struct TaskTracker {
    inner: Arc,
}

/// Represents a task tracked by a [`TaskTracker`].
#[must_use]
#[derive(Debug)]
pub struct TaskTrackerToken {
    task_tracker: TaskTracker,
}

struct TaskTrackerInner {
    /// Keeps track of the state.
    ///
    /// The lowest bit is whether the task tracker is closed.
    ///
    /// The rest of the bits count the number of tracked tasks.
    state: AtomicUsize,
    /// Used to notify when the last task exits.
    on_last_exit: Notify,
}
```

Dies ist das raffinierteste Speicherlayout dieses Kapitels:**Ein`AtomicUsize`kodiert gleichzeitig „ob geschlossen" und „Aufgabenzähler"**. Das niedrigste Bit ist das Shutdown-Flag, die übrigen Bits sind die Aufgabenanzahl (da der Aufgabenzähler bei jedem`+2`, ist das niedrigste Bit immer 0). So benötigt`is_closed_and_empty`nur einen einzigen atomaren Ladevorgang:

[FACT:tokio-util/src/task/task_tracker.rs:216-222]

```rust
fn is_closed_and_empty(&self) -> bool {
    // If empty and closed bit set, then we are done.
    //
    // The acquire load will synchronize with the release store of any previous call to
    // `set_closed` and `drop_task`.
    self.state.load(Ordering::Acquire) == 1
}
```

> **[Design Inference & Architectural Trade-offs]**
> `state == 1`bedeutet „Shutdown-Bit ist 1, Zähler ist 0". Warum nicht zwei atomare Variablen? Zwei Variablen erfordern zwei Ladevorgänge und können nicht atomar feststellen, ob „beide Bedingungen gleichzeitig erfüllt sind". Die Einzelvariablen-Kodierung macht`is_closed_and_empty`zu einem einzigen`Acquire`-Ladevorgang und benötigt auf dem schnellen Pfad von`wait`keine Sperre.

## Szenario-getriebener Walkthrough: Race zwischen close und drop_task

Betrachten wir ein typisches Szenario: Der Hauptthread ruft`tracker.close()`auf, während gleichzeitig die letzte Aufgabe beendet wird (`TaskTrackerToken::drop`ruft`drop_task`auf). Beide können nebenläufig sein, und es muss garantiert werden, dass unabhängig davon, wer zuerst kommt,`wait()`aufgeweckt werden kann.

Zuerst`set_closed`：

[FACT:tokio-util/src/task/task_tracker.rs:225-249]

```rust
fn set_closed(&self) -> bool {
    // The AcqRel ordering makes the closed bit behave like a `Mutex` for synchronization
    // purposes. ...
    let state = self.state.fetch_or(1, Ordering::AcqRel);

    // If there are no tasks, and if it was not already closed:
    if state == 0 {
        self.notify_now();
    }

    (state & 1) == 0
}
```

`fetch_or(1, AcqRel)`setzt atomar das Shutdown-Bit und gibt den alten Wert zurück. Wenn der alte Wert 0 ist (vorher nicht geschlossen und keine Aufgaben), bedeutet dies „nach dem Schließen sofort leer + geschlossen erfüllt", und`notify_now`wird aufgerufen. Der Rückgabewert`(state & 1) == 0`bedeutet „dieser Aufruf hat den Zustand tatsächlich verändert".

Nun zu`drop_task`：

[FACT:tokio-util/src/task/task_tracker.rs:264-271]

```rust
fn drop_task(&self) {
    let state = self.state.fetch_sub(2, Ordering::Release);

    // If this was the last task and we are closed:
    if state == 3 {
        self.notify_now();
    }
}
```

`fetch_sub(2, Release)`dekrementiert den Zähler. Wenn der alte Wert 3 ist (binär`11`: Shutdown-Bit 1 + Zähler 1), bedeutet dies „dies ist die letzte Aufgabe und bereits geschlossen", und`notify_now`。

wird aufgerufen. Race-Analyse der beiden Pfade:

- **close wird zuerst ausgeführt**：`set_closed`sieht den alten Wert`2`(Zähler 1, nicht geschlossen), benachrichtigt nicht. Anschließend sieht`drop_task`den alten Wert`3`, benachrichtigt. ✓
- **drop_task wird zuerst ausgeführt**：`drop_task`sieht den alten Wert`2`(Zähler 1, nicht geschlossen), benachrichtigt nicht. Anschließend sieht`set_closed`den alten Wert`0`(Zähler 0, nicht geschlossen), benachrichtigt. ✓
- **Nebenläufig**：`fetch_or`und`fetch_sub`sind atomar; unabhängig von der Verschachtelungsreihenfolge wird immer einer die Kombination „geschlossen + leer" sehen und benachrichtigen. ✓

`notify_now`enthält einen leicht zu übersehenden`Acquire`-Ladevorgang:

[FACT:tokio-util/src/task/task_tracker.rs:274-285]

```rust
#[cold]
fn notify_now(&self) {
    // Insert an acquire fence. This matters for `drop_task` but doesn't matter for
    // `set_closed` since it already uses AcqRel.
    //
    // This synchronizes with the release store of any other call to `drop_task`, and with the
    // release store in the call to `set_closed`. That ensures that everything that happened
    // before those other calls to `drop_task` or `set_closed` will be visible after this load,
    // and those things will also be visible to anything woken by the call to `notify_waiters`.
    self.state.load(Ordering::Acquire);

    self.on_last_exit.notify_waiters();
}
```

Warum verwendet`drop_task``Release`statt`AcqRel`? Weil`drop_task`s`fetch_sub`nur „vorherige Schreibvorgänge für nachfolgende Leser sichtbar machen" muss (Release-Semantik) und nicht „Schreibvorgänge anderer Threads von zuvor sehen" muss (Acquire-Semantik). Aber`notify_now`benötigt Acquire, um happens-before herzustellen: Es stellt sicher, dass alle Aufräumarbeiten, die vor dem Beenden der Aufgabe durchgeführt wurden, für den Code nach der Rückkehr von`wait()`sichtbar sind. Das Ergebnis dieses`load`wird verworfen, einzig wegen seines Speicherordnungs-Nebeneffekts – dies ist eine typische Verwendung eines „fence-artigen Ladevorgangs" in Rust-Atomoperationen.

## Design-Überlegungen: ABA-Widerstand von wait und drop-Semantik von TrackedFuture

`wait`gibt ein`TaskTrackerWaitFuture`zurück, das intern`Notified`：

[FACT:tokio-util/src/task/task_tracker.rs:318-327]

```rust
pub fn wait(&self) -> TaskTrackerWaitFuture {
    TaskTrackerWaitFuture {
        future: self.inner.on_last_exit.notified(),
        inner: if self.inner.is_closed_and_empty() {
            None
        } else {
            Some(&self.inner)
        },
    }
}
```

Kopieren`inner`Beachten Sie das Feld`None`，`poll`: Wenn es beim Erstellen bereits „geschlossen und leer" ist, wird es direkt auf`Ready`gesetzt und kehrt sofort zurück

. Dies ist der schnelle Pfad.

[FACT:tokio-util/src/task/task_tracker.rs:304-307]

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

Kopieren`Notify::notified()`Diese Garantie stammt aus der Semantik von`Notified`: Der Future registriert sich bei der Erstellung als „Wartender", und selbst wenn`notify_waiters`aufgerufen wird, bevor er`poll`wird, wird er beim ersten`poll`die Benachrichtigung sehen.`TaskTrackerWaitFuture::poll`Die Implementierung von

[FACT:tokio-util/src/task/task_tracker.rs:697-712]

```rust
fn poll(self: Pin, cx: &mut Context) -> Poll {
    let me = self.project();

    let inner = match me.inner.as_ref() {
        None => return Poll::Ready(()),
        Some(inner) => inner,
    };

    let ready = inner.is_closed_and_empty() || me.future.poll(cx).is_ready();
    if ready {
        *me.inner = None;
        Poll::Ready(())
    } else {
        Poll::Pending
    }
}
```

Kopieren`poll`Jedes Mal`is_closed_and_empty()`wird zuerst`poll` `Notified`geprüft, dann`Notified`. Diese Reihenfolge garantiert: Selbst wenn

`TrackedFuture`aus irgendeinem Grund nicht aufgeweckt wird, fängt die Zustandsprüfung es auf.`TaskTracker`Die drop-Semantik von`JoinSet`ist der zentrale Unterschied zwischen

[FACT:tokio-util/src/task/task_tracker.rs:488-494]

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

:`Ready`Kopieren`TrackedFuture`Dies bedeutet: Selbst wenn der Future bereits`TaskTracker`zurückgegeben hat, solange

[FACT:tokio-util/src/task/task_tracker.rs:33-35]

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken`die Aufgabe als noch laufend. Die Dokumentation erklärt, warum dieses Design wichtig ist:`Drop`Kopieren

[FACT:tokio-util/src/task/task_tracker.rs:670-672]

```rust
impl Drop for TaskTrackerToken {
    /// Dropping the token indicates to the [`TaskTracker`] that the task has exited.
    #[inline]
    fn drop(&mut self) {
        self.task_tracker.inner.drop_task();
    }
}
```

`TrackedFuture`von`pin_project!`ist der Auslösepunkt für die Zählerdekrementierung:`token`Kopieren`future`Durch`token`werden`spawn_blocking`und

[FACT:tokio-util/src/task/task_tracker.rs:452-464]

zusammengepackt, und der drop von
