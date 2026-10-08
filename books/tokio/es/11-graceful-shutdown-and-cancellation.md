# Capítulo 11: Ecosistema de Stream y capa de herramientas: mecanismos de extensión de tokio-stream y tokio-util

En el capítulo anterior desglosamos el mecanismo a nivel de bytes de Framed: Decoder corta BytesMut en tramas, Sink escribe las tramas de vuelta, y así el límite de abstracción de la E/S asíncrona queda claro. Pero una trama es solo un contenedor de datos; una implementación real de protocolo se encontrará inmediatamente con tres problemas que ni tokio::io ni Framed resuelven: iteración asíncrona — Framed implementa Stream, pero Stream solo tiene poll_next, no tiene next().await, filter, take, merge; escribir poll_fn a mano es verboso y propenso a errores de seguridad ante cancelación; conjuntos dinámicos de tareas — un servicio de chat necesita suscribirse simultáneamente a N canales, los canales se unen y salen en cualquier momento, y el número de ramas de select! es fijo en tiempo de compilación, incapaz de expresar conjuntos de flujos que aumentan o disminuyen en tiempo de ejecución; cancelación estructurada — select! puede cancelar una sola rama, pero no puede propagar la detención de todo el árbol de tareas, ni puede esperar a que todas las tareas realmente terminen. tokio-stream y tokio-util nacen precisamente para estas tres cosas, y su principio de diseño clave es no empezar desde cero: cada combinador de StreamExt es solo un envoltorio sobre poll_next, StreamMap reutiliza la semántica de registro de Waker, CancellationToken se construye directamente sobre tokio::sync::Notify, TaskTracker codifica todo el estado con un AtomicUsize. Entenderlos es, en esencia, entender cómo hacer abstracciones de coste cero sobre los mecanismos existentes de Waker y scheduling. Este capítulo avanza progresivamente en tres capas: iteración, colecciones y cancelación: primero veremos cómo StreamExt convierte poll_next en un iterador componible, luego cómo StreamMap y TaskTracker gestionan colecciones dinámicas, y finalmente cómo CancellationToken propaga la señal de cancelación a todo el árbol de tareas mediante un árbol.

# StreamExt: convertir poll_next en un iterador componible

## Modelo intuitivo

`Stream`es a`Future`, como`Iterator`es a un valor:`Future`produce «un valor»,`Stream`produce «una secuencia de valores». Pero`Stream`solo define`poll_next`esta única primitiva, igual que`Iterator`solo define`next`. Si no existiera`StreamExt`, cada filtrado, mapeo o truncamiento requeriría escribir a mano un cierre`poll_fn`y gestionar manualmente`Pin`—esto es precisamente lo más doloroso para los primeros usuarios del crate`futures`.`StreamExt`El papel de`Stream`es dotar a`Iterator`de un ecosistema de combinadores como el de

. Sin él, el desastre al que se enfrenta el sistema no es la falta de funcionalidad, sino**el colapso sistémico de la seguridad ante cancelación**: cada`poll_fn`escrito a mano puede, al ser cancelado por`select!`, perder un elemento que ya había sido`poll`.

## Estructura de datos y diseño de memoria

`StreamExt`es un**trait de extensión**, que en sí mismo no contiene datos:

[FACT:tokio-stream/src/stream_ext.rs:106-106]

```rust
pub trait StreamExt: Stream {
```

Todos sus métodos devuelven una**estructura concreta de combinador**, en lugar de`Box<dyn Stream>`. Esta es la decisión de diseño clave:`map`devuelve`Map<Self, F>`，`filter`devuelve`Filter<Self, F>`，`take`devuelve`Take<Self>`. Estas estructuras son envoltorios genéricos sin asignación en heap, y el compilador puede inline toda la cadena en capas de llamadas a`poll_next`.

Nótese el blanket impl del trait:

[FACT:tokio-stream/src/stream_ext.rs:1213-1213]

```rust
impl StreamExt for St where St: Stream {}
```

Cualquier`Stream`obtiene automáticamente todos los combinadores, sin necesidad de implementación manual.`?Sized`permite que`dyn Stream`también disfrute de los métodos de extensión.

La declaración de módulos de los combinadores revela la superficie completa de capacidades de este trait:

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

Aquí hay una distinción que merece atención:`next`、`try_next`、`all`、`any`、`fold`、`collect`devuelve**Future**（`Next`、`TryNext`、`AllFuture`…), porque consumen todo el flujo en un solo valor; mientras que`map`、`filter`、`take`etc. devuelven**Stream**, porque mantienen la forma del flujo.`next`El tipo de retorno de`Next<'_, Self>`es

[FACT:tokio-stream/src/stream_ext.rs:144-149]

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin`Copiar`next`La restricción de`Pin`es deliberada:`!Unpin`no toma posesión del flujo, solo lo toma prestado, por lo que no puede`Box::pin`el flujo. Si el flujo es`pin_mut!`, el usuario debe primero

[FACT:tokio-stream/src/stream_ext.rs:116-121]

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## . La documentación señala explícitamente este compromiso:`merge`del polling

`merge`es el mejor ejemplo para entender cómo los combinadores reutilizan el Waker. Intercala la producción de dos flujos, y**garantiza la equidad**——si ambos flujos están listos a la vez, produce alternadamente. La documentación advierte específicamente contra el encadenamiento de llamadas`merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321]

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge`de la firma requiere que ambos flujos tengan el mismo tipo`Item`:

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

Cuando el llamador`.next().await`, el flujo de ejecución es el siguiente:

1. `Next::poll`llama a`Merge::poll_next`。

2. `Merge`mantiene internamente una bandera booleana de «a quién le tocó la última vez». Primero`poll`el flujo que no produjo la última vez; si`Pending`, entonces`poll`el otro.

3. Si ambos`Pending`，`Merge`devuelven`Pending`, pero**los Wakers de ambos flujos ya están registrados**——cualquiera que esté listo despertará la tarea actual.

4. Si un flujo devuelve`Ready(None)`(fin),`Merge`registra que ese flujo terminó, y a partir de entonces solo`poll`el otro flujo, hasta que también termine.

La clave aquí es:`Merge`no tiene su propia lógica de gestión de Waker; pasa`cx`tal cual a los dos flujos internos`poll_next`。**El registro del Waker corre completamente a cargo de los flujos subyacentes**，`Merge`solo decide «a quién preguntar primero esta vez». Este es el significado literal de «reutilizar el mecanismo de Waker subyacente».

`merge_size_hints`La función auxiliar muestra cómo los combinadores fusionan las pistas de capacidad:

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

Nótese la elección entre`saturating_add`y`checked_add`: para el límite inferior se usa suma saturante (mejor subestimar que desbordar con panic), para el límite superior se usa suma verificada (si alguno es desconocido, el total es desconocido). Esta es la forma típica de manejar el contrato de`size_hint`.

## Reflexión de diseño: seguridad ante cancelación y protección contra panic de`chunks_timeout`

`StreamExt`La documentación de**Cancel safety**anota`next`en cada método. Tomemos

[FACT:tokio-stream/src/stream_ext.rs:123-127]

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next`Copia`Next`es seguro ante cancelación porque solo toma prestado el flujo, no consume elementos——`next`cuando el future se descarta, el estado del propio flujo no cambia, y la próxima vez`poll`。

volverá a`chunks_timeout`Pero no todos los combinadores son seguros ante cancelación.

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
> `#[track_caller]`〔Inferencia de diseño y compensaciones arquitectónicas〕`assert!`hace que la ubicación del panic apunte al llamador en lugar del interior de la biblioteca,`max_size == 0`rechaza`max_size == 0`，`ChunksTimeout`en la fase de construcción. ¿Por qué debe verificarse en la fase de construcción? Si se permitiera

`timeout`, la lógica de procesamiento por lotes caería en un bucle infinito de «nunca acumular un lote completo» o produciría lotes vacíos, y este tipo de bug es extremadamente difícil de localizar en tiempo de ejecución. El panic en la fase de construcción adelanta el error al punto observable más temprano.`timeout_repeating`La diferencia entre`timeout`y**también merece atención:**；`timeout_repeating`devuelve un error tras el timeout, pero`Interval`continúa haciendo polling del flujo interno

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

# Copia

## Copia

`select!`StreamMap: colección dinámica de flujos y polling equitativo`StreamMap`Modelo intuitivo`select!`tiene un número de ramas fijo en tiempo de compilación. Pero el número de canales a los que se debe suscribir un servicio de chat, o el número de conexiones que debe rastrear un crawler, solo se conocen en tiempo de ejecución.`next`es precisamente «`(key, value)`que se puede añadir o eliminar en tiempo de ejecución»: coloca cualquier cantidad de flujos en una colección, y cada vez`mpsc`devuelve

## , indicándote de qué flujo proviene el valor. Sin él, solo podrías meter todos los flujos en un

`StreamMap`canal, con una capa adicional de sobrecarga de reenvío.`Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208]

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

El almacenamiento de

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
> La documentación explica claramente el costo de esta elección:`HashMap`Copia`StreamMap`〔Inferencia de diseño y compensaciones arquitectónicas〕**¿Por qué no usar**? Porque la operación central de`Vec`es`swap_remove`hacer polling de todos los flujos`HashMap`, no buscar por clave.`poll_next`El escaneo lineal de`insert`es amigable con la caché de CPU, y`remove`es O(1). Si se usara

`insert`, cada

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

`remove`El escaneo O(n) de`swap_remove`y

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

## La implementación de

`StreamMap`refleja la semántica de «eliminar primero, insertar después»:`poll_next_entry`Copia**usa**para intercambiar el elemento eliminado con el último elemento y luego hacer pop, evitando el desplazamiento O(n):

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

Walkthrough guiado por escenarios: punto de inicio aleatorio y corrección del cursor en poll_next_entry

**El núcleo de** `thread_rng_n`es`FastRand`. Comienza el polling desde`xorshift64+`un punto de inicio aleatorio

[FACT:tokio-stream/src/stream_map.rs:765-768]

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n`Copia`% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792]

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**Primero, el punto de inicio aleatorio.`swap_remove`usa**thread-local`idx`, basado en el algoritmo`None`:`swap_remove`Copia`idx`usa la multiplicación y módulo de Lemire en lugar de**Copia**Segundo,`start`la corrección del cursor tras`idx < start && start <= self.entries.len()`. Cuando el flujo en el índice`idx = idx.wrapping_add(1) % len`devuelve`idx == len`y es eliminado,

**mueve el último elemento a`Poll::Pending`. Este elemento movido puede**ya haber sido sometido a polling`Pending`(si su índice original estaba antes de

`poll_next`). El código usa`poll_next_entry`para detectar esta situación y, si es así, lo salta (

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

), el cursor vuelve a 0.`ready!`Tercero,`poll_next_entry`la semántica de`Pending`. Si tras recorrer una vuelta ningún flujo está listo y la colección no está vacía, devuelve`poll_next`. En ese momento los Wakers de todos los flujos ya están registrados, y cualquiera que esté listo despertará.`Pending`。`K: Clone`añade la clave sobre`key.clone()`。

## :

`next_many`Copia`StreamMap`Nótese la macro

[FACT:tokio-stream/src/stream_map.rs:581-583]

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

devuelve

[FACT:tokio-stream/src/stream_map.rs:573-578]

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

devuelve inmediatamente`next_many`La restricción proviene de aquí**Reflexión de diseño: semántica por lotes y seguridad ante cancelación de next_many`buffer`**es la versión por lotes de`buffer`, que recoge tantos elementos listos como sea posible de una vez:`buffer`Copia

`poll_next_many`Su garantía de seguridad ante cancelación es crucial:`poll_next_entry`Copia

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

es seguro ante cancelación? Porque hace`while added < limit`push de los elementos inmediatamente en el`for`proporcionado por el llamador, en lugar de almacenarlos temporalmente en el interior. Si el future se descarta, los elementos ya insertados siguen en`should_loop = true`, sin perderse. Pero esto también implica que: al descartarse,`limit`puede que ya tenga algunos elementos——el llamador necesita saberlo.

[FACT:tokio-stream/src/stream_map.rs:588-591]

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint`La implementación de muestra cómo agregar las pistas de capacidad de múltiples flujos:

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

Igual que`merge_size_hints`el mismo patrón: saturación del límite inferior con suma, verificación del límite superior con suma, y si alguno es desconocido, el conjunto es desconocido.

A continuación, se describe con un diagrama de flujo`poll_next_entry`la ruta de decisión de:

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

# TaskTracker: codificar todo el estado con un único AtomicUsize

## Modelo intuitivo

El cierre elegante requiere dos cosas:**notificar a las tareas que se detengan**（`CancellationToken`se encarga de ello), y**esperar a que las tareas realmente salgan**（`TaskTracker`se encarga de ello).`TaskTracker`Es como una combinación de «contador de tareas + interruptor de cierre»: mientras haya tareas en ejecución, o no se haya llamado a`close`，`wait()`no retornará. Sin él, solo podrías usar`JoinSet`pero`JoinSet`acumularía el valor de retorno de cada tarea, y un servicio de larga duración sufriría OOM.

## Estructura de datos y diseño de memoria

`TaskTracker`es un envoltorio de`Arc`:

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

Este es el diseño de memoria más ingenioso de este capítulo:**un`AtomicUsize`codifica simultáneamente «si está cerrado» y «el conteo de tareas»**. El bit menos significativo es la bandera de cierre, y los bits restantes son el número de tareas (porque el conteo de tareas se incrementa cada vez`+2`, el bit menos significativo siempre es 0). Así,`is_closed_and_empty`solo necesita una carga atómica:

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
> `state == 1`significa «bit de cierre en 1, conteo en 0». ¿Por qué no usar dos variables atómicas? Dos variables requerirían dos cargas y no podrían determinar atómicamente «que se cumplan ambas condiciones a la vez». La codificación en una sola variable hace que`is_closed_and_empty`sea una única carga`Acquire`, y en la ruta rápida de`wait`no se necesita bloqueo.

## Walkthrough guiado por escenarios: la carrera entre close y drop_task

Considera un escenario típico: el hilo principal llama a`tracker.close()`, mientras la última tarea está saliendo (`TaskTrackerToken::drop`llama a`drop_task`). Ambos pueden ser concurrentes, y debe garantizarse que, sin importar quién vaya primero,`wait()`pueda ser despertado.

Primero veamos`set_closed`：

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

`fetch_or(1, AcqRel)`establece atómicamente el bit de cierre y devuelve el valor antiguo. Si el valor antiguo es 0 (no estaba cerrado y no había tareas), significa que «tras el cierre se cumple inmediatamente vacío + cerrado», y se llama a`notify_now`. El valor de retorno`(state & 1) == 0`indica que «esta llamada realmente cambió el estado».

Ahora veamos`drop_task`：

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

`fetch_sub(2, Release)`decrementa el conteo. Si el valor antiguo es 3 (binario`11`: bit de cierre 1 + conteo 1), significa que «esta es la última tarea y ya está cerrado», y se llama a`notify_now`。

Análisis de la carrera entre las dos rutas:

- **close se ejecuta primero**：`set_closed`ve el valor antiguo`2`(conteo 1, no cerrado), no notifica. Luego`drop_task`ve el valor antiguo`3`, notifica. ✓
- **drop_task se ejecuta primero**：`drop_task`ve el valor antiguo`2`(conteo 1, no cerrado), no notifica. Luego`set_closed`ve el valor antiguo`0`(conteo 0, no cerrado), notifica. ✓
- **Concurrencia**：`fetch_or`y`fetch_sub`son atómicas; sin importar el orden de intercalado, siempre habrá una que vea la combinación «cerrado + vacío» y notifique. ✓

`notify_now`Hay una carga`Acquire`fácil de pasar por alto en

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

Copiar`drop_task`¿Por qué`Release`usa`AcqRel`en lugar de`drop_task`? Porque el`fetch_sub`de`notify_now`solo necesita «hacer visibles las escrituras previas para lectores posteriores» (semántica Release), no necesita «ver las escrituras previas de otros hilos» (semántica Acquire). Pero`wait()`necesita Acquire para establecer happens-before: garantizar que todo el trabajo de limpieza realizado antes de que la tarea salga sea visible para el código posterior al retorno de`load`. El resultado de este

## se descarta, puramente por su efecto secundario de ordenamiento de memoria; este es el uso típico de «carga tipo fence» en las operaciones atómicas de Rust.

`wait`Reflexión de diseño: la resistencia a ABA de wait y la semántica de drop de TrackedFuture`TaskTrackerWaitFuture`devuelve un`Notified`：

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

Copiar`inner`Nota el campo`None`，`poll`: si al crearse ya está «cerrado y vacío», se establece directamente como`Ready`y retorna inmediatamente

. Esta es la ruta rápida.

[FACT:tokio-util/src/task/task_tracker.rs:304-307]

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

Copiar`Notify::notified()`Esta garantía proviene de la semántica de`Notified`:`notify_waiters`el future registra su identidad de «esperador» en el momento de su creación; incluso si`poll`se llama antes de que sea`poll`, verá la notificación en su primer`TaskTrackerWaitFuture::poll`.

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

Copiar`poll`Cada vez que`is_closed_and_empty()`primero verifica`poll` `Notified`, y luego`Notified`. Este orden garantiza que: incluso si

`TrackedFuture`no es despertado por alguna razón, la verificación de estado también sirve como respaldo.`TaskTracker`La semántica de drop de`JoinSet`es la diferencia central entre

[FACT:tokio-util/src/task/task_tracker.rs:488-494]

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

:`Ready`Copiar`TrackedFuture`Esto significa que: incluso si el future ya ha retornado`TaskTracker`, mientras

[FACT:tokio-util/src/task/task_tracker.rs:33-35]

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken`considera que la tarea sigue existiendo. La documentación explica por qué este diseño es importante:`Drop`Copiar

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

`TrackedFuture`de`pin_project!`es el punto de activación del decremento del conteo:`token`Copiar`future`empaqueta`token`y`spawn_blocking`mediante

[FACT:tokio-util/src/task/task_tracker.rs:452-464]

, y el drop de
