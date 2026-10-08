# Глава 11: Экосистема Stream и утилиты: механизмы расширения tokio-stream и tokio-util

В предыдущей главе мы разобрали байтовый механизм Framed: Decoder нарезает BytesMut на кадры, Sink записывает кадры обратно — так становится ясна абстрактная граница асинхронного ввода-вывода. Но кадр — это лишь контейнер данных, и реальная реализация протокола сразу же сталкивается с тремя проблемами, которые не решают ни tokio::io, ни Framed: асинхронная итерация — Framed реализует Stream, но у Stream есть только poll_next, нет next().await, filter, take, merge; писать poll_fn вручную и многословно, и легко ошибиться в безопасности отмены; динамический набор задач — чат-сервису нужно одновременно подписаться на N каналов, каналы присоединяются и покидают в любой момент, а количество ветвей select! фиксировано на этапе компиляции и не может выразить набор потоков, изменяющийся во время выполнения; структурированная отмена — select! может отменить одну ветвь, но не может распространить остановку всего дерева задач вниз и не может дождаться фактического завершения всех задач. tokio-stream и tokio-util созданы именно для этих трёх вещей, и их ключевой принцип проектирования — не изобретать заново: каждый комбинатор StreamExt — это лишь обёртка над poll_next, StreamMap переиспользует семантику регистрации Waker, CancellationToken напрямую строится поверх tokio::sync::Notify, а TaskTracker кодирует всё состояние одним AtomicUsize. Понимание их — это по сути понимание того, как делать абстракции с нулевой стоимостью поверх существующих Waker и механизмов планирования. Эта глава последовательно продвигается по трём слоям: итерация, коллекции, отмена: сначала посмотрим, как StreamExt превращает poll_next в комбинируемый итератор, затем — как StreamMap и TaskTracker управляют динамическими коллекциями, и наконец — как CancellationToken с помощью дерева распространяет сигнал отмены на всё дерево задач.

# StreamExt: превращаем poll_next в комбинируемый итератор

## Интуитивная модель

`Stream`по отношению к`Future`, как`Iterator`по отношению к значению:`Future`производит «одно значение»,`Stream`производит «последовательность значений». Но`Stream`определяет только`poll_next`этот один примитив, как`Iterator`определяет только`next`. Без`StreamExt`каждая фильтрация, отображение, усечение требовали бы ручного написания замыкания`poll_fn`и ручного управления`Pin`— именно это было самым болезненным местом для ранних пользователей crate`futures`.`StreamExt`Роль`Stream`— снабдить`Iterator`такой экосистемой комбинаторов, как у

. Без неё система сталкивается не с отсутствием функциональности, а с**системным обрушением безопасности отмены**: каждый написанный вручную`poll_fn`может при отмене через`select!`потерять уже`poll`элемент.

## Структуры данных и размещение в памяти

`StreamExt`— это**расширяющий trait**, сам по себе не хранящий данных:

[FACT:tokio-stream/src/stream_ext.rs:106-106]

```rust
pub trait StreamExt: Stream {
```

Все его методы возвращают**конкретную структуру-комбинатор**, а не`Box<dyn Stream>`. Это ключевое проектное решение:`map`возвращает`Map<Self, F>`，`filter`возвращает`Filter<Self, F>`，`take`возвращает`Take<Self>`. Эти структуры — обобщённые обёртки с нулевым выделением памяти в куче; компилятор может встроить всю цепочку в слой за слоем вызовов`poll_next`.

Обратите внимание на blanket impl трейта:

[FACT:tokio-stream/src/stream_ext.rs:1213-1213]

```rust
impl StreamExt for St where St: Stream {}
```

Любой`Stream`автоматически получает все комбинаторы без ручной реализации.`?Sized`позволяет`dyn Stream`также пользоваться методами расширения.

Объявления модулей комбинаторов раскрывают полную поверхность возможностей этого трейта:

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

Здесь есть值得注意 различие:`next`、`try_next`、`all`、`any`、`fold`、`collect`возвращает**Future**（`Next`、`TryNext`、`AllFuture`…), потому что они потребляют весь поток в одно значение; а`map`、`filter`、`take`и т.п. возвращают**Stream**, потому что сохраняют форму потока.`next`Тип возвращаемого значения —`Next<'_, Self>`, с параметром времени жизни, потому что он лишь заимствует поток:

[FACT:tokio-stream/src/stream_ext.rs:144-149]

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin`Ограничение`next`намеренно:`Pin`не получает владение потоком, только заимствует, поэтому нельзя`!Unpin`поток. Если поток —`Box::pin`, пользователь должен сначала`pin_mut!`или

[FACT:tokio-stream/src/stream_ext.rs:116-121]

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## копировать`merge`опрос

`merge`— лучший пример для понимания того, как комбинаторы переиспользуют Waker. Он чередует вывод двух потоков и**гарантирует справедливость**— если оба потока готовы одновременно, вывод чередуется. Документация специально предупреждает против цепочечного вызова`merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321]

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge`Сигнатура требует, чтобы оба потока имели одинаковый`Item`тип:

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

Когда вызывающая сторона`.next().await`, поток выполнения следующий:

1. `Next::poll`вызов`Merge::poll_next`。

2. `Merge`Внутри поддерживается булев флаг «чей черёд был в прошлый раз». Сначала`poll`тот поток, который не выдал значение в прошлый раз; если`Pending`, затем`poll`другой.

3. Если оба`Pending`，`Merge`возвращают`Pending`, но**Waker каждого из двух потоков уже зарегистрированы**— готовность любого разбудит текущую задачу.

4. Если один поток возвращает`Ready(None)`(завершение),`Merge`записывает, что этот поток завершён, и после этого`poll`только другой поток, пока и он не завершится.

Ключевой момент здесь:`Merge`не имеет собственной логики управления Waker, он передаёт`cx`как есть внутренним двум потокам`poll_next`。**Регистрация Waker полностью лежит на нижележащих потоках**，`Merge`только решает, «кого спросить первым на этот раз». Это и есть буквальный смысл «переиспользования механизма Waker нижележащих потоков».

`merge_size_hints`Вспомогательная функция показывает, как комбинатор объединяет подсказки о ёмкости:

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

Обратите внимание на выбор между`saturating_add`и`checked_add`: для нижней границы используется насыщающее сложение (лучше недооценить, чем переполниться с panic), для верхней — проверяемое сложение (если хоть одно неизвестно, то всё неизвестно). Это типичный способ обработки контракта`size_hint`.

## Размышления о дизайне: безопасность отмены и защита от panic в`chunks_timeout`

`StreamExt`Документация помечает для каждого метода**Cancel safety**. Возьмём`next`в качестве примера:

[FACT:tokio-stream/src/stream_ext.rs:123-127]

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next`безопасен для отмены, потому что он только заимствует поток и не потребляет элементы —`Next`когда future уничтожается, состояние самого потока не меняется, и следующий`next`заново`poll`。

Но не все комбинаторы безопасны для отмены.`chunks_timeout`выполняет проверку параметров уже при конструировании:

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
> `#[track_caller]`заставляет место panic указывать на вызывающую сторону, а не на внутренности библиотеки,`assert!`отвергает`max_size == 0`уже на этапе конструирования. Почему проверка обязана быть на этапе конструирования? Если разрешить`max_size == 0`，`ChunksTimeout`логика пакетной обработки попадёт в бесконечный цикл «никогда не набрать полный пакет» или будет выдавать пустые пакеты, а такие баги крайне трудно локализовать во время выполнения. Panic на этапе конструирования переносит ошибку в самую раннюю наблюдаемую точку.

`timeout`Различие между`timeout_repeating`и`timeout`также заслуживает внимания:**возвращает ошибку после тайм-аута, но**；`timeout_repeating`продолжает опрашивать внутренний поток`Interval`же в соответствии с

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

# копия

## StreamMap: динамический набор потоков и справедливый опрос

`select!`Интуитивная модель`StreamMap`Количество ветвей фиксировано на этапе компиляции. Но количество каналов, на которые нужно подписаться чат-сервису, и количество соединений, которые должен отслеживать краулер, становятся известны только во время выполнения.`select!`— это «`next`, который можно добавлять и удалять во время выполнения»: он помещает произвольное количество потоков в набор, и каждый`(key, value)`возвращает`mpsc`, сообщая, из какого потока пришло значение. Без него вам пришлось бы запихнуть все потоки в один

## канал, что добавило бы лишние накладные расходы на пересылку.

`StreamMap`Структура данных и размещение в памяти`Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208]

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

предельно простое — один

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
> копия`HashMap`〔Проектные соображения и архитектурные компромиссы〕`StreamMap`Почему не используется**? Потому что ключевая операция**—`Vec`опрос всех потоков`swap_remove`, а не поиск по ключу.`HashMap`Линейное сканирование дружественно к кэшу CPU, и`poll_next`имеет сложность O(1). Если бы использовался`insert`, каждый`remove`требовал бы обхода хэш-корзин, что ухудшило бы локальность кэша.

`insert`и

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

`remove`Реализация`swap_remove`воплощает семантику «сначала удалить, потом вставить»:

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

## использует

`StreamMap`чтобы обменять удаляемый элемент с последним и затем извлечь его, избегая O(n) перемещений:`poll_next_entry`копия**Сценарный Walkthrough: случайная начальная точка и коррекция курсора в poll_next_entry**Ядро

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

. Он начинает опрос со

**случайной начальной точки** `thread_rng_n`, чтобы гарантировать справедливость — если всегда начинать с индекса 0, первый поток заморит голодом остальные:`FastRand`копия`xorshift64+`В этом коде три тонких момента, разберём их по порядку:

[FACT:tokio-stream/src/stream_map.rs:765-768]

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n`использует потоково-локальный`% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792]

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**:`swap_remove`копия**использует умножение и взятие по модулю Лемьера вместо`idx`копия`None`Второй —`swap_remove`коррекция курсора после`idx`. Когда поток с индексом**возвращает**и удаляется,`start`перемещает последний элемент на место`idx < start && start <= self.entries.len()`. Этот перемещённый элемент может`idx = idx.wrapping_add(1) % len`уже быть опрошенным`idx == len`(если его исходный индекс был до

**). Код использует`Poll::Pending`для обнаружения этого случая и, если так, пропускает его (**). Если удалён последний элемент (`Pending`), курсор сбрасывается на 0.

`poll_next`Третий —`poll_next_entry`семантика

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

. В этот момент Waker всех потоков уже зарегистрированы, и готовность любого разбудит задачу.`ready!`поверх`poll_next_entry`добавляет key:`Pending`копия`poll_next`Обратите внимание на макрос`Pending`。`K: Clone`: если`key.clone()`。

## возвращает

`next_many`, весь`StreamMap`немедленно возвращает

[FACT:tokio-stream/src/stream_map.rs:581-583]

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

Размышления о дизайне: пакетная семантика next_many и безопасность отмены

[FACT:tokio-stream/src/stream_map.rs:573-578]

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

пакетная версия`next_many`, собирающая как можно больше готовых элементов за один раз:**копия`buffer`**Его гарантия безопасности отмены критически важна:`buffer`копия`buffer`Почему

`poll_next_many`безопасен для отмены? Потому что он`poll_next_entry`немедленно помещает элементы в предоставленный вызывающей стороной

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

и не теряются. Но это также означает: при уничтожении`while added < limit`может уже содержать часть элементов — вызывающая сторона должна это знать.`for`Структура цикла в`should_loop = true`сложнее, чем в`limit`, потому что он должен собрать как можно больше за один проход:

[FACT:tokio-stream/src/stream_map.rs:588-591]

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint`реализация демонстрирует, как агрегировать подсказки о ёмкости нескольких потоков:

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

Такой же шаблон, как и`merge_size_hints`: насыщающее сложение для нижней границы, проверяемое сложение для верхней границы, и если любая из них неизвестна, то всё неизвестно.

Ниже приведена блок-схема, иллюстрирующая`poll_next_entry`путь принятия решений:

```mermaid
)
    Pipe->>RtA: событие доступности для чтения
    Pipe->>RtB: событие доступности для чтения
    Note over RtA,RtB: оба Runtime конкурируют за чтение, прочитать байт может только один
```

---

# TaskTracker: кодирование всего состояния в одном AtomicUsize

## Интуитивная модель

Для корректного завершения нужны две вещи:**Уведомить задачи о необходимости остановиться**（`CancellationToken`отвечает за это), а также**Дождаться фактического выхода задач**（`TaskTracker`отвечает за это).`TaskTracker`подобен объединению «счётчика задач + переключателя завершения»: он не вернётся, пока есть работающие задачи или пока не вызван`close`，`wait()`Без него пришлось бы использовать`JoinSet`, но`JoinSet`накапливает возвращаемые значения каждой задачи, что при длительной работе сервиса приведёт к OOM.

## Структура данных и размещение в памяти

`TaskTracker`представляет собой обёртку над`Arc`:

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

Это самое изящное размещение в памяти в данной главе:**Один`AtomicUsize`одновременно кодирует «закрыто ли» и «количество задач»**. Младший бит — флаг закрытия, остальные биты — количество задач (поскольку счётчик задач каждый раз`+2`, младший бит всегда равен 0). Таким образом,`is_closed_and_empty`требует всего одной атомарной загрузки:

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
> `state == 1`означает «бит закрытия равен 1, счётчик равен 0». Почему бы не использовать две атомарные переменные? Две переменные требуют двух загрузок и не позволяют атомарно определить «одновременное выполнение обоих условий». Кодирование в одной переменной делает`is_closed_and_empty`одной`Acquire`загрузкой и не требует блокировки на быстром пути`wait`.

## Сценарий-ориентированный разбор: гонка между close и drop_task

Рассмотрим типичный сценарий: главный поток вызывает`tracker.close()`, одновременно последняя задача завершается (`TaskTrackerToken::drop`вызывает`drop_task`). Оба могут выполняться параллельно, и необходимо гарантировать, что независимо от того, кто первый,`wait()`будет разбужен.

Сначала рассмотрим`set_closed`：

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

`fetch_or(1, AcqRel)`атомарно устанавливает бит закрытия и возвращает старое значение. Если старое значение равно 0 (ранее не закрыто и нет задач), это означает «после закрытия немедленно выполнено условие пусто+закрыто», вызывается`notify_now`. Возвращаемое значение`(state & 1) == 0`означает «этот вызов действительно изменил состояние».

Теперь рассмотрим`drop_task`：

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

`fetch_sub(2, Release)`уменьшает счётчик. Если старое значение равно 3 (двоичное`11`: бит закрытия 1 + счётчик 1), это означает «это последняя задача и уже закрыто», вызывается`notify_now`。

Анализ гонки двух путей:

- **close выполняется первым**：`set_closed`видит старое значение`2`(счётчик 1, не закрыто), не уведомляет. Затем`drop_task`видит старое значение`3`, уведомляет. ✓
- **drop_task выполняется первым**：`drop_task`видит старое значение`2`(счётчик 1, не закрыто), не уведомляет. Затем`set_closed`видит старое значение`0`(счётчик 0, не закрыто), уведомляет. ✓
- **Параллельно**：`fetch_or`и`fetch_sub`атомарны, независимо от порядка чередования, всегда найдётся один, который увидит комбинацию «закрыто + пусто» и уведомит. ✓

`notify_now`Внутри`Acquire`есть легко упускаемая

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

Копировать`drop_task`Почему`Release`использует`AcqRel`, а не`drop_task`? Потому что`fetch_sub`для`notify_now`требует лишь «сделать предыдущие записи видимыми для последующих читателей» (семантика Release), а не «увидеть записи других потоков, сделанные ранее» (семантика Acquire). Но`wait()`требует Acquire для установления happens-before: гарантировать, что вся работа по очистке, выполненная до выхода задачи, видна коду после возврата`load`. Результат этой

## отбрасывается исключительно ради её побочного эффекта на порядок памяти — это типичное использование «fence-подобной загрузки» в атомарных операциях Rust.

`wait`Размышления о дизайне: устойчивость wait к ABA и семантика drop у TrackedFuture`TaskTrackerWaitFuture`возвращает`Notified`：

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

Копировать`inner`Обратите внимание на поле`None`，`poll`: если при создании уже «закрыто и пусто», оно сразу устанавливается в`Ready`и немедленно возвращает

. Это быстрый путь.

[FACT:tokio-util/src/task/task_tracker.rs:304-307]

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

Копировать`Notify::notified()`Эта гарантия следует из семантики`Notified`:`notify_waiters`future при создании регистрирует себя как «ожидающий», поэтому даже если`poll`был вызван до того, как он был`poll`, он увидит уведомление при первом`TaskTrackerWaitFuture::poll`.

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

:`poll`Копировать`is_closed_and_empty()`Каждый раз`poll` `Notified`сначала проверяет`Notified`, затем

`TrackedFuture`. Этот порядок гарантирует: даже если`TaskTracker`по какой-то причине не был разбужен, проверка состояния послужит запасным вариантом.`JoinSet`Семантика drop у

[FACT:tokio-util/src/task/task_tracker.rs:488-494]

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

и`Ready`:`TrackedFuture`Копировать`TaskTracker`Это означает: даже если future уже вернул

[FACT:tokio-util/src/task/task_tracker.rs:33-35]

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken`сам ещё не был drop,`Drop`считает, что задача всё ещё выполняется. В документации объясняется, почему этот дизайн важен:

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

`TrackedFuture`для`pin_project!`является точкой запуска уменьшения счётчика:`token`Копировать`future`через`token`упаковывает`spawn_blocking`и

[FACT:tokio-util/src/task/task_tracker.rs:452-464]

, и drop
