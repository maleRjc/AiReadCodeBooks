# Chapter 11: The Stream Ecosystem and Tooling Layer: Extension Mechanisms of tokio-stream and tokio-util

In the previous chapter, we broke down the byte-level mechanics of Framed: Decoder splits BytesMut into frames, Sink writes frames back, and the abstraction boundary of asynchronous I/O becomes clear. But frames are only containers for data, and real protocol implementations immediately encounter three problems that neither tokio::io nor Framed solves: asynchronous iteration—Framed implements Stream, but Stream only has poll_next, not next().await, filter, take, or merge, and hand-writing poll_fn is both verbose and prone to pitfalls in cancellation safety; dynamic task sets—a chat service needs to subscribe to N channels simultaneously, with channels joining and leaving at any time, while the number of branches in select! is fixed at compile time and cannot express a stream set that changes at runtime; structured cancellation—select! can cancel a single branch, but it cannot propagate the shutdown of the entire task tree, nor can it wait for all tasks to actually exit. tokio-stream and tokio-util were born precisely for these three things, and their key design principle is not to start from scratch: every combinator in StreamExt is just a wrapper around poll_next, StreamMap reuses the registration semantics of Waker, CancellationToken is built directly on top of tokio::sync::Notify, and TaskTracker encodes all state with an AtomicUsize. Understanding them is essentially understanding how to build zero-cost abstractions on top of the existing Waker and scheduling mechanisms. This chapter progresses through three layers: iteration, collections, and cancellation: first we look at how StreamExt turns poll_next into a composable iterator, then at how StreamMap and TaskTracker manage dynamic collections, and finally at how CancellationToken uses a tree to propagate cancellation signals to the entire task tree.

# StreamExt: Turning poll_next into a composable iterator

## Intuitive model

`Stream`is to`Future`as`Iterator`is to values:`Future`produces "one value,"`Stream`produces "a sequence of values." But`Stream`defines only`poll_next`this one primitive, just as`Iterator`defines only`next`. Without`StreamExt`, every filter, map, and truncate operation would require hand-writing a`poll_fn`closure and manually managing`Pin`—this was exactly the most painful part for early users of the`futures`crate.`StreamExt`The role of`Stream`is to give`Iterator`a combinator ecosystem like that of

. Without it, the disaster the system faces is not missing functionality, but**a systemic collapse of cancellation safety**: every hand-written`poll_fn`may lose an element that has already been`select!`when it is cancelled by`poll`.

## Data structures and memory layout

`StreamExt`is an**extension trait**, and it holds no data itself:

[FACT:tokio-stream/src/stream_ext.rs:106-106]

```rust
pub trait StreamExt: Stream {
```

All of its methods return a**concrete combinator struct**, not`Box<dyn Stream>`. This is the key design:`map`returns`Map<Self, F>`，`filter`returns`Filter<Self, F>`，`take`returns`Take<Self>`. These structs are all zero-heap-allocation generic wrappers, and the compiler can inline the entire chain into layers of`poll_next`calls.

Note the blanket impl of the trait:

[FACT:tokio-stream/src/stream_ext.rs:1213-1213]

```rust
impl StreamExt for St where St: Stream {}
```

Any`Stream`automatically gains all combinators, with no manual implementation required.`?Sized`allows`dyn Stream`to also enjoy extension methods.

The module declarations of the combinators reveal the full capability surface of this trait:

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

There is a noteworthy distinction here:`next`、`try_next`、`all`、`any`、`fold`、`collect`returns**Future**（`Next`、`TryNext`、`AllFuture`...), because they consume the entire stream into a single value; while`map`、`filter`、`take`and others return**Stream**, because they preserve the shape of the stream.`next`The return type of`Next<'_, Self>`is

[FACT:tokio-stream/src/stream_ext.rs:144-149]

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin`Copy`next`The`Pin`constraint is deliberate:`!Unpin`does not take ownership of the stream, only borrows it, and therefore cannot`Box::pin`the stream. If the stream is`pin_mut!`, the user must first

[FACT:tokio-stream/src/stream_ext.rs:116-121]

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## . The documentation explicitly points out this tradeoff:`merge`polling of

`merge`is the best example for understanding how combinators reuse Waker. It interleaves the outputs of two streams, and**guarantees fairness**— if both streams are ready simultaneously, it alternates outputs. The documentation specifically warns against chaining calls to`merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321]

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge`requires that both streams have the same`Item`type:

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

When the caller`.next().await`the execution flow is as follows:

1. `Next::poll`calls`Merge::poll_next`。

2. `Merge`internally maintains a boolean flag for "whose turn it was last time." It first`poll`the stream that did not produce last time; if`Pending`then`poll`the other one.

3. If both`Pending`，`Merge`return`Pending`but**both streams' respective Wakers have been registered**— either becoming ready will wake the current task.

4. If one stream returns`Ready(None)`(finished),`Merge`records that the stream has ended, and thereafter only`poll`the other stream until it also ends.

The key here is:`Merge`has no Waker management logic of its own; it passes`cx`as-is to the internal two streams'`poll_next`。**Waker registration is entirely handled by the underlying streams**，`Merge`only decides "whom to ask first this time." This is the literal meaning of "reusing the underlying Waker mechanism."

`merge_size_hints`The helper function demonstrates how combinators merge capacity hints:

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

Note the choice of`saturating_add`and`checked_add`: the lower bound uses saturating addition (better to underestimate than to overflow and panic), and the upper bound uses checked addition (if either is unknown, the whole is unknown). This is the typical way of handling the`size_hint`contract.

## Design considerations: cancellation safety and`chunks_timeout`'s panic protection

`StreamExt`'s documentation annotates each method with**Cancel safety**. Taking`next`as an example:

[FACT:tokio-stream/src/stream_ext.rs:123-127]

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next`is cancellation-safe because it only borrows the stream and does not consume elements —`Next`when the future is dropped, the stream's own state is unchanged, and the next`next`will re-`poll`。

But not all combinators are cancellation-safe.`chunks_timeout`performs parameter validation at construction time:

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
> `#[track_caller]`makes the panic location point to the caller rather than the library internals,`assert!`rejecting at construction time`max_size == 0`. Why must it be checked at construction time? If`max_size == 0`，`ChunksTimeout`'s batching logic would fall into an infinite loop of "never accumulating a full batch" or produce empty batches, and such bugs are extremely difficult to locate at runtime. A construction-time panic moves the error to the earliest observable point.

`timeout`The difference between`timeout_repeating`and`timeout`is also worth noting:**returns an error after the timeout, but**；`timeout_repeating`continues polling the inner stream`Interval`instead, per

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

# Copy

## StreamMap: Dynamic stream collections and fair polling

`select!`Intuitive model`StreamMap`'s number of branches is fixed at compile time. But the number of channels a chat service needs to subscribe to, or the number of connections a crawler needs to track, are only known at runtime.`select!`is a "runtime-addable/removable`next`": it puts any number of streams into a collection, and each`(key, value)`returns`mpsc`, telling you which stream the value came from. Without it, you could only stuff all streams into a single

## channel, adding an extra layer of forwarding overhead.

`StreamMap`Data structure and memory layout`Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208]

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

Copy

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
> [Design inference and architectural trade-offs]`HashMap`Why not use`StreamMap`? Because**'s core operation is**polling all streams`Vec`, not lookup by key.`swap_remove`'s linear scan is CPU-cache-friendly, and`HashMap`is O(1). If`poll_next`were used, each`insert`would have to traverse hash buckets, with worse cache locality.`remove`and

`insert`'s O(n) scan is acceptable under the "small-scale stream collection" assumption.

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

`remove`Copy`swap_remove`uses

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

## Copy

`StreamMap`Scenario-driven Walkthrough: poll_next_entry's random start point and cursor correction`poll_next_entry`The core of**is**. It starts polling from

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

to guarantee fairness — if it always started from index 0, the first stream would starve the later ones:

**Copy** `thread_rng_n`This code has three ingenious aspects, broken down one by one:`FastRand`First, the random start point.`xorshift64+`uses a thread-local

[FACT:tokio-stream/src/stream_map.rs:765-768]

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n`algorithm:`% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792]

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**uses Lemire's multiplicative modulo instead of`swap_remove`Copy**Second,`idx`the cursor correction after`None`. When the stream at index`swap_remove`returns`idx`and is removed,**moves the last element to**. This moved element may`start`have already been polled`idx < start && start <= self.entries.len()`(if its original index was before`idx = idx.wrapping_add(1) % len`). The code uses`idx == len`to detect this case, and if so skips it (

**). If the removed element was the last one (`Poll::Pending`), the cursor wraps around to 0.**Third,`Pending`'s semantics.

`poll_next`If a full traversal finds no stream ready and the collection is non-empty, it returns`poll_next_entry`. At this point all streams' Wakers have been registered, and any becoming ready will wake it.

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

:`ready!`Copy`poll_next_entry`Note the`Pending`macro: if`poll_next`returns`Pending`。`K: Clone`, the entire`key.clone()`。

## immediately returns

`next_many`The constraint comes from the`StreamMap`here

[FACT:tokio-stream/src/stream_map.rs:581-583]

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

is the batch version of

[FACT:tokio-stream/src/stream_map.rs:573-578]

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

Copy`next_many`Its cancellation-safety guarantee is crucial:**Copy`buffer`**Why is`buffer`cancellation-safe? Because it`buffer`immediately pushes elements into the caller-provided

`poll_next_many`, rather than buffering them internally. If the future is dropped, the already-pushed elements are still in`poll_next_entry`and will not be lost. But this also means: when dropped,

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

's loop structure is more complex than`while added < limit`'s, because it must collect as many as possible within one round:`for`Copy`should_loop = true`The outer`limit`combined with the inner

[FACT:tokio-stream/src/stream_map.rs:588-591]

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint`The implementation demonstrates how to aggregate capacity hints from multiple streams:

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

Same as`merge_size_hints`the same pattern: lower bound saturating add, upper bound checked add, if either is unknown then the whole is unknown.

Below is a flowchart depicting`poll_next_entry`the decision path of:

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

# TaskTracker: Encoding all state with a single AtomicUsize

## Intuitive model

Graceful shutdown requires two things:**Notifying tasks to stop**（`CancellationToken`is responsible for), and**Waiting for tasks to actually exit**（`TaskTracker`is responsible for).`TaskTracker`is like a "task counter + shutdown switch" hybrid: as long as there are still tasks running, or`close`，`wait()`has not been called, it will not return. Without it, you could only use`JoinSet`, but`JoinSet`would accumulate the return value of each task, and a long-running service would OOM.

## Data structure and memory layout

`TaskTracker`is a`Arc`wrapper:

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

This is the most ingenious memory layout in this chapter:**A single`AtomicUsize`simultaneously encodes "whether closed" and "task count"**. The lowest bit is the closed flag, and the remaining bits are the task count (because the task count is`+2`each time, the lowest bit is always 0). This way`is_closed_and_empty`only needs one atomic load:

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
> `state == 1`means "closed bit is 1, count is 0". Why not use two atomic variables? Two variables require two loads, and cannot atomically determine "both conditions are satisfied at the same time". Single-variable encoding makes`is_closed_and_empty`a single`Acquire`load, and on the fast path of`wait`no lock is needed.

## Scenario-driven Walkthrough: the race between close and drop_task

Consider a typical scenario: the main thread calls`tracker.close()`, while the last task is exiting (`TaskTrackerToken::drop`calls`drop_task`). The two may be concurrent, and it must be guaranteed that no matter which happens first,`wait()`can be woken up.

First look at`set_closed`：

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

`fetch_or(1, AcqRel)`atomically sets the closed bit and returns the old value. If the old value is 0 (previously not closed and no tasks), it means "after closing, empty + closed is immediately satisfied", so call`notify_now`. The return value`(state & 1) == 0`indicates "this call actually changed the state".

Next look at`drop_task`：

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

`fetch_sub(2, Release)`decrements the count. If the old value is 3 (binary`11`: closed bit 1 + count 1), it means "this is the last task and it is already closed", so call`notify_now`。

Race analysis of the two paths:

- **close executes first**：`set_closed`sees the old value`2`(count 1, not closed), and does not notify. Then`drop_task`sees the old value`3`, and notifies. ✓
- **drop_task executes first**：`drop_task`sees the old value`2`(count 1, not closed), and does not notify. Then`set_closed`sees the old value`0`(count 0, not closed), and notifies. ✓
- **Concurrent**：`fetch_or`and`fetch_sub`are atomic, so no matter the interleaving order, one of them will always see the "closed + empty" combination and notify. ✓

`notify_now`There is an easily overlooked`Acquire`load in:

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

Why does`drop_task`use`Release`instead of`AcqRel`? Because`drop_task`'s`fetch_sub`only needs to "make previous writes visible to subsequent readers" (Release semantics), and does not need to "see writes from other threads before this" (Acquire semantics). But`notify_now`needs Acquire to establish happens-before: ensuring that all cleanup work done before the task exits is visible to the code after`wait()`returns. The result of this`load`is discarded purely for its memory-ordering side effect—this is a typical use of a "fence-style load" in Rust atomic operations.

## Design thinking: wait's ABA resistance and TrackedFuture's drop semantics

`wait`returns a`TaskTrackerWaitFuture`, which internally holds`Notified`：

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

Note the`inner`field: if it is already "closed and empty" at creation time, directly set it to`None`，`poll`and immediately return`Ready`. This is the fast path.

The documentation particularly emphasizes ABA resistance:

[FACT:tokio-util/src/task/task_tracker.rs:304-307]

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

This guarantee comes from the semantics of`Notify::notified()`:`Notified`the future registers itself as a "waiter" at creation time, so even if`notify_waiters`is called before it is`poll`, it will still see the notification on its first`poll`.`TaskTrackerWaitFuture::poll`'s implementation:

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

Each`poll`first checks`is_closed_and_empty()`, then`poll` `Notified`. This order guarantees that even if`Notified`is not woken for some reason, the state check can still serve as a fallback.

`TrackedFuture`'s drop semantics are the core difference between`TaskTracker`and`JoinSet`:

[FACT:tokio-util/src/task/task_tracker.rs:488-494]

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

This means: even if the future has already returned`Ready`, as long as`TrackedFuture`itself has not been dropped,`TaskTracker`still considers the task to be alive. The documentation explains why this design is important:

[FACT:tokio-util/src/task/task_tracker.rs:33-35]

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken`'s`Drop`is the trigger point for count decrement:

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

`TrackedFuture`By`pin_project!`packaging`token`and`future`together,`token`'s drop automatically triggers the count decrement.`spawn_blocking`explicitly manages the token:

[FACT:tokio-util/src/task/task_tracker.rs:452-464]

At this point, StreamExt has turned poll_next into a composable iterator, StreamMap and TaskTracker give a dynamic task set a home, and CancellationToken uses a tree to propagate cancellation signals to the entire task tree. The common point of these three layers of extensions is that they do not introduce new scheduling primitives, but instead recombine existing mechanisms such as Waker, Notify, and atomic counting into higher-level abstractions. But a key question then emerges: when these combinators, task sets, and cancellation trees run concurrently on the same scheduler, how can we ensure that a task does not starve other tasks by not yielding for a long time? The next chapter will dive into Tokio's coop cooperative budget mechanism, looking at how each task consumes budget within a scheduling cycle, actively yields when exhausted, and how budget is passed through thread-local storage, thereby solving this classic problem.
