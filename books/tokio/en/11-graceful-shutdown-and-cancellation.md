# Chapter 11: Graceful Shutdown & Cancellation Safety: Lifecycle Management in Practice


上一章我们拆解了 Framed 的字节级机制：Decoder 把 BytesMut 切成帧，Sink 把帧写回，异步 I/O 的抽象边界由此清晰。但帧只是数据的容器，真实协议实现紧接着就会遇到三个 tokio::io 与 Framed 都不解决的问题：异步迭代——Framed 实现了 Stream，但 Stream 只有 poll_next，没有 next().await、filter、take、merge，手写 poll_fn 既啰嗦又容易在取消安全上踩坑；动态任务集合——一个聊天服务要同时订阅 N 个频道，频道随时加入退出，而 select! 的分支数量是编译期固定的，无法表达运行时增减的流集合；结构化取消——select! 能取消单个分支，但无法把整个任务树停工这件事传播下去，也无法等待所有任务真正退出。tokio-stream 与 tokio-util 正是为这三件事而生，它们的关键设计原则是不另起炉灶：StreamExt 的每个组合子都只是对 poll_next 的包装，StreamMap 复用 Waker 的注册语义，CancellationToken 直接建立在 tokio::sync::Notify 之上，TaskTracker 用一个 AtomicUsize 编码全部状态。理解它们，本质上是理解如何在既有 Waker 与调度机制上做零成本抽象。本章按迭代、集合、取消三层递进：先看 StreamExt 如何把 poll_next 变成可组合的迭代器，再看 StreamMap 与 TaskTracker 如何管理动态集合，最后看 CancellationToken 如何用一棵树把取消信号传播到整个任务树。


## Intuitive Architectural Model

`Stream` 之于 `Future`，正如 `Iterator` 之于值：`Future` 产出「一个值」，`Stream` 产出「一串值」。但 `Stream` 只定义了 `poll_next` 这一个原语，就像 `Iterator` 只定义了 `next`。若没有 `StreamExt`，每次过滤、映射、截断都要手写 `poll_fn` 闭包并手动管理 `Pin`——这正是 `futures` crate 早期用户最痛苦的地方。`StreamExt` 的角色，就是给 `Stream` 装上 `Iterator` 那样的组合子生态。

若没有它，系统面临的灾难不是功能缺失，而是**取消安全性的系统性崩塌**：每个手写的 `poll_fn` 都可能在被 `select!` 取消时丢失一个已经 `poll` 出来的元素。

## Data Structures & Memory Layout

`StreamExt` 是一个**扩展 trait**，本身不持有数据：

[FACT:tokio-stream/src/stream_ext.rs:106-106](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L106-L106)

```rust
pub trait StreamExt: Stream {
```

它的所有方法都返回一个**具体的组合子结构体**，而非 `Box<dyn Stream>`。这是关键设计：`map` 返回 `Map<Self, F>`，`filter` 返回 `Filter<Self, F>`，`take` 返回 `Take<Self>`。这些结构体都是零堆分配的泛型包装，编译器可以把整条链内联成一层层 `poll_next` 调用。

注意 trait 的 blanket impl：

[FACT:tokio-stream/src/stream_ext.rs:1213-1213](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L1213-L1213)

```rust
impl StreamExt for St where St: Stream {}
```

任何 `Stream` 自动获得全部组合子，无需手动实现。`?Sized` 允许 `dyn Stream` 也享受扩展方法。

组合子的模块声明揭示了这个 trait 的完整能力面：

[FACT:tokio-stream/src/stream_ext.rs:4-59](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L4-L59)

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

这里有一个值得注意的区分：`next`、`try_next`、`all`、`any`、`fold`、`collect` 返回的是 **Future**（`Next`、`TryNext`、`AllFuture`……），因为它们把整个流消费成一个值；而 `map`、`filter`、`take` 等返回的是 **Stream**，因为它们保持流的形态。`next` 的返回类型是 `Next<'_, Self>`，带生命周期参数，因为它只借用流：

[FACT:tokio-stream/src/stream_ext.rs:144-149](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L144-L149)

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin` 约束是刻意的：`next` 不取得流的所有权，只借用，因此无法把流 `Pin` 住。若流是 `!Unpin`，用户必须先 `Box::pin` 或 `pin_mut!`。文档明确点出了这个权衡：

[FACT:tokio-stream/src/stream_ext.rs:116-121](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L116-L121)

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## 场景驱动 Walkthrough：一次 `merge` 的轮询

`merge` 是理解组合子如何复用 Waker 的最佳样本。它把两个流交错产出，且**保证公平性**——若两个流同时就绪，交替产出。文档特意警告不要链式调用 `merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L319-L321)

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge` 的签名要求两个流的 `Item` 类型相同：

[FACT:tokio-stream/src/stream_ext.rs:398-404](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L398-L404)

```rust
fn merge(self, other: U) -> Merge
where
    U: Stream,
    Self: Sized,
{
    Merge::new(self, other)
}
```

当调用方 `.next().await` 时，执行流如下：

1. `Next::poll` 调用 `Merge::poll_next`。

2. `Merge` 内部维护一个「上次轮到谁」的布尔标志。它先 `poll` 上次未产出的那个流；若 `Pending`，再 `poll` 另一个。

3. 若两个都 `Pending`，`Merge` 返回 `Pending`，但**两个流各自的 Waker 都已注册**——任一就绪都会唤醒当前任务。

4. 若一个流返回 `Ready(None)`（结束），`Merge` 记录该流已结束，此后只 `poll` 另一个流，直到它也结束。

这里的关键是：`Merge` 没有自己的 Waker 管理逻辑，它把 `cx` 原样传给内部两个流的 `poll_next`。**Waker 的注册完全由底层流负责**，`Merge` 只是决定「这次先问谁」。这正是「复用底层 Waker 机制」的字面含义。

`merge_size_hints` 辅助函数展示了组合子如何合并容量提示：

[FACT:tokio-stream/src/stream_ext.rs:1216-1226](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L1216-L1226)

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

注意 `saturating_add` 与 `checked_add` 的选择：下界用饱和加法（宁可低估不可溢出 panic），上界用检查加法（任一未知则整体未知）。这是 `size_hint` 契约的典型处理方式。

## 设计思考：取消安全与 `chunks_timeout` 的 panic 防护

`StreamExt` 的文档对每个方法都标注了 **Cancel safety**。以 `next` 为例：

[FACT:tokio-stream/src/stream_ext.rs:123-127](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L123-L127)

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next` 之所以取消安全，是因为它只借用流、不消费元素——`Next` future 被 drop 时，流本身状态不变，下次 `next` 会重新 `poll`。

但并非所有组合子都取消安全。`chunks_timeout` 在构造时就做了参数校验：

[FACT:tokio-stream/src/stream_ext.rs:1178-1185](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L1178-L1185)

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

> **〔Design Inference & Architectural Trade-offs〕**
> `#[track_caller]` 让 panic 位置指向调用方而非库内部，`assert!` 在构造期就拒绝 `max_size == 0`。为什么必须在构造期检查？ 若允许 `max_size == 0`，`ChunksTimeout` 的批处理逻辑会陷入「永远攒不满一批」的死循环或产出空批次，而这类 bug 在运行时极难定位。构造期 panic 把错误提前到最早可观测点。

`timeout` 与 `timeout_repeating` 的差异也值得注意：`timeout` 在超时后返回一个错误，但**继续轮询内层流**；`timeout_repeating` 则按 `Interval` 持续产出超时错误，直到内层流产出值。文档用两个例子精确刻画了这个区别：

[FACT:tokio-stream/src/stream_ext.rs:985-1001](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L985-L1001)

```rust
/// Once a timeout error is received, no further events will be received
/// unless the wrapped stream yields a value (timeouts do not repeat).
```

[FACT:tokio-stream/src/stream_ext.rs:1071-1072](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_ext.rs#L1071-L1072)

```rust
/// Timeout errors will be continuously produced at the specified interval
/// until the wrapped stream yields a value.
```

---


## Intuitive Architectural Model

`select!` 的分支数在编译期固定。但聊天服务要订阅的频道数、爬虫要跟踪的连接数，都是运行时才知道的。`StreamMap` 就是「运行时可增删的 `select!`」：它把任意多个流放进一个集合，每次 `next` 返回 `(key, value)`，告诉你这个值来自哪个流。若没有它，你只能把所有流塞进一个 `mpsc` 通道，多一层转发开销。

## Data Structures & Memory Layout

`StreamMap` 的存储极其朴素——一个 `Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L204-L208)

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

文档明确说明了这个选择的代价：

[FACT:tokio-stream/src/stream_map.rs:38-44](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L38-L44)

```rust
/// `StreamMap` is backed by a `Vec`. There is no guarantee that this
/// internal implementation detail will persist in future versions, but it is
/// important to know the runtime implications. In general, `StreamMap` works
/// best with a "smallish" number of streams as all entries are scanned on
/// insert, remove, and polling. In cases where a large number of streams need
/// to be merged, it may be advisable to use tasks sending values on a shared
/// [`mpsc`] channel.
```

> **〔Design Inference & Architectural Trade-offs〕**
> 为什么不用 `HashMap`？ 因为 `StreamMap` 的核心操作是**轮询所有流**，而非按键查找。`Vec` 的线性扫描对 CPU 缓存友好，且 `swap_remove` 是 O(1)。若用 `HashMap`，每次 `poll_next` 都要遍历哈希桶，缓存局部性更差。`insert` 和 `remove` 的 O(n) 扫描在「小规模流集合」假设下可接受。

`insert` 的实现体现了「先删后插」的语义：

[FACT:tokio-stream/src/stream_map.rs:446-454](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L446-L454)

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

`remove` 用 `swap_remove` 把被删元素与末尾元素交换后弹出，避免 O(n) 搬移：

[FACT:tokio-stream/src/stream_map.rs:471-483](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L471-L483)

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

## 场景驱动 Walkthrough：poll_next_entry 的随机起点与游标修正

`StreamMap` 的核心是 `poll_next_entry`。它从**随机起点**开始轮询，以保证公平性——若总从索引 0 开始，第一个流会饿死后面的流：

[FACT:tokio-stream/src/stream_map.rs:515-550](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L515-L550)

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

这段代码有三个精妙之处，逐一拆解：

**第一，随机起点。** `thread_rng_n` 使用线程局部 `FastRand`，基于 `xorshift64+` 算法：

[FACT:tokio-stream/src/stream_map.rs:765-768](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L765-L768)

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n` 用 Lemire 的乘法取模替代 `% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L787-L792)

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**第二，`swap_remove` 后的游标修正。** 当索引 `idx` 的流返回 `None` 被移除时，`swap_remove` 会把末尾元素搬到 `idx`。这个被搬来的元素可能**已经被轮询过**（如果它的原索引在 `start` 之前）。代码用 `idx < start && start <= self.entries.len()` 检测这种情况，若是则跳过它（`idx = idx.wrapping_add(1) % len`）。若被移除的是最后一个元素（`idx == len`），游标回绕到 0。

**第三，`Poll::Pending` 的语义。** 若遍历一圈没有任何流就绪，且集合非空，返回 `Pending`。此时所有流的 Waker 都已注册，任一就绪都会唤醒。

`poll_next` 在 `poll_next_entry` 之上补上 key：

[FACT:tokio-stream/src/stream_map.rs:676-683](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L676-L683)

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

注意 `ready!` 宏：若 `poll_next_entry` 返回 `Pending`，整个 `poll_next` 立即返回 `Pending`。`K: Clone` 约束来自这里的 `key.clone()`。

## 设计思考：next_many 的批量语义与取消安全

`next_many` 是 `StreamMap` 的批量版本，一次尽可能多地收集就绪元素：

[FACT:tokio-stream/src/stream_map.rs:581-583](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L581-L583)

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

它的取消安全保证很关键：

[FACT:tokio-stream/src/stream_map.rs:573-578](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L573-L578)

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

为什么 `next_many` 取消安全？因为它把元素**立即 push 进调用方提供的 `buffer`**，而不是暂存在内部。若 future 被 drop，已 push 的元素仍在 `buffer` 里，不会丢失。但这也意味着：被 drop 时 `buffer` 可能已有部分元素——调用方需要知道这一点。

`poll_next_many` 的循环结构比 `poll_next_entry` 复杂，因为它要在一轮内尽可能多地收集：

[FACT:tokio-stream/src/stream_map.rs:597-666](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L597-L666)

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

外层 `while added < limit` 配合内层 `for` 构成「多轮扫描」：只要上一轮有流产出过值（`should_loop = true`），就再扫一轮，直到攒够 `limit` 或一轮无产出。返回值的三种情况精确对应文档：

[FACT:tokio-stream/src/stream_map.rs:588-591](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L588-L591)

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint` 的实现展示了如何聚合多个流的容量提示：

[FACT:tokio-stream/src/stream_map.rs:685-701](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-stream/src/stream_map.rs#L685-L701)

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

与 `merge_size_hints` 同样的模式：下界饱和加，上界检查加，任一未知则整体未知。

下面用一张流程图刻画 `poll_next_entry` 的决策路径：

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


## Intuitive Architectural Model

优雅关闭需要两件事：**通知任务停工**（`CancellationToken` 负责），以及**等待任务真正退出**（`TaskTracker` 负责）。`TaskTracker` 就像一个「任务计数器 + 关闭开关」的合体：只要还有任务在跑，或者还没调用 `close`，`wait()` 就不会返回。若没有它，你只能用 `JoinSet`，但 `JoinSet` 会累积每个任务的返回值，长期运行的服务会 OOM。

## Data Structures & Memory Layout

`TaskTracker` 是一个 `Arc` 包装：

[FACT:tokio-util/src/task/task_tracker.rs:158-178](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L158-L178)

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

这是本章最精妙的内存布局：**一个 `AtomicUsize` 同时编码「是否关闭」和「任务计数」**。最低位是关闭标志，其余位是任务数（因为任务计数每次 `+2`，最低位永远是 0）。这样 `is_closed_and_empty` 只需一次原子加载：

[FACT:tokio-util/src/task/task_tracker.rs:216-222](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L216-L222)

```rust
fn is_closed_and_empty(&self) -> bool {
    // If empty and closed bit set, then we are done.
    //
    // The acquire load will synchronize with the release store of any previous call to
    // `set_closed` and `drop_task`.
    self.state.load(Ordering::Acquire) == 1
}
```

> **〔Design Inference & Architectural Trade-offs〕**
> `state == 1` 意味着「关闭位为 1，计数为 0」。为什么不用两个原子变量？ 两个变量需要两次加载，且无法原子地判断「同时满足两个条件」。单变量编码让 `is_closed_and_empty` 成为一次 `Acquire` 加载，且在 `wait` 的快速路径上无需加锁。

## 场景驱动 Walkthrough：close 与 drop_task 的竞态

考虑一个典型场景：主线程调用 `tracker.close()`，同时最后一个任务正在退出（`TaskTrackerToken::drop` 调用 `drop_task`）。两者可能并发，必须保证无论谁先，`wait()` 都能被唤醒。

先看 `set_closed`：

[FACT:tokio-util/src/task/task_tracker.rs:225-249](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L225-L249)

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

`fetch_or(1, AcqRel)` 原子地设置关闭位并返回旧值。若旧值为 0（之前未关闭且无任务），说明「关闭后立即满足空+关闭」，调用 `notify_now`。返回值 `(state & 1) == 0` 表示「这次调用确实改变了状态」。

再看 `drop_task`：

[FACT:tokio-util/src/task/task_tracker.rs:264-271](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L264-L271)

```rust
fn drop_task(&self) {
    let state = self.state.fetch_sub(2, Ordering::Release);

    // If this was the last task and we are closed:
    if state == 3 {
        self.notify_now();
    }
}
```

`fetch_sub(2, Release)` 减计数。若旧值为 3（二进制 `11`：关闭位 1 + 计数 1），说明「这是最后一个任务且已关闭」，调用 `notify_now`。

两个路径的竞态分析：

- **close 先执行**：`set_closed` 看到旧值 `2`（计数 1，未关闭），不通知。随后 `drop_task` 看到旧值 `3`，通知。✓
- **drop_task 先执行**：`drop_task` 看到旧值 `2`（计数 1，未关闭），不通知。随后 `set_closed` 看到旧值 `0`（计数 0，未关闭），通知。✓
- **并发**：`fetch_or` 和 `fetch_sub` 是原子的，无论交错顺序，总有一个会看到「关闭 + 空」的组合并通知。✓

`notify_now` 里有一个容易被忽略的 `Acquire` 加载：

[FACT:tokio-util/src/task/task_tracker.rs:274-285](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L274-L285)

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

为什么 `drop_task` 用 `Release` 而非 `AcqRel`？因为 `drop_task` 的 `fetch_sub` 只需要「让之前的写对后续读者可见」（Release 语义），不需要「看到之前其他线程的写」（Acquire 语义）。但 `notify_now` 需要 Acquire 来建立 happens-before：确保任务退出前做的所有清理工作，对 `wait()` 返回后的代码可见。这个 `load` 的结果被丢弃，纯粹是为了它的内存序副作用——这是 Rust 原子操作中「fence 式加载」的典型用法。

## 设计思考：wait 的 ABA 抵抗与 TrackedFuture 的 drop 语义

`wait` 返回一个 `TaskTrackerWaitFuture`，它内部持有 `Notified`：

[FACT:tokio-util/src/task/task_tracker.rs:318-327](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L318-L327)

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

注意 `inner` 字段：若创建时已经「关闭且空」，直接设为 `None`，`poll` 时立即返回 `Ready`。这是快速路径。

文档特别强调了 ABA 抵抗：

[FACT:tokio-util/src/task/task_tracker.rs:304-307](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L304-L307)

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

这个保证来自 `Notify::notified()` 的语义：`Notified` future 在创建时就注册了「等待者」身份，即使 `notify_waiters` 在它被 `poll` 之前调用，它也会在首次 `poll` 时看到通知。`TaskTrackerWaitFuture::poll` 的实现：

[FACT:tokio-util/src/task/task_tracker.rs:697-712](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L697-L712)

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

每次 `poll` 都先检查 `is_closed_and_empty()`，再 `poll` `Notified`。这个顺序保证：即使 `Notified` 因为某种原因没被唤醒，状态检查也能兜底。

`TrackedFuture` 的 drop 语义是 `TaskTracker` 与 `JoinSet` 的核心差异：

[FACT:tokio-util/src/task/task_tracker.rs:488-494](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L488-L494)

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

这意味着：即使 future 已经返回 `Ready`，只要 `TrackedFuture` 本身还没被 drop，`TaskTracker` 就认为任务还在。文档解释了为什么这个设计重要：

[FACT:tokio-util/src/task/task_tracker.rs:33-35](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L33-L35)

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken` 的 `Drop` 是计数递减的触发点：

[FACT:tokio-util/src/task/task_tracker.rs:670-672](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L670-L672)

```rust
impl Drop for TaskTrackerToken {
    /// Dropping the token indicates to the [`TaskTracker`] that the task has exited.
    #[inline]
    fn drop(&mut self) {
        self.task_tracker.inner.drop_task();
    }
}
```

`TrackedFuture` 通过 `pin_project!` 把 `token` 与 `future` 打包，`token` 的 drop 自动触发计数递减。`spawn_blocking` 则显式管理 token：

[FACT:tokio-util/src/task/task_tracker.rs:452-464](https://github.com/tokio-rs/tokio/blob/e800714ad714f1d996ddd56265b550d879349c0a/tokio-util/src/task/task_tracker.rs#L452-L464)

至此，StreamExt 把 poll_next 变成了可组合的迭代器，StreamMap 与 TaskTracker 让动态任务集合有了归属，CancellationToken 则用一棵树把取消信号传播到整个任务树。这三层扩展的共同点是：它们没有引入新的调度原语，而是把 Waker、Notify 和原子计数这些既有机制重新组合成更高层的抽象。但一个关键问题随之浮现：当这些组合子、任务集合和取消树在同一个调度器上并发运行时，如何保证某个任务不会因为长时间不让出而饿死其他任务？下一章将深入 Tokio 的 coop 协作预算机制，看每个任务在一次调度周期内如何消耗预算、耗尽后主动让出，以及 budget 如何在线程本地存储中传递，从而解决这个经典问题。