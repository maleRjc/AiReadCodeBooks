# 第 11 章：Stream 生態與工具層：tokio-stream 與 tokio-util 的擴展機制

上一章我們拆解了 Framed 的位元組級機制：Decoder 把 BytesMut 切成幀，Sink 把幀寫回，非同步 I/O 的抽象邊界由此清晰。但幀只是資料的容器，真實協定實作緊接著就會遇到三個 tokio::io 與 Framed 都不解決的問題：非同步迭代——Framed 實作了 Stream，但 Stream 只有 poll_next，沒有 next().await、filter、take、merge，手寫 poll_fn 既囉嗦又容易在取消安全上踩坑；動態任務集合——一個聊天服務要同時訂閱 N 個頻道，頻道隨時加入退出，而 select! 的分支數量是編譯期固定的，無法表達執行時增減的流集合；結構化取消——select! 能取消單個分支，但無法把整個任務樹停工這件事傳播下去，也無法等待所有任務真正退出。tokio-stream 與 tokio-util 正是為這三件事而生，它們的關鍵設計原則是不另起爐灶：StreamExt 的每個組合子都只是對 poll_next 的包裝，StreamMap 複用 Waker 的註冊語義，CancellationToken 直接建立在 tokio::sync::Notify 之上，TaskTracker 用一個 AtomicUsize 編碼全部狀態。理解它們，本質上是理解如何在既有 Waker 與排程機制上做零成本抽象。本章按迭代、集合、取消三層遞進：先看 StreamExt 如何把 poll_next 變成可組合的迭代器，再看 StreamMap 與 TaskTracker 如何管理動態集合，最後看 CancellationToken 如何用一棵樹把取消信號傳播到整個任務樹。

# StreamExt：把 poll_next 變成可組合的迭代器

## 直覺模型

`Stream`之於`Future`，正如`Iterator`之於值：`Future`產出「一個值」，`Stream`產出「一串值」。但`Stream`只定義了`poll_next`這一個原語，就像`Iterator`只定義了`next`。若沒有`StreamExt`，每次過濾、映射、截斷都要手寫`poll_fn`閉包並手動管理`Pin`——這正是`futures`crate 早期使用者最痛苦的地方。`StreamExt`的角色，就是給`Stream`裝上`Iterator`那樣的組合子生態。

若沒有它，系統面臨的災難不是功能缺失，而是**取消安全性的系統性崩塌**：每個手寫的`poll_fn`都可能在被`select!`取消時丟失一個已經`poll`出來的元素。

## 資料結構與記憶體佈局

`StreamExt`是一個**擴展 trait**，本身不持有資料：

[FACT:tokio-stream/src/stream_ext.rs:106-106]

```rust
pub trait StreamExt: Stream {
```

它的所有方法都返回一個**具體的組合子結構體**，而非`Box<dyn Stream>`。這是關鍵設計：`map`返回`Map<Self, F>`，`filter`返回`Filter<Self, F>`，`take`返回`Take<Self>`。這些結構體都是零堆分配的泛型包裝，編譯器可以把整條鏈內聯成一層層`poll_next`呼叫。

注意 trait 的 blanket impl：

[FACT:tokio-stream/src/stream_ext.rs:1213-1213]

```rust
impl StreamExt for St where St: Stream {}
```

任何`Stream`自動獲得全部組合子，無需手動實作。`?Sized`允許`dyn Stream`也享受擴展方法。

組合子的模組宣告揭示了這個 trait 的完整能力面：

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

這裡有一個值得注意的區分：`next`、`try_next`、`all`、`any`、`fold`、`collect`返回的是**Future**（`Next`、`TryNext`、`AllFuture`……），因為它們把整個流消費成一個值；而`map`、`filter`、`take`等返回的是**Stream**，因為它們保持流的形態。`next`的返回類型是`Next<'_, Self>`，帶生命週期參數，因為它只借用流：

[FACT:tokio-stream/src/stream_ext.rs:144-149]

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin`約束是刻意的：`next`不取得流的所有權，只借用，因此無法把流`Pin`住。若流是`!Unpin`，使用者必須先`Box::pin`或`pin_mut!`。文件明確點出了這個取捨：

[FACT:tokio-stream/src/stream_ext.rs:116-121]

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## 場景驅動 Walkthrough：一次`merge`的輪詢

`merge`是理解組合子如何復用 Waker 的最佳樣本。它把兩個流交錯產出，且**保證公平性**——若兩個流同時就緒，交替產出。文檔特意警告不要鏈式調用`merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321]

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge`的簽名要求兩個流的`Item`類型相同：

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

當調用方`.next().await`時，執行流如下：

1. `Next::poll`調用`Merge::poll_next`。

2. `Merge`內部維護一個「上次輪到誰」的布爾標誌。它先`poll`上次未產出的那個流；若`Pending`，再`poll`另一個。

3. 若兩個都`Pending`，`Merge`返回`Pending`，但**兩個流各自的 Waker 都已註冊**——任一就緒都會喚醒當前任務。

4. 若一個流返回`Ready(None)`（結束），`Merge`記錄該流已結束，此後只`poll`另一個流，直到它也結束。

這裡的關鍵是：`Merge`沒有自己的 Waker 管理邏輯，它把`cx`原樣傳給內部兩個流的`poll_next`。**Waker 的註冊完全由底層流負責**，`Merge`只是決定「這次先問誰」。這正是「復用底層 Waker 機制」的字面含義。

`merge_size_hints`輔助函數展示了組合子如何合併容量提示：

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

注意`saturating_add`與`checked_add`的選擇：下界用飽和加法（寧可低估不可溢出 panic），上界用檢查加法（任一未知則整體未知）。這是`size_hint`契約的典型處理方式。

## 設計思考：取消安全與`chunks_timeout`的 panic 防護

`StreamExt`的文檔對每個方法都標註了**Cancel safety**。以`next`為例：

[FACT:tokio-stream/src/stream_ext.rs:123-127]

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next`之所以取消安全，是因為它只借用流、不消費元素——`Next`future 被 drop 時，流本身狀態不變，下次`next`會重新`poll`。

但並非所有組合子都取消安全。`chunks_timeout`在構造時就做了參數校驗：

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
> `#[track_caller]`讓 panic 位置指向調用方而非庫內部，`assert!`在構造期就拒絕`max_size == 0`。為什麼必須在構造期檢查？ 若允許`max_size == 0`，`ChunksTimeout`的批處理邏輯會陷入「永遠攢不滿一批」的死循環或產出空批次，而這類 bug 在運行時極難定位。構造期 panic 把錯誤提前到最早可觀測點。

`timeout`與`timeout_repeating`的差異也值得注意：`timeout`在超時後返回一個錯誤，但**繼續輪詢內層流**；`timeout_repeating`則按`Interval`持續產出超時錯誤，直到內層流產出值。文檔用兩個例子精確刻畫了這個區別：

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

# StreamMap：動態流集合與公平輪詢

## 直覺模型

`select!`的分支數在編譯期固定。但聊天服務要訂閱的頻道數、爬蟲要跟蹤的連接數，都是運行時才知道的。`StreamMap`就是「運行時可增刪的`select!`」：它把任意多個流放進一個集合，每次`next`返回`(key, value)`，告訴你這個值來自哪個流。若沒有它，你只能把所有流塞進一個`mpsc`通道，多一層轉發開銷。

## 數據結構與內存佈局

`StreamMap`的存儲極其樸素——一個`Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208]

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

文檔明確說明了這個選擇的代價：

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
> 為什麼不用`HashMap`？ 因為`StreamMap`的核心操作是**輪詢所有流**，而非按鍵查找。`Vec`的線性掃描對 CPU 緩存友好，且`swap_remove`是 O(1)。若用`HashMap`，每次`poll_next`都要遍歷哈希桶，緩存局部性更差。`insert`和`remove`的 O(n) 掃描在「小規模流集合」假設下可接受。

`insert`的實現體現了「先刪後插」的語義：

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

`remove`用`swap_remove`把被刪元素與末尾元素交換後彈出，避免 O(n) 搬移：

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

## 場景驅動 Walkthrough：poll_next_entry 的隨機起點與游標修正

`StreamMap`的核心是`poll_next_entry`。它從**隨機起點**開始輪詢，以保證公平性——若總從索引 0 開始，第一個流會餓死後面的流：

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

這段代碼有三個精妙之處，逐一拆解：

**第一，隨機起點。** `thread_rng_n`使用線程局部`FastRand`，基於`xorshift64+`算法：

[FACT:tokio-stream/src/stream_map.rs:765-768]

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n`用 Lemire 的乘法取模替代`% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792]

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**第二，`swap_remove`後的游標修正。**當索引`idx`的流返回`None`被移除時，`swap_remove`會把末尾元素搬到`idx`。這個被搬來的元素可能**已經被輪詢過**（如果它的原索引在`start`之前）。代碼用`idx < start && start <= self.entries.len()`檢測這種情況，若是則跳過它（`idx = idx.wrapping_add(1) % len`）。若被移除的是最後一個元素（`idx == len`），游標回繞到 0。

**第三，`Poll::Pending`的語義。**若遍歷一圈沒有任何流就緒，且集合非空，返回`Pending`。此時所有流的 Waker 都已註冊，任一就緒都會喚醒。

`poll_next`在`poll_next_entry`之上補上 key：

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

注意`ready!`宏：若`poll_next_entry`返回`Pending`，整個`poll_next`立即返回`Pending`。`K: Clone`約束來自這裡的`key.clone()`。

## 設計思考：next_many 的批量語義與取消安全

`next_many`是`StreamMap`的批量版本，一次盡可能多地收集就緒元素：

[FACT:tokio-stream/src/stream_map.rs:581-583]

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

它的取消安全保證很關鍵：

[FACT:tokio-stream/src/stream_map.rs:573-578]

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

為什麼`next_many`取消安全？因為它把元素**立即 push 進調用方提供的`buffer`**，而不是暫存在內部。若 future 被 drop，已 push 的元素仍在`buffer`裡，不會丟失。但這也意味著：被 drop 時`buffer`可能已有部分元素——調用方需要知道這一點。

`poll_next_many`的循環結構比`poll_next_entry`複雜，因為它要在一輪內盡可能多地收集：

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

外層`while added < limit`配合內層`for`構成「多輪掃描」：只要上一輪有流產出過值（`should_loop = true`），就再掃一輪，直到攢夠`limit`或一輪無產出。返回值的三種情況精確對應文檔：

[FACT:tokio-stream/src/stream_map.rs:588-591]

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint`的實作展示了如何聚合多個流的容量提示：

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

與`merge_size_hints`同樣的模式：下界飽和加，上界檢查加，任一未知則整體未知。

下面用一張流程圖刻畫`poll_next_entry`的決策路徑：

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

# TaskTracker：用單個 AtomicUsize 編碼全部狀態

## 直覺模型

優雅關閉需要兩件事：**通知任務停工**（`CancellationToken`負責），以及**等待任務真正退出**（`TaskTracker`負責）。`TaskTracker`就像一個「任務計數器 + 關閉開關」的合體：只要還有任務在跑，或者還沒呼叫`close`，`wait()`就不會返回。若沒有它，你只能用`JoinSet`，但`JoinSet`會累積每個任務的返回值，長期運行的服務會 OOM。

## 資料結構與記憶體佈局

`TaskTracker`是一個`Arc`包裝：

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

這是本章最精妙的記憶體佈局：**一個`AtomicUsize`同時編碼「是否關閉」和「任務計數」**。最低位是關閉標誌，其餘位是任務數（因為任務計數每次`+2`，最低位永遠是 0）。這樣`is_closed_and_empty`只需一次原子載入：

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
> `state == 1`意味著「關閉位為 1，計數為 0」。為什麼不用兩個原子變數？ 兩個變數需要兩次載入，且無法原子地判斷「同時滿足兩個條件」。單變數編碼讓`is_closed_and_empty`成為一次`Acquire`載入，且在`wait`的快速路徑上無需加鎖。

## 場景驅動 Walkthrough：close 與 drop_task 的競態

考慮一個典型場景：主執行緒呼叫`tracker.close()`，同時最後一個任務正在退出（`TaskTrackerToken::drop`呼叫`drop_task`）。兩者可能並發，必須保證無論誰先，`wait()`都能被喚醒。

先看`set_closed`：

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

`fetch_or(1, AcqRel)`原子地設置關閉位並返回舊值。若舊值為 0（之前未關閉且無任務），說明「關閉後立即滿足空+關閉」，呼叫`notify_now`。返回值`(state & 1) == 0`表示「這次呼叫確實改變了狀態」。

再看`drop_task`：

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

`fetch_sub(2, Release)`減計數。若舊值為 3（二進位`11`：關閉位 1 + 計數 1），說明「這是最後一個任務且已關閉」，呼叫`notify_now`。

兩個路徑的競態分析：

- **close 先執行**：`set_closed`看到舊值`2`（計數 1，未關閉），不通知。隨後`drop_task`看到舊值`3`，通知。✓
- **drop_task 先執行**：`drop_task`看到舊值`2`（計數 1，未關閉），不通知。隨後`set_closed`看到舊值`0`（計數 0，未關閉），通知。✓
- **並發**：`fetch_or`和`fetch_sub`是原子的，無論交錯順序，總有一個會看到「關閉 + 空」的組合並通知。✓

`notify_now`裡有一個容易被忽略的`Acquire`載入：

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

為什麼`drop_task`用`Release`而非`AcqRel`？因為`drop_task`的`fetch_sub`只需要「讓之前的寫對後續讀者可見」（Release 語義），不需要「看到之前其他執行緒的寫」（Acquire 語義）。但`notify_now`需要 Acquire 來建立 happens-before：確保任務退出前做的所有清理工作，對`wait()`返回後的程式碼可見。這個`load`的結果被丟棄，純粹是為了它的記憶體序副作用——這是 Rust 原子操作中「fence 式載入」的典型用法。

## 設計思考：wait 的 ABA 抵抗與 TrackedFuture 的 drop 語義

`wait`返回一個`TaskTrackerWaitFuture`，它內部持有`Notified`：

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

注意`inner`欄位：若建立時已經「關閉且空」，直接設為`None`，`poll`時立即返回`Ready`。這是快速路徑。

文件特別強調了 ABA 抵抗：

[FACT:tokio-util/src/task/task_tracker.rs:304-307]

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

這個保證來自`Notify::notified()`的語義：`Notified`future 在建立時就註冊了「等待者」身份，即使`notify_waiters`在它被`poll`之前呼叫，它也會在首次`poll`時看到通知。`TaskTrackerWaitFuture::poll`的實作：

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

每次`poll`都先檢查`is_closed_and_empty()`，再`poll` `Notified`。這個順序保證：即使`Notified`因為某種原因沒被喚醒，狀態檢查也能兜底。

`TrackedFuture`的 drop 語義是`TaskTracker`與`JoinSet`的核心差異：

[FACT:tokio-util/src/task/task_tracker.rs:488-494]

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

這意味著：即使 future 已經返回`Ready`，只要`TrackedFuture`本身還沒被 drop，`TaskTracker`就認為任務還在。文件解釋了為什麼這個設計重要：

[FACT:tokio-util/src/task/task_tracker.rs:33-35]

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken`的`Drop`是計數遞減的觸發點：

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

`TrackedFuture`透過`pin_project!`把`token`與`future`打包，`token`的 drop 自動觸發計數遞減。`spawn_blocking`則顯式管理 token：

[FACT:tokio-util/src/task/task_tracker.rs:452-464]

至此，StreamExt 把 poll_next 變成了可組合的迭代器，StreamMap 與 TaskTracker 讓動態任務集合有了歸屬，CancellationToken 則用一棵樹把取消信號傳播到整個任務樹。這三層擴展的共同點是：它們沒有引入新的調度原語，而是把 Waker、Notify 和原子計數這些既有機制重新組合成更高層的抽象。但一個關鍵問題隨之浮現：當這些組合子、任務集合和取消樹在同一個調度器上並發運行時，如何保證某個任務不會因為長時間不讓出而餓死其他任務？下一章將深入 Tokio 的 coop 協作預算機制，看每個任務在一次調度週期內如何消耗預算、耗盡後主動讓出，以及 budget 如何線上程本地存儲中傳遞，從而解決這個經典問題。
