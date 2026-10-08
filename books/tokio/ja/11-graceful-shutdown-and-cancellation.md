# 第 11 章：Stream エコシステムとツール層：tokio-stream と tokio-util の拡張メカニズム

前章では Framed のバイトレベル機構を分解した。Decoder が BytesMut をフレームに分割し、Sink がフレームを書き戻す。これにより非同期 I/O の抽象境界が明確になった。しかしフレームはデータの容器にすぎず、実際のプロトコル実装では直ちに tokio::io と Framed のどちらも解決しない3つの問題に直面する。非同期イテレーション——Framed は Stream を実装しているが、Stream には poll_next しかなく、next().await、filter、take、merge がない。手書きの poll_fn は冗長で、キャンセル安全性でつまずきやすい。動的タスク集合——チャットサービスが N 個のチャンネルを同時に購読し、チャンネルが随時参加・退出する場合、select! の分岐数はコンパイル時に固定され、実行時の増減するストリーム集合を表現できない。構造化キャンセル——select! は単一分岐をキャンセルできるが、タスクツリー全体の停止を伝播できず、すべてのタスクが実際に終了するのを待つこともできない。tokio-stream と tokio-util はまさにこの3つのために生まれ、その重要な設計原則は「別の釜戸を起こさない」ことである。StreamExt の各コンビネータは poll_next のラッパーにすぎず、StreamMap は Waker の登録セマンティクスを再利用し、CancellationToken は tokio::sync::Notify の上に直接構築され、TaskTracker は AtomicUsize 1つで全状態をエンコードする。これらを理解することは、本質的に既存の Waker とスケジューリング機構の上でゼロコスト抽象をどう作るかを理解することである。本章はイテレーション、集合、キャンセルの3層を順に進む。まず StreamExt が poll_next をどのように組み合わせ可能なイテレータに変えるかを見て、次に StreamMap と TaskTracker が動的集合をどう管理するかを見て、最後に CancellationToken が1本のツリーでキャンセル信号をタスクツリー全体にどう伝播するかを見る。

# StreamExt：poll_next を組み合わせ可能なイテレータに変える

## 直感モデル

`Stream`は`Future`にとって、`Iterator`が値にとってそうであるのと同じである：`Future`は「1つの値」を生成し、`Stream`は「一連の値」を生成する。しかし`Stream`は`poll_next`という1つのプリミティブしか定義しておらず、`Iterator`が`next`だけを定義しているのと同じである。もし`StreamExt`がなければ、フィルタ、マップ、切り詰めのたびに`poll_fn`クロージャを手書きし、`Pin`を手動管理する必要がある——これこそが`futures`crate の初期ユーザーが最も苦しんだ点である。`StreamExt`の役割は、`Stream`に`Iterator`のようなコンビネータエコシステムを装着することである。

もしそれがなければ、システムが直面する災難は機能欠如ではなく、**キャンセル安全性の体系的崩壊**である：手書きの`poll_fn`はそれぞれ、`select!`によってキャンセルされたときに、すでに`poll`された要素を失う可能性がある。

## データ構造とメモリレイアウト

`StreamExt`は**拡張 trait**であり、自身はデータを保持しない：

[FACT:tokio-stream/src/stream_ext.rs:106-106]

```rust
pub trait StreamExt: Stream {
```

そのすべてのメソッドは**具体的なコンビネータ構造体**を返し、`Box<dyn Stream>`ではない。これが重要な設計である：`map`は`Map<Self, F>`，`filter`を返し`Filter<Self, F>`，`take`は`Take<Self>`を返す。これらの構造体はすべてゼロヒープ割り当てのジェネリックラッパーであり、コンパイラはチェーン全体を何層もの`poll_next`呼び出しにインライン化できる。

trait の blanket impl に注意：

[FACT:tokio-stream/src/stream_ext.rs:1213-1213]

```rust
impl StreamExt for St where St: Stream {}
```

任意の`Stream`が自動的にすべてのコンビネータを獲得し、手動実装は不要。`?Sized`は`dyn Stream`も拡張メソッドを享受できるようにする。

コンビネータのモジュール宣言はこの trait の完全な能力面を明らかにする：

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

ここで注目すべき区別がある：`next`、`try_next`、`all`、`any`、`fold`、`collect`が返すのは**Future**（`Next`、`TryNext`、`AllFuture`……）、なぜならそれらはストリーム全体を1つの値に消費するからであり、`map`、`filter`、`take`などが返すのは**Stream**、なぜならそれらはストリームの形態を保つからである。`next`の戻り型は`Next<'_, Self>`であり、ライフタイムパラメータを持ち、ストリームを借用するだけである：

[FACT:tokio-stream/src/stream_ext.rs:144-149]

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin`制約は意図的である：`next`はストリームの所有権を取得せず、借用するだけなので、ストリームを`Pin`ことはできない。もしストリームが`!Unpin`なら、ユーザーはまず`Box::pin`または`pin_mut!`する必要がある。ドキュメントはこのトレードオフを明確に指摘している：

[FACT:tokio-stream/src/stream_ext.rs:116-121]

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## シナリオ駆動 Walkthrough：一度の`merge`のポーリング

`merge`は、コンビネータがどのように Waker を再利用するかを理解するための最良のサンプルである。これは2つのストリームを交互に産出し、かつ**公平性を保証する**——両方のストリームが同時に準備完了した場合、交互に産出する。ドキュメントは特にチェーン呼び出しを避けるよう警告している`merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321]

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge`のシグネチャは、2つのストリームの`Item`型が同じであることを要求する：

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

呼び出し側が`.next().await`を呼び出したとき、実行フローは以下の通り：

1. `Next::poll`が`Merge::poll_next`。

2. `Merge`を呼び出す。内部では「前回どちらの番だったか」を示すブールフラグを維持している。まず`poll`前回産出しなかった方のストリームをポーリングし、もし`Pending`なら、次に`poll`もう一方をポーリングする。

3. 両方とも`Pending`，`Merge`を返した場合`Pending`を返すが、**両方のストリームそれぞれの Waker はすでに登録済み**——どちらかが準備完了すれば現在のタスクが起動される。

4. 一方のストリームが`Ready(None)`（終了）を返した場合、`Merge`はそのストリームが終了したことを記録し、以降は`poll`もう一方のストリームのみをポーリングし、それも終了するまで続ける。

ここでの鍵は：`Merge`は独自の Waker 管理ロジックを持たず、`cx`をそのまま内部の2つのストリームの`poll_next`。**に渡す。Waker の登録は完全に基層のストリームが担当する**，`Merge`は単に「今回どちらに先に問い合わせるか」を決めるだけである。これこそが「基層の Waker メカニズムを再利用する」という言葉の文字通りの意味である。

`merge_size_hints`補助関数は、コンビネータがどのように容量ヒントを統合するかを示している：

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

注意`saturating_add`と`checked_add`の選択：下限には飽和加算を使用し（過小評価は許容するがオーバーフロー panic は避ける）、上限には検査付き加算を使用する（いずれかが未知なら全体が未知）。これは`size_hint`契約の典型的な処理方法である。

## 設計上の考察：キャンセル安全性と`chunks_timeout`の panic 防護

`StreamExt`のドキュメントは、各メソッドに**Cancel safety**を注記している。`next`を例にとると：

[FACT:tokio-stream/src/stream_ext.rs:123-127]

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next`がキャンセル安全である理由は、ストリームを借用するだけで要素を消費しないため——`Next`future が drop されたとき、ストリーム自体の状態は変わらず、次回の`next`は再び`poll`。

を呼び出す。しかし、すべてのコンビネータがキャンセル安全というわけではない。`chunks_timeout`は構築時にパラメータ検証を行う：

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
> `#[track_caller]`panic の位置をライブラリ内部ではなく呼び出し側に向けさせ、`assert!`構築段階で`max_size == 0`を拒否する。なぜ構築段階でチェックしなければならないのか？ もし`max_size == 0`，`ChunksTimeout`を許可すると、バッチ処理ロジックが「永遠に1バッチ分貯まらない」無限ループに陥るか、空のバッチを産出することになり、この種のバグは実行時に特定するのが極めて困難である。構築段階での panic は、エラーを最も早い観測可能な時点に前倒しする。

`timeout`と`timeout_repeating`の差異も注目に値する：`timeout`はタイムアウト後にエラーを返すが、**内側のストリームのポーリングを続ける**；`timeout_repeating`は`Interval`に従ってタイムアウトエラーを産出し続け、内側のストリームが値を産出するまで続ける。ドキュメントは2つの例でこの違いを正確に描写している：

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

# StreamMap：動的ストリーム集合と公平なポーリング

## 直感的モデル

`select!`の分岐数はコンパイル時に固定される。しかし、チャットサービスが購読するチャンネル数や、クローラーが追跡する接続数は、実行時にしかわからない。`StreamMap`は「実行時に追加・削除可能な`select!`」である：任意の数のストリームを1つの集合に入れ、毎回`next`が`(key, value)`を返し、この値がどのストリームから来たかを教えてくれる。これがなければ、すべてのストリームを1つの`mpsc`チャネルに詰め込むしかなく、余分な転送オーバーヘッドが生じる。

## データ構造とメモリレイアウト

`StreamMap`のストレージは極めて素朴である——1つの`Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208]

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

ドキュメントはこの選択の代償を明確に説明している：

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
> なぜ`HashMap`を使わないのか？ なぜなら`StreamMap`の核心的操作は**すべてのストリームをポーリングする**ことであり、キーによる検索ではないからである。`Vec`の線形スキャンは CPU キャッシュに優しく、かつ`swap_remove`は O(1) である。もし`HashMap`を使えば、毎回の`poll_next`でハッシュバケットを走査する必要があり、キャッシュ局所性が悪化する。`insert`と`remove`の O(n) スキャンは「小規模なストリーム集合」という仮定の下では許容できる。

`insert`の実装は「先に削除してから挿入」というセマンティクスを体現している：

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

`remove`は`swap_remove`を使って削除対象の要素を末尾要素と交換してからポップし、O(n) の移動を避ける：

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

## シナリオ駆動ウォークスルー：poll_next_entry のランダム開始点とカーソル修正

`StreamMap`の核心は`poll_next_entry`である。これは**ランダムな開始点**からポーリングを開始し、公平性を保証する——もし常にインデックス0から始めると、最初のストリームが後続のストリームを飢えさせる：

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

このコードには3つの巧妙な点があり、一つずつ分解する：

**第一に、ランダムな開始点。** `thread_rng_n`はスレッドローカルの`FastRand`を使用し、`xorshift64+`アルゴリズムに基づく：

[FACT:tokio-stream/src/stream_map.rs:765-768]

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n`は Lemire の乗算剰余を`% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792]

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**複製`swap_remove`第二に、**後のカーソル修正。`idx`インデックス`None`のストリームが`swap_remove`を返し`idx`が削除されると、**は末尾要素を**に移動する。この移動された要素は`start`すでにポーリング済みである可能性がある`idx < start && start <= self.entries.len()`（もしその元のインデックスが`idx = idx.wrapping_add(1) % len`より前であれば）。コードは`idx == len`でこの状況を検出し、該当する場合はスキップする（

**）。削除されたのが最後の要素である場合（`Poll::Pending`）、カーソルは0にラップアラウンドする。**第三に、`Pending`のセマンティクス。

`poll_next`一周走査してもどのストリームも準備完了でなく、かつ集合が空でない場合、`poll_next_entry`を返す。このときすべてのストリームの Waker はすでに登録済みであり、どれかが準備完了すれば起動される。

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

の上に key を補う：`ready!`複製`poll_next_entry`注意`Pending`マクロ：もし`poll_next`が`Pending`。`K: Clone`を返した場合、全体の`key.clone()`。

## が直ちに

`next_many`を返す。制約はここでの`StreamMap`に由来する

[FACT:tokio-stream/src/stream_map.rs:581-583]

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

は

[FACT:tokio-stream/src/stream_map.rs:573-578]

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

複製`next_many`そのキャンセル安全保証は極めて重要である：**複製`buffer`**なぜ`buffer`はキャンセル安全なのか？ なぜなら要素を`buffer`直ちに呼び出し側が提供する

`poll_next_many`に push し、内部に一時保存しないからである。もし future が drop されても、すでに push された要素は`poll_next_entry`に残っており、失われない。しかしこれはつまり：drop されたとき

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

のループ構造は`while added < limit`よりも複雑である。なぜなら1ラウンド内でできるだけ多く収集する必要があるから：`for`複製`should_loop = true`外側の`limit`と内側の

[FACT:tokio-stream/src/stream_map.rs:588-591]

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint`は、複数のストリームの容量ヒントをどのように集約するかを示しています：

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

と`merge_size_hints`同じパターン：下限は飽和加算、上限はチェック付き加算、いずれかが不明なら全体が不明。

次に、フローチャートで`poll_next_entry`の決定パスを描写します：

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

# TaskTracker：単一の AtomicUsize ですべての状態をエンコードする

## 直感モデル

優雅なシャットダウンには2つのことが必要です：**タスクに停止を通知すること**（`CancellationToken`が担当）、および**タスクが実際に終了するのを待つこと**（`TaskTracker`が担当）。`TaskTracker`は「タスクカウンター + シャットダウンスイッチ」の合体のようなものです：実行中のタスクがまだあるか、または`close`，`wait()`がまだ呼び出されていなければ、戻りません。これがなければ、`JoinSet`しか使えませんが、`JoinSet`は各タスクの戻り値を蓄積するため、長時間実行されるサービスでは OOM になります。

## データ構造とメモリレイアウト

`TaskTracker`は`Arc`のラッパーです：

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

これは本章で最も精妙なメモリレイアウトです：**1つの`AtomicUsize`が「閉鎖済みかどうか」と「タスク数」を同時にエンコードします**。最下位ビットは閉鎖フラグで、残りのビットはタスク数です（タスクカウントは毎回`+2`されるため、最下位ビットは常に 0 です）。これにより`is_closed_and_empty`は1回のアトミックロードだけで済みます：

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
> `state == 1`は「閉鎖ビットが 1、カウントが 0」を意味します。なぜ2つのアトミック変数を使わないのか？ 2つの変数は2回のロードが必要で、「両方の条件を同時に満たす」ことをアトミックに判定できません。単一変数エンコードにより`is_closed_and_empty`は1回の`Acquire`ロードとなり、`wait`の高速パス上でロックが不要になります。

## シナリオ駆動ウォークスルー：close と drop_task の競合

典型的なシナリオを考えます：メインスレッドが`tracker.close()`を呼び出し、同時に最後のタスクが終了中です（`TaskTrackerToken::drop`が`drop_task`を呼び出す）。両者は並行する可能性があり、どちらが先でも`wait()`が起床できることを保証する必要があります。

まず`set_closed`：

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

`fetch_or(1, AcqRel)`はアトミックに閉鎖ビットを設定し、古い値を返します。古い値が 0（以前に閉鎖されておらず、タスクもない）なら、「閉鎖後すぐに空+閉鎖を満たす」ことを意味し、`notify_now`を呼び出します。戻り値`(state & 1) == 0`は「今回の呼び出しが実際に状態を変更した」ことを示します。

次に`drop_task`：

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

`fetch_sub(2, Release)`はカウントを減算します。古い値が 3（バイナリ`11`：閉鎖ビット 1 + カウント 1）なら、「これが最後のタスクで、かつ閉鎖済み」を意味し、`notify_now`。

を呼び出します。2つのパスの競合分析：

- **close が先に実行**：`set_closed`は古い値`2`（カウント 1、未閉鎖）を見て、通知しません。その後`drop_task`は古い値`3`を見て、通知します。✓
- **drop_task が先に実行**：`drop_task`は古い値`2`（カウント 1、未閉鎖）を見て、通知しません。その後`set_closed`は古い値`0`（カウント 0、未閉鎖）を見て、通知します。✓
- **並行**：`fetch_or`と`fetch_sub`はアトミックであり、どのようなインターリーブ順序でも、必ずどちらかが「閉鎖 + 空」の組み合わせを見て通知します。✓

`notify_now`には見落とされがちな`Acquire`ロードがあります：

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

なぜ`drop_task`は`Release`ではなく`AcqRel`を使うのか？ なぜなら`drop_task`の`fetch_sub`は「以前の書き込みを後続の読者に可視化する」（Release セマンティクス）だけで十分で、「以前の他のスレッドの書き込みを見る」（Acquire セマンティクス）は不要だからです。しかし`notify_now`は happens-before を確立するために Acquire が必要です：タスク終了前に行われたすべてのクリーンアップ作業が、`wait()`の戻り値以降のコードに可視であることを保証します。この`load`の結果は破棄され、純粋にそのメモリ順序の副作用のためです——これは Rust のアトミック操作における「フェンス的ロード」の典型的な用法です。

## 設計思考：wait の ABA 耐性と TrackedFuture の drop セマンティクス

`wait`は`TaskTrackerWaitFuture`を返し、その内部は`Notified`：

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

コピー`inner`フィールドに注意：`None`，`poll`作成時にすでに「閉鎖かつ空」なら、直接`Ready`に設定し、

時に即座に

[FACT:tokio-util/src/task/task_tracker.rs:304-307]

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

ドキュメントは特に ABA 耐性を強調しています：`Notify::notified()`コピー`Notified`この保証は`notify_waiters`のセマンティクスに由来します：`poll`future は作成時に「待機者」の身分を登録し、たとえ`poll`がそれが`TaskTrackerWaitFuture::poll`される前に呼び出されても、最初の

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

の実装：`poll`コピー`is_closed_and_empty()`毎回の`poll` `Notified`はまず`Notified`をチェックし、次に

`TrackedFuture`します。この順序は保証します：たとえ`TaskTracker`が何らかの理由で起床されなくても、状態チェックがフォールバックできます。`JoinSet`の drop セマンティクスは

[FACT:tokio-util/src/task/task_tracker.rs:488-494]

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

の核心的な違いです：`Ready`コピー`TrackedFuture`これは意味します：たとえ future がすでに`TaskTracker`を返していても、

[FACT:tokio-util/src/task/task_tracker.rs:33-35]

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken`はタスクがまだ存在すると見なします。ドキュメントはこの設計がなぜ重要かを説明しています：`Drop`コピー

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

`TrackedFuture`はカウント減算のトリガーポイントです：`pin_project!`コピー`token`は`future`を通じて`token`と`spawn_blocking`をパッケージ化し、

[FACT:tokio-util/src/task/task_tracker.rs:452-464]

の drop が自動的にカウント減算をトリガーします。
