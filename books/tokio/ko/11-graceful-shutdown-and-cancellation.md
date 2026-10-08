# 제 11 장: Stream 생태계와 도구 계층: tokio-stream과 tokio-util의 확장 메커니즘

지난 장에서 우리는 Framed의 바이트 수준 메커니즘을 분해했다. Decoder는 BytesMut을 프레임으로 자르고, Sink는 프레임을 다시 쓴다. 이로써 비동기 I/O의 추상화 경계가 명확해진다. 하지만 프레임은 데이터의 컨테이너일 뿐이며, 실제 프로토콜 구현은 곧바로 tokio::io와 Framed 모두 해결하지 못하는 세 가지 문제에 직면한다. 비동기 반복 — Framed는 Stream을 구현하지만 Stream에는 poll_next만 있고 next().await, filter, take, merge가 없으며, poll_fn을 직접 작성하는 것은 장황할 뿐만 아니라 취소 안전성에서 함정을 밟기 쉽다. 동적 작업 집합 — 채팅 서비스가 N개의 채널을 동시에 구독해야 하고 채널이 수시로 가입하고 탈퇴하는데, select!의 분기 수는 컴파일 시점에 고정되어 런타임에 증감하는 스트림 집합을 표현할 수 없다. 구조적 취소 — select!는 단일 분기를 취소할 수 있지만, 전체 작업 트리를 중단시키는 일을 전파할 수도 없고 모든 작업이 실제로 종료되기를 기다릴 수도 없다. tokio-stream과 tokio-util은 바로 이 세 가지를 위해 태어났으며, 이들의 핵심 설계 원칙은 별도의 체계를 새로 만들지 않는 것이다. StreamExt의 각 조합자는 단지 poll_next에 대한 래퍼일 뿐이고, StreamMap은 Waker의 등록 의미론을 재사용하며, CancellationToken은 tokio::sync::Notify 위에 직접 세워지고, TaskTracker는 AtomicUsize 하나로 모든 상태를 인코딩한다. 이들을 이해하는 것은 본질적으로 기존 Waker와 스케줄링 메커니즘 위에서 제로 비용 추상화를 하는 방법을 이해하는 것이다. 이 장은 반복, 집합, 취소의 세 계층으로 점진한다. 먼저 StreamExt가 어떻게 poll_next를 조합 가능한 반복자로 바꾸는지 보고, 다음으로 StreamMap과 TaskTracker가 어떻게 동적 집합을 관리하는지 보며, 마지막으로 CancellationToken이 어떻게 트리 하나로 취소 신호를 전체 작업 트리에 전파하는지 본다.

# StreamExt: poll_next를 조합 가능한 반복자로 바꾸기

## 직관적 모델

`Stream`은(는)`Future`에 대해, 마치`Iterator`이(가) 값에 대해 그러하듯:`Future`은 '하나의 값'을 산출하고,`Stream`은 '일련의 값'을 산출한다. 하지만`Stream`은`poll_next`이 하나의 원시 연산만 정의하며, 마치`Iterator`이`next`만 정의한 것과 같다. 만약`StreamExt`이 없다면, 매번 필터링, 매핑, 절단마다`poll_fn`클로저를 직접 작성하고`Pin`을 수동으로 관리해야 한다 — 이것이 바로`futures`crate 초기 사용자들이 가장 고통스러워했던 부분이다.`StreamExt`의 역할은 바로`Stream`에`Iterator`과 같은 조합자 생태계를 붙여주는 것이다.

만약 그것이 없다면, 시스템이 직면하는 재앙은 기능 부재가 아니라**취소 안전성의 체계적 붕괴**이다: 손으로 작성한 모든`poll_fn`은`select!`에 의해 취소될 때 이미`poll`되어 나온 요소 하나를 잃을 수 있다.

## 데이터 구조와 메모리 레이아웃

`StreamExt`은**확장 trait**이며, 자체적으로 데이터를 보유하지 않는다:

[FACT:tokio-stream/src/stream_ext.rs:106-106]

```rust
pub trait StreamExt: Stream {
```

그 모든 메서드는**구체적인 조합자 구조체**를 반환하며,`Box<dyn Stream>`이 아니다. 이것이 핵심 설계다:`map`은`Map<Self, F>`，`filter`을 반환하고`Filter<Self, F>`，`take`은`Take<Self>`을 반환한다. 이 구조체들은 모두 힙 할당이 없는 제네릭 래퍼이며, 컴파일러는 전체 체인을 여러 겹의`poll_next`호출로 인라인할 수 있다.

trait의 blanket impl에 주목하라:

[FACT:tokio-stream/src/stream_ext.rs:1213-1213]

```rust
impl StreamExt for St where St: Stream {}
```

어떤`Stream`이든 자동으로 모든 조합자를 얻으며, 수동 구현이 필요 없다.`?Sized`은`dyn Stream`도 확장 메서드를 누릴 수 있게 해준다.

조합자의 모듈 선언은 이 trait의 완전한 능력 면모를 드러낸다:

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

여기서 주목할 만한 구분이 있다:`next`、`try_next`、`all`、`any`、`fold`、`collect`이 반환하는 것은**Future**（`Next`、`TryNext`、`AllFuture`이다...), 왜냐하면 그것들은 전체 스트림을 하나의 값으로 소비하기 때문이다. 반면`map`、`filter`、`take`등이 반환하는 것은**Stream**이며, 왜냐하면 그것들은 스트림의 형태를 유지하기 때문이다.`next`의 반환 타입은`Next<'_, Self>`이며, 라이프타임 매개변수를 가지는데, 이는 스트림을 빌리기만 하기 때문이다:

[FACT:tokio-stream/src/stream_ext.rs:144-149]

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin`제약은 의도적이다:`next`은 스트림의 소유권을 얻지 않고 빌리기만 하므로, 스트림을`Pin`할 수 없다. 만약 스트림이`!Unpin`이라면, 사용자는 먼저`Box::pin`하거나`pin_mut!`해야 한다. 문서는 이 트레이드오프를 명확히 지적한다:

[FACT:tokio-stream/src/stream_ext.rs:116-121]

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## 시나리오 기반 Walkthrough: 한 번의`merge`의 폴링

`merge`은 조합자가 Waker를 어떻게 재사용하는지 이해하는 최고의 샘플이다. 두 스트림을 교차로 산출하며,**공정성을 보장한다**—두 스트림이 동시에 준비되면 교대로 산출한다. 문서는 특별히 체인 호출을 하지 말라고 경고한다`merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321]

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge`의 시그니처는 두 스트림의`Item`타입이 같을 것을 요구한다:

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

호출자가`.next().await`할 때, 실행 흐름은 다음과 같다:

1. `Next::poll`이`Merge::poll_next`。

2. `Merge`을 호출하면 내부적으로 "지난번에 누구 차례였는지"를 나타내는 불리언 플래그를 유지한다. 먼저`poll`지난번에 산출하지 않은 스트림을 폴링하고, 만약`Pending`이면 다시`poll`다른 쪽을 폴링한다.

3. 만약 둘 다`Pending`，`Merge`을 반환하면`Pending`을 반환하지만,**두 스트림 각각의 Waker가 이미 등록되어 있으므로**—어느 하나라도 준비되면 현재 태스크를 깨운다.

4. 만약 한 스트림이`Ready(None)`(종료)를 반환하면,`Merge`은 해당 스트림이 종료되었음을 기록하고, 이후에는`poll`다른 스트림만 폴링하며, 그것도 종료될 때까지 계속한다.

여기서 핵심은:`Merge`은 자체적인 Waker 관리 로직이 없으며,`cx`을 그대로 내부 두 스트림의`poll_next`。**에 전달한다. Waker 등록은 전적으로 하위 스트림이 담당하며,**，`Merge`은 단지 "이번에 누구를 먼저 물어볼지"만 결정한다. 이것이 바로 "하위 Waker 메커니즘을 재사용한다"는 말의 문자 그대로의 의미다.

`merge_size_hints`보조 함수는 조합자가 용량 힌트를 어떻게 병합하는지 보여준다:

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

주목할 점은`saturating_add`과`checked_add`의 선택이다: 하한은 포화 덧셈(과소평가할지언정 오버플로 panic은 피함)을 사용하고, 상한은 검사 덧셈(어느 하나라도 미지면 전체가 미지)을 사용한다. 이것이`size_hint`계약의 전형적인 처리 방식이다.

## 설계 고찰: 취소 안전성과`chunks_timeout`의 panic 방어

`StreamExt`의 문서는 모든 메서드에**Cancel safety**을 표기한다. 예를 들어`next`의 경우:

[FACT:tokio-stream/src/stream_ext.rs:123-127]

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next`이 취소 안전한 이유는 스트림을 빌리기만 하고 요소를 소비하지 않기 때문이다—`Next`future가 drop될 때 스트림 자체의 상태는 변하지 않고, 다음번`next`은 다시`poll`。

을 호출한다. 하지만 모든 조합자가 취소 안전한 것은 아니다.`chunks_timeout`은 생성 시점에 매개변수 검증을 수행한다:

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
> `#[track_caller]`은 panic 위치가 라이브러리 내부가 아닌 호출자를 가리키게 하며,`assert!`은 생성 시점에`max_size == 0`을 거부한다. 왜 반드시 생성 시점에 검사해야 하는가? 만약`max_size == 0`，`ChunksTimeout`을 허용하면 배치 처리 로직이 "영원히 한 배치를 채우지 못하는" 무한 루프에 빠지거나 빈 배치를 산출하게 되는데, 이런 종류의 버그는 런타임에 극히 찾아내기 어렵다. 생성 시점 panic은 오류를 가장 이른 관측 가능 지점으로 앞당긴다.

`timeout`과`timeout_repeating`의 차이도 주목할 만하다:`timeout`은 타임아웃 후 오류를 반환하지만,**내부 스트림을 계속 폴링한다.**；`timeout_repeating`은`Interval`에 따라 내부 스트림이 값을 산출할 때까지 계속 타임아웃 오류를 산출한다. 문서는 두 가지 예제로 이 차이를 정확히 묘사한다:

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

# StreamMap: 동적 스트림 집합과 공정 폴링

## 직관적 모델

`select!`의 분기 수는 컴파일 시점에 고정된다. 하지만 채팅 서비스가 구독할 채널 수, 크롤러가 추적할 연결 수는 모두 런타임에야 알 수 있다.`StreamMap`은 "런타임에 추가/삭제 가능한`select!`"이다: 임의 개수의 스트림을 하나의 집합에 넣고, 매번`next`이`(key, value)`을 반환하여 이 값이 어느 스트림에서 왔는지 알려준다. 이것이 없다면 모든 스트림을 하나의`mpsc`채널에 밀어넣어야 하며, 한 계층의 전달 오버헤드가 추가된다.

## 데이터 구조와 메모리 레이아웃

`StreamMap`의 저장 방식은 극히 소박하다—하나의`Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208]

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

문서는 이 선택의 대가를 명확히 설명한다:

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
> 왜`HashMap`을 사용하지 않는가? 왜냐하면`StreamMap`의 핵심 연산은**모든 스트림 폴링**이지, 키로 조회하는 것이 아니기 때문이다.`Vec`의 선형 스캔은 CPU 캐시에 친화적이며,`swap_remove`은 O(1)이다. 만약`HashMap`을 사용하면 매번`poll_next`마다 해시 버킷을 순회해야 하므로 캐시 지역성이 더 나빠진다.`insert`과`remove`의 O(n) 스캔은 "소규모 스트림 집합" 가정 하에 수용 가능하다.

`insert`의 구현은 "먼저 삭제 후 삽입" 의미론을 구현한다:

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

`remove`은`swap_remove`을 사용하여 삭제된 요소를 마지막 요소와 교환한 후 pop하여 O(n) 이동을 피한다:

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

## 시나리오 기반 Walkthrough: poll_next_entry의 무작위 시작점과 커서 보정

`StreamMap`의 핵심은`poll_next_entry`이다. 이것은**무작위 시작점**부터 폴링을 시작하여 공정성을 보장한다—만약 항상 인덱스 0부터 시작하면 첫 번째 스트림이 뒤의 스트림들을 기아 상태로 만들 것이다:

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

이 코드에는 세 가지 정묘한 점이 있으며, 하나씩 분석한다:

**첫째, 무작위 시작점.** `thread_rng_n`은 스레드 로컬`FastRand`을 사용하며,`xorshift64+`알고리즘에 기반한다:

[FACT:tokio-stream/src/stream_map.rs:765-768]

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n`은 Lemire의 곱셈 모듈로를 사용하여`% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792]

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**복사`swap_remove`둘째,**이후의 커서 보정.`idx`인덱스`None`의 스트림이`swap_remove`을 반환하여 제거될 때,`idx`은 마지막 요소를**로 옮긴다. 이 옮겨진 요소는**이미 폴링되었을 수 있다`start`(만약 원래 인덱스가`idx < start && start <= self.entries.len()`이전이라면). 코드는`idx = idx.wrapping_add(1) % len`으로 이 상황을 감지하고, 그렇다면 건너뛴다(`idx == len`). 만약 제거된 것이 마지막 요소라면(

**), 커서는 0으로 되돌아간다.`Poll::Pending`셋째,**의 의미론.`Pending`만약 한 바퀴 순회했는데 어떤 스트림도 준비되지 않았고 집합이 비어 있지 않다면,

`poll_next`을 반환한다. 이때 모든 스트림의 Waker가 이미 등록되어 있으므로, 어느 하나라도 준비되면 깨운다.`poll_next_entry`은

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

복사`ready!`주목할 점은`poll_next_entry`매크로이다: 만약`Pending`이`poll_next`을 반환하면, 전체`Pending`。`K: Clone`이 즉시`key.clone()`。

## 을 반환한다. 제약은 여기의

`next_many`에서 온다. 설계 고찰: next_many의 배치 의미론과 취소 안전성`StreamMap`은

[FACT:tokio-stream/src/stream_map.rs:581-583]

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

복사

[FACT:tokio-stream/src/stream_map.rs:573-578]

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

복사`next_many`왜**이 취소 안전한가? 왜냐하면 요소를`buffer`**즉시 호출자가 제공한`buffer`에 push하고, 내부에 임시 저장하지 않기 때문이다. 만약 future가 drop되면 이미 push된 요소는 여전히`buffer`안에 있어 손실되지 않는다. 하지만 이것은 또한 의미한다: drop될 때

`poll_next_many`에 이미 일부 요소가 있을 수 있다—호출자는 이 점을 알아야 한다.`poll_next_entry`의 루프 구조는

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

복사`while added < limit`외부`for`과 내부`should_loop = true`가 "다중 라운드 스캔"을 구성한다: 이전 라운드에 스트림이 값을 산출했으면(`limit`), 한 라운드 더 스캔하여

[FACT:tokio-stream/src/stream_map.rs:588-591]

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint`여러 스트림의 용량 힌트를 집계하는 방법을 보여줍니다:

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

와`merge_size_hints`동일한 패턴: 하한은 포화 덧셈, 상한은 검사 덧셈, 어느 하나라도 미지면 전체가 미지.

아래는 흐름도로`poll_next_entry`의 의사결정 경로를 묘사합니다:

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

# TaskTracker: 단일 AtomicUsize로 모든 상태를 인코딩

## 직관적 모델

우아한 종료에는 두 가지가 필요합니다:**작업에 중지를 통지**（`CancellationToken`가 담당), 그리고**작업이 실제로 종료될 때까지 대기**（`TaskTracker`가 담당).`TaskTracker`은 「작업 카운터 + 종료 스위치」의 결합체와 같습니다: 아직 실행 중인 작업이 있거나`close`，`wait()`이 호출되지 않았다면 반환하지 않습니다. 이것이 없다면`JoinSet`만 사용할 수 있지만,`JoinSet`은 각 작업의 반환값을 누적하므로 장기 실행 서비스는 OOM이 발생합니다.

## 데이터 구조와 메모리 레이아웃

`TaskTracker`은`Arc`래퍼입니다:

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

이것은 이 장에서 가장 정교한 메모리 레이아웃입니다:**하나의`AtomicUsize`이 「종료 여부」와 「작업 수」를 동시에 인코딩합니다**. 최하위 비트는 종료 플래그, 나머지 비트는 작업 수입니다 (작업 카운트가 매번`+2`이므로 최하위 비트는 항상 0). 이렇게 하면`is_closed_and_empty`은 단 한 번의 원자적 로드만 필요합니다:

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
> `state == 1`은 「종료 비트가 1, 카운트가 0」을 의미합니다. 왜 두 개의 원자적 변수를 사용하지 않을까요? 두 변수는 두 번의 로드가 필요하고 「두 조건을 동시에 만족」하는 것을 원자적으로 판단할 수 없습니다. 단일 변수 인코딩은`is_closed_and_empty`을 한 번의`Acquire`로드로 만들고,`wait`의 빠른 경로에서 잠금이 필요 없게 합니다.

## 시나리오 기반 Walkthrough: close와 drop_task의 경쟁

전형적인 시나리오를 고려해봅시다: 메인 스레드가`tracker.close()`을 호출하고, 동시에 마지막 작업이 종료 중입니다 (`TaskTrackerToken::drop`이`drop_task`을 호출). 둘은 동시에 실행될 수 있으며, 누가 먼저든`wait()`이 깨어날 수 있음을 보장해야 합니다.

먼저`set_closed`：

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

`fetch_or(1, AcqRel)`은 원자적으로 종료 비트를 설정하고 이전 값을 반환합니다. 이전 값이 0이면 (이전에 종료되지 않았고 작업도 없음), 「종료 후 즉시 빈 상태 + 종료 충족」을 의미하므로`notify_now`을 호출합니다. 반환값`(state & 1) == 0`은 「이번 호출이 실제로 상태를 변경했음」을 나타냅니다.

다음으로`drop_task`：

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

`fetch_sub(2, Release)`은 카운트를 감소시킵니다. 이전 값이 3이면 (이진수`11`: 종료 비트 1 + 카운트 1), 「이것이 마지막 작업이고 이미 종료됨」을 의미하므로`notify_now`。

을 호출합니다. 두 경로의 경쟁 분석:

- **close가 먼저 실행**：`set_closed`은 이전 값`2`(카운트 1, 미종료)을 보고 통지하지 않습니다. 이후`drop_task`이 이전 값`3`을 보고 통지합니다. ✓
- **drop_task가 먼저 실행**：`drop_task`은 이전 값`2`(카운트 1, 미종료)을 보고 통지하지 않습니다. 이후`set_closed`이 이전 값`0`(카운트 0, 미종료)을 보고 통지합니다. ✓
- **동시 실행**：`fetch_or`과`fetch_sub`은 원자적이므로, 어떤 교차 순서든 항상 하나는 「종료 + 빈 상태」 조합을 보고 통지합니다. ✓

`notify_now`에는 간과하기 쉬운`Acquire`로드가 있습니다:

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

왜`drop_task`이`Release`대신`AcqRel`을 사용할까요?`drop_task`의`fetch_sub`은 「이전 쓰기를 후속 읽기자에게 가시화」만 필요하고 (Release 의미론), 「이전 다른 스레드의 쓰기를 보는 것」 (Acquire 의미론)은 필요하지 않기 때문입니다. 하지만`notify_now`은 happens-before를 확립하기 위해 Acquire가 필요합니다: 작업 종료 전에 수행한 모든 정리 작업이`wait()`반환 후의 코드에 가시적임을 보장합니다. 이`load`의 결과는 버려지며, 순전히 메모리 순서 부작용을 위한 것입니다 — 이것은 Rust 원자적 연산에서 「fence식 로드」의 전형적인 사용법입니다.

## 설계 사고: wait의 ABA 저항과 TrackedFuture의 drop 의미론

`wait`은`TaskTrackerWaitFuture`을 반환하며, 내부적으로`Notified`：

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

복사`inner`필드에 주목하세요: 생성 시 이미 「종료되고 빈 상태」라면,`None`，`poll`으로 직접 설정하여`Ready`시 즉시 반환합니다. 이것이 빠른 경로입니다.

문서는 ABA 저항을 특별히 강조합니다:

[FACT:tokio-util/src/task/task_tracker.rs:304-307]

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

이 보장은`Notify::notified()`의 의미론에서 비롯됩니다:`Notified`future는 생성 시 「대기자」 신원을 등록하므로,`notify_waiters`이 그것이`poll`되기 전에 호출되더라도, 첫`poll`시 통지를 보게 됩니다.`TaskTrackerWaitFuture::poll`의 구현:

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

매번`poll`은 먼저`is_closed_and_empty()`을 확인하고, 그 다음`poll` `Notified`을 합니다. 이 순서는 다음을 보장합니다:`Notified`이 어떤 이유로 깨어나지 않더라도 상태 확인이 안전망 역할을 합니다.

`TrackedFuture`의 drop 의미론은`TaskTracker`과`JoinSet`의 핵심 차이입니다:

[FACT:tokio-util/src/task/task_tracker.rs:488-494]

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

이것은 다음을 의미합니다: future가 이미`Ready`을 반환했더라도,`TrackedFuture`자체가 아직 drop되지 않았다면,`TaskTracker`은 작업이 아직 있다고 간주합니다. 문서는 이 설계가 왜 중요한지 설명합니다:

[FACT:tokio-util/src/task/task_tracker.rs:33-35]

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken`의`Drop`은 카운트 감소의 트리거 지점입니다:

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

`TrackedFuture`은`pin_project!`을 통해`token`과`future`을 함께 묶어,`token`의 drop이 자동으로 카운트 감소를 트리거합니다.`spawn_blocking`은 token을 명시적으로 관리합니다:

[FACT:tokio-util/src/task/task_tracker.rs:452-464]

여기까지 StreamExt는 poll_next를 조합 가능한 반복자로 만들었고, StreamMap과 TaskTracker는 동적 작업 집합에 귀속을 부여했으며, CancellationToken은 트리로 취소 신호를 전체 작업 트리에 전파합니다. 이 세 계층 확장의 공통점은: 새로운 스케줄링 원시 요소를 도입하지 않고, Waker, Notify, 원자적 카운트 같은 기존 메커니즘을 재조합하여 더 높은 수준의 추상화를 만든 것입니다. 하지만 핵심 질문이 떠오릅니다: 이 조합자, 작업 집합, 취소 트리가 같은 스케줄러에서 동시에 실행될 때, 어떤 작업이 오랫동안 양보하지 않아 다른 작업을 기아 상태로 만들지 않도록 어떻게 보장할까요? 다음 장에서는 Tokio의 coop 협력 예산 메커니즘을 깊이 파고들어, 각 작업이 한 스케줄링 주기에서 예산을 어떻게 소비하고, 소진 후 능동적으로 양보하며, budget이 스레드 로컬 저장소에서 어떻게 전달되는지 살펴봄으로써 이 고전적 문제를 해결합니다.
