# Capítulo 11: Ecossistema Stream e camada de ferramentas: mecanismos de extensão do tokio-stream e tokio-util

No capítulo anterior, desmontamos o mecanismo em nível de bytes do Framed: o Decoder divide BytesMut em quadros, o Sink escreve os quadros de volta, e a fronteira de abstração de I/O assíncrono torna-se clara. Mas quadros são apenas contêineres de dados; implementações reais de protocolo imediatamente encontram três problemas que nem tokio::io nem Framed resolvem: iteração assíncrona — Framed implementa Stream, mas Stream só tem poll_next, sem next().await, filter, take, merge; escrever poll_fn manualmente é verboso e propenso a erros de segurança de cancelamento; conjuntos dinâmicos de tarefas — um serviço de chat precisa se inscrever simultaneamente em N canais, que entram e saem a qualquer momento, enquanto o número de ramos do select! é fixo em tempo de compilação, incapaz de expressar conjuntos de streams que aumentam ou diminuem em tempo de execução; cancelamento estruturado — select! pode cancelar um único ramo, mas não pode propagar a parada de toda a árvore de tarefas, nem esperar que todas as tarefas realmente terminem. tokio-stream e tokio-util nasceram exatamente para essas três coisas, e seu princípio de design chave é não começar do zero: cada combinador do StreamExt é apenas um wrapper sobre poll_next, StreamMap reutiliza a semântica de registro do Waker, CancellationToken é construído diretamente sobre tokio::sync::Notify, e TaskTracker codifica todo o estado com um AtomicUsize. Entendê-los é, essencialmente, entender como fazer abstrações de custo zero sobre os mecanismos existentes de Waker e agendamento. Este capítulo progride em três camadas: iteração, coleções e cancelamento: primeiro veremos como StreamExt transforma poll_next em um iterador componível, depois como StreamMap e TaskTracker gerenciam coleções dinâmicas, e finalmente como CancellationToken usa uma árvore para propagar sinais de cancelamento por toda a árvore de tarefas.

# StreamExt: transformando poll_next em um iterador componível

## Modelo intuitivo

`Stream`é para`Future`, assim como`Iterator`é para valores:`Future`produz "um valor",`Stream`produz "uma sequência de valores". Mas`Stream`define apenas`poll_next`como primitiva, assim como`Iterator`define apenas`next`. Sem`StreamExt`, cada filtragem, mapeamento ou truncamento exigiria escrever manualmente closures de`poll_fn`e gerenciar manualmente`Pin`— isso é exatamente o ponto mais doloroso para os primeiros usuários do crate`futures`.`StreamExt`O papel de`Stream`é equipar`Iterator`com um ecossistema de combinadores como

. Sem ele, o desastre que o sistema enfrenta não é falta de funcionalidade, mas**colapso sistêmico da segurança de cancelamento**: cada`poll_fn`escrito manualmente pode, ao ser cancelado por`select!`, perder um elemento já`poll`produzido.

## Estrutura de dados e layout de memória

`StreamExt`é uma**trait de extensão**, que não armazena dados por si só:

[FACT:tokio-stream/src/stream_ext.rs:106-106]

```rust
pub trait StreamExt: Stream {
```

Todos os seus métodos retornam uma**struct concreta de combinador**, em vez de`Box<dyn Stream>`. Este é o design chave:`map`retorna`Map<Self, F>`，`filter`retorna`Filter<Self, F>`，`take`retorna`Take<Self>`. Essas structs são wrappers genéricos sem alocação no heap, e o compilador pode inline toda a cadeia em camadas de chamadas`poll_next`.

Note o blanket impl da trait:

[FACT:tokio-stream/src/stream_ext.rs:1213-1213]

```rust
impl StreamExt for St where St: Stream {}
```

Qualquer`Stream`obtém automaticamente todos os combinadores, sem necessidade de implementação manual.`?Sized`permite que`dyn Stream`também desfrute de métodos de extensão.

A declaração de módulo dos combinadores revela a superfície completa de capacidades desta trait:

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

Há uma distinção digna de nota aqui:`next`、`try_next`、`all`、`any`、`fold`、`collect`retorna**Future**（`Next`、`TryNext`、`AllFuture`...), porque consomem todo o stream em um valor; enquanto`map`、`filter`、`take`etc. retornam**Stream**, porque mantêm a forma do stream.`next`O tipo de retorno de`Next<'_, Self>`é

[FACT:tokio-stream/src/stream_ext.rs:144-149]

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin`Copiar`next`A restrição de`Pin`é intencional:`!Unpin`não obtém a posse do stream, apenas empresta, portanto não pode`Box::pin`o stream. Se o stream for`pin_mut!`, o usuário deve primeiro

[FACT:tokio-stream/src/stream_ext.rs:116-121]

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## . A documentação aponta explicitamente este trade-off:`merge`polling de

`merge`é o melhor exemplo para entender como os combinadores reutilizam o Waker. Ele intercala a produção de dois streams e**garante justiça**— se ambos os streams estiverem prontos ao mesmo tempo, alterna a produção. A documentação alerta explicitamente para não encadear chamadas`merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321]

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge`exige que ambos os streams tenham o`Item`mesmo tipo:

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

Quando o chamador`.next().await`, o fluxo de execução é o seguinte:

1. `Next::poll`chama`Merge::poll_next`。

2. `Merge`mantém internamente um sinalizador booleano de "quem foi o último". Primeiro`poll`o stream que não produziu na última vez; se`Pending`, então`poll`o outro.

3. Se ambos`Pending`，`Merge`retornam`Pending`, mas**os Wakers de ambos os streams já estão registrados**— qualquer um que fique pronto acordará a tarefa atual.

4. Se um stream retorna`Ready(None)`(fim),`Merge`registra que esse stream terminou e, a partir daí, só`poll`o outro stream, até que ele também termine.

O ponto-chave aqui é:`Merge`não tem lógica própria de gerenciamento de Waker; ele passa`cx`como está para os dois streams internos`poll_next`。**O registro do Waker é totalmente responsabilidade dos streams subjacentes**，`Merge`apenas decide "a quem perguntar primeiro desta vez". Esse é o sentido literal de "reutilizar o mecanismo de Waker subjacente".

`merge_size_hints`A função auxiliar mostra como os combinadores combinam dicas de capacidade:

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

Observe a escolha entre`saturating_add`e`checked_add`: o limite inferior usa adição saturante (prefira subestimar a estourar com panic), o limite superior usa adição verificada (se qualquer um for desconhecido, o todo é desconhecido). Essa é a forma típica de lidar com o contrato de`size_hint`.

## Reflexão de design: cancel safety e`chunks_timeout`proteção contra panic de

`StreamExt`A documentação de**Cancel safety**anota`next`em cada método. Tomando

[FACT:tokio-stream/src/stream_ext.rs:123-127]

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next`clone`Next`é cancel safe porque apenas empresta o stream, não consome elementos —`next`quando o future é dropado, o estado do próprio stream não muda, e na próxima`poll`。

irá re`chunks_timeout`Mas nem todos os combinadores são cancel safe.

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
> `#[track_caller]`〔Inferência de design e trade-offs arquiteturais〕`assert!`faz a posição do panic apontar para o chamador em vez de para dentro da biblioteca,`max_size == 0`rejeita`max_size == 0`，`ChunksTimeout`já na fase de construção. Por que é obrigatório verificar na construção? Se permitir

`timeout`a lógica de batching de`timeout_repeating`cairia em um loop infinito de "nunca acumular um lote completo" ou produziria lotes vazios, e esse tipo de bug é extremamente difícil de localizar em tempo de execução. O panic na construção antecipa o erro para o ponto observável mais cedo possível.`timeout`A diferença entre**e**；`timeout_repeating`também merece atenção:`Interval`retorna um erro após o timeout, mas

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

# , até que o stream interno produza um valor. A documentação descreve precisamente essa diferença com dois exemplos:

## clone

`select!`clone`StreamMap`StreamMap: coleção dinâmica de streams e polling justo`select!`Modelo intuitivo`next`O número de ramos de`(key, value)`é fixo em tempo de compilação. Mas o número de canais que um serviço de chat precisa assinar, ou o número de conexões que um crawler precisa rastrear, só é conhecido em tempo de execução.`mpsc`é exatamente um

## que pode ser adicionado/removido em tempo de execução: ele coloca qualquer quantidade de streams em uma coleção, e cada

`StreamMap`retorna`Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208]

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

canal, adicionando uma camada extra de overhead de encaminhamento.

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
> é extremamente simples — um`HashMap`clone`StreamMap`A documentação explica explicitamente o custo dessa escolha:**clone**〔Inferência de design e trade-offs arquiteturais〕`Vec`Por que não usar`swap_remove`? Porque a operação central de`HashMap`é`poll_next`fazer polling de todos os streams`insert`, e não busca por chave.`remove`A varredura linear de

`insert`é amigável ao cache da CPU, e

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

`remove`, cada`swap_remove`teria que percorrer os buckets de hash, com localidade de cache pior.

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

## e

`StreamMap`é aceitável sob a premissa de "coleções pequenas de streams".`poll_next_entry`A implementação de**reflete a semântica de "remover antes de inserir":**clone

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

para trocar o elemento removido com o último elemento e então fazer pop, evitando movimentação O(n):

**clone** `thread_rng_n`Walkthrough orientado a cenários: ponto de partida aleatório e correção de cursor em poll_next_entry`FastRand`O núcleo de`xorshift64+`é

[FACT:tokio-stream/src/stream_map.rs:765-768]

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n`um ponto de partida aleatório`% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792]

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**clone`swap_remove`Este trecho de código tem três sutilezas; vamos destrinchá-las uma a uma:**Primeiro, o ponto de partida aleatório.`idx`usa`None`thread-local`swap_remove`, baseado no algoritmo`idx`:**clone**usa a multiplicação e módulo de Lemire em vez de`start`clone`idx < start && start <= self.entries.len()`Segundo,`idx = idx.wrapping_add(1) % len`a correção do cursor após`idx == len`.

**Quando o stream no índice`Poll::Pending`retorna**e é removido,`Pending`move o último elemento para

`poll_next`. Esse elemento movido pode`poll_next_entry`já ter sido pollado

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

). O código usa`ready!`para detectar esse caso e, se for, pula ele (`poll_next_entry`). Se o removido for o último elemento (`Pending`), o cursor volta para 0.`poll_next`Terceiro,`Pending`。`K: Clone`a semântica de`key.clone()`。

## .

`next_many`Se percorrer um ciclo inteiro sem nenhum stream pronto, e a coleção não estiver vazia, retorna`StreamMap`. Nesse momento, os Wakers de todos os streams já estão registrados; qualquer um que fique pronto acordará.

[FACT:tokio-stream/src/stream_map.rs:581-583]

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

:

[FACT:tokio-stream/src/stream_map.rs:573-578]

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

Observe a macro`next_many`: se**retorna`buffer`**, todo o`buffer`retorna imediatamente`buffer`A restrição vem daqui

`poll_next_many`Reflexão de design: semântica em lote de next_many e cancel safety`poll_next_entry`é a versão em lote de

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

clone`while added < limit`Sua garantia de cancel safety é crucial:`for`clone`should_loop = true`Por que`limit`é cancel safe? Porque ele faz

[FACT:tokio-stream/src/stream_map.rs:588-591]

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint`A implementação de  mostra como agregar dicas de capacidade de múltiplos streams:

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

O mesmo padrão de`merge_size_hints`: saturação do limite inferior com adição, verificação do limite superior com adição, e se qualquer um for desconhecido, o todo é desconhecido.

A seguir, um fluxograma descreve o caminho de decisão de`poll_next_entry`:

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

# TaskTracker: codificando todo o estado com um único AtomicUsize

## Modelo intuitivo

O encerramento gracioso requer duas coisas:**Notificar as tarefas para pararem**（`CancellationToken`é responsável), e**Aguardar que as tarefas realmente terminem**（`TaskTracker`é responsável).`TaskTracker`é como uma fusão de "contador de tarefas + interruptor de encerramento": enquanto houver tarefas em execução, ou enquanto  não for chamado,`close`，`wait()`não retornará. Sem ele, você só poderia usar`JoinSet`, mas`JoinSet`acumularia o valor de retorno de cada tarefa, e um serviço de longa duração sofreria OOM.

## Estrutura de dados e layout de memória

`TaskTracker`é um wrapper de`Arc`:

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

Este é o layout de memória mais engenhoso deste capítulo:**Um único`AtomicUsize`codifica simultaneamente "se está encerrado" e "contagem de tarefas"**. O bit menos significativo é o flag de encerramento, e os demais bits são a contagem de tarefas (porque a contagem de tarefas é incrementada de`+2`a cada vez, o bit menos significativo é sempre 0). Assim,`is_closed_and_empty`precisa de apenas um carregamento atômico:

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
> `state == 1`significa "bit de encerramento = 1, contagem = 0". Por que não usar duas variáveis atômicas? Duas variáveis exigiriam dois carregamentos e não permitiriam determinar atomicamente "as duas condições satisfeitas ao mesmo tempo". A codificação em uma única variável faz com que`is_closed_and_empty`seja um único carregamento`Acquire`, e no caminho rápido de`wait`não requer bloqueio.

## Walkthrough orientado a cenários: a corrida entre close e drop_task

Considere um cenário típico: a thread principal chama`tracker.close()`, enquanto a última tarefa está saindo (`TaskTrackerToken::drop`chama`drop_task`). Ambos podem ser concorrentes, e é preciso garantir que, independentemente de quem vier primeiro,`wait()`possa ser despertado.

Vejamos primeiro`set_closed`：

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

`fetch_or(1, AcqRel)`define atomicamente o bit de encerramento e retorna o valor antigo. Se o valor antigo for 0 (não encerrado antes e sem tarefas), isso significa "após o encerramento, satisfaz imediatamente vazio + encerrado", então chama`notify_now`. O valor de retorno`(state & 1) == 0`indica "esta chamada realmente alterou o estado".

Vejamos agora`drop_task`：

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

`fetch_sub(2, Release)`decrementa a contagem. Se o valor antigo for 3 (binário`11`: bit de encerramento 1 + contagem 1), isso significa "esta é a última tarefa e já está encerrado", então chama`notify_now`。

Análise de corrida dos dois caminhos:

- **close executa primeiro**：`set_closed`vê o valor antigo`2`(contagem 1, não encerrado), não notifica. Em seguida,`drop_task`vê o valor antigo`3`, notifica. ✓
- **drop_task executa primeiro**：`drop_task`vê o valor antigo`2`(contagem 1, não encerrado), não notifica. Em seguida,`set_closed`vê o valor antigo`0`(contagem 0, não encerrado), notifica. ✓
- **Concorrência**：`fetch_or`e`fetch_sub`são atômicos; independentemente da ordem de intercalação, sempre haverá um que verá a combinação "encerrado + vazio" e notificará. ✓

`notify_now`Há em  um carregamento`Acquire`facilmente ignorado:

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

Por que`drop_task`usa`Release`em vez de`AcqRel`? Porque o`drop_task`de`fetch_sub`só precisa "tornar as escritas anteriores visíveis para leitores subsequentes" (semântica Release), e não precisa "ver as escritas anteriores de outras threads" (semântica Acquire). Mas`notify_now`precisa de Acquire para estabelecer happens-before: garantir que todo o trabalho de limpeza feito antes da saída da tarefa seja visível para o código após o retorno de`wait()`. O resultado deste`load`é descartado, puramente por seu efeito colateral de ordenação de memória — este é um uso típico de "carregamento estilo fence" em operações atômicas do Rust.

## Reflexão de design: a resistência a ABA de wait e a semântica de drop de TrackedFuture

`wait`retorna um`TaskTrackerWaitFuture`, que internamente mantém`Notified`：

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

Observe o campo`inner`: se no momento da criação já estiver "encerrado e vazio", define diretamente como`None`，`poll`e retorna imediatamente`Ready`. Este é o caminho rápido.

A documentação enfatiza especialmente a resistência a ABA:

[FACT:tokio-util/src/task/task_tracker.rs:304-307]

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

Esta garantia vem da semântica de`Notify::notified()`:`Notified`o future registra sua identidade de "esperante" no momento da criação; mesmo que`notify_waiters`seja chamado antes de ele ser`poll`, ele verá a notificação no primeiro`poll`.`TaskTrackerWaitFuture::poll`A implementação de

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

Copiar`poll`A cada`is_closed_and_empty()`, primeiro verifica`poll` `Notified`, depois`Notified`. Esta ordem garante que: mesmo que

`TrackedFuture`por algum motivo não seja despertado, a verificação de estado também serve como fallback.`TaskTracker`A semântica de drop de`JoinSet`é a diferença central entre

[FACT:tokio-util/src/task/task_tracker.rs:488-494]

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

:`Ready`Copiar`TrackedFuture`Isso significa: mesmo que o future já tenha retornado`TaskTracker`, desde que

[FACT:tokio-util/src/task/task_tracker.rs:33-35]

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken`considera que a tarefa ainda está ativa. A documentação explica por que este design é importante:`Drop`Copiar

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

`TrackedFuture`de`pin_project!`é o ponto de disparo do decremento da contagem:`token`Copiar`future`empacota`token`e`spawn_blocking`através de

[FACT:tokio-util/src/task/task_tracker.rs:452-464]

, e o drop de
