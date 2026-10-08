# Chapitre 11 : Écosystème Stream et couche d'outils : mécanismes d'extension de tokio-stream et tokio-util

Dans le chapitre précédent, nous avons décomposé le mécanisme au niveau des octets de Framed : le Decoder découpe BytesMut en trames, le Sink réécrit les trames, et la frontière d'abstraction de l'I/O asynchrone devient ainsi claire. Mais une trame n'est qu'un conteneur de données ; une implémentation réelle de protocole rencontre immédiatement trois problèmes que ni tokio::io ni Framed ne résolvent : l'itération asynchrone — Framed implémente Stream, mais Stream n'a que poll_next, sans next().await, filter, take, merge ; écrire poll_fn à la main est à la fois verbeux et propice aux pièges de la sécurité d'annulation ; l'ensemble de tâches dynamique — un service de chat doit s'abonner simultanément à N canaux, les canaux rejoignant et quittant à tout moment, alors que le nombre de branches de select! est fixé à la compilation et ne peut exprimer un ensemble de flux variant à l'exécution ; l'annulation structurée — select! peut annuler une branche unique, mais ne peut propager l'arrêt de tout l'arbre de tâches, ni attendre que toutes les tâches aient réellement terminé. tokio-stream et tokio-util sont nés précisément pour ces trois choses, et leur principe de conception clé est de ne pas repartir de zéro : chaque combinateur de StreamExt n'est qu'un emballage autour de poll_next, StreamMap réutilise la sémantique d'enregistrement de Waker, CancellationToken se construit directement sur tokio::sync::Notify, et TaskTracker encode tout son état dans un AtomicUsize. Les comprendre, c'est essentiellement comprendre comment réaliser une abstraction à coût nul sur les mécanismes existants de Waker et d'ordonnancement. Ce chapitre progresse en trois couches : itération, collections, annulation : d'abord comment StreamExt transforme poll_next en itérateur composable, puis comment StreamMap et TaskTracker gèrent les collections dynamiques, et enfin comment CancellationToken propage le signal d'annulation à tout l'arbre de tâches au moyen d'un arbre.

# StreamExt : transformer poll_next en itérateur composable

## Modèle intuitif

`Stream`est à`Future`, ce que`Iterator`est à une valeur :`Future`produit « une valeur »,`Stream`produit « une suite de valeurs ». Mais`Stream`ne définit que`poll_next`comme unique primitive, tout comme`Iterator`ne définit que`next`. Sans`StreamExt`, chaque filtrage, mapping, troncature nécessiterait d'écrire à la main une closure`poll_fn`et de gérer manuellement`Pin`— c'est précisément là que les premiers utilisateurs du crate`futures`souffraient le plus.`StreamExt`Le rôle de`Stream`est de doter`Iterator`d'un écosystème de combinateurs comme

. Sans lui, la catastrophe à laquelle le système ferait face ne serait pas un manque de fonctionnalités, mais**un effondrement systémique de la sécurité d'annulation**: chaque`poll_fn`écrit à la main pourrait, lors d'une annulation par`select!`, perdre un élément déjà`poll`.

## Structures de données et disposition mémoire

`StreamExt`est un**trait d'extension**, qui ne détient lui-même aucune donnée :

[FACT:tokio-stream/src/stream_ext.rs:106-106]

```rust
pub trait StreamExt: Stream {
```

Toutes ses méthodes renvoient une**structure de combinateur concrète**, et non`Box<dyn Stream>`. C'est une conception clé :`map`renvoie`Map<Self, F>`，`filter`renvoie`Filter<Self, F>`，`take`renvoie`Take<Self>`. Ces structures sont toutes des emballages génériques sans allocation sur le tas, et le compilateur peut inliner toute la chaîne en une succession d'appels`poll_next`.

Noter le blanket impl du trait :

[FACT:tokio-stream/src/stream_ext.rs:1213-1213]

```rust
impl StreamExt for St where St: Stream {}
```

Tout`Stream`obtient automatiquement tous les combinateurs, sans implémentation manuelle.`?Sized`permet à`dyn Stream`de bénéficier aussi des méthodes d'extension.

La déclaration des modules de combinateurs révèle la surface de capacité complète de ce trait :

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

Il y a ici une distinction à noter :`next`、`try_next`、`all`、`any`、`fold`、`collect`renvoie**Future**（`Next`、`TryNext`、`AllFuture`…), car ils consomment tout le flux en une valeur ; tandis que`map`、`filter`、`take`etc. renvoient**Stream**, car ils préservent la forme du flux.`next`Le type de retour de`Next<'_, Self>`est

[FACT:tokio-stream/src/stream_ext.rs:144-149]

```rust
fn next(&mut self) -> Next
where
    Self: Unpin,
{
    Next::new(self)
}
```

`Self: Unpin`Copier`next`La contrainte`Pin`est délibérée :`!Unpin`ne prend pas possession du flux, il l'emprunte seulement, et ne peut donc pas`Box::pin`le flux. Si le flux est`pin_mut!`, l'utilisateur doit d'abord

[FACT:tokio-stream/src/stream_ext.rs:116-121]

```rust
/// Note that because `next` doesn't take ownership over the stream,
/// the [`Stream`] type must be [`Unpin`]. If you want to use `next` with
/// a [`!Unpin`](Unpin) stream, you'll first have to pin the stream. This can
/// be done by boxing the stream using [`Box::pin`] or
/// pinning it to the stack using the `pin_mut!` macro from the `pin_utils`
/// crate.
```

## . La documentation souligne explicitement ce compromis :`merge`de l'interrogation

`merge`est le meilleur exemple pour comprendre comment les combinateurs réutilisent le Waker. Il entrelace la production de deux flux, et**garantit l'équité**— si les deux flux sont prêts simultanément, ils produisent en alternance. La documentation avertit explicitement de ne pas enchaîner les appels`merge`：

[FACT:tokio-stream/src/stream_ext.rs:319-321]

```rust
/// simultaneously, the merge stream alternates between them. This provides
/// some level of fairness. You should not chain calls to `merge`, as this
/// will break the fairness of the merging.
```

`merge`de la signature exige que les deux flux aient le même`Item`type :

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

Lorsque l'appelant`.next().await`, le flux d'exécution est le suivant :

1. `Next::poll`appelle`Merge::poll_next`。

2. `Merge`maintient en interne un indicateur booléen « à qui le tour la dernière fois ». Il commence par`poll`le flux qui n'a pas produit la dernière fois ; si`Pending`, puis`poll`l'autre.

3. Si les deux`Pending`，`Merge`retournent`Pending`, mais**les Wakers respectifs des deux flux sont déjà enregistrés**— dès que l'un est prêt, la tâche courante est réveillée.

4. Si un flux retourne`Ready(None)`(fin),`Merge`enregistre que ce flux est terminé, et ensuite ne`poll`l'autre flux, jusqu'à ce qu'il se termine aussi.

Le point clé ici est :`Merge`n'a pas sa propre logique de gestion de Waker, il transmet`cx`tel quel aux deux flux internes`poll_next`。**L'enregistrement du Waker est entièrement à la charge des flux sous-jacents**，`Merge`décide seulement « à qui demander en premier cette fois ». C'est exactement le sens littéral de « réutiliser le mécanisme de Waker sous-jacent ».

`merge_size_hints`La fonction auxiliaire montre comment les combinateurs fusionnent les indications de capacité :

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

Notez le choix entre`saturating_add`et`checked_add`: la borne inférieure utilise l'addition saturante (plutôt sous-estimer que déborder en panic), la borne supérieure utilise l'addition vérifiée (si l'un est inconnu, l'ensemble est inconnu). C'est le traitement typique du contrat de`size_hint`.

## Réflexion de conception : sécurité à l'annulation et`chunks_timeout`protection contre les panics

`StreamExt`La documentation annote chaque méthode avec**Cancel safety**. Prenons`next`comme exemple :

[FACT:tokio-stream/src/stream_ext.rs:123-127]

```rust
/// # Cancel safety
///
/// This method is cancel safe. The returned future only
/// holds onto a reference to the underlying stream,
/// so dropping it will never lose a value.
```

`next`est sûr à l'annulation parce qu'il emprunte seulement le flux, ne consomme pas d'éléments —`Next`lorsque le future est drop, l'état du flux lui-même reste inchangé, le prochain`next`refera`poll`。

Mais tous les combinateurs ne sont pas sûrs à l'annulation.`chunks_timeout`effectue une validation des paramètres dès la construction :

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
> `#[track_caller]`fait pointer la position du panic vers l'appelant plutôt que vers l'intérieur de la bibliothèque,`assert!`rejette dès la phase de construction`max_size == 0`. Pourquoi faut-il vérifier à la construction ? Si l'on autorisait`max_size == 0`，`ChunksTimeout`la logique de traitement par lots tomberait dans une boucle infinie « jamais assez pour remplir un lot » ou produirait des lots vides, et ce type de bug est extrêmement difficile à localiser à l'exécution. Le panic à la construction avance l'erreur au point observable le plus précoce.

`timeout`et`timeout_repeating`La différence mérite aussi attention :`timeout`retourne une erreur après le timeout, mais**continue d'interroger le flux interne**；`timeout_repeating`quant à lui, selon`Interval`produit continuellement des erreurs de timeout, jusqu'à ce que le flux interne produise une valeur. La documentation décrit précisément cette différence avec deux exemples :

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

# StreamMap : ensemble dynamique de flux et interrogation équitable

## Modèle intuitif

`select!`Le nombre de branches est fixé à la compilation. Mais le nombre de canaux à souscrire pour un service de chat, le nombre de connexions à suivre pour un crawler, ne sont connus qu'à l'exécution.`StreamMap`est précisément un`select!`ajoutable et supprimable à l'exécution : il place un nombre arbitraire de flux dans un ensemble, chaque`next`retourne`(key, value)`, vous indiquant de quel flux provient cette valeur. Sans lui, vous ne pourriez entasser tous les flux dans un`mpsc`canal, avec une couche de surcoût de transfert en plus.

## Structure de données et disposition mémoire

`StreamMap`Le stockage est extrêmement simple — un`Vec`：

[FACT:tokio-stream/src/stream_map.rs:204-208]

```rust
#[derive(Debug)]
pub struct StreamMap {
    /// Streams stored in the map
    entries: Vec,
}
```

La documentation explique explicitement le coût de ce choix :

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
> Pourquoi ne pas utiliser`HashMap`? Parce que`StreamMap`l'opération centrale de**interroger tous les flux**, et non une recherche par clé.`Vec`Le balayage linéaire de`swap_remove`est favorable au cache CPU, et`HashMap`est O(1). Si l'on utilisait`poll_next`, chaque`insert`devrait parcourir les buckets de hachage, avec une localité de cache bien pire.`remove`et

`insert`Le balayage O(n) de

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

`remove`L'implémentation de`swap_remove`reflète la sémantique « supprimer puis insérer » :

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

## utilise

`StreamMap`pour échanger l'élément supprimé avec le dernier élément puis le dépiler, évitant un déplacement O(n) :`poll_next_entry`Copie**Parcours guidé par scénario : point de départ aléatoire et correction du curseur de poll_next_entry**Le cœur de

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

. Il part d'un

**point de départ aléatoire** `thread_rng_n`pour commencer l'interrogation, afin de garantir l'équité — si l'on commençait toujours à l'indice 0, le premier flux affamerait les suivants :`FastRand`Copie`xorshift64+`Ce code comporte trois subtilités, décomposées une à une :

[FACT:tokio-stream/src/stream_map.rs:765-768]

```rust
/// Implement `xorshift64+`: 2 32-bit `xorshift` sequences added together.
/// Shift triplet `[17,7,16]` was calculated as indicated in Marsaglia's
/// `Xorshift` paper
```

`fastrand_n`utilise un`% n`：

[FACT:tokio-stream/src/stream_map.rs:787-792]

```rust
pub(crate) fn fastrand_n(&self, n: u32) -> u32 {
    // This is similar to fastrand() % n, but faster.
    // See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let mul = (self.fastrand() as u64).wrapping_mul(n as u64);
    (mul >> 32) as u32
}
```

**:`swap_remove`Copie**utilise le modulo multiplicatif de Lemire à la place de`idx`Copie`None`Deuxièmement,`swap_remove`la correction du curseur après`idx`. Lorsque l'indice**du flux retourne**est retiré,`start`déplace le dernier élément vers`idx < start && start <= self.entries.len()`. Cet élément déplacé peut`idx = idx.wrapping_add(1) % len`avoir déjà été interrogé`idx == len`(si son indice d'origine était avant

**). Le code utilise`Poll::Pending`pour détecter ce cas, et si oui le saute (**). Si c'est le dernier élément qui est retiré (`Pending`), le curseur revient à 0.

`poll_next`Troisièmement,`poll_next_entry`la sémantique de

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

. À ce moment, les Wakers de tous les flux sont enregistrés, et dès que l'un est prêt, il y a réveil.`ready!`ajoute la clé au-dessus de`poll_next_entry`:`Pending`Copie`poll_next`Notez la macro`Pending`。`K: Clone`: si`key.clone()`。

## retourne

`next_many`, tout`StreamMap`retourne immédiatement

[FACT:tokio-stream/src/stream_map.rs:581-583]

```rust
pub async fn next_many(&mut self, buffer: &mut Vec, limit: usize) -> usize {
    poll_fn(|cx| self.poll_next_many(cx, buffer, limit)).await
}
```

Réflexion de conception : sémantique par lots et sécurité à l'annulation de next_many

[FACT:tokio-stream/src/stream_map.rs:573-578]

```rust
/// # Cancel safety
///
/// This method is cancel safe. If `next_many` is used as the event in a
/// [`tokio::select!`] statement and some other branch completes first,
/// it is guaranteed that no items were received on any of the underlying
/// streams.
```

, collectant autant d'éléments prêts que possible en une fois :`next_many`Copie**Sa garantie de sécurité à l'annulation est cruciale :`buffer`**Copie`buffer`Pourquoi`buffer`est-il sûr à l'annulation ? Parce qu'il pousse les éléments

`poll_next_many`immédiatement dans le`poll_next_entry`fourni par l'appelant, plutôt que de les stocker temporairement en interne. Si le future est drop, les éléments déjà poussés restent dans

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

peut déjà contenir une partie des éléments — l'appelant doit en être conscient.`while added < limit`La structure de boucle de`for`est plus complexe que`should_loop = true`, car il doit collecter autant que possible en un seul tour :`limit`Copie

[FACT:tokio-stream/src/stream_map.rs:588-591]

```rust
/// * `Poll::Pending` if no items are available but the `StreamMap` is not empty.
/// * `Poll::Ready(count)` where `count` is the number of items successfully received and
///   stored in `buffer`. This can be less than, or equal to, `limit`.
/// * `Poll::Ready(0)` if `limit` is set to zero or when the `StreamMap` is empty.
```

`size_hint`L'implémentation de  montre comment agréger les indices de capacité de plusieurs flux :

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

Identique à`merge_size_hints`: saturation de la borne inférieure par addition, vérification de la borne supérieure par addition, et si l'un est inconnu, l'ensemble est inconnu.

Ci-dessous, un diagramme de flux illustre`poll_next_entry`le chemin de décision de :

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

# TaskTracker : encoder tous les états avec un seul AtomicUsize

## Modèle intuitif

Une fermeture élégante nécessite deux choses :**notifier aux tâches l'arrêt du travail**（`CancellationToken`en est responsable), et**attendre que les tâches se terminent réellement**（`TaskTracker`en est responsable).`TaskTracker`est comme une combinaison de « compteur de tâches + interrupteur de fermeture » : tant qu'il y a des tâches en cours d'exécution, ou que`close`，`wait()`n'a pas été appelé, il ne retournera pas. Sans lui, vous ne pourriez utiliser que`JoinSet`, mais`JoinSet`accumulerait la valeur de retour de chaque tâche, et un service fonctionnant à long terme provoquerait un OOM.

## Structure de données et disposition mémoire

`TaskTracker`est un`Arc`wrapper :

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

C'est la disposition mémoire la plus ingénieuse de ce chapitre :**un`AtomicUsize`encode simultanément « est-ce fermé » et « le compteur de tâches »**. Le bit le plus bas est le drapeau de fermeture, les autres bits sont le nombre de tâches (car le compteur de tâches à chaque`+2`, le bit le plus bas est toujours 0). Ainsi`is_closed_and_empty`ne nécessite qu'un seul chargement atomique :

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
> `state == 1`signifie « le bit de fermeture est 1, le compteur est 0 ». Pourquoi ne pas utiliser deux variables atomiques ? Deux variables nécessitent deux chargements, et il est impossible de déterminer atomiquement « les deux conditions sont satisfaites simultanément ». L'encodage à variable unique fait de`is_closed_and_empty`un seul`Acquire`chargement, et sur le chemin rapide de`wait`aucun verrou n'est nécessaire.

## Parcours guidé par scénario : la course entre close et drop_task

Considérons un scénario typique : le thread principal appelle`tracker.close()`, tandis que la dernière tâche est en train de se terminer (`TaskTrackerToken::drop`appelle`drop_task`). Les deux peuvent être concurrents, et il faut garantir que quel que soit celui qui agit en premier,`wait()`puisse être réveillé.

Regardons d'abord`set_closed`：

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

`fetch_or(1, AcqRel)`définit atomiquement le bit de fermeture et retourne l'ancienne valeur. Si l'ancienne valeur est 0 (précédemment non fermé et aucune tâche), cela signifie « après fermeture, immédiatement satisfait vide + fermé », on appelle`notify_now`. La valeur de retour`(state & 1) == 0`indique « cet appel a effectivement changé l'état ».

Regardons ensuite`drop_task`：

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

`fetch_sub(2, Release)`décrémente le compteur. Si l'ancienne valeur est 3 (binaire`11`: bit de fermeture 1 + compteur 1), cela signifie « c'est la dernière tâche et déjà fermé », on appelle`notify_now`。

Analyse de course des deux chemins :

- **close s'exécute en premier**：`set_closed`voit l'ancienne valeur`2`(compteur 1, non fermé), ne notifie pas. Ensuite`drop_task`voit l'ancienne valeur`3`, notifie. ✓
- **drop_task s'exécute en premier**：`drop_task`voit l'ancienne valeur`2`(compteur 1, non fermé), ne notifie pas. Ensuite`set_closed`voit l'ancienne valeur`0`(compteur 0, non fermé), notifie. ✓
- **Concurrence**：`fetch_or`et`fetch_sub`sont atomiques, quel que soit l'ordre d'entrelacement, il y en aura toujours un qui verra la combinaison « fermé + vide » et notifiera. ✓

`notify_now`contient un`Acquire`chargement facile à négliger :

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

Pourquoi`drop_task`utilise`Release`plutôt que`AcqRel`? Parce que le`drop_task`de`fetch_sub`nécessite seulement « rendre les écritures précédentes visibles aux lecteurs suivants » (sémantique Release), et non « voir les écritures des autres threads précédents » (sémantique Acquire). Mais`notify_now`nécessite Acquire pour établir le happens-before : garantir que tout le travail de nettoyage effectué avant la fin de la tâche soit visible pour le code après le retour de`wait()`. Le résultat de ce`load`est jeté, purement pour son effet de bord sur l'ordre mémoire — c'est un usage typique de « chargement de type fence » dans les opérations atomiques de Rust.

## Réflexion de conception : la résistance à l'ABA de wait et la sémantique de drop de TrackedFuture

`wait`retourne un`TaskTrackerWaitFuture`, qui détient en interne`Notified`：

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

Notons le champ`inner`: si à la création c'est déjà « fermé et vide », on le met directement à`None`，`poll`et on retourne immédiatement`Ready`. C'est le chemin rapide.

La documentation souligne particulièrement la résistance à l'ABA :

[FACT:tokio-util/src/task/task_tracker.rs:304-307]

```rust
/// The `wait` future is resistant against [ABA problems][aba]. That is, if the `TaskTracker`
/// becomes both closed and empty for a short amount of time, then it is guarantee that all
/// `wait` futures that were created before the short time interval will trigger, even if they
/// are not polled during that short time interval.
```

Cette garantie provient de la sémantique de`Notify::notified()`:`Notified`le future enregistre son identité de « waiter » dès sa création, même si`notify_waiters`est appelé avant qu'il ne soit`poll`, il verra la notification lors de son premier`poll`.`TaskTrackerWaitFuture::poll`L'implémentation de

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

Copier`poll`Chaque`is_closed_and_empty()`vérifie d'abord`poll` `Notified`, puis`Notified`. Cet ordre garantit : même si

`TrackedFuture`n'est pas réveillé pour une raison quelconque, la vérification d'état peut servir de filet de sécurité.`TaskTracker`La sémantique de drop de`JoinSet`est la différence fondamentale entre

[FACT:tokio-util/src/task/task_tracker.rs:488-494]

```rust
/// The task is removed from the collection when it is dropped, not when [`poll`] returns
/// [`Poll::Ready`].
```

Copier`Ready`Cela signifie : même si le future a déjà retourné`TrackedFuture`, tant que`TaskTracker`lui-même n'a pas été drop,

[FACT:tokio-util/src/task/task_tracker.rs:33-35]

```rust
/// When a call to [`wait`] returns, it is guaranteed that all tracked tasks have exited and that
/// the destructor of the future has finished running. However, there might be a short amount of
/// time where [`JoinHandle::is_finished`] returns false.
```

`TaskTrackerToken`Copier`Drop`Le

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

`TrackedFuture`est le point de déclenchement de la décrémentation du compteur :`pin_project!`Copier`token`empaquette`future`et`token`via`spawn_blocking`, le drop de

[FACT:tokio-util/src/task/task_tracker.rs:452-464]

déclenche automatiquement la décrémentation du compteur.
