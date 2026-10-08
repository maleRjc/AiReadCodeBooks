# Глава 5: Уведомление о готовности I/O: как Reactor переводит события epoll в пробуждение Waker

В предыдущей главе мы проследили главный цикл worker-потока: задача poll-ится, при возврате Pending Waker сохраняется куда-то, после готовности события Waker срабатывает, и задача снова ставится в очередь. Но что это за «куда-то»? Как Waker находится при поступлении события epoll? Именно на это должен ответить Reactor. Сначала построим интуитивную модель: представьте весь механизм уведомления о готовности I/O как систему вызова по номеру в ресторане — клиент (задача), сделав заказ, не стоит и не ждёт у окна, а берёт пейджер (Waker) и возвращается на место; кухня (ядро epoll), приготовив блюдо, сообщает стойке (Reactor), которая по номеру заказа (Token) находит соответствующий пейджер и нажимает кнопку. Без этой системы каждой задаче пришлось бы опрашивать socket, и CPU сгорел бы; либо использовались бы блокирующие потоки — один поток на соединение, и масштабирование было бы невозможно. Reactor в Tokio состоит из трёх файлов, образующих трёхуровневую структуру со строгим разделением обязанностей: driver.rs — это сам цикл событий, владеющий mio::Poll, отвечающий за вызов poll() для блокирующего ожидания событий ядра и перевод событий в чтение/запись ScheduledIo; registration.rs — это пользовательский регистрационный дескриптор, который хранится внутри TcpStream и предоставляет API poll_read_ready / poll_write_ready и т. п.; scheduled_io.rs — это слот состояния каждого fd, хранящий биты готовности чтения/записи и список Waker, являясь мостом между событиями и задачами. Связи сборки модулей можно посмотреть в tokio/src/runtime/io/mod.rs:5-16: driver экспортирует Driver, Handle, ReadyEvent, registration экспортирует Registration, scheduled_io экспортирует ScheduledIo. Следующая схема фиксирует полный поток данных, который нужно проследить в этой главе: TcpStream → Registration → ScheduledIo → Handle/Driver → ядро → обратно в ScheduledIo → Waker. Далее мы разберём это слой за слоем.

# Уровень драйвера:`Driver`и`Handle`разделение обязанностей

## Интуитивная модель

`Driver`— это**единственная сущность, владеющая`mio::Poll`, к ней можно обращаться только из одного потока**— это требование эксклюзивности цикла событий. А`&mut`— это`Handle`клонируемая точка входа для регистрации, разделяемая между потоками**клонируемая, разделяемая между потоками точка регистрации**, любой поток, желающий зарегистрировать новый fd, делает это через него. Без этого разделения пришлось бы либо`mio::Poll`блокировать, либо возвращать все регистрации в поток driver (что вводит межпоточную очередь сообщений). Tokio выбирает, чтобы`Handle`напрямую владел клоном`mio::Registry`, регистрация может выполняться параллельно, и только фактическое ожидание событий требует эксклюзивного доступа.

## Раскладка памяти и поля

Сначала рассмотрим`Driver`поля[FACT:tokio/src/runtime/io/driver.rs:25-38]：

- `signal_ready: bool`: пришло ли событие Unix-сигнала, используется для signal-драйвера.
- `events: mio::Events`: основной буфер событий, переиспользуется между вызовами`turn`, чтобы избежать выделения памяти каждый раз.
- `events_busy: Option<mio::Events>`：**Специальный буфер для неблокирующего poll**, существует только когда`max_io_events_per_busy_tick`установлен.
- `poll: mio::Poll`: обёртка над очередью событий ядра.

Теперь рассмотрим`Handle` [FACT:tokio/src/runtime/io/driver.rs:41-75]：

- `registry: mio::Registry`：`mio::Poll::registry()`клон, используемый для`register`/`deregister`。
- `registrations: RegistrationSet`: множество всех активных регистраций, отвечает за выделение`Token`и`ScheduledIo`。
- `synced: Mutex<registration_set::Synced>`: защищает синхронизированное состояние`RegistrationSet`.
- `waker: mio::Waker`: используется для пробуждения driver, заблокированного в`turn`, из любого потока.
- `metrics: IoDriverMetrics`: подсчитывает количество fd, количество готовых событий.

Здесь есть ключевое проектное решение:`events_busy`существование[FACT:tokio/src/runtime/io/driver.rs:25-38]предназначено для решения**проблемы, когда неблокирующий poll поглощает события**. Комментарий[FACT:tokio/src/runtime/io/driver.rs:189-190]ясно говорит: если события, извлечённые неблокирующим poll, остаются в основном буфере, следующий poll их не увидит; используя отдельный буфер, необработанные события остаются в очереди ядра, и следующий poll вернёт их снова.

## Пошагово: одно выполнение`turn`

`turn`— это основная функция driver[FACT:tokio/src/runtime/io/driver.rs:184-261]. Предположим, worker-поток обнаружил, что нет задач для выполнения, и вызывает`park` → `turn(handle, None)`для блокирующего ожидания:

**Шаг первый**: утверждается, что не shutdown[FACT:tokio/src/runtime/io/driver.rs:185], и освобождаются регистрации, ожидающие очистки[FACT:tokio/src/runtime/io/driver.rs:187]。`release_pending_registrations`проверяет`needs_release()`, и если есть, вызывает`registrations.release()` [FACT:tokio/src/runtime/io/driver.rs:336-340]。

**Шаг второй**: выбирается буфер событий[FACT:tokio/src/runtime/io/driver.rs:191-194]. Если`max_wait`равен нулю и`events_busy`существует, используется busy-буфер; иначе используется основной буфер.

**Шаг третий**: вызывается`self.poll.poll(events, max_wait)` [FACT:tokio/src/runtime/io/driver.rs:198]. Это место, где происходит реальная блокировка на epoll_wait. Обработка ошибок очень сдержанная:`Interrupted`просто игнорируется (прерывание сигналом — это нормально)[FACT:tokio/src/runtime/io/driver.rs:200], под WASI`InvalidInput`также игнорируется[FACT:tokio/src/runtime/io/driver.rs:201-205], другие ошибки приводят к panic[FACT:tokio/src/runtime/io/driver.rs:206]。

**Шаг четвёртый**: перебираются события[FACT:tokio/src/runtime/io/driver.rs:211-233]. Для каждого`event`：

- если`token == TOKEN_WAKEUP`(значение 0)[FACT:tokio/src/runtime/io/driver.rs:214], ничего не делается — это`unpark`используется для прерывания блокировки.
- Если`token == TOKEN_SIGNAL`(значение 1)[FACT:tokio/src/runtime/io/driver.rs:216], устанавливается`signal_ready = true`。
- иначе это обычное событие I/O[FACT:tokio/src/runtime/io/driver.rs:218-231]: преобразуется`mio::Ready`в Tokio-представление`Ready`, с помощью`EXPOSE_IO.from_exposed_addr(token.0)`token восстанавливается в указатель`*const ScheduledIo`, затем`set_readiness(Tick::Set, |curr| curr | ready)`накапливаются биты готовности, и далее`io.wake(ready)`запускается соответствующее направление`Waker`。

Здесь`EXPOSE_IO`— это`PtrExposeDomain<ScheduledIo>` [FACT:tokio/src/runtime/io/mod.rs:21-22], который «экспонирует» указатель как`usize`в качестве`mio::Token`. Комментарий о безопасности[FACT:tokio/src/runtime/io/driver.rs:222-225]объясняет, почему это unsafe-преобразование безопасно: указатель не освобождается до тех пор, пока не будет отменена регистрация в mio**и**driver больше не выполняет параллельный poll, и driver владеет`Arc<ScheduledIo>`.

**Шаг пятый**: обработка очереди завершений io_uring (только Linux + tokio_unstable)[FACT:tokio/src/runtime/io/driver.rs:235-258], включая цикл flush при переполнении CQ.

**Шаг шестой**: накопление метрик[FACT:tokio/src/runtime/io/driver.rs:265-267]。

```mermaid
flowchart TD
    start["turn(handle, max_wait)"] --> assert["debug_assert!(!is_shutdown)"]
    assert --> release["release_pending_registrations()"]
    release --> pick{"max_wait == 0и events_busy существует?"}
    pick -->|да| busy["events = events_busy"]
    pick -->|нет| main["events = events"]
    busy --> poll["poll.poll(events, max_wait)"]
    main --> poll
    poll --> pollres{"poll вернул?"}
    pollres -->|"Ok / Interrupted"| iter["обход events.iter()"]
    pollres -->|"другие Err"| panic["panic!(unexpected error)"]
    iter --> tok{"event.token()?"}
    tok -->|"TOKEN_WAKEUP"| skip["игнорировать, используется только для прерывания блокировки"]
    tok -->|"TOKEN_SIGNAL"| sig["signal_ready = true"]
    tok -->|"обычный fd token"| cast["EXPOSE_IO.from_exposed_addr(token.0)"]
    cast --> setr["io.set_readiness(Tick::Set, curr | ready)"]
    setr --> wake["io.wake(ready)"]
    wake --> iter
    skip --> iter
    sig --> iter
    iter --> uring["dispatch_completions() (io-uring)"]
    uring --> metrics["metrics.incr_ready_count_by(ready_count)"]
```

## Проектное размышление: почему`Handle`должен владеть`mio::Waker`

`unpark` [FACT:tokio/src/runtime/io/driver.rs:280-283]вызывает`self.waker.wake()`. Этот`mio::Waker`при`Driver::new`регистрируется через`TOKEN_WAKEUP`с помощью[FACT:tokio/src/runtime/io/driver.rs:124]. Когда driver заблокирован в`poll.poll()`, другой поток, вызывая`unpark`, помещает в epoll событие`TOKEN_WAKEUP`,`poll`немедленно возвращается, при переборе видит этот token и просто пропускает его[FACT:tokio/src/runtime/io/driver.rs:214-215]。

> **[Design Inference & Architectural Trade-offs]**
> Этот механизм используется в`deregister_source`при[FACT:tokio/src/runtime/io/driver.rs:315-334]: после отмены регистрации source, если`registrations.deregister`возвращает true (означая, что это последняя ссылка), выполняется`unpark()`. Почему? Потому что driver может быть заблокирован в`poll`в ожидании события для этого fd, а fd уже отменён, и ядро больше не сгенерирует событие; необходимо активно разбудить driver, чтобы он перепроверил набор регистраций и, возможно, вышел из блокировки. Иначе driver будет спать до тайм-аута`max_wait`, задерживая shutdown.

Ещё одна деталь:`deregister_source`сначала вызывается`self.registry.deregister(source)` [FACT:tokio/src/runtime/io/driver.rs:322], затем очищается`registrations` [FACT:tokio/src/runtime/io/driver.rs:315-334]. Комментарий[FACT:tokio/src/runtime/io/driver.rs:320-321]говорит «Cleanup ALWAYS happens» — даже если отмена регистрации на уровне ОС не удалась, внутреннее состояние всё равно очищается, и только потом возвращается ошибка ОС[FACT:tokio/src/runtime/io/driver.rs:336-340]. Это типичный**паттерн, где очистка ресурсов имеет приоритет над распространением ошибки**.

# Слой регистрации:`Registration`как`Waker`сохраняется в`ScheduledIo`

## Интуитивная модель

`Registration`— это**контракт между задачей и fd**. Он содержит две вещи: один`scheduler::Handle`(используется для доступа к runtime при необходимости), один`Arc<ScheduledIo>`(слот состояния fd). Когда задача вызывает`poll_read_ready`,`Registration`передаёт`Waker`на хранение в`ScheduledIo`; когда driver получает событие, он извлекает`ScheduledIo`из`Waker`и пробуждает.

## Раскладка памяти и поля

`Registration`содержит только два поля[FACT:tokio/src/runtime/io/registration.rs:46-54]：

- `handle: scheduler::Handle`: дескриптор runtime, комментарий[FACT:tokio/src/runtime/io/registration.rs:46-54]говорит «TODO: this can probably be moved into ScheduledIo», что указывает на то, что автор считает расположение этого поля возможным для оптимизации.
- `shared: Arc<ScheduledIo>`: разделяемое состояние,`Arc`гарантирует, что и driver, и задача могут получить к нему доступ.

> **[Design Inference & Architectural Trade-offs]**
> Обратите внимание, что`Registration`вручную реализует`Send`и`Sync` [FACT:tokio/src/runtime/io/registration.rs:57-58]. Зачем нужен unsafe impl? Потому что`scheduler::Handle`внутри может содержать поля, не являющиеся`Send`/`Sync`(например,`Rc`), но сценарий использования`Registration`требует, чтобы он мог пересекать потоки. Комментарий к документации[FACT:tokio/src/runtime/io/registration.rs:28-33]задаёт ключевое ограничение:**вызывающий должен гарантировать, что не более двух задач одновременно используют один и тот же`Registration`**, одна на чтение, одна на запись. Нарушение этого ограничения хотя и безопасно с точки зрения памяти, но приводит к потере уведомлений и зависанию задач.

## Step-by-Step：`poll_read_ready`Цепочка вызовов

Предположим, задача в`TcpStream::poll_read`обнаруживается, что в socket нет данных, необходимо зарегистрировать интерес чтения. Цепочка вызовов:`TcpStream::poll_read_priv` → `PollEvented::poll_read` → `Registration::poll_read_io` → `poll_io` → `poll_ready`。

`poll_ready`— это ядро[FACT:tokio/src/runtime/io/registration.rs:155-171]：

**Первый шаг**：`trace_leaf()` [FACT:tokio/src/runtime/io/registration.rs:160], используется для tracing-инструментирования.

**Второй шаг**：`coop::poll_proceed(cx)` [FACT:tokio/src/runtime/io/registration.rs:155-171]. Это механизм кооперативного бюджета, о котором пойдёт речь в главе 12. Если бюджет исчерпан, возвращается`Pending`и регистрируется специальный`Waker`, чтобы задача была перепланирована в следующем раунде.

**Третий шаг**：`self.shared.poll_readiness(cx, direction)` [FACT:tokio/src/runtime/io/registration.rs:155-171]. Это место, где происходит реальное взаимодействие с`ScheduledIo`: проверяется текущий бит готовности, и если уже готово — немедленно возвращается`Ready`; иначе`cx.waker()`сохраняется в`ScheduledIo`в соответствующий слот направления, возвращается`Pending`。

**Четвёртый шаг**: проверяется`ev.is_shutdown` [FACT:tokio/src/runtime/io/registration.rs:155-171]. Если runtime завершается, возвращается`RUNTIME_SHUTTING_DOWN_ERROR`。

**Пятый шаг**：`coop.made_progress()` [FACT:tokio/src/runtime/io/registration.rs:169], отмечается расход бюджета, возвращается событие готовности.

`poll_io`поверх`poll_ready`добавляет слой цикла повторных попыток[FACT:tokio/src/runtime/io/registration.rs:173-192]：

```rust
loop {
    let ev = ready!(self.poll_ready(cx, direction))?;
    match f() {
        Ok(ret) => return Poll::Ready(Ok(ret)),
        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
            self.clear_readiness(ev);
        }
        Err(e) => return Poll::Ready(Err(e)),
    }
}
```

Здесь отражена**readiness — это подсказка, а не гарантия**ключевая идея:`poll_ready`говорит, что доступно для чтения, но при реальном`read()`может вернуться`WouldBlock`(например, другой поток успел забрать данные). В этом случае необходимо`clear_readiness(ev)` [FACT:tokio/src/runtime/io/registration.rs:187]сбросить бит готовности и затем в цикле снова ждать. Если не сбросить, задача попадёт в busy-loop «думает, что можно читать → read fails → снова думает, что можно читать».

## Размышления о дизайне:`try_io`и`async_io`разделение обязанностей

`try_io` [FACT:tokio/src/runtime/io/registration.rs:194-213]— синхронная версия: сначала`ready_event(interest)`проверяется бит готовности, если пусто — сразу возвращается`WouldBlock` [FACT:tokio/src/runtime/io/registration.rs:194-213]; иначе выполняется`f()`, и если`f()`возвращает`WouldBlock`, то сбрасывается бит готовности[FACT:tokio/src/runtime/io/registration.rs:207-210]. Он**не регистрирует Waker**, подходит для`try_read`таких сценариев «попробовал и ушёл».

`async_io` [FACT:tokio/src/runtime/io/registration.rs:225-245]— асинхронная версия:`readiness(interest).await`регистрирует Waker и ждёт, затем при выполнении`f()`，`WouldBlock`сбрасывает бит готовности и зацикливается. Обратите внимание, что в цикле также вызывается`coop::poll_proceed` [FACT:tokio/src/runtime/io/registration.rs:233], чтобы не исчерпать бюджет при множественных`WouldBlock`повторных попытках.

## Производственные подводные камни:`Drop`Очистка Waker в

`Registration::drop` [FACT:tokio/src/runtime/io/registration.rs:253-262]вызывает`self.shared.clear_wakers()`. Комментарий[FACT:tokio/src/runtime/io/registration.rs:253-262]объясняет причину:`ScheduledIo`хранящийся в`Waker`может держать`Arc<driver::Inner>`, а`driver::Inner`в свою очередь держит`ScheduledIo`, образуя циклическую ссылку. Очистка Waker — это способ разорвать цикл. Но комментарий также признаёт, что это «imperfect solution» — если`Registration`сам сохранён в`Waker`, цикл всё равно остаётся. Это проблема, обсуждаемая в tokio-rs/tokio#3481.

> **[Design Inference & Architectural Trade-offs]**
> В production поведение таково: если множество соединений было drop, но runtime не завершился, память не освобождается немедленно до следующего`clear_wakers`или завершение работы runtime. Для сервисов с длительными соединениями это обычно не проблема; но для сценариев с короткими соединениями и частым созданием/уничтожением нужно следить за моментом освобождения`ScheduledIo`.

# От`TcpStream::read`до`Waker`полная цепочка пробуждения

## Интуитивная модель

Теперь свяжем три слоя вместе. Пользователь на`TcpStream`вызывает`.read().await`, фактически выполняется`AsyncRead::poll_read` → `PollEvented::poll_read` → `Registration::poll_read_io`. Когда данных нет,`Waker`сохраняется в`ScheduledIo`; когда epoll сообщает о готовности к чтению, driver извлекает из`ScheduledIo``Waker`и пробуждает, задача перепланируется, и при повторном poll`poll_readiness`обнаруживает, что бит готовности установлен, и сразу возвращает`Ready`，`read()`успешно.

## Step-by-Step: одно полное ожидание чтения

**Этап первый: регистрация интереса**。`TcpStream::new` [FACT:tokio/src/net/tcp/stream.rs:166-169]вызывает`PollEvented::new(connected)`, который внутри вызывает`Registration::new_with_interest_and_handle` [FACT:tokio/src/runtime/io/registration.rs:73-81], а затем`handle.driver().io().add_source(io, interest)` [FACT:tokio/src/runtime/io/registration.rs:73-81]。

`add_source` [FACT:tokio/src/runtime/io/driver.rs:288-312]делает три вещи:

1. `registrations.allocate(&mut synced.lock())`выделяет`ScheduledIo`, получает`token` [FACT:tokio/src/runtime/io/driver.rs:293-294]。

2. `self.registry.register(source, token, interest.to_mio())`регистрирует[FACT:tokio/src/runtime/io/driver.rs:298]в ядре. Если неудача,**необходимо**удалить только что выделенный`ScheduledIo`из множества[FACT:tokio/src/runtime/io/driver.rs:300-303], иначе утечка.

3. `metrics.incr_fd_count()`подсчитывает[FACT:tokio/src/runtime/io/driver.rs:309]。

**Этап второй: ожидание готовности**. Задача poll`TcpStream::poll_read` [FACT:tokio/src/net/tcp/stream.rs:1492-1498] → `poll_read_priv` [FACT:tokio/src/net/tcp/stream.rs:1451-1458] → `PollEvented::poll_read` → `Registration::poll_read_io` [FACT:tokio/src/runtime/io/registration.rs:133-139] → `poll_io` → `poll_ready` → `ScheduledIo::poll_readiness`. Если в этот момент не готово,`Waker`сохраняется в`ScheduledIo`слот чтения, возвращается`Pending`。

**Этап третий: приход события**. driver`turn`из`poll.poll()`получает событие[FACT:tokio/src/runtime/io/driver.rs:198], при обходе для каждого события fd выполняет`io.set_readiness(Tick::Set, |curr| curr | ready)`и`io.wake(ready)` [FACT:tokio/src/runtime/io/driver.rs:228-229]。`wake`внутри извлекает`Waker`соответствующего направления и вызывает`wake()`。

**Этап четвёртый: перепланирование задачи**。`Waker::wake()`повторно ставит задачу в локальную очередь worker (рассказывалось в предыдущей главе). worker снова poll-ит эту задачу,`poll_readiness`обнаруживает, что бит готовности установлен, возвращает`Ready`，`read()`успешно.

```mermaid
sequenceDiagram
    participant Task as "Задача (worker-поток)"
    participant Reg as "Registration"
    participant SIO as "ScheduledIo"
    participant Drv as "Driver (поток ввода-вывода)"
    participant OS as "epoll/kqueue"

    Task->>Reg: "poll_read_ready(cx)"
    Reg->>SIO: "poll_readiness(cx, Read)"
    SIO-->>Reg: "Pending (Waker сохранён в слоте чтения)"
    Reg-->>Task: "Poll::Pending"
    Note over Task: задача уступает, worker идёт выполнять другие задачи
    Drv->>OS: "poll.poll(events, max_wait)"
    OS-->>Drv: "event(token=fd_ptr, READABLE)"
    Drv->>SIO: "set_readiness(Tick::Set, curr | READABLE)"
    Drv->>SIO: "wake(READABLE)"
    SIO->>Task: "Waker::wake() повторная постановка в очередь"
    Note over Task: worker снова опрашивает эту задачу
    Task->>Reg: "poll_read_ready(cx)"
    Reg->>SIO: "poll_readiness(cx, Read)"
    SIO-->>Reg: "Ready(ReadyEvent{ready: READABLE})"
    Reg-->>Task: "Poll::Ready(Ok(ev))"
    Task->>Task: "read() успешно возвращает данные"
```

## Важное ответвление:`assume_ready`оптимизация

`TcpStream::new_accepted` [FACT:tokio/src/net/tcp/stream.rs:174-181]— заметная оптимизация.`accept`возвращаемый socket естественно доступен для записи и обычно уже содержит первую партию байтов от peer. Если ждать первого события driver, при высокой нагрузке это событие может оказаться позади событий всех установленных соединений, вызывая задержку. Поэтому`new_accepted`напрямую вызывает`assume_ready(Ready::READABLE | Ready::WRITABLE)` [FACT:tokio/src/net/tcp/stream.rs:174-181]。

`assume_ready`, комментарий[FACT:tokio/src/runtime/io/registration.rs:103-105]гласит: «A wrong guess costs one`WouldBlock`, which clears the readiness again.» — цена ошибочной догадки — всего один`WouldBlock`，`poll_io`цикл очистит бит готовности и снова будет ждать. Это дизайн**оптимистичной догадки + быстрого исправления ошибок**.

## Размышления о дизайне: почему I/O-драйвер отделён от планировщика

> **[Design Inference & Architectural Trade-offs]**
> Судя по структуре исходников,`Driver`и worker-потоки разделены:`Driver`размещается в некотором выделенном месте runtime (обычно`block_on`поток или специальный I/O-поток), а worker-потоки держат только`Handle`. Такое разделение даёт несколько преимуществ:

1. **Регистрация без блокировок**：`Handle`держит`mio::Registry`клон, любой worker может параллельно регистрировать новые fd, не возвращаясь в поток driver.

2. **Централизация ожидания событий**: только один поток блокируется на`epoll_wait`, что позволяет избежать проблемы thundering herd при одновременном poll одного и того же epoll fd из нескольких потоков.

3. **Короткий путь пробуждения**: driver, получив событие, напрямую оперирует`ScheduledIo`и вызывает`Waker::wake()`，`wake()`внутри ставит задачу в очередь worker, без межпоточной передачи сообщений.

Цена — необходимость`ScheduledIo`обрабатывать конкурентный доступ (`set_readiness`и`poll_readiness`могут происходить одновременно), что решается атомарными операциями и внутренними блокировками.

## Производственные подводные камни:`is_shutdown`и`RUNTIME_SHUTTING_DOWN_ERROR`

`poll_ready`проверяет`ev.is_shutdown` [FACT:tokio/src/runtime/io/registration.rs:155-171], если истина — возвращает`gone()` [FACT:tokio/src/runtime/io/registration.rs:265-267], то есть`RUNTIME_SHUTTING_DOWN_ERROR`。

> **[Design Inference & Architectural Trade-offs]**
> Смысл этой проверки в том, что: при завершении runtime driver`shutdown` [FACT:tokio/src/runtime/io/driver.rs:174-182]обойдёт все регистрации и вызовет`io.shutdown()`, установит`is_shutdown`и разбудит всех ожидающих. Если не проверять этот флаг, задача может попытаться прочитать сокет после того, как runtime уже прекратил планирование, что приведёт к неопределённому поведению или зависанию. В производственной среде, если вы видите`RUNTIME_SHUTTING_DOWN_ERROR`, это обычно означает, что какая-то задача всё ещё выполняется после drop runtime — проверьте, нет ли`spawn`задач, которые не были корректно присоединены через join.

Ещё одна ловушка —`deregister_source`из`unpark` [FACT:tokio/src/runtime/io/driver.rs:328]. Если driver в данный момент заблокирован в`poll`, и в этот момент последний`Registration`был drop,`unpark`разбудит driver. Но если driver не находится в состоянии блокировки (например, обрабатывает другие события),`unpark`лишь заставит следующий`turn`немедленно вернуть[FACT:tokio/src/runtime/io/driver.rs:280-283]. Эта семантика описана в комментариях к документации`Handle::unpark`.

# Проектное решение: три ключевых компромисса Reactor

**Компромисс первый:`Token`использовать указатели вместо индексов**。`EXPOSE_IO.from_exposed_addr(token.0)` [FACT:tokio/src/runtime/io/driver.rs:220]Рассматривать`mio::Token`напрямую как адрес`*const ScheduledIo`. Это позволяет избежать поддержки`Token → ScheduledIo`таблица сопоставления, поиск за O(1) и без блокировок. Цена — безопасность зависит от строгого управления жизненным циклом: указатель может быть освобождён только после дерегистрации и прекращения опроса драйвером[FACT:tokio/src/runtime/io/driver.rs:222-225]。

**Компромисс второй: два слота Waker для чтения и записи**。`Registration`Документация[FACT:tokio/src/runtime/io/registration.rs:24-26]гласит: «A registration instance represents two separate readiness streams» — для чтения и записи имеется независимый`Waker`слот. Это позволяет задачам чтения и записи одного и того же сокета регистрироваться отдельно, не мешая друг другу. Но`poll_read_ready`комментарий[FACT:tokio/src/net/tcp/stream.rs:549-552]предупреждает: при многократном вызове`poll_read_ready`/`poll_read`/`poll_peek`сохраняется только последний`Waker`— для направления чтения существует только один слот.

**Компромисс третий:`events_busy`независимый буфер**. Тест[FACT:tokio/src/runtime/io/driver.rs:364-386]подтвердил это поведение:`Driver::new(16, Some(2))`создаётся драйвер с ёмкостью busy равной 2, после регистрации 5 источников, доступных для чтения, неблокирующий`turn`извлекает только 2 события[FACT:tokio/src/runtime/io/driver.rs:375-376], остальные 3 остаются в очереди ядра и будут заблокированы в следующий раз`turn`получено[FACT:tokio/src/runtime/io/driver.rs:379-380]. Это предотвращает ситуацию, когда неблокирующий poll за один раз поглощает все события, что приводит к голоданию последующих poll.

# Резюме главы

В этой главе прослежен`TcpStream::read`полный путь Reactor, стоящий за

- **Уровень драйвера**：`Driver`эксклюзивно`mio::Poll`，`turn`блокирующее ожидание события, с помощью`EXPOSE_IO`восстановить`Token`в`ScheduledIo`указатель, вызвать`set_readiness` + `wake`запустить`Waker`。`Handle`предоставляет точку регистрации, доступную для межпоточного использования,`unpark`используется для прерывания блокировки.
- **Слой регистрации**：`Registration`Содержит`Arc<ScheduledIo>`，`poll_ready`Проверяет бит готовности или сохраняет`Waker`，`poll_io`Использует`WouldBlock`Цикл повторных попыток для обработки ложных срабатываний,`try_io`/`async_io`Обслуживает синхронные и асинхронные сценарии по отдельности.
- **Слой состояния**：`ScheduledIo`Является слотом состояния fd, хранит биты готовности чтения/записи и двойной`Waker`Слот, является единственным мостом между событиями и задачами.

# Вопросы для размышления и самопроверки в этой главе

Q1: Если удалить`poll_io`в`WouldBlock`ветви`self.clear_readiness(ev)`то в каком сценарии это приведёт к busy-loop задачи? Почему?

**Справочный анализ**：`poll_io`цикл[FACT:tokio/src/runtime/io/registration.rs:173-192]в`f()`возврат`WouldBlock`вызывается при`clear_readiness(ev)` [FACT:tokio/src/runtime/io/registration.rs:187]。`ev`является`poll_ready`возвращаемым`ReadyEvent`, содержит текущие биты готовности.`clear_readiness`удалит эти биты из`ScheduledIo`.

Если не очистить, при следующем вызове цикла`poll_ready` → `poll_readiness`,`ScheduledIo`по-прежнему сохраняются старые биты «доступно для чтения»,`poll_readiness`немедленно вернёт`Ready`(поскольку биты готовности не пусты), затем`f()`снова выполнит`read()`, если в socket действительно нет данных, и снова возвращается`WouldBlock`, цикл продолжается. Поскольку бит готовности никогда не сбрасывается, этот цикл никогда не войдёт в`Pending`, задача будет постоянно занимать CPU опросом.

Сценарий срабатывания: несколько задач совместно используют одно направление чтения одного socket (хотя`Registration`документация[FACT:tokio/src/runtime/io/registration.rs:28-33]говорит, что максимум две задачи, но для направления чтения есть только один слот), или`try_read`и`poll_read`используются совместно. Более распространённый случай: после того как epoll сообщил о готовности к чтению, другой поток успел первым вычитать данные, и`read()`текущей задачи возвращает`WouldBlock`, в этот момент необходимо сбросить бит готовности, иначе будут бесконечные повторные попытки.

Q2: `add_source`При`registry.register`почему при сбое вызывается`registrations.remove`? Что произойдёт, если не вызвать?

**Справочный анализ**：`add_source` [FACT:tokio/src/runtime/io/driver.rs:288-312]Сначала`registrations.allocate`Выделить`ScheduledIo` [FACT:tokio/src/runtime/io/driver.rs:293], затем`registry.register`зарегистрировать в ядре[FACT:tokio/src/runtime/io/driver.rs:298]. Если регистрация не удалась,`ScheduledIo`уже выделен, но не связан ни с одним fd; если не удалить, он навсегда останется в`RegistrationSet`.

Комментарий[FACT:tokio/src/runtime/io/driver.rs:296-297]явно говорит: «we should remove the`scheduled_io` from the `registrations` set if registering the `source` with the OS fails. Otherwise it will leak the `scheduled_io`.» — это утечка памяти.

`remove`Вызов[FACT:tokio/src/runtime/io/driver.rs:300-303]обёрнут в блок unsafe, потому что`ScheduledIo`является`RegistrationSet`частью, и операция удаления должна гарантировать отсутствие других ссылок. Последствия утечки:`RegistrationSet`постоянно растёт,`Token`пространство расходуется впустую, что в конечном итоге может привести к`allocate`Сбой или исчерпание памяти. В сценариях с частым созданием/уничтожением соединений (например, серверы с короткими соединениями), если вероятность сбоя регистрации высока (например, исчерпание fd), утечка ускоряет исчерпание ресурсов.

Q3: `deregister_source`, почему`unpark()`только при`registrations.deregister`возвращает true? Какие проблемы возникнут, если вызывать безусловно?

**Справочный анализ**：`deregister_source` [FACT:tokio/src/runtime/io/driver.rs:315-334]Логика такова: сначала`registry.deregister(source)`отменяет регистрацию в ядре[FACT:tokio/src/runtime/io/driver.rs:322], затем`registrations.deregister`очищает внутреннее состояние[FACT:tokio/src/runtime/io/driver.rs:315-334], если возвращает true, то`unpark()` [FACT:tokio/src/runtime/io/driver.rs:328]。

`registrations.deregister`возврат true означает, что это последняя ссылка,`ScheduledIo`действительно удалён. В этот момент driver может быть заблокирован в`poll`в ожидании события для этого fd, но fd уже отменён, и ядро больше не будет генерировать события.`unpark`через`mio::Waker`помещает в epoll`TOKEN_WAKEUP`Событие[FACT:tokio/src/runtime/io/driver.rs:280-283], позволяя`poll`немедленно вернуться, driver повторно проверяет набор регистраций и может выйти из блокировки.

Если безусловно вызывать`unpark`: каждая отмена регистрации не последней ссылки пробуждает driver, вызывая ненужные пробуждения. В сценариях, где множество соединений совместно используют один и тот же`ScheduledIo`(например,`TcpStream`из`split`после разделения на половины чтения и записи), каждое удаление одной половины пробуждает driver, увеличивая нагрузку на CPU. Что ещё серьёзнее

本章我们拆解了 Reactor 如何把 epoll 事件翻译成 Waker 唤醒：从 TcpStream 的 poll_read_ready 出发，经过 Registration 的注册与查询，落到 ScheduledIo 的就绪位与 Waker 槽位，再由 Driver 在事件循环中根据 Token 定位并触发唤醒。关键设计包括：Token 即指针实现 O(1) 查找，读写双 Waker 槽位支持并发读写分离，events_busy 独立缓冲区防止事件饥饿，assume_ready 乐观猜测优化 accept 场景。至此，I/O 就绪通知的闭环已经完整。但异步运行时还需要处理另一类「就绪」——时间。下一章我们将剖析 tokio::time::sleep 与 timeout 的实现：定时器如何被插入时间轮、时间轮如何按到期时间分级、driver 如何计算下一次 park 的超时并触发到期任务。你会看到「时间也是一种 I/O 事件」这一统一抽象，以及 start_paused 与 test clock 如何让时间在测试中可控。
