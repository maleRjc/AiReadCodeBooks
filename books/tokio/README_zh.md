# Tokio 异步底层机制与运行时剖析

[![GitHub stars](https://img.shields.io/badge/GitHub-tokio-rs%2Ftokio-blue?logo=github)](https://github.com/tokio-rs/tokio)
[![Stars](https://img.shields.io/badge/Stars-28.5k-yellow)](#)
[![Chapters](https://img.shields.io/badge/章节数-14-emerald)](#)
[![Language](https://img.shields.io/badge/技术栈-Rust-purple)](#)
[![在线交互阅读器](https://img.shields.io/badge/在线阅读器-aireadcode.com-cyan)](https://aireadcode.com/books/tokio.html)

> 本书以「一个异步任务从诞生到消亡」为主线，逐层拆解 Tokio 作为 Rust 异步运行时的核心设计：Future/Waker 协作式调度、工作窃取调度器、I/O Reactor 驱动与分级时间轮。

---

## 🌐 多语言导航
- **[English Edition](README.md)**: View English chapter index and translations.
- **中文原著 (当前)**: 完整 14~25 章精读目录见下方列表。
- **[网页端双栏交互精读器](https://aireadcode.com/books/tokio.html)**: 支持实时代码切片联动与 `[FACT]` 行号溯源验证。

---

## 📚 专著目录导读

| 章节 | 章节名称 | 在线精读入口 |
| :---: | :--- | :---: |
| **01** | 第 1 章：异步哲学与核心模型：Future、Waker 与协作式调度 | [立即阅读](zh/01-async-philosophy-future-waker.md) |
| **02** | 第 2 章：Runtime 装配工厂：Builder 如何把线程池与驱动拼装成一个运行时 | [立即阅读](zh/02-runtime-assembly-and-builder.md) |
| **03** | 第 3 章：一粒任务的诞生：spawn 如何把一个 Future 变成可调度实体 | [立即阅读](zh/03-task-spawn-and-lifecycle.md) |
| **04** | 第 4 章：调度循环的心跳：poll 循环与工作窃取 (Work-Stealing) 算法 | [立即阅读](zh/04-scheduler-poll-and-work-stealing.md) |
| **05** | 第 5 章：I/O 驱动与就绪通知：Reactor 如何把 epoll 事件转化为 Waker 唤醒 | [立即阅读](zh/05-io-driver-reactor-epoll.md) |
| **06** | 第 6 章：时间轮与高精度定时器：Time Driver 的分级时间轮实现 | [立即阅读](zh/06-time-wheel-driver.md) |
| **07** | 第 7 章：同步原语深度剖析：tokio::sync (Mutex, RwLock, Notify, mpsc) | [立即阅读](zh/07-sync-primitives.md) |
| **08** | 第 8 章：阻塞线程池与外部互操作：spawn_blocking 与任务隔离 | [立即阅读](zh/08-blocking-thread-pool.md) |
| **09** | 第 9 章：协作式调度预算：coop 机制如何防止异步任务饥饿 | [立即阅读](zh/09-cooperative-budget-scheduler.md) |
| **10** | 第 10 章：背压与流控机制：异步通道的缓冲区管理与流处理 | [立即阅读](zh/10-backpressure-and-channels.md) |
| **11** | 第 11 章：优雅停机与取消安全：Cancel Safety 与生命周期管理 | [立即阅读](zh/11-graceful-shutdown-and-cancellation.md) |
| **12** | 第 12 章：生产环境观测与排障：tokio-console 与 tracing 埋点实战 | [立即阅读](zh/12-observability-console-tracing.md) |
| **13** | 第 13 章：性能调优与高并发陷阱：上下文切换、内存分配与死锁排查 | [立即阅读](zh/13-performance-tuning-and-pitfalls.md) |
| **14** | 第 14 章：演进历程与架构沉思：Tokio 从微内核到工业级运行时的权衡 | [立即阅读](zh/14-runtime-evolution-and-tradeoffs.md) |

---

## 🔍 阅读建议
1. **GitHub 沉浸精读**：点击上述章节链接，直接在 GitHub Markdown 阅读完整讲解、源码切片与设计推断。
2. **网页端真实源码联动**：访问 [aireadcode.com/books/tokio.html](https://aireadcode.com/books/tokio.html) 体验双栏联动与行号高亮。
3. **本地代码一键成书**：下载 [AiReadCode 桌面客户端](https://aireadcode.com/#downloads)，一键将任意复杂代码仓库扫描成书。

## 📄 版权与协议
本专著由 AiReadCode 源码编撰引擎自动化深度生成，遵循 **CC BY-NC 4.0** 开源知识共享协议。文中所引用源码归原开源项目所有。
