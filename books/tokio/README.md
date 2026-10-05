# Tokio Internals: From Future to Production Asynchronous Runtime

[![GitHub stars](https://img.shields.io/badge/GitHub-tokio-rs%2Ftokio-blue?logo=github)](https://github.com/tokio-rs/tokio)
[![Stars](https://img.shields.io/badge/Stars-28.5k-yellow)](#)
[![Chapters](https://img.shields.io/badge/Chapters-14-emerald)](#)
[![Language](https://img.shields.io/badge/Language-Rust-purple)](#)
[![Interactive Reader](https://img.shields.io/badge/Web%20Reader-aireadcode.com-cyan)](https://aireadcode.com/books/tokio.html)

> A systemic architectural breakdown of Tokio, tracing the complete lifecycle of asynchronous tasks: Future/Waker cooperative polling, work-stealing scheduler, I/O reactor (epoll), and hierarchical timing wheels.

---

## 🌐 Language Navigation
- **English Edition (Current)**: Table of Contents below.
- **[中文版 (Chinese Edition)](README_zh.md)**: 访问全书中文目录与章节内容。
- **[Interactive Live Reader](https://aireadcode.com/books/tokio.html)**: Read with dual-pane real-time code inspector and `[FACT]` verification anchors.

---

## 📚 Table of Contents

| Chapter | Title | Markdown Link |
| :---: | :--- | :---: |
| **01** | Chapter 01: Async Philosophy & Core Mental Model: Future, Waker & Cooperative Scheduling | [Read Online](en/01-async-philosophy-future-waker.md) |
| **02** | Chapter 02: Runtime Assembly: How Builder Composes Drivers & Thread Pools | [Read Online](en/02-runtime-assembly-and-builder.md) |
| **03** | Chapter 03: Task Genesis: How spawn Converts a Future into a Schedulable Unit | [Read Online](en/03-task-spawn-and-lifecycle.md) |
| **04** | Chapter 04: The Scheduler Heartbeat: Poll Loops & Work-Stealing Mechanics | [Read Online](en/04-scheduler-poll-and-work-stealing.md) |
| **05** | Chapter 05: I/O Driver & Readiness: Transforming epoll Events into Waker Notifications | [Read Online](en/05-io-driver-reactor-epoll.md) |
| **06** | Chapter 06: Timer Wheel Driver: Hierarchical Timing Wheels for Ultra-Low Latency | [Read Online](en/06-time-wheel-driver.md) |
| **07** | Chapter 07: Synchronization Primitives: Deep Dive into tokio::sync (Mutex, Notify, mpsc) | [Read Online](en/07-sync-primitives.md) |
| **08** | Chapter 08: Blocking Thread Pool: spawn_blocking & Worker Thread Isolation | [Read Online](en/08-blocking-thread-pool.md) |
| **09** | Chapter 09: Cooperative Scheduling Budget: How the coop Mechanism Prevents Starvation | [Read Online](en/09-cooperative-budget-scheduler.md) |
| **10** | Chapter 10: Backpressure & Flow Control: Channel Buffer Management & Stream Processing | [Read Online](en/10-backpressure-and-channels.md) |
| **11** | Chapter 11: Graceful Shutdown & Cancellation Safety: Lifecycle Management in Practice | [Read Online](en/11-graceful-shutdown-and-cancellation.md) |
| **12** | Chapter 12: Production Observability & Diagnostics: tokio-console & Distributed Tracing | [Read Online](en/12-observability-console-tracing.md) |
| **13** | Chapter 13: Performance Tuning & Concurrency Pitfalls: Context Switching, Allocations & Deadlocks | [Read Online](en/13-performance-tuning-and-pitfalls.md) |
| **14** | Chapter 14: Architectural Evolution & Trade-offs: From Micro-Kernel to Industrial Runtime | [Read Online](en/14-runtime-evolution-and-tradeoffs.md) |

---

## 🔍 How to Read
1. **GitHub Markdown Reader**: Click any chapter link above to browse full source code walkthroughs directly in GitHub.
2. **Interactive Dual-Pane Reader**: Visit [aireadcode.com/books/tokio.html](https://aireadcode.com/books/tokio.html) to view the synchronized code inspector.
3. **Desktop App**: Open your own local clone with the [AiReadCode Desktop Client](https://aireadcode.com/#downloads) to generate architecture books for any repository.

## 📄 License
This work is released under **CC BY-NC 4.0** for open technical education. Code snippets follow the upstream repository's open-source license (tokio-rs/tokio).
