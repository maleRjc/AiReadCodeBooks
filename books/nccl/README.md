# NCCL Deep Dive: The GPU Journey of an AllReduce Collective

[![GitHub stars](https://img.shields.io/badge/GitHub-NVIDIA%2Fnccl-blue?logo=github)](https://github.com/NVIDIA/nccl)
[![Stars](https://img.shields.io/badge/Stars-6.2k-yellow)](#)
[![Chapters](https://img.shields.io/badge/Chapters-25-emerald)](#)
[![Language](https://img.shields.io/badge/Language-C++%2FCUDA-purple)](#)
[![Interactive Reader](https://img.shields.io/badge/Web%20Reader-aireadcode.com-cyan)](https://malerjc.github.io/AiReadCodeBooks/books/nccl/#ch-01)

> A 25-chapter tour-de-force of NVIDIA NCCL: from topology graph discovery and algorithm tuning to multi-channel kernel scheduling, device-side LL/LL128/Simple protocols, and InfiniBand GPUDirect RDMA.

---

## 🌐 Language Navigation
- **English Edition (Current)**: Table of Contents below.
- **[中文版 (Chinese Edition)](README_zh.md)**: 访问全书中文目录与章节内容。
- **[Interactive Live Reader](https://malerjc.github.io/AiReadCodeBooks/books/nccl/#ch-01)**: Read with dual-pane real-time code inspector and `[FACT]` verification anchors.

---

## 📚 Table of Contents

| Chapter | Title | Markdown Link |
| :---: | :--- | :---: |
| **01** | Chapter 01: Execution & Phenomenon: External Behavior of an AllReduce | [Read Online](en/01-allreduce-external-behavior.md) |
| **02** | Chapter 02: Core Abstraction Model: Collectives, Topology, Algorithms & Transports | [Read Online](en/02-core-abstractions-topology-transport.md) |
| **03** | Chapter 03: Initialization: How ncclCommInitRank Forms a Communicator Domain | [Read Online](en/03-ncclcomminitrank-initialization.md) |
| **04** | Chapter 04: Topology Discovery & Graph Search: Mapping Multi-GPU Interconnects | [Read Online](en/04-topology-discovery-and-graph-search.md) |
| **05** | Chapter 05: Tuning & Protocol Selection: Deciding Optimal Paths and Channels | [Read Online](en/05-tuning-algorithm-and-protocol-selection.md) |
| **06** | Chapter 06: Collective Dispatch: Converting ncclAllReduce into Executable Tasks | [Read Online](en/06-ncclallreduce-task-dispatch.md) |
| **07** | Chapter 07: Task Scheduler: Multi-Channel & Kernel Execution Orchestration | [Read Online](en/07-task-scheduler-and-channels.md) |
| **08** | Chapter 08: Kernel Launch & Device Execution: From Host Dispatch to GPU Warps | [Read Online](en/08-kernel-launch-and-gpu-execution.md) |
| **09** | Chapter 09: Device Primitives: Data Transport in LL, LL128 & Simple Protocols | [Read Online](en/09-device-primitives-ll-ll128-simple.md) |
| **10** | Chapter 10: Collective Kernels: Device Implementations of AllReduce & AllGather | [Read Online](en/10-collective-kernels-allreduce-allgather.md) |
| **11** | Chapter 11: Transport Abstraction: Unifying P2P, SHM, NET & NVLS Interfaces | [Read Online](en/11-transport-layer-p2p-shm-net-nvls.md) |
| **12** | Chapter 12: Proxy Threads: Decoupling Asynchronous Network I/O from GPU Kernels | [Read Online](en/12-proxy-threads-async-io-scheduler.md) |
| **13** | Chapter 13: InfiniBand Networking: verbs Wrappers & GPUDirect RDMA Architecture | [Read Online](en/13-infiniband-net-ib-verbs-gpudirect-rdma.md) |
| **14** | Chapter 14: Symmetric Memory & NVLS: Hardware Multicast & LSA Direct Addressing | [Read Online](en/14-symmetric-memory-and-nvls-multicast.md) |
| **15** | Chapter 15: RMA & GIN: Remote Memory Access & GPU-Direct Interconnect Evolution | [Read Online](en/15-rma-and-gin-remote-gpu-communication.md) |
| **16** | Chapter 16: Plugin Ecosystem: Tuner Hooks, Profiler Interfaces & Environment Tuning | [Read Online](en/16-plugin-ecosystem-and-env-variables.md) |
| **17** | Chapter 17: RAS & Fault Tolerance: Link Failure Detection & Graceful Degradation | [Read Online](en/17-ras-fault-tolerance-and-degradation.md) |
| **18** | Chapter 18: Memory Allocator & Registration Cache: Host-to-Device Memory Optimization | [Read Online](en/18-memory-allocator-and-registration-cache.md) |
| **19** | Chapter 19: Device Communicator & ABI Compatibility: devcomm Structure & Kernel Contracts | [Read Online](en/19-device-communication-abi-devcomm.md) |
| **20** | Chapter 20: Device Native APIs & Kernel Fusion: Collective Fusion in Custom Kernels | [Read Online](en/20-device-api-and-kernel-fusion.md) |
| **21** | Chapter 21: Performance Tuning in Practice: nccl-tests, Benchmarks & Methodology | [Read Online](en/21-performance-tuning-and-benchmarking.md) |
| **22** | Chapter 22: Production Troubleshooting: Deadlocks, Timeouts & Diagnosis Workflows | [Read Online](en/22-production-troubleshooting-and-pitfalls.md) |
| **23** | Chapter 23: Ecosystem Extensions: nccl4py, nccl4rust, nccl_ep & nccl_ubx Bindings | [Read Online](en/23-ecosystem-nccl4py-nccl4rust-nccl-ep.md) |
| **24** | Chapter 24: Architectural Evolution: From Static Communication to Programmable Fabric | [Read Online](en/24-architecture-evolution-programmable-comms.md) |
| **25** | Chapter 25: Architectural Retrospective: The Complete Journey & Essence of AllReduce | [Read Online](en/25-retrospective-allreduce-complete-journey.md) |

---

## 🔍 How to Read
1. **GitHub Markdown Reader**: Click any chapter link above to browse full source code walkthroughs directly in GitHub.
2. **Interactive Dual-Pane Reader**: Visit [malerjc.github.io/AiReadCodeBooks/books/nccl/#ch-01](https://malerjc.github.io/AiReadCodeBooks/books/nccl/#ch-01) to view the synchronized code inspector.
3. **Desktop App**: Open your own local clone with the [AiReadCode Desktop Client](https://aireadcode.com/#downloads) to generate architecture books for any repository.

## 📄 License
This work is released under **CC BY-NC 4.0** for open technical education. Code snippets follow the upstream repository's open-source license (NVIDIA/nccl).
