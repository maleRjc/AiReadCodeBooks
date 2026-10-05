# NCCL 源码解读：一个 AllReduce 的 GPU 旅程

[![GitHub stars](https://img.shields.io/badge/GitHub-NVIDIA%2Fnccl-blue?logo=github)](https://github.com/NVIDIA/nccl)
[![Stars](https://img.shields.io/badge/Stars-6.2k-yellow)](#)
[![Chapters](https://img.shields.io/badge/章节数-25-emerald)](#)
[![Language](https://img.shields.io/badge/技术栈-C++%2FCUDA-purple)](#)
[![在线交互阅读器](https://img.shields.io/badge/在线阅读器-aireadcode.com-cyan)](https://aireadcode.com/books/nccl.html)

> 全书 25 章系统拆解 NVIDIA NCCL 通信库内核：从初始化拓扑图搜索、算法协议选型，到多 channel 任务调度、CUDA Kernel 启动、LL/LL128/Simple 数据搬运与 InfiniBand GPUDirect RDMA。

---

## 🌐 多语言导航
- **[English Edition](README.md)**: View English chapter index and translations.
- **中文原著 (当前)**: 完整 14~25 章精读目录见下方列表。
- **[网页端双栏交互精读器](https://aireadcode.com/books/nccl.html)**: 支持实时代码切片联动与 `[FACT]` 行号溯源验证。

---

## 📚 专著目录导读

| 章节 | 章节名称 | 在线精读入口 |
| :---: | :--- | :---: |
| **01** | 第 1 章：运行与现象：从一个 AllReduce 开始看外部行为 | [立即阅读](zh/01-allreduce-external-behavior.md) |
| **02** | 第 2 章：核心抽象模型：通信算子、拓扑、算法、协议与传输层 | [立即阅读](zh/02-core-abstractions-topology-transport.md) |
| **03** | 第 3 章：初始化入局：ncclCommInitRank 如何把一群孤立进程建立成通信域 | [立即阅读](zh/03-ncclcomminitrank-initialization.md) |
| **04** | 第 4 章：拓扑发现与图搜索：NCCL 如何“看清”多 GPU 系统的物理互联 | [立即阅读](zh/04-topology-discovery-and-graph-search.md) |
| **05** | 第 5 章：算法与协议选型：tuning 模块如何决定通信路径 | [立即阅读](zh/05-tuning-algorithm-and-protocol-selection.md) |
| **06** | 第 6 章：算子下发全景：ncclAllReduce 如何变成一个可执行的 kernel 任务 | [立即阅读](zh/06-ncclallreduce-task-dispatch.md) |
| **07** | 第 7 章：任务调度器：task_sched 如何编排多 channel 与 kernel 的执行顺序 | [立即阅读](zh/07-task-scheduler-and-channels.md) |
| **08** | 第 8 章：Kernel 启动与设备端执行：从 host 侧调用到 GPU 线程块起跑 | [立即阅读](zh/08-kernel-launch-and-gpu-execution.md) |
| **09** | 第 9 章：设备端通信原语：LL、LL128、Simple 三种协议的数据搬运实现 | [立即阅读](zh/09-device-primitives-ll-ll128-simple.md) |
| **10** | 第 10 章：集体通信算法内核：AllReduce、AllGather、ReduceScatter 的设备端实现 | [立即阅读](zh/10-collective-kernels-allreduce-allgather.md) |
| **11** | 第 11 章：传输层抽象：P2P、SHM、NET、NVLS 如何统一在同一套接口下 | [立即阅读](zh/11-transport-layer-p2p-shm-net-nvls.md) |
| **12** | 第 12 章：代理线程异步调度：proxy.cc 如何解耦 I/O 与 kernel 执行 | [立即阅读](zh/12-proxy-threads-async-io-scheduler.md) |
| **13** | 第 13 章：InfiniBand 网络传输：net_ib 如何封装 verbs 与 GPUDirect RDMA | [立即阅读](zh/13-infiniband-net-ib-verbs-gpudirect-rdma.md) |
| **14** | 第 14 章：对称内存与 NVLS：多播加速与 LSA 设备端直接寻址 | [立即阅读](zh/14-symmetric-memory-and-nvls-multicast.md) |
| **15** | 第 15 章：RMA 与 GIN：远端内存访问与 GPU 直连通信的演进 | [立即阅读](zh/15-rma-and-gin-remote-gpu-communication.md) |
| **16** | 第 16 章：插件生态与环境变量：net、tuner、profiler、env 如何扩展 NCCL 行为 | [立即阅读](zh/16-plugin-ecosystem-and-env-variables.md) |
| **17** | 第 17 章：RAS 机制与容错：链路故障检测、心跳与优雅降级 | [立即阅读](zh/17-ras-fault-tolerance-and-degradation.md) |
| **18** | 第 18 章：内存分配与显存管理：allocator、注册缓存与用户注册内存优化 | [立即阅读](zh/18-memory-allocator-and-registration-cache.md) |
| **19** | 第 19 章：设备端通信域与 ABI 兼容：devcomm 与 kernel 的通信契约 | [立即阅读](zh/19-device-communication-abi-devcomm.md) |
| **20** | 第 20 章：设备端原生 API 与算子融合：nccl_device 与 kernel fusion 实践 | [立即阅读](zh/20-device-api-and-kernel-fusion.md) |
| **21** | 第 21 章：性能调优实战：tuning 实操、benchmark 工具与调优方法论 | [立即阅读](zh/21-performance-tuning-and-benchmarking.md) |
| **22** | 第 22 章：生产排障与踩坑：常见死锁、超时、版本不匹配与排查方案 | [立即阅读](zh/22-production-troubleshooting-and-pitfalls.md) |
| **23** | 第 23 章：生态扩展：nccl4py、nccl4rust、nccl_ep、nccl_ubx 等周边项目 | [立即阅读](zh/23-ecosystem-nccl4py-nccl4rust-nccl-ep.md) |
| **24** | 第 24 章：架构演进与未来方向：从静态通信到可编程通信 | [立即阅读](zh/24-architecture-evolution-programmable-comms.md) |
| **25** | 第 25 章：全景回顾与思考：一个 AllReduce 的终极旅程与设计精髓 | [立即阅读](zh/25-retrospective-allreduce-complete-journey.md) |

---

## 🔍 阅读建议
1. **GitHub 沉浸精读**：点击上述章节链接，直接在 GitHub Markdown 阅读完整讲解、源码切片与设计推断。
2. **网页端真实源码联动**：访问 [aireadcode.com/books/nccl.html](https://aireadcode.com/books/nccl.html) 体验双栏联动与行号高亮。
3. **本地代码一键成书**：下载 [AiReadCode 桌面客户端](https://aireadcode.com/#downloads)，一键将任意复杂代码仓库扫描成书。

## 📄 版权与协议
本专著由 AiReadCode 源码编撰引擎自动化深度生成，遵循 **CC BY-NC 4.0** 开源知识共享协议。文中所引用源码归原开源项目所有。
