# 📚 AiReadCodeBooks — 开源架构百万行专著书库

<p align="center">
  <a href="https://aireadcode.com"><img src="https://img.shields.io/badge/官方网站-aireadcode.com-00e5ff?style=for-the-badge&logo=googlechrome&logoColor=white" alt="官方网站"></a>
  <a href="#-旗舰首发专著"><img src="https://img.shields.io/badge/首发专著-4%20本全集上线-10b981?style=for-the-badge" alt="首发专著"></a>
  <a href="#-100-本开源专著规划矩阵"><img src="https://img.shields.io/badge/架构矩阵-100%20本系统专著-8b5cf6?style=for-the-badge" alt="100本矩阵"></a>
  <a href="#-标准多语言目录规范"><img src="https://img.shields.io/badge/多语言支持-中文%20%7C%20English-f59e0b?style=for-the-badge" alt="多语言支持"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/开源协议-CC%20BY--NC%204.0-ec4899?style=for-the-badge" alt="开源协议"></a>
</p>

<p align="center">
  <b>把数十万乃至百万行的大型开源名作，编撰为条理清晰、深度可读、行号真实可查的标准技术专著。</b><br>
  每本专著均涵盖全景架构剖析、逐行精读、设计推断以及带行号溯源的 <code>[FACT:file:lines]</code> 真实代码锚点。
</p>

---

## 🌐 多语言与阅读入口
- **中文原著入口 (当前)**: 查看中文专著目录、100本开源专著大纲矩阵与精读导引。
- **[English Edition](README.md)**: Access English master catalog and translated monographs.
- **[网页端双栏交互阅读器](https://aireadcode.com/books/)**: 体验与官方源码库联动的双栏高亮代码检视器。
- **[AiReadCode 桌面客户端](https://aireadcode.com/#downloads)**: 无论是大型开源项目还是私有企业代码，一键即可扫描成书。

---

## 🌟 旗舰首发专著

以下 4 本深度专著已完整编撰完毕，并同时提供 **中文 (`zh/`)** 与 **英文 (`en/`)** 完整章节：

| 专著书名 | 赛道领域 | 技术栈 | GitHub Stars | 章节规模 | 在线交互阅读器 | GitHub Markdown 直达 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **[Vue 3 源码与工程化全景架构](books/vue3/)** | 前端底层与全栈运行时 | `TypeScript` | ★ 45.8k | 14 章系统专著 | [在线交互精读](https://aireadcode.com/books/vue3.html) | [中文章节入口](books/vue3/zh/01-monorepo-philosophy.md) · [English](books/vue3/en/01-monorepo-philosophy.md) |
| **[Tokio 异步底层机制与运行时剖析](books/tokio/)** | 系统级开发与高性能基础设施 | `Rust` | ★ 28.5k | 14 章系统专著 | [在线交互精读](https://aireadcode.com/books/tokio.html) | [中文章节入口](books/tokio/zh/01-async-philosophy-future-waker.md) · [English](books/tokio/en/01-async-philosophy-future-waker.md) |
| **[vLLM 高性能推理引擎核心实现](books/vllm/)** | AI Infra 与大模型系统工程 | `Python/C++/CUDA` | ★ 35.2k | 14 章系统专著 | [在线交互精读](https://aireadcode.com/books/vllm.html) | [中文章节入口](books/vllm/zh/01-design-philosophy-and-architecture.md) · [English](books/vllm/en/01-design-philosophy-and-architecture.md) |
| **[NCCL 源码解读：一个 AllReduce 的 GPU 旅程](books/nccl/)** | AI Infra 与大模型系统工程 | `C++/CUDA` | ★ 6.2k | 25 章系统专著 | [在线交互精读](https://aireadcode.com/books/nccl.html) | [中文章节入口](books/nccl/zh/01-allreduce-external-behavior.md) · [English](books/nccl/en/01-allreduce-external-behavior.md) |

---

## 🗺️ 100 本开源专著规划矩阵

本仓库系统性覆盖 5 大关键技术领域的 100 本优质开源专著，遵循标准化多语言结构逐步发布：

### 赛道 1：AI Infra 与大模型系统工程 (20 本)
| 序号 | 专著名称 | 对应开源仓库 | 核心技术栈 | Stars | 计划章节 | 发布状态 |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **vLLM**：vLLM 高性能推理引擎核心实现 | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | `Python/C++/CUDA` | 35k+ | 14 | [✅ 已上线](books/vllm/) |
| 2 | **NCCL**：NCCL 源码解读：一个 AllReduce 的 GPU 旅程 | [`NVIDIA/nccl`](https://github.com/NVIDIA/nccl) | `C++/CUDA` | 6k+ | 25 | [✅ 已上线](books/nccl/) |
| 3 | **llama.cpp**：llama.cpp 极简纯 C++ 边缘推理架构 | [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp) | `C/C++` | 68k+ | 14 | ⏳ 排期编撰中 |
| 4 | **Ollama**：Ollama 本地大模型运行时与容器化调度 | [`ollama/ollama`](https://github.com/ollama/ollama) | `Go/C++` | 95k+ | 14 | ⏳ 排期编撰中 |
| 5 | **DeepSpeed**：DeepSpeed 显存优化与 ZeRO 阶段全拆解 | [`microsoft/DeepSpeed`](https://github.com/microsoft/DeepSpeed) | `Python/C++` | 35k+ | 16 | ⏳ 排期编撰中 |
| 6 | **Megatron-LM**：Megatron-LM 3D 并行混合训练机制 | [`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM) | `Python/CUDA` | 12k+ | 16 | ⏳ 排期编撰中 |
| 7 | **TensorRT-LLM**：TensorRT-LLM 硬件极致加速与算子图融合 | [`NVIDIA/TensorRT-LLM`](https://github.com/NVIDIA/TensorRT-LLM) | `C++/Python` | 9k+ | 14 | ⏳ 排期编撰中 |
| 8 | **SGLang**：SGLang 结构化输出与 RadixAttention 缓存机制 | [`sgl-project/sglang`](https://github.com/sgl-project/sglang) | `Python/C++` | 10k+ | 14 | ⏳ 排期编撰中 |
| 9 | **LangChain**：LangChain 智能体编排与 LCEL 流水线设计 | [`langchain-ai/langchain`](https://github.com/langchain-ai/langchain) | `Python` | 95k+ | 14 | ⏳ 排期编撰中 |
| 10 | **LlamaIndex**：LlamaIndex 企业级 RAG 架构与多层索引系统 | [`run-llama/llama_index`](https://github.com/run-llama/llama_index) | `Python` | 38k+ | 14 | ⏳ 排期编撰中 |
| 11 | **Transformers**：Hugging Face Transformers 模型生态与 AutoModel 调度 | [`huggingface/transformers`](https://github.com/huggingface/transformers) | `Python` | 135k+ | 16 | ⏳ 排期编撰中 |
| 12 | **PyTorch Core**：PyTorch 核心机制：Autograd 自动求导与 ATen 算子库 | [`pytorch/pytorch`](https://github.com/pytorch/pytorch) | `C++/Python` | 85k+ | 20 | ⏳ 排期编撰中 |
| 13 | **Triton**：OpenAI Triton 编译器与 GPU 高性能编程模型 | [`triton-lang/triton`](https://github.com/triton-lang/triton) | `Python/C++` | 13k+ | 14 | ⏳ 排期编撰中 |
| 14 | **FlashAttention**：FlashAttention IO 感知与 GPU SRAM 平铺算法实现 | [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) | `CUDA/C++` | 16k+ | 12 | ⏳ 排期编撰中 |
| 15 | **Axolotl**：Axolotl 大模型微调编排框架全流程解析 | [`axolotl-ai-cloud/axolotl`](https://github.com/axolotl-ai-cloud/axolotl) | `Python` | 7k+ | 12 | ⏳ 排期编撰中 |
| 16 | **FastChat**：FastChat 多模型分布式评测与对话服务器 | [`lm-sys/FastChat`](https://github.com/lm-sys/FastChat) | `Python` | 35k+ | 12 | ⏳ 排期编撰中 |
| 17 | **TGI**：Hugging Face TGI 高并发 Rust 服务端架构 | [`huggingface/text-generation-inference`](https://github.com/huggingface/text-generation-inference) | `Rust/Python` | 9k+ | 14 | ⏳ 排期编撰中 |
| 18 | **AutoGPT**：AutoGPT 自主循环与多智能体认知架构 | [`Significant-Gravitas/AutoGPT`](https://github.com/Significant-Gravitas/AutoGPT) | `Python` | 168k+ | 12 | ⏳ 排期编撰中 |
| 19 | **CrewAI**：CrewAI 多智能体协作模型与任务委托机制 | [`crewAIInc/crewAI`](https://github.com/crewAIInc/crewAI) | `Python` | 25k+ | 12 | ⏳ 排期编撰中 |
| 20 | **Dify**：Dify 大模型工作流与全链路可视化调度引擎 | [`langgenius/dify`](https://github.com/langgenius/dify) | `Python/TypeScript` | 65k+ | 14 | ⏳ 排期编撰中 |

### 赛道 2：系统级开发与高性能基础设施 (20 本)
| 序号 | 专著名称 | 对应开源仓库 | 核心技术栈 | Stars | 计划章节 | 发布状态 |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 21 | **Tokio**：Tokio 异步底层机制与运行时剖析 | [`tokio-rs/tokio`](https://github.com/tokio-rs/tokio) | `Rust` | 28k+ | 14 | [✅ 已上线](books/tokio/) |
| 22 | **Redis**：Redis 源码剖析：事件循环、数据结构与持久化机制 | [`redis/redis`](https://github.com/redis/redis) | `C` | 66k+ | 16 | ⏳ 排期编撰中 |
| 23 | **Nginx**：Nginx 高性能异步多进程架构与模块机制 | [`nginx/nginx`](https://github.com/nginx/nginx) | `C` | 24k+ | 16 | ⏳ 排期编撰中 |
| 24 | **SQLite**：SQLite 嵌入式数据库核心：B-Tree、VDBE 与 WAL 机制 | [`sqlite/sqlite`](https://github.com/sqlite/sqlite) | `C` | 6k+ | 16 | ⏳ 排期编撰中 |
| 25 | **Linux Kernel**：Linux 内核精读：CFS 调度器、虚拟内存与 VFS 架构 | [`torvalds/linux`](https://github.com/torvalds/linux) | `C` | 180k+ | 24 | ⏳ 排期编撰中 |
| 26 | **RocksDB**：RocksDB LSM-Tree 存储引擎与 Compaction 机制 | [`facebook/rocksdb`](https://github.com/facebook/rocksdb) | `C++` | 30k+ | 16 | ⏳ 排期编撰中 |
| 27 | **ClickHouse**：ClickHouse 向量化执行引擎与列式存储机制 | [`ClickHouse/ClickHouse`](https://github.com/ClickHouse/ClickHouse) | `C++` | 38k+ | 18 | ⏳ 排期编撰中 |
| 28 | **TiDB**：TiDB 分布式数据库内核：SQL 解析、优化器与事务调度 | [`pingcap/tidb`](https://github.com/pingcap/tidb) | `Go` | 37k+ | 16 | ⏳ 排期编撰中 |
| 29 | **Envoy**：Envoy 服务代理架构与 xDS 动态配置机制 | [`envoyproxy/envoy`](https://github.com/envoyproxy/envoy) | `C++` | 30k+ | 16 | ⏳ 排期编撰中 |
| 30 | **BCC / eBPF**：eBPF/BCC 内核探测与可观测性系统实践 | [`iovisor/bcc`](https://github.com/iovisor/bcc) | `C/Python/Lua` | 20k+ | 14 | ⏳ 排期编撰中 |
| 31 | **DuckDB**：DuckDB 嵌入式分析型数据库与向量化流水线 | [`duckdb/duckdb`](https://github.com/duckdb/duckdb) | `C++` | 26k+ | 14 | ⏳ 排期编撰中 |
| 32 | **Memcached**：Memcached 内存分配 slab 机制与多线程事件循环 | [`memcached/memcached`](https://github.com/memcached/memcached) | `C` | 14k+ | 12 | ⏳ 排期编撰中 |
| 33 | **etcd**：etcd 分布式共识机制：Raft 算法与 MVCC 存储引擎 | [`etcd-io/etcd`](https://github.com/etcd-io/etcd) | `Go` | 47k+ | 16 | ⏳ 排期编撰中 |
| 34 | **Apache Arrow**：Apache Arrow 内存列式格式与跨语言零拷贝通信 | [`apache/arrow`](https://github.com/apache/arrow) | `C++/Rust/Python` | 15k+ | 14 | ⏳ 排期编撰中 |
| 35 | **DPDK**：DPDK 极速用户态网络栈与轮询驱动架构 | [`DPDK/dpdk`](https://github.com/DPDK/dpdk) | `C` | 3k+ | 14 | ⏳ 排期编撰中 |
| 36 | **io_uring / liburing**：io_uring Linux 原生异步 I/O 内核环形缓冲区 | [`axboe/liburing`](https://github.com/axboe/liburing) | `C` | 5k+ | 12 | ⏳ 排期编撰中 |
| 37 | **Rust Standard Library**：Rust 标准库底层剖析：所有权抽象、并发原语与 Allocator | [`rust-lang/rust`](https://github.com/rust-lang/rust) | `Rust` | 98k+ | 18 | ⏳ 排期编撰中 |
| 38 | **QEMU**：QEMU 硬件仿真与 TCG 动态二进制翻译机制 | [`qemu/qemu`](https://github.com/qemu/qemu) | `C` | 11k+ | 16 | ⏳ 排期编撰中 |
| 39 | **PostgreSQL**：PostgreSQL 内核：执行引擎、并发控制与 WAL 日志 | [`postgres/postgres`](https://github.com/postgres/postgres) | `C` | 16k+ | 20 | ⏳ 排期编撰中 |
| 40 | **FreeBSD Kernel**：FreeBSD 内核设计精要：UFS、Jails 隔离与网络栈 | [`freebsd/freebsd-src`](https://github.com/freebsd/freebsd-src) | `C` | 8k+ | 18 | ⏳ 排期编撰中 |

### 赛道 3：前端底层、框架生态与全栈运行时 (20 本)
| 序号 | 专著名称 | 对应开源仓库 | 核心技术栈 | Stars | 计划章节 | 发布状态 |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 41 | **Vue 3 Core**：Vue 3 源码与工程化全景架构 | [`vuejs/core`](https://github.com/vuejs/core) | `TypeScript` | 46k+ | 14 | [✅ 已上线](books/vue3/) |
| 42 | **React**：React 核心机制：Fiber 树调和、Lane 优先级与并发模式 | [`facebook/react`](https://github.com/facebook/react) | `JavaScript/TypeScript` | 228k+ | 16 | ⏳ 排期编撰中 |
| 43 | **Node.js**：Node.js 运行时：V8 引擎集成、libuv 事件循环与原生绑定 | [`nodejs/node`](https://github.com/nodejs/node) | `C++/JavaScript` | 106k+ | 18 | ⏳ 排期编撰中 |
| 44 | **Deno**：Deno 现代化运行时架构：V8 Rusty 桥接与安全沙箱 | [`denoland/deno`](https://github.com/denoland/deno) | `Rust/TypeScript` | 95k+ | 16 | ⏳ 排期编撰中 |
| 45 | **Bun**：Bun 极速全栈运行时：JavaScriptCore 与 Zig 原生系统调用 | [`oven-sh/bun`](https://github.com/oven-sh/bun) | `Zig/C++` | 75k+ | 16 | ⏳ 排期编撰中 |
| 46 | **Next.js**：Next.js 全栈框架：App Router、RSC 渲染与 Turbopack | [`vercel/next.js`](https://github.com/vercel/next.js) | `JavaScript/Rust` | 125k+ | 16 | ⏳ 排期编撰中 |
| 47 | **Vite**：Vite 下一代前端构建工具：原生 ESM 开发服务器与 Rolldown | [`vitejs/vite`](https://github.com/vitejs/vite) | `TypeScript/Rust` | 68k+ | 14 | ⏳ 排期编撰中 |
| 48 | **Svelte**：Svelte 5 编译器架构：Runes 反应性与零运行时代码生成 | [`sveltejs/svelte`](https://github.com/sveltejs/svelte) | `TypeScript` | 80k+ | 14 | ⏳ 排期编撰中 |
| 49 | **Angular**：Angular 企业级架构：依赖注入容器、Signals 与 Ivy 引擎 | [`angular/angular`](https://github.com/angular/angular) | `TypeScript` | 95k+ | 16 | ⏳ 排期编撰中 |
| 50 | **TypeScript Compiler**：TypeScript 编译器内核：类型推断、Checker 与 AST 转换 | [`microsoft/TypeScript`](https://github.com/microsoft/TypeScript) | `TypeScript` | 100k+ | 18 | ⏳ 排期编撰中 |
| 51 | **esbuild**：esbuild 极速打包器：并行 AST 解析、链接器与代码压缩 | [`evanw/esbuild`](https://github.com/evanw/esbuild) | `Go` | 38k+ | 14 | ⏳ 排期编撰中 |
| 52 | **Turbopack**：Turbopack 增量计算计算图与 Rust 打包引擎 | [`vercel/turbo`](https://github.com/vercel/turbo) | `Rust` | 27k+ | 14 | ⏳ 排期编撰中 |
| 53 | **Electron**：Electron 桌面框架：Chromium 与 Node.js 跨进程多线程通信 | [`electron/electron`](https://github.com/electron/electron) | `C++/JavaScript` | 115k+ | 16 | ⏳ 排期编撰中 |
| 54 | **Tauri**：Tauri 轻量桌面应用内核：WRY 网页视图与安全隔离机制 | [`tauri-apps/tauri`](https://github.com/tauri-apps/tauri) | `Rust/TypeScript` | 84k+ | 14 | ⏳ 排期编撰中 |
| 55 | **Zustand**：Zustand 极简状态管理哲学与 React 订阅闭环 | [`pmndrs/zustand`](https://github.com/pmndrs/zustand) | `TypeScript` | 45k+ | 10 | ⏳ 排期编撰中 |
| 56 | **TanStack Query**：TanStack Query 异步状态机与客户端缓存设计 | [`TanStack/query`](https://github.com/TanStack/query) | `TypeScript` | 42k+ | 12 | ⏳ 排期编撰中 |
| 57 | **Redux Toolkit**：Redux Toolkit 规范化状态流与 Immer 不变性实践 | [`reduxjs/redux-toolkit`](https://github.com/reduxjs/redux-toolkit) | `TypeScript` | 11k+ | 12 | ⏳ 排期编撰中 |
| 58 | **Three.js**：Three.js 3D 渲染引擎：场景图、着色器与 WebGL/WebGPU 抽象 | [`mrdoob/three.js`](https://github.com/mrdoob/three.js) | `JavaScript` | 100k+ | 16 | ⏳ 排期编撰中 |
| 59 | **Astro**：Astro 群岛架构 (Islands Architecture) 与零 JS 静态优先 | [`withastro/astro`](https://github.com/withastro/astro) | `TypeScript` | 48k+ | 14 | ⏳ 排期编撰中 |
| 60 | **Remix**：Remix 全栈 Web 框架：嵌套路由、数据加载器与 Web 标准优先 | [`remix-run/remix`](https://github.com/remix-run/remix) | `TypeScript` | 30k+ | 14 | ⏳ 排期编撰中 |

### 赛道 4：云原生、微服务与分布式计算系统 (20 本)
| 序号 | 专著名称 | 对应开源仓库 | 核心技术栈 | Stars | 计划章节 | 发布状态 |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 61 | **Kubernetes**：Kubernetes 控制面设计：API Server、Scheduler 与 Controller Manager | [`kubernetes/kubernetes`](https://github.com/kubernetes/kubernetes) | `Go` | 110k+ | 24 | ⏳ 排期编撰中 |
| 62 | **Docker / Moby**：Docker 容器引擎核心：containerd 调度与 Linux Namespace 隔离 | [`moby/moby`](https://github.com/moby/moby) | `Go` | 68k+ | 18 | ⏳ 排期编撰中 |
| 63 | **Istio**：Istio 服务网格架构：Pilot 拓扑分发与 Envoy 数据面调度 | [`istio/istio`](https://github.com/istio/istio) | `Go` | 35k+ | 16 | ⏳ 排期编撰中 |
| 64 | **Prometheus**：Prometheus 监控引擎：TSDB 时序数据库与 PromQL 解析器 | [`prometheus/prometheus`](https://github.com/prometheus/prometheus) | `Go` | 56k+ | 16 | ⏳ 排期编撰中 |
| 65 | **OpenTelemetry Core**：OpenTelemetry Collector 遥测数据管道与流水线处理器 | [`open-telemetry/opentelemetry-collector`](https://github.com/open-telemetry/opentelemetry-collector) | `Go` | 5k+ | 14 | ⏳ 排期编撰中 |
| 66 | **Cilium**：Cilium eBPF 云原生网络与网络安全策略引擎 | [`cilium/cilium`](https://github.com/cilium/cilium) | `Go/C` | 20k+ | 16 | ⏳ 排期编撰中 |
| 67 | **Apache Kafka**：Apache Kafka 分布式日志流：Partition 副本、KRaft 与零拷贝 I/O | [`apache/kafka`](https://github.com/apache/kafka) | `Java/Scala` | 28k+ | 18 | ⏳ 排期编撰中 |
| 68 | **Apache Spark**：Apache Spark 大数据计算引擎：DAG 调度、Tungsten 与 Catalyst 优化器 | [`apache/spark`](https://github.com/apache/spark) | `Scala/Java` | 38k+ | 20 | ⏳ 排期编撰中 |
| 69 | **Apache Flink**：Apache Flink 流批一体引擎：Chandy-Lamport 状态快照与流式拓扑 | [`apache/flink`](https://github.com/apache/flink) | `Java/Scala` | 24k+ | 18 | ⏳ 排期编撰中 |
| 70 | **Milvus**：Milvus 向量数据库架构：向量索引、段合并与分布式执行流 | [`milvus-io/milvus`](https://github.com/milvus-io/milvus) | `Go/C++` | 30k+ | 16 | ⏳ 排期编撰中 |
| 71 | **Qdrant**：Qdrant Rust 向量搜索引擎：HNSW 层次图与负载过滤机制 | [`qdrant/qdrant`](https://github.com/qdrant/qdrant) | `Rust` | 20k+ | 14 | ⏳ 排期编撰中 |
| 72 | **MinIO**：MinIO 高性能对象存储：Erasure Coding 纠删码与 S3 兼容架构 | [`minio/minio`](https://github.com/minio/minio) | `Go` | 46k+ | 16 | ⏳ 排期编撰中 |
| 73 | **Traefik**：Traefik 云原生边缘路由：动态服务发现与中间件链 | [`traefik/traefik`](https://github.com/traefik/traefik) | `Go` | 50k+ | 14 | ⏳ 排期编撰中 |
| 74 | **Helm**：Helm 包管理器内核：Chart 渲染引擎与 Kubernetes 资源生命周期 | [`helm/helm`](https://github.com/helm/helm) | `Go` | 26k+ | 12 | ⏳ 排期编撰中 |
| 75 | **Harbor**：Harbor 企业级制品库：多租户隔离、镜像复制与安全扫描 | [`goharbor/harbor`](https://github.com/goharbor/harbor) | `Go` | 24k+ | 14 | ⏳ 排期编撰中 |
| 76 | **Consul**：Consul 服务发现与配置共享：Gossip 协议与跨数据中心同步 | [`hashicorp/consul`](https://github.com/hashicorp/consul) | `Go` | 28k+ | 14 | ⏳ 排期编撰中 |
| 77 | **CoreDNS**：CoreDNS 插件化 DNS 服务器架构与 Kubernetes 域名解析 | [`coredns/coredns`](https://github.com/coredns/coredns) | `Go` | 12k+ | 12 | ⏳ 排期编撰中 |
| 78 | **Apache Pulsar**：Apache Pulsar 计算存储分离架构：BookKeeper 与分层存储 | [`apache/pulsar`](https://github.com/apache/pulsar) | `Java` | 14k+ | 16 | ⏳ 排期编撰中 |
| 79 | **Linkerd**：Linkerd 超轻量服务网格：Rust 微代理 linkerd2-proxy 设计 | [`linkerd/linkerd2`](https://github.com/linkerd/linkerd2) | `Rust/Go` | 11k+ | 14 | ⏳ 排期编撰中 |
| 80 | **Dapr**：Dapr 分布式应用运行时：边车模型与通用状态/发布订阅构建块 | [`dapr/dapr`](https://github.com/dapr/dapr) | `Go` | 23k+ | 14 | ⏳ 排期编撰中 |

### 赛道 5：开发者工具链、编译器与编程语言运行时 (20 本)
| 序号 | 专著名称 | 对应开源仓库 | 核心技术栈 | Stars | 计划章节 | 发布状态 |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 81 | **Git Core**：Git 底层原理：对象存储模型、Packfile 与三方合并算法 | [`git/git`](https://github.com/git/git) | `C` | 54k+ | 18 | ⏳ 排期编撰中 |
| 82 | **VS Code**：VS Code 架构精要：插件进程隔离机制、Monaco 编辑器与 LSP 通信 | [`microsoft/vscode`](https://github.com/microsoft/vscode) | `TypeScript` | 162k+ | 20 | ⏳ 排期编撰中 |
| 83 | **Rust Compiler**：Rust 编译器架构：HIR、MIR 中间表示、Borrow Checker 与 LLVM 代码生成 | [`rust-lang/rust`](https://github.com/rust-lang/rust) | `Rust` | 98k+ | 20 | ⏳ 排期编撰中 |
| 84 | **LLVM Core**：LLVM 架构全景：LLVM IR 中间表示、Pass 优化流水线与后端目标生成 | [`llvm/llvm-project`](https://github.com/llvm/llvm-project) | `C++` | 30k+ | 22 | ⏳ 排期编撰中 |
| 85 | **CPython**：CPython 解释器内核：字节码虚拟机、GIL 锁与内存垃圾回收 | [`python/cpython`](https://github.com/python/cpython) | `C` | 63k+ | 18 | ⏳ 排期编撰中 |
| 86 | **Go Runtime**：Go 运行时深潜：GMP 调度器、三色标记清除 GC 与抢占机制 | [`golang/go`](https://github.com/golang/go) | `Go` | 125k+ | 18 | ⏳ 排期编撰中 |
| 87 | **Neovim**：Neovim 现代化编辑器架构：RPC 架构、Lua API 与异步事件循环 | [`neovim/neovim`](https://github.com/neovim/neovim) | `C/Lua` | 84k+ | 16 | ⏳ 排期编撰中 |
| 88 | **Ripgrep**：Ripgrep 极速文本搜索：并行目录遍历、SIMD 硬件加速与正则表达式引擎 | [`BurntSushi/ripgrep`](https://github.com/BurntSushi/ripgrep) | `Rust` | 48k+ | 12 | ⏳ 排期编撰中 |
| 89 | **Starship**：Starship 极速跨平台提示符：并行模块渲染与异步配置加载 | [`starship/starship`](https://github.com/starship/starship) | `Rust` | 45k+ | 10 | ⏳ 排期编撰中 |
| 90 | **Helix Editor**：Helix 模态编辑器内核：Tree-sitter 语法高亮与 Rope 文本数据结构 | [`helix-editor/helix`](https://github.com/helix-editor/helix) | `Rust` | 35k+ | 14 | ⏳ 排期编撰中 |
| 91 | **Zig Compiler**：Zig 编译器架构：comptime 编译期执行与直接自举代码生成 | [`ziglang/zig`](https://github.com/ziglang/zig) | `Zig/C++` | 36k+ | 16 | ⏳ 排期编撰中 |
| 92 | **Cargo**：Cargo 包管理工具：依赖解析 PubGrub 算法与并发编译编排 | [`rust-lang/cargo`](https://github.com/rust-lang/cargo) | `Rust` | 13k+ | 14 | ⏳ 排期编撰中 |
| 93 | **Fish Shell**：Fish Shell 交互式命令行：自动补全管道与 Rust 重构实践 | [`fish-shell/fish-shell`](https://github.com/fish-shell/fish-shell) | `Rust` | 26k+ | 12 | ⏳ 排期编撰中 |
| 94 | **Tmux**：Tmux 终端复用器：客户端-服务端架构与伪终端 PTY 管理 | [`tmux/tmux`](https://github.com/tmux/tmux) | `C` | 38k+ | 12 | ⏳ 排期编撰中 |
| 95 | **Alacritty**：Alacritty GPU 加速终端：OpenGL 文本渲染与低延迟输入循环 | [`alacritty/alacritty`](https://github.com/alacritty/alacritty) | `Rust` | 56k+ | 12 | ⏳ 排期编撰中 |
| 96 | **fzf**：fzf 交互式模糊查找器：Smith-Waterman 模糊匹配与并行流式处理 | [`junegunn/fzf`](https://github.com/junegunn/fzf) | `Go` | 65k+ | 10 | ⏳ 排期编撰中 |
| 97 | **SWC**：SWC 极速编译器：基于 Rust 的 JavaScript/TypeScript 转换器 | [`swc-project/swc`](https://github.com/swc-project/swc) | `Rust` | 32k+ | 14 | ⏳ 排期编撰中 |
| 98 | **Ruff**：Ruff 极速 Python 代码分析器：Rust 实现的并行 Linter 与 Formatter | [`astral-sh/ruff`](https://github.com/astral-sh/ruff) | `Rust` | 36k+ | 12 | ⏳ 排期编撰中 |
| 99 | **Biome**：Biome 一体化 Web 开发工具链：语法容错解析器与代码格式化引擎 | [`biomejs/biome`](https://github.com/biomejs/biome) | `Rust` | 18k+ | 14 | ⏳ 排期编撰中 |
| 100 | **Zed Editor**：Zed 高性能协作代码编辑器：GPUI 渲染框架与 CRDT 实时协作 | [`zed-industries/zed`](https://github.com/zed-industries/zed) | `Rust` | 50k+ | 16 | ⏳ 排期编撰中 |

---

## 📁 标准多语言目录规范

每个开源专著均遵循统一、严谨的目录结构，确保与自动化流水线完全兼容：

```text
AiReadCodeBooks/
├── README.md                      # 英文顶级导航与 100 本专著矩阵
├── README_zh.md                   # 中文顶级导航与 100 本专著矩阵
├── LICENSE                        # CC BY-NC 4.0 知识共享协议
├── doc/
│   └── AiReadCodeBooks_100本开源多语言专著生成与运营落地方案.md
├── scripts/
│   ├── book_matrix_100.json       # 100 本专著完整结构化元数据列表
│   └── update_catalog.py          # 自动化同步脚本
└── books/
    ├── <slug>/                    # 例如 vue3, tokio, vllm, nccl
    │   ├── README.md              # 英文版专著导读与章节目录
    │   ├── README_zh.md           # 中文版专著导读与章节目录
    │   ├── meta.json              # 仓库信息、Stars、技术栈、章节索引
    │   ├── zh/                    # 中文完整精读章节
    │   │   ├── 01-<slug>.md
    │   │   └── ...
    │   └── en/                    # 英文完整精读章节
    │       ├── 01-<slug>.md
    │       └── ...
```

---

## ⚡ 核心特色与编撰方法论

1. **FACT 行号级精准锚定**：所有代码引用均带有 `[FACT:path/to/file:Lstart-Lend]` 标签，可追溯到 Git Commit 真实代码行，杜绝大模型臆造。
2. **四层认知递进叙事**：
   - `第 1 层：宏观认知` — 建立直觉心智模型与架构全景图。
   - `第 2 层：端到端主干链路` — 串联一个请求、一个任务或一次构建的完整生命周期。
   - `第 3 层：核心子系统机制` — 深度剖析内存管理、调度器、协议驱动与状态机。
   - `第 4 层：架构权衡与避坑指南` — 探讨设计取舍、生产死锁排查与性能基准测试。
3. **沉浸式双栏交互体验**：在 [aireadcode.com/books](https://aireadcode.com/books) 网页端点击任何 `[FACT]` 药丸，右侧代码面板将自动载入全量源文件并滚动高亮对应切片。

---

## 🤝 专著许愿与参与贡献
- **提交新书许愿**：欢迎提交 Issue 标题为 `[Book Request] <仓库名称>`，我们将根据社区需求优先排期编撰。
- **提交改进与勘误**：欢迎提交 Pull Request 完善章节细节、修正笔误或补充多语言版本（如 `ja/`, `es/`, `ko/`）。

---

## 📄 开源许可协议
文字阐述、架构解析与图表采用 **署名-非商业性使用 4.0 国际许可协议 ([CC BY-NC 4.0](LICENSE))**。
书中所引用的各项目开源代码版权归各自原项目所有者所有。
