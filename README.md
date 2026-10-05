# 📚 AiReadCodeBooks — The Multilingual Open-Source Architecture Library

<p align="center">
  <a href="https://aireadcode.com"><img src="https://img.shields.io/badge/Official%20Website-aireadcode.com-00e5ff?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Official Website"></a>
  <a href="#-flagship-available-monographs"><img src="https://img.shields.io/badge/Flagship%20Monographs-4%20Released-10b981?style=for-the-badge" alt="Flagship Monographs"></a>
  <a href="#-100-open-source-books-roadmap-matrix"><img src="https://img.shields.io/badge/Architecture%20Matrix-100%20Books-8b5cf6?style=for-the-badge" alt="100 Books Matrix"></a>
  <a href="#-multilingual-directory-standard"><img src="https://img.shields.io/badge/Multilingual-EN%20%7C%20ZH-f59e0b?style=for-the-badge" alt="Multilingual"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-ec4899?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <b>Transform complex, multi-million-line open-source codebases into clean, structured, and fact-verified technical textbooks.</b><br>
  Every monograph features end-to-end architecture diagrams, code walkthroughs, design inferences, and verbatim <code>[FACT:file:lines]</code> source verification.
</p>

---

## 🌐 Language Navigation / 多语言入口
- **English Edition (Current)**: Master documentation, 100-book matrix & English monographs.
- **[中文版入口 (Chinese Edition)](README_zh.md)**: 查看中文主目录、100本开源专著大纲矩阵与精读指引。
- **[Interactive Live Web Readers](https://aireadcode.com/books/)**: Experience real-time dual-pane reading with live source code synchronization.
- **[AiReadCode Desktop Client](https://aireadcode.com/#downloads)**: Turn your own repositories into structured architectural books with one click.

---

## 🌟 Flagship Available Monographs

The following 4 monographs are fully generated, audited, and ready to read online in both **English (`en/`)** and **Chinese (`zh/`)**:

| Flagship Book | Domain Track | Tech Stack | Stars | Chapters | Online Web Reader | GitHub Markdown |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **[Vue 3 Core Architecture](books/vue3/)** | Frontend & Web Runtimes | `TypeScript` | ★ 45.8k | 14 Chapters | [Interactive Reader](https://aireadcode.com/books/vue3.html) | [Read (EN)](books/vue3/en/01-monorepo-philosophy.md) · [阅读 (ZH)](books/vue3/zh/01-monorepo-philosophy.md) |
| **[Tokio Internals & Runtime](books/tokio/)** | Systems & High-Performance | `Rust` | ★ 28.5k | 14 Chapters | [Interactive Reader](https://aireadcode.com/books/tokio.html) | [Read (EN)](books/tokio/en/01-async-philosophy-future-waker.md) · [阅读 (ZH)](books/tokio/zh/01-async-philosophy-future-waker.md) |
| **[Inside vLLM Serving Engine](books/vllm/)** | AI Infra & Deep Learning | `Python/C++/CUDA` | ★ 35.2k | 14 Chapters | [Interactive Reader](https://aireadcode.com/books/vllm.html) | [Read (EN)](books/vllm/en/01-design-philosophy-and-architecture.md) · [阅读 (ZH)](books/vllm/zh/01-design-philosophy-and-architecture.md) |
| **[NCCL Deep Dive: AllReduce](books/nccl/)** | AI Infra & Deep Learning | `C++/CUDA` | ★ 6.2k | 25 Chapters | [Interactive Reader](https://aireadcode.com/books/nccl.html) | [Read (EN)](books/nccl/en/01-allreduce-external-behavior.md) · [阅读 (ZH)](books/nccl/zh/01-allreduce-external-behavior.md) |

---

## 🗺️ 100 Open-Source Books Roadmap Matrix

We are systematically releasing 100 complete monographs across 5 core technology domains. Each book undergoes full AST parsing, architectural chunking, deep inference synthesis, and verbatim source verification.

### Track 1: AI Infra & Deep Learning Systems (20 Books)
| # | Monograph Name | Repository | Tech Stack | Stars | Chapters | Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **vLLM**: Inside vLLM: High-Throughput LLM Serving Engine Architecture | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | `Python/C++/CUDA` | 35k+ | 14 | [✅ Available](books/vllm/) |
| 2 | **NCCL**: NCCL Deep Dive: The GPU Journey of an AllReduce Collective | [`NVIDIA/nccl`](https://github.com/NVIDIA/nccl) | `C++/CUDA` | 6k+ | 25 | [✅ Available](books/nccl/) |
| 3 | **llama.cpp**: llama.cpp Architecture: Pure C/C++ Edge LLM Inference Engine | [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp) | `C/C++` | 68k+ | 14 | ⏳ Queued |
| 4 | **Ollama**: Ollama Internals: Local LLM Runtime & Containerized Serving | [`ollama/ollama`](https://github.com/ollama/ollama) | `Go/C++` | 95k+ | 14 | ⏳ Queued |
| 5 | **DeepSpeed**: DeepSpeed: Memory Optimization & ZeRO Distributed Training | [`microsoft/DeepSpeed`](https://github.com/microsoft/DeepSpeed) | `Python/C++` | 35k+ | 16 | ⏳ Queued |
| 6 | **Megatron-LM**: Megatron-LM: 3D Parallelism for Large Scale Model Training | [`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM) | `Python/CUDA` | 12k+ | 16 | ⏳ Queued |
| 7 | **TensorRT-LLM**: TensorRT-LLM: Hardware Acceleration & Graph Fusion | [`NVIDIA/TensorRT-LLM`](https://github.com/NVIDIA/TensorRT-LLM) | `C++/Python` | 9k+ | 14 | ⏳ Queued |
| 8 | **SGLang**: SGLang: Structured Generation & RadixAttention Internals | [`sgl-project/sglang`](https://github.com/sgl-project/sglang) | `Python/C++` | 10k+ | 14 | ⏳ Queued |
| 9 | **LangChain**: LangChain Architecture: Agent Orchestration & LCEL Pipeline | [`langchain-ai/langchain`](https://github.com/langchain-ai/langchain) | `Python` | 95k+ | 14 | ⏳ Queued |
| 10 | **LlamaIndex**: LlamaIndex: Enterprise RAG Architecture & Multi-Stage Indexing | [`run-llama/llama_index`](https://github.com/run-llama/llama_index) | `Python` | 38k+ | 14 | ⏳ Queued |
| 11 | **Transformers**: Hugging Face Transformers: Model Abstractions & AutoModel Pipeline | [`huggingface/transformers`](https://github.com/huggingface/transformers) | `Python` | 135k+ | 16 | ⏳ Queued |
| 12 | **PyTorch Core**: PyTorch Core: Autograd Computation Graph & ATen Dispatcher | [`pytorch/pytorch`](https://github.com/pytorch/pytorch) | `C++/Python` | 85k+ | 20 | ⏳ Queued |
| 13 | **Triton**: OpenAI Triton: Compiler Architecture & GPU Kernel Generation | [`triton-lang/triton`](https://github.com/triton-lang/triton) | `Python/C++` | 13k+ | 14 | ⏳ Queued |
| 14 | **FlashAttention**: FlashAttention: Fast & Memory-Efficient Exact Attention with IO-Awareness | [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) | `CUDA/C++` | 16k+ | 12 | ⏳ Queued |
| 15 | **Axolotl**: Axolotl: Modular LLM Fine-Tuning Pipeline & Architecture | [`axolotl-ai-cloud/axolotl`](https://github.com/axolotl-ai-cloud/axolotl) | `Python` | 7k+ | 12 | ⏳ Queued |
| 16 | **FastChat**: FastChat: Distributed Model Serving & Chatbot Arena Pipeline | [`lm-sys/FastChat`](https://github.com/lm-sys/FastChat) | `Python` | 35k+ | 12 | ⏳ Queued |
| 17 | **TGI**: Hugging Face TGI: High-Performance Rust Web Server & LLM Serving | [`huggingface/text-generation-inference`](https://github.com/huggingface/text-generation-inference) | `Rust/Python` | 9k+ | 14 | ⏳ Queued |
| 18 | **AutoGPT**: AutoGPT: Autonomous Agent Execution Loop & Architecture | [`Significant-Gravitas/AutoGPT`](https://github.com/Significant-Gravitas/AutoGPT) | `Python` | 168k+ | 12 | ⏳ Queued |
| 19 | **CrewAI**: CrewAI: Multi-Agent Role Collaboration & Task Delegation System | [`crewAIInc/crewAI`](https://github.com/crewAIInc/crewAI) | `Python` | 25k+ | 12 | ⏳ Queued |
| 20 | **Dify**: Dify: LLM Workflow Orchestration & Hybrid Agent Platform | [`langgenius/dify`](https://github.com/langgenius/dify) | `Python/TypeScript` | 65k+ | 14 | ⏳ Queued |

### Track 2: Systems, High-Performance & Infrastructure (20 Books)
| # | Monograph Name | Repository | Tech Stack | Stars | Chapters | Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 21 | **Tokio**: Tokio Internals: From Future to Production Asynchronous Runtime | [`tokio-rs/tokio`](https://github.com/tokio-rs/tokio) | `Rust` | 28k+ | 14 | [✅ Available](books/tokio/) |
| 22 | **Redis**: Inside Redis: Event Loop, Memory Structures & Persistence Engine | [`redis/redis`](https://github.com/redis/redis) | `C` | 66k+ | 16 | ⏳ Queued |
| 23 | **Nginx**: Nginx Architecture: High-Performance Multi-Process Event Model | [`nginx/nginx`](https://github.com/nginx/nginx) | `C` | 24k+ | 16 | ⏳ Queued |
| 24 | **SQLite**: The Architecture of SQLite: B-Tree, VDBE & WAL Logging Engine | [`sqlite/sqlite`](https://github.com/sqlite/sqlite) | `C` | 6k+ | 16 | ⏳ Queued |
| 25 | **Linux Kernel**: Linux Kernel Internals: CFS Scheduler, VMM & VFS Architecture | [`torvalds/linux`](https://github.com/torvalds/linux) | `C` | 180k+ | 24 | ⏳ Queued |
| 26 | **RocksDB**: RocksDB: High-Performance LSM-Tree Storage Engine & Compaction | [`facebook/rocksdb`](https://github.com/facebook/rocksdb) | `C++` | 30k+ | 16 | ⏳ Queued |
| 27 | **ClickHouse**: ClickHouse: Vectorized Query Execution Engine & Columnar Storage | [`ClickHouse/ClickHouse`](https://github.com/ClickHouse/ClickHouse) | `C++` | 38k+ | 18 | ⏳ Queued |
| 28 | **TiDB**: TiDB Internals: SQL Parser, Cost-Based Optimizer & Distributed Transactions | [`pingcap/tidb`](https://github.com/pingcap/tidb) | `Go` | 37k+ | 16 | ⏳ Queued |
| 29 | **Envoy**: Envoy Proxy: High-Performance Event-Driven Architecture & xDS Control Plane | [`envoyproxy/envoy`](https://github.com/envoyproxy/envoy) | `C++` | 30k+ | 16 | ⏳ Queued |
| 30 | **BCC / eBPF**: eBPF & BCC: Kernel Tracing, Observability & Performance Tools | [`iovisor/bcc`](https://github.com/iovisor/bcc) | `C/Python/Lua` | 20k+ | 14 | ⏳ Queued |
| 31 | **DuckDB**: DuckDB Architecture: In-Process Analytical Engine & Vectorized Pipelines | [`duckdb/duckdb`](https://github.com/duckdb/duckdb) | `C++` | 26k+ | 14 | ⏳ Queued |
| 32 | **Memcached**: Memcached: Slab Allocator & Multi-Threaded Event Loop | [`memcached/memcached`](https://github.com/memcached/memcached) | `C` | 14k+ | 12 | ⏳ Queued |
| 33 | **etcd**: Inside etcd: Raft Consensus Protocol & MVCC Storage Engine | [`etcd-io/etcd`](https://github.com/etcd-io/etcd) | `Go` | 47k+ | 16 | ⏳ Queued |
| 34 | **Apache Arrow**: Apache Arrow: Columnar In-Memory Format & Zero-Copy Interoperability | [`apache/arrow`](https://github.com/apache/arrow) | `C++/Rust/Python` | 15k+ | 14 | ⏳ Queued |
| 35 | **DPDK**: DPDK: Kernel-Bypass Networking & Poll Mode Drivers | [`DPDK/dpdk`](https://github.com/DPDK/dpdk) | `C` | 3k+ | 14 | ⏳ Queued |
| 36 | **io_uring / liburing**: io_uring & liburing: Linux Asynchronous I/O Ring Buffers & Zero-Copy | [`axboe/liburing`](https://github.com/axboe/liburing) | `C` | 5k+ | 12 | ⏳ Queued |
| 37 | **Rust Standard Library**: The Rust Standard Library: Core Abstractions, Concurrency & Allocators | [`rust-lang/rust`](https://github.com/rust-lang/rust) | `Rust` | 98k+ | 18 | ⏳ Queued |
| 38 | **QEMU**: QEMU: Hardware Virtualization & Tiny Code Generator (TCG) | [`qemu/qemu`](https://github.com/qemu/qemu) | `C` | 11k+ | 16 | ⏳ Queued |
| 39 | **PostgreSQL**: PostgreSQL Internals: Query Executor, MVCC & Write-Ahead Logging | [`postgres/postgres`](https://github.com/postgres/postgres) | `C` | 16k+ | 20 | ⏳ Queued |
| 40 | **FreeBSD Kernel**: FreeBSD Kernel: Process Scheduling, Jails & Network Subsystems | [`freebsd/freebsd-src`](https://github.com/freebsd/freebsd-src) | `C` | 8k+ | 18 | ⏳ Queued |

### Track 3: Frontend, Fullstack & Web Runtimes (20 Books)
| # | Monograph Name | Repository | Tech Stack | Stars | Chapters | Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 41 | **Vue 3 Core**: Vue 3 Core Architecture: From Source Code to Full-Pipeline Engineering | [`vuejs/core`](https://github.com/vuejs/core) | `TypeScript` | 46k+ | 14 | [✅ Available](books/vue3/) |
| 42 | **React**: Inside React: Fiber Reconciliation, Lane Model & Concurrent Mode | [`facebook/react`](https://github.com/facebook/react) | `JavaScript/TypeScript` | 228k+ | 16 | ⏳ Queued |
| 43 | **Node.js**: Node.js Architecture: V8 Engine Integration, libuv Loop & Native Addons | [`nodejs/node`](https://github.com/nodejs/node) | `C++/JavaScript` | 106k+ | 18 | ⏳ Queued |
| 44 | **Deno**: Deno Internals: Rust-V8 Bridge, Secure Sandbox & Web Standards | [`denoland/deno`](https://github.com/denoland/deno) | `Rust/TypeScript` | 95k+ | 16 | ⏳ Queued |
| 45 | **Bun**: Bun Architecture: JavaScriptCore Integration & Zig System Call Pipeline | [`oven-sh/bun`](https://github.com/oven-sh/bun) | `Zig/C++` | 75k+ | 16 | ⏳ Queued |
| 46 | **Next.js**: Next.js Architecture: App Router, React Server Components & Turbopack | [`vercel/next.js`](https://github.com/vercel/next.js) | `JavaScript/Rust` | 125k+ | 16 | ⏳ Queued |
| 47 | **Vite**: Vite: No-Bundle Dev Server, ESM Transformation & Rolldown Bundler | [`vitejs/vite`](https://github.com/vitejs/vite) | `TypeScript/Rust` | 68k+ | 14 | ⏳ Queued |
| 48 | **Svelte**: Svelte 5 Internals: Compiler Architecture, Runes & Zero-Runtime Codegen | [`sveltejs/svelte`](https://github.com/sveltejs/svelte) | `TypeScript` | 80k+ | 14 | ⏳ Queued |
| 49 | **Angular**: Angular Architecture: Dependency Injection, Signals & Ivy Compiler | [`angular/angular`](https://github.com/angular/angular) | `TypeScript` | 95k+ | 16 | ⏳ Queued |
| 50 | **TypeScript Compiler**: Inside TypeScript: Parser, Type Checker & AST Transformation Engine | [`microsoft/TypeScript`](https://github.com/microsoft/TypeScript) | `TypeScript` | 100k+ | 18 | ⏳ Queued |
| 51 | **esbuild**: esbuild Architecture: Parallel AST Parsing, Linker & Minification in Go | [`evanw/esbuild`](https://github.com/evanw/esbuild) | `Go` | 38k+ | 14 | ⏳ Queued |
| 52 | **Turbopack**: Turbopack: Incremental Computation Engine & Rust Asset Bundling | [`vercel/turbo`](https://github.com/vercel/turbo) | `Rust` | 27k+ | 14 | ⏳ Queued |
| 53 | **Electron**: Electron Architecture: Chromium & Node.js Integration with IPC Pipelines | [`electron/electron`](https://github.com/electron/electron) | `C++/JavaScript` | 115k+ | 16 | ⏳ Queued |
| 54 | **Tauri**: Tauri Architecture: WRY Webview, Inter-Process Bridge & Security Model | [`tauri-apps/tauri`](https://github.com/tauri-apps/tauri) | `Rust/TypeScript` | 84k+ | 14 | ⏳ Queued |
| 55 | **Zustand**: Zustand Internals: Minimalist State Management & React Sync Subscriptions | [`pmndrs/zustand`](https://github.com/pmndrs/zustand) | `TypeScript` | 45k+ | 10 | ⏳ Queued |
| 56 | **TanStack Query**: TanStack Query: Async State Machines, Caching Strategies & GC Mechanics | [`TanStack/query`](https://github.com/TanStack/query) | `TypeScript` | 42k+ | 12 | ⏳ Queued |
| 57 | **Redux Toolkit**: Redux Toolkit: Opinionated Redux, Immer Immutability & RTK Query | [`reduxjs/redux-toolkit`](https://github.com/reduxjs/redux-toolkit) | `TypeScript` | 11k+ | 12 | ⏳ Queued |
| 58 | **Three.js**: Three.js Architecture: Scene Graph, Shaders & WebGPU Abstractions | [`mrdoob/three.js`](https://github.com/mrdoob/three.js) | `JavaScript` | 100k+ | 16 | ⏳ Queued |
| 59 | **Astro**: Astro Internals: Islands Architecture & Zero-JS Static Generation | [`withastro/astro`](https://github.com/withastro/astro) | `TypeScript` | 48k+ | 14 | ⏳ Queued |
| 60 | **Remix**: Remix Architecture: Nested Routing, Data Loaders & Web Standards Focus | [`remix-run/remix`](https://github.com/remix-run/remix) | `TypeScript` | 30k+ | 14 | ⏳ Queued |

### Track 4: Cloud Native, Microservices & Distributed Computing (20 Books)
| # | Monograph Name | Repository | Tech Stack | Stars | Chapters | Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 61 | **Kubernetes**: Kubernetes Architecture: Control Plane, Declarative Reconciliation & Kubelet | [`kubernetes/kubernetes`](https://github.com/kubernetes/kubernetes) | `Go` | 110k+ | 24 | ⏳ Queued |
| 62 | **Docker / Moby**: Docker & Moby Architecture: containerd, runc & Namespace Isolation | [`moby/moby`](https://github.com/moby/moby) | `Go` | 68k+ | 18 | ⏳ Queued |
| 63 | **Istio**: Istio Service Mesh: Pilot Control Plane & Envoy Sidecar Management | [`istio/istio`](https://github.com/istio/istio) | `Go` | 35k+ | 16 | ⏳ Queued |
| 64 | **Prometheus**: Prometheus Internals: TSDB Time-Series Engine & PromQL Parser | [`prometheus/prometheus`](https://github.com/prometheus/prometheus) | `Go` | 56k+ | 16 | ⏳ Queued |
| 65 | **OpenTelemetry Core**: OpenTelemetry Collector: Telemetry Pipelines, Receivers & Processors | [`open-telemetry/opentelemetry-collector`](https://github.com/open-telemetry/opentelemetry-collector) | `Go` | 5k+ | 14 | ⏳ Queued |
| 66 | **Cilium**: Cilium Architecture: eBPF Cloud-Native Networking & Security Policies | [`cilium/cilium`](https://github.com/cilium/cilium) | `Go/C` | 20k+ | 16 | ⏳ Queued |
| 67 | **Apache Kafka**: Apache Kafka Internals: Partition Replication, KRaft & Zero-Copy I/O | [`apache/kafka`](https://github.com/apache/kafka) | `Java/Scala` | 28k+ | 18 | ⏳ Queued |
| 68 | **Apache Spark**: Apache Spark: DAG Scheduler, Tungsten Engine & Catalyst Optimizer | [`apache/spark`](https://github.com/apache/spark) | `Scala/Java` | 38k+ | 20 | ⏳ Queued |
| 69 | **Apache Flink**: Apache Flink: Streaming State Machines, Checkpoints & Event-Time Engine | [`apache/flink`](https://github.com/apache/flink) | `Java/Scala` | 24k+ | 18 | ⏳ Queued |
| 70 | **Milvus**: Milvus Architecture: Vector Indexing Engines & Distributed Storage | [`milvus-io/milvus`](https://github.com/milvus-io/milvus) | `Go/C++` | 30k+ | 16 | ⏳ Queued |
| 71 | **Qdrant**: Qdrant Internals: HNSW Graphs & Filtered Vector Search in Rust | [`qdrant/qdrant`](https://github.com/qdrant/qdrant) | `Rust` | 20k+ | 14 | ⏳ Queued |
| 72 | **MinIO**: MinIO Architecture: Erasure Coding, S3 Compatibility & Object Storage | [`minio/minio`](https://github.com/minio/minio) | `Go` | 46k+ | 16 | ⏳ Queued |
| 73 | **Traefik**: Traefik: Edge Router Architecture & Dynamic Provider Discovery | [`traefik/traefik`](https://github.com/traefik/traefik) | `Go` | 50k+ | 14 | ⏳ Queued |
| 74 | **Helm**: Inside Helm: Chart Template Engine & Kubernetes Release Management | [`helm/helm`](https://github.com/helm/helm) | `Go` | 26k+ | 12 | ⏳ Queued |
| 75 | **Harbor**: Harbor Registry: Multi-Tenant Architecture & Image Replication Engine | [`goharbor/harbor`](https://github.com/goharbor/harbor) | `Go` | 24k+ | 14 | ⏳ Queued |
| 76 | **Consul**: Consul Architecture: Gossip Protocol, Serf & Multi-DC Service Discovery | [`hashicorp/consul`](https://github.com/hashicorp/consul) | `Go` | 28k+ | 14 | ⏳ Queued |
| 77 | **CoreDNS**: CoreDNS Architecture: Plugin-Driven DNS Server & K8s Resolution | [`coredns/coredns`](https://github.com/coredns/coredns) | `Go` | 12k+ | 12 | ⏳ Queued |
| 78 | **Apache Pulsar**: Apache Pulsar: Segment-Centric Architecture & BookKeeper Storage Engine | [`apache/pulsar`](https://github.com/apache/pulsar) | `Java` | 14k+ | 16 | ⏳ Queued |
| 79 | **Linkerd**: Linkerd Service Mesh: Micro-Proxy linkerd2-proxy Architecture in Rust | [`linkerd/linkerd2`](https://github.com/linkerd/linkerd2) | `Rust/Go` | 11k+ | 14 | ⏳ Queued |
| 80 | **Dapr**: Dapr Internals: Distributed Application Runtime Building Blocks | [`dapr/dapr`](https://github.com/dapr/dapr) | `Go` | 23k+ | 14 | ⏳ Queued |

### Track 5: Developer Tools, Compilers & Languages (20 Books)
| # | Monograph Name | Repository | Tech Stack | Stars | Chapters | Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 81 | **Git Core**: Git Internals: Content-Addressable Storage, Packfile & 3-Way Merge | [`git/git`](https://github.com/git/git) | `C` | 54k+ | 18 | ⏳ Queued |
| 82 | **VS Code**: VS Code Architecture: Extension Host Isolation, Monaco & LSP Pipeline | [`microsoft/vscode`](https://github.com/microsoft/vscode) | `TypeScript` | 162k+ | 20 | ⏳ Queued |
| 83 | **Rust Compiler**: Inside rustc: HIR, MIR Representations, Borrow Checker & LLVM Codegen | [`rust-lang/rust`](https://github.com/rust-lang/rust) | `Rust` | 98k+ | 20 | ⏳ Queued |
| 84 | **LLVM Core**: The Architecture of LLVM: IR Design, Optimization Passes & Target Backends | [`llvm/llvm-project`](https://github.com/llvm/llvm-project) | `C++` | 30k+ | 22 | ⏳ Queued |
| 85 | **CPython**: Inside CPython: Bytecode VM, GIL Concurrency & Tracing Garbage Collector | [`python/cpython`](https://github.com/python/cpython) | `C` | 63k+ | 18 | ⏳ Queued |
| 86 | **Go Runtime**: The Go Runtime: GMP Scheduler, Tri-Color Mark-Sweep GC & Preemption | [`golang/go`](https://github.com/golang/go) | `Go` | 125k+ | 18 | ⏳ Queued |
| 87 | **Neovim**: Neovim Architecture: MessagePack-RPC, Embedded Lua Engine & Async UI | [`neovim/neovim`](https://github.com/neovim/neovim) | `C/Lua` | 84k+ | 16 | ⏳ Queued |
| 88 | **Ripgrep**: Ripgrep Internals: Parallel Directory Walk, SIMD Acceleration & Regex Engine | [`BurntSushi/ripgrep`](https://github.com/BurntSushi/ripgrep) | `Rust` | 48k+ | 12 | ⏳ Queued |
| 89 | **Starship**: Starship Prompt: Parallel Module Rendering & Zero-Lag Architecture | [`starship/starship`](https://github.com/starship/starship) | `Rust` | 45k+ | 10 | ⏳ Queued |
| 90 | **Helix Editor**: Helix Editor Architecture: Tree-sitter Syntax Engine & Rope Buffer | [`helix-editor/helix`](https://github.com/helix-editor/helix) | `Rust` | 35k+ | 14 | ⏳ Queued |
| 91 | **Zig Compiler**: Zig Compiler Internals: comptime Execution & Self-Hosted Codegen | [`ziglang/zig`](https://github.com/ziglang/zig) | `Zig/C++` | 36k+ | 16 | ⏳ Queued |
| 92 | **Cargo**: Cargo Internals: PubGrub Dependency Resolution & Build Orchestration | [`rust-lang/cargo`](https://github.com/rust-lang/cargo) | `Rust` | 13k+ | 14 | ⏳ Queued |
| 93 | **Fish Shell**: Fish Shell Architecture: Interactive Autosuggestions & Rust Rewrite | [`fish-shell/fish-shell`](https://github.com/fish-shell/fish-shell) | `Rust` | 26k+ | 12 | ⏳ Queued |
| 94 | **Tmux**: Inside Tmux: Client-Server Architecture & Pseudo-Terminal (PTY) Management | [`tmux/tmux`](https://github.com/tmux/tmux) | `C` | 38k+ | 12 | ⏳ Queued |
| 95 | **Alacritty**: Alacritty Architecture: GPU-Accelerated OpenGL Text Rendering | [`alacritty/alacritty`](https://github.com/alacritty/alacritty) | `Rust` | 56k+ | 12 | ⏳ Queued |
| 96 | **fzf**: fzf Internals: Smith-Waterman Fuzzy Match Algorithm & Streaming Engine | [`junegunn/fzf`](https://github.com/junegunn/fzf) | `Go` | 65k+ | 10 | ⏳ Queued |
| 97 | **SWC**: SWC Architecture: High-Performance Rust JS/TS Transpiler & Minifier | [`swc-project/swc`](https://github.com/swc-project/swc) | `Rust` | 32k+ | 14 | ⏳ Queued |
| 98 | **Ruff**: Ruff Internals: Blazing Fast Python Linter & Formatter in Rust | [`astral-sh/ruff`](https://github.com/astral-sh/ruff) | `Rust` | 36k+ | 12 | ⏳ Queued |
| 99 | **Biome**: Biome Architecture: Resilient Parser, Linter & Formatter in Rust | [`biomejs/biome`](https://github.com/biomejs/biome) | `Rust` | 18k+ | 14 | ⏳ Queued |
| 100 | **Zed Editor**: Zed Editor Architecture: GPUI GPU Framework & CRDT Collaborative Engine | [`zed-industries/zed`](https://github.com/zed-industries/zed) | `Rust` | 50k+ | 16 | ⏳ Queued |

---

## 📁 Multilingual Directory Standard

Every book repository follows a strict, zero-ambiguity structure designed for automated updates and seamless multi-language navigation:

```text
AiReadCodeBooks/
├── README.md                      # English Master Catalog & 100-Book Matrix
├── README_zh.md                   # Chinese Master Catalog & 100-Book Matrix
├── LICENSE                        # CC BY-NC 4.0 License
├── doc/
│   └── AiReadCodeBooks_100本开源多语言专著生成与运营落地方案.md
├── scripts/
│   ├── book_matrix_100.json       # Structured 100-repository metadata registry
│   └── update_catalog.py          # Automatic README & matrix sync script
└── books/
    ├── <slug>/                    # e.g., vue3, tokio, vllm, nccl
    │   ├── README.md              # English book introduction & chapter index
    │   ├── README_zh.md           # Chinese book introduction & chapter index
    │   ├── meta.json              # Repository metadata, stars, chapters, tags
    │   ├── en/                    # Complete English chapters
    │   │   ├── 01-<slug>.md
    │   │   └── ...
    │   └── zh/                    # Complete Chinese chapters
    │       ├── 01-<slug>.md
    │       └── ...
```

---

## ⚡ Key Methodological Features

1. **FACT Line-Level Grounding**: Every code slice cited in the books is tagged with `[FACT:path/to/file:Lstart-Lend]`, ensuring verifiable correctness against upstream Git commits.
2. **Four-Layer Progressive Narrative**:
   - `Layer 1: Macro Mental Model` — Intuitive abstractions and system architecture maps.
   - `Layer 2: End-to-End Main Lifecycle` — Tracing a complete request, packet, or build step.
   - `Layer 3: Core Subsystems` — Deep dive into memory allocators, schedulers, and drivers.
   - `Layer 4: Trade-offs & Production Pitfalls` — Design inferences, deadlocks, and tuning benchmarks.
3. **Dual-Pane Interactive Experience**: Readers on [aireadcode.com/books](https://aireadcode.com/books/) can click any `[FACT]` badge to view full, syntax-highlighted source code in a synchronized side inspector.

---

## 🤝 Contributing & Requesting Books
- **Request a Monograph**: Open an issue titled `[Book Request] <Repo Name>` with your target GitHub URL.
- **Submit Translations & Fixes**: PRs improving technical explanations, fixing typos, or adding new language editions (`ja/`, `es/`, `ko/`) are warmly welcome.

---

## 📄 License
Text content and architectural diagrams are licensed under **Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](LICENSE))**.
Code snippets quoted in the texts belong to their respective original open-source authors under their project licenses.
