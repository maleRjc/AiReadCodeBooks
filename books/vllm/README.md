# Inside vLLM: High-Throughput LLM Serving Engine Architecture

[![GitHub stars](https://img.shields.io/badge/GitHub-vllm-project%2Fvllm-blue?logo=github)](https://github.com/vllm-project/vllm)
[![Stars](https://img.shields.io/badge/Stars-35.2k-yellow)](#)
[![Chapters](https://img.shields.io/badge/Chapters-14-emerald)](#)
[![Language](https://img.shields.io/badge/Language-Python%2FC++%2FCUDA-purple)](#)
[![Interactive Reader](https://img.shields.io/badge/Web%20Reader-aireadcode.com-cyan)](https://malerjc.github.io/AiReadCodeBooks/books/vllm/#ch-01)

> An in-depth architectural breakdown of vLLM, covering end-to-end request scheduling, PagedAttention memory virtualization, continuous batching, CUDA Graph acceleration, and speculative decoding.

---

## 🌐 Language Navigation
- **English Edition (Current)**: Table of Contents below.
- **[中文版 (Chinese Edition)](README_zh.md)**: 访问全书中文目录与章节内容。
- **[Interactive Live Reader](https://malerjc.github.io/AiReadCodeBooks/books/vllm/#ch-01)**: Read with dual-pane real-time code inspector and `[FACT]` verification anchors.

---

## 📚 Table of Contents

| Chapter | Title | Markdown Link |
| :---: | :--- | :---: |
| **01** | Chapter 01: vLLM Design Philosophy & High-Throughput Inference Architecture | [Read Online](en/01-design-philosophy-and-architecture.md) |
| **02** | Chapter 02: Core Abstractions & Data Structures: Request, Sequence & KV Cache | [Read Online](en/02-core-abstractions-request-sequence-kvcache.md) |
| **03** | Chapter 03: Request Lifecycle: End-to-End Flow from HTTP/CLI to EngineCore | [Read Online](en/03-request-lifecycle-http-to-enginecore.md) |
| **04** | Chapter 04: Memory Management & PagedAttention: KV Cache Virtualization | [Read Online](en/04-paged-attention-and-kv-cache-virtualization.md) |
| **05** | Chapter 05: Continuous Batching Engine: Iteration-Level Dynamic Scheduling | [Read Online](en/05-continuous-batching-and-scheduling.md) |
| **06** | Chapter 06: Model Execution & Distributed Parallelism: ModelRunner, Worker & Tensor Parallel | [Read Online](en/06-model-runner-and-distributed-parallelism.md) |
| **07** | Chapter 07: CUDA Graph & Execution Acceleration: Static Graph Capture & Low Latency | [Read Online](en/07-cuda-graph-and-execution-acceleration.md) |
| **08** | Chapter 08: Distributed KV Cache & Chunked Prefill: Prefix Caching Architecture | [Read Online](en/08-kv-transfer-and-chunked-prefill.md) |
| **09** | Chapter 09: Speculative Decoding: Implementation Mechanics & Speedup Benchmarks | [Read Online](en/09-speculative-decoding-implementation.md) |
| **10** | Chapter 10: Model Quantization: AWQ, GPTQ, FP8 & Optimized GEMM Kernels | [Read Online](en/10-quantization-awq-gptq-fp8.md) |
| **11** | Chapter 11: Server Concurrency: AsyncLLMEngine & OpenAI-Compatible REST Server | [Read Online](en/11-async-llm-engine-and-api-server.md) |
| **12** | Chapter 12: Performance Profiling: Throughput Optimization, TTFT/ITL Breakdown | [Read Online](en/12-performance-profiling-and-benchmarking.md) |
| **13** | Chapter 13: Production Deployment & Stability: Memory Leaks, Deadlock Prevention & HA | [Read Online](en/13-production-deployment-and-stability.md) |
| **14** | Chapter 14: Architectural Evolution: From vLLM v1 to Future Inference Systems | [Read Online](en/14-architecture-evolution-v1-and-future.md) |

---

## 🔍 How to Read
1. **GitHub Markdown Reader**: Click any chapter link above to browse full source code walkthroughs directly in GitHub.
2. **Interactive Dual-Pane Reader**: Visit [malerjc.github.io/AiReadCodeBooks/books/vllm/#ch-01](https://malerjc.github.io/AiReadCodeBooks/books/vllm/#ch-01) to view the synchronized code inspector.
3. **Desktop App**: Open your own local clone with the [AiReadCode Desktop Client](https://aireadcode.com/#downloads) to generate architecture books for any repository.

## 📄 License
This work is released under **CC BY-NC 4.0** for open technical education. Code snippets follow the upstream repository's open-source license (vllm-project/vllm).
