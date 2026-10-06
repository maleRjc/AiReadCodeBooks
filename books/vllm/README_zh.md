# vLLM 高性能推理引擎核心实现

[![GitHub stars](https://img.shields.io/badge/GitHub-vllm-project%2Fvllm-blue?logo=github)](https://github.com/vllm-project/vllm)
[![Stars](https://img.shields.io/badge/Stars-35.2k-yellow)](#)
[![Chapters](https://img.shields.io/badge/章节数-14-emerald)](#)
[![Language](https://img.shields.io/badge/技术栈-Python%2FC++%2FCUDA-purple)](#)
[![在线交互阅读器](https://img.shields.io/badge/在线双栏阅读-GitHub_Pages-cyan?logo=github)](https://malerjc.github.io/AiReadCodeBooks/books/vllm/#ch-01)

> 本书系统剖析 vLLM 高吞吐 LLM 推理引擎的内部实现：端到端请求调度、PagedAttention 内存虚拟化、连续批处理、CUDA Graph 静态图加速与投机解码。

---

## 🌐 多语言导航
- **[English Edition](README.md)**: View English chapter index and translations.
- **中文原著 (当前)**: 完整 14~25 章精读目录见下方列表。
- **[网页端双栏交互精读器](https://malerjc.github.io/AiReadCodeBooks/books/vllm/#ch-01)**: 支持实时代码切片联动与 `[FACT]` 行号溯源验证。

---

## 📚 专著目录导读

| 章节 | 章节名称 | 在线精读入口 |
| :---: | :--- | :---: |
| **01** | 第 1 章：vLLM 设计哲学与宏观架构：高吞吐大模型推理引擎 | [立即阅读](zh/01-design-philosophy-and-architecture.md) |
| **02** | 第 2 章：核心抽象与数据结构：Request、Sequence 与 KV Cache | [立即阅读](zh/02-core-abstractions-request-sequence-kvcache.md) |
| **03** | 第 3 章：请求生命周期：从 HTTP/CLI 到 EngineCore 的端到端链路 | [立即阅读](zh/03-request-lifecycle-http-to-enginecore.md) |
| **04** | 第 4 章：资源感知与内存管理：PagedAttention 与 KV Cache 显存虚拟化 | [立即阅读](zh/04-paged-attention-and-kv-cache-virtualization.md) |
| **05** | 第 5 章：连续批处理引擎：Continuous Batching 与迭代级调度 | [立即阅读](zh/05-continuous-batching-and-scheduling.md) |
| **06** | 第 6 章：模型执行器与分布式并行：ModelRunner、Worker 与 Tensor/Pipeline Parallel | [立即阅读](zh/06-model-runner-and-distributed-parallelism.md) |
| **07** | 第 7 章：CUDA Graph 与执行加速：静态图捕获与低延迟调度 | [立即阅读](zh/07-cuda-graph-and-execution-acceleration.md) |
| **08** | 第 8 章：KV 传输与多节点缓存：Prefix Caching 与 Chunked Prefill | [立即阅读](zh/08-kv-transfer-and-chunked-prefill.md) |
| **09** | 第 9 章：投机解码加速：Speculative Decoding 的实现与加速比评测 | [立即阅读](zh/09-speculative-decoding-implementation.md) |
| **10** | 第 10 章：量化与压缩支持：AWQ、GPTQ、FP8 与量化内核实现 | [立即阅读](zh/10-quantization-awq-gptq-fp8.md) |
| **11** | 第 11 章：服务端并发与架构：AsyncLLMEngine 与 OpenAI 兼容 API 适配 | [立即阅读](zh/11-async-llm-engine-and-api-server.md) |
| **12** | 第 12 章：性能剖析与基准测试：吞吐量优化、TTFT/ITL 延迟拆解与 Profiling | [立即阅读](zh/12-performance-profiling-and-benchmarking.md) |
| **13** | 第 13 章：生产部署与稳定性：GPU 显存泄漏排查、死锁预防与高可用 | [立即阅读](zh/13-production-deployment-and-stability.md) |
| **14** | 第 14 章：演进历程与架构前瞻：vLLM v1 到未来推理系统的演化路径 | [立即阅读](zh/14-architecture-evolution-v1-and-future.md) |

---

## 🔍 阅读建议
1. **GitHub 沉浸精读**：点击上述章节链接，直接在 GitHub Markdown 阅读完整讲解、源码切片与设计推断。
2. **网页端真实源码联动**：访问 [malerjc.github.io/AiReadCodeBooks/books/vllm/#ch-01](https://malerjc.github.io/AiReadCodeBooks/books/vllm/#ch-01) 体验双栏联动与行号高亮。
3. **本地代码一键成书**：下载 [AiReadCode 桌面客户端](https://aireadcode.com/#downloads)，一键将任意复杂代码仓库扫描成书。

## 📄 版权与协议
本专著由 AiReadCode 源码编撰引擎自动化深度生成，遵循 **CC BY-NC 4.0** 开源知识共享协议。文中所引用源码归原开源项目所有。
