# Vue 3 源码与工程化全景架构

[![GitHub stars](https://img.shields.io/badge/GitHub-vuejs%2Fcore-blue?logo=github)](https://github.com/vuejs/core)
[![Stars](https://img.shields.io/badge/Stars-45.8k-yellow)](#)
[![Chapters](https://img.shields.io/badge/章节数-14-emerald)](#)
[![Language](https://img.shields.io/badge/技术栈-TypeScript-purple)](#)
[![在线交互阅读器](https://img.shields.io/badge/在线双栏阅读-GitHub_Pages-cyan?logo=github)](https://malerjc.github.io/AiReadCodeBooks/books/vue3/#ch-01)

> 本书以 Vue 官方 core 仓库为蓝本，围绕「一次代码提交如何变成可发布的 npm 包」这一主线，逐层剖析 monorepo 工程化架构、构建流水线、类型测试体系、Playground 调试工具链与发布自动化机制。

---

## 🌐 多语言导航
- **[English Edition](README.md)**: View English chapter index and translations.
- **中文原著 (当前)**: 完整 14~25 章精读目录见下方列表。
- **[网页端双栏交互精读器](https://malerjc.github.io/AiReadCodeBooks/books/vue3/#ch-01)**: 支持实时代码切片联动与 `[FACT]` 行号溯源验证。

---

## 📚 专著目录导读

| 章节 | 章节名称 | 在线精读入口 |
| :---: | :--- | :---: |
| **01** | 第 1 章：宏观认知：core 仓库的工程化哲学 | [立即阅读](zh/01-monorepo-philosophy.md) |
| **02** | 第 2 章：构建闭环：一次构建的端到端调用链 | [立即阅读](zh/02-build-pipeline.md) |
| **03** | 第 3 章：动态构建链路：dev 脚本与 SFC 预编译协议 | [立即阅读](zh/03-dev-scripts-and-sfc.md) |
| **04** | 第 4 章：魔鬼在细节：Tree-shaking 语义与类型声明生成 | [立即阅读](zh/04-tree-shaking-and-typing.md) |
| **05** | 第 5 章：类型测试流水线：源码与类型契约的守门人 | [立即阅读](zh/05-type-testing-pipeline.md) |
| **06** | 第 6 章：模块化与 Monorepo：packages 与 packages-private 的解耦设计 | [立即阅读](zh/06-monorepo-package-decoupling.md) |
| **07** | 第 7 章：核心响应式子系统：@vue/reactivity 的双向绑定与调度 | [立即阅读](zh/07-reactivity-subsystem.md) |
| **08** | 第 8 章：运行时核心：@vue/runtime-core 的虚拟 DOM 与组件生命周期 | [立即阅读](zh/08-runtime-core-vdom.md) |
| **09** | 第 9 章：编译器核心：@vue/compiler-core 的 AST 转换与代码生成 | [立即阅读](zh/09-compiler-core-ast.md) |
| **10** | 第 10 章：SFC 单文件组件编译：@vue/compiler-sfc 的解析与代码块分割 | [立即阅读](zh/10-compiler-sfc-parsing.md) |
| **11** | 第 11 章：平台特定运行时：@vue/runtime-dom 的 DOM 操作与事件绑定 | [立即阅读](zh/11-runtime-dom-events.md) |
| **12** | 第 12 章：开发调试与生态工具链：sfc-playground 与 template-explorer 的工程化支撑 | [立即阅读](zh/12-tooling-and-playground.md) |
| **13** | 第 13 章：性能优化与打包权衡：Tree-shaking、Feature Flags 与 Rollup 插件设计 | [立即阅读](zh/13-performance-and-feature-flags.md) |
| **14** | 第 14 章：架构演进与未来展望：Vue 3 源码的设计权衡与避坑指南 | [立即阅读](zh/14-architecture-evolution-and-best-practices.md) |

---

## 🔍 阅读建议
1. **GitHub 沉浸精读**：点击上述章节链接，直接在 GitHub Markdown 阅读完整讲解、源码切片与设计推断。
2. **网页端真实源码联动**：访问 [malerjc.github.io/AiReadCodeBooks/books/vue3/#ch-01](https://malerjc.github.io/AiReadCodeBooks/books/vue3/#ch-01) 体验双栏联动与行号高亮。
3. **本地代码一键成书**：下载 [AiReadCode 桌面客户端](https://aireadcode.com/#downloads)，一键将任意复杂代码仓库扫描成书。

## 📄 版权与协议
本专著由 AiReadCode 源码编撰引擎自动化深度生成，遵循 **CC BY-NC 4.0** 开源知识共享协议。文中所引用源码归原开源项目所有。
