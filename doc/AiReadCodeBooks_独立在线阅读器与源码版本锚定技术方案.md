# AiReadCodeBooks 独立在线阅读器与源码版本锚定技术方案

> **文档标识**：AiReadCode-DOC-2026-READER-DECOUPLING  
> **制定日期**：2026-10-06  
> **核心目标**：使 `AiReadCodeBooks` 仓库完全脱离 `www.aireadcode.com` 后端与域名依赖，实现双栏交互式在线阅读；通过 Commit 锚定与轻量切片机制，在无需 Clone 巨大源码的前提下，确保 `[FACT]` 行号溯源永久 100% 精准不漂移。

---

## 目录
1. [背景与核心诉求](#一背景与核心诉求)
2. [总体架构：纯前端去中心化双栏阅读器](#二总体架构纯前端去中心化双栏阅读器)
3. [三大解耦改造实现](#三大解耦改造实现)
4. [源码版本防漂移机制（Commit 锚定与切片固化）](#四源码版本防漂移机制commit-锚定与切片固化)
5. [AiReadCodeBooks 仓库目录结构设计](#五aireadcodebooks-仓库目录结构设计)
6. [自动化构建与 GitHub Pages 部署流水线](#六自动化构建与-github-pages-部署流水线)
7. [分步实施与落地路线图](#七分步实施与落地路线图)

---

## 一、背景与核心诉求

在 `AiReadCode` 项目体系中，用户通过 AI 分析开源项目并生成系统性的架构专著（如 `books/vue3`、`books/tokio` 等）。目前官网展示的在线阅读页面（如 `https://www.malerjc.github.io/AiReadCodeBooks/books/vue3/#ch-01#ch-01`）具备优秀的双栏交互体验。

但在开源书籍开源化运作与长期留存中，存在两个核心诉求：
1. **完全解耦，去中心化独立运行**：
   - 不依赖 `www.aireadcode.com` 任何服务器、API 与域名；
   - 所有的书籍文本、元数据、阅读器资源完全自包含在开源仓库 [maleRjc/AiReadCodeBooks](https://github.com/maleRjc/AiReadCodeBooks) 中；
   - 支持通过 GitHub Pages 免费全球分发，或直接在离线本地双击打开阅读。
2. **源码溯源（FACT）防漂移与版本一致性**：
   - 专著正文中包含大量精细代码引注（如 `[FACT:packages/core/src/index.ts:45-50]`）；
   - 上游官方开源库后续会不断产生新的 Commit，若盲目跟踪 `main` 分支，代码行号必然错位甚至 404；
   - **不能**把每个项目几百 MB / 数 GB 的全量代码库直接 clone 存入书籍仓库（否则 100 本书会直接撑爆 GitHub 仓库）；
   - 必须通过科学的工程机制，实现**零存储膨胀、行号永久 100% 对齐**。

---

## 二、总体架构：纯前端去中心化双栏阅读器

官网的在线阅读器本质上是 **纯前端单页架构（HTML5 + Vanilla CSS + Vanilla JS）**，无需任何后端计算即可运行：

```
+-----------------------------------------------------------------------------------+
|  顶部导航栏 (Navbar): 专著标题 / GitHub 徽章 / 深浅主题切换 (Dark/Light Mode)       |
+---------------------+-------------------------------+-----------------------------+
|  左侧章节目录 (TOC)  |  中间专著阅读区 (Reader Pane)  |  右侧真实源码检视器 (Code)    |
|                     |                               |                             |
|  - 01 宏观架构       |  Breadcrumbs / 章节标题        |  当前文件: index.ts (L45-50)|
|  - 02 构建闭环       |  Markdown 排版正文             |  - 自动定位与滚动           |
|  - 03 响应式核心     |  - 〔设计推断与架构权衡〕框      |  - 目标行号高亮与闪烁提示     |
|  ...                |  - [FACT:xxx:45-50] 药丸      |  - 复制代码选段 / 复制全文   |
|  - 14 最佳实践       |  - 自动代码复制按钮             |  - 真实代码行号完全对齐     |
|                     |                               |                             |
|  (URL Hash 联动)    |  (#ch-01 ~ #ch-14 平滑切换)   |  (双重保险: CDN + 本地缓存)  |
+---------------------+-------------------------------+-----------------------------+
```

---

## 三、三大解耦改造实现

| 依赖点 | 原官网依赖方式 | 独立于 AiReadCodeBooks 的去中心化方案 |
| :--- | :--- | :--- |
| **1. 页面托管** | 依赖官网服务器 Nginx 托管 `/books/vue3.html` | **GitHub Pages 免费原生托管**：<br>地址：`https://malerjc.github.io/AiReadCodeBooks/books/vue3/#ch-01`<br>（全球 CDN 自动加速、0 运维成本） |
| **2. CSS / 脚本资源** | 引用主站 `../style.css`、`../theme.js`、`../favicon.svg` | **样式与主题自包含**：<br>将黑曜石科技暗色/极简纯净白主题、TOC 联动逻辑内聚于仓库的 `assets/` 目录下（或直接内联至单个 HTML） |
| **3. 源码检视器** | 发起官网专有请求：<br>`fetch('/sources/' + bookSlug + '/' + file)` | **两级去中心化获取**：<br>① 主力：通过 jsDelivr / GitHub Raw 从指定不可变 Commit 节点拉取；<br>② 兜底：从本地仅几百 KB 的 `snippets.json` 极速加载 |

---

## 四、源码版本防漂移机制（Commit 锚定与切片固化）

### 1. 为什么不能 Clone 完整源码到书籍仓库？
- 大型开源项目（如 PyTorch、vLLM、LLVM、Vue 等）单个源码目录通常在 **50MB ~ 2GB** 不等；
- 若计划收录 100 本开源专著，直接存放完整源码将导致仓库体积达到 **数十 GB**；
- GitHub 对单个仓库推荐上限为 1GB~5GB，单文件超 100MB 会被拒绝 Push；
- 结论：**严禁将外部完整源码提交到书籍仓库**。

### 2. 解法一：Git Commit Hash 永久锚定（0 额外存储成本）
Git 采用 SHA 内容寻址，**Commit Hash 对应的文件内容快照是永久固定、不可更改的**。

在专著元数据 `meta.json` 中，除了记录 `repo` 之外，强制锁定生成文章时的精准 Commit SHA：

```json
{
  "slug": "vue3",
  "titleZh": "Vue 3 源码与工程化全景架构",
  "repo": "vuejs/core",
  "branch": "main",
  "commit": "add68a0ab73d4040a372c22d96c97b43a99e9e9c",
  "version": "v3.5.13"
}
```

当读者在阅读器中点击 `[FACT:packages/core/src/index.ts:45-50]` 时：
- **请求地址**：
  ```text
  https://cdn.jsdelivr.net/gh/vuejs/core@add68a0ab73d4040/packages/core/src/index.ts
  ```
  或者通过 GitHub 原始节点备用：
  ```text
  https://raw.githubusercontent.com/vuejs/core/add68a0ab73d4040/packages/core/src/index.ts
  ```
- **核心收益**：
  - 无论未来 5 年内官方仓库提交了多少万次代码，该 Commit 快照中的行号永远不变；
  - jsDelivr 支持全网免费 CDN 与开放 CORS 跨域，全球加载极其迅速；
  - 书籍仓库不需要存放原项目任何一个 `.ts` / `.py` 文件。

### 3. 解法二：离线切片抽取缓存（Snippets Cache，仅 300~500 KB）
作为第二重保险，在编译生成专著的同时，自动运行提取器：
1. 扫描该书全部 Markdown 中出现的 `[FACT:path:lines]` 标签；
2. 仅把被引用的真实代码行（及前后 4~5 行上下文）抽取出来，组装为 `snippets.json`；
3. 全书引用的代码片段打包后通常只有 **300 KB ~ 500 KB**。

```json
{
  "packages/core/src/index.ts:45-50": {
    "file": "packages/core/src/index.ts",
    "lines": "45-50",
    "items": [
      { "n": 41, "t": "export * from '@vue/reactivity'", "h": 0 },
      { "n": 45, "t": "export function compile() {", "h": 1 },
      { "n": 50, "t": "}", "h": 1 },
      { "n": 55, "t": "// context end", "h": 0 }
    ]
  }
}
```

**双重保险协同逻辑：**
- **网络通畅时**：优先通过 Commit Hash 获取完整文件，读者不仅能看到引用行，还能在右侧上下滚动浏览整个原文件；
- **弱网、离线或 CDN 受阻时**：前端捕获到 Fetch 失败，立即毫秒级回退至本地的 `snippets.json` 显示精准切片；
- 保证 100% 可用率，永久不失效。

---

## 五、AiReadCodeBooks 仓库目录结构设计

推荐在 [maleRjc/AiReadCodeBooks](https://github.com/maleRjc/AiReadCodeBooks) 中按如下规范组织：

```text
AiReadCodeBooks/
├── .github/
│   └── workflows/
│       └── deploy-pages.yml         # GitHub Actions: 自动编译并部署至 GitHub Pages
├── assets/                          # 全局自包含阅读器样式与公共脚本 (零外部 CDN 锁)
│   ├── reader.css                   # 双栏自适应布局、黑曜石科技暗色/纯净白主题、代码高亮
│   ├── reader.js                    # 目录 Hash 路由、FACT 联动、Commit CDN 抓取器
│   └── favicon.svg
├── books/
│   ├── vue3/
│   │   ├── zh/                      # 14 章中文 Markdown 源码
│   │   │   ├── 01-monorepo-philosophy.md
│   │   │   └── ...
│   │   ├── en/                      # 14 章英文 Markdown 源码
│   │   ├── meta.json                # 包含精确 commit hash 与章节映射
│   │   ├── snippets.json            # 离线兜底切片 (约 400KB)
│   │   ├── index.html               # 编译后的单页交互式阅读器 (对应 #ch-01)
│   │   └── README_zh.md             # 目录导航，顶部挂载在线阅读直达徽章
│   ├── tokio/
│   ├── vllm/
│   └── nccl/
├── doc/
│   └── AiReadCodeBooks_独立在线阅读器与源码版本锚定技术方案.md
├── scripts/
│   ├── build_readers.py             # 本地一键编译所有书籍为 index.html 与 snippets.json
│   └── book_matrix_100.json
├── index.html                       # 书库总览大厅（开源书架，展示 100 本开源书矩阵）
└── README.md
```

---

## 六、自动化构建与 GitHub Pages 部署流水线

通过 GitHub Actions 实现“撰写/更新 Markdown，自动发布在线阅读器”的全自动闭环。

在 `.github/workflows/deploy-pages.yml` 中：

```yaml
name: Deploy Open Source Books to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Build Readers and Snippets
        run: |
          python scripts/build_readers.py

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload Artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: "."

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

---

## 七、分步实施与落地路线图

| 步骤 | 行动项 | 交付成果 |
| :---: | :--- | :--- |
| **Step 1** | **元数据规范升级** | 在已生成的 `books/*/meta.json` 中统一补充生成时的准确 `commit` SHA 与 `version` |
| **Step 2** | **提取自包含样式与驱动** | 制作独立的 `assets/reader.css` 与 `assets/reader.js`，彻底移除对 `www.aireadcode.com/sources/` 的硬编码调用 |
| **Step 3** | **编写独立编译脚本** | 在 `AiReadCodeBooks/scripts/build_readers.py` 实现纯静态生成，支持将任意 `zh/*.md` 生成 `index.html` 并提取 `snippets.json` |
| **Step 4** | **开启 GitHub Pages** | 在 GitHub 仓库 Settings -> Pages 中开启部署，在线链接形如 `https://malerjc.github.io/AiReadCodeBooks/books/vue3/#ch-01` |
| **Step 5** | **更新徽章与导读入口** | 将各书籍 `README_zh.md` 的在线阅读链接直接切为 GitHub Pages 独立地址 |

---

> **结论**：通过上述方案，`AiReadCodeBooks` 既获得了如同官方网站般的极致在线交互阅读体验，又彻底摆脱了中心化服务器的依赖；同时借助 Git 提交哈希锚定，确保哪怕数年后代码行号依然百分之百精准吻合。
