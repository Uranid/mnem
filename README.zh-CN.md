<div align="center">

<img src="assets/logo/mnem-banner.svg" alt="mnem: Git for AI Agent Knowledge" />

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Uranid/mnem/ci.yml?style=for-the-badge&label=CI)](https://github.com/Uranid/mnem/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mnem-cli?style=for-the-badge)](https://crates.io/crates/mnem-cli)
[![PyPI](https://img.shields.io/pypi/v/mnem-cli?style=for-the-badge)](https://pypi.org/project/mnem-cli/)
[![npm](https://img.shields.io/npm/v/mnem-cli?style=for-the-badge)](https://www.npmjs.com/package/mnem-cli)
[![MSRV 1.95](https://img.shields.io/badge/MSRV-1.95-orange?style=for-the-badge)](rust-toolchain.toml)
[![Runs on Linux macOS Windows WASM](https://img.shields.io/badge/runs%20on-linux%20%7C%20macos%20%7C%20windows%20%7C%20wasm-2ea44f?style=for-the-badge)](#安装)

</div>

<div align="center">

[English](README.md) &nbsp;·&nbsp; [中文](README.zh-CN.md) &nbsp;·&nbsp; [Español](README.es.md)

</div>

<hr>

<div align="center">

https://github.com/user-attachments/assets/bd744a7e-8e89-4531-bd96-fdee0030c390

</div>

<hr>

1. [问题所在](#问题所在)
2. [什么是 mnem](#什么是-mnem)
3. [性能基准](#性能基准)
4. [你能获得什么](#你能获得什么)
5. [安装](#安装)
6. [快速入门](#快速入门)
7. [接入与取消接入](#mnem-integrate---接入任何-agent-宿主)
8. [命令](#命令)
9. [MCP 工具](#mcp-工具)
10. [Python API](#python-api-mnem-py)
11. [GraphRAG](#graphrag)
12. [与其他工具的比较](#与其他工具的比较)
13. [不适合使用 mnem 的场景](#不适合使用-mnem-的场景)
14. [文档](#文档)
15. [贡献](#贡献)

<hr>

## 问题所在

> **影响对象：** 如果你使用 AI 编程助手（Claude Code、Cursor、Gemini CLI 等），或者正在构建需要 AI Agent 在会话间记住信息的软件，mnem 正是为解决这一问题而生。

> **每个会话都从零开始。**

- **会话相互隔离。** 在 Claude Code（一款 AI 编程助手）中规划一次迁移。明天打开 Cursor（另一款 AI 编程助手）。那个 Agent 对此一无所知。
- **无法检查的记忆不是真正的记忆。** Agent 的上下文发生了变化，你不知道是什么、何时、为什么。没有日志。
- **约定规则在扁平文件中腐烂。** 六个工程师，六份各自悄然发散的 `AGENTS.md` 文件（许多 AI 工具会自动读取的 Agent 配置文件）。没有合并，没有历史，无法判断哪份是最新的。

> 你的代码库有 git，你 Agent 的知识却什么都没有。

<hr>

## 什么是 mnem

> **不熟悉 git 或版本控制？** Git 是一种随时间保存文件编号快照的软件，让你可以追踪变更、撤销错误、协同协作。mnem 对 AI Agent 的知识做同样的事：每次写入都是一个可分支、可 diff、可合并或回滚的已保存快照。

**AI Agent 知识的 git。** 一个持久化、版本化的 AI Agent 知识层，在所有测试基准上达到最佳或并列最佳的召回率（召回率 = 返回正确结果的比例，越高越好）。

**知识图谱**是一种条目之间可以相互链接的可搜索事实存储，可以把它想象成你的 AI Agent 可以写入和读取的智能笔记本。例如：写入"部署窗口为每周二 UTC 10-11 时"，将其链接到发布检查清单，之后用普通中文询问"我们的部署计划是什么？"即可检索到它。（对于想了解技术术语的人：事实以节点形式存储，通过类型化关系边相连，如 `part_of`、`relates_to`、`depends_on` 等。）

技能、决策和上下文以可查询图谱的形式存储在项目文件夹中。提交 `.mnem/` 目录，它就随代码一起移动。用可供整个团队版本化、diff 和合并的东西替换陈旧的 `.cursorrules`（Cursor 的项目规则文件）和 `AGENTS.md` 文件。

检索在一次遍历中融合向量搜索（按语义查找结果，而非仅精确词语，如"deploy schedule"能找到"deploy window"）、关键词搜索（精确词语）和图遍历（沿条目间的链接跟踪）。每次查询都精确报告消耗了多少 token 以及过滤掉了什么，不会有任何内容被静默丢弃。单个二进制文件（一个可执行文件），无需运行服务器。一条命令即可接入 Claude Code、Cursor、Gemini CLI 或任何 MCP（Model Context Protocol，为 AI 工具提供外部能力访问的标准）宿主；可从 CLI、HTTP 或 Python 使用。

> **对于团队：** 将 `.mnem/` 与代码一同提交，每位队友的 Agent 都从相同的知识基线出发。参见 [mnem push / mnem pull](#10-mnem-push--mnem-pull--mnem-clone---与远端同步) 了解 CI 同步方式。

## 性能基准

**在六个公开数据集上与 mem0 和 MemPalace 进行了正面对比测试。mnem 在五个数据集上领先‡†；在 LongMemEval 上与 MemPalace 并列。**

<div align="center"><img src="assets/benchmarks/benchmarks.svg" alt="mnem public benchmarks" /></div>

<details>
<summary><b>方法论、脚注、查询速度与复现步骤</b></summary>

> **方法论：** mem0 数据为我们在相同测试框架下的复现结果，mem0 未在这些数据集上发布 R@K（Recall at top K，即前 K 个结果中正确答案的比例）头条数字。MemPalace 头条数字已在我们的测试框架下交叉验证。这是公开披露，而非隐瞒：可复现的产物与二进制文件一同发布。

默认测试框架嵌入器：MiniLM-L6-v2（ONNX 格式的小型预训练文本模型，ONNX 是 AI 模型的开放文件格式，无需单独安装），各系统使用完全相同的字节。FinanceBench 在所有系统上使用 bge-large 以公平比较（见 † 脚注）。不使用 LLM 重排序。每次运行样本数：LongMemEval 500 问，LoCoMo 完整数据集（约 1986 问），ConvoMem 每类别 50 问，MemBench 每配置 100 问。所有基准测试仅使用密集检索（不含稀疏/BM25 通道）。复现方法：`bash benchmarks/harness/run_bench.sh`。

<sup>mem0 列：我们在相同测试框架下的复现结果（mem0 未在这些数据集上发布 R@K 头条数字）。MemPalace 列：公开头条数字，已在我们的测试框架下交叉验证。原始产物：[`benchmarks/results/v0.1.0/`](benchmarks/results/v0.1.0/)。† FinanceBench 在所有系统上均使用 Ollama bge-large（1024 维）；MemPalace 展示的是最佳配置下的结果（bge-large 直连 ChromaDB）；mem0 在存储前对记忆应用了 LLM 提取。流水线说明：mnem FinanceBench 运行使用了混合检索（`--hybrid-boost --query-expand`）；MemPalace bge-large 使用纯向量检索，流水线不同。完整方法论：[`benchmarks/results/analysis/financebench.md`](benchmarks/results/analysis/financebench.md)。‡ LoCoMo：mnem 使用 MAX-over-turn-hits 会话评分（宽松）；MemPalace 使用逐轮聚合（更严格），分数反映的是不同评估方法。参见 [`benchmarks/results/analysis/locomo.md`](benchmarks/results/analysis/locomo.md)。</sup>

### 查询速度

<div align="center"><img src="assets/benchmarks/query-speed.svg" alt="mnem query speed" /></div>

<details>
<summary><b>复现方法</b></summary>

```bash
mnem bench fetch longmemeval     # 下载数据集（一次性，264 MB）
mnem bench                       # TUI 界面；交互式选择基准测试
mnem bench run --benches longmemeval --limit 50 --non-interactive
mnem bench results ./bench-out   # 从上次运行的结果重新渲染

# 传统 bash 工具（官方标题数字的标准路径）
bash benchmarks/harness/run_bench.sh
```

方法论、原始产物、各基准测试详细分类：[`benchmarks/`](benchmarks/) 和 [`docs/src/benchmarks/`](docs/src/benchmarks/)。

</details>

</details>

<hr>

## 你能获得什么

<sup><img src="assets/legend/unique.svg" width="12" height="12" alt="unique"> mnem 独有 &nbsp;·&nbsp; <img src="assets/legend/rare.svg" width="12" height="12" alt="rare"> 同类中罕见</sup>

| | | |
|:---:|:---|:---|
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **从任意文件或代码库即时构建知识图谱，无需调用 LLM。** 导入源代码、PDF、Markdown 文档或对话导出，mnem 自动处理一切。一条命令，支持 30 余种文件格式，自动解析并建索引。 | [了解更多](docs/features/rich-ingest.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **像 git 一样对知识进行分支、diff 和合并。** 每次写入都是一个版本化提交。在分支上实验，准备好后合并，你的知识图谱拥有与代码库相同的操作原语。 | [了解更多](docs/features/versioned-memory.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **用版本化、可查询的图谱替代扁平的 Agent 文件。** `.cursorrules` 和 `AGENTS.md` 无法被 diff 或合并。mnem 可以，导出你的图谱，导入队友的图谱，合并你需要的部分。 | [了解更多](docs/features/skills-graph.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **精确看到检索找到了什么、跳过了什么，以及花费了多少代价。** 每次查询都返回 `tokens_used`、`candidates_seen` 和 `dropped`。不会在 token 预算处发生静默截断。 | [了解更多](docs/features/token-transparency.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **相同输入，相同输出，任意机器（存储层）。** 每份内容都会根据其精确字节获得唯一指纹。两次存储同一事实，mnem 会自动去重，无论是哪台机器、哪个会话或哪个用户摄入的。检索结果按近似相似度排序，不同运行间可能略有差异。 | [了解更多](docs/features/content-addressing.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **可在浏览器标签页中运行。**（进阶功能，仅使用 CLI 的话可跳过。）相同的二进制文件通过 WASM（WebAssembly，一种在浏览器中运行编译代码的方式）在 Chrome 中运行，并可在 AWS Lambda 中部署（约 40 MB）。无需 Python，无需外部数据库。WASM 绑定单独发布，参见 [`docs/features/wasm-edge.md`](docs/features/wasm-edge.md)。 | [了解更多](docs/features/wasm-edge.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **在所有测试基准上达到最佳或并列最佳召回率。** 在六个公开基准测试中五个领先（召回率 = 返回正确结果的比例，越高越好）。所有数据均可通过附带的测试框架复现。详见上方[性能基准](#性能基准)。 | [了解更多](docs/features/benchmarks.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **零配置启动，之后可切换任意提供商。** 一个小型预训练文本模型自动在进程内运行（二进制文件总计约 40 MB，无需配置）。通过 `config.toml`（一个简单的键值配置文件）中的一行配置即可切换至 Ollama、OpenAI 或 Cohere。 | [了解更多](docs/features/providers.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **CLI（命令行工具）、HTTP（Web API）、MCP 和 Python，共用同一引擎。** `mnem integrate` 将 MCP 服务器接入 Claude Code、Cursor、Gemini CLI 以及任何支持 MCP 的工具。 | [了解更多](docs/features/integrations.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **单个约 40 MB 的二进制文件，无需其他任何东西。** 无需后台服务（守护进程），无需云端，无需账户。完全离线运行。同一二进制文件同时驱动 CLI 和 HTTP 服务器。 | [了解更多](docs/features/single-binary.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **无 API 调用、确定性的摄入。** 建索引时不调用 LLM。相同文件始终产生完全相同的节点，完全可重现且便于审计。重新摄入未更改的文件不会产生任何新节点。 | [了解更多](docs/features/deterministic-ingest.md) |
| | **向量、关键词和图搜索一次完成。** 为跨文档查询启用多跳遍历（沿多个相连条目的链接链追踪），对快速单文档查找则可跳过。 | [了解更多](docs/features/hybrid-retrieval.md) |

<hr>

## 安装

**前提条件：** 检查你已有什么：`python --version` / `node --version` / `cargo --version`。都没有？[安装 Python](https://www.python.org/downloads) 或 [安装 Node.js](https://nodejs.org/en/download)，两者均免费且包含 pip/npm。需要 Cargo 的话：[通过 rustup 安装 Rust](https://rustup.rs/)（免费，同时安装 `cargo`）。

> **想用 Python 从自己的应用调用 mnem？** `pip install mnem-cli` 给你提供的是 `mnem` 命令行工具。如果要在 Python 代码中导入 mnem（`import pymnem`），请改用 `pip install mnem-py`，参见 [Python API](#python-api-mnem-py)。

**选择一种**（如果你有 Python，推荐使用 pip）：

**pip (Python) - 推荐** · 预构建二进制，内置嵌入器，即装即用

```bash
pip install mnem-cli
```

**npm (Node.js)** · 预构建二进制，内置嵌入器，即装即用

```bash
npm install -g mnem-cli
```

**Cargo (Rust)** · 从源码编译，首次运行约需 5-15 分钟

```bash
# Linux 专属：sudo apt-get install g++ (Debian/Ubuntu/WSL)  或  sudo dnf install gcc-c++ (Fedora/RHEL)
cargo install --locked mnem-cli --features bundled-embedder
```

**Docker** · 运行 HTTP 服务器，无需本地安装

```bash
docker run --rm -p 9876:9876 -e MNEM_HTTP_ALLOW_NON_LOOPBACK=1 \
  ghcr.io/uranid/mnem:latest http --bind 0.0.0.0:9876
```

```bash
mnem --version    # 确认安装成功
```

> **如果提示 `mnem: command not found`：** 先尝试打开新终端（PATH 变更只对新会话生效）。在 Linux 上，pip 安装路径为 `~/.local/bin`，如果该路径不在 PATH 中，运行 `export PATH="$HOME/.local/bin:$PATH"`，然后将该行添加到 `~/.bashrc`（一次性修复，文件修改后永久生效）。在 Windows 上：1. 运行 `pip show mnem-cli`。2. 复制 `Location` 值（如 `C:\Users\you\AppData\Roaming\Python\Python312\site-packages`）。3. 将 `site-packages` 替换为 `Scripts` 得到 Scripts 文件夹路径。4. 打开系统属性 - 环境变量 - Path - 编辑 - 新建 - 粘贴 Scripts 路径 - 确定。5. 打开新的命令提示符（PATH 变更需要新窗口才能生效）。

> [!NOTE]
> `--locked` 固定经过测试的精确依赖版本。`--features bundled-embedder` 将嵌入器（约 40 MB）打包进二进制文件，使 `mnem retrieve` 即刻可用，无需额外配置。**此标志仅适用于 Cargo**；pip 和 npm 已预置内置嵌入器。如不使用该标志（且未在 `config.toml` 中配置其他提供商），`mnem retrieve` 会报错"embedder not configured"。

<details>
<summary>示例 <code>.mnem/config.toml</code>（Ollama 示例）</summary>

```toml
[embed]
provider = "ollama"
model    = "nomic-embed-text"
base_url = "http://localhost:11434"
```

完整配置键列表：[`docs/src/configuration.md`](docs/src/configuration.md)。

</details>

<details>
<summary><b>macOS / Linux</b></summary>

没有 Cargo？[通过 rustup 安装](https://rustup.rs/)（同时安装 `rustc`）。

```bash
# 链接内置 ONNX Runtime 需要 C++ 标准库（仅限 Linux）
sudo apt-get install g++          # Debian / Ubuntu / WSL
# sudo dnf install gcc-c++        # Fedora / RHEL
```

```bash
cargo install --locked mnem-cli --features bundled-embedder

# CUDA 加速嵌入器（Linux，NVIDIA GPU）
cargo install --locked mnem-cli --features bundled-embedder-cuda
```

安装后找不到 `mnem`，说明 `~/.cargo/bin` 不在 `$PATH` 中。

**rustup 安装**：加载环境变量（或打开新终端）：
```bash
source ~/.cargo/env
```

**系统 Rust（apt/dnf）**：永久添加到 PATH：
```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

</details>

<details>
<summary><b>Windows</b></summary>

没有 Cargo？[通过 rustup 安装](https://rustup.rs/)（同时安装 `rustc`）。

```powershell
cargo install --locked mnem-cli --features bundled-embedder

# DirectML 加速嵌入器（Windows，支持任意 GPU 厂商）
cargo install --locked mnem-cli --features bundled-embedder-directml
```

</details>

<details>
<summary><b>npm / Node.js</b></summary>

没有 npm？[安装 Node.js](https://nodejs.org/en/download)（npm 已内置，需要 Node 18+）。

```bash
npm install -g mnem-cli
mnem --version

# 或不全局安装（一次性使用）
npx mnem-cli --version
```

安装时会自动下载适用于你当前平台的预构建原生二进制文件。需要 Node 18+。已内置嵌入器，无需 Ollama 或 API 密钥。

</details>

<details>
<summary><b>pip (PyPI)</b></summary>

没有 pip？[安装 Python](https://www.python.org/downloads/)（pip 随 Python 3.4+ 一同提供）。

```bash
pip install mnem-cli
mnem --version
```

以 manylinux / macOS / Windows wheel 形式发布 `mnem` 二进制文件，已预置内置嵌入器。

</details>

<details>
<summary><b>Docker</b></summary>

没有 Docker？[安装 Docker Desktop](https://docs.docker.com/get-started/get-docker/)。

```bash
# 临时使用（容器停止后数据丢失，仅用于快速测试）：
docker run --rm -p 9876:9876 \
  -e MNEM_HTTP_ALLOW_NON_LOOPBACK=1 \
  ghcr.io/uranid/mnem:latest http --bind 0.0.0.0:9876

# 持久使用（重启后数据保留，正式使用推荐）：
mkdir -p ./mnem-data
docker run -p 9876:9876 \
  -v "$(pwd)/mnem-data:/data" -w /data \
  -e MNEM_HTTP_ALLOW_NON_LOOPBACK=1 \
  ghcr.io/uranid/mnem:latest http --bind 0.0.0.0:9876
# Windows (PowerShell)：将 $(pwd) 替换为 ${PWD}；Windows (cmd.exe)：将 $(pwd) 替换为 %cd%
```

镜像已包含内置嵌入器。`--bind 0.0.0.0:9876` 标志和 `MNEM_HTTP_ALLOW_NON_LOOPBACK=1` 环境变量是 Docker 内运行的必要配置，确保端口映射（`-p 9876:9876`）生效；默认回环绑定在宿主机上无法访问。在容器内运行 `mnem mcp` 可使用 MCP 服务器接口。

> **全新卷（无历史数据）：** 如果挂载的 `/data` 目录中没有 `.mnem/`，`mnem http` 会在首次启动时自动初始化新图谱，无需手动执行 `mnem init`。后续重启时将复用现有图谱。

> **镜像标签固定：** `ghcr.io/uranid/mnem:latest` 始终指向最新发布版本。生产部署时，建议固定到具体版本标签（如 `ghcr.io/uranid/mnem:v0.1.0`），以避免意外升级。

> **⚠️ 默认无身份验证：** 上述示例将 API 暴露在无令牌保护的状态下。任何能访问 9876 端口的人都可以读写你的图谱。在绑定到非回环地址之前，请在服务端设置 `MNEM_HTTP_AUTH_TOKEN`，并在客户端使用 `--token-env`，参见 `mnem push`/`mnem pull` 下的身份验证说明。

</details>

<details>
<summary><b>从源码构建</b></summary>

```bash
# 链接内置 ONNX Runtime 需要 C++ 标准库（仅限 Linux）
sudo apt-get install g++          # Debian / Ubuntu / WSL
# sudo dnf install gcc-c++        # Fedora / RHEL
```

```bash
git clone https://github.com/Uranid/mnem
cd mnem
cargo install --path crates/mnem-cli --features bundled-embedder
```

需要 Rust 1.95+。如有需要：`rustup install 1.95 && rustup default 1.95`。

</details>

```bash
mnem --version
mnem doctor        # 检测嵌入器、存储和配置，输出绿/黄/红状态清单
```

完整安装矩阵：[`docs/src/install.md`](docs/src/install.md)。

> **想将 mnem 嵌入 Python 应用？** 上面的 `pip install mnem-cli` 以 wheel 形式发布的是 **CLI 二进制文件**。原生 **Python API**（`import pymnem`）位于独立的包中。请跳转至 **[Python API (mnem-py) ↓](#python-api-mnem-py)**，查看 `pip install mnem-py` 的安装方式和代码示例。

<hr>

## 快速入门

**第一步：立即体验（独立运行，无需 AI 助手）**

```bash
mkdir my-graph
cd my-graph
mnem init          # 每个项目只需运行一次 - 创建存储知识图谱的 .mnem/ 文件夹
mnem ingest --text "mnem is a versioned knowledge graph for AI agents"
mnem retrieve "what does mnem do"
```

> 每个项目在使用 `mnem ingest` 或 `mnem retrieve` 之前，必须先运行一次 `mnem init`，它会创建存储图谱的 `.mnem/` 文件夹。如果出现问题，运行 `mnem doctor`。

预期输出：
```
[1] score=0.94  mnem is a versioned knowledge graph for AI agents
    tokens_used=12  candidates_seen=1  dropped=0
```

**第二步（可选）：接入你的 AI 助手**

> **前提条件：** 此示例使用 Claude Code。没有的话？在 [claude.ai/code](https://claude.ai/code) 免费下载。没有 Agent？跳过第二步，`mnem retrieve` 可独立使用。

> **工作目录：** 接入后，请从 `my-graph/`（或其子目录）打开 Claude Code。从其他文件夹启动意味着它找不到这个图谱。

```bash
# 第一个会话：添加事实并接入 Agent
mnem init     # 如果第一步已运行可跳过
mnem ingest --text "The API retry policy uses exponential backoff with a 3-attempt limit"
mnem integrate claude-code    # Cursor：使用 `mnem integrate cursor`

# 第二个会话（第二天，新终端）：记忆持续存在
cd my-graph
mnem retrieve "what is our API retry policy"
```

`mnem integrate` 完成后，关闭并重新打开应用程序（不仅仅是终端）。验证方法：打开任意会话并发送消息，Claude 回复之前你应该能看到 `mnem: N item(s)` 作为系统消息出现在对话顶部。`0 item(s)` 表示图谱为空，但接入是正常工作的。

> **本地图谱与全局图谱：** 项目目录中的 `.mnem/` 存储项目专属记忆。`~/.mnemglobal/.mnem/`（全局图谱，其中 `~` 表示主目录，Windows 上为 `C:\Users\you`，Linux/macOS 上为 `/home/you`）存储跨所有项目的事实：个人偏好、团队共享约定、跨仓库实体等。使用 `mnem global retrieve` 和 `mnem global add` 来操作它。

**下一步：**
- 摄入文件：`mnem ingest README.md`（或 `mnem ingest your-docs/ --recursive` 摄入整个目录）
- 接入 AI 助手：`mnem integrate`（支持 Claude Code、Cursor 等）
- 随意提问：`mnem retrieve "你的问题"`

五分钟从零上手。完整演练参见 [`docs/src/quickstart.md`](docs/src/quickstart.md)。

<hr>

## `mnem integrate` - 接入任何 Agent 宿主

> **不使用 Claude Code、Cursor 或其他 AI 编程助手？** 跳过此节，`mnem integrate` 只在你希望这些工具自动使用 mnem 时才需要。

> **Claude Code、Cursor 等工具必须已安装。** `mnem integrate` 会检测哪些工具已存在，先运行 `mnem integrate --check` 查看检测到的工具。

一条命令将三件事写入你的 Agent 宿主：**MCP 服务器**（让 Agent 访问 `mnem_retrieve` 和 `mnem_commit` 等 mnem 工具）、**自动检索触发器**（在每条消息前自动运行 `mnem retrieve`，在 Agent 回复前将相关记忆注入上下文；仅限 Claude Code，以"UserPromptSubmit hook"实现）以及 **mnem 系统提示**（告诉 Agent 如何使用 mnem）。重启宿主（将 Claude Code 或 Cursor 作为应用程序完全关闭并重新启动，而不仅仅是终端），Agent 即开始自动使用 mnem。验证方法：打开新会话并发送任意消息，你应该能看到一行类似 `mnem: 3 item(s): [...]` 的内容作为系统消息出现在对话顶部，在 Claude 发言前注入。`0 item(s)` 没问题，表示图谱为空，接入是正常工作的。

> **故障排除：** 没有看到 `mnem: N item(s)`？
> - 确保你已完全关闭并重新打开**应用程序**（不仅仅是终端），即完全关闭 Claude Code 或 Cursor 窗口并重新启动它
> - 从包含 `.mnem/` 文件夹的目录（或其子目录）内打开应用程序，从其他文件夹打开 Claude Code 则找不到那个项目的图谱
> - 运行 `mnem doctor` 检查嵌入器和存储是否正常
> - 运行 `mnem integrate --check` 确认宿主是否正确接入

```bash
mnem integrate                           # 交互式：检测已安装的宿主并提示选择
mnem integrate claude-code               # 接入指定宿主，跳过交互式检测
mnem integrate --all                     # 无需提示，接入所有检测到的宿主

mnem integrate --check                   # 报告所有宿主的接入状态；不做任何修改
mnem integrate --dry-run                 # 预览将写入的内容，不实际修改任何文件
mnem integrate --show claude-code        # 打印 MCP JSON 块，供手动复制粘贴

mnem integrate --no-hooks                # 跳过 UserPromptSubmit 钩子接入
mnem integrate --no-system-prompt        # 跳过系统提示接入
mnem integrate --target-repo ~/notes     # 将 MCP 服务器指向指定图谱而非全局图谱
```

**接入内容：**
- **MCP 服务器**（`mcpServers.mnem`）- Agent 通过 `mnem mcp --repo <graph>` 获得完整的 mnem 工具访问权限；默认指向全局图谱（`~/.mnemglobal/.mnem`）
- **自动检索触发器**（仅限 Claude Code；以"UserPromptSubmit hook"实现）- 在每条消息前运行 `mnem retrieve`，在模型看到你的提示之前将相关记忆注入上下文
- **系统提示** - mnem 使用说明注入宿主的项目规则文件

钩子始终优先查询项目的 `.mnem/`（从当前目录向上查找），若未找到则自动回退至 `mnem global retrieve`。无论在设置期间选择哪个默认知识图谱，钩子和系统提示的行为保持一致。仅当你希望 MCP 服务器指向全局图谱以外的位置时，才需要使用 `--target-repo`。

自动检测并配置：
- Claude Code
- Claude Desktop
- Cursor
- Continue
- Zed
- Gemini CLI

任何其他支持 MCP 的宿主均可通过手动编辑 `mcpServers` 条目，指向 `mnem mcp --repo <path>` 来接入，参见 [`docs/src/mcp.md`](docs/src/mcp.md)。

Agent 将获得完整的 mnem 工具集作为原生工具：检索、提交、摄入、软删除（tombstone）、遍历、全局图谱访问等。无需额外守护进程，无需管理端口。完整工具参考：[`docs/src/mcp.md`](docs/src/mcp.md)。

<details>
<summary>从宿主移除 mnem</summary>

```bash
mnem unintegrate                  # 交互式：选择要移除 mnem 的宿主
mnem unintegrate claude-code      # 移除单个宿主
mnem unintegrate --all            # 移除所有已接入的宿主
```

运行 `mnem unintegrate --help` 查看所有选项。

</details>

<hr>

## 命令

> **术语速查：** **node（节点）** = 图谱中的单个条目（一个事实、文档块或实体，任何你存储的内容）。**edge（边）** = 两个节点之间的类型化链接（`depends_on`、`relates_to`、`part_of` 等）。**CID** = 内容寻址 ID，基于精确字节的唯一指纹，每个节点、边和提交都有一个。**HEAD** = 当前操作日志的末端（最新提交，与 git 中概念相同）。**op-log** = 所有写操作的仅追加日志。**ref** = 指向某个提交 CID 的命名指针（如 `refs/heads/main`，与 git 的分支或标签相同）。

每个命令都接受 `--help` 查看完整的标志参考。完整 CLI 参考：[`docs/src/cli.md`](docs/src/cli.md)。

---

### 1. `mnem init` - 初始化知识图谱

在当前目录创建 `.mnem/` 存储。将其与代码库一同提交，使每位开发者和 Agent 从相同的基线出发。

```bash
mnem init
```

> **示例：** 你的团队随 API 服务一同发布 AI Agent。在仓库根目录运行一次 `mnem init`，克隆该仓库的每位工程师都将拥有与 Agent 训练所用相同的知识库。

<details>
<summary>健康检查与诊断</summary>

```bash
mnem doctor    # 检测嵌入器、存储和配置 - 输出绿/黄/红状态清单
mnem stats     # 一览节点、边、引用及存储大小
```

</details>

---

### 2. `mnem ingest` - 向图谱添加文档

在一次遍历中将文件或目录解析为 `Doc`、`Chunk` 和 `Entity` 节点。摄入时无需 LLM，具有确定性且便于审计：相同字节始终产生相同的 CID（内容寻址 ID，从内容字节自动计算的唯一指纹，每个节点、边和提交都有一个）。

```bash
mnem ingest architecture.md
mnem ingest --recursive docs/               # 批量导入整个目录
```

文件类型根据扩展名自动检测：Markdown 使用感知标题结构的分块，源代码（`.rs`、`.py`、`.ts`、`.go` 等）使用 Tree-sitter 函数/类级别解析，PDF 使用滑动窗口文本提取，均自动处理，无需任何标志。

> **示例：** 一个 Agent 在入职你的平台时，摄入 `ARCHITECTURE.md`、`runbooks/` 目录以及所有 ADR 文件。后续每个 Agent 都能检索到相同的结构化知识，无需重新逐一读取文件。

<details>
<summary>更多选项</summary>

```bash
mnem ingest --text "Deploy window is Tuesdays 10-11 AM UTC"  # 直接导入内联文本，无需文件
mnem ingest src/ --recursive                # 导入 src/ 下的所有源文件
mnem ingest --chunker recursive report.pdf  # PDF 使用显式递归分块
mnem ingest --extractor keybert notes.md    # 关键短语增强，提升稀疏检索效果
mnem ingest --max-tokens 256 notes.md       # 更小的分块，实现细粒度检索
```

</details>

---

### 3. `mnem add` - 写入单个事实和关系

提交单个事实节点，或用类型化边连接两个实体。这是最基础的写操作原语，当你需要精确控制图谱内容时使用。可选的 `--label` 标签（如 `Fact`、`Convention`、`Decision`）对节点分类，便于之后按类型过滤检索结果。

```bash
mnem add node -s "Deploy window is Tuesdays 10-11 AM UTC"
```

> **示例：** 对话中途，Agent 发现了一个未记录的约束。它立即提交这个发现，确保所有下游 Agent 都从同一份共享事实出发，不再在不同会话中重复发现同一边界情况。

<details>
<summary>更多写入选项</summary>

```bash
mnem add node --label Fact -s "The payments API uses idempotency keys for all POST requests"
mnem add node --label Convention -s "All REST APIs are versioned under /v1/"
mnem add edge --from <uuid> --to <uuid> --label depends_on        # 连接两个已有节点
```

</details>

<details>
<summary>读取和删除节点</summary>

```bash
mnem get <uuid>                                                    # 通过 UUID 获取节点
mnem get <uuid> --content                                         # 包含完整内容正文

mnem tombstone <uuid>                                             # 软删除：从检索中隐藏，保留在审计日志中
mnem tombstone <uuid> --reason "superseded by v2 decision"        # 记录原因
mnem delete <uuid>                                                # 硬删除：无审计记录

mnem global get <uuid>                                            # 在全局图谱中查找节点
mnem global tombstone <uuid>                                      # 在全局图谱中软删除
```

</details>

---

### 4. `mnem retrieve` - 搜索图谱

在一次遍历中进行混合语义 + 关键词 + 图检索。精确返回找到了什么、跳过了什么、使用了多少 token，不会在 token 预算处发生静默截断。

```bash
mnem retrieve "what did we decide about the API rate-limit design"
```

> **示例：** 三个迭代后，新工程师问 Agent"为什么我们的重试逻辑是指数级的？"Agent 检索到包含完整原因说明的原始决策节点，无需任何人特意记得单独记录它。

<details>
<summary>更多选项</summary>

```bash
mnem -R ~/notes retrieve "query"           # 显式指定目标图谱
mnem retrieve "..." --limit 20             # 返回更多结果
mnem retrieve "..." --graph-expand 20      # 启用多跳图遍历
mnem retrieve "..." --graph-expand 20 --community-filter --graph-mode ppr
mnem retrieve "..." --rerank cohere:rerank-english-v3.0
mnem retrieve "..." --vector-cap 512       # 扩大密集候选池
mnem retrieve "..." --explain              # 输出各条目的分项通道得分（vector、sparse、graph_expand、rerank）
```

完整标志参考参见 [GraphRAG](#graphrag)。

</details>

---

### 5. `mnem global` - 跨项目、跨会话记忆

位于 `~/.mnemglobal/.mnem/`（其中 `~` 是你的主目录：Windows 上为 `C:\Users\you`，Linux/macOS 上为 `/home/you`）的第二个图谱，随 Agent 走遍任何地方，跨仓库、跨团队、跨会话。用于共享约定、供应商决策以及出现在每个项目中的实体。

```bash
mnem global retrieve "what payment provider do we use"
mnem global add node --label Convention -s "All REST APIs are versioned under /v1/"
```

> **示例：** 你的平台有十几个微服务，每个都有自己的 `.mnem/`。全局图谱存储全团队的约定、共享实体定义和跨服务决策。任何服务上的任何 Agent 都可以查询它，无需知道事实来自哪个仓库。

<details>
<summary>更多选项和本地与全局使用指南</summary>

```bash
mnem global ingest contacts.md
mnem global add node --label Entity:Person \
  --prop name=Alice -s "Alice leads the infra team"
mnem global get <uuid>
mnem global tombstone <uuid>
```

**本地与全局的使用场景：**

| 使用本地 `.mnem/` 的场景 | 使用 `mnem global` 的场景 |
|------------------------|----------------------|
| 项目专属的事实、决策、代码上下文 | 跨所有项目的人员、偏好和事实 |
| 随仓库一同传递的单仓库记忆 | 希望每个会话和每个 Agent 都能看到的知识 |
| 任何你会与代码一同提交的内容 | 跨会话的连续性 |

`mnem integrate` 命令会将 Agent 配置为优先读取本地图谱，并在需要时自动回退到全局图谱，正常使用时无需手动切换。

</details>

---

### 6. `mnem status` / `mnem log` - 查看历史记录

查看图谱的当前状态并逆序遍历操作日志。

```bash
mnem status    # 显示 op-head CID、最新提交、所有命名引用及标签计数
mnem log       # 逆序遍历操作日志，显示最近 20 条
```

<details>
<summary>更多选项</summary>

```bash
mnem stats              # 紧凑单行摘要：CID 数、引用数、标签名
mnem log -n 50          # 显示最近 50 条
mnem log --oneline      # 每条操作一行的紧凑格式
mnem log --format json  # 机器可读的 JSON 流
```

</details>

---

### 7. `mnem diff` / `mnem show` - 对比快照与检查块

精确查看任意两个操作 CID 之间的变化：引用差异加上节点/边的结构 diff。按 CID 解码任意块进行详细取证。

```bash
mnem log          # 列出提交及其 CID - 从这里复制 CID 以供下方使用
mnem diff HEAD <cid>
```

> **示例：** 一个 Agent 连夜运行并提交了数百个新事实。合并进 `main` 之前，审查者将 `HEAD` 与运行前快照进行 diff，确认没有意外添加或删除任何内容。

<details>
<summary>更多选项</summary>

```bash
mnem diff <op-a-cid> <op-b-cid>   # 对比任意两个操作

mnem show               # 解码并格式化显示当前 op-head 块
mnem show <cid>         # 按 CID 解码任意块（Node、Edge、Commit、Operation 等）

mnem cat-file <cid>                # 将任意块的原始 DAG-CBOR 字节输出到 stdout
mnem cat-file <cid> --json         # 解码为 DAG-JSON 并格式化输出（可管道传给 jq）
```

</details>

---

### 8. `mnem branch` - 创建和管理分支

像分支代码一样分支知识图谱。每个分支都是一条独立的提交线，自由实验，准备好后合并回来。

```bash
mnem branch create agentic-workflow
```

> **示例：** 两个 Agent 正在测试摘要流水线的两种竞争方案。每个在自己的分支上工作，`approach-a` 和 `approach-b`，随时提交发现。审查者将获胜方案的分支合并回 `main`，保留两次实验的完整历史。

<details>
<summary>更多选项</summary>

```bash
mnem branch list                        # 列出所有分支；* 标记当前分支
mnem branch create <name> <start>       # 从引用、分支名或 CID 创建分支
mnem branch create <name> --from HEAD   # 显式 --from 形式；解析规则与上方相同
mnem branch delete <name>               # 删除本地分支指针
```

</details>

---

### 9. `mnem merge` - 合并分支

3-way 图谱合并，与 `git merge` 相同的模型，但操作对象是知识。冲突写入 `.mnem/MERGE_CONFLICTS.json` 供明确解决。

```bash
mnem merge agentic-workflow
```

> **示例：** Agent A 花了一周处理客户访谈；Agent B 并行处理支持工单。合并将两个知识库干净地融合，没有任何事实被静默覆盖，每个节点的完整来源均得以保留。

<details>
<summary>更多选项</summary>

```bash
mnem merge <branch> --strategy=ours     # 自动解决冲突：保留当前侧
mnem merge <branch> --strategy=theirs   # 自动解决冲突：采用传入侧
mnem merge <branch> --dry-run           # 预览结果，不持久化任何内容
mnem merge --continue                   # 编辑 MERGE_CONFLICTS.json 后继续完成合并
mnem merge --abort                      # 取消合并；从 ORIG_HEAD 还原 HEAD
```

</details>

---

### 10. `mnem push` / `mnem pull` / `mnem clone` - 与远端同步

像推拉代码一样推拉知识图谱。传输格式为标准 CAR v1（Content Addressed aRchive，一种 IPFS 兼容的二进制格式）。

> **首次推送前**，注册一个远端：`mnem remote add origin <url>`，其中 `<url>` 是你的服务器地址，例如 `http://my-server:9876` 或 `https://mnem.example.com`（完整远端命令列表见下方更多选项）。
>
> **自建服务器？** 在目标机器上设置 `MNEM_HTTP_ALLOW_NON_LOOPBACK=1` 并运行 `mnem http --bind 0.0.0.0:9876`。驱动 CLI 的同一二进制文件也提供 HTTP 服务，无需单独安装或守护进程。然后在客户端执行 `mnem remote add origin http://<server-ip>:9876`。
>
> **身份验证（bearer token）：** 默认情况下，`mnem http` 没有身份验证。要保护 push/pull，服务端和客户端必须使用相同的令牌：
> ```bash
> # 服务端：设置令牌并启动服务器
> export MNEM_HTTP_AUTH_TOKEN=my-secret-token
> MNEM_HTTP_ALLOW_NON_LOOPBACK=1 mnem http --bind 0.0.0.0:9876
>
> # 客户端：注册指向持有相同令牌环境变量的远端
> export MNEM_REMOTE_ORIGIN_TOKEN=my-secret-token
> mnem remote add origin http://my-server:9876 --token-env MNEM_REMOTE_ORIGIN_TOKEN
> mnem push   # 令牌错误或缺失时以"authentication failed (HTTP 401)"退出
> ```
> 令牌错误或缺失时，服务器返回 HTTP 401 拒绝请求。永远不要在命令中硬编码令牌值，使用环境变量。如果推送时 `MNEM_REMOTE_ORIGIN_TOKEN` 未设置或为空，`mnem push` 在发起任何网络请求前就以"missing authentication token"退出。

```bash
mnem push          # 将 HEAD 推送到 origin/main
mnem pull          # 将 origin/main 快进合并到 HEAD
```

> **单写者说明：** `mnem push` 和 `mnem pull` 会获取本地存储的写锁。如果 `mnem http` 正在对同一存储运行，推送将阻塞直到服务器释放任何进行中的写操作（不会损坏数据，但会等待）。如果需要干净地推送而不等待，请先停止 `mnem http`。对于写入同一远端的并发 CI 流水线，使用外部队列或通过 `mnem merge` 合并的独立仓库。

> **示例：** 在 CI 中运行的 Agent 在每次构建后提交新发现并推送。开发者机器上的 Agent 在会话开始时拉取，整个团队无需任何手动同步即可从相同的知识基线工作。

<details>
<summary>更多选项</summary>

```bash
mnem push <remote> <branch>             # 推送指定分支
mnem pull <remote> <branch>             # 从指定远端/分支拉取

mnem fetch                              # 仅抓取，不合并（使用默认远端）
mnem fetch <remote>                     # 从指定远端抓取

mnem clone <url> [<dir>]                # 将 CAR 归档克隆到 <dir>
mnem clone file:///tmp/repo.car ./copy  # 从本地文件路径克隆
mnem clone ./repo.car ./copy            # 裸路径简写（必须以 .car 结尾）

mnem remote add <name> <url>                         # 注册远端
mnem remote add <name> <url> \
  --token-env MNEM_REMOTE_ORIGIN_TOKEN               # 通过环境变量提供 bearer token
mnem remote list                                     # 列出所有已配置的远端
mnem remote show <name>                              # 显示 URL 和功能信息
mnem remote remove <name>                            # 移除远端条目
```

</details>

---

### 11. `mnem query` - 结构化图查询

带可选边遍历的精确属性过滤。无需计算嵌入，快速且具有确定性。

```bash
mnem query --where name=Alice
```

> **示例：** 一个 Agent 从入职文档中构建组织架构图。之后，另一个 Agent 运行 `mnem query --where kind=Person --with-outgoing reports_to` 重建完整的汇报结构，无需文本搜索。

<details>
<summary>更多选项</summary>

```bash
mnem query --where kind=Person -n 25             # 增大结果数量上限
mnem query --where kind=Person \
  --with-outgoing knows                          # 跟踪出向 "knows" 边
mnem query --where status=active \
  --with-outgoing depends_on \
  --with-outgoing depends_on                     # 链接多跳遍历

mnem blame <node-uuid>                           # 列出某节点的所有入向边
mnem blame <node-uuid> --etype authored          # 按边类型过滤
mnem blame <node-uuid> --first-writer            # 显示每条边最早的祖先提交（BFS）

# mnem ref：管理命名引用（通过 CID 管理分支/标签）
mnem ref list                         # 列出所有引用（refs/heads/*、refs/remotes/* 等）
mnem ref set <name> <target-cid>      # 将引用指向特定提交 CID
mnem ref delete <name>                # 删除命名引用
```

</details>

---

### 12. `mnem reindex` - 管理向量嵌入

为节点补充或更新向量嵌入。在添加新的嵌入提供商或切换模型后运行。

> **`mnem http` 运行时执行 `mnem reindex`？** `mnem reindex` 是写操作，会获取单写者锁，因此会等待任何进行中的写操作完成后再开始。进行中的 HTTP 读操作（`mnem retrieve`）在重建索引期间仍可继续工作，但可能看到旧嵌入，直到重建索引提交落地。如果需要一致的时间点快照，请先停止 HTTP 服务器。

```bash
mnem reindex
```

<details>
<summary>更多选项</summary>

```bash
mnem reindex --label Doc              # 仅处理指定标签的节点
mnem reindex --since <commit>         # 仅处理 <commit> 之后添加或修改的节点
mnem reindex --force                  # 重新嵌入已建索引的节点
mnem reindex --dry-run                # 统计将被嵌入的数量，不实际调用提供商

mnem embed --force                    # 重新嵌入已建索引的节点
mnem embed --label Person             # 仅处理该标签的节点
```

</details>

---

### 13. `mnem export` / `mnem import` - 备份与还原

将任意快照导出为标准 CAR v1 归档文件。可在任意机器、任意平台上导入。

```bash
mnem export backup.car
```

> **示例：** 在大批量摄入之前，导出当前快照。如果摄入产生了意外结果，导入快照以还原到精确的之前状态。

<details>
<summary>更多选项</summary>

```bash
mnem export -                              # 将 CAR 写入 stdout（可通过 SSH 管道传输）
mnem export --from refs/heads/main out.car # 从指定引用导出
mnem export --from <cid> backup.car        # 从指定提交 CID 导出

mnem import <path>                         # 将 CAR 归档导入当前仓库
mnem import -                              # 从 stdin 读取 CAR
```

</details>

---

### 14. `mnem config` - 配置 mnem

设置作者身份、嵌入提供商和 API 端点。API 密钥存储在环境变量中，不写入磁盘。

```bash
mnem config set user.name "ci-agent"
mnem config set embed.provider ollama
```

<details>
<summary>所有配置键</summary>

```bash
mnem config set user.email agent@example.com
mnem config set embed.model nomic-embed-text
mnem config set embed.base_url http://localhost:11434
mnem config get embed.provider
mnem config unset embed.provider
mnem config list
```

已知配置键：`user.name`、`user.email`、`user.key`、`user.agent_id`、`embed.provider`、`embed.model`、`embed.api_key_env`、`embed.base_url`。

</details>

---

### 15. `mnem mcp` / `mnem http` - 暴露图服务

将 mnem 以 MCP 服务器（stdio，供 agent 宿主使用）或 HTTP JSON API（供直接调用的服务使用）形式对外提供。

> **注意：** 你很少需要直接运行 `mnem mcp`。如果你使用了 `mnem integrate`，你的 AI 宿主（Claude Code、Cursor 等）会在需要 mnem 工具时自动在后台启动它。当你希望通过 HTTP 从服务或脚本调用 mnem 时，请使用 `mnem http`。

```bash
mnem mcp                 # 以 stdio 启动 MCP JSON-RPC 服务器
mnem http                # 启动 HTTP JSON API（默认仅监听本机）
```

> `mnem http` 在前台运行；按 Ctrl+C 停止。如需持久后台服务器，请使用操作系统进程管理器（Linux/macOS 上如 `nohup mnem http &`，或 Windows 服务包装器）。

> **并发：** `mnem http` 支持任意数量的并发读者，但每次只允许一个写者（单写者锁）。如果需要在 `mnem http` 运行时执行 `mnem reindex`，请参阅[不适合使用 mnem 的场景](#不适合使用-mnem-的场景)了解行为细节。在 HTTP 服务器运行期间执行 `mnem push`/`mnem pull`，请先停止服务器或通过外部队列协调。

> **示例：** 一个后端服务在启动时运行 `mnem http`。集群中的每个 agent 调用同一个 HTTP 端点，共享知识，无需各实例维护本地状态。

<details>
<summary>更多选项</summary>

```bash
mnem mcp --repo ~/notes            # 将 MCP 服务器指向特定图

# HTTP 绑定与网络
mnem http --bind 127.0.0.1:9876    # 默认本机回环绑定
mnem http --bind 0.0.0.0:9876      # 在所有接口上暴露（需设置 MNEM_HTTP_ALLOW_NON_LOOPBACK=1）
mnem http --in-memory              # 内存临时存储（无需 .mnem/）
mnem http --metrics                # 强制开启 /metrics 端点
mnem http --no-metrics             # 强制关闭 /metrics 端点

mnem repos list                    # 列出所有通过 mnem integrate 注册的仓库
mnem repos set-default <path>      # 将某仓库设为默认（无需 -R）
mnem repos prune                   # 移除不再存在路径的注册条目
```

</details>

---

### 16. `mnem completions` - Shell 补全

为你的 shell 生成并安装 Tab 补全。

```bash
# bash（如目录不存在请先创建）：
mkdir -p ~/.local/share/bash-completion/completions
mnem completions bash > ~/.local/share/bash-completion/completions/mnem

# zsh（先创建目录；并在 ~/.zshrc 中添加 fpath 条目）：
mkdir -p ~/.zsh/completions
mnem completions zsh > ~/.zsh/completions/_mnem
# 如 ~/.zshrc 中尚未包含以下内容，请添加：
#   fpath=(~/.zsh/completions $fpath); autoload -Uz compinit && compinit
```

<details>
<summary>所有 Shell</summary>

```bash
mnem completions bash
mnem completions zsh
mnem completions fish
mnem completions powershell
mnem completions elvish
```

</details>

---

### 全局标志：`-R <path>`

将任意命令重定向至指定仓库目录，绕过从当前目录向上查找的逻辑。

```bash
mnem -R ~/notes status
mnem -R ~/notes log
mnem -R ~/notes retrieve "query"
```

<hr>

## MCP 工具

通过 `mnem integrate` 接入后，agent 将获得 **22 个原生 MCP 工具**，均以 `mnem_` 为前缀（21 个稳定 + 1 个功能门控）。每个响应均携带包含 `bytes`、`latency_micros` 和 `tokens_estimate` 的 `_meta` 字段，供调用方推算自身成本。写操作会将 `agent_id` 和 `task_id` 写入提交元数据，使溯源始终可查。

> **从这里开始：** 你的 agent 大部分时间会用到 `mnem_retrieve` 和 `mnem_commit`。以下表格为完整参考，你无需单独配置每个工具。

启动服务器：`mnem mcp --repo <path>`（或让 `mnem integrate` 自动连接）。

完整参考：[`docs/src/mcp.md`](docs/src/mcp.md)。

### 内省

| 工具 | 描述 |
|------|------|
| `mnem_stats` | 仓库概览：op-head、最新提交、ref 摘要、已知标签。开销低；建议 agent 首次接触新图时优先调用。 |
| `mnem_schema` | 检查当前提交中的节点标签和边谓词。在编写查询或遍历前调用，以发现图中的内容。 |
| `mnem_list_nodes` | 枚举当前 head 的节点，可按标签过滤。每个节点返回 UUID + 标签 + 摘要。 |
| `mnem_list_tags` | 列出仓库中所有具名标签（`refs/tags/*`）。 |
| `mnem_recent` | 从 HEAD 向后遍历 op-log。返回最近 N 个操作，包含时间、作者、`agent_id`、`task_id` 和消息。 |

### 检索

| 工具 | 描述 |
|------|------|
| `mnem_retrieve` | **主要检索工具。** 混合语义 + 稀疏 + 图搜索。返回预渲染为文本的节点，以及 `tokens_used` / `dropped` / `candidates_seen` 元数据。支持图展开、社区过滤、PPR 和交叉编码器重排序。 |
| `mnem_global_retrieve` | 与 `mnem_retrieve` 相同，但始终针对全局图（`~/.mnemglobal/.mnem/`）。用于跨项目、跨会话记忆。 |
| `mnem_search` | 精确属性匹配，可选边遍历。速度快，确定性强，无需嵌入。 |
| `mnem_vector_search` | 对存储的节点嵌入进行原始余弦相似度最近邻搜索。传入模型名称和查询向量，返回 top-k 匹配结果。 |
| `mnem_get_node` | 通过 UUID 获取单个节点。返回完整属性、内容大小和出边数量。 |
| `mnem_traverse` | 从起始节点出发，列出通过指定边标签可到达的出邻居。 |
| `mnem_incoming_edges` | 列出所有指向某节点的边（反向查找）。等价于 CLI 中的 `mnem blame`。 |

### 写操作

| 工具 | 描述 |
|------|------|
| `mnem_commit` | 将节点和/或边作为单次提交添加。返回新 op-id、提交 CID 和已创建节点的 UUID。 |
| `mnem_commit_relation` | 复合写操作：解析或创建一个主语节点，解析或创建一个宾语节点，并用有类型的边连接它们，一次调用完成。避免重复实体问题（见下方示例）。 |
| `mnem_resolve_or_create` | 通过主键属性查找或创建节点。若存在匹配的 `(label, anchor-property) == value`，则返回其 UUID；否则提交新节点。 |
| `mnem_ingest` | 将文件路径或内联文本摄入为 `Doc + Chunk + Entity` 子图。接受 `{path: "notes.md"}` 或 `{text: "...", source: "label"}`。分块选项：`auto`、`paragraph`、`recursive`、`sentence_recursive`、`session`、`structural`。 |
| `mnem_global_ingest` | 与 `mnem_ingest` 相同，但写入全局图。适用于需要跨所有会话和项目查询的文档。 |
| `mnem_global_add` | 直接向全局图写入节点和/或边。适用于跨多个项目出现的共享实体（人物、组织、约定）。 |

`mnem_commit_relation` 示例 - 一次调用关联两个实体：

```json
{
  "subject": "Alice",
  "subject_kind": "Entity:Person",
  "predicate": "works_at",
  "object": "Globex",
  "object_kind": "Entity:Organization",
  "agent_id": "onboarding-agent"
}
```

### 删除

| 工具 | 描述 |
|------|------|
| `mnem_tombstone_node` | 软删除：将节点标记为已遗忘。默认从检索结果中隐藏，但节点 CID 和所有先前提交保持完整以供审计。当用户说"忘掉 X"或撤销同意时使用。 |
| `mnem_global_tombstone_node` | 与 `mnem_tombstone_node` 相同，但操作全局图。 |
| `mnem_delete_node` | 硬删除：从当前 head 提交中移除节点。引用该节点的先前提交仍可寻址。仅在目标是释放存储而非内存清理时使用。 |

### 可选（功能门控）

| 工具 | 描述 |
|------|------|
| `mnem_community_summarize` | 对调用方提供的节点 UUID 集合进行抽取式 Centroid + MMR（最大边际相关性，促进多样性的选择）摘要。无 LLM 调用 - 在接近社区质心与多样性之间平衡，选取 k 个句子。通过 `summarize` cargo 功能启用。 |

<hr>

## Python API (mnem-py) - 密集向量检索（v0.1.0）

> **包名说明：** `pip install mnem-py`（PyPI 包名）· `import pymnem`（Python 导入名）。这是同一个库的两个不同名称。`mnem-cli`（CLI 工具）和 `mnem-py`（本 Python 库）是独立的包。

当你希望直接从 Python（3.8+）读写 mnem 图时，可使用 `mnem-py`，无需 CLI 二进制文件。相同的检索引擎，无需 Rust 工具链（Linux、macOS 和 Windows 均提供预构建 wheel）。

> **v0.1.0 功能范围：** `mnem-py` 目前仅支持**密集向量检索**。关键词搜索（BM25/SPLADE）和图遍历（`--graph-expand`、`--graph-mode ppr`）尚不支持从 Python 调用。如需这些功能，请使用 CLI（`mnem retrieve "..."`）或 HTTP API（`mnem http`），两者均可与 `mnem-py` 写入的同一磁盘图配合使用。

```bash
pip install mnem-py
pip install sentence-transformers   # 可选 - 或使用 OpenAI、Cohere 等提供的嵌入
```

`mnem-py` 通过**密集向量**进行存储和检索：你在 Python 中计算嵌入并传给 mnem。

> [!WARNING]
> 检索时的 `MODEL_NAME` 必须与摄入时的 `MODEL_NAME` 一致。**不匹配时会静默返回零结果**，不会抛出异常。`add_embedding_f32` 必须紧接其配对的 `add_node` 调用之后；在 `add_node` 之前调用会报错。
> **恢复模型不匹配：** 在 `.mnem/config.toml` 中配置正确的模型后，从 CLI 运行 `mnem reindex`（或 `mnem reindex --label <label>`）- 这会为所有匹配节点重建嵌入，不改变节点内容。

```python
import pymnem
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # 下载一次，约 23 MB
MODEL_NAME = "all-MiniLM-L6-v2"                    # 摄入和检索时必须一致

# open_or_init：如果 "my-graph/" 中不存在 .mnem/，则创建（无需 `mnem init`）
# 这替代了 CLI 快速入门中的 `mnem init` 步骤 - 你无需单独运行 mnem init。
# 路径陷阱："my-graph/" 是相对于 Python 运行时的工作目录的。
# 从不同目录运行此脚本会打开（或创建）不同的图。
# 使用绝对路径以避免此问题：pathlib.Path.home() / "my-graph"
# init_memory()：仅限内存 - 进程退出时数据丢失；适合测试
repo = pymnem.Repo.open_or_init("my-graph/")

# transaction(author, message)：两者均为必填字符串；author 标记谁写入了提交，message 是备注
with repo.transaction(author="agent", message="seed") as tx:
    for text in ["Alice lives in Berlin", "Bob moved to Paris"]:
        tx.add_node(ntype="Memory", summary=text)  # ntype 对应 CLI 中的 --label（如 mnem retrieve --label Memory）
        tx.add_embedding_f32(MODEL_NAME, model.encode(text).tolist())  # 必须紧接 add_node 之后

# token_budget：返回摘要的近似 token 数上限（达到上限时 mnem 停止添加结果）
# result 是 RetrieveResult - 可迭代，同时具有 .tokens_used / .tokens_budget 属性
query_vec = model.encode("Alice Berlin").tolist()
result = repo.retrieve(vector=query_vec, model=MODEL_NAME, token_budget=500, limit=5)
for item in result:
    print(f"{item.score:.3f}  {item.summary}")
print(f"tokens_used={result.tokens_used}  tokens_budget={result.tokens_budget}")  # 无静默截断
```

> **任何嵌入模型均可使用。** 将 `SentenceTransformer("all-MiniLM-L6-v2")` 替换为任何返回固定长度浮点列表的模型，并在摄入和检索时使用相同的 `MODEL_NAME` 字符串。例如使用 OpenAI：`vec = openai.OpenAI().embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding`，将 `MODEL_NAME = "text-embedding-3-small"`。Cohere 和任何本地 HuggingFace 模型的用法相同。

完整 API 接口 - `query`、`update_node`、`delete_node`、磁盘持久化、标签过滤：[`crates/mnem-py/README.md`](crates/mnem-py/README.md) 或[在 GitHub 上查看](https://github.com/Uranid/mnem/tree/main/crates/mnem-py)。

<hr>

## GraphRAG（进阶）

GraphRAG 在向量搜索基础上扩展了图遍历：沿边追溯到相关节点，按社区聚类，按图距离评分。每个阶段一个标志，按查询按需开启。向量搜索单独即可处理大多数查询；对于多文档或多跳问题，请启用图阶段。

### 阶段与标志

| 阶段 | 标志 | 作用 |
|------|------|------|
| **向量通道** | 始终开启 | 基于每次提交的密集嵌入的近似最近邻索引。通过 `config.toml` 配置模型。 |
| **稀疏通道** | 配置驱动 | BM25 + SPLADE 关键词评分，与向量结果融合。通过 `config.toml` 中的 `[sparse]` 块启用。 |
| **向量候选池** | `--vector-cap <N>` | 将密集池大小从默认的 256 提升。越高 = 长尾召回越好，成本越高。 |
| **结果数量** | `--limit <N>` | 最终返回集合（默认无限制）。简写：`-n`。 |
| **图展开** | `--graph-expand <N>` | 通过已标记的边添加 top-K 种子的 N 个邻居。图开启时推荐的审计默认值为 `20`。 |
| **图模式** | `--graph-mode <decay\|ppr>` | `decay`（默认）按跳数距离加权。`ppr` 使用个性化 PageRank；多跳召回更好，成本更高。 |
| **社区过滤** | `--community-filter` | 对内容聚类；融合前丢弃覆盖率低的簇。 |
| **KeyBERT 提取** | `mnem ingest --extractor keybert` | 摄入时进行关键短语提取；增强稀疏和社区信号。 |
| **摘要生成** | `--summarize` | 对 top-K 进行 Centroid + MMR 摘要，具备多样性。 |
| **交叉编码器重排序** | `--rerank <provider:model>` | 融合后重新排序。支持 `cohere:rerank-english-v3.0`、`voyage:rerank-1`、本地模型。 |

### 快速示例

```bash
# 纯密集向量基准
mnem retrieve "what does this project do"

# 添加多跳图遍历
mnem retrieve "..." --graph-expand 20

# 完整堆栈：图展开 + 社区过滤 + PPR + 重排序
mnem retrieve "..." --graph-expand 20 --community-filter --graph-mode ppr --rerank cohere:rerank-english-v3.0

# 在顶部叠加交叉编码器重排序器
mnem retrieve "..." --graph-expand 20 --community-filter --rerank cohere:rerank-english-v3.0

# 使用 KeyBERT 关键短语丰富摄入（增强稀疏 + 社区信号）
mnem ingest --extractor keybert notes.md
```

### 何时启用

- **单文档语料库、简单查询**：关闭图，仅向量搜索即可
- **多跳/组合问题**：`--graph-expand 20`
- **具有跨文档引用的长历史**：添加 `--community-filter`
- **需要召回上限**：在顶部叠加 `--rerank`
- **关键短语丰富摄入**：摄入时使用 `mnem ingest --extractor keybert`

完整检索架构：[`docs/src/cli.md`](docs/src/cli.md)（retrieve 标志）

<hr>

## 与其他工具的比较

<sup>✅ 完全支持 &nbsp;·&nbsp; ~ 部分或有限支持 &nbsp;·&nbsp; ✗ 不支持 &nbsp;·&nbsp; n/a 不适用</sup>

|  | **mnem** | **mem0** | **Graphiti** | **Letta** | **Supermemory** | **MemPalace** | **Cognee** |
|--|:--------:|:--------:|:------------:|:---------:|:---------------:|:-------------:|:----------:|
| 本地优先/离线 | ✅ | ~ | ✗ | ~ | ✗ | ✅ | ~ |
| 版本化历史 | ✅ | ✗ | ✗ | ~ | ✗ | ✗ | ✗ |
| 分支与合并 | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 内容寻址存储 *（相同内容始终获得相同 ID；自动去重相同事实）* | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| WASM / 边缘可部署 | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 无 API 摄入 | ✅ | ~ | ✗ | ✗ | ✗ | ✅ | ~ |
| Token 预算透明度 | ✅ | ✗ | ✗ | ~ | ✗ | ✗ | ✗ |
| 单一二进制，无守护进程 | ✅ | ✗ | ✗ | ✗ | n/a | ✗ | ✗ |
| 无需外部数据库 | ✅ | ~ | ✗ | ✗ | n/a | ✗ | ~ |
| 知识图谱 | ✅ | ~ | ✅ | ✗ | ✗ | ~ | ✅ |
| 混合检索（向量 + 稀疏 + 图） | ✅ | ~ | ✅ | ~ | ~ | ~ | ~ |
| MCP 原生 | ✅ | ~ | ~ | ✅ | ✅ | ✅ | ~ |
| 开源 | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT | MIT | Apache-2.0 |

<sup>~ = 部分或有限支持 &nbsp;·&nbsp; mem0 知识图谱：v1.1+ 新增图记忆（Neo4j 或内存） &nbsp;·&nbsp; Graphiti 需要 Neo4j、FalkorDB 或 Kuzu（Kuzu 可嵌入）；摄入需要 LLM API 密钥 &nbsp;·&nbsp; Letta 本地默认使用 SQLite；生产环境需要 PostgreSQL &nbsp;·&nbsp; MemPalace 需要 ChromaDB &nbsp;·&nbsp; Supermemory 企业自托管需要 Cloudflare + Postgres + OpenAI &nbsp;·&nbsp; Cognee 使用 Kuzu + LanceDB（均可嵌入）；图提取需要 LLM API 密钥 &nbsp;·&nbsp; 最后验证：2026 年 5 月</sup>

深度对比：

- [mnem vs mem0](docs/src/comparisons/mem0.md) - agent 记忆层，OSS 领导者
- [mnem vs MemPalace](docs/src/comparisons/mempalace.md) - 基准对标
- [mnem vs Graphiti](docs/src/comparisons/graphify.md) - AI 编码助手知识图谱工具
- [mnem vs Letta](docs/src/comparisons/letta.md) - agent 记忆框架（原 MemGPT）
- [mnem vs Supermemory](docs/src/comparisons/supermemory.md) - 云托管记忆服务
- [mnem vs Cognee](docs/src/comparisons/cognee.md) - 面向 agent 的 KG 替代方案

完整矩阵：[`docs/src/comparisons/README.md`](docs/src/comparisons/README.md)。

<hr>

## 不适合使用 mnem 的场景

> **v0.1.0 成熟度说明：** mnem 处于 1.0 之前阶段。CLI 命令、MCP 工具名称和 Python 绑定在 v0.1.x 中保持稳定；磁盘存储格式向前兼容。次要版本之间可能发生破坏性变更 - 在生产环境升级前请查看 [CHANGELOG](CHANGELOG.md)。

- **你需要事务性 OLTP**（在线事务处理 - 为高容量行级 INSERT/UPDATE/DELETE 设计的数据库，如支付账本或库存系统）。mnem 是仅追加的版本化历史；行级 UPDATE/DELETE 语义不是其模型。
- **你需要 10k+ QPS 下 50 ms 以内的云规模检索**（每秒查询数）。mnem 是本地优先的。多区域分片检索在路线图上，v1 中尚未提供。
- **你需要并发多写者访问。** redb 存储是单写者的（ACID = 原子性、一致性、隔离性、持久性；通过写时复制 B 树实现崩溃安全）- 每次一个写者，多个并发读者。两个并发写者不会损坏数据（第二个写操作会阻塞直到第一个释放锁），但也不会自动合并。并发 agent 写操作需要外部队列或通过 `mnem merge` 合并的独立仓库。

> 寻找托管记忆、多区域副本、跨团队共享图或托管远程层？为 mnem 提供这些功能的兄弟项目正在积极开发中 - 敬请关注。

<hr>

## Crates

| Crate | 角色 |
|-------|------|
| [`mnem-cli`](crates/mnem-cli) | `mnem` 二进制 - 一个命令搞定一切 |
| [`mnem-core`](crates/mnem-core) | 图模型、检索、索引、辅助程序 |
| [`mnem-http`](crates/mnem-http) | HTTP JSON 服务器 |
| [`mnem-mcp`](crates/mnem-mcp) | MCP 服务器（stdio） |
| [`mnem-py`](crates/mnem-py) | PyO3 Python 绑定 |
| [`mnem-embed-providers`](crates/mnem-embed-providers) | ONNX 内置、Ollama、OpenAI、Cohere |
| [`mnem-sparse-providers`](crates/mnem-sparse-providers) | BM25、SPLADE-onnx |
| [`mnem-rerank-providers`](crates/mnem-rerank-providers) | Cohere、Voyage |
| [`mnem-llm-providers`](crates/mnem-llm-providers) | OpenAI、Anthropic、Ollama |
| [`mnem-ingest`](crates/mnem-ingest) | 解析 + 分块 + 提取流水线 |
| [`mnem-extract`](crates/mnem-extract) | 实体提取（KeyBERT、统计 NER） |
| [`mnem-ner-providers`](crates/mnem-ner-providers) | NER 提供商 trait + 内置提供商（`RuleNer`、`NullNer`） |
| [`mnem-bench`](crates/mnem-bench) | 基准测试工具（LongMemEval、LoCoMo 等） |
| [`mnem-graphrag`](crates/mnem-graphrag) | 社区摘要、Centroid + MMR |
| [`mnem-ann`](crates/mnem-ann) | HNSW 包装器 |
| [`mnem-backend-redb`](crates/mnem-backend-redb) | redb 后端存储 |
| [`mnem-transport`](crates/mnem-transport) | CAR 编解码器 + 远程帧 |

<hr>

## 文档

- [快速入门](docs/src/quickstart.md) - 五分钟演练
- [安装](docs/src/install.md) - 各平台安装矩阵
- [CLI 参考](docs/src/cli.md) - 每个子命令和标志
- [MCP 服务器](docs/src/mcp.md) - 暴露的工具、客户端连接
- [核心概念](docs/src/core-concepts.md) - CID、提交、标签
- [配置](docs/src/configuration.md) - 环境变量、config.toml
- [基准测试方法论](docs/src/benchmarks/methodology.md)
- [复现基准测试](docs/src/benchmarks/reproduce.md)
- [嵌入提供商](docs/src/guides/embed-providers.md)
- [迁移](docs/src/migrations/)
- [GitHub Issues](https://github.com/Uranid/mnem/issues) - 问题、Bug 报告、功能请求

<hr>

## 贡献

欢迎提交 Issue 和 PR。本地构建与测试：

```bash
cargo build --features bundled-embedder
cargo test
```

- [`CONTRIBUTING.md`](CONTRIBUTING.md) - 分支规范、代码审查礼仪、如何提交 PR
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) - 行为准则（Contributor Covenant 2.1）
- [`SECURITY.md`](SECURITY.md) - 漏洞披露政策

## 许可证

[Apache-2.0](LICENSE)。第三方归属见 [`NOTICE`](NOTICE)。

<hr>

⭐ **觉得 mnem 有用？** 一个 Star 是来自满意的构建者最有力的信号，它能帮助下一位在记忆问题上苦苦挣扎的 Agent 开发者找到这个仓库。我们认真阅读每一个 Issue、每一个 PR、每一条提及。告诉我们你用它构建了什么。
