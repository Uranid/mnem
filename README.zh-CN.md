<div align="center">

<img src="assets/logo/mnem-banner.svg" alt="mnem: Git for AI Memory" />

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

> [!NOTE]
> 此文档为社区翻译版本，最新内容以 [English README](README.md) 为准。

1. [问题所在](#问题所在)
2. [基准测试](#基准测试)
3. [安装](#安装)
4. [快速入门](#快速入门)
5. [接入](#mnem-integrate---接入任何-agent-宿主)
6. [它是什么](#它是什么)
7. [命令](#命令)
8. [Python API (mnem-py)](#python-api-mnem-py)
9. [GraphRAG](#graphrag)
10. [与其他工具对比](#与其他工具对比)
11. [文档](#文档)
12. [贡献](#贡献)

<hr>

## 问题所在

> **转录记录不是记忆。**

你对此已有一个心智模型：git。提交带有可以 diff 和回滚的历史，可以合并的分支，以及每一个决策及其原因的日志。而你的 Agent 记忆却什么都没有。它只是把转录内容粘贴回提示词，或者是一个你无法检查或编辑的搜索索引。约定规则存放在扁平的 `.cursorrules` 文件里 - 有用，但既不可查询也不可版本化。而且会话是相互隔离的：今天和 Claude Code 一起规划一次迁移，明天打开 Cursor，那个 agent 对此一无所知。

mnem 将 git 模型引入 Agent 知识管理。每一次写入都是一个**基于内容寻址的提交** - 相同的字节，相同的 CID，适用于任何机器。技能、决策和笔记存储在一个**可版本化、可分支、可合并的知识图谱**中：`diff` 两个会话之间的变化，`revert` 一批错误的事实，像合并分支一样 `merge` 来自两个 agent 的知识。

检索是**混合且透明的**：向量检索 + 关键词检索 + 图遍历一次完成，并配有明确的 token 预算 - mnem 会精确报告找到了什么、跳过了什么，以及消耗了多少 token。**只需修改一行配置，即可替换嵌入模型、重排序器或 LLM。** 一条 `mnem integrate` 命令即可将其接入 Claude Code、Cursor、Codex、Gemini CLI 或任何 MCP 宿主。单个约 40 MB 的二进制文件，无需守护进程，无需云服务，无需 API 密钥。

> 合上笔记本。明天再打开。你的 Agent 还记得。

<hr>

## 基准测试

**在六个公开数据集上与 mem0 和 MemPalace 进行了正面对比测试。mnem 在所有数据集上均领先。**

使用 ONNX MiniLM-L6-v2 嵌入模型，每个系统上的字节完全相同，不使用 LLM 重排序。复现方法：`bash benchmarks/harness/run_bench.sh`。

<div align="center"><img src="assets/benchmarks/benchmarks.svg" alt="mnem public benchmarks" /></div>

<sup>mem0 列：我们在相同测试框架下的复现结果（mem0 未在这些数据集上发布 R@K 指标）。MemPalace 列：公开的头条数字，已在我们的测试框架下交叉验证。原始产物：[`benchmarks/results/v0.1.0/`](benchmarks/results/v0.1.0/)。† FinanceBench 在所有系统上均使用 Ollama bge-large（1024 维）；MemPalace 展示的是最佳配置下的结果（bge-large 直连 ChromaDB）；mem0 在存储前对记忆应用了 LLM 提取。完整方法论：[`benchmarks/results/analysis/financebench.md`](benchmarks/results/analysis/financebench.md)。</sup>

### 查询速度

<div align="center"><img src="assets/benchmarks/query-speed.svg" alt="mnem query speed" /></div>

<details>
<summary><b>复现方法</b></summary>

```bash
mnem bench fetch longmemeval     # download datasets (one-time, 264 MB)
mnem bench                       # TUI; select benchmarks interactively
mnem bench run --benches longmemeval --limit 50 --non-interactive
mnem bench results ./bench-out   # re-render results from a prior run

# Legacy bash harness (canonical path for headline numbers)
bash benchmarks/harness/run_bench.sh
```

方法论、原始产物、各基准测试详细分类：[`benchmarks/`](benchmarks/) 和 [`docs/src/benchmarks/`](docs/src/benchmarks/)。

</details>

<hr>

## 安装

**选择你已有的任意一种方式，任何一种都可以。** 各平台的完整说明见下文。

```bash
# if you have Cargo (Rust): recommended for dev machines
cargo install --locked mnem-cli --features bundled-embedder

# if you have pip (Python)
pip install mnem-cli

# if you have npm (Node.js)
npm install -g mnem-cli
```

```bash
mnem --version    # confirm install
```

> [!NOTE]
> `--features bundled-embedder` 会内置一个进程内 ONNX MiniLM-L6-v2 模型，使 `mnem retrieve` 无需任何配置即可使用。如果你想通过 `.mnem/config.toml` 使用自己的嵌入器（Ollama、OpenAI、Cohere），可以省略该标志。

<details>
<summary><b>macOS / Linux</b></summary>

没有 Cargo？[通过 rustup 安装](https://rustup.rs/)（同时会安装 `rustc`）。

```bash
# C++ stdlib required to link the bundled ONNX Runtime (Linux only)
sudo apt-get install g++          # Debian / Ubuntu / WSL
# sudo dnf install gcc-c++        # Fedora / RHEL
```

```bash
cargo install --locked mnem-cli --features bundled-embedder

# CUDA-accelerated embedder (Linux, NVIDIA GPU)
cargo install --locked mnem-cli --features bundled-embedder-cuda
```

如果安装后找不到 `mnem`，说明 `~/.cargo/bin` 不在 `$PATH` 中。

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

没有 Cargo？[通过 rustup 安装](https://rustup.rs/)（同时会安装 `rustc`）。

```powershell
cargo install --locked mnem-cli --features bundled-embedder

# DirectML-accelerated embedder (any GPU vendor on Windows)
cargo install --locked mnem-cli --features bundled-embedder-directml
```

</details>

<details>
<summary><b>npm / Node.js</b></summary>

没有 npm？[安装 Node.js](https://nodejs.org/en/download)（npm 已内置，需要 Node 18+）。

```bash
npm install -g mnem-cli
mnem --version

# or without a global install (one-shot)
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
docker run --rm -p 9876:9876 ghcr.io/uranid/mnem:latest http serve
```

镜像已包含内置嵌入器。在容器内运行 `mnem mcp` 可使用 MCP 服务器接口。

</details>

<details>
<summary><b>从源码构建</b></summary>

```bash
# C++ stdlib required to link the bundled ONNX Runtime (Linux only)
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
mnem doctor        # checks embedder + store + config, prints a green/yellow/red checklist
```

完整安装矩阵：[`docs/src/install.md`](docs/src/install.md)。

> **想将 mnem 嵌入 Python 应用？** 上面的 `pip install mnem-cli` 以 wheel 形式发布的是 **CLI 二进制文件**。原生 **Python API**（`import mnem`）位于独立的包中。请跳转至 **[Python API (mnem-py) ↓](#python-api-mnem-py)**，查看 `pip install mnem-py` 的安装方式和代码示例。

<hr>

## 快速入门

```bash
mkdir my-graph && cd my-graph
mnem init
mnem ingest README.md
mnem retrieve "what does this project do"
```

从零开始，五分钟上手。完整演练请参见 [`docs/src/quickstart.md`](docs/src/quickstart.md)。

<hr>

## `mnem integrate` - 接入任何 Agent 宿主

一条命令即可将 **MCP 服务器条目**、**UserPromptSubmit 钩子**（适用于支持该功能的宿主）以及 **mnem 系统提示**写入宿主的项目规则文件。重启宿主后，Agent 即自动开始使用 mnem。

```bash
mnem integrate                           # interactive: detect installed hosts and prompt
mnem integrate claude-code               # wire a specific host, skip interactive detection
mnem integrate --all                     # wire every detected host without prompting

mnem integrate --check                   # report wired state for all hosts; nothing changes
mnem integrate --dry-run                 # preview what would be written without changing anything
mnem integrate --show claude-code        # print the MCP JSON block for manual copy-paste

mnem integrate --no-hooks                # skip UserPromptSubmit hook wiring
mnem integrate --no-system-prompt        # skip system prompt wiring
mnem integrate --target-repo ~/notes     # point the MCP server at a specific graph, not the global one
```

**接入内容：**
- **MCP 服务器**（`mcpServers.mnem`）- Agent 通过 `mnem mcp --repo <graph>` 获得完整的 mnem 工具访问权限；默认指向全局图（`~/.mnemglobal/.mnem`）
- **UserPromptSubmit 钩子**（仅限 Claude Code）- 在每条消息前运行 `mnem retrieve`，自动将相关记忆注入上下文
- **系统提示** - mnem 使用说明注入宿主的项目规则文件

钩子始终优先查询项目的 `.mnem/`（从当前目录向上查找），若未找到则自动回退至 `mnem global retrieve`。无论在设置期间选择哪个默认知识图，钩子和系统提示的行为保持一致。仅当你希望 MCP 服务器指向全局图以外的位置时，才需要使用 `--target-repo`。

自动检测并配置：
- Claude Code
- Claude Desktop
- Cursor
- Continue
- Zed
- Gemini CLI

任何其他支持 MCP 的宿主均可通过手动编辑 `mcpServers` 条目，指向 `mnem mcp --repo <path>` 来接入 - 参见 [`docs/src/mcp.md`](docs/src/mcp.md)。

Agent 将获得完整的 mnem 工具集作为原生工具：检索、提交、摄入、软删除（tombstone）、遍历、全局图访问等。无需额外守护进程，无需管理端口。完整工具参考：[`docs/src/mcp.md`](docs/src/mcp.md)。

<hr>

## 它是什么

**一个内容寻址知识图谱，具备混合 GraphRAG 检索、版本化提交和确定性摄入，作为 AI Agent 的持久记忆基底而构建。**

每个节点都携带由 DAG-CBOR + BLAKE3 派生的密码学身份：相同内容在任意机器上生成相同的 CID。检索在单次遍历中融合向量（HNSW）、稀疏（BM25/SPLADE）和多跳图遍历（通过 RRF），每次响应都精确报告看到了哪些候选项，以及哪些在 token 预算内被丢弃。摄入无需 LLM。单一二进制文件。无需云端。可编译至 `wasm32`。

## 你能获得什么

以下每一项均以通俗描述开头，随后给出技术细节。
标签说明：<img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> = 当前 Agent 记忆领域中 mnem 独有 &nbsp;·&nbsp; <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> = 罕见（1-2 个竞品，通常为付费方案）&nbsp;·&nbsp;（无标签）= 常见功能，实现质量优秀。

### 像 git 一样运作的记忆

- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **像 git 一样分支、对比和合并，但操作对象是 Agent 所知道的内容。** 每次写入都是带有 Ed25519 签名历史的版本化提交。两个 Agent（或两台机器）离线写入同一作用域后，通过 3-way 图 + 嵌入合并来协调，而非"最后写入优先"。→ [核心概念](docs/src/core-concepts.md)
- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **相同输入在任何计算机上始终落在相同地址。** 每个节点、树、附属数据和提交均通过规范化 DAG-CBOR + BLAKE3 进行内容寻址。相同内容折叠为同一个 CID。确定性和可重放性自然具备，无需额外代价，而非空洞的口号。→ [核心概念](docs/src/core-concepts.md)
- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **技能成为可查询的图，而非扁平的 Markdown。** 用版本化、可分支、可合并的图替代 `AGENTS.md` 和 `.cursorrules`。导出你的图，导入队友的图，对比两者，合并你需要的部分。

### 过程透明的检索

- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **没有内容会在 token 预算处无声消失。** 每次检索都将 `tokens_used`、`candidates_seen` 和 `dropped` 计数器作为一等响应字段返回。其他 Agent 记忆系统均不暴露这些信息。
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **在所有公开基准测试中均达到同类最优召回率。** 在相同嵌入器下，较开源竞品高出 **LoCoMo R@5 +0.218**、**MemBench +0.120**、**ConvoMem +0.047**。在 LongMemEval 上与 MemPalace 持平（R@5 0.966）。所有数据均可通过附带的测试框架复现。→ [基准测试](#基准测试)
- **在单次遍历中按语义、关键词和关系同时搜索。** 混合 GraphRAG：向量（HNSW）+ 稀疏（BM25/SPLADE）+ 多跳图遍历，通过 RRF 融合。图遍历按需启用：多跳有效时开启，稠密向量已饱和时关闭。

### 随处可运行

- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **可在浏览器标签页中运行。** `mnem-core` 不依赖 tokio、不依赖文件系统、不依赖网络。相同的检索代码可原封不动地编译至 `wasm32`：Chrome、Cloudflare Workers、Lambda 冷启动。Graphiti 和 mem0 是 Python + 外部数据库的技术栈，无法部署到边缘节点。
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **单个约 40 MB 的二进制文件，无守护进程、无云端、无需账户。** 内嵌 redb 存储，完全离线运行。同一镜像同时驱动 CLI 和 HTTP 服务器。
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **数秒内即插即用。** 内置 ONNX MiniLM-L6-v2 在进程内运行：无需 Ollama、无需 API 密钥、无需冷启动网络请求。只需 `mnem init` 即可开始检索。mem0 和 Graphiti 在摄入时都需要外部 LLM 端点。→ [安装](docs/src/install.md)
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **一行配置即可切换任意提供商。** 嵌入器、稀疏编码器、重排序器和 LLM 均由配置驱动。一个参数即可从本地 ONNX 切换至托管的 Cohere。无需 fork，无需重新编译。→ [嵌入提供商](docs/src/guides/embed-providers.md)
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **一个核心，四种入口。** CLI、HTTP、MCP 和 Python 均封装同一引擎。`mnem integrate` 将 MCP 服务器接入 Claude Code、Cursor、Codex、Gemini CLI 以及任何支持 MCP 协议的工具。→ [CLI 参考](docs/src/cli.md) &nbsp;·&nbsp; [MCP](docs/src/mcp.md)

### 信任信号

- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **相同字节输入始终产生相同 CID 输出。** 确定性摄入：摄入时不使用 LLM，解析 + 分块 + 抽取均为统计方法（KeyBERT 可选）。便于审计、经过模糊测试，跨机器字节完全一致。→ [摄入流水线](docs/src/guides/ingest.md)
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **以函数粒度摄入代码；以句子边界分割散文。** 源文件（10 种语言，基于 tree-sitter）每个函数、类或结构体生成一个独立 chunk；文本和 PDF 按 Unicode 句子边界切分，永不截断句子中间。Markdown、法律文本、对话、YAML、Shell 等 30 余种格式自动识别。→ [丰富摄入流水线](docs/features/rich-ingest.md)
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **在基础设施数据库层进行属性测试与模糊测试。** 解析器经过属性测试和模糊测试验证；CAR 往返转换和合并提交字节完全一致。这一信任保障通常只在底层数据库中才能见到。

### mnem 的适用场景

- 知识跨多个会话持续积累，且需要对历史进行推理。
- 两个 Agent（或两台机器）编辑同一份记忆，需要干净地协调合并。
- 审计至关重要：相同输入、相同输出，可在任意计算机上重放。
- 部署环境为边缘节点、离线或隔离网络（浏览器、Cloudflare Workers、Lambda 冷启动）。

<hr>

## 命令

每个命令都接受 `--help` 查看完整的标志参考。

### 初始化与健康检查

```bash
mnem init      # create a new graph in the current directory
mnem doctor    # probe embedder + store + config; green/yellow/red checklist
mnem stats     # nodes, edges, refs, embedder health, repo size
```

### 添加知识

```bash
mnem ingest notes.md                        # parse a file into Doc + Chunk + Entity nodes
mnem ingest --recursive docs/               # ingest a directory recursively
mnem ingest --chunker recursive report.pdf  # PDF with sliding-window chunking
```

```bash
mnem add node -s "Alice leads the infra team"                       # label defaults to "Node"
mnem add node --label Fact -s "Alice leads the infra team"          # add a single fact node
mnem add edge --from <uuid> --to <uuid> --label works_at            # connect two nodes
```

```bash
mnem get <uuid>                                                     # fetch a node by UUID: ntype, summary, props
mnem get <uuid> --content                                           # also print the full content body
mnem tombstone <uuid>                                               # soft-delete: excluded from retrieval, kept in audit log
mnem tombstone <uuid> --reason "superseded by newer decision"       # with reason recorded in op-log
mnem delete <uuid>                                                  # hard-delete: no audit trail
mnem global get <uuid>                                              # look up a node in the global graph
mnem global tombstone <uuid>                                        # tombstone a node in the global graph
```

> 摄入流水线是确定性的：摄入时不调用 LLM，相同的字节输入始终产生相同的 CID 输出。便于审计且经过模糊测试。

### 检索知识

```bash
mnem retrieve "what did we decide about the API design"  # searches local .mnem/ first, falls back to global
mnem -R ~/notes retrieve "query"                         # target a specific graph explicitly
```

`-R <path>` 是一个全局标志，可将任意命令重定向到指定的仓库目录。它会覆盖从当前目录向上查找的逻辑，以及通过 `mnem integrate` 设置的任何默认值。适用于所有子命令：`mnem -R ~/notes status`、`mnem -R ~/notes log` 等。

混合检索：向量检索（HNSW）+ 稀疏检索（BM25/SPLADE）+ 图遍历，通过 RRF 融合。调优标志请参见 [GraphRAG](#graphrag)。

### 全局知识图

> [!NOTE]
> mnem 有两个作用域：**本地知识图**（项目目录中的 `.mnem/`）和**全局知识图**（`~/.mnemglobal/.mnem/`）。全局知识图用于跨项目、跨会话的事实，这些事实应在任何地方都跟随你。

**本地与全局的使用场景：**

| 使用本地 `.mnem/` 的场景 | 使用 `mnem global` 的场景 |
|------------------------|----------------------|
| 项目专属的事实、决策、代码上下文 | 跨所有项目的人员、偏好和事实 |
| 随仓库一同传递的单仓库记忆 | 希望每个会话和每个 Agent 都能看到的知识 |
| 任何你会与代码一同提交的内容 | 跨会话的连续性 |

`mnem global` 是 `mnem` 的完整镜像，但仅操作全局知识图：

```bash
mnem global retrieve "what is Alice's current role"     # search the global graph only
mnem global ingest contacts.md                          # ingest a file into the global graph
mnem global add node --label Entity:Person \
  --prop name=Alice -s "Alice leads the infra team"     # add a node to the global graph
```

`mnem integrate` 命令会将 Agent 配置为优先读取本地图，并在需要时自动回退到全局图 - 正常使用时无需手动切换。

### 状态与检查

```bash
mnem status           # op-head CID, head commit, all named refs, label counts, MERGING marker
mnem stats            # one-line: op, commit, content CID, ref count, label names
```

### 历史记录

```bash
mnem log              # walk op-log backwards from HEAD, default last 20 entries
mnem log -n 50        # show last 50 entries
mnem log --oneline    # compact one-line-per-op format
mnem log --format json # machine-readable JSON stream

mnem show             # decode and pretty-print the current op-head block
mnem show <cid>       # decode any block by CID (Node, Edge, Commit, Operation, View, ...)

mnem diff <op-a-cid> <op-b-cid>   # ref deltas + node/edge structural diff between two ops
mnem diff HEAD <cid>               # diff current op against a specific op CID
```

### 分支与合并

```bash
mnem branch list                        # list all refs/heads/* branches; * marks current
mnem branch create <name>               # create branch at current HEAD
mnem branch create <name> <start>       # branch from a ref name, branch name, or CID
mnem branch create <name> --from HEAD   # explicit --from form; same resolution as above
mnem branch delete <name>               # delete a local branch pointer

mnem merge <branch>                     # 3-way merge <branch> into current HEAD
mnem merge <branch> --strategy=ours     # auto-resolve conflicts: keep current side
mnem merge <branch> --strategy=theirs   # auto-resolve conflicts: take incoming side
mnem merge <branch> --dry-run           # preview outcome without persisting anything
mnem merge --continue                   # finish after editing .mnem/MERGE_CONFLICTS.json
mnem merge --abort                      # cancel, restore HEAD from .mnem/ORIG_HEAD

mnem pull                               # fast-forward origin/main into HEAD (default)
mnem pull <remote> <branch>             # fast-forward <remote>/<branch> into HEAD
```

### 远程操作

```bash
mnem remote add <name> <url>            # register a remote (stores in .mnem/config.toml)
mnem remote add <name> <url> \
  --token-env MNEM_REMOTE_ORIGIN_TOKEN  # name the env var that holds the bearer token
mnem remote list                        # list all configured remotes with their URLs
mnem remote show <name>                 # show URL + capabilities for one remote
mnem remote remove <name>               # remove a remote entry

mnem fetch                              # fetch from origin (default)
mnem fetch <remote>                     # fetch from a named remote; token via env var

mnem push                               # push HEAD to origin/main (default)
mnem push <remote> <branch>             # push a specific branch to a named remote

mnem clone <url> [<dir>]                # clone a CAR archive into <dir>; file:// and bare .car paths supported
mnem clone file:///tmp/repo.car ./copy  # clone from a local file URL
mnem clone ./repo.car ./copy            # bare path shorthand (must end in .car)
```

### 查询与图遍历

```bash
mnem query --where name=Alice                    # exact property match, default 10 results
mnem query --where kind=Person -n 25             # increase result limit
mnem query --where kind=Person \
  --with-outgoing knows                          # match nodes + follow outgoing "knows" edges
mnem query --where status=active \
  --with-outgoing depends_on \
  --with-outgoing depends_on                     # repeat --with-outgoing to chain hops

mnem blame <node-uuid>                           # list all incoming edges to a node
mnem blame <node-uuid> --etype authored          # filter to one edge type
```

### 命名引用

```bash
mnem ref list                         # list all refs (refs/heads/*, refs/remotes/*, ...)
mnem ref set <name> <target-cid>      # point a ref at a specific commit CID
mnem ref delete <name>                # delete a named ref
```

### 向量嵌入

```bash
mnem embed                            # backfill embeddings for every node missing a vector
mnem embed --force                    # re-embed even nodes that already have a vector
mnem embed --label Person             # restrict to nodes of one label
mnem embed --dry-run                  # count what would be embedded without calling the provider

mnem reindex                          # alias for embed; preferred name after C7 rename
mnem reindex --label Doc              # restrict to one label
mnem reindex --since <commit>         # only nodes added/changed after <commit>
mnem reindex --force                  # re-embed already-indexed nodes
mnem reindex --dry-run                # count without calling the provider
```

### 底层块访问

```bash
mnem cat-file <cid>          # emit raw DAG-CBOR bytes for a block to stdout
mnem cat-file <cid> --json   # decode to DAG-JSON and pretty-print (pipe into jq)
```

### 导出与导入

```bash
mnem export <path>                        # export HEAD as a CAR v1 archive
mnem export -                             # write CAR to stdout (pipe over SSH etc.)
mnem export --from refs/heads/main out.car  # export from a specific ref
mnem export --from <cid> backup.car       # export from a specific commit CID

mnem import <path>                        # import a CAR archive into the current repo
mnem import -                             # read CAR from stdin
```

### 配置

```bash
mnem config set user.name Alice           # set author name
mnem config set user.email alice@example.com
mnem config set embed.provider ollama     # embedder: openai | ollama
mnem config set embed.model nomic-embed-text
mnem config set embed.base_url http://localhost:11434  # override provider endpoint
mnem config get embed.provider            # print the current value of a key
mnem config unset embed.provider          # remove a key
mnem config list                          # print all set keys and their values
```

已知键：`user.name`、`user.email`、`user.key`、`user.agent_id`、`embed.provider`、`embed.model`、`embed.api_key_env`、`embed.base_url`。API 密钥存储在环境变量中，不存储在配置文件里。

### 仓库注册表

```bash
mnem repos list              # list all repos registered with mnem integrate
mnem repos set-default <path>  # mark a repo as the default for mnem without -R
mnem repos prune             # remove registry entries for paths that no longer exist
```

### 服务器

```bash
mnem mcp                       # start the MCP JSON-RPC server over stdio
mnem mcp --repo ~/notes        # point the MCP server at a specific graph
mnem http serve                # start the HTTP JSON API (loopback by default)
```

### 基准测试命令

```bash
mnem bench                                       # interactive TUI; select benchmarks to run
mnem bench run --benches longmemeval --limit 50  # run a specific benchmark suite
mnem bench fetch longmemeval                     # download benchmark datasets
mnem bench results ./bench-out                   # re-render results from a prior run
```

### Shell 补全

```bash
mnem completions bash        # emit bash completion script
mnem completions zsh         # zsh
mnem completions fish        # fish
mnem completions powershell  # PowerShell
mnem completions elvish      # Elvish

# Install (bash):
mnem completions bash > ~/.local/share/bash-completion/completions/mnem
# Install (zsh):
mnem completions zsh > ~/.zsh/completions/_mnem
```

完整 CLI 参考：[`docs/src/cli.md`](docs/src/cli.md)。

<hr>

## Python API (mnem-py)

当你希望直接从 Python 读写 mnem 图时，请使用 `mnem-py`，无需 CLI 二进制文件。检索引擎与 CLI 完全相同，通过 PyO3 绑定暴露。

```bash
pip install mnem-py
pip install sentence-transformers   # brings ~200 MB of deps (torch, transformers)
```

`mnem-py` 通过**密集向量**进行存储与检索：你在 Python 中计算嵌入，然后将其传入 mnem。`SentenceTransformer("all-MiniLM-L6-v2")` 在首次使用时会从 HuggingFace Hub 下载约 23 MB 的模型并缓存至 `~/.cache/huggingface/`，后续所有调用完全本地运行，无需网络连接。

```python
import pymnem
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # downloaded once, ~23 MB
MODEL_NAME = "all-MiniLM-L6-v2"                    # key mnem uses to match stored vectors

repo = pymnem.Repo.init_memory()                    # in-memory; open_or_init() for disk

# Write: compute an embedding for each node and attach it
with repo.transaction(author="agent", message="seed") as tx:
    for text in ["Alice lives in Berlin", "Bob moved to Paris"]:
        tx.add_node(ntype="Memory", summary=text)
        tx.add_embedding_f32(MODEL_NAME, model.encode(text).tolist())

# Retrieve: compute a query vector with the same model, mnem ranks under token budget
query_vec = model.encode("Alice Berlin").tolist()
result = repo.retrieve(vector=query_vec, model=MODEL_NAME, token_budget=500, limit=5)
for item in result:
    print(f"{item.score:.3f}  {item.summary}")
# result.tokens_used / result.tokens_budget  - no silent truncation
```

完整 API 接口（`query`、`update_node`、`delete_node`、磁盘持久化、标签过滤）：[`crates/mnem-py/README.md`](crates/mnem-py/README.md)。

<hr>

## GraphRAG

mnem 内置了 GraphRAG。每个阶段一个开关，按需启用，从不强制要求。对于大多数查询，单独使用向量搜索就已足够 - 当查询跨越多个文档、需要多跳推理或需要组合式回答时，再开启图阶段。

### 阶段与参数

| 阶段 | 参数 | 功能说明 |
|-------|------|------|
| **向量通道** | 始终开启 | 基于每次提交的密集嵌入（默认 384 维 MiniLM）构建 HNSW 索引。 |
| **稀疏通道** | 配置驱动 | BM25 + SPLADE-onnx，通过 Reciprocal Rank Fusion 与向量融合。由 `config.toml` 中的 `[sparse]` 块控制开关。 |
| **向量候选池** | `--vector-cap <N>` | 将密集候选池大小从默认的 256 提升。数值越大，长尾召回越好，但代价相应增加。 |
| **结果数量** | `--limit <N>` | 最终返回的结果数（默认 10）。简写形式：`-n`。 |
| **图扩展** | `--graph-expand <N>` | 通过 authored 类型的边，为 top-K 种子节点添加 N 个邻居。开启图模式时，建议的审计默认值为 `20`。 |
| **图模式** | `--graph-mode <decay\|ppr>` | `decay`（默认）= 按跳数进行指数衰减加权。`ppr` = 在混合邻接索引上执行 Personalised PageRank，适用于多跳场景的论文级评分。 |
| **社区过滤** | `--community-filter` | 运行 Leiden 社区检测，在融合前丢弃覆盖率低的社区。默认覆盖率阈值：`0.5`。 |
| **KeyBERT 抽取** | `mnem ingest --extractor keybert` | 摄入时进行关键词短语增强。强化稀疏信号与社区信号。在摄入时传入，而非检索时。 |
| **摘要生成** | `--summarize` | 对 top-K 结果进行质心 + MMR 摘要，兼顾多样性。 |
| **交叉编码器重排序** | `--rerank <provider:model>` | 融合后重新排序。支持 `cohere:rerank-english-v3.0`、`voyage:rerank-1` 及本地模型。 |

### 快速示例

```bash
# Dense baseline
mnem retrieve "what does this project do"

# Add multi-hop graph traversal
mnem retrieve "..." --graph-expand 20

# Full stack: graph-expand + community-filter + PPR + rerank
mnem retrieve "..." --graph-expand 20 --community-filter --graph-mode ppr --rerank cohere:rerank-english-v3.0

# Stack a cross-encoder reranker on top
mnem retrieve "..." --graph-expand 20 --community-filter --rerank cohere:rerank-english-v3.0

# Ingest with KeyBERT keyphrase enrichment (strengthens sparse + community signals)
mnem ingest --extractor keybert notes.md
```

### 何时启用

- **单文档语料库，简单查询**：关闭图模式，单独使用向量搜索即可
- **多跳 / 组合式问题**：`--graph-expand 20`
- **跨文档引用的长历史记录**：添加 `--community-filter`
- **需要提升召回上限**：在顶层叠加 `--rerank`
- **关键词短语增强摄入**：摄入时使用 `mnem ingest --extractor keybert`

完整检索架构：[`docs/src/cli.md`](docs/src/cli.md)（检索参数说明）

<hr>

## 与其他工具对比

- [mnem vs mem0](docs/src/comparisons/mem0.md) - 智能体记忆层，开源领域领头羊
- [mnem vs MemPalace](docs/src/comparisons/mempalace.md) - 方法论同类工具
- [mnem vs Supermemory](docs/src/comparisons/supermemory.md) - 闭源云端既有方案
- [mnem vs Cognee](docs/src/comparisons/cognee.md) - 面向智能体的知识图谱替代方案
- [mnem vs Letta](docs/src/comparisons/letta.md) - 智能体记忆框架
- [mnem vs graphify](docs/src/comparisons/graphify.md) - 轻量级图工具

完整对比矩阵：[`docs/src/comparisons/README.md`](docs/src/comparisons/README.md)。

<hr>

## 何时不适合使用 mnem

- **你需要事务型 OLTP。** mnem 是仅追加的带版本历史的存储；行级 UPDATE/DELETE 语义不在其设计模型内。
- **你需要在 10k+ QPS 下实现低于 50 ms 的云端规模检索。** mnem 以本地优先为核心。多区域分片检索在路线图中，尚未包含在 v1 内。

> 需要托管记忆、多区域副本、团队共享图谱，或托管远端层？一个为 mnem 带来上述能力的兄弟项目正在积极开发中 - 敬请关注。

<hr>

## 代码包（Crates）

| crate（包）| 职责 |
|-------|------|
| [`mnem-cli`](crates/mnem-cli) | `mnem` 二进制文件 - 统一命令入口 |
| [`mnem-core`](crates/mnem-core) | 图模型、检索、索引、附属数据 |
| [`mnem-http`](crates/mnem-http) | HTTP JSON 服务器 |
| [`mnem-mcp`](crates/mnem-mcp) | MCP 服务器（stdio） |
| [`mnem-py`](crates/mnem-py) | PyO3 Python 绑定 |
| [`mnem-embed-providers`](crates/mnem-embed-providers) | ONNX 内置、Ollama、OpenAI、Cohere |
| [`mnem-sparse-providers`](crates/mnem-sparse-providers) | BM25、SPLADE-onnx |
| [`mnem-rerank-providers`](crates/mnem-rerank-providers) | Cohere、Voyage |
| [`mnem-llm-providers`](crates/mnem-llm-providers) | OpenAI、Anthropic、Ollama |
| [`mnem-ingest`](crates/mnem-ingest) | 解析 + 分块 + 抽取流水线 |
| [`mnem-extract`](crates/mnem-extract) | 实体抽取（KeyBERT、统计 NER） |
| [`mnem-ner-providers`](crates/mnem-ner-providers) | NER 提供者 trait 及内置提供者（`RuleNer`、`NullNer`） |
| [`mnem-bench`](crates/mnem-bench) | 基准测试框架（LongMemEval、LoCoMo 等） |
| [`mnem-graphrag`](crates/mnem-graphrag) | 社区摘要、质心 + MMR |
| [`mnem-ann`](crates/mnem-ann) | HNSW 封装 |
| [`mnem-backend-redb`](crates/mnem-backend-redb) | 基于 redb 的存储后端 |
| [`mnem-transport`](crates/mnem-transport) | CAR 编解码 + 远端帧协议 |

<hr>

## 文档

- [快速入门](docs/src/quickstart.md) - 五分钟上手指南
- [安装](docs/src/install.md) - 各平台安装矩阵
- [CLI 参考](docs/src/cli.md) - 所有子命令与参数
- [MCP 服务器](docs/src/mcp.md) - 暴露的工具及客户端接入方式
- [核心概念](docs/src/core-concepts.md) - CID、提交、标签
- [配置](docs/src/configuration.md) - 环境变量、config.toml
- [基准测试方法论](docs/src/benchmarks/methodology.md)
- [复现基准测试](docs/src/benchmarks/reproduce.md)
- [嵌入提供商](docs/src/guides/embed-providers.md)
- [迁移指南](docs/src/migrations/)

<hr>

## 贡献

欢迎提交 Issue 和 PR。从这里开始：

- [`CONTRIBUTING.md`](CONTRIBUTING.md) - 分支规范、Review 礼仪、如何提交 PR
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) - 行为准则（贡献者公约 2.1）
- [`SECURITY.md`](SECURITY.md) - 漏洞披露政策

## 许可证

[Apache-2.0](LICENSE)。第三方归属说明请参见 [`NOTICE`](NOTICE)。

<hr>

## 取消集成 / 移除

```bash
mnem unintegrate                  # interactive: pick which hosts to remove mnem from
mnem unintegrate claude-code      # remove one host
mnem unintegrate --all            # remove all wired hosts
```

运行 `mnem unintegrate --help` 查看所有选项。

<hr>

⭐ **觉得 mnem 有用？** 一个 Star 是来自满意的构建者最有力的信号 - 它能帮助下一位在记忆问题上苦苦挣扎的智能体开发者找到这个仓库。我们会认真阅读每一个 Issue、每一个 PR、每一条提及。告诉我们你用它构建了什么。
