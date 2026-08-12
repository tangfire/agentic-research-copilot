# Agentic Research Runtime 学习总览

这份文档按“第一次看代码”的节奏写。你不用先懂所有文件，先抓住主链路。

如果你想系统学习，而不是只快速扫一遍，先读同目录下的 `agentic_research_runtime_deep_learning_guide_zh.md`。那份文档会按学习路线、代码路径、动手实验和面试追问把项目完整讲一遍。

## 这个项目到底是什么

它不是普通聊天机器人，也不是后台 CRUD。它现在更准确的定位是“Agentic Research Runtime 实验项目”：用户给一个开放式技术问题，系统先规划，再选择 Web Search、GitHub MCP、本地 RAG 等证据源，收集证据，写报告，最后验证报告有没有引用和质量问题。

你可以把它理解成一个自己学习 Deep Research / Codex 类系统工程机制的实验系统。它面向开源项目调研、工程技术选型、架构风险分析和本地技术语料 grounding，但不应该被包装成要替代成熟产品的商业 Copilot。它不是私人资料助手，也不是只分析 GitHub 项目的工具。

核心链路：

```text
问题 -> 计划 -> 监督者分派研究 -> 搜索/检索/工具调用 -> 证据压缩 -> 报告 -> 验证/评估 -> trace 复盘
```

现在核心里没有 project memory。之前的 memory 模块已经删除，因为它会让项目显得像在堆功能，但当前真正强的是 agentic research 和 RAG 证据链路。

## 先理解几个英文词

| 词 | 这里是什么意思 |
| --- | --- |
| Agent | 一个承担明确职责的模块，比如 planner、researcher、reporter。不是随便起名的类。 |
| Provider | 真模型调用层。`providers.py` 负责把请求发给 OpenAI-compatible 模型，并要求模型返回结构化 JSON。 |
| Schema | 数据契约。`schemas.py` 规定一次研究中的 request、plan、route、evidence、report 长什么样。 |
| RAG | Retrieval-Augmented Generation，检索增强生成。这里指先检索证据，再让模型基于证据写报告。 |
| Agentic RAG | 不是固定 top-k 检索，而是 agent 会改写 query、选择工具、判断证据够不够。 |
| MCP | Model Context Protocol，外部工具协议。这里主要用来接 GitHub MCP，补充 repo、code、issue、PR、release 这些开发事实源，不再使用本地 workbench MCP。 |
| Trace | 运行轨迹。记录谁交给谁、调用了什么工具、用了什么 query、评估结果如何。 |
| Evaluation | 自动质量评估。检查引用覆盖、证据充分性、上下文相关性等。 |

## 代码阅读顺序

1. `src/agentic_research_copilot/schemas.py`
2. `src/agentic_research_copilot/providers.py`
3. `src/agentic_research_copilot/graph_runtime.py`
4. `src/agentic_research_copilot/pipeline.py`
5. `src/agentic_research_copilot/agents`
6. `src/agentic_research_copilot/retrieval/store.py`
7. `src/agentic_research_copilot/evaluation.py`
8. `src/agentic_research_copilot/server.py`

## 最重要的几条链路

### 1. 规划链路

`PlannerAgent` 调用 `model_provider.draft_plan()`，把用户 topic 变成：

- `research_brief`：研究简报。
- `PlanItem[]`：多个子问题。
- `success_criteria`：成功标准。

你要看：

- `agents/planner.py`
- `providers.py` 的 `draft_plan`
- `schemas.py` 的 `PlannerContract` 和 `PlanItem`

### 2. 监督者链路

`ResearchSupervisor` 不是直接搜索，它先决定怎么分派任务。核心输出是 `SupervisorToolCall`：

- `think_tool`：先反思。
- `ConductResearch`：分派某个研究单元。
- `ResearchComplete`：声明完成条件。

你要看：

- `agents/supervisor.py`
- `providers.py` 的 `supervise_research`
- `pipeline.py` 的 `_routes_from_supervisor_decision`

### 3. 研究者工具链路

`ResearchAgent` 对每个研究单元跑一个 bounded loop，也就是有上限的循环。每一轮由模型决定：

- 继续 web search。
- 调外部 MCP tool，例如 GitHub MCP 的 repo/code/issue/PR/release 工具。
- 停止并返回 ResearchComplete。

你要看：

- `agents/researcher.py`
- `providers.py` 的 `decide_researcher_action`

### 4. RAG 检索链路

本项目不是简单 `query -> top_k chunks -> answer`。

它做了：

- child chunk 精确检索。
- parent/neighbor context expansion，也就是返回命中 chunk 周围的上下文。
- Qdrant dense retrieval。
- SQLite FTS5/BM25 keyword retrieval。
- entity/relation graph signal。
- fusion 和 rerank。

你要看：

- `retrieval/store.py`
- `retrieval/fulltext.py`
- `retrieval/rerank.py`

### 5. 报告和验证链路

报告不是凭空写。`pipeline.py` 先用 plan、notes、evidence 构造 `ReportSection`，再让 reporter 模型组织语言。之后 verifier/evaluator 检查质量。

你要看：

- `pipeline.py` 的 `_build_sections`
- `agents/reporter.py`
- `agents/verifier.py`
- `evaluation.py`

### 6. Trace 和复盘链路

每次运行都会记录：

- checkpoint
- handoff
- tool_call
- evaluation
- final report

你要看：

- `graph_runtime.py` 的 `_append_trace` 和 `_checkpoint`
- `storage.py`
- `ledger.py`

## MCP 怎么理解

现在 MCP 不是本项目的主功能，而是外部工具扩展。

正确理解：

```text
Researcher -> mcp_tools.py -> 外部 MCP Server -> EvidenceItem
```

推荐接 GitHub MCP，因为它补充的是 Tavily 和本地 RAG 都不擅长的开发事实源：仓库文件、源码、issue、PR、release。不要接完整 deep research MCP，因为那会和本项目重复。

现在 MCP 不是简单把整句话塞进 query。模型会看到 MCP tool catalog，然后决定：

- `mcp_tool_name`：调用哪个外部工具。
- `mcp_tool_args`：传什么结构化参数，比如 `owner`、`repo`、`path`、`query`。

## 学到什么

学完这个项目，你应该能讲清楚：

- 大模型应用为什么需要 schema。
- agent 为什么要有 supervisor。
- RAG 为什么不能只做 top-k。
- query rewrite、tool selection、evidence sufficiency 怎么串起来。
- citation-backed report 怎么避免模型乱造引用。
- trace/evaluation 为什么是工业化项目和玩具项目的分界。
- 为什么删掉不够真实的 memory 和本地 MCP workbench 反而让项目更可信。
