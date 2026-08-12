# 面试讲解笔记

## 一句话

这是一个参考 Open Deep Research / Deep Research 类系统自己实现的 **Agentic Research Runtime 实验项目**。它把开放式技术问题拆成研究计划，由 supervisor 分派研究单元，自动选择 Web Search、GitHub MCP 和本地 Agentic RAG 等证据源，最后生成带引用、可验证、可复盘的技术调研报告。

更诚实的说法是：

> 我不是在做一个要打败 Codex 的商业 Copilot，而是在学习并复现复杂研究 Agent 背后的工程机制：LangGraph 状态流、结构化 tool call、多证据路由、RAG grounding、citation synthesis、verifier/evaluator 和 trace replay。

## 面试官问“用 Codex 不就好了？”

可以这样回答：

> 对，作为最终用户工具，Codex / Deep Research 当然更强。我这个项目不是想替代它，而是把这类系统背后的 agent runtime 拆出来自己实现了一遍。重点在工程机制：怎么把问题拆成计划，怎么让 supervisor 做结构化工具决策，怎么把 Web、GitHub MCP、本地 RAG 都统一成证据，怎么约束引用，怎么评估 groundedness，怎么保存 trace 并 replay 一次运行。

这比说“我做了一个更好的研究助手”稳得多。

## 现在还需要补什么

不要继续堆“看起来像 Agent 的功能”。现在最应该补的是面试可展示的硬证据：

1. **2-3 个稳定 demo topic**
   - 开源项目尽调：输入 GitHub repo，分析架构、核心代码、issue 风险、PR 活跃度、release 变化。
   - 技术选型 memo：比较 LangGraph / AutoGen、Qdrant / Milvus、不同 RAG 设计。
   - 本地语料研究：上传论文/架构文档，展示 child chunk、parent context、BM25、graph signal、rerank。

2. **真实 demo 资产**
   - 每个 topic 保存 report、trace、evaluation、source index、route metadata。
   - 不要只现场跑，网络和模型 provider 很容易出意外。

3. **更小但更准的 eval set**
   - 给每个样例标 expected sources、expected evidence types、expected terms。
   - 重点测 citation coverage、evidence sufficiency、context precision、unsupported sections。

4. **GitHub MCP smoke demo**
   - 明确需要哪些工具：`get_file_contents`、`search_code`、`list_issues`、`list_pull_requests`、`get_latest_release`。
   - 明确如果 auth/network 失败，就展示已保存的 run bundle。

5. **一键导出 run bundle**
   - 面试前最好能导出一个目录：request、report markdown、trace JSON、evaluation JSON、source index、runtime config summary。

6. **前端只围绕证据和复盘打磨**
   - 重点是 plan、evidence、citations、quality gates、trace。
   - 不要现在重做成大而全产品 UI。

## 现在的边界

已经从核心里删除：

- project memory 模块
- 本地 workbench MCP server
- `/v1/memory` API

原因：这些功能在没有稳定历史语料和真实长期使用数据时，容易显得像“框架很全，但实验不实”。现在保留更能讲清楚的主链路：规划、监督、检索、证据、报告、验证、评估、trace。

## 核心链路

```text
clarify
-> planner
-> research_supervisor
-> parallel_research
-> reporter
-> verifier_evaluator
-> finalize
```

你面试时不要把重点放在 CRUD。真正值得讲的是：

- `schemas.py` 里的结构化契约。
- `providers.py` 里的真模型调用。
- `graph_runtime.py` 里的 LangGraph 状态图。
- `pipeline.py` 里的应用装配和跨模块编排。
- `agents/` 里各 agent 如何把决策交给 provider。
- `retrieval/store.py` 里的 Agentic RAG：child chunk 检索、parent/neighbor 上下文扩展、dense、BM25、graph fusion、rerank。
- `evaluation.py` 里的质量指标。

## 技术选型怎么讲

- LangGraph：研究流程有状态、分支、验证、返工和 trace，不适合写成一条简单 chain。
- FastAPI：只是本地 API，用来暴露 run、job、document、trace、evaluation。
- Qdrant：负责向量检索。
- SQLite FTS5/BM25：负责关键词精确匹配，补 dense retrieval 的短板。
- graph signal：参考 LightRAG，只作为检索增强信号，不包装成完整 GraphRAG 平台。
- reranker：把 dense、BM25、graph 融合后的候选再按 query 相关性排序。
- MCP：现在是外部工具接口，不是本地自调用 demo。

## MCP 选型

不要接那种“深度研究助手 MCP”，因为它会和本项目的 planner/supervisor/reporter 重复。

更适合：

- GitHub MCP remote read-only endpoint：补充 repo、code、issue、PR、release 这些开发事实源。
- 论文/学术搜索 MCP：只有 demo 真的需要论文元数据时再考虑。

现在这条链路不是简单 `query -> MCP`，而是模型先看 MCP tool catalog，再决定 `mcp_tool_name` 和 `mcp_tool_args`。例如分析开源项目时，Tavily 查背景，GitHub MCP 查 README、源码、issue 风险、PR 活跃度和 release 变化。

如果以后把本项目做成 MCP Server，应该暴露：

- `run_research`
- `search_local_corpus`
- `inspect_research_run`

这应该是一个对外 facade，不应该恢复之前的本地 workbench MCP。

## 项目是不是玩具

现在不是简单玩具项目，因为它有：

- 结构化 plan 和 supervisor tool call。
- bounded researcher loop。
- 多证据通道。
- 真实 RAG 检索链路。
- citation-backed report。
- verifier + evaluator。
- trace、checkpoint、run artifact。
- 测试覆盖。

但也不要过度包装。它不是：

- 分布式研究平台。
- 企业级知识库。
- 浏览器自动化 agent。
- 通用 agent SDK。
- 长期个性化 memory 系统。

最好的定位是：**自己学习复杂 LLM Agent 工程机制的实验项目**。它的价值不是“市场必须买单”，而是你能讲清楚一个 research agent runtime 怎么设计、怎么观测、怎么评估、怎么失败、怎么迭代。

## 可以怎么写简历

推荐标题：

> Agentic Research Runtime：面向复杂技术调研的多证据 Agent 实验系统

推荐 bullet：

> 基于 LangGraph、OpenAI-compatible Provider、Qdrant、SQLite FTS5/BM25、Reranker 和 FastAPI 实现 Agentic Research Runtime，支持复杂问题规划、ODR-style supervisor tool call、并发研究单元、Web/GitHub MCP/本地文档多证据路由、引用报告生成、验证评估与 trace replay。

可以强调：

- 将复杂问题拆成可执行研究单元。
- 用 `ConductResearch` 携带工具选择、query rewrite、证据阈值和检索模式。
- 设计 child 检索 + parent/neighbor context expansion，兼顾精确命中和上下文完整性。
- 使用 dense + BM25 + graph signal + rerank 提升召回和排序。
- 用 verifier/evaluator 检查 citation precision、evidence sufficiency、context precision 等指标。
- 主动删除 memory 和本地 MCP workbench，避免噱头模块影响项目可信度。
