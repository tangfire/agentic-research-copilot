# 代码阅读路线

## 第 0 步：先不要乱看

这个项目文件很多，但不是每个文件都同等重要。你先围绕“研究运行一次怎么走完”来看。

主线：

```text
ResearchRequest
-> LangGraphResearchRuntime
-> PlannerAgent
-> SupervisorAgent
-> ResearchAgent / DocumentStore
-> ReporterAgent
-> VerifierAgent + RAGEvaluator
-> ResearchRun
```

## 第 1 步：看 `schemas.py`

先理解数据对象：

- `ResearchRequest`：用户请求。
- `PlanItem`：计划里的一个研究问题。
- `RetrievalRoute`：这个问题用 web、vector，还是 MCP。
- `EvidenceItem`：所有证据的统一格式。
- `ResearchNote`：研究单元压缩后的发现。
- `ResearchReport`：最终报告。
- `ResearchRun`：一次运行的完整 artifact。

读这个文件时只问一个问题：一次研究运行有哪些中间产物？

## 第 2 步：看 `providers.py`

这里是真模型调用。agent 文件大多很薄，真正让大模型做决定的是 provider。

重点看这些方法：

- `clarify_request`
- `draft_plan`
- `supervise_research`
- `decide_researcher_action`
- `compose_report`
- `assess_report`
- `extract_knowledge_graph`
- `extract_graph_query`

读这个文件时只问一个问题：哪些决策是模型做的，模型必须返回什么 JSON？

## 第 3 步：看 `graph_runtime.py`

这是运行图。

节点顺序：

```text
supervisor_start
planner
research_supervisor
parallel_research
reporter
verifier_evaluator
revision_prepare 或 finalize
```

读这个文件时只问一个问题：状态从一个节点到下一个节点时，多了哪些字段？

## 第 4 步：看 `pipeline.py`

`pipeline.py` 是总装配层，不是某个算法文件。

它负责：

- 初始化 provider。
- 初始化 DocumentStore。
- 初始化 researcher/reporter/verifier/supervisor。
- 把 supervisor tool call 转成可执行 route。
- 并发执行 plan item。
- 构建 report sections。
- 估计 confidence。
- 保存 run。

读这个文件时只问一个问题：LangGraph 节点为什么都要回到 `ResearchCopilot` 拿能力？

## 第 5 步：看 `agents/`

`agents/` 不是复杂算法目录。它主要是 provider 的封装层。

- `planner.py`：把 request 交给 provider 生成 plan。
- `supervisor.py`：规范化 `ConductResearch`。
- `researcher.py`：执行 search/MCP loop。
- `reporter.py`：调用 provider 写报告。
- `verifier.py`：调用 provider 做验证。

重点是 `researcher.py`，因为它有真实循环。

## 第 6 步：看 `retrieval/store.py`

这是最像“工程难点”的文件之一。

你要看懂：

- 文档怎么切 child chunk。
- 命中 child chunk 后为什么要扩展 parent/neighbor context。
- dense retrieval 和 BM25 怎么融合。
- graph entity/relation signal 怎么参与排序。
- reranker 什么时候执行。

## 第 7 步：看 `evaluation.py`

这个文件告诉你系统怎么判断“不是随便生成了一段话”。

指标包括：

- plan coverage
- evidence sufficiency
- source diversity
- citation precision
- citation source coverage
- context precision
- faithfulness proxy

## 最后再看 API

`server.py` 只是把能力暴露出来。它不是项目核心。

你要知道 API 有：

- research run
- research job
- document ingest/search
- trace/evaluation
- runtime config/provider check

现在没有 memory API。
