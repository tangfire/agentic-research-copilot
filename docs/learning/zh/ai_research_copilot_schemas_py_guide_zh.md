# `schemas.py` 阅读指南

## 这个文件负责什么

`schemas.py` 定义项目里的数据契约。契约就是：每个 agent、provider、API、storage 之间传递的数据必须长什么样。

你先把它理解成整套系统的类型地图。

## 研究请求

### `ResearchRequest`

用户输入：

- `topic`：研究主题。
- `depth`：`quick`、`standard`、`deep`。
- `include_private_docs`：是否使用本地文档检索。
- `max_sections`：最多生成几个报告章节。
- `max_revisions`：最多返工几次。

当前没有旧的 memory 开关字段，因为 memory 已经从核心删除。

## 计划

### `PlanItem`

一个 plan item 就是一个子研究问题。

重要字段：

- `id`：唯一 ID。
- `question`：具体问题。
- `purpose`：这个问题为什么重要。
- `search_query`：给搜索/检索用的 query。
- `requires_research`：是否需要收集证据。
- `evidence_count`：这个子问题最后拿到了多少证据。

## 工具和路线

### `ResearchToolName`

当前工具只有：

```python
"web_search" | "vector_retrieval" | "mcp_tool"
```

- `web_search`：外部搜索。
- `vector_retrieval`：本地文档 RAG 检索。
- `mcp_tool`：外部 MCP 工具。

### `RetrievalRoute`

它告诉系统一个 plan item 应该怎么找证据。

重要字段：

- `mode`：`external`、`internal`、`hybrid`。
- `selected_tools`：用哪些工具。
- `web_queries`：外部搜索 queries。
- `internal_queries`：本地检索 queries。
- `min_evidence`：最低证据数量。
- `min_sources`：最低来源数量。
- `sufficiency_criteria`：什么叫证据足够。

## 证据

### `EvidenceItem`

所有证据统一成这个对象。

来源可能是：

- web search
- source reader
- document chunk
- external MCP tool
- run artifact

重要字段：

- `title`
- `source`
- `kind`
- `url`
- `snippet`
- `content`
- `score`
- `metadata`

这样 reporter、verifier、evaluator 不需要关心证据具体从哪里来。

## Supervisor 契约

### `SupervisorToolCall`

监督者只允许输出三种 tool call：

- `think_tool`
- `ConductResearch`
- `ResearchComplete`

`ConductResearch` 会携带：

- `plan_item_ids`
- `research_topic`
- `mode`
- `selected_tools`
- `web_queries`
- `internal_queries`
- `min_evidence`
- `min_sources`
- `sufficiency_criteria`

这就是项目 agentic 的关键：不是代码固定路线，而是模型用结构化契约决定研究路线。

## Researcher 契约

### `ResearcherToolDecisionContract`

研究者每一轮可以决定：

- `think_tool`
- `web_search`
- `mcp_tool`
- `ResearchComplete`

如果选择 MCP，现在会带两类信息：

- `mcp_tool_name`：具体调用哪个外部 MCP 工具，例如 GitHub MCP 的 `search_code` 或 `get_file_contents`。
- `mcp_tool_args`：结构化参数，例如 `owner`、`repo`、`path`、`query`。

这一步很重要：GitHub MCP 不是简单把整句话塞进 query，而是要求模型按工具 schema 生成参数。

## 报告和运行结果

### `ReportSection`

报告中的一个章节，必须带 citations。

### `ResearchReport`

最终报告，包括：

- title
- summary
- sections
- citations
- highlights
- recommendations
- source_index

### `ResearchRun`

一次完整运行的 artifact。

它保存：

- request
- plan
- routes
- search queries
- notes
- evidence
- web hits
- document hits
- trace
- handoffs
- checkpoints
- report
- evaluation
- issues
- status

## 质量评估

### `RAGEvaluation`

这些字段用来证明项目不是只生成一段文本：

- `plan_coverage`
- `retrieval_hit_rate`
- `evidence_sufficiency`
- `tool_selection_coverage`
- `source_quality_score`
- `context_precision`
- `faithfulness_proxy`
- `citation_precision`
- `citation_source_coverage`
- `source_diversity`

## 看这个文件的目标

你要能回答：

- 一次 research run 有哪些结构化中间产物？
- supervisor 和 researcher 分别能做哪些动作？
- report 为什么不能随便编 citation？
- memory 为什么已经不在当前 schema 里？
