# schemas.py 代码阅读指南

对应源码：

```text
src\agentic_research_copilot\schemas.py
```

一句话定位：

> `schemas.py` 是整个项目的数据契约层。它不负责调用模型、不负责检索、不负责生成报告，而是定义各模块之间传递的结构化对象：用户请求、研究计划、工具决策、证据、报告、评估、运行轨迹、后台任务和完整运行结果。

如果 `pipeline.py` 是总协调层，`graph_runtime.py` 是工作流层，那么 `schemas.py` 就是所有模块共同遵守的“接口协议”。

## 1. 为什么先学 schemas.py

你看大模型项目源码时，很容易被 prompt、provider、RAG、Agent 这些词绕晕。`schemas.py` 可以帮你先建立全局地图，因为它回答了三个问题：

1. 系统里有哪些核心对象？
2. 每个对象有哪些字段？
3. 这些对象如何从用户输入一路流到最终报告？

这比直接读业务代码更稳定。

你先记住这条数据流：

```text
ResearchRequest
-> PlannerContract
-> PlanItem
-> SupervisorDecisionContract
-> RetrievalRoute
-> EvidenceItem / ResearchNote
-> ResearchReport
-> VerificationContract / RAGEvaluation
-> ResearchRun
```

面试时可以说：

> 我没有让各模块随便传字典或字符串，而是用 Pydantic schema 固定数据契约。这样模型输出、工具调用、RAG 证据、报告引用、评估结果和运行轨迹都可以被验证、保存和复盘。

## 2. 文件顶部：工具函数和工具类型

位置：

```text
schemas.py:1-18
```

### `_utc_now`

作用：

```text
生成当前 UTC 时间字符串。
```

很多对象都有 `created_at`、`queued_at`、`last_updated` 这类字段，用它做默认值。

为什么用 UTC：

> 系统可能跨进程、跨服务保存运行记录，用 UTC 时间可以避免本地时区差异带来的混乱。

### `_none_to_empty_list`

作用：

```text
把 None 转成空列表。
```

这个函数很重要。真实大模型做结构化输出时，本来应该返回列表的字段，有时会返回 `null`。例如：

```json
{
  "web_queries": null,
  "selected_tools": null
}
```

如果程序直接把 `null` 当列表遍历，就会报错。所以 schema 层把这些 `None` 统一转成 `[]`。

面试说法：

> 我在 schema 层处理了 LLM structured output 的格式漂移问题，例如把 `null` list 规范化为空列表，避免模型输出的小异常传染到整个工作流。

### `ResearchToolName`

定义：

```python
ResearchToolName = Literal["web_search", "vector_retrieval", "memory_recall", "mcp_tool"]
```

中文解释：

| 工具名 | 作用 |
| --- | --- |
| `web_search` | 联网搜索 |
| `vector_retrieval` | 本地 RAG 文档检索 |
| `memory_recall` | 读取系统记忆 |
| `mcp_tool` | 调用 MCP 工具 |

这个类型限制了 supervisor 只能选择这些工具，避免模型输出一个系统不存在的工具名。

## 3. 输入和计划对象

### 3.1 `ResearchRequest`

位置：

```text
schemas.py:20-27
```

作用：

```text
用户提交的一次研究请求。
```

字段解释：

| 字段 | 中文含义 | 作用 |
| --- | --- | --- |
| `topic` | 研究主题 | 用户输入的核心问题，最少 3 个字符 |
| `depth` | 研究深度 | `quick`、`standard`、`deep` |
| `include_private_docs` | 是否使用本地文档 | 控制能否走本地 RAG |
| `use_memory` | 是否使用系统记忆 | 控制能否召回和写入 memory |
| `max_sections` | 报告最多章节数 | 控制报告长度 |
| `max_revisions` | 最大修订次数 | 控制 verifier/evaluator 失败后的重试预算 |

它被谁创建：

- API 请求。
- 测试代码。
- replay 重新运行旧 run 时。

它被谁消费：

- `pipeline.py`
- `graph_runtime.py`
- planner、supervisor、researcher、reporter、verifier

面试说法：

> `ResearchRequest` 把用户问题和运行约束放在一起，例如是否使用本地文档、是否使用记忆、最大修订次数。这样研究流程不是完全自由生成，而是受请求参数约束。

### 3.2 `PlanItem`

位置：

```text
schemas.py:29-37
```

作用：

```text
规划器把复杂问题拆出来的一个研究子任务。
```

字段解释：

| 字段 | 中文含义 |
| --- | --- |
| `id` | 计划项 ID |
| `question` | 这个子任务要回答的问题 |
| `purpose` | 这个子任务为什么存在 |
| `status` | 当前状态：pending、running、done |
| `requires_research` | 是否需要真正查资料 |
| `search_query` | 初始搜索查询 |
| `evidence_count` | 已收集证据数量 |
| `revision_hint` | 修订时的提示 |

它从哪里来：

```text
PlannerAgent
-> PlannerContract.plan
-> list[PlanItem]
```

它去哪里：

- supervisor 根据 `PlanItem` 决定工具。
- researcher 根据 `PlanItem` 收集证据。
- evaluator 根据 `PlanItem` 检查计划覆盖率。

### 3.3 `SearchQuery`

位置：

```text
schemas.py:40-46
```

作用：

```text
记录一次搜索查询或查询改写。
```

字段解释：

| 字段 | 中文含义 |
| --- | --- |
| `query` | 查询文本 |
| `intent` | 查询意图 |
| `plan_item_id` | 对应哪个计划项 |
| `tool` | 使用哪个工具 |
| `rewrite_index` | 第几次查询改写 |
| `revision` | 第几轮修订 |

为什么要有它：

> 查询不是临时字符串，而是可记录对象。这样 trace 和 evaluation 可以知道系统到底查了什么，是否做过 query rewrite。

## 4. 证据和资料对象

### 4.1 `EvidenceItem`

位置：

```text
schemas.py:49-57
```

作用：

```text
系统中最核心的证据对象。
```

无论证据来自哪里，最后都统一成 `EvidenceItem`：

- 联网搜索结果。
- 网页 raw content 压缩结果。
- 本地 RAG 文档片段。
- 系统记忆。
- MCP 工具返回。
- 运行 artifact。

字段解释：

| 字段 | 中文含义 |
| --- | --- |
| `title` | 证据标题 |
| `source` | 来源名称 |
| `kind` | 证据类型，例如 web、document-chunk、memory |
| `url` | 来源网址 |
| `snippet` | 短摘要 |
| `content` | 详细正文 |
| `score` | 相关性或质量分 |
| `metadata` | 额外信息，例如 chunk id、page number、retrieval score |

这是你必须熟悉的对象。

面试说法：

> 我把 web、RAG、memory 和 MCP 返回内容都统一成 `EvidenceItem`，这样 reporter、verifier、evaluator 不需要关心证据来自哪里，只需要基于同一种证据契约处理引用和质量检查。

### 4.2 `SourceCompressionContract`

位置：

```text
schemas.py:60-64
```

作用：

```text
SourceReader 调用模型压缩网页正文时，要求模型按这个格式返回。
```

字段：

| 字段 | 作用 |
| --- | --- |
| `summary` | 压缩总结 |
| `key_excerpts` | 关键摘录 |
| `relevance` | 与查询的相关性 |
| `limitations` | 局限或注意事项 |

它解决的问题：

> 网页 raw content 太长太乱，不能直接给 reporter。压缩后要保留摘要、关键摘录、相关性和局限，才能变成可引用证据。

### 4.3 `ChunkContextContract`

位置：

```text
schemas.py:67-71
```

作用：

```text
文档入库时，为 chunk 生成上下文前缀。
```

字段：

| 字段 | 作用 |
| --- | --- |
| `context` | 这个 chunk 所属文档和章节背景 |
| `key_terms` | 关键术语 |
| `provenance_hint` | 来源提示 |
| `confidence` | 上下文置信度 |

它和 RAG 的关系：

> chunk 本身可能很短，脱离标题和章节后不好检索。`ChunkContextContract` 让模型为 chunk 补背景，然后再进入 embedding 和 BM25 索引。

## 5. 研究执行对象

### 5.1 `ResearchNote`

位置：

```text
schemas.py:74-84
```

作用：

```text
一个计划项完成研究后的压缩笔记。
```

字段解释：

| 字段 | 中文含义 |
| --- | --- |
| `plan_item_id` | 对应哪个计划项 |
| `question` | 研究问题 |
| `finding` | 压缩后的发现 |
| `evidence_titles` | 使用了哪些证据标题 |
| `confidence` | 该发现的置信度 |
| `sufficiency_score` | 证据充分性分数 |
| `gaps` | 还缺什么证据 |
| `follow_up_queries` | 后续建议查询 |
| `research_iterations` | researcher 每轮动作 |
| `completed_reason` | 为什么停止研究 |

它从哪里来：

```text
ResearchAgent / ResearchWorkflow.compress_findings
```

它去哪里：

- reporter 生成章节。
- trace/checkpoint 保存。
- evaluation 检查证据充分性。

### 5.2 `RetrievalRoute`

位置：

```text
schemas.py:87-99
```

作用：

```text
一个计划项的检索路线。
```

它回答：

```text
这个 plan item 要用互联网、本地文档，还是两者都用？
具体查哪些 query？
至少需要多少证据和来源？
```

字段解释：

| 字段 | 中文含义 |
| --- | --- |
| `plan_item_id` | 对应计划项 |
| `mode` | `external`、`internal`、`hybrid` |
| `web_query` | 主联网搜索查询 |
| `internal_query` | 主本地检索查询 |
| `reason` | 为什么这样路由 |
| `selected_tools` | 选择的工具列表 |
| `web_queries` | 多个联网查询 |
| `internal_queries` | 多个本地检索查询 |
| `memory_query` | 记忆查询 |
| `min_evidence` | 最低证据数 |
| `min_sources` | 最低来源数 |
| `sufficiency_criteria` | 证据充分标准 |

三个 mode：

| mode | 中文解释 |
| --- | --- |
| `external` | 只用互联网搜索 |
| `internal` | 只用本地 RAG |
| `hybrid` | 互联网和本地 RAG 都用 |

面试说法：

> `RetrievalRoute` 是 supervisor 决策落地后的可执行路线。它把模型的工具选择转成代码可以执行、trace 可以记录、evaluator 可以检查的结构化对象。

## 6. 语料库和记忆对象

### 6.1 `CorpusProfile`

位置：

```text
schemas.py:102-114
```

作用：

```text
描述当前本地资料库的状态。
```

字段解释：

| 字段 | 中文含义 |
| --- | --- |
| `document_count` | 文档数量 |
| `source_count` | 来源数量 |
| `source_names` | 来源名称列表 |
| `document_kinds` | 文档类型统计 |
| `keyword_signals` | 高频关键词信号 |
| `has_private_docs` | 是否有本地文档 |
| `has_reference_docs` | 是否有参考文档 |
| `vector_backend` | 向量检索后端 |
| `keyword_backend` | 关键词检索后端 |
| `embedding_dimensions` | 向量维度 |
| `collection_name` | Qdrant collection 名称 |
| `last_updated` | 更新时间 |

它给谁用：

- planner 判断本地资料库是否可用。
- supervisor 决定是否能选 `vector_retrieval`。
- runtime_config 展示系统状态。
- evaluator 解释私有资料命中情况。

### 6.2 `MemoryRecord`

位置：

```text
schemas.py:117-127
```

作用：

```text
系统记忆的一条记录。
```

字段解释：

| 字段 | 中文含义 |
| --- | --- |
| `key` | 记忆键 |
| `value` | 记忆内容 |
| `layer` | 记忆层：session、canonical、summary |
| `tags` | 标签 |
| `run_id` | 来源 run |
| `session_id` | 来源 session |
| `topic` | 主题 |
| `confidence` | 置信度 |
| `created_at` | 创建时间 |
| `metadata` | 额外治理信息 |

三层记忆：

| layer | 中文解释 |
| --- | --- |
| `session` | 当前会话或当前 run 的临时记忆 |
| `summary` | 某个主题的总结记忆 |
| `canonical` | 相对稳定、可复用的事实 |

## 7. LLM 输出契约

这一组对象很重要，因为它们约束大模型输出，不让模型只返回一段自由文本。

### 7.1 `ClarificationContract`

位置：

```text
schemas.py:130-135
```

作用：

```text
判断用户问题是否需要澄清。
```

字段：

| 字段 | 作用 |
| --- | --- |
| `need_clarification` | 是否需要追问 |
| `question` | 要问用户的问题 |
| `verification` | 为什么判断需要或不需要澄清 |
| `missing_dimensions` | 缺失哪些维度 |
| `confidence` | 判断置信度 |

### 7.2 `PlannerContract`

位置：

```text
schemas.py:138-144
```

作用：

```text
规划器输出的结构化计划。
```

字段：

| 字段 | 作用 |
| --- | --- |
| `research_brief` | 研究简报 |
| `plan` | 多个 `PlanItem` |
| `assumptions` | 假设 |
| `success_criteria` | 成功标准 |
| `revision_budget` | 修订预算 |
| `confidence` | 规划置信度 |

### 7.3 `SupervisorToolCall`

位置：

```text
schemas.py:147-172
```

作用：

```text
研究监督器输出的一个工具调用。
```

允许的工具调用名：

| name | 中文作用 |
| --- | --- |
| `think_tool` | 先反思当前计划和证据要求 |
| `ConductResearch` | 委派具体研究任务 |
| `ResearchComplete` | 标记研究可以结束 |

它包含：

- 计划项 ID。
- 工具选择。
- web/internal 查询。
- memory 查询。
- 最低证据数。
- 最低来源数。
- 证据充分条件。

重点 validator：

```python
@field_validator(..., mode="before")
def _normalize_optional_lists(...)
```

它会把这些字段的 `None` 转成 `[]`：

- `plan_item_ids`
- `selected_tools`
- `web_queries`
- `internal_queries`
- `sufficiency_criteria`

这是现实工程里很重要的一层防御。

### 7.4 `SupervisorDecisionContract`

位置：

```text
schemas.py:175-185
```

作用：

```text
研究监督器一次完整决策。
```

字段：

| 字段 | 作用 |
| --- | --- |
| `reflection` | 整体反思 |
| `tool_calls` | 多个 `SupervisorToolCall` |
| `completion_criteria` | 研究完成标准 |
| `max_concurrent_research_units` | 最大并发研究单元 |
| `confidence` | 决策置信度 |

它和 `RetrievalRoute` 的关系：

```text
SupervisorDecisionContract
-> pipeline._routes_from_supervisor_decision
-> RetrievalRoute
```

也就是说，supervisor 的模型输出不会直接执行，先要被 materialize 成路线。

### 7.5 `ResearcherToolDecisionContract`

位置：

```text
schemas.py:188-195
```

作用：

```text
研究执行器每一轮决定下一步动作。
```

允许动作：

| action | 中文作用 |
| --- | --- |
| `think_tool` | 先思考证据缺口 |
| `web_search` | 联网搜索 |
| `mcp_tool` | 调用 MCP 工具 |
| `ResearchComplete` | 停止研究 |

字段：

- `query`
- `mcp_tool_name`
- `rationale`
- `reflection`
- `completion_reason`
- `confidence`

这对应 `ResearchAgent.collect_iterative` 的有上限工具循环。

## 8. 校验和报告输出契约

### 8.1 `VerificationContract`

位置：

```text
schemas.py:198-204
```

作用：

```text
VerifierAgent 对报告的校验结果。
```

字段：

| 字段 | 中文含义 |
| --- | --- |
| `issues` | 普通问题 |
| `critical_issues` | 严重问题 |
| `should_revise` | 是否建议修订 |
| `revision_reason` | 修订原因 |
| `confidence` | 校验置信度 |
| `coverage_score` | 覆盖率分数 |

### 8.2 `ReporterSectionDraft`

位置：

```text
schemas.py:207-210
```

作用：

```text
报告模型输出的章节草稿。
```

关键字段：

```text
citation_indexes
```

它不是直接放 URL，而是引用证据列表中的编号。

### 8.3 `ReporterContract`

位置：

```text
schemas.py:213-220
```

作用：

```text
报告生成器的结构化模型输出。
```

字段：

- `title`
- `summary`
- `sections`
- `highlights`
- `recommendations`
- `source_index`
- `confidence`

它还不是最终 `ResearchReport`，还需要 reporter 把 citation indexes 映射回真实 `EvidenceItem`。

## 9. 运行轨迹对象

### 9.1 `AgentHandoff`

位置：

```text
schemas.py:223-229
```

作用：

```text
记录一次 Agent 之间的交接。
```

例如：

```text
supervisor -> planner
planner -> research_supervisor
supervisor -> reporter
```

字段：

- `from_agent`
- `to_agent`
- `step`
- `reason`
- `revision`
- `created_at`

### 9.2 `RunTraceEvent`

位置：

```text
schemas.py:232-249
```

作用：

```text
记录一次运行中的事件。
```

事件类型：

| kind | 中文解释 |
| --- | --- |
| `handoff` | Agent 交接 |
| `tool_call` | 工具调用 |
| `step` | 普通步骤 |
| `memory_write` | 写入记忆 |
| `verification` | 报告校验 |
| `checkpoint` | 检查点 |
| `failure` | 失败事件 |
| `evaluation` | 质量评估 |

字段中比较重要的：

| 字段 | 作用 |
| --- | --- |
| `actor` | 谁做的 |
| `message` | 做了什么 |
| `step` | 哪个步骤 |
| `status` | 状态 |
| `tool_name` | 调用了哪个工具 |
| `provider` / `model` | 用了哪个模型或 provider |
| `tokens_in` / `tokens_out` | token 使用量 |
| `latency_ms` | 耗时 |
| `metadata` | 额外信息 |

面试说法：

> Trace 不是日志字符串，而是结构化事件。它能记录工具调用、模型、token、耗时、证据数量和失败原因，所以一次 run 可以被复盘。

## 10. 报告、评估和最终产物

### 10.1 `ReportSection`

位置：

```text
schemas.py:252-257
```

作用：

```text
最终报告中的一个章节。
```

字段：

- `heading`
- `content`
- `citations`
- `evidence_count`
- `source_summary`

### 10.2 `ResearchReport`

位置：

```text
schemas.py:260-269
```

作用：

```text
最终报告对象。
```

字段：

| 字段 | 中文含义 |
| --- | --- |
| `title` | 报告标题 |
| `summary` | 摘要 |
| `sections` | 多个章节 |
| `citations` | 全部引用证据 |
| `confidence` | 报告置信度 |
| `highlights` | 亮点 |
| `recommendations` | 建议 |
| `source_index` | 来源索引 |
| `source_count` | 来源数量 |

重点：

> `ResearchReport.citations` 是 `EvidenceItem` 列表，不是模型随便生成的字符串引用。

### 10.3 `RAGEvaluation`

位置：

```text
schemas.py:272-290
```

作用：

```text
一次报告的自动质量评估结果。
```

字段很多，但可以按问题分类：

| 类别 | 字段 |
| --- | --- |
| 计划是否覆盖 | `plan_coverage` |
| 检索是否有效 | `retrieval_hit_rate`, `private_retrieval_hit_rate` |
| 证据是否足够 | `evidence_sufficiency`, `source_diversity` |
| 工具选择是否合理 | `tool_selection_coverage`, `query_rewrite_count` |
| 来源质量 | `source_quality_score` |
| 上下文质量 | `context_precision`, `context_recall` |
| 是否贴合证据 | `faithfulness_proxy` |
| 引用是否准确 | `citation_precision`, `citation_source_coverage` |
| 失败细节 | `insufficient_plan_items`, `unsupported_sections`, `notes` |
| 是否通过 | `passed` |

面试说法：

> `RAGEvaluation` 把一次报告的质量问题结构化下来，不是只给一个“好/不好”。这让 demo 能展示为什么一次 run 通过或失败。

## 11. 检查点、后台任务和完整运行结果

### 11.1 `RunCheckpoint`

位置：

```text
schemas.py:293-297
```

作用：

```text
记录工作流运行到某个关键阶段时的状态摘要。
```

字段：

- `run_id`
- `stage`
- `payload`
- `created_at`

`payload` 是字典，用来存不同阶段的关键信息，例如 plan 数量、证据数量、评估结果。

### 11.2 `ResearchJob`

位置：

```text
schemas.py:300-312
```

作用：

```text
后台任务对象。
```

字段：

| 字段 | 中文含义 |
| --- | --- |
| `job_id` | 任务 ID |
| `request` | 对应的研究请求 |
| `status` | queued、running、completed、failed、cancelled |
| `run_id` | 完成后关联的 run |
| `error` | 错误信息 |
| `attempts` | 已尝试次数 |
| `max_attempts` | 最大尝试次数 |
| `timeout_seconds` | 超时时间 |
| `cancel_requested` | 是否请求取消 |
| `queued_at` | 入队时间 |
| `started_at` | 开始时间 |
| `finished_at` | 结束时间 |

### 11.3 `ResearchRun`

位置：

```text
schemas.py:315-341
```

作用：

```text
一次研究任务的完整档案。
```

这是 `schemas.py` 里最完整的对象。

它包含：

| 字段 | 中文含义 |
| --- | --- |
| `run_id` | 运行 ID |
| `job_id` | 后台任务 ID |
| `request` | 原始请求 |
| `research_brief` | 研究简报 |
| `corpus_profile` | 资料库状态 |
| `supervisor_decision` | supervisor 决策 |
| `plan` | 研究计划 |
| `search_queries` | 查询记录 |
| `retrieval_routes` | 检索路线 |
| `notes` | 研究笔记 |
| `evidence` | 全部证据 |
| `web_hits` | 联网证据 |
| `memory_hits` | 记忆证据 |
| `document_hits` | 本地文档证据 |
| `checkpoints` | 检查点 |
| `trace` | 运行轨迹 |
| `handoffs` | Agent 交接 |
| `report` | 最终报告 |
| `evaluation` | 质量评估 |
| `issues` | 校验问题 |
| `status` | 运行状态 |
| `revision_count` | 修订次数 |
| `failure_reason` | 失败原因 |
| `started_at` / `finished_at` | 开始和结束时间 |
| `duration_ms` | 耗时 |

面试说法：

> `ResearchRun` 是完整可复盘 artifact。它保存了从输入请求到最终报告之间的全部关键对象：计划、路线、证据、trace、checkpoint、evaluation 和 memory 相关证据。

## 12. 这些 schema 在主链路中怎么流动

### 12.1 用户提交问题

```text
API / test / replay
-> ResearchRequest
```

### 12.2 澄清阶段

```text
ResearchRequest
-> ClarificationContract
```

如果 `need_clarification=True`，说明问题不够明确。

### 12.3 规划阶段

```text
ResearchRequest + CorpusProfile + MemoryRecord[]
-> PlannerContract
-> PlanItem[]
```

### 12.4 研究监督阶段

```text
PlanItem[] + CorpusProfile
-> SupervisorDecisionContract
-> SupervisorToolCall[]
-> RetrievalRoute[]
```

### 12.5 研究执行阶段

```text
RetrievalRoute
-> web_search / vector_retrieval / memory_recall / mcp_tool
-> EvidenceItem[]
-> ResearchNote
```

### 12.6 报告阶段

```text
EvidenceItem[] + ReportSection drafts
-> ReporterContract
-> ResearchReport
```

### 12.7 质量闭环

```text
ResearchReport + EvidenceItem[] + PlanItem[]
-> VerificationContract
-> RAGEvaluation
-> passed 或 revision
```

### 12.8 最终保存

```text
所有中间对象
-> ResearchRun
-> SQLiteStore
```

## 13. 哪些对象最值得你先背熟

第一优先级：

- `ResearchRequest`
- `PlanItem`
- `RetrievalRoute`
- `EvidenceItem`
- `ResearchReport`
- `RAGEvaluation`
- `ResearchRun`

第二优先级：

- `PlannerContract`
- `SupervisorDecisionContract`
- `SupervisorToolCall`
- `ResearcherToolDecisionContract`
- `ResearchNote`
- `RunTraceEvent`

第三优先级：

- `SourceCompressionContract`
- `ChunkContextContract`
- `ClarificationContract`
- `AgentHandoff`
- `RunCheckpoint`
- `ResearchJob`
- `CorpusProfile`
- `MemoryRecord`

## 14. 常见误区

### 误区一：schemas.py 只是类型文件，不重要

不对。

它定义的是系统边界。模型输出、Agent 交接、RAG 证据、报告引用、评估结果和运行产物都依赖这里的契约。

### 误区二：EvidenceItem 只是搜索结果

不对。

`EvidenceItem` 是统一证据对象。web、document、memory、MCP、run artifact 都会转成它。

### 误区三：ReporterContract 就是最终报告

不对。

`ReporterContract` 是模型输出契约，`ResearchReport` 才是最终报告对象。中间还要把 citation indexes 映射回真实 evidence。

### 误区四：RAGEvaluation 是公开 benchmark

不对。

它是本地任务级质量评估对象，用于检查一次 run 的计划覆盖、证据质量、引用质量和失败原因。

### 误区五：MemoryRecord 等于聊天历史

不对。

它是分层记忆对象，有 layer、confidence、topic、metadata 等治理字段。

## 15. 读 schemas.py 的顺序

建议按这个顺序看：

1. `ResearchRequest`
2. `PlanItem`
3. `EvidenceItem`
4. `RetrievalRoute`
5. `PlannerContract`
6. `SupervisorToolCall`
7. `SupervisorDecisionContract`
8. `ResearcherToolDecisionContract`
9. `ResearchNote`
10. `ResearchReport`
11. `RAGEvaluation`
12. `RunTraceEvent`
13. `RunCheckpoint`
14. `ResearchJob`
15. `ResearchRun`

第一遍可以先跳过：

- `SourceCompressionContract`
- `ChunkContextContract`
- `AgentHandoff` 的细节
- 每个时间字段的格式

## 16. 自测问题

读完 `schemas.py` 后，你应该能回答：

1. `ResearchRequest` 控制了一次研究的哪些行为？
2. `PlanItem` 和 `SearchQuery` 有什么区别？
3. 为什么所有证据都要统一成 `EvidenceItem`？
4. `RetrievalRoute.mode` 的三个值分别是什么意思？
5. `SupervisorToolCall` 和 `RetrievalRoute` 有什么关系？
6. 为什么要把 `None` list 转成空列表？
7. `ReporterContract` 和 `ResearchReport` 有什么区别？
8. `VerificationContract` 和 `RAGEvaluation` 分别解决什么问题？
9. `RunTraceEvent` 为什么比普通日志更有价值？
10. `ResearchJob` 和 `ResearchRun` 有什么区别？
11. `ResearchRun` 为什么可以作为 demo artifact？
12. 如果面试官问“这个项目如何防止模型随便返回格式”，你会怎么回答？

## 17. 面试时怎么讲 schemas.py

可以这样讲：

> `schemas.py` 是项目的数据契约层。它用 Pydantic 定义了从输入请求到最终运行产物的全部结构化对象，包括 `ResearchRequest`、`PlanItem`、`RetrievalRoute`、`EvidenceItem`、`ResearchReport`、`RAGEvaluation` 和 `ResearchRun`。模型输出也不是自由文本，而是被约束成 `PlannerContract`、`SupervisorDecisionContract`、`ResearcherToolDecisionContract`、`ReporterContract` 和 `VerificationContract`。这样系统可以在每一步验证字段、记录 trace、保存 checkpoint，并把报告引用映射回真实证据。针对大模型可能把 list 返回成 `null` 的问题，schema 层还做了规范化处理，提升了真实 provider 下的健壮性。

最短版本：

> `schemas.py` 定义了这个大模型研究系统的对象边界。它让规划、工具选择、证据、报告、评估和运行记录都变成可验证、可保存、可复盘的数据结构。

