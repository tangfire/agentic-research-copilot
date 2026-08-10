# AI Research Copilot 代码阅读指南

这份文档是给“已经看过项目学习文档，但还没真正读代码”的你用的。

目标不是把所有源码读完，而是在较短时间内掌握这个项目的主干：

1. 用户问题从哪里进入系统。
2. LangGraph 工作流如何一步步推进。
3. 各个 Agent 分别负责什么。
4. RAG 如何从文档和网页中生成可引用证据。
5. 报告如何生成、校验、评估、修订和写入记忆。
6. 面试官追问代码时，你知道该打开哪个文件、哪个函数。

项目根目录：

```text
D:\kn\projects\agentic-research-copilot
```

## 0. 总原则

不要从 `src` 目录第一行开始读。

正确方式是：沿着一次真实研究任务读代码。

也就是先记住这条主链路：

```text
用户提交研究问题
-> ResearchCopilot.run
-> LangGraphResearchRuntime.run
-> 读取记忆
-> 生成研究计划
-> 决定工具和检索路线
-> 执行搜索、本地 RAG、MCP 工具
-> 汇总证据
-> 生成引用报告
-> 校验和质量评估
-> 必要时修订
-> 写入记忆
-> 保存完整运行结果
```

第一次读代码只问四件事：

| 问题 | 你要找什么 |
| --- | --- |
| 输入是什么 | 用户问题、计划、证据、报告还是运行状态 |
| 调用了谁 | Agent、检索器、模型 provider、数据库、工具 |
| 输出是什么 | 计划、证据、报告、评估、运行记录 |
| 失败怎么办 | 报错、修订、跳过、回退、写 trace |

## 1. 第一站：入口文件

先看这个文件：

```text
src\agentic_research_copilot\pipeline.py
```

重点函数：

```python
ResearchCopilot.run(...)
```

它是一次研究任务的总入口。

你要看懂：

```python
from .graph_runtime import LangGraphResearchRuntime

return LangGraphResearchRuntime(self).run(request, job_id=job_id)
```

这说明项目现在只有一套研究编排：LangGraph 工作流。

阅读重点：

- `ResearchCopilot` 是系统总协调对象。
- `run()` 收到 `ResearchRequest`。
- 直接把任务交给 `LangGraphResearchRuntime`。
- 旧的 `_run_custom_workflow` 已经删除，不需要再寻找或比较第二套流程。

你读完这里要能回答：

> 用户问题进入系统后，为什么会跳到 `graph_runtime.py`？

## 2. 第二站：数据结构

再看这个文件：

```text
src\agentic_research_copilot\schemas.py
```

不要全读，先只看这些类：

| 类名 | 中文作用 | 为什么重要 |
| --- | --- | --- |
| `ResearchRequest` | 研究请求 | 用户问题和运行配置 |
| `PlanItem` | 研究计划项 | 一个复杂问题拆出来的小任务 |
| `RetrievalRoute` | 检索路线 | 决定用互联网、本地文档还是混合检索 |
| `EvidenceItem` | 证据对象 | 报告引用的最小证据单位 |
| `SupervisorDecisionContract` | 研究监督器决策 | 规定工具调用、查询和证据要求 |
| `ResearchReport` | 最终报告 | 报告标题、摘要、章节、引用和置信度 |
| `RunTraceEvent` | 运行轨迹事件 | 记录系统每一步发生了什么 |
| `RunCheckpoint` | 检查点 | 保存关键步骤状态 |
| `ResearchRun` | 完整运行结果 | 一次任务所有结果的总对象 |

读这些类时，不要背字段。你只要理解：

```text
ResearchRequest 是输入
PlanItem 是规划结果
RetrievalRoute 是工具路线
EvidenceItem 是证据
ResearchReport 是输出报告
ResearchRun 是一次完整运行的档案
```

你读完这里要能回答：

> 为什么这个项目不是把所有内容都放进一个字符串，而是定义这么多结构化对象？

面试回答：

> 因为复杂研究流程需要可检查的中间状态。计划、检索路线、证据、报告、评估和运行轨迹必须分开建模，否则后续校验、修订和回放都很难做。

## 3. 第三站：LangGraph 工作流骨架

看这个文件：

```text
src\agentic_research_copilot\graph_runtime.py
```

先找：

```python
ResearchGraphState
```

它是 LangGraph 工作流中的状态对象。可以理解成一次任务在每个节点之间传递的“工作台”。

重点字段：

| 字段 | 中文作用 |
| --- | --- |
| `request` | 原始研究请求 |
| `memory_hits` | 召回到的记忆证据 |
| `final_plan` | 最终研究计划 |
| `final_retrieval_routes` | 最终检索路线 |
| `final_evidence` | 汇总后的全部证据 |
| `final_report` | 最终报告 |
| `final_evaluation` | 质量评估结果 |
| `trace` | 运行轨迹 |
| `checkpoints` | 检查点 |
| `revision_count` | 修订次数 |
| `needs_revision` | 是否需要修订 |

然后找：

```python
_build_graph(...)
```

你会看到节点顺序：

```text
supervisor_start
-> memory_recall
-> planner
-> research_supervisor
-> parallel_research
-> reporter
-> verifier_evaluator
-> revision_prepare 或 memory_write
-> finalize
```

这一段是项目主干。你应该先把它画下来。

## 4. 工作流节点逐个读

下面是你读 `graph_runtime.py` 的顺序。

### 4.1 `_supervisor_start`

作用：

- 创建 `run_id`。
- 初始化 `trace`、`checkpoints`、`revision_count` 等状态。
- 写入第一批检查点。

你要看：

- 它如何初始化 `ResearchGraphState`。
- 它如何记录 `langgraph.runtime` 和 `supervisor.start`。

读完要能说：

> 这个节点相当于一次研究任务的启动器，负责创建运行上下文和可观测性记录。

### 4.2 `_memory_recall`

作用：

- 如果用户启用了记忆，就从 memory store 中召回相关记忆。
- 把记忆转换成 `EvidenceItem`，参与后续报告生成。

你要看：

- `self.copilot._recall_memory_context(...)`
- `self.copilot._memory_records_to_evidence(...)`

读完要能说：

> 记忆不是直接拼聊天历史，而是先召回相关记忆，再转成证据对象，和其他证据一起进入报告链路。

### 4.3 `_planner`

作用：

- 读取用户问题、语料库状态和记忆。
- 调用 `PlannerAgent` 生成研究简报和计划项。
- 调用 router 生成候选检索路线。

关键调用：

```python
self.copilot.planner.draft(...)
self.copilot.router.build_routes(...)
```

读完要能说：

> Planner 负责把宽泛问题拆成结构化研究计划，并结合资料库状态生成初步路线。

### 4.4 `_research_supervisor`

作用：

- 调用 `SupervisorAgent`。
- 决定每个计划项用哪些工具。
- 生成最终 `RetrievalRoute`。
- 生成搜索查询。

关键调用：

```python
self.copilot.supervisor_agent.decide(...)
self.copilot._routes_from_supervisor_decision(...)
self.copilot.workflow.build_queries(...)
```

读完要能说：

> ResearchSupervisor 不是直接收集证据，而是负责工具选择、查询改写和证据要求，相当于研究任务调度器。

### 4.5 `_parallel_research`

作用：

- 对每个计划项执行研究。
- 收集互联网证据、本地文档证据、记忆证据和 MCP 工具证据。
- 汇总成 `final_evidence`。

关键调用：

```python
self.copilot._research_plan_items(...)
```

这一段很重要，因为真正的搜索和 RAG 会从这里被触发。

读完要能说：

> 这个节点根据检索路线执行一个或多个研究子任务，并把不同来源的证据统一成 `EvidenceItem`。

### 4.6 `_reporter`

作用：

- 根据研究计划和证据构造报告章节。
- 对证据排序。
- 调用 `ReporterAgent` 生成引用报告。

关键调用：

```python
self.copilot._build_sections(...)
self.copilot._rank_evidence_for_report(...)
self.copilot.reporter.build_report(...)
```

读完要能说：

> Reporter 不是从零编报告，而是在已有证据基础上组织章节，并把引用绑定到证据对象。

### 4.7 `_verifier_evaluator`

作用：

- 调用大模型校验器找报告问题。
- 调用自动评估器计算质量指标。
- 判断是否需要修订。

关键调用：

```python
self.copilot.verifier.assess(...)
self.copilot.evaluator.evaluate(...)
```

读完要能说：

> Verifier 偏模型判断，Evaluator 偏规则和指标检查，两者一起决定报告能不能通过质量门控。

### 4.8 `_revision_prepare`

作用：

- 如果质量不通过，增加修订次数。
- 写入修订原因。
- 回到 `planner` 重新走一轮。

你要看：

- 它如何更新 `revision_count`。
- `_route_after_verification` 如何决定走 `revise` 还是 `finish`。

读完要能说：

> 修订不是简单让模型重写，而是把失败原因带回规划阶段，重新规划和补证据。

### 4.9 `_memory_write`

作用：

- 如果启用 memory，就把本次运行中值得保存的信息写入记忆。

关键调用：

```python
self.copilot._build_memory_artifacts(...)
self.copilot.storage.save_memory(...)
```

读完要能说：

> 记忆写入发生在质量闭环之后，避免把失败结果随便沉淀成长期记忆。

### 4.10 `_finalize`

作用：

- 构造完整 `ResearchRun`。
- 写入 ledger 和 storage。
- 记录运行完成事件。

关键对象：

```python
ResearchRun(...)
```

读完要能说：

> `ResearchRun` 是一次任务的完整档案，包含请求、计划、路线、证据、报告、评估、轨迹和检查点。

## 5. Agent 代码怎么读

Agent 文件都在：

```text
src\agentic_research_copilot\agents
```

第一遍只读这五个：

| 文件 | 重点类 | 重点函数 | 作用 |
| --- | --- | --- | --- |
| `planner.py` | `PlannerAgent` | `draft` | 生成研究计划 |
| `supervisor.py` | `SupervisorAgent` | `decide`, `_normalize` | 决定工具和路线，并修正模型输出 |
| `researcher.py` | `ResearchAgent` | `collect_iterative` | 执行有上限的搜索和工具循环 |
| `reporter.py` | `ReporterAgent` | `build_report` | 根据证据生成引用报告 |
| `verifier.py` | `VerifierAgent` | `assess` | 检查报告问题 |

### 5.1 `PlannerAgent`

文件：

```text
src\agentic_research_copilot\agents\planner.py
```

重点函数：

```python
draft(...)
```

它本身不复杂，主要是调用模型 provider：

```python
self.model_provider.draft_plan(...)
```

你要理解：

> Planner 的工程价值不在复杂 Python 逻辑，而在把模型输出约束成 `PlannerContract`，也就是结构化计划。

### 5.2 `SupervisorAgent`

文件：

```text
src\agentic_research_copilot\agents\supervisor.py
```

重点看两个函数：

```python
decide(...)
_normalize(...)
```

`decide` 调用模型，让模型决定研究工具和查询。

`_normalize` 更值得重点看，因为它处理模型输出不可靠的问题：

- 如果缺少 `think_tool`，自动补一个。
- 如果 `ConductResearch` 指向不存在的计划项，会过滤掉。
- 如果某个必须研究的计划项没有被分配，会补一个 fallback 研究指令。
- 如果没有 `ResearchComplete`，会补结束信号。
- 如果没有本地资料，就移除 `vector_retrieval`。
- 如果工具列表为空，就至少补 `web_search`。

这段可以作为面试亮点：

> 大模型输出不是天然可信的，所以 Supervisor 层会对工具调用计划做结构化规范化，保证后续研究链路不会因为空字段、非法工具或漏分配任务而断掉。

### 5.3 `ResearchAgent`

文件：

```text
src\agentic_research_copilot\agents\researcher.py
```

重点函数：

```python
collect_iterative(...)
```

这是研究执行器的核心循环：

```text
准备查询队列
-> 检查证据缺口
-> 让模型决定下一步动作
-> 可能执行 think_tool
-> 可能执行 web_search
-> 可能执行 mcp_tool
-> 合并并去重证据
-> 检查证据数量和来源数量是否足够
-> 不够则生成 follow-up query
-> 达到条件或超过最大次数后结束
```

重点变量：

| 变量 | 中文含义 |
| --- | --- |
| `query_queue` | 待执行查询队列 |
| `evidence` | 已收集证据 |
| `iterations` | 每轮研究动作记录 |
| `min_evidence` | 最低证据数量 |
| `min_sources` | 最低来源数量 |
| `completed_reason` | 停止原因 |
| `follow_up_queries` | 后续补充查询 |

你要特别看：

```python
self.model_provider.decide_researcher_action(...)
```

它让模型在有限工具中做选择。

但真正执行工具的是代码：

```python
self.collect(...)
self.collect_mcp(...)
```

面试说法：

> Researcher 不是自由行动的 Agent，而是受最大迭代次数约束的工具循环。每一轮都会记录动作、查询、新证据数、总证据数、来源数、缺口和停止原因。

### 5.4 `ReporterAgent`

文件：

```text
src\agentic_research_copilot\agents\reporter.py
```

重点函数：

```python
build_report(...)
```

关键点：

- 模型生成 `ReporterContract`。
- 报告中的 `citation_indexes` 必须映射到已有 `EvidenceItem`。
- 如果模型没给出有效引用，会回退到 fallback section。

面试说法：

> Reporter 不是让模型随便编引用，而是把模型输出的引用编号映射回已有证据列表，保证 citation 和 evidence 绑定。

### 5.5 `VerifierAgent`

文件：

```text
src\agentic_research_copilot\agents\verifier.py
```

重点函数：

```python
assess(...)
```

它调用模型 provider：

```python
self.model_provider.assess_report(...)
```

它负责输出：

- 报告问题。
- 严重问题。
- 覆盖率。
- 是否建议修订。

## 6. RAG 代码怎么读

RAG 是这个项目最值得学的部分。

先看：

```text
src\agentic_research_copilot\retrieval\store.py
```

如果你想看逐函数拆解版，先打开这篇单独文档：

```text
D:\kn\projects\agentic-research-copilot\docs\learning\zh\ai_research_copilot_retrieval_store_py_guide_zh.md
```

重点函数：

```python
DocumentStore.search(...)
```

这个函数可以拆成 7 步：

```text
1. 拼接 query、context、purpose
2. 对查询做 tokenize 和 embedding
3. 用 Qdrant 做语义向量召回
4. 用 SQLite FTS5/BM25 做关键词召回
5. 合并 dense 和 BM25 候选
6. 用轻量实体关系图补充候选
7. 用 reranker 重排
8. 补回父段落和相邻 chunk
9. 返回 EvidenceItem
```

重点函数顺序：

| 函数 | 作用 |
| --- | --- |
| `search` | 检索入口 |
| `_search_qdrant` | Qdrant 语义向量检索 |
| `_merge_keyword_candidates` | 合并 BM25 关键词检索结果 |
| `_fuse_dense_keyword_score` | dense 和 BM25 分数融合 |
| `_merge_graph_candidates` | 加入轻量图关系候选 |
| `_search_graph` | 根据实体共现关系找候选 |
| `_rerank` | 调用重排器 |
| `_parent_context_for_child` | 补回父段落和相邻片段 |

你读 `store.py` 时先不要看所有工具函数。先把上面 8 个函数串起来。

### 6.1 文档入库怎么读

还是在 `store.py`。

重点看：

```python
add(...)
extend(...)
_index_document(...)
```

文档入库大致是：

```text
EvidenceItem 文档
-> 生成 document_id
-> 切成 child chunks
-> 为每个 chunk 构造 contextual_text
-> 计算 embedding
-> 写入 BM25 关键词索引
-> 写入轻量 graph 索引
-> 写入 Qdrant 向量库
```

这里你要理解两个概念：

- `chunk.text`：原始小片段。
- `chunk.contextual_text`：加了文档标题、来源、章节等上下文后用于检索的文本。

面试说法：

> 入库时不是直接把原文切块后向量化，而是为 chunk 补充上下文前缀，使孤立片段在检索时仍然带有文档和章节背景。

### 6.2 BM25 怎么读

文件：

```text
src\agentic_research_copilot\retrieval\fulltext.py
```

重点看：

```python
SQLiteBM25Index.search(...)
```

你只要理解：

> SQLite FTS5 负责全文索引，BM25 根据关键词匹配程度排序。它补充向量检索对专有名词、缩写和精确指标不敏感的问题。

### 6.3 重排怎么读

文件：

```text
src\agentic_research_copilot\retrieval\rerank.py
```

重点看：

```python
DashScopeReranker.rerank(...)
RuleBasedReranker.rerank(...)
```

区别：

- `DashScopeReranker`：真实 Qwen/DashScope 重排服务。
- `RuleBasedReranker`：本地规则重排，主要用于 fallback 和测试。

严格模式下：

> 如果真实 provider 配置不对，不能悄悄回退，要暴露错误。

## 7. 外部网页阅读怎么读

文件：

```text
src\agentic_research_copilot\source_reader.py
```

重点函数：

```python
SourceReader.read(...)
```

它处理的是互联网搜索返回的网页正文。

三种策略：

| 策略 | 中文解释 |
| --- | --- |
| `extract` | 直接抽取与查询相关的句子或段落 |
| `model_compress` | 让模型把网页正文压缩成短证据 |
| `chunk_rerank_compress` | 先切块、重排、补相邻块，再让模型压缩 |

重点函数顺序：

| 函数 | 作用 |
| --- | --- |
| `read` | 阅读入口 |
| `_chunk_rerank_compress` | 切块、排序、扩展、压缩 |
| `_model_compress` | 模型压缩 |
| `_extract` | 规则抽取 |
| `split_text` | 长网页切块 |
| `rank_chunks_lexical` | 基于关键词排序 |
| `expand_neighbor_chunks` | 补相邻片段 |
| `stitch_chunks` | 拼回上下文 |

你要理解：

> 搜索结果摘要太短，网页正文又太长，所以 source reader 把 raw content 加工成短而可引用的证据。

## 8. 质量评估怎么读

文件：

```text
src\agentic_research_copilot\evaluation.py
```

重点类：

```python
RAGEvaluator
```

重点函数：

```python
evaluate(...)
```

你要看它如何计算：

| 指标 | 检查什么 |
| --- | --- |
| `plan_coverage` | 是否覆盖研究计划 |
| `evidence_sufficiency` | 证据是否足够 |
| `tool_selection_coverage` | 工具使用是否符合计划 |
| `context_precision` | 取回证据是否相关 |
| `context_recall` | 重要证据是否找全 |
| `faithfulness_proxy` | 报告是否贴合证据 |
| `citation_precision` | 引用是否能映射到证据 |
| `citation_source_coverage` | 引用来源是否足够 |
| `unsupported_sections` | 是否存在无证据章节 |

你不需要一开始背公式。先理解它在抓哪些失败：

- 有计划没回答。
- 有结论没证据。
- 有引用但对不上证据。
- 来源太少。
- 报告和证据不一致。
- 本地检索没有命中。

## 9. 记忆代码怎么读

文件：

```text
src\agentic_research_copilot\memory\store.py
```

重点类：

```python
MemoryStore
```

重点函数：

| 函数 | 作用 |
| --- | --- |
| `add_session_note` | 添加会话记忆 |
| `add_summary` | 添加主题总结记忆 |
| `add_fact` | 添加稳定事实记忆 |
| `search` | 召回相关记忆 |

重点看：

- 它如何区分 `session`、`summary`、`canonical`。
- 它如何处理冲突。
- 它如何结合关键词和 embedding 相似度召回记忆。

面试说法：

> Memory 不是聊天历史拼接，而是分层、带置信度、带冲突治理的可召回状态。

## 10. API 和后台任务怎么读

这部分不是核心亮点，但要知道系统怎么跑起来。

### 10.1 FastAPI

文件：

```text
src\agentic_research_copilot\server.py
```

重点看：

```python
create_app(...)
```

你只要理解：

- API 接收文档、记忆和研究任务。
- API 可以查询任务状态、结果、trace、evaluation。
- API 层不是简历重点，只是让研究链路可操作、可检查。

### 10.2 Celery

文件：

```text
src\agentic_research_copilot\celery_app.py
```

重点看：

```python
execute_job(...)
```

你只要理解：

- Celery worker 从队列中取出 job。
- 调用 `ResearchCopilot.run(...)`。
- 执行结果写回 storage。
- 任务状态可以被 API 查询。

面试说法：

> Celery/Redis 的作用是把耗时研究任务从 HTTP 请求里解耦出来，避免真实模型、搜索和向量检索导致接口长时间阻塞。

## 11. 测试怎么读

测试文件比源码更适合你快速理解“这个模块想保证什么”。

推荐顺序：

| 测试文件 | 看什么 |
| --- | --- |
| `tests\test_pipeline.py` | 研究主链路、source reader、MCP、任务状态 |
| `tests\test_retrieval.py` | RAG、BM25、图关系、重排、父级上下文 |
| `tests\test_memory.py` | 记忆分层、冲突治理、embedding 召回 |
| `tests\test_verifier.py` | 无证据报告是否会被拦截 |
| `tests\test_schemas.py` | 模型返回异常结构时是否被规范化 |
| `tests\test_settings.py` | 严格 provider、BOM `.env`、真实配置 |
| `tests\test_celery_app.py` | worker 初始化和任务状态可观测性 |

读测试时不要只看断言怎么写，而要写下：

```text
这个测试在防止哪一种系统失败？
```

举例：

- `test_supervisor_contract_treats_null_lists_as_empty_lists`：防止大模型把列表字段返回 `null` 导致系统崩。
- `test_document_store_uses_real_bm25_keyword_index_for_exact_terms`：证明 BM25 不是摆设。
- `test_canonical_memory_conflicts_are_marked_for_review`：证明稳定记忆不会被随便覆盖。
- `test_strict_provider_mode_reports_missing_real_config`：证明严格模式不会悄悄 fallback。

## 12. 推荐三天阅读路线

### 第 1 天：读主工作流

文件：

```text
pipeline.py
graph_runtime.py
schemas.py
```

目标：

- 能画出 LangGraph 节点图。
- 能说清楚 `ResearchGraphState` 中关键字段在哪里产生。
- 能解释一次任务最终如何变成 `ResearchRun`。

当天必须能回答：

```text
用户问题从哪里进入？
哪个节点负责规划？
哪个节点负责研究监督？
哪个节点收集证据？
哪个节点生成报告？
哪个节点决定是否修订？
最终结果在哪里保存？
```

### 第 2 天：读研究执行和 RAG

文件：

```text
agents\researcher.py
retrieval\store.py
retrieval\fulltext.py
retrieval\rerank.py
source_reader.py
```

目标：

- 能讲清楚 `collect_iterative` 的循环。
- 能讲清楚 Qdrant、BM25、graph、rerank、parent context 的顺序。
- 能讲清楚网页 raw content 怎么变成可引用证据。

当天必须能回答：

```text
Researcher 为什么不是无限自主搜索？
BM25 补了向量检索什么短板？
图关系信号是不是完整 GraphRAG？
重排为什么放在候选融合之后？
为什么命中 child chunk 后还要补 parent context？
```

### 第 3 天：读质量闭环和工程问题

文件：

```text
agents\reporter.py
agents\verifier.py
evaluation.py
memory\store.py
server.py
celery_app.py
tests
```

目标：

- 能讲清楚 citation 如何绑定 evidence。
- 能讲清楚 evaluator 抓哪些质量问题。
- 能讲清楚 memory 为什么不是聊天历史。
- 能讲清楚 Celery/Redis 为什么存在。
- 能从测试里讲出真实问题和修复。

当天必须能回答：

```text
报告引用如何防止模型编来源？
质量评估为什么不是公开 benchmark？
失败 run 为什么有价值？
哪些结果会写入 memory？
Celery queued 问题为什么是可观测性问题？
```

## 13. 推荐断点

如果你用 PyCharm 或 VS Code 调试，可以先打这些断点：

```text
src\agentic_research_copilot\pipeline.py
ResearchCopilot.run

src\agentic_research_copilot\graph_runtime.py
_planner
_research_supervisor
_parallel_research
_reporter
_verifier_evaluator
_memory_write
_finalize

src\agentic_research_copilot\agents\researcher.py
ResearchAgent.collect_iterative

src\agentic_research_copilot\retrieval\store.py
DocumentStore.search

src\agentic_research_copilot\source_reader.py
SourceReader.read
```

每次断下来只看三个东西：

```text
当前输入是什么？
当前新增了什么状态？
当前要调用哪个下一步？
```

不要第一次调试就展开所有对象。尤其是 `EvidenceItem` 很多时，只看数量、来源、标题和 metadata。

## 14. 你要画的四张图

学完代码后，建议你自己画四张图。

### 图一：工作流图

```text
request
-> memory_recall
-> planner
-> research_supervisor
-> parallel_research
-> reporter
-> verifier_evaluator
-> revision 或 memory_write
-> finalize
```

### 图二：状态流转图

```text
ResearchRequest
-> final_plan
-> final_retrieval_routes
-> final_evidence
-> final_report
-> final_evaluation
-> ResearchRun
```

### 图三：证据生成图

```text
互联网搜索结果
-> SourceReader
-> EvidenceItem

本地论文文档
-> DocumentStore.search
-> Qdrant + BM25 + graph + rerank + parent context
-> EvidenceItem
```

### 图四：质量闭环图

```text
ResearchReport + EvidenceItem
-> Verifier
-> Evaluator
-> passed
   或
-> revision_prepare
-> planner
```

## 15. 面试前代码掌握标准

你不用做到每行都背，但至少要达到下面标准。

### 必须熟

- `ResearchCopilot.run` 为什么进入 LangGraph。
- `_build_graph` 的节点顺序。
- `ResearchGraphState` 的关键字段。
- `SupervisorAgent._normalize` 为什么重要。
- `ResearchAgent.collect_iterative` 的循环逻辑。
- `DocumentStore.search` 的 RAG 检索顺序。
- `ReporterAgent.build_report` 如何绑定引用。
- `RAGEvaluator.evaluate` 检查哪些质量问题。

### 可以一般熟

- 具体 prompt 文字。
- 所有 API endpoint。
- SQLite 每张表的字段。
- Docker Compose 每个配置项。
- 全部 provider 适配器细节。

### 不要花太多时间

- 静态前端页面。
- CRUD 细节。
- 所有测试 fixture 的实现细节。
- 每个 fallback 分支的完整代码。

## 16. 最终你要能讲出的代码版总结

> 用户请求进入 `ResearchCopilot.run` 后，直接交给 `LangGraphResearchRuntime`。LangGraph 用 `ResearchGraphState` 保存一次研究任务的状态，并按节点执行记忆召回、规划、研究监督、并行研究、报告生成、校验评估、修订和记忆写入。规划器生成 `PlanItem`，研究监督器生成 `RetrievalRoute` 和工具调用，研究执行器通过有上限的循环收集互联网、本地 RAG 和 MCP 证据。本地 RAG 由 `DocumentStore.search` 完成，内部结合 Qdrant 语义检索、SQLite BM25 关键词检索、轻量实体关系候选、重排和父级上下文扩展。报告生成器把引用编号映射回已有 `EvidenceItem`，校验器和评估器检查引用、证据和计划覆盖。最终所有计划、证据、报告、评估、trace 和 checkpoint 都保存到 `ResearchRun`，用于复盘和面试展示。
