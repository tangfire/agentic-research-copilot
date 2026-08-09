# pipeline.py 代码阅读指南

对应源码：

```text
D:\kn\projects\agentic-research-copilot\src\agentic_research_copilot\pipeline.py
```

这份文档专门解释 `pipeline.py`。你现在读这个文件时，先不要追每个工具函数的细节，而是先搞懂它在整个系统里的位置。

一句话定位：

> `pipeline.py` 里的 `ResearchCopilot` 是整个项目的总协调对象。它负责装配模型、搜索、RAG、记忆、评估、存储、任务队列和 Agent；研究任务从这里进入后，统一交给 `graph_runtime.py` 的 LangGraph 工作流执行。

可以把 `pipeline.py` 理解成四个角色：

| 角色 | 中文解释 | 对应代码 |
| --- | --- | --- |
| 系统装配器 | 创建模型、检索、记忆、存储、Agent 等核心组件 | `ResearchCopilot.__init__` |
| 对外门面 | 给 API 层提供添加文档、提交任务、查询 run/job 等方法 | `add_document`, `submit_job`, `get_run` |
| 任务执行入口 | 接收 `ResearchRequest`，统一进入 LangGraph 工作流 | `run` |
| 工作流辅助函数库 | 给 `graph_runtime.py` 提供研究、路由、记忆、报告组装等 helper | `_research_plan_items`, `_routes_from_supervisor_decision`, `_build_sections` |

## 1. 先记住主调用关系

项目实际运行时，大致是：

```text
server.py
-> ResearchCopilot()
-> ResearchCopilot.submit_job(...) 或 ResearchCopilot.run(...)
-> ResearchCopilot.run(...)
-> LangGraphResearchRuntime(self).run(...)
-> graph_runtime.py 中的各个节点
-> 回调 pipeline.py 中的 helper 函数
-> 保存 ResearchRun
```

最重要的一点：

> 当前默认研究链路不是直接在 `pipeline.py` 里跑完整流程，而是 `pipeline.py` 把任务交给 `graph_runtime.py`。不过 `graph_runtime.py` 会大量使用 `pipeline.py` 中已经装配好的组件和 helper 函数。

你第一次看 `pipeline.py` 时，要带着这个问题：

```text
这个函数是在“装配组件”、 “对外提供接口”，还是“给 LangGraph 节点提供辅助能力”？
```

## 2. 文件顶部：导入和小数据类

位置：

```text
pipeline.py:1-53
```

这里主要导入项目所有核心模块。

你不用背所有 import，只要按职责分组理解：

| 导入内容 | 中文作用 |
| --- | --- |
| `PlannerAgent`, `ResearchAgent`, `ReporterAgent`, `SupervisorAgent`, `VerifierAgent` | 五个 Agent 角色 |
| `DocumentReader` | 读取本地文件，例如 Markdown、HTML、PDF |
| `RAGEvaluator` | 对报告和证据做质量评估 |
| `JobLedger`, `RunLedger` | 内存中的 job/run 记录账本 |
| `build_mcp_tool` | 构建 MCP 工具调用入口 |
| `build_model_provider`, `build_embedding_provider` | 构建大模型和向量模型 provider |
| `MemoryStore` | 分层记忆 |
| `DocumentStore` | 本地 RAG 文档库 |
| `SQLiteStore` | SQLite 持久化存储 |
| `RetrievalCoordinator` | 初步生成检索路线 |
| `ResearchWorkflow` | 非模型的工作流辅助逻辑 |

### `PlanItemResearchResult`

位置：

```text
pipeline.py:44-52
```

这是一个小的结果容器，用来保存一个研究计划项执行后的结果：

| 字段 | 中文作用 |
| --- | --- |
| `item_id` | 计划项 ID |
| `web_evidence` | 联网搜索得到的证据 |
| `document_evidence` | 本地 RAG 检索得到的证据 |
| `note` | 对该计划项的压缩研究笔记 |
| `web_latency_ms` | 联网搜索耗时 |
| `document_latency_ms` | 本地检索耗时 |

这个类主要被 `_research_plan_item` 和 `_research_plan_items` 使用。

## 3. `ResearchCopilot.__init__`：系统总装配

位置：

```text
pipeline.py:54-129
```

这是整个文件最重要的初始化部分。

你可以把它理解成系统启动时的“组装车间”。

### 3.1 加载配置和严格模式校验

代码重点：

```python
self.settings = settings or load_settings()
if self.settings.strict_providers:
    require_real_provider_config(self.settings)
```

中文解释：

- `settings` 是项目配置，包括模型、搜索、Qdrant、Celery、SQLite 等配置。
- `strict_providers` 是严格真实服务模式。
- 如果严格模式开启，必须检查真实 provider 配置是否完整。

面试说法：

> 初始化阶段就会校验真实 provider 配置，避免系统悄悄 fallback 到假数据或本地简化实现。

### 3.2 构建模型、向量、MCP 和重排器

代码重点：

```python
self.model_provider = build_model_provider(self.settings)
self.embedding_provider = build_embedding_provider(self.settings, self.model_provider)
self.mcp_tool = build_mcp_tool(self.settings)
self.reranker = build_reranker(...)
```

中文解释：

| 组件 | 作用 |
| --- | --- |
| `model_provider` | 调用大语言模型，用于规划、监督、报告、校验 |
| `embedding_provider` | 把文本转成向量，用于语义检索 |
| `mcp_tool` | 调用 MCP 工具，例如资料库检索、记忆读取、运行检查 |
| `reranker` | 对候选证据重新排序 |

注意：

- 严格模式下，`reranker` 的 fallback 会被关闭。
- 非严格模式下，本地规则重排可以作为测试或离线 fallback。

### 3.3 构建 Memory 和 DocumentStore

代码重点：

```python
self.memory = MemoryStore(self.embedding_provider)
self.documents = DocumentStore(...)
```

中文解释：

`MemoryStore` 负责系统记忆。

`DocumentStore` 是本地 RAG 的核心，里面装配了：

- embedding provider
- Qdrant 配置
- BM25 融合策略
- 轻量图关系开关
- graph entity 数量配置
- reranker
- contextualizer provider
- fallback 策略

这里要注意一个重要设计：

> `pipeline.py` 不在这里实现 RAG 算法，而是把 RAG 所需配置和组件传给 `DocumentStore`。真正的检索逻辑在 `retrieval/store.py`。

### 3.4 构建存储、账本、遥测和路由器

代码重点：

```python
self.telemetry = TelemetryLog()
self.ledger = RunLedger()
self.jobs = JobLedger()
self.storage = SQLiteStore(...)
self.router = RetrievalCoordinator(...)
```

中文解释：

| 组件 | 作用 |
| --- | --- |
| `TelemetryLog` | 记录运行事件 |
| `RunLedger` | 内存中的 run 列表 |
| `JobLedger` | 内存中的 job 列表 |
| `SQLiteStore` | 持久化文档、记忆、任务、运行结果 |
| `RetrievalCoordinator` | 根据问题和 corpus 状态生成初步检索路线 |

你可以这样理解：

> `ledger` 是内存视图，`storage` 是 SQLite 持久化，`telemetry` 是事件日志。

### 3.5 构建五个 Agent

代码重点：

```python
self.planner = PlannerAgent(...)
self.researcher = ResearchAgent(...)
self.verifier = VerifierAgent(...)
self.reporter = ReporterAgent(...)
self.supervisor_agent = SupervisorAgent(...)
```

中文解释：

| Agent | 作用 |
| --- | --- |
| `PlannerAgent` | 把用户问题拆成研究计划 |
| `SupervisorAgent` | 决定工具选择、查询和证据要求 |
| `ResearchAgent` | 执行联网搜索、MCP 工具和网页阅读 |
| `ReporterAgent` | 生成引用报告 |
| `VerifierAgent` | 检查报告质量和修订需求 |

注意：

`ResearchAgent` 初始化时会传入：

- `search_tool`
- `model_provider`
- `embedding_provider`
- `mcp_tool`
- source reader 配置
- 最大研究迭代次数

所以 `ResearchAgent` 是真正会执行工具循环的 Agent。

### 3.6 恢复状态和 seed 参考知识

代码重点：

```python
self._restore_state()
if self.settings.seed_reference_knowledge:
    self._seed_reference_knowledge()
```

中文解释：

- `_restore_state` 从 SQLite 恢复文档、记忆、job 和 run。
- `_seed_reference_knowledge` 会写入一些项目说明文档作为内置参考资料。

注意：

> seed reference knowledge 只是帮助 demo 和自检，不是用户上传语料，也不是核心算法。

## 4. 文档和记忆相关对外方法

位置：

```text
pipeline.py:131-282
```

这些方法主要给 API 层调用。

### 4.1 `add_document`

作用：

```text
把一条文档证据加入 DocumentStore，并保存到 SQLite。
```

代码逻辑：

```text
self.documents.add(...)
-> self.storage.save_document(...)
-> return document
```

你要理解：

> `documents.add` 负责建索引，`storage.save_document` 负责持久化。

### 4.2 `ingest_document_path`

作用：

```text
读取本地文件路径，把文件解析成多个 segment，然后逐个 add_document。
```

代码逻辑：

```text
DocumentReader.read_path(...)
-> segments
-> add_document(...)
```

这里把“文件解析”和“文档索引”分开了：

- `DocumentReader` 负责解析文件。
- `DocumentStore` 负责索引和检索。

### 4.3 `delete_document` / `clear_documents`

作用：

- 从检索索引删除文档。
- 从 SQLite 删除文档。

这属于支撑能力，不是面试主线。

### 4.4 `clear_history`

作用：

```text
清理 runs、jobs、telemetry，必要时也清理 memory。
```

这里要注意：

- 默认不清 memory。
- 只有 `include_memory=True` 时才清记忆。

### 4.5 `clarify`

作用：

```text
判断用户问题是否太模糊，是否需要追问。
```

代码逻辑：

```text
读取 corpus_profile
-> 召回相关 memory
-> model_provider.clarify_request(...)
-> 写 telemetry
-> 返回 ClarificationContract
```

面试说法：

> clarify 是研究开始前的安全前门。问题太宽泛时，系统先要求用户补充范围，而不是直接开始查资料。

### 4.6 `add_memory`

作用：

```text
添加系统记忆，并保存到 SQLite。
```

它根据 `layer` 选择不同写入方法：

| layer | 调用 |
| --- | --- |
| `summary` | `memory.add_summary` |
| `canonical` | `memory.add_fact` |
| 其他 | `memory.add_session_note` |

你要理解：

> `pipeline.py` 不实现 memory 算法，它只是把外部请求路由到 `MemoryStore`，并负责保存。

## 5. Run 和 Job 相关对外方法

位置：

```text
pipeline.py:283-350
```

### 5.1 `list_runs` / `get_run`

作用：

- 从 SQLite 读取已有 run。
- 同步到内存 `RunLedger`。

简单理解：

```text
SQLite 是真实持久化来源
RunLedger 是当前进程里的快速视图
```

### 5.2 `submit_job`

作用：

```text
提交一个后台研究任务。
```

代码逻辑：

```text
创建 ResearchJob，状态为 queued
-> 记录 job
-> 如果配置 celery，尝试提交到 Celery
-> 如果 Celery 提交失败且 strict mode 开启，直接失败
-> 否则使用本进程 ThreadPoolExecutor 执行
```

重点：

> Celery 是优先后台队列；本地线程池是非严格模式下的 fallback。

### 5.3 `list_jobs` / `get_job`

作用：

- 查询任务列表。
- 查询单个任务状态。

### 5.4 `cancel_job`

作用：

- 如果任务还在 `queued`，直接取消。
- 如果任务已经运行，标记 `cancel_requested=True`。

注意：

运行中的 job 不是立刻强杀，而是在安全边界停止。

面试说法：

> 取消任务不是粗暴杀线程，而是通过状态标记让运行中的任务在安全边界停止，避免破坏 run artifact 保存。

## 6. `runtime_config`：运行配置说明书

位置：

```text
pipeline.py:361-679
```

这个函数很长，但第一遍不要被吓到。

它不是核心执行路径，而是返回一大段“系统当前运行配置和能力说明”。

它主要服务于：

- API 查询运行配置。
- demo readiness 检查。
- 面试时展示系统技术选型。
- MCP 工具读取当前运行状态。

你可以把它理解成：

> `runtime_config` 是系统自我介绍，不是系统执行算法。

里面包括：

| 字段 | 作用 |
| --- | --- |
| `product` | 产品定位 |
| `orchestration` | LangGraph、checkpoint、strict provider 信息 |
| `modeling` | 模型和 embedding 配置 |
| `job_execution` | Celery/Redis/线程队列配置 |
| `provider_readiness` | 真实 provider 是否就绪 |
| `reference_designs` | 参考设计说明 |
| `agents` | 各 Agent 职责 |
| `tool_registry` | web_search、MCP、document retrieval 等工具状态 |
| `retrieval` | RAG 检索策略 |
| `memory` | 记忆策略 |
| `observability` | trace、checkpoint、telemetry |
| `evaluation` | 评估指标 |
| `storage` | SQLite 存储配置 |

阅读建议：

> 这里适合查配置，不适合第一遍逐行读。

## 7. Job 执行内部函数

位置：

```text
pipeline.py:680-885
```

这段是后台任务执行逻辑。

### 7.1 `replay`

作用：

```text
拿旧 run 的 request 重新跑一次。
```

代码逻辑：

```text
get_run(run_id)
-> run.request
-> self.run(...)
```

### 7.2 `_ensure_job_executor`

作用：

```text
创建本进程单线程任务池。
```

这个是 Celery 不可用或未启用时的本地 fallback。

### 7.3 `_submit_celery_job`

作用：

```text
尝试把 job_id 发给 Celery worker。
```

如果失败：

- 写 telemetry。
- 返回 `False`。

### 7.4 `_execute_job`

作用：

```text
真正执行一个后台 job。
```

核心流程：

```text
读取 job
-> 如果取消，标记 cancelled
-> 改状态为 running
-> 按 max_attempts 重试
-> 每次尝试调用 self.run(...)
-> 成功后写 run_id 和 completed
-> 失败后写 failed 和 error
```

这段你要重点理解：

> 后台任务最终还是调用 `ResearchCopilot.run`。也就是说，不管任务从 API 同步进入还是 job 异步进入，最后都会进入同一个研究入口。

### 7.5 `_restore_state`

作用：

```text
进程启动时，从 SQLite 恢复 documents、memory、jobs、runs。
```

它还会处理中断状态：

- 如果本地线程队列的任务在上次进程退出时还在 `queued/running`，重启后标记 failed。
- 如果是 Celery 队列，则保留 queued/running，因为 worker 可能还会处理。
- 如果任务已经 cancel_requested，则恢复为 cancelled。

面试说法：

> SQLite 是任务状态的持久化来源，API 进程和 worker 进程都能通过它观察任务状态。

## 8. `_seed_reference_knowledge`：内置参考资料

位置：

```text
pipeline.py:886-959
```

作用：

```text
给系统加入一些项目自身文档作为默认参考知识。
```

包括：

- README 项目定位。
- 架构文档。
- source map。
- hardening roadmap。
- project positioning memory。

注意：

> 这不是用户语料，也不是实验论文 corpus。它是为了让系统能自解释和 demo 自检。

第一遍可以快速略读。

## 9. `run`：真正入口

位置：

```text
pipeline.py:956-960
```

这是你必须看懂的函数。

代码逻辑：

```python
def run(self, request: ResearchRequest, *, job_id: str | None = None) -> ResearchRun:
    from .graph_runtime import LangGraphResearchRuntime

    return LangGraphResearchRuntime(self).run(request, job_id=job_id)
```

中文解释：

- `ResearchCopilot.run` 统一进入 `graph_runtime.py`。
- 项目现在只保留 LangGraph 编排，不再通过配置切换第二套工作流。

当前项目只有 LangGraph 这一条主线，所以你只要记住：

```text
ResearchCopilot.run
-> LangGraphResearchRuntime(self).run
```

面试说法：

> `pipeline.py` 是统一入口，会直接把研究任务交给 LangGraph runtime。这样 API、job worker 和 replay 都可以复用同一套研究入口。

## 10. 为什么现在只有一套工作流

项目早期曾经在 `pipeline.py` 中保留一套手写的兼容工作流。它和 LangGraph 版本重复实现了规划、研究、报告、评估和修订，容易让学习者误以为项目有两套主架构。

现在已经做了收敛：

- 删除了 `pipeline.py` 中的 `_run_custom_workflow`。
- `ResearchCopilot.run` 直接进入 `LangGraphResearchRuntime`。
- `AppSettings.orchestration_runtime` 只允许 `langgraph`。
- 删除了 README 和架构文档中关于切换 `custom` 的说明。

因此你现在只需要学习这一条：

```text
ResearchCopilot.run
-> LangGraphResearchRuntime.run
-> graph_runtime.py 的显式节点
```

这不是删掉核心能力，而是删掉重复编排代码。研究、RAG、评估、记忆和后台任务 helper 仍然保留，并由 LangGraph 节点调用。

## 11. `_research_plan_items`：并行执行研究计划项

位置：

```text
pipeline.py:961-1027
```

作用：

```text
对多个 PlanItem 执行研究，并返回每个计划项的 PlanItemResearchResult。
```

核心逻辑：

```text
找出需要执行的计划项
-> 根据 supervisor 的并发限制和 settings.research_max_workers 决定 worker 数
-> 如果 worker=1，串行执行
-> 如果 worker>1，用 ThreadPoolExecutor 并行执行
-> 每个计划项调用 _research_plan_item
```

重要变量：

| 变量 | 中文含义 |
| --- | --- |
| `runnable_items` | 真正要执行研究的计划项 |
| `supervisor_worker_limit` | supervisor 输出的最大并发研究单元数 |
| `max_workers` | 实际线程数 |
| `results` | 每个计划项的研究结果 |

严格模式下：

> 如果某个并行研究任务失败，会直接抛出异常；非严格模式下会生成一个低置信度失败 note，避免整个 demo 崩掉。

## 12. `_conduct_plan_items`：确定哪些计划项被委派

位置：

```text
pipeline.py:1028-1050
```

作用：

```text
根据 SupervisorDecisionContract 中的 ConductResearch 调用，决定哪些 PlanItem 应该被执行。
```

如果 supervisor 没有明确委派，就回退到所有有 route 的计划项。

你要理解：

> 不是所有 plan item 都一定要跑研究；只有需要研究并且被 route/supervisor 覆盖的才进入执行。

## 13. `_research_plan_item`：单个计划项怎么找证据

位置：

```text
pipeline.py:1051-1125
```

这是 `pipeline.py` 中最值得认真读的 helper 之一。

作用：

```text
根据 route 决定一个计划项要不要走外部搜索、本地 RAG，最后生成研究笔记。
```

核心流程：

```text
初始化 web_evidence、document_evidence
-> 如果 route 是 external/hybrid 或需要 MCP，调用 researcher.collect_iterative
-> 如果 route 是 internal/hybrid 且本地文档可用，调用 documents.search
-> 合并 web 和 document 证据
-> workflow.compress_findings 生成 ResearchNote
-> 返回 PlanItemResearchResult
```

### 外部搜索/MCP 路径

触发条件：

```text
route.mode in {"external", "hybrid"}
或 selected_tools 中包含 mcp_tool
```

关键调用：

```python
self.researcher.collect_iterative(...)
```

这会进入 `agents/researcher.py`，执行有最大迭代次数的工具循环。

### 本地 RAG 路径

触发条件：

```text
route.mode in {"internal", "hybrid"}
request.include_private_docs == True
corpus_profile.has_private_docs == True
```

关键调用：

```python
self.documents.search(...)
```

这会进入 `retrieval/store.py` 的 `DocumentStore.search`，执行 Qdrant + BM25 + graph + rerank + parent context。

面试说法：

> `_research_plan_item` 是外部搜索和本地 RAG 的汇合点。它根据 `RetrievalRoute` 选择 web、MCP、vector retrieval 或 hybrid，然后统一输出 `EvidenceItem` 和 `ResearchNote`。

## 14. Memory helper：记忆召回和写入

位置：

```text
pipeline.py:1126-1194
```

### `_recall_memory_context`

作用：

```text
根据 topic 和 run_id 从不同 memory layer 召回记忆。
```

召回顺序：

```text
summary topic memory
-> canonical topic memory
-> session run memory
-> general topic memory
-> 去重
```

### `_build_memory_artifacts`

作用：

```text
根据最终报告生成要写入 memory 的记录。
```

它会写：

| 记忆类型 | 条件 |
| --- | --- |
| session note | 每次 run 都会写 |
| summary memory | 每次 run 都会写 |
| canonical fact | 只有 completed、confidence >= 0.6 且有 citation 时写 |

关键点：

> 稳定事实不是随便写的，必须是完成状态、置信度够、并且有引用支撑。

## 15. Route helper：把 supervisor 决策变成可执行路线

位置：

```text
pipeline.py:1195-1398
```

这部分很重要，因为它把模型输出转换成代码能执行的检索路线。

### `_routes_from_supervisor_decision`

作用：

```text
遍历 supervisor 的 ConductResearch 调用，为每个 plan item 生成 RetrievalRoute。
```

如果某个必须研究的计划项没被 supervisor 覆盖，会使用 route hint 或 fallback route 补上。

### `_route_from_conduct_call`

作用：

```text
把一个 ConductResearch tool call 转成 RetrievalRoute。
```

它会确定：

| 字段 | 中文作用 |
| --- | --- |
| `selected_tools` | 使用哪些工具 |
| `mode` | external、internal 还是 hybrid |
| `web_queries` | 联网搜索查询 |
| `internal_queries` | 本地资料检索查询 |
| `memory_query` | 记忆召回查询 |
| `min_evidence` | 最少证据数 |
| `min_sources` | 最少来源数 |
| `sufficiency_criteria` | 证据充分条件 |

### `_normalize_supervisor_tools`

作用：

```text
清洗模型输出的工具列表。
```

它会：

- 移除未知工具。
- 去重。
- 如果没启用 memory，就移除 `memory_recall`。
- 如果没启用本地文档或没有 corpus，就移除 `vector_retrieval`。

### `_mode_from_tools`

作用：

```text
根据工具组合决定 route mode。
```

规则：

| 工具 | mode |
| --- | --- |
| `web_search` + `vector_retrieval` | `hybrid` |
| 只有 `vector_retrieval` | `internal` |
| 其他 | `external` |

面试说法：

> Supervisor 可以让模型决定工具，但真正执行前会经过 route materialization，把模型输出清洗成可执行、可校验的 `RetrievalRoute`。

## 16. Run artifact 和报告章节 helper

位置：

```text
pipeline.py:1399-1606
```

### `_build_run_artifact_evidence`

作用：

```text
把一次 run 的统计信息也变成 EvidenceItem。
```

它会记录：

- plan 数量。
- search query 数量。
- retrieval route 数量。
- web hits 数量。
- memory hits 数量。
- document hits 数量。
- route mix。
- tool counts。
- query rewrite count。
- revision count。

为什么这样做？

> 这样最终报告不仅引用外部资料和本地文档，还能引用本次运行自身的过程指标，增强可复盘性。

### `_build_sections`

作用：

```text
基于 evidence、notes、routes 和 search queries 构造报告章节草稿。
```

注意：

这里生成的是给 `ReporterAgent` 的 section 草稿，不是最终报告。

它现在不是生成固定的系统介绍章节，而是按 `plan` 里的 `PlanItem` 生成 topic 相关章节。

核心逻辑是：

1. 先对 evidence 排序，过滤掉 `run-artifact` 这类过程指标，优先保留真正回答 topic 的证据。
2. 按 `metadata.plan_item_id` 把 evidence 归到对应计划项。
3. 用 `ResearchNote.evidence_titles` 再补充一次 citation 对齐。
4. 每个计划项生成一个 `ReportSection`，heading 来自 `PlanItem.question`。
5. section content 会融合：
   - 用户 topic。
   - 计划项目的 question 和 purpose。
   - `ResearchNote.finding`。
   - citation 的标题、snippet、content 摘要。
   - retrieval route 和 search query 信息。
   - 如果有 gaps 或 follow-up queries，也会写进章节。

这个改动很重要：

> 早期实现曾经把 `_build_sections` 写成固定的系统自述章节，例如 `Problem framing`、`Execution flow`、`Contextual grounding`。这会让报告偏离用户真正的研究 topic。现在章节草稿已经改成围绕 `plan + notes + evidence` 生成，`ReporterAgent` 再在这个 topic 相关草稿上做最终合成。

你要理解：

> `_build_sections` 是规则层的 topic 报告骨架，`ReporterAgent` 再用模型把这些骨架合成为更自然的报告。它的职责不是介绍这个项目怎么运行，而是把前面研究阶段收集到的结果变成可以写报告的章节草稿。

## 17. Confidence、evidence 和排序 helper

位置：

```text
pipeline.py:1626-1749
```

### `_estimate_confidence`

作用：

```text
根据证据数量、本地文档命中、memory 命中、来源数量、plan 数量估计报告置信度。
```

它不是严格统计学置信度，而是工程上的启发式分数。

### `_memory_records_to_evidence`

作用：

```text
把 memory record 转成 EvidenceItem。
```

这样 memory 可以和 web evidence、document evidence 一起进入报告生成和评估。

### `_ensure_seed_document`

作用：

```text
避免重复添加 seed reference document。
```

### `_dedupe_evidence`

作用：

```text
按 url 或 kind/source/title/content 去重证据。
```

### `_rank_evidence_for_report`

作用：

```text
对证据排序，让更可靠的证据优先进入报告引用。
```

### `_report_evidence_weight`

作用：

```text
给不同来源证据打权重。
```

加分来源：

- document chunk
- run artifact
- paper
- web summary
- official reference
- architecture/source map
- docs、GitHub、arXiv、PubMed 等来源

减分来源：

- memory
- YouTube
- Reddit

注意：

> 这里不是把弱来源删掉，而是在报告证据排序时降低权重。弱来源仍然可以在 trace 里看到。

## 18. 第一次阅读顺序

你现在看 `pipeline.py`，按这个顺序：

1. `ResearchCopilot.__init__`
2. `run`
3. `_research_plan_items`
4. `_research_plan_item`
5. `_routes_from_supervisor_decision`
6. `_route_from_conduct_call`
7. `_recall_memory_context`
8. `_build_memory_artifacts`
9. `_build_sections`
10. `_rank_evidence_for_report`
11. `submit_job`
12. `_execute_job`
13. `runtime_config`

其中第一遍可以跳过：

- `runtime_config` 的每个字段。
- `clear_documents`、`clear_history` 等管理接口。
- `_seed_reference_knowledge` 的具体文本。

## 19. 你要能画出的 pipeline.py 图

### 图一：初始化组件图

```text
ResearchCopilot.__init__
-> settings
-> model_provider / embedding_provider
-> mcp_tool / reranker
-> MemoryStore
-> DocumentStore
-> SQLiteStore
-> TelemetryLog / RunLedger / JobLedger
-> RetrievalCoordinator
-> PlannerAgent / SupervisorAgent / ResearchAgent / ReporterAgent / VerifierAgent
-> restore_state
```

### 图二：运行入口图

```text
server.py 或 celery_app.py
-> ResearchCopilot.run(request)
-> LangGraphResearchRuntime(self).run(request)
```

### 图三：研究 helper 图

```text
graph_runtime._parallel_research
-> pipeline._research_plan_items
-> pipeline._research_plan_item
-> researcher.collect_iterative
-> documents.search
-> workflow.compress_findings
-> PlanItemResearchResult
```

### 图四：路由 helper 图

```text
SupervisorDecisionContract
-> _routes_from_supervisor_decision
-> _route_from_conduct_call
-> _normalize_supervisor_tools
-> _mode_from_tools
-> RetrievalRoute
```

### 图五：报告 helper 图

```text
EvidenceItem[]
-> _build_run_artifact_evidence
-> _rank_evidence_for_report
-> _build_sections
-> ReporterAgent.build_report
-> VerifierAgent.assess
-> RAGEvaluator.evaluate
```

## 20. 面试时怎么讲 pipeline.py

可以这样说：

> `pipeline.py` 是系统的 orchestration facade，也就是总协调层。它负责在启动时装配模型 provider、embedding provider、MCP 工具、Qdrant 文档库、MemoryStore、RAGEvaluator、SQLiteStore、Telemetry、JobLedger 和五个 Agent。对外它提供文档入库、记忆写入、任务提交、run 查询等 API 支撑能力；对内它把 `ResearchRequest` 统一交给 LangGraph runtime，并给 LangGraph 节点提供研究执行、检索路线 materialization、记忆召回、报告章节构建和结果持久化等 helper。真正的 Agent 决策在 `agents` 目录，真正的 RAG 检索在 `retrieval/store.py`，而 `pipeline.py` 的价值是把这些模块组合成一次可运行、可追踪、可评估的研究任务。

如果面试官追问“为什么不把所有逻辑都写在 LangGraph 里”，可以回答：

> `graph_runtime.py` 更适合表达节点和状态流转，`pipeline.py` 更适合承载可复用的工程能力，例如文档入库、job 状态、storage、memory、route materialization 和 evidence 排序。这样 API、Celery worker、replay 和 LangGraph 节点都能复用同一个 `ResearchCopilot` 对象。

## 21. 读完 pipeline.py 后的自测问题

你读完后要能回答：

1. `ResearchCopilot.__init__` 装配了哪些核心组件？
2. 严格 provider 模式在哪里校验？
3. `add_document` 为什么既调用 `documents.add` 又调用 `storage.save_document`？
4. `submit_job` 如何在 Celery 和本地线程池之间选择？
5. `run` 为什么会跳到 `LangGraphResearchRuntime`？
6. 为什么 `ResearchCopilot.run` 现在只进入 LangGraph？
7. `_research_plan_item` 如何决定走 web search、本地 RAG 或 hybrid？
8. `_routes_from_supervisor_decision` 解决了什么问题？
9. `memory` 如何被转成 `EvidenceItem`？
10. `_build_sections` 和 `ReporterAgent.build_report` 有什么区别？
11. `_rank_evidence_for_report` 为什么不直接删除弱来源？
12. 最终 `ResearchRun` 保存了哪些信息？

能回答这 12 个问题，说明你已经基本看懂 `pipeline.py` 的主线了。
