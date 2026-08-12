# Agentic Research Runtime 深度学习手册

这份文档是给“想通过这个项目把 Agent / RAG / LangGraph / MCP / Evaluation 真正学明白”的你写的。

它不是 README，也不是面试包装话术。它的目标是帮你建立一张脑内地图：一次复杂技术调研请求从 API 进入系统以后，如何被拆成计划，如何被 supervisor 路由到不同证据通道，如何检索本地文档，如何调用外部工具，如何生成报告，如何验证质量，最后如何被 trace 和 replay。

读完以后，你应该能做到三件事：

1. 给别人讲清楚这个项目为什么不是普通 RAG。
2. 顺着一次 run 找到每个核心对象在哪个文件里产生、变形、保存。
3. 面试官追问 LangGraph、tool call、RAG、MCP、evaluation、trace 时，你能讲实现细节，而不是只讲概念。

## 0. 先定心：这个项目到底学什么

这个项目现在最准确的定位是：

> Agentic Research Runtime：一个用于学习和复现 Deep Research / Codex 类系统工程机制的实验项目。

这句话里有几个关键词。

`Runtime` 表示它不只是 prompt，不只是聊天页面。它关心一次任务怎么被执行、怎么被拆分、怎么调用工具、怎么保存中间状态、怎么验证、怎么复盘。

`Agentic` 表示系统里有模型参与决策，例如规划、监督、工具选择、查询改写、报告写作、质量判断。但它不是“越多 agent 越高级”。每个 agent 都必须有清楚职责。

`Research` 表示任务不是 CRUD，也不是简单问答，而是开放式技术调研：需要拆问题、找证据、比较来源、承认证据缺口。

`Experiment` 表示它是学习项目，不要把它包装成打败 Codex 的商业产品。成熟产品当然更强。这个项目的价值是你亲手实现了那类系统背后的工程骨架。

## 1. 推荐学习路线

不要从最大文件 `retrieval/store.py` 开始硬啃。建议分四遍读。

### 第一遍：跑通主线

目标：知道一次请求大概怎么流动。

读这些文件：

1. `docs/product-positioning.md`
2. `docs/architecture.md`
3. `src/agentic_research_copilot/schemas.py`
4. `src/agentic_research_copilot/server.py`
5. `src/agentic_research_copilot/pipeline.py`
6. `src/agentic_research_copilot/graph_runtime.py`

这一遍不要纠结每个字段。你只要抓住：

```text
API -> ResearchCopilot -> LangGraphResearchRuntime -> agents/tools/retrieval -> report/evaluation/trace -> storage
```

### 第二遍：理解 Agent 编排

目标：知道 planner、supervisor、researcher、reporter、verifier 分别干什么。

读这些文件：

1. `src/agentic_research_copilot/agents/planner.py`
2. `src/agentic_research_copilot/agents/supervisor.py`
3. `src/agentic_research_copilot/agents/researcher.py`
4. `src/agentic_research_copilot/agents/reporter.py`
5. `src/agentic_research_copilot/agents/verifier.py`
6. `src/agentic_research_copilot/providers.py`

这一遍要重点看：agent 本身通常很薄，真正让模型做判断的是 provider 方法。

### 第三遍：理解证据系统

目标：知道 web、GitHub MCP、本地 RAG 为什么最后都能进入同一套报告和评估链路。

读这些文件：

1. `src/agentic_research_copilot/schemas.py` 里的 `EvidenceItem`
2. `src/agentic_research_copilot/search.py`
3. `src/agentic_research_copilot/source_reader.py`
4. `src/agentic_research_copilot/mcp_tools.py`
5. `src/agentic_research_copilot/retrieval/store.py`
6. `src/agentic_research_copilot/evaluation.py`

这一遍要抓住一句话：

> 不管证据来自 Tavily、GitHub MCP、本地文档，最后都要变成 `EvidenceItem`，否则 reporter、verifier、evaluator 就没法统一处理。

### 第四遍：准备面试讲法

目标：能把实现讲成工程问题，而不是功能列表。

读这些文件：

1. `docs/interview-notes.zh-CN.md`
2. `docs/resume-demo-runbook.md`
3. `docs/hardening-roadmap.md`
4. `examples/eval-dataset.jsonl`

这一遍要形成自己的回答：

- 为什么不是普通 RAG？
- 为什么需要 LangGraph？
- supervisor 为什么不直接搜索？
- GitHub MCP 为什么只是证据通道，不是主产品？
- 为什么删掉 memory 反而更可信？
- 如果 Codex 已经能做，为什么这个项目还有价值？

## 2. 总体架构：从入口到结果

先看一张文字版架构图：

```text
用户问题
  |
  v
FastAPI server.py
  |
  v
ResearchCopilot pipeline.py
  |
  v
LangGraphResearchRuntime graph_runtime.py
  |
  +--> Clarifier
  +--> PlannerAgent
  +--> SupervisorAgent
  +--> ResearchAgent
  |      +--> web_search
  |      +--> source_reader
  |      +--> mcp_tool
  |
  +--> DocumentStore / Retriever
  |      +--> Qdrant dense retrieval
  |      +--> SQLite FTS5 / BM25
  |      +--> graph signal
  |      +--> reranker
  |
  +--> ReporterAgent
  +--> VerifierAgent
  +--> RAGEvaluator
  |
  v
ResearchRun / ResearchReport / Trace / Evaluation
  |
  v
SQLite storage + API endpoints
```

这张图里最重要的是分层。

API 层负责“把功能暴露出去”。它不应该包含复杂研究逻辑。

Pipeline 层负责“把组件装起来”。它知道系统里有哪些 provider、retriever、agent、storage。

Graph runtime 层负责“按状态图执行任务”。它决定先规划、再监督、再并发研究、再报告、再验证。

Agent 层负责“把某类决策封装成清楚职责”。例如 planner 只负责计划，verifier 只负责检查报告。

Provider 层负责“真正调用模型”。这样测试时可以换 deterministic provider，真实 demo 时可以换 OpenAI-compatible provider。

Retrieval 层负责“从本地语料里找证据”。它不是简单 top-k。

Evaluation / Trace 层负责“证明系统怎么运行、质量如何”。这是面试里区别玩具项目的关键。

## 3. 数据契约：先读 `schemas.py`

如果只能认真读一个文件的前半部分，先读 `src/agentic_research_copilot/schemas.py`。

大模型应用容易乱，是因为模型输入输出天然不稳定。这个项目用 schema 把每一步的输入输出固定下来。

### 3.1 `ResearchRequest`

`ResearchRequest` 是用户请求。

你需要关注：

- `topic`：研究问题。
- `depth`：研究深度。
- `include_private_docs`：是否使用本地文档。

它是最开始进入系统的对象，后面 planner、supervisor、retriever、reporter 都会围绕它工作。

### 3.2 `PlanItem`

`PlanItem` 是 planner 拆出来的子问题。

一个复杂 topic 不应该直接丢给搜索工具。planner 会把它拆成多个 focused plan item，例如：

- 分析项目架构。
- 查关键实现文件。
- 查 issue 风险。
- 查 release 变化。
- 对照本地采用标准。

面试时可以说：

> 我没有让模型一次性回答开放问题，而是先把问题变成结构化计划，再让 supervisor 对每个 plan item 决定证据路线。

### 3.3 `RetrievalRoute`

`RetrievalRoute` 是证据路由。

它回答几个问题：

- 这个 plan item 应该查 web 吗？
- 应该查本地文档吗？
- 应该调 MCP 吗？
- 检索 query 是什么？
- 需要多少证据才算够？

这是项目从“聊天机器人”变成“研究 runtime”的关键对象之一。因为它让证据选择变成可观察、可测试、可解释的结构。

### 3.4 `EvidenceItem`

`EvidenceItem` 是证据统一格式。

不同来源的证据会被统一成它：

- web search result
- source reader 压缩后的网页内容
- GitHub MCP 返回的 repo/code/issue/PR/release 内容
- 本地 RAG 检索到的文档 chunk
- run artifact 里总结出的计划、路线、评估信息

为什么要统一？

因为 reporter 不应该关心“证据从哪里来的底层细节”。它只需要知道 title、source、snippet、url、metadata。verifier 和 evaluator 也一样。

### 3.5 `ResearchNote`

`ResearchNote` 是 researcher 对某个研究单元的阶段性结论。

它通常来自一个 bounded researcher loop。研究者不是只拿证据，还要压缩成 note，供后续 reporter 使用。

### 3.6 `ResearchReport`

`ResearchReport` 是最终报告。

重点字段通常包括：

- sections
- source index
- confidence
- citations

面试时不要说“我让模型写 markdown”。更准确是：

> reporter 基于已有 notes 和 evidence 生成结构化报告，报告里的 section 和 citation 会再经过 verifier/evaluator 检查。

### 3.7 `RAGEvaluation`

`RAGEvaluation` 是质量评估结果。

它不是完美的学术评测，但很适合项目面试，因为它证明你考虑了质量闭环。

常见指标包括：

- citation coverage
- evidence sufficiency
- source diversity
- context precision
- unsupported sections

## 4. API 层：`server.py` 只做入口

`src/agentic_research_copilot/server.py` 里最重要的是 `create_app()`。

它暴露几类接口。

研究接口：

```text
POST /v1/research/clarify
POST /v1/research/runs
GET  /v1/research/runs
GET  /v1/research/runs/{run_id}
GET  /v1/research/runs/{run_id}/trace
GET  /v1/research/runs/{run_id}/evaluation
POST /v1/research/runs/{run_id}/replay
```

任务接口：

```text
POST /v1/research/jobs
GET  /v1/research/jobs/{job_id}/status
GET  /v1/research/jobs/{job_id}/result
```

文档接口：

```text
POST /v1/documents
POST /v1/documents/ingest
GET  /v1/documents/search
```

运行时接口：

```text
GET /v1/runtime/config
GET /v1/runtime/provider-check
```

注意：现在没有 `/v1/memory`。这不是漏写，而是刻意删除。

理解 API 层时要记住：

> server.py 不应该变复杂。它应该把请求交给 `ResearchCopilot`，而不是自己实现研究逻辑。

## 5. 总装配层：`pipeline.py` 是系统中枢

`src/agentic_research_copilot/pipeline.py` 里的 `ResearchCopilot` 是应用 facade。

你可以把它理解成“把所有零件装成一台机器”的地方。

### 5.1 初始化时装了什么

`ResearchCopilot.__init__` 会组装这些组件：

- settings
- model provider
- embedding provider
- search tool
- MCP tool registry
- document store
- planner
- supervisor
- researcher
- reporter
- verifier
- evaluator
- storage
- telemetry
- LangGraph runtime

这就是为什么它叫 pipeline。它不是单个 agent，而是把多个能力串成一个完整运行时。

### 5.2 `run()`

`ResearchCopilot.run()` 是同步执行一次研究的入口。

它通常把请求交给 `LangGraphResearchRuntime.run()`。真正的状态图执行在 `graph_runtime.py`。

这是一种很好的工程分层：

- pipeline 知道“有什么组件”。
- graph runtime 知道“这些组件按什么顺序执行”。

### 5.3 文档管理

`add_document()` 和 `ingest_document_path()` 会把本地资料放进 `DocumentStore`。

这里的本地资料不是“长期记忆”，而是当前 research runtime 可检索的 evidence corpus。

区别很重要：

- memory 暗示长期用户画像、偏好、历史经验。
- corpus 表示本次或本项目可检索资料。

这个项目保留 corpus，删除 memory，是为了让边界更诚实。

### 5.4 job 管理

`submit_job()`、`_execute_job()`、`cancel_job()` 支持异步任务。

这可以让前端提交研究任务后轮询状态。但它不应该被夸成分布式执行平台。

面试时可以说：

> 当前是 single-node job execution，Celery/Redis 只是可选的进程分离，不是分布式平台承诺。

### 5.5 section 构造

`_build_sections()` 是报告生成前很重要的一步。

它会把 plan、notes、evidence 变成 reporter 可以使用的 section draft。这样 reporter 不是凭空写，而是被已有证据约束。

## 6. LangGraph 层：为什么不是一条 chain

`src/agentic_research_copilot/graph_runtime.py` 里的 `LangGraphResearchRuntime` 是项目的核心运行时。

它存在的原因是：复杂研究不是一条线。

一次研究可能需要：

- 先澄清问题。
- 生成计划。
- supervisor 决定分派哪些研究单元。
- 多个研究单元可以并发。
- 搜索和本地检索都可能产生证据。
- reporter 生成报告。
- verifier/evaluator 发现证据不足。
- 如果还有 revision budget，就回到 planner 或继续补证据。
- 最后 finalize 并保存。

这种流程适合状态图，不适合一条普通 chain。

### 6.1 核心节点

当前图大概是：

```text
supervisor_start
-> planner
-> research_supervisor
-> parallel_research
-> reporter
-> verifier_evaluator
-> revision_prepare or finalize
```

你读代码时可以按这些方法找：

- `_supervisor_start`
- `_planner`
- `_research_supervisor`
- `_parallel_research`
- `_reporter`
- `_verifier_evaluator`
- `_revision_prepare`
- `_finalize`

### 6.2 state 是什么

`ResearchGraphState` 是图里的状态对象。

可以把它理解成一次 run 的工作台，上面放着：

- request
- run_id
- research_brief
- plan
- routes
- notes
- evidence
- report
- evaluation
- trace
- revision_count
- errors

每个节点读 state 的一部分，写回新的字段。

这就是 LangGraph 的好处：你不用把所有变量塞进函数参数，也不用让每个 agent 私自保存状态。

### 6.3 checkpoint 和 trace

`_checkpoint()` 负责记录关键阶段快照。

`_append_trace()` 负责记录细粒度事件。

面试里可以这样讲：

> 我不只保存最终答案，也保存计划、工具调用、证据路线、验证结果和失败信息。这样一次 run 可以被复盘，而不是只有一个不可解释的模型输出。

### 6.4 revision loop

`_route_after_verification()` 决定验证后是 finish 还是 revise。

如果 verifier/evaluator 认为引用不足、证据不足、source diversity 不够，而且 revision budget 还没用完，就进入 `_revision_prepare()`。

这就是研究 runtime 和普通问答的区别之一：

> 系统不是生成一次就结束，而是可以根据质量检查结果返工。

## 7. Agent 层：职责要小，不要玄学

这个项目里有几个 agent，但不要把它理解成“多 agent 越多越强”。

真正好的 agent 拆分是：每个 agent 都有明确职责和输入输出。

### 7.1 PlannerAgent

文件：`src/agentic_research_copilot/agents/planner.py`

职责：把 topic 变成 research brief 和 plan items。

它本身不负责搜索、不负责写最终报告。

背后的模型方法是 `provider.draft_plan()`。

学习重点：

- planner 输出必须结构化。
- plan item 要足够具体，后面才能路由证据。
- plan item 不能只是换个说法重复 topic。

### 7.2 SupervisorAgent

文件：`src/agentic_research_copilot/agents/supervisor.py`

职责：决定研究怎么分派。

它会处理 ODR-style tool calls：

- `think_tool`
- `ConductResearch`
- `ResearchComplete`

重点不是名字，而是行为：

- 先想清楚证据缺口。
- 再把某个研究单元分派出去。
- 分派时携带工具偏好、query、证据阈值等。

学习重点：

> supervisor 不是为了表演“多 agent”，而是为了让开放问题先经过结构化调度，再进入工具层。

### 7.3 ResearchAgent

文件：`src/agentic_research_copilot/agents/researcher.py`

职责：执行一个 plan item 的证据收集。

它有三条常见路径：

1. `collect()`：普通外部搜索。
2. `collect_mcp()`：外部 MCP 工具调用。
3. `collect_iterative()`：bounded researcher loop。

bounded loop 的意思是：研究者可以多轮查证据，但有最大轮数和停止条件。

为什么需要 bounded？

因为 agent 如果没有边界，很容易无限搜索、重复搜索、花太多 token。工程系统必须有上限。

学习重点：

- 每轮模型都要决定下一步。
- 证据要去重。
- 不够时要能生成 follow-up query。
- 够了时要 ResearchComplete。

### 7.4 ReporterAgent

文件：`src/agentic_research_copilot/agents/reporter.py`

职责：把 notes 和 evidence 写成最终报告。

reporter 的危险点是最容易幻觉。所以这个项目让 reporter 基于已有 section drafts 和 evidence 写报告，然后再进入 verifier/evaluator。

学习重点：

> reporter 不是“自由作文”，而是“基于证据的综合写作”。

### 7.5 VerifierAgent

文件：`src/agentic_research_copilot/agents/verifier.py`

职责：检查报告有没有明显质量问题。

它会关注：

- 是否有引用。
- 是否有 unsupported section。
- 是否需要 revision。
- 需要补什么证据。

学习重点：

> verifier 是质量门，不是另一个 reporter。

## 8. Provider 层：真模型调用都在这里

`src/agentic_research_copilot/providers.py` 是理解 LLM 工程化的关键。

这个文件负责把“我要模型做某件事”变成稳定的 API 调用和结构化返回。

### 8.1 为什么需要 provider 抽象

如果你在 planner、researcher、reporter 里到处直接写 HTTP 请求，系统会很乱。

provider 抽象带来几个好处：

- 真实模型和 deterministic test double 可以互换。
- prompt 和 JSON schema 集中管理。
- usage/cost/latency 可以统一记录。
- 不同模型 provider 可以走同一套接口。

### 8.2 核心方法

你要重点看这些方法：

- `clarify_request`
- `draft_plan`
- `supervise_research`
- `decide_researcher_action`
- `compose_report`
- `assess_report`
- `compress_source`
- `contextualize_chunk`
- `extract_knowledge_graph`
- `extract_graph_query`
- `embed_text`
- `embed_texts`

这些方法对应了研究 runtime 的各个智能步骤。

### 8.3 structured JSON 的意义

项目要求模型返回结构化 JSON，而不是随便一段自然语言。

比如 supervisor 需要返回工具调用，researcher 需要返回下一步动作，verifier 需要返回是否需要 revision。

如果没有结构化输出，后面的代码就只能用脆弱的字符串解析。

面试时可以说：

> 我把 LLM 输出当作不稳定外部输入，所以用 Pydantic schema 和 normalization，把模型响应约束成可以被程序消费的数据契约。

### 8.4 normalization 的意义

`providers.py` 里有很多 `_normalize_*` 函数。

这不是啰嗦，而是现实工程需要。

模型可能：

- 少返回字段。
- 返回空字符串。
- 把 list 写成 string。
- 返回多余字段。
- 工具参数格式不稳定。

normalization 负责把这些不稳定输出修正到 schema 可接受的范围。

## 9. RAG 层：为什么不是普通 top-k

`src/agentic_research_copilot/retrieval/store.py` 是这个项目最硬的技术点之一。

它做的不是：

```text
query -> embedding -> top 5 chunks -> answer
```

更接近：

```text
document ingest
  -> chunk
  -> contextualize child chunk
  -> dense index
  -> BM25 index
  -> graph entity/relation extraction

query
  -> dense search
  -> keyword search
  -> graph search
  -> score fusion
  -> rerank
  -> parent/neighbor context expansion
  -> EvidenceItem
```

### 9.1 child chunk

child chunk 是较小的检索单元。

小 chunk 的好处是精确。比如用户问一个具体概念，小 chunk 更容易命中。

坏处是上下文可能不完整。所以系统后面还要做 parent/neighbor expansion。

### 9.2 parent / neighbor context expansion

命中 child chunk 后，系统会补充父级或相邻上下文。

这样 synthesis 阶段拿到的不是孤零零一句话，而是带上下文的证据。

面试时可以这样讲：

> 我用 child chunk 提升命中精度，再用 parent/neighbor expansion 保证生成阶段的上下文完整性。

### 9.3 dense retrieval

Qdrant dense retrieval 负责语义召回。

优点：

- 同义表达也能匹配。
- 用户 query 和文档措辞不完全一致时仍然能找出相关内容。

缺点：

- 对专有名词、版本号、函数名、指标名不一定稳。

### 9.4 SQLite FTS5 / BM25

BM25 负责关键词召回。

它对这些内容特别重要：

- 函数名
- 类名
- repo 名
- 论文名
- 指标名
- 缩写
- 版本号

这就是为什么 dense + BM25 比单一向量检索更稳。

### 9.5 graph signal

项目参考 LightRAG 的思想，抽取 entity 和 relationship。

但要注意边界：

> 这里不是完整 GraphRAG 平台，而是把 graph 当作一个 retrieval signal。

它的作用是：如果 query 里出现某些实体或关系，系统可以把相关 chunk 加入候选，再和 dense/BM25 一起融合。

### 9.6 rerank

融合之后还要 rerank。

原因是 dense、BM25、graph 各自的分数尺度不同，候选也可能有噪音。reranker 根据 query 和候选内容再排序，让最终返回更贴近问题。

### 9.7 读 `retrieval/store.py` 的方法

不要从头读到尾。建议按功能读：

1. 先看 `DocumentStore.add()`，理解文档怎么进入系统。
2. 再看 `_index_document()`，理解 chunk 和索引。
3. 再看 `search()`，理解检索入口。
4. 再看 `_search_qdrant()`、`_search_local()`、`_search_graph()`。
5. 再看 `_merge_keyword_candidates()`、`_merge_graph_candidates()`、`_fuse_dense_keyword_score()`。
6. 最后看 `_rerank()` 和 `_parent_context_for_child()`。

## 10. MCP 层：外部工具不是主产品

`src/agentic_research_copilot/mcp_tools.py` 是外部 MCP 工具适配层。

这个项目现在最推荐接 GitHub MCP，因为它补的是开发事实源：

- repo files
- source code
- issues
- pull requests
- releases

### 10.1 为什么 MCP 不是主角

如果你接一个“deep research MCP”，那它会自己规划、搜索、写报告，和本项目的 planner/supervisor/reporter 重复。

所以这个项目的 MCP 边界应该是：

```text
Researcher -> mcp_tools.py -> GitHub MCP -> EvidenceItem(kind="mcp")
```

也就是说，MCP 只是证据通道，不是替代 runtime。

### 10.2 allowlist

`ARC_MCP_TOOLS` 是 allowlist。

这样做有几个好处：

- demo 更稳定。
- 工具范围可解释。
- 减少模型乱调工具。
- 面试时能讲安全边界。

### 10.3 structured args

GitHub MCP 不是把整个问题当搜索 query。

provider 会尝试提取结构化参数，例如：

- `owner`
- `repo`
- `path`
- `query`
- `issue_number`

这就是为什么 `providers.py` 里有 `_github_repository_hints()`，`mcp_tools.py` 里有 `_payload_for_tool()`。

面试时可以说：

> 我把 MCP 当成结构化外部工具边界。模型先看到 tool catalog，再选择 tool name 和 tool args，系统只允许调用 allowlist 中的工具，并把结果转换成 EvidenceItem。

## 11. Evaluation 层：怎么证明答案有质量

`src/agentic_research_copilot/evaluation.py` 里的 `RAGEvaluator` 是质量评估层。

它不等于完美自动评测，但它解决了一个面试里非常重要的问题：

> 你的系统怎么知道自己答得好不好？

### 11.1 citation coverage

检查报告引用覆盖。

如果报告很多结论没有 citation，就说明 reporter 可能在自由发挥。

### 11.2 evidence sufficiency

检查证据数量和多样性是否足够。

复杂问题不能只靠一个来源。

### 11.3 context precision

检查报告内容和证据上下文是否相关。

这不是严格事实核验，但能作为 proxy metric。

### 11.4 source quality

不同来源质量不同。

比如官方文档、repo 文件、release、issue 可能比随机博客更适合某些问题。

### 11.5 unsupported sections

如果某个 section 没有足够 evidence 支撑，就应该被标记。

这和 verifier 配合，可以触发 revision。

## 12. Trace / Replay：玩具项目和工程项目的分界

很多学生项目只展示最终回答。这个项目的亮点是能展示过程。

一次 run 应该能看到：

- request
- plan
- supervisor decision
- route
- tool calls
- evidence
- report
- evaluation
- checkpoints
- errors

这些信息让你可以回答：

- 为什么用了 web search？
- 为什么用了 GitHub MCP？
- 为什么用了本地 RAG？
- 哪些证据被引用？
- 哪些 section 证据不足？
- 如果失败，失败在哪一步？

面试时可以强调：

> 我把 research agent 当成可观测系统来做，不只关心最终自然语言输出，也关心运行轨迹和质量指标。

## 13. 跑起来：最小学习实验

先用 deterministic / 默认配置跑测试，不要一上来就折腾真实 provider。

### 13.1 安装

```powershell
pip install -e .[dev]
```

如果你要读 PDF、网页文档或真实 MCP，可以再装对应 extra：

```powershell
pip install -e .[dev,documents,mcp]
```

### 13.2 跑测试

```powershell
pytest -q
```

这一步的目的不是为了“显示测试通过”，而是让你知道项目有一条 deterministic 路径，可以不依赖真实模型完成基本行为验证。

### 13.3 启动 API

```powershell
uvicorn agentic_research_copilot.server:create_app --factory --host 127.0.0.1 --port 8000
```

然后访问：

```text
http://127.0.0.1:8000
```

### 13.4 添加一段本地资料

你可以用 API 添加一段项目说明：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/documents `
  -ContentType "application/json" `
  -Body '{"title":"Project Positioning Note","source":"local-note","content":"This project should be presented as an Agentic Research Runtime rather than a commercial Codex replacement. It focuses on planning, evidence routing, Agentic RAG, verification, evaluation, and trace replay."}'
```

### 13.5 提交一次研究

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/research/runs `
  -ContentType "application/json" `
  -Body '{"topic":"How should this project be positioned for AI engineering interviews?","depth":"standard","include_private_docs":true}'
```

保存返回的 `run_id`。

### 13.6 查看 trace 和 evaluation

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/research/runs/<run_id>/trace
Invoke-RestMethod http://127.0.0.1:8000/v1/research/runs/<run_id>/evaluation
```

学习时不要只看 final report。你要看：

- plan 是怎么拆的。
- route 选择了什么。
- evidence 来自哪里。
- evaluation 认为哪里不足。

## 14. 真实 provider 学习路径

真实 provider 不是第一天必须跑，但你最终要理解。

关键配置在 `.env.example` 和 `.env.real.example`。

常见配置包括：

```text
ARC_STRICT_PROVIDERS=true
ARC_MODEL_PROVIDER=openai_compatible
ARC_MODEL_BASE_URL=...
ARC_MODEL_API_KEY=...
ARC_EMBEDDING_PROVIDER=openai_compatible
ARC_EMBEDDING_BASE_URL=...
ARC_EMBEDDING_API_KEY=...
ARC_SEARCH_PROVIDER=tavily
ARC_SEARCH_API_KEY=...
ARC_QDRANT_URL=http://127.0.0.1:6333
ARC_RERANK_PROVIDER=dashscope
ARC_RERANK_API_KEY=...
```

理解这些配置时，不要背环境变量名。你要理解每个 provider 对应 runtime 哪一层：

- model provider：规划、监督、工具决策、报告、验证。
- embedding provider：本地文档向量化和 query embedding。
- search provider：外部 web evidence。
- rerank provider：本地检索候选排序。
- Qdrant：dense vector index。
- SQLite：runs、jobs、trace、BM25、checkpoint。

## 15. GitHub MCP 学习实验

先理解边界，再配置。

推荐配置：

```powershell
$env:ARC_MCP_ENABLED="true"
$env:ARC_MCP_SERVER_URL="https://api.githubcopilot.com/mcp/readonly"
$env:ARC_MCP_TOOLS="search_repositories,get_file_contents,search_code,list_issues,issue_read,search_issues,list_pull_requests,pull_request_read,get_latest_release"
$env:ARC_MCP_AUTH_REQUIRED="true"
$env:ARC_MCP_AUTH_TOKEN="<github-token>"
```

推荐 topic：

```text
Analyze https://github.com/langchain-ai/open_deep_research:
explain its research workflow, inspect relevant implementation files,
summarize issue risks, review recent pull request activity, and report release changes.
Use Tavily for ecosystem context and GitHub MCP for repository-level evidence.
```

观察点：

- provider 是否识别出 `owner/repo`。
- researcher 是否选择 `mcp_tool`。
- `mcp_tool_name` 是哪个。
- `mcp_tool_args` 是否结构化。
- MCP 返回结果是否变成 `EvidenceItem(kind="mcp")`。
- final report 是否引用了 GitHub evidence。

如果 auth 或网络失败，不要慌。这正好说明为什么面试前要保存 run bundle。

## 16. 怎么读测试

测试不是只为 CI 服务。对学习项目来说，测试是最好的“行为说明书”。

建议这样读：

- `tests/test_schemas.py`：看数据契约期望。
- `tests/test_routing.py`：看 route 怎么决定 external/internal/hybrid。
- `tests/test_retrieval.py`：看本地检索行为。
- `tests/test_researcher_mcp.py`：看 MCP 决策和 evidence 转换。
- `tests/test_pipeline.py`：看完整 pipeline 行为。
- `tests/test_server.py`：看 API surface。
- `tests/test_verifier.py`：看报告验证。

读测试时问自己：

- 这个测试证明了哪个工程边界？
- 如果我改坏了这个测试，面试里哪句叙事会不成立？
- 这个测试是真实质量保障，还是只是实现细节？

## 17. 你要真正掌握的 10 个问题

学完以后，用自己的话回答这些问题。

### 17.1 为什么这个项目不是普通 RAG？

因为它不是固定 `query -> top-k chunks -> answer`。

它有：

- planner 拆问题。
- supervisor 路由证据。
- researcher bounded loop。
- web / MCP / local RAG 多证据通道。
- dense + BM25 + graph + rerank。
- citation-backed report。
- verifier/evaluator。
- trace/replay。

### 17.2 为什么需要 schema？

因为 LLM 输出不稳定，而程序需要稳定的数据结构。

schema 把自然语言模型输出变成可验证、可归一化、可测试的工程契约。

### 17.3 为什么需要 supervisor？

因为开放式问题不能直接搜索。

supervisor 的价值是决定：

- 哪些 plan item 要做。
- 每个 item 用哪些工具。
- 证据不够时怎么办。
- 什么时候可以 ResearchComplete。

### 17.4 为什么需要 LangGraph？

因为研究流程有状态、分支、并发、checkpoint、revision loop。

一条 chain 很难表达这些控制流。

### 17.5 为什么 dense retrieval 不够？

dense retrieval 擅长语义相似，但对精确词、代码名、版本号、指标名不一定稳。

所以需要 BM25。

### 17.6 为什么 BM25 也不够？

BM25 擅长关键词，但不懂语义相似。

所以需要 dense。

### 17.7 graph signal 的边界是什么？

它不是完整知识图谱产品。

它只是一个 retrieval signal，用 entity/relation 帮助召回相关 chunk。

### 17.8 GitHub MCP 的价值是什么？

它提供 repository source-of-truth evidence。

Tavily 可以查外部背景，但 GitHub MCP 更适合查源码、issue、PR、release。

### 17.9 为什么删掉 memory？

因为没有真实长期使用数据和强 corpus 时，memory 容易像噱头。

删掉后，项目边界更清楚：专注 research runtime、evidence、RAG、trace、evaluation。

### 17.10 Codex 已经能做，为什么这个项目还有价值？

因为目标不同。

Codex 是成熟产品，这个项目是学习和复现底层机制。你不是证明自己做了更强工具，而是证明你理解并实现了：

- stateful orchestration
- structured tool calls
- evidence routing
- local retrieval
- citation grounding
- quality evaluation
- replayable trace

## 18. 七天学习计划

### Day 1：只看定位和总览

读：

- `README.md`
- `docs/product-positioning.md`
- `docs/architecture.md`
- 本文档第 0-3 章

产出：

- 用 5 句话写出项目定位。
- 画出一张从 API 到 report 的流程图。

### Day 2：读 schema 和 API

读：

- `src/agentic_research_copilot/schemas.py`
- `src/agentic_research_copilot/server.py`

产出：

- 列出一次 run 里最重要的 8 个对象。
- 说明 `/v1/research/runs/{run_id}/trace` 和 `/evaluation` 为什么重要。

### Day 3：读 pipeline 和 graph runtime

读：

- `src/agentic_research_copilot/pipeline.py`
- `src/agentic_research_copilot/graph_runtime.py`

产出：

- 写出 graph 的节点顺序。
- 解释 revision loop 的触发条件。

### Day 4：读 agents 和 provider

读：

- `src/agentic_research_copilot/agents/`
- `src/agentic_research_copilot/providers.py`

产出：

- 用表格写出每个 agent 的输入、输出、职责。
- 找出 5 个 provider 方法分别服务哪个 agent。

### Day 5：读 RAG

读：

- `src/agentic_research_copilot/retrieval/store.py`
- `src/agentic_research_copilot/retrieval/fulltext.py`
- `src/agentic_research_copilot/retrieval/rerank.py`

产出：

- 解释 child chunk 和 parent context。
- 解释 dense + BM25 + graph + rerank 的组合价值。

### Day 6：读 MCP、evaluation、trace

读：

- `src/agentic_research_copilot/mcp_tools.py`
- `src/agentic_research_copilot/evaluation.py`
- `src/agentic_research_copilot/storage.py`
- `src/agentic_research_copilot/telemetry.py`

产出：

- 解释 MCP allowlist。
- 解释 `EvidenceItem(kind="mcp")` 的意义。
- 解释 citation coverage 和 evidence sufficiency。

### Day 7：跑 demo，准备讲解

读：

- `docs/resume-demo-runbook.md`
- `docs/interview-notes.zh-CN.md`
- `examples/eval-dataset.jsonl`

产出：

- 跑一次本地 deterministic run。
- 保存 run_id。
- 查看 report、trace、evaluation。
- 准备 3 分钟项目讲解。

## 19. 三分钟讲解模板

可以这样讲：

```text
这个项目我现在把它定位成 Agentic Research Runtime，不是要做一个商业 Copilot 去和 Codex 比。

它学习的是 Deep Research 这类系统背后的工程机制：用户给一个开放式技术问题后，系统先用 planner 生成 research brief 和 plan item，再由 supervisor 通过结构化 tool call 分派研究单元。researcher 对每个单元跑 bounded loop，按证据缺口选择 web search、GitHub MCP 或本地 Agentic RAG。

本地 RAG 不是简单 top-k，我做了 child chunk 检索、parent/neighbor context expansion、Qdrant dense、SQLite BM25、graph signal fusion 和 rerank。外部证据、本地证据、MCP 证据最后都会统一成 EvidenceItem，reporter 基于 notes 和 evidence 生成 citation-backed report。

最后 verifier 和 evaluator 会检查 citation coverage、evidence sufficiency、source diversity、unsupported sections，并把 plan、route、tool call、evidence、evaluation 都保存成 trace，方便 replay。

所以这个项目的重点不是答案本身，而是一个 research agent run 怎么被规划、执行、验证和复盘。
```

## 20. 常见追问和回答方向

### Q1：为什么不用 Codex？

答：

```text
Codex 作为产品当然更强。我这个项目不是替代它，而是学习和实现这类产品背后的 runtime 机制。重点是我能解释 state graph、structured tool call、evidence routing、Agentic RAG、citation verification、evaluation 和 trace replay 怎么做。
```

### Q2：本地 RAG 是不是硬凑？

答：

```text
如果只是几段短文本，确实直接放 prompt 更简单。所以我没有把本地 RAG 说成主需求。它在这个项目里是 evidence channel，用于论文、架构文档、采用标准等超过 prompt 或需要复用检索的语料。主链路仍然是 research runtime。
```

### Q3：MCP 是不是只是套壳？

答：

```text
不是。这里 MCP 不是完整 agent，而是外部证据工具边界。比如 GitHub MCP 提供 repo/code/issue/PR/release 这些 developer source-of-truth。planner/supervisor/reporter 仍然是本项目自己的 runtime。
```

### Q4：为什么删 memory？

答：

```text
因为没有真实长期使用数据时，memory 很容易变成噱头模块。删除后项目边界更清楚，专注研究编排、证据检索、引用验证、评估和 trace。
```

### Q5：这个项目最大技术难点是什么？

答：

```text
不是单个 API，而是把不稳定的 LLM 决策变成可执行、可观察、可评估的系统。具体包括 schema contract、structured tool decision、bounded researcher loop、多证据融合、citation-backed synthesis、verifier/evaluator 和 replayable trace。
```

## 21. 你接下来最该补的东西

如果你学完文档，下一步不要急着加新 agent。

最该补的是：

1. `run bundle exporter`
2. 一套 Open-source Due Diligence demo bundle
3. 一套 Technical Decision Memo demo bundle
4. 一套 Local Corpus Research demo bundle
5. eval dataset 里更明确的 expected evidence type
6. 前端围绕 evidence、quality gates、trace 的展示 polish

这些东西会让项目更像“可复盘的工程系统”，而不是“我又加了一个功能”。

## 22. 最后提醒

你学习这个项目时，最容易掉进两个坑。

第一个坑：只背概念。

不要只说 LangGraph、RAG、MCP、Evaluation。你要能说出这些概念在代码里对应哪个对象、哪个方法、哪个数据流。

第二个坑：过度产品化。

不要说“我做了一个比 Codex 更好的研究助手”。要说“我做了一个可检查的 research-agent runtime，用来学习和复现这类系统背后的工程机制”。

只要你按这个思路学，这个项目就不是玩具。它会变成你理解 AI 工程系统的一条主线。
