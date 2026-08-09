# providers.py 代码阅读指南

对应源码：
```text
D:\kn\projects\agentic-research-copilot\src\agentic_research_copilot\providers.py
D:\kn\projects\agentic-research-copilot\src\agentic_research_copilot\provider_base.py
D:\kn\projects\agentic-research-copilot\src\agentic_research_copilot\deterministic_provider.py
```

一句话定位：
> `providers.py` 是这个项目的真实模型适配层。`provider_base.py` 放统一接口和 `ModelUsage`，`providers.py` 放 OpenAI-compatible 真模型实现和 builder，`deterministic_provider.py` 放测试/离线用的 deterministic test double。

你可以把它理解成整个系统的“模型适配器”：

```text
PlannerAgent.draft()
-> model_provider.draft_plan(...)

SupervisorAgent.decide()
-> model_provider.supervise_research(...)

ResearchAgent.collect_iterative()
-> model_provider.decide_researcher_action(...)

ReporterAgent.compose()
-> model_provider.compose_report(...)

VerifierAgent.assess()
-> model_provider.assess_report(...)
```

所以你的直觉是对的：只看 `agents/` 会觉得没什么，因为真正的大模型方法都在 provider 层。现在 deterministic 已经拆到单独文件，读 `providers.py` 会更接近真实运行主链路。

## 1. 这个文件解决什么问题

大模型应用里最容易混乱的是：

1. 不同 provider 的调用方式不一样。
2. prompt 容易散落在各个 Agent 里。
3. 模型输出不稳定，需要结构化校验。
4. 测试不能每次都真实调用 API。

这组 provider 文件主要解决这些问题：

| 问题 | 这里的设计 |
| --- | --- |
| 模型接口不统一 | 用 `ResearchModelProvider` 协议统一方法 |
| 真模型和测试模型不同 | 真模型在 `providers.py`，测试替身在 `deterministic_provider.py` |
| 模型输出不稳定 | 每个方法都返回 Pydantic schema |
| 要记录成本和延迟 | 每次调用返回 `ModelUsage` |
| 要支持 embedding | 同一个 provider 协议里包含 `embed_text` / `embed_texts` |

## 2. 先看三个核心对象

### `ModelUsage`

位置：
```text
provider_base.py:27-37
```

它记录一次模型调用的用量：

- provider
- model
- prompt tokens
- completion tokens
- cost
- latency

这些信息后面会进入 trace，方便你解释“这个 Agent 到底有没有调模型、调了哪个模型、花了多少 token”。

### `ResearchModelProvider`

位置：
```text
provider_base.py:40-119
```

这是一个 `Protocol`，也就是接口约定。它规定所有模型 provider 必须实现这些方法：

| 方法 | 被谁调用 | 作用 |
| --- | --- | --- |
| `clarify_request` | `ResearchCopilot.clarify` | 判断用户问题是否需要澄清 |
| `draft_plan` | `PlannerAgent` | 生成 research brief 和 plan |
| `supervise_research` | `SupervisorAgent` | 生成 ODR 风格 tool calls |
| `decide_researcher_action` | `ResearchAgent` | 决定研究员下一步是 search、think、MCP 还是 complete |
| `compose_report` | `ReporterAgent` | 把章节草稿和证据合成最终报告 contract |
| `assess_report` | `VerifierAgent` | 检查报告质量，决定是否返工 |
| `compress_source` | `SourceReader` | 把网页原文压缩成可引用证据 |
| `contextualize_chunk` | `DocumentStore` | 给文档 chunk 生成索引用上下文前缀 |
| `embed_text` / `embed_texts` | `MemoryStore` / `DocumentStore` | 生成向量 |

这就是模型能力的总接口。

### 两个实现类

| 类 | 文件 | 用途 |
| --- | --- |
| `OpenAICompatibleResearchModelProvider` | `providers.py` | 真模型 provider，通过 OpenAI-compatible API 调 chat/completions 和 embeddings |
| `DeterministicResearchModelProvider` | `deterministic_provider.py` | 本地 deterministic test double，用于测试、CI、离线复现 |

## 3. 本地 deterministic 要不要删

不建议现在删。

原因不是“项目还想靠假数据包装”，而是它承担了几个工程职责：

1. 单元测试和 CI 不能依赖真实 API key。
2. 离线开发需要稳定、可复现的输出。
3. 它实现了和真模型一样的 `ResearchModelProvider` 接口，可以做 contract test。
4. strict real-provider demo 已经能通过配置禁用静默 fallback。

真正需要对面试官强调的是：

> deterministic provider 是测试替身和离线脚手架，不是真实 demo 路径。真实运行用 `OpenAICompatibleResearchModelProvider`，并通过 `ARC_STRICT_PROVIDERS=true` 在启动阶段拒绝 deterministic model、deterministic embedding、无 key search、规则 reranker、内存 Qdrant 等配置。

也就是说，正确做法不是删掉 deterministic，而是把边界讲清楚：

```text
测试 / CI / 离线复现 -> deterministic
真实 demo / 面试展示 -> openai_compatible + strict_providers
```

为了让主链路更好读，代码现在已经把它拆出去了：

```text
providers.py               -> 真模型 provider + builder
provider_base.py           -> ModelUsage + ResearchModelProvider 协议
deterministic_provider.py  -> 测试替身和离线启发式实现
```

## 4. 真模型类：`OpenAICompatibleResearchModelProvider`

位置：
```text
providers.py:67-484
```

这是你现在最应该看的类。

它支持任何 OpenAI-compatible 的服务，只要对方提供：

- `/chat/completions`
- `/embeddings`
- Bearer token 认证
- JSON response

构造函数接收：

- `base_url`
- `api_key`
- `chat_model`
- `embedding_model`
- `timeout_seconds`
- `temperature`
- `embedding_dimensions`

这说明项目没有把自己绑定死在某一个供应商上。DeepSeek、DashScope、OpenAI-compatible relay 等，只要接口兼容，都可以接。

## 5. 真模型方法逐个看

### `clarify_request`

位置：
```text
providers.py:89-118
```

作用：
```text
判断用户问题是否足够具体，是否需要先问澄清问题。
```

输入包括：

- `ResearchRequest`
- `CorpusProfile`
- `memory_records`

输出是：

- `ClarificationContract`
- `ModelUsage`

这个方法对应 Open Deep Research 里的 clarify phase。它不会随便问问题，而是只有在 scope、目标受众、决策背景、来源类型缺失时才问一个简短问题。

### `decide_researcher_action`

位置：
```text
providers.py:120-171
```

作用：
```text
在研究员循环里决定下一步动作。
```

它给模型的输入包括：

- 当前 `PlanItem`
- 可用工具 `available_tools`
- 已经搜索过的 query
- 已收集 evidence
- 当前 gaps
- 当前第几轮 iteration
- 最大轮次 max_iterations

模型必须选一个动作：

| action | 含义 |
| --- | --- |
| `think_tool` | 先反思，不立刻搜 |
| `web_search` | 继续外部搜索 |
| `mcp_tool` | 调配置好的 MCP 工具 |
| `ResearchComplete` | 当前研究单元结束 |

这个方法让 `ResearchAgent.collect_iterative()` 不再是固定规则搜索，而是一个可由模型控制的 bounded loop。

### `draft_plan`

位置：
```text
providers.py:173-209
```

作用：
```text
把用户 topic 拆成 research brief 和 3-5 个 PlanItem。
```

它要求模型做到：

- 写清楚研究目标和约束。
- 每个 plan item 是独立可研究的问题。
- 每个 plan item 有 `purpose`。
- 每个 plan item 有更适合检索的 `search_query`。
- 如果是 revision，要吸收 `revision_notes`。
- 如果有 private docs，要包含能被本地文档 grounding 的问题。

这个方法对应 `PlannerAgent.draft()`。

### `supervise_research`

位置：
```text
providers.py:211-256
```

作用：
```text
让监督者决定哪些研究单元要执行、每个单元用什么工具、怎么改写 query。
```

它输出的是 `SupervisorDecisionContract`，里面最关键的是 `tool_calls`。

这些 tool calls 模仿 Open Deep Research 风格：

- `think_tool`
- `ConductResearch`
- `ResearchComplete`

每个 `ConductResearch` 应该带：

- `plan_item_ids`
- `mode`
- `selected_tools`
- `web_queries`
- `internal_queries`
- `memory_query`
- `min_evidence`
- `min_sources`
- `sufficiency_criteria`

这一步是 agentic 的关键，因为它不是固定路线，而是让模型根据 plan、route hints、corpus profile 和 memory 来决定检索策略。

### `assess_report`

位置：
```text
providers.py:258-282
```

作用：
```text
检查最终报告有没有问题，是否需要返工。
```

输入包括：

- report
- evidence
- plan
- revision_count
- max_revisions

输出是 `VerificationContract`，包括：

- issues
- critical_issues
- should_revise
- revision_reason
- confidence
- coverage_score

这个方法对应 `VerifierAgent.assess()`，它影响 `graph_runtime.py` 里是否走 revision loop。

### `compose_report`

位置：
```text
providers.py:284-327
```

作用：
```text
把 pipeline 生成的 topic 章节草稿和 evidence_index 合成最终 ReporterContract。
```

这里特别重要：

> `compose_report` 不能发明 citations。它只能用 `citation_indexes` 引用传入的 `evidence_index`。

这就是为什么前面修 `_build_sections` 很关键：如果传给 reporter 的 draft sections 偏题，真模型也会在偏题草稿上继续写。现在 `_build_sections` 已经改为围绕 `plan + notes + evidence` 生成 topic 相关章节。

### `compress_source`

位置：
```text
providers.py:329-361
```

作用：
```text
把网页或原文内容压缩成下游可用证据。
```

它要求模型保留：

- 具体事实
- 数字
- 日期
- 命名实体
- caveats
- 和 query 相关的 key excerpts

输出是 `SourceCompressionContract`，会被 `SourceReader` 用来提升搜索结果质量。

### `contextualize_chunk`

位置：
```text
providers.py:363-405
```

作用：
```text
在文档入库时，为 chunk 生成 contextual retrieval prefix。
```

这不是回答用户问题，而是给 RAG 索引用的。

它会基于：

- document title
- source
- metadata
- document excerpt
- chunk text
- chunk index

生成 50-100 token 左右的上下文，让孤立 chunk 在 embedding 和 BM25 检索时更有语境。

### `embed_text` / `embed_texts`

位置：
```text
providers.py:407-427
```

作用：
```text
调用 OpenAI-compatible embeddings 接口生成向量。
```

它会请求：

```text
POST /embeddings
```

payload 包括：

- model
- input
- dimensions

返回向量和 `ModelUsage`。

## 6. 真模型调用底座：`_chat_structured`

位置：
```text
providers.py:429-464
```

这是所有 chat 类方法共用的底座。

它做了这几件事：

1. 构造 `/chat/completions` 请求。
2. system prompt 放角色说明。
3. user message 里同时塞入：
   - schema
   - input payload
4. 要求 `response_format={"type": "json_object"}`。
5. 取出 response content。
6. 用 `_extract_json_object(...)` 提取 JSON。
7. 用 `response_model.model_validate_json(...)` 转成 Pydantic schema。
8. 记录 `ModelUsage`。

这条链路是项目“大模型工程化”的核心：

```text
prompt + payload + schema
-> OpenAI-compatible chat completion
-> JSON object
-> Pydantic validation
-> typed contract
-> Agent / Graph 节点继续执行
```

面试里可以这样讲：

> 我没有让 LLM 自由输出文本再手写解析，而是把每个模型能力设计成结构化 contract。provider 统一把 schema 和 input 发给 OpenAI-compatible 接口，返回后用 Pydantic 校验成明确的数据对象，再交给 LangGraph 节点继续执行。

## 7. provider 构建函数

### `build_model_provider`

位置：
```text
providers.py:487-501
```

逻辑：

- 如果 `settings.model_provider == "openai_compatible"` 且有 `model_base_url`，返回真模型 provider。
- 否则返回 deterministic provider。

注意：

> 在非严格模式下，它允许回到 deterministic，方便测试和离线开发。严格模式下，`ResearchCopilot.__init__` 会先调用 `require_real_provider_config(...)`，配置不真实就直接失败。

### `build_embedding_provider`

位置：
```text
providers.py:503-523
```

逻辑：

- `embedding_provider="deterministic"`：本地 hash embedding。
- `embedding_provider="openai_compatible"`：单独的 embedding endpoint。
- `embedding_provider="model"`：复用主 model provider。

这让 chat model 和 embedding model 可以分开配置。

## 8. 后处理和防御性 helper

文件底部还有一批 helper，主要做三类事情：

| helper | 作用 |
| --- | --- |
| `_normalize_clarification_contract` | 清理 clarification 输出，避免空问题或空确认 |
| `_normalize_researcher_action` | 修正非法 action，例如没有 MCP 工具却选择 `mcp_tool` |
| `_normalize_chunk_context_contract` | 限制 chunk context 长度和 key terms |
| `_extract_chat_content` | 从 OpenAI-compatible response 里取 message content |
| `_extract_json_object` | 从模型输出里提取 JSON |
| `_scalar_metadata` | 控制传给 chunk contextualizer 的 metadata |
| `_limit_words` | 限制模型返回的上下文长度 |

这些 helper 的意义是：

> 即使模型输出有小偏差，也尽量在 provider 层把它规范成稳定 contract，不把脏数据扩散到 graph runtime。

deterministic 相关的 `_hashed_dense_vector`、`_heuristic_source_compression`、`_heuristic_chunk_context` 已经挪到 `deterministic_provider.py`，读真模型链路时可以先跳过。

## 9. 读代码顺序

建议按这个顺序读：

1. `ModelUsage`
2. `ResearchModelProvider`
3. `OpenAICompatibleResearchModelProvider._chat_structured`
4. `OpenAICompatibleResearchModelProvider.draft_plan`
5. `OpenAICompatibleResearchModelProvider.supervise_research`
6. `OpenAICompatibleResearchModelProvider.decide_researcher_action`
7. `OpenAICompatibleResearchModelProvider.compose_report`
8. `OpenAICompatibleResearchModelProvider.assess_report`
9. `build_model_provider` / `build_embedding_provider`
10. 最后看 `deterministic_provider.py`，把它理解成 test double

不要一开始就陷进 deterministic 的启发式规则里。你秋招讲项目时，重点应该讲真模型 provider 和结构化 contract。

## 10. 面试时怎么讲

可以这样讲：

> 这个项目没有把 prompt 散落在每个 agent 里，而是抽象了 `ResearchModelProvider`。Planner、Supervisor、Researcher、Reporter、Verifier 都只依赖这个接口。真实运行时用 `OpenAICompatibleResearchModelProvider`，它把 schema 和 input payload 发给 `/chat/completions`，要求 JSON object，然后用 Pydantic 校验成结构化 contract。测试和 CI 使用 deterministic provider 作为 test double，严格 demo 模式通过 `ARC_STRICT_PROVIDERS=true` 禁止 deterministic 和无 key fallback。

如果面试官问“为什么不删本地兜底”，你可以说：

> 我把 deterministic 当成测试替身，而不是产品路径。它保证 CI 和离线测试稳定；真实 demo 通过 strict provider config 运行，启动时会检查 chat model、embedding、search、reranker、Qdrant 和 LangGraph checkpoint 配置，避免悄悄退回本地实现。

## 11. 你读完应该能回答的问题

1. 为什么 `agents/` 看起来很薄。
2. `ResearchModelProvider` 统一了哪些模型能力。
3. 真模型调用为什么要带 schema。
4. `compose_report` 为什么只能引用 `evidence_index`。
5. deterministic provider 为什么不等于玩具路径。
6. strict provider mode 如何防止真实 demo 悄悄 fallback。

如果这些问题能答清楚，你就能把这个项目的大模型工程化部分讲明白。
