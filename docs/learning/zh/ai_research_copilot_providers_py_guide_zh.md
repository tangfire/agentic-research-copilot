# `providers.py` 阅读指南

## 这个文件为什么重要

如果你想知道“真模型到底在哪里调用”，就看 `providers.py`。

`agents/` 只是封装。`providers.py` 才是把 request 发给大模型、要求模型返回结构化 JSON 的地方。

## Provider 是什么

Provider 是模型适配层。

它负责：

- 调 chat model。
- 调 embedding model。
- 传 system prompt。
- 传 user payload。
- 传 JSON schema。
- 解析模型返回。
- 记录 tokens、latency、provider、model。

## 核心类

### `OpenAICompatibleResearchModelProvider`

这是实模型 provider。只要服务兼容 OpenAI API 风格，就可以接入。

重要初始化参数：

- `base_url`
- `api_key`
- `chat_model`
- `embedding_model`
- `timeout_seconds`
- `temperature`
- `embedding_dimensions`

## 关键方法

### `clarify_request`

判断用户问题是否太模糊。

输出 `ClarificationContract`：

- 是否需要澄清。
- 问什么问题。
- 如果不用澄清，就给 verification。

### `draft_plan`

生成研究计划。

输入：

- `ResearchRequest`
- `CorpusProfile`
- revision 信息

输出：

- `PlannerContract`

### `supervise_research`

让模型当 research supervisor。

它必须返回：

- `think_tool`
- `ConductResearch`
- `ResearchComplete`

每个 `ConductResearch` 要说明：

- 研究哪个 plan item。
- 用什么工具。
- 用哪些 queries。
- 证据够不够的标准是什么。

### `decide_researcher_action`

让模型决定 researcher 下一步做什么。

可选动作：

- `web_search`
- `mcp_tool`
- `think_tool`
- `ResearchComplete`

这是 bounded researcher loop 的核心。

现在这里还会收到 `mcp_tools` catalog。模型如果选择 `mcp_tool`，需要返回：

- `mcp_tool_name`：具体工具名。
- `mcp_tool_args`：结构化参数。

所以 GitHub MCP 的调用不是靠程序猜，而是让模型根据工具 schema 决定参数，再由 `mcp_tools.py` 执行。

### `compose_report`

把 sections 和 evidence 交给模型写报告。

注意：模型只能引用已有 evidence。代码会把 citation indexes 映射回真实 `EvidenceItem`。

### `assess_report`

让模型检查报告质量。

### `compress_source`

把 provider raw content 压缩成摘要和关键摘录。

### `contextualize_chunk`

在文档入库时，为 chunk 生成 contextual retrieval prefix。意思是：给孤立 chunk 补一段上下文描述，让检索时更容易命中。

### `extract_knowledge_graph`

从 chunk 中抽取实体和关系。

这不是完整 GraphRAG，只是给检索增加 graph signal。

### `extract_graph_query`

把用户 query 拆成：

- local keywords：偏实体、局部概念。
- global keywords：偏关系、主题。

然后 retrieval 层可以用这些关键词查 graph index。

### `embed_text` / `embed_texts`

生成向量，供 Qdrant 检索使用。

## `_chat_structured`

这是结构化模型调用的底层方法。

它做的事：

1. 拼 messages。
2. 调 OpenAI-compatible `/chat/completions`。
3. 要求模型输出 JSON。
4. 解析成 Pydantic model。
5. 返回 contract 和 usage。

## `deterministic_provider.py` 是什么

它是测试替身，不是真模型。

用途：

- 离线测试。
- CI。
- 没有 key 时跑通契约。

现在你本地 strict provider 已经用真 key，所以学习主线应该看 `providers.py`，不是只看 deterministic fallback。

## 你要能回答

- 哪些决策是 LLM 做的？
- 每个 LLM 决策必须返回哪个 schema？
- 为什么 agent 文件看起来薄？
- 为什么结构化 JSON 对工业化大模型应用很重要？
- graph extraction 为什么只是检索信号，不是完整知识图谱系统？
