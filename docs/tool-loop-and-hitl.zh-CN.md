# Tool Loop And HITL

本文档解释本项目 v2 的工具循环和 human-in-the-loop 设计。重点不是把所有工具调用都拦下来，而是先建立一套可解释、可观察、可测试的 tool policy 层。

参考资料：

- LangGraph human-in-the-loop: https://docs.langchain.com/oss/python/langgraph/interrupts
- OpenAI Agents SDK tools: https://openai.github.io/openai-agents-python/tools/
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- OpenHands: https://github.com/OpenHands/OpenHands
- Microsoft Agent Framework: https://learn.microsoft.com/en-us/agent-framework/overview/

## 1. Tool Loop 是什么

一个研究型 agent 通常不是：

```text
user question -> LLM -> final answer
```

而是：

```text
user question
-> plan
-> choose tool
-> call tool
-> observe result
-> decide if enough
-> call next tool or finish
-> synthesize report
-> evaluate
```

在本项目里，底层 research runtime 已经有 bounded tool loop：

- `ResearchSupervisor` 决定哪些 plan item 需要研究。
- `Researcher` 在 `web_search`、`mcp_tool`、`ResearchComplete` 之间选择。
- `Retriever` 负责本地知识库、向量、BM25、图信号融合。
- `Reporter` 生成 citation-backed report。
- `Verifier` 和 `RAGEvaluator` 检查质量。

代码路径：

```text
src/agentic_research_copilot/agents/research_supervisor.py
src/agentic_research_copilot/agents/researcher.py
src/agentic_research_copilot/retrieval/store.py
src/agentic_research_copilot/graph_runtime.py
```

## 2. v2 为什么还要 Tool Policy

“代码能调用工具”和“agent 能解释地选择工具”不是一回事。

v1 的问题：

- 工具能力主要藏在代码和 trace 里。
- 前端不知道哪些工具启用、哪些缺 token、哪些需要审批。
- MCP 缺 token 时虽然不会成功调用，但 session 里没有 approval artifact。
- 面试讲 tool loop 时，证据不够直观。

v2 补了三个对象。

### 2.1 AgentToolDefinition

表示工具注册表：

```text
name
channel: web | vector | mcp | local
description
input_schema
enabled
requires_auth
auth_configured
risk_level
approval_required
failure_mode
metadata
```

API：

```text
GET /v1/agent/tools
```

默认策略：

- `web_search`: enabled，low risk，不需要 approval。
- `vector_retrieval`: enabled，low risk，不需要 approval。
- `mcp_tool`: 只有配置可用时 enabled。配置了但缺 auth 时显示 unavailable，并需要 approval。

### 2.2 ToolInvocation

表示某次工具调用或被策略层拦下的工具动作：

```text
invocation_id
session_id
run_id
tool_name
status: pending_approval | running | completed | failed | skipped
arguments
result_preview
evidence_ids
latency_ms
error
metadata
```

API：

```text
GET /v1/agent/sessions/{session_id}/tool-invocations
```

v2 做了两类同步：

- MCP unavailable 会生成 `pending_approval` invocation。
- research runtime 完成后，`RunTraceEvent(kind="tool_call")` 会同步成 session 里的 tool invocation。

### 2.3 ApprovalRequest

表示需要用户确认或知情的高风险动作：

```text
approval_id
session_id
invocation_id
reason
requested_action
status: pending | approved | rejected | expired
metadata
```

API：

```text
POST /v1/agent/sessions/{session_id}/approvals/{approval_id}/approve
POST /v1/agent/sessions/{session_id}/approvals/{approval_id}/reject
```

v2 的 approval 不是完整 durable interrupt。它的作用是：

- 明确告诉用户：GitHub MCP 配了，但 token 缺失。
- 不把 MCP evidence 伪装成成功。
- 允许用户在 UI 里 approve/reject 这个状态。
- 把 action 变成可审计的 step 和 invocation。

## 3. 为什么不直接做 LangGraph Interrupt

LangGraph interrupt 很适合真正的可暂停、可恢复、人类审批工作流。但当前项目的底层 research runtime 已经比较完整，如果为了 v2 强行把每个 tool call 都改成 interrupt，会带来三个问题：

1. 改动面很大，容易破坏已有 research pipeline。
2. 静态 Web UI 没有 WebSocket/SSE，用户体验会被半成品 interrupt 卡住。
3. 秋招讲项目时，重点应该是能解释工程边界，而不是堆一个还不稳定的高级特性。

所以 v2 的选择是：

```text
先做可观察 approval
再做真正 durable interrupt
```

v3 可以进一步升级：

- research runtime 在执行 medium/high risk tool 前写 pending step。
- graph checkpoint 暂停。
- UI approve 后恢复 graph。
- reject 后走 alternative route 或降级到 web/local evidence。

## 4. GitHub MCP 的正确降级

GitHub MCP 是本项目推荐的外部工具，因为它补的是 Web 和本地知识库不容易稳定拿到的开发者证据：

- repo metadata
- code search
- issue
- pull request
- release
- changelog

但 MCP token 缺失时，正确行为不是“继续假装有 GitHub 证据”。

正确行为：

```text
MCP configured = true
auth token configured = false
available = false
tool registry: mcp_tool unavailable
approval_required = true
session approval request: pending
tool invocation: pending_approval 或 skipped
research runtime: 允许 web + local 继续跑
```

脚本：

```powershell
python scripts/check_github_mcp.py
```

无 token 时允许失败，但必须输出明确 JSON，提示需要这些变量之一：

```text
ARC_MCP_AUTH_TOKEN
GH_TOKEN
GITHUB_TOKEN
GITHUB_PERSONAL_ACCESS_TOKEN
```

## 5. Step Stream 怎么和 Tool Loop 对齐

v2 新增 `AgentRunStep`：

```text
message | planning | approval | tool_call | retrieval | research | report | verification | evaluation | failure
```

运行中：

- 用户发消息，写 `message` step。
- 需要澄清，写 skipped `planning` step。
- 生成计划，写 completed `planning` step。
- 确认计划，写 running `research` step。
- MCP 不可用，写 pending `approval` step。

运行完成后：

- `RunTraceEvent` 同步成 session-visible step。
- `tool_call` trace 同步成 `ToolInvocation`。
- report、verification、evaluation 也出现在 steps 里。

API：

```text
GET /v1/agent/sessions/{session_id}/steps
GET /v1/agent/sessions/{session_id}/events
```

Workbench 现在用 polling，每 2 秒刷新一次 session bundle。

## 6. 面试讲法

可以这样讲：

> 我把工具调用分成 registry、policy、invocation、approval 四层。registry 描述 agent 能用什么工具，policy 判断工具是否启用和是否需要审批，invocation 记录具体调用或被拦截的动作，approval 让用户对风险动作做确认。v2 先实现 MCP unavailable 的 gating 和可观测 approval，后续可以升级成 LangGraph durable interrupt。

被问“为什么不直接把 shell/file write 也做进去”：

> 因为这个项目定位是研究型 agent，不是 OpenHands/Codex 替代品。v2 明确不支持 destructive local tools，先把 web、vector、MCP 证据链做扎实，避免工具权限边界失控。

被问“approval 有什么实际用”：

> 实际用处是让 agent 在不确定或缺权限时诚实降级。比如 GitHub MCP 缺 token，系统不会编造 GitHub evidence，而是把这个状态写入 tool registry、approval request、tool invocation 和 UI。

