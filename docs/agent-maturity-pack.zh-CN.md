# Agent Maturity Pack v2

本文档解释这次 v2 升级到底在补什么，以及为什么这些点比“能对话、能跑研究任务”更像主流开源 agent 项目的工程形态。

参考资料：

- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- Open Deep Research: https://github.com/langchain-ai/open_deep_research
- CrewAI docs: https://docs.crewai.com/
- OpenHands: https://github.com/OpenHands/OpenHands
- Mem0 docs: https://docs.mem0.ai/platform/overview
- Microsoft Agent Framework: https://learn.microsoft.com/en-us/agent-framework/overview/

## 1. 主流 Agent 项目在卷什么

现在秋招里说 agent，不能只说“我把用户问题发给大模型，再调用几个工具”。成熟项目一般在卷这些工程能力：

1. **Stateful execution**
   - agent 有 session、run、step、message、checkpoint。
   - 长任务不靠一个函数调用硬撑，状态可以被查询、恢复、复盘。
   - LangGraph 强调 long-running、stateful agent、persistence、human-in-the-loop。

2. **Tool loop**
   - agent 不是一次 prompt，而是不断选择工具、执行工具、观察结果、决定下一步。
   - OpenAI Agents SDK 把 agents、tools、handoffs、guardrails、sessions、tracing 作为核心原语。
   - Open Deep Research 则体现 plan -> research -> compress -> report -> eval 的研究型 workflow。

3. **Human-in-the-loop**
   - 人不只是最后看答案，而是在计划、风险工具、关键决策处审批。
   - 研究任务成本高，直接开跑容易浪费 token 和时间，所以 plan confirmation 是合理边界。
   - 更成熟的系统会支持 durable interrupt；本项目 v2 先实现可观察 approval 和 MCP gating。

4. **Memory and knowledge**
   - session memory 解决当前对话上下文。
   - user/project memory 解决跨 session 的偏好和团队约束。
   - Mem0 的核心启发是：从对话中抽取事实，存储，检索时只召回相关 memory，避免用户重复说同样约束。

5. **Observability**
   - agent 每一步在干什么、用了什么工具、拿到了多少证据、哪里失败，都要可见。
   - OpenHands 这种工程 agent 很强调事件流、工具边界、运行轨迹和可控环境。
   - LangSmith、CrewAI、OpenAI tracing 也都说明 observability 已经是 agent 工程的核心能力。

6. **Evaluation and guardrails**
   - 不是“生成了报告就算成功”。
   - 要评估 citation、context recall、faithfulness、工具选择、约束覆盖。
   - 本项目这次重点补 `constraint_coverage`，因为真实实验里一度出现覆盖偏低，说明约束写进 memory 后不一定稳定进入最终报告。

## 2. 本项目 v1 差在哪里

v1 已经有：

- `ConversationalResearchAgent`
- SQLite memory
- plan confirmation
- background research job
- trace/evaluation
- 静态 Agent Workbench

但 v1 还偏“能跑通”：

- UI 只能事后看 trace，运行时阶段不够清楚。
- 工具是代码能调用，但没有显式 tool registry、risk、auth、approval。
- memory 能写入，但缺少 extraction result 和 memory quality 视角。
- 团队约束会进入 prompt，但最终报告是否逐条覆盖没有质量门。
- MCP token 缺失时能 fail fast，但 session 里没有统一的 tool policy 和 approval artifact。

所以 v2 的目标不是重写成 LangGraph/CrewAI/OpenHands，而是把它们共同强调的 agent 工程能力补到当前项目里。

## 3. v2 补了什么

核心链路变成：

```text
Agent Session
-> Memory + Local KB
-> Plan Draft
-> Confirmation
-> Step Stream
-> Tool Policy / Approval
-> Research Runtime
-> Constraint Coverage Gate
-> Report / Trace / Eval
-> Learning Docs
```

### 3.1 AgentRunStep

代码路径：

- `src/agentic_research_copilot/schemas.py`
- `src/agentic_research_copilot/storage.py`
- `src/agentic_research_copilot/agent.py`
- `src/agentic_research_copilot/server.py`

新增 `AgentRunStep`，把 session 内发生的动作变成一等公民：

- user message received
- clarification required
- plan draft generated
- research job started
- run trace synchronized
- report generated
- verification/evaluation completed
- approval requested/resolved

API：

```text
GET /v1/agent/sessions/{session_id}/steps
GET /v1/agent/sessions/{session_id}/events
```

面试讲法：

> 我没有只保存最终报告，而是把 agent session 的关键阶段都落成 step。底层 research runtime 仍然有 RunTraceEvent，v2 会把 trace 同步成 session-visible steps，前端可以轮询展示运行阶段。

### 3.2 Tool Registry / Invocation / Approval

代码路径：

- `src/agentic_research_copilot/schemas.py`
- `src/agentic_research_copilot/agent.py`
- `src/agentic_research_copilot/server.py`
- `apps/web/index.html`

新增对象：

- `AgentToolDefinition`
- `ToolInvocation`
- `ApprovalRequest`

默认工具：

- `web_search`: 公开 web 证据，低风险。
- `vector_retrieval`: 本地知识库和 project memory，低风险。
- `mcp_tool`: 外部 MCP 工具，默认中风险，缺 auth 或未知工具时需要 approval/gating。

API：

```text
GET  /v1/agent/tools
GET  /v1/agent/sessions/{session_id}/tool-invocations
POST /v1/agent/sessions/{session_id}/approvals/{approval_id}/approve
POST /v1/agent/sessions/{session_id}/approvals/{approval_id}/reject
```

面试讲法：

> v2 先实现 tool policy 的可观测层：每个工具有 channel、auth、risk、approval_required。GitHub MCP 缺 token 时不会伪装成功，而是生成 pending approval 和 skipped invocation。真正中断底层 tool loop 的 durable interrupt 留给 v3。

### 3.2.1 Skill Packs

v2 还把 skill 从“prompt 片段”升级成“可加载能力包”：

- `skill.json` 描述场景、输入、评估重点和脚本清单
- `SKILL.md` 描述操作步骤和边界
- `skill_registry.py` 负责扫描、注册和执行受控 preflight
- `skill_registry` 只允许 manifest 声明的本地脚本，使用 JSON stdin/stdout 和超时

这件事的价值不在于“模型会跑脚本”，而在于：

- skill 可以被展示出来
- skill 可以被解释
- skill 可以影响 planning
- skill 可以把结果写回 session 和 export bundle

它仍然不是插件市场，也不是通用 shell agent；它只是把技能做成了一个小而真实的工程边界。

### 3.3 Memory Quality

代码路径：

- `src/agentic_research_copilot/agent.py`
- `src/agentic_research_copilot/storage.py`
- `docs/memory-and-constraint-eval.zh-CN.md`

新增 `MemoryExtractionResult`，记录一次 user message 后：

- 候选 memory
- accepted
- rejected
- reason
- extractor metadata

API：

```text
GET /v1/agent/sessions/{session_id}/memory/evaluation
```

这个 evaluation 现在是 proxy：

- `memory_precision = accepted / candidates`
- `memory_recall = 有 accepted memory 的用户消息数 / 用户消息数`

它不是严格学术指标，但能暴露产品问题：用户明明给了团队约束，agent 是否真的抽出来、保存了、后续用了。

### 3.4 Constraint Coverage Gate

代码路径：

- `src/agentic_research_copilot/constraint_evaluation.py`
- `src/agentic_research_copilot/pipeline.py`
- `src/agentic_research_copilot/agent.py`
- `scripts/run_memory_constraint_eval.py`

新增 `ConstraintCoverage`，逐条检查 hard constraint 是否被报告或证据覆盖：

- constraint content
- covered
- matched sections
- matched evidence
- confidence
- reason

阈值：

- `< 0.6`: evaluation notes 记录 warning。
- `< 0.4`: 标记 evaluation failed，但保留报告。

API：

```text
GET /v1/research/runs/{run_id}/constraint-coverage
```

面试讲法：

> 真实实验里 constraint recall 低不是坏事，而是产品问题被评估系统抓出来了。v2 把团队约束提升成 hard constraints，并在 reporter/evaluator 后增加 constraint coverage gate，避免 memory 只停留在 prompt 里。

## 4. Workbench 现在怎么看

静态 UI 仍在：

```text
apps/web/index.html
```

右侧 Inspector 现在有四类检查区：

- `Plan`: brief、plan items、确认按钮。
- `Memory`: user/project/session memory、extraction result、memory quality、delete。
- `Tools`: tool registry、MCP auth status、approval requests、tool invocation history、steps、routes、trace。
- `Quality`: citation、faithfulness、context recall、constraint coverage、evaluation notes。

session `researching` 时每 2 秒轮询一次 bundle。v2 没引入 WebSocket，避免为了炫技增加前端构建复杂度。

## 5. 和参考项目的关系

本项目借鉴的是工程思想，不是把别人项目包一层。

| 参考项目 | 借鉴点 | 本项目实现 |
| --- | --- | --- |
| LangGraph | durable/stateful workflow、checkpoint、HITL、streaming 思想 | 底层 research graph + SQLite trace/checkpoint + polling steps |
| OpenAI Agents SDK | tools、handoffs、guardrails、sessions、tracing | tool registry、approval、agent session、trace/eval |
| Open Deep Research | plan -> research -> report -> eval | planner、research supervisor、reporter、verifier/evaluator |
| CrewAI | memory、knowledge、flows、guardrails、observability | SQLite memory、local KB、quality inspector |
| OpenHands | 实时事件、工具边界、workspace 控制台 | Workbench + tool boundary + non-destructive policy |
| Mem0 | memory extraction / recall / evaluation | SQLite memory + extraction result + memory evaluation |
| Microsoft Agent Framework | agent vs workflow 边界、state、telemetry、long-running HITL | chat facade + workflow runtime 解耦 |

## 6. 秋招怎么讲

一句话：

> 我做了一个 conversational research agent runtime，面向技术采用评审。它支持 session memory、interactive planning、human confirmation、tool policy、approval、trace、constraint coverage 和 evaluation。

三段式：

1. **为什么做**
   - 技术决策不是普通 RAG 问答。
   - 同一个开源库是否可用，取决于团队栈、部署约束、维护能力、风险偏好。

2. **怎么做**
   - chat session 收集目标和约束。
   - memory extractor 保存长期偏好和项目约束。
   - planner 生成计划，用户确认后才启动 research job。
   - runtime 用 web/local/MCP 工具收集证据，产出 report/trace/eval。
   - v2 增加 step stream、tool policy、approval、memory eval、constraint gate。

3. **难点**
   - 如何避免 agent 无限跑。
   - 如何让工具调用可解释。
   - 如何让 memory 不变成垃圾桶。
   - 如何证明报告覆盖了用户约束。
   - 如何在 MCP token 缺失时明确降级，不假装成功。

诚实边界：

> 它不是 Codex/OpenHands/Deep Research 替代品，也不是完整通用 agent 平台。它是一个单用户、本地、可解释的研究型 agent runtime，用来学习和展示 agent 工程机制。
