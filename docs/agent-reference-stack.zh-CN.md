# Agent 参考栈

这个项目已经从“提交研究任务的 workflow runtime”升级为更接近主流开源 agent 工程形态的研究型 agent：有 session、memory、计划确认、step stream、tool registry、approval、constraint coverage、run 结果回挂和静态 Agent Workbench。

后续继续迭代时，建议按下面这套开源参考栈来抄模式，不要整套照搬。

## 1. 研究主流程

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Open Deep Research](https://github.com/langchain-ai/open_deep_research)

抄这些：

- 任务拆解
- 计划 -> 搜索 -> 压缩 -> 报告 -> 验证
- checkpoint / resume
- human-in-the-loop
- structured output
- evaluation

这两套最适合做你现在这个“技术采用 memo / 研究型 agent”的骨架。

## 2. 记忆层

- [Mem0](https://github.com/mem0ai/mem0)

抄这些：

- User / Session / Agent 三层记忆
- 长期偏好
- 历史对话复用
- 把“团队约束”变成可检索记忆，而不是每次手填 prompt

本项目 v2 的实现选择：

- 不直接引入 Mem0 SDK
- 用 SQLite 保存 `user | project | session` 三层 memory
- 用轻量 extractor 写入明确偏好、团队约束和 session fact
- `project` memory 同步进入本地 DocumentStore，参与向量/BM25/图检索
- 保存 `MemoryExtractionResult`，让 memory 候选、accepted、rejected 都可观察
- 把 project/constraint memory 当成 hard constraint，进入 constraint coverage gate

这层最适合补你说的 `memory`，也能证明 memory 不是只写进数据库，而是会进入 planning、retrieval 和 evaluation。

## 3. 交互壳

- [Open WebUI](https://github.com/open-webui/open-webui)
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)

抄这些：

- 聊天入口
- 本地知识库
- 权限 / 资源绑定
- 记忆 / notes / workspace
- 模型路由

如果你想把项目包装成“能真正每天打开用的研究台”，这层很重要。

本项目 v2 的实现选择：

- 不引入 React/Vite
- 继续用 `apps/web/index.html`
- 左侧 session list，中间 chat timeline，右侧 plan/memory/tools/quality inspector
- 研究前必须确认 plan，避免聊天一发就开长任务
- session `researching` 时轮询 steps、tool invocation、approval、quality 状态

## 4. 多 agent / 工具循环

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [OpenHands](https://github.com/OpenHands/OpenHands)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [CrewAI](https://docs.crewai.com/)

抄这些：

- multi-agent workflow
- sequential / concurrent / handoff
- tool loop
- long-running tasks
- tool registry / policy / approval
- observability / restartability

`AutoGen` 也能看，但官方现在写得很明确：它进入 maintenance mode 了，新用户更适合直接看 Microsoft Agent Framework。

## 5. 你这个项目最合适的组合

我建议你的主线是：

1. `LangGraph` 负责状态图和执行边界
2. `Open Deep Research` 负责研究 memo 的工作流范式
3. `Mem0` 负责 memory 分层思想
4. `Open WebUI` 或 `AnythingLLM` 负责对话壳和知识库入口
5. `OpenAI Agents SDK`、`Microsoft Agent Framework`、`OpenHands`、`CrewAI` 负责 tool policy、handoff、HITL、observability 参考

## 6. 不建议的方向

- 直接做成普通聊天机器人
- 只做知识库问答
- 把 deterministic 假跑当成真实 agent
- 把一个完整外部 deep-research agent 整个包进来当“自己的 agent”

## 7. 下一步实施顺序

已完成：

1. chat session
2. 本地知识库和 project memory 绑定
3. SQLite memory 层
4. plan confirmation
5. Web Agent Workbench
6. GitHub MCP 缺 token 时明确暴露 unavailable，不伪装 evidence
7. AgentRunStep / ToolInvocation / ApprovalRequest
8. MemoryExtractionResult / memory evaluation
9. ConstraintCoverage gate

继续可以做：

1. LLM memory extractor 和人工标注 eval set
2. WebSocket/SSE streaming trace
3. session export bundle
4. 2-3 个真实 adoption memo demo corpus
5. LangGraph durable interrupt / pause / resume

这样做出来的东西，秋招里会更像“你真的做过 agent”，而不是只做过一个带 RAG 的 demo。
