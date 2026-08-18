# AI Research Copilot 秋招讲解手册

## 1. 项目一句话

一个面向技术调研和开源引入评审的 conversational research agent runtime：用户在会话里输入 repo、技术问题和团队约束，系统先把这些约束落进 workspace profile，再通过 skill/playbook 选择、可确认研究计划、step stream、tool policy、approval gate、web、本地知识库、GitHub MCP 等机制收集证据，最后输出带 citation、trace、constraint coverage、evaluation 和 export bundle 的技术采用 memo。

更口语一点：

> 我做的是一个研究型 agent 工作台，不是普通问答。它会记住团队约束，先把 workspace 和 skill 选出来，再和用户确认研究计划，最后跑一个可追踪、可审批、可评估、可复盘的 research workflow。

## 2. 最适合展示的真实场景

场景：小型研发团队准备引入一个开源技术，例如 LangGraph、LlamaIndex、Qdrant、Temporal、Milvus、OpenHands。

输入：

- 仓库或技术对象：`langchain-ai/langgraph`
- 决策问题：是否适合作为研究型 agent 的 workflow runtime
- 团队约束：Python/FastAPI、单机 Docker Compose、需要 checkpoint、需要可回滚、团队 5 人、秋招项目要能讲清楚

输出：

- adoption memo
- source index
- research trace
- retrieval/tool route
- step stream
- tool registry / approval request
- workspace profile / selected skill
- evaluation metrics
- constraint coverage
- evidence gaps

这不是“自己玩玩”的方向，因为小团队真的会遇到“要不要引入某个开源库”的问题。普通搜索只能给链接，通用聊天只能给建议；这个项目强调的是可配置约束、可确认计划、证据链、可复盘过程和质量指标。

另外，这个项目现在还有一层 skill pack。它不是插件市场，也不是给模型开任意 shell，而是把研究场景收束成可加载的能力包：`skill.json + SKILL.md + optional scripts/`。agent 会先选 skill，再读技能说明，必要时跑一个受控 preflight，把结果写回 plan 和 session。这样面试时就能讲清楚“skill 不是贴个 markdown，而是可发现、可加载、可执行、可审计”。

## 3. 为什么不是普通 RAG

普通 RAG 通常是：

```text
question -> retrieve chunks -> answer
```

这个项目是：

```text
session -> memory -> clarify -> plan -> confirm -> step stream -> tool policy/approval -> supervise tool loop -> web/vector/MCP evidence -> report -> constraint gate -> verify/evaluate -> trace/replay
```

关键区别：

- 有 session state：不是一次性问答，可以在同一研究任务里补充约束。
- 有 memory：团队约束和用户偏好不用每次复制 prompt。
- 有 interactive planning：先生成 plan draft，用户确认后才启动长任务。
- 有 tool loop：研究 agent 会在 web search、vector retrieval、MCP tool 之间选择。
- 有 tool policy：工具状态、auth、risk、approval_required 都是可见的。
- 有 HITL approval：MCP token 缺失或风险动作不会被伪装成成功。
- 有本地知识库：团队约束、架构文档、论文、README 可以进入 DocumentStore。
- 有图增强检索：实体/关系信号和 dense/BM25 一起参与候选召回。
- 有 evaluation：输出 citation precision、retrieval hit rate、context precision、faithfulness proxy、constraint coverage 等指标。
- 有 trace/replay：可以解释每一步怎么来的，而不是只给最后答案。

面试时可以说：

> 我没有把它做成简单 RAG，因为技术采用评审不是只回答一个事实，而是要把问题拆成多个研究子问题，跨来源收集证据，最后给出可审计的决策 memo。

现在这版还多了一个控制面：workspace profile 负责团队约束，skill/playbook 负责场景收束，export bundle 负责把一次研究变成可复盘的交付物。

## 4. Agent 技术点怎么讲

### 4.1 Agent Session

对应文件：

- `src/agentic_research_copilot/agent.py`
- `src/agentic_research_copilot/schemas.py`
- `src/agentic_research_copilot/storage.py`
- `src/agentic_research_copilot/server.py`

可以讲：

> 我单独加了一层 `ConversationalResearchAgent`，它不重写底层 research pipeline，只负责 session state、message、memory、plan confirmation 和 job binding。这样聊天入口和研究执行是解耦的。

状态流：

```text
collecting -> planning -> awaiting_confirmation -> researching -> completed / failed
```

这个点能体现你理解“agent 不是一个 prompt”，而是状态机、外部工具、用户确认和长任务状态的组合。

### 4.2 Agent Step 和 Observability

新增对象：

- `AgentRunStep`
- `ToolInvocation`
- `ApprovalRequest`

可以讲：

> v2 我补了 session-visible step stream。用户发消息、计划生成、确认计划、启动研究、工具调用、报告、验证和评估都会写成 step。底层 runtime 仍然保留 RunTraceEvent，session 完成后会把 trace 同步成 steps，Workbench 可以轮询展示。

### 4.2.1 Skill Pack

代码路径：

- `src/agentic_research_copilot/skill_registry.py`
- `src/agentic_research_copilot/agent.py`
- `src/agentic_research_copilot/server.py`
- `skills/open_source_adoption_review/`

可以讲：

> 我没有把 skill 做成一个无限扩张的插件系统，而是做成可发现的 skill pack。每个 pack 由 manifest、Markdown 说明和可选脚本组成。agent 启动时会扫描这些 pack，按场景选 skill，先执行受控 preflight，再把结果注入 planning。

这段的重点不是“能跑脚本很酷”，而是：

- skill 是有 manifest 的
- skill 的说明会被 agent 读取
- skill 的脚本是白名单式、相对路径、JSON stdin/stdout、带超时的
- skill 的结果会写回 session 和 plan metadata

这比“只是贴一段提示词”更像真正的工程能力。

这能体现你理解 agent 工程里的 observability，而不是只保存最终 answer。

### 4.3 Memory

Memory 分三层：

- `user`: 长期偏好，例如秋招目标、喜欢的技术栈
- `project`: 团队约束，例如部署方式、语言、评审标准
- `session`: 当前会话里的临时事实

实现选择：

- v2 用 SQLite，不引入 Mem0 SDK
- 每次 user message 后做轻量 memory extraction
- 保存 `MemoryExtractionResult`
- project memory 同步写入 DocumentStore
- planning 时自动注入 relevant memory
- project/constraint memory 进入 constraint coverage gate

可以讲：

> 我借鉴 Mem0 的 memory 分层，但没有直接依赖 Mem0。因为这个项目是学习和面试项目，我更想自己实现一版最小闭环，能讲清楚 memory 是怎么进入 planning 和 retrieval 的。

### 4.4 Tool Loop 和 Tool Policy

底层 researcher 的工具包括：

- `web_search`
- `vector_retrieval`
- `mcp_tool`

核心不是“工具越多越好”，而是：

- planner 先拆研究子问题
- supervisor 决定每个子问题怎么搜
- route materializer 把 tool-call 参数变成可执行 route
- researcher bounded loop 收集证据
- tool registry 显示 enabled/auth/risk/approval_required
- approval request 显示 MCP 缺 token 或风险动作
- reporter 只能引用已有 evidence
- evaluator 检查质量

可以讲：

> 我把 tool loop 控制在 bounded iteration 内，避免 agent 无限跑。v2 又补了 tool registry、tool invocation 和 approval request，所以每个工具的可用性、风险和失败方式都是可观察的。

### 4.5 Human-in-the-loop

这个项目的关键产品决策是“研究前确认”：

- 用户输入完整问题后，不直接启动 research
- agent 先生成 plan draft
- 用户确认后再启动后台 job

可以讲：

> 技术研究任务成本比较高，直接开跑容易浪费 token 和时间。所以我加了 confirmation gate，这也是 LangGraph/Open Deep Research 这类系统里很重要的人机协作边界。

v2 还加了 approval model：

> 现在 GitHub MCP 如果缺 token，会生成 pending approval 和 pending/skipped tool invocation，而不是把 web evidence 冒充 MCP evidence。真正中断底层图执行的 durable interrupt 留给下一版。

### 4.6 Constraint Coverage

新增对象：

- `ConstraintCoverage`

可以讲：

> 我在真实实验里发现 constraint_recall 只有 0.25，说明 memory 写进去了不等于报告真的覆盖了团队约束。v2 把 project memory 标为 hard constraint，注入 planner，并在报告后做 constraint coverage gate。覆盖低于 0.6 进入 warning，低于 0.4 标记 evaluation failed，但保留报告用于复盘。

### 4.7 面试时怎么解释 skill

如果面试官问“这不就是 markdown 文件吗”，你可以直接回答：

> 外壳确实借鉴了 markdown skill，但在项目里它已经不是静态文档了。它有 manifest、有注册表、有 session 级选择、有 preflight 脚本、有 plan 注入、有运行时事件和 export bundle，所以它是一个受控能力包，不是单纯提示词。

如果继续问“为什么不做大而全的插件市场”，就答：

> 这个项目的目标不是做通用平台，而是做一个秋招能讲清楚的 research agent runtime。所以我优先把技能包做成可发现、可加载、可审计的最小闭环，而不是先铺一个很大的插件生态。

## 5. 如何现场演示

先启动服务：

```powershell
uvicorn agentic_research_copilot.server:create_app --factory --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/
```

推荐输入：

```text
我们团队是 5 人 Python/FastAPI，单机 Docker Compose 部署，必须可回滚。
请评估 langchain-ai/langgraph 是否适合作为研究型 agent 的 workflow runtime，
输出 adoption memo，并关注可观测性、checkpoint、工具循环和秋招展示价值。
```

演示顺序：

1. 新建 session。
2. 输入上面的约束和研究问题。
3. 展示右侧 memory：团队约束被抽取为 project memory。
4. 展示 plan draft：research brief、plan items、assumptions、success criteria。
5. 点击“确认并开始研究”。
6. 展示 job status 从 researching 到 completed。
7. 展示 report。
8. 看 Tools 区：tool registry、approval requests、tool invocation、steps。
9. 看 Quality 区：citation precision、context recall、faithfulness、constraint coverage。
10. 说明 GitHub MCP 如果没 token，会显示 unavailable，不会伪装成 MCP evidence。

## 6. 面试官问“为什么不用 Codex/Deep Research”

推荐回答：

> Codex 和 Deep Research 作为最终产品肯定更强。我这个项目不是要替代它们，而是学习并实现它们背后的一部分工程机制：session state、memory、interactive planning、tool policy、approval、tool loop、RAG evidence routing、citation grounding、constraint coverage、evaluation 和 trace replay。对我来说，价值在于我能解释一个研究型 agent 从输入到报告的每个内部阶段。

如果面试官继续问“那你的项目有什么用”：

> 它可以作为小团队的技术采用评审台。团队先保存自己的技术栈、部署限制和评审规则，以后评估一个开源库时，不需要每次复制约束。agent 先给出研究计划，经确认后再收集 GitHub、Web 和本地文档证据，最后输出可复盘的 adoption memo。

## 7. 面试官可能追问的问题

### Q1：为什么需要 memory？

答：

> 技术决策不是孤立问题。比如同一个库，对大公司和 5 人小团队的结论可能不一样。memory 让团队约束稳定进入 planning 和 retrieval，而不是靠用户每次手写 prompt。

### Q2：为什么需要图结构？

答：

> 这个项目里的图不是为了炫技，而是用于补充 dense/BM25 的召回盲区。技术文档里有很多实体和关系，例如 framework、checkpoint、runtime、deployment、evaluation、MCP tool。图信号可以把相关实体附近的 chunk 纳入候选，再和 dense/BM25 结果融合，最后 rerank。

诚实补一句：

> 现在它是 LightRAG-inspired 的轻量图增强，不是完整 GraphRAG 平台。

### Q3：deterministic 模式是不是假的？

答：

> deterministic 模式只是测试替身，用来保证 CI 和离线回归稳定。产品证明必须看 real mode：真实模型、真实搜索、真实 embedding/rerank、真实 trace/eval。项目里也保留了 adoption memo experiment 来跑真实链路。

### Q4：GitHub MCP 没 token 怎么办？

答：

> 后端会把 MCP 标记为 unavailable，并提示需要 `ARC_MCP_AUTH_TOKEN/GH_TOKEN/GITHUB_TOKEN/GITHUB_PERSONAL_ACCESS_TOKEN`。web+local run 可以继续，但不会把 web evidence 冒充成 MCP evidence。v2 还会在 session 里生成 approval request 和 tool invocation，方便 UI 和 trace 复盘。

### Q5：为什么不直接接一个 deep research MCP？

答：

> 那会和项目自己的 planner/supervisor/reporter 重叠，架构上讲不清。更合理的是接 GitHub MCP 这种垂直工具，让它提供 repository/source-of-truth evidence，而不是替代本项目的研究工作流。

### Q6：你的 HITL 够成熟吗？

答：

> v2 已经有两个 HITL gate：研究前 plan confirmation，以及 MCP unavailable/risky action 的 approval artifact。但我不会夸成完整 LangGraph durable interrupt。现在是 observable HITL，下一步可以在 graph node 层实现 checkpoint pause/resume。

## 8. 当前边界，怎么诚实说

可以承认：

- 这是单用户本地 workbench，不是多租户 SaaS。
- memory extractor v1 是轻量规则，后续可以换成 LLM extractor 加 eval。
- approval v2 是可观测审批，不是完整 durable interrupt。
- deterministic 模式不是产品效果证明。
- GitHub MCP 依赖 token 和网络，缺失时只能跑 web+local。
- evaluation 是工程质量代理指标，不是严格学术 benchmark。
- 图增强是轻量实体/关系召回，不是完整知识图谱系统。

同时强调：

> 我没有回避这些边界，而是把它们显式写进 runtime config、docs、UI 状态和测试里。这个项目最强的地方是可解释、可观测、可复盘。

## 9. 简历写法

简历项目 bullet 可以这样写：

- 设计并实现 conversational research agent：支持 session state、SQLite memory、interactive planning、human confirmation、step stream 和后台 research job binding。
- 构建 Agentic RAG 检索层：结合 Qdrant dense retrieval、SQLite FTS5/BM25、LightRAG-inspired entity/relation graph signal 和 reranking，支持本地文档与 project memory grounding。
- 实现 ODR-style research workflow：clarify、plan、research supervisor、bounded tool loop、reporter、verifier/evaluator、trace/replay。
- 设计 tool registry/policy/invocation/approval 模型，支持 GitHub MCP repository/code/issue/PR/release evidence，并在 token 缺失时 fail fast 或显式降级。
- 建设可观测质量体系：输出 source index、retrieval route、tool calls、checkpoint、citation precision、context precision、faithfulness proxy、constraint coverage 等评估指标。

## 10. 最后一句定位

这个项目不要包装成“我做了一个 Codex”。更好的说法是：

> 我做了一个能讲清内部机制的研究型 agent runtime。它把聊天入口、长期约束、计划确认、工具策略、审批、工具循环、本地知识库、引用报告和评估复盘连成了一个完整闭环。
