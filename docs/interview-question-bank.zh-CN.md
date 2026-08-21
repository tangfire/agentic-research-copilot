# Research Desk 秋招问答包

这份文档不是把题目机械复写一遍，而是把常见的 agent 面试问题改写成适合本项目的问答卡片。

项目语境固定为：**面向小型研发团队的开源引入评审 Research Agent Workbench**。

核心链路：

```text
team constraints / repo / technical question
-> session memory
-> skill selection
-> plan confirmation
-> tool loop
-> report / trace / eval / constraint coverage
-> replay bundle
```

## 1. 项目介绍

### Q1: 你这个项目一句话是什么？

考察点：能不能把项目收束到一个明确场景。

30 秒答法：

> 我做的是一个面向小型研发团队的开源引入评审 Research Agent Workbench。用户输入 repo、技术问题和团队约束后，系统会先记忆约束、选 skill、出计划，确认后再跑 research workflow，最后给出带证据、trace、evaluation 和 constraint coverage 的 adoption memo。

2 分钟答法：

> 这个项目不是普通聊天，也不是泛化 SaaS agent。我把它收束成一个非常具体的工作流：评估一个开源库是否适合团队引入。因为这个任务天然需要多轮澄清、计划确认、工具调用、证据整合和复盘，所以我把 session、memory、skill、tool policy、approval、trace、eval 都做成了一条闭环。

诚实边界：

> 它不是要替代 Codex 或 Deep Research，而是在学习那类系统背后的工程机制。

对应 artifact：`README.md`、`docs/architecture.md`、`apps/web/index.html`

### Q2: 为什么这个项目不是玩具？

考察点：闭环是否成立。

30 秒答法：

> 因为它不是只输出答案，而是有 session、workspace、memory、skill、plan confirmation、tool registry、approval request、trace、replay 和 constraint coverage。真正可讲的是从输入到证据再到评估的完整链路。

对应 artifact：`src/agentic_research_copilot/agent.py`、`src/agentic_research_copilot/pipeline.py`

### Q3: 你为什么把场景固定成开源引入评审？

考察点：有没有合理收束。

30 秒答法：

> 因为这是最容易真实部署、真实演示、也最容易说清楚价值的场景。小团队真的会评估要不要引入 LangGraph、OpenHands、Qdrant、LlamaIndex 这类开源库，而且这个任务天然需要证据链，不适合只靠一句话回答。

## 2. 多 Agent / 角色划分

### Q4: 你的项目里有哪些 Agent？

考察点：角色是否清楚，是否真的分工。

30 秒答法：

> 主链路里有 ConversationalResearchAgent 做会话入口和确认门，底层 research runtime 里有 planner、research supervisor、三个 specialist worker、reporter、verifier、evaluator。v4 里 `RepoSignalAgent`、`ArchitectureFitAgent`、`OpsRiskAgent` 会在 research stage 内执行各自的工具循环，并把证据归属、route decision 和冲突写进 run artifact。

对应 artifact：`src/agentic_research_copilot/agent.py`、`src/agentic_research_copilot/multi_agent_harness.py`

### Q5: 为什么要拆成三个专长 worker，而不是一个大 Agent？

考察点：是不是硬凑多 Agent。

30 秒答法：

> 因为开源引入评审本身就有三类稳定问题：仓库事实是否可信、架构是否适配、部署和运维风险是否可接受。拆成三个 worker 能让 planner 的路由、报告的覆盖和 review 的证据更清楚，而不是一个责任边界同时管所有东西。

2 分钟答法：

> 我不是为了多 Agent 而多 Agent。拆角色的前提是每个 worker 有稳定职责和触发信号：RepoSignalAgent 看 repo 信号和开发事实，ArchitectureFitAgent 看架构和集成成本，OpsRiskAgent 看部署、回滚、依赖和风险。这样最终 memo 里的每一段都能对应一个责任边界和工具证据。

对应 artifact：`src/agentic_research_copilot/multi_agent_harness.py`

### Q6: 多 Agent 的价值是什么？能不能退化成单 Agent？

考察点：是否理解系统复杂度。

30 秒答法：

> 可以退化成单 Agent 做 ablation，但主模式下三个 worker 的价值是让工具边界、证据归属和失败定位更清楚。真正执行还是同一条 LangGraph workflow，不是先跑一遍节点再偷偷跑另一套 agent。

## 3. 工具与技能

### Q7: Agent 怎么发现和调用工具？

考察点：工具边界是否显式。

30 秒答法：

> 我把工具做成 tool registry，里面有 channel、risk、auth、approval_required 和 failure mode。agent 不是“想用就用”，而是先看工具定义，再决定 web、vector 还是 MCP。

对应 artifact：`src/agentic_research_copilot/agent.py`、`apps/web/index.html`

### Q8: Skill 是什么？和 Tool 有什么区别？

考察点：概念边界是否清楚。

30 秒答法：

> Skill 是场景 playbook，Tool 是执行原语。Skill 负责告诉 agent 在什么场景下该怎么研究、该问什么、该看什么；Tool 负责真正去搜、去查、去读。

2 分钟答法：

> 在这个项目里 skill pack 由 `skill.json`、`SKILL.md` 和可选 scripts 组成。它会影响 session 里的 skill selection、plan draft、required inputs 和 evaluation focus，但它不是插件市场，也不是随便执行 shell 的入口。

对应 artifact：`src/agentic_research_copilot/skill_registry.py`、`skills/`

### Q9: 你的 skill 真能跑脚本吗？

考察点：是不是只有提示词。

30 秒答法：

> 能，但只限于 skill manifest 声明的脚本，而且是 JSON stdin/stdout、相对路径、超时控制、白名单式执行。它的作用是做 preflight 或场景检查，不是开放式 shell。

### Q10: GitHub MCP 缺 token 怎么办？

考察点：降级是否诚实。

30 秒答法：

> 缺 token 就标 unavailable，不伪装成成功，也不把 web evidence 冒充 MCP evidence。UI 和 API 都会显示 approval / pending / skipped 之类的状态。

## 4. 记忆与 workspace

### Q11: 你为什么要做 memory？

考察点：是否理解跨会话约束。

30 秒答法：

> 因为小团队每次都重复讲“我们是几个人、什么栈、怎么部署、不能接受什么风险”很浪费。我把这些抽成 project memory 和 workspace profile，后续 planning 就能自动注入。

### Q12: workspace 和 memory 有什么区别？

考察点：状态建模是否合理。

30 秒答法：

> workspace 是正式的团队约束容器，memory 是从对话中抽取出来的长期、项目或 session 事实。workspace 更像显式配置，memory 更像会话里学出来的东西。

### Q13: 为什么你的 memory 还要做 extraction result？

考察点：有没有评估意识。

30 秒答法：

> 因为只存 memory 不代表存对了。我需要知道候选里哪些被接受、哪些被拒绝、为什么拒绝，这样才能评估 memory precision 和 recall proxy。

## 5. 可观测性与回放

### Q14: 你怎么证明 agent 不是黑箱？

考察点：trace 和 step 是否可见。

30 秒答法：

> 我把 message、plan、approval、tool invocation、route decision、report、verification、evaluation 都变成 step 和 event。用户能看到的是运行过程，不只是最终答案。

### Q15: Trace 和 replay 有什么区别？

考察点：是否理解回放边界。

30 秒答法：

> Trace 是当次运行的记录，replay 是拿冻结产物重新组装一个可展示副本，不应该重新打工具。这样才能稳定复盘，而不是每次 replay 都变成新运行。

### Q16: 为什么你要做 frozen replay？

考察点：是否理解评价一致性。

30 秒答法：

> 因为如果 replay 重新联网、重新调用模型，结果就不稳定了。面试里问 replay，真正想看的是你有没有把运行证据冻结下来并保持可复现。

## 6. 评测与质量门

### Q17: 你怎么判断这个项目有没有用？

考察点：有没有质量指标。

30 秒答法：

> 我不只看报告生成成功，还看 citation precision、context recall、faithfulness proxy、memory precision/recall proxy、constraint coverage、tool success rate 和 replay fidelity。

### Q18: constraint coverage 是什么？

考察点：有没有把业务约束落到评测。

30 秒答法：

> 它是检查团队约束有没有被最终报告和证据覆盖。因为真实实验里经常出现 memory 有了，但报告没讲全，所以我把它做成质量门。

### Q19: 评测低了代表项目失败吗？

考察点：有没有诚实边界。

30 秒答法：

> 不代表失败，代表暴露了真实问题。像 constraint coverage 低，说明我需要改 prompt、改 planner、改报告结构或改 memory 注入方式。评测的价值就是把这个问题显性化。

## 7. 面试官高频追问

### Q20: 这个项目哪些模块是你做的？

考察点：ownership 是否清楚。

30 秒答法：

> 核心 runtime、session/memory/workspace 层、tool policy、workbench UI、evaluation 和 docs 都是我自己做的。它不是团队拆分项目，我负责的是整个系统主链路。

### Q21: 这个项目和 Codex / Deep Research 的关系是什么？

考察点：定位是否清楚。

30 秒答法：

> 它不是替代品，而是学习那类系统的工程骨架：session、memory、tool loop、approval、trace、eval 和 replay。

### Q22: 你为什么不直接说自己做了个通用 agent？

考察点：有没有过度包装。

30 秒答法：

> 因为通用 agent 这个说法太虚。我更愿意把范围收紧成开源引入评审，这样功能、指标、demo 和面试问答都能闭环。

## 8. 可以直接背的收尾

> 我做的不是一个聊天壳，而是一个有 session、memory、skill、tool policy、approval、trace 和 evaluation 的 research agent runtime。它专门解决小团队做开源引入评审时“约束重复说、计划不透明、工具不好控、报告不好复盘”的问题。

> 这项目的价值不在于替代大模型产品，而在于我能把一个研究型 agent 从输入、规划、执行到复盘的每一层讲清楚，并且能在代码里指出对应实现。

## 9. 附录：基础题怎么接回项目

### Q23: Agent 和 workflow 有什么区别？

考察点：能否把抽象讲清楚。

30 秒答法：

> Agent 更像有决策能力的执行体，workflow 更像明确状态和依赖的执行图。这个项目里我更强调 workflow，因为研究任务需要可追踪、可恢复、可评测。

### Q24: 多任务有依赖时怎么并发调度？

考察点：有没有工程基本功。

30 秒答法：

> 先做依赖图，再找可并发节点，最后在共享状态上做隔离和合并。这个项目里对应的就是 plan item、route decision 和 researcher loop。

### Q25: shared_ptr 为什么会循环引用？

考察点：基础 C++ 是否扎实。

30 秒答法：

> 因为引用计数互相加了，谁都释放不掉。解决方法通常是用 `weak_ptr` 打断环。

### Q26: 虚表和虚表指针是什么？

考察点：虚函数机制是否清楚。

30 秒答法：

> 虚表里放的是虚函数地址，虚表指针通常挂在对象里。调用虚函数时，先通过 vptr 找到 vtable，再间接调用对应实现。

### Q27: mmap 和 writev 这类系统题怎么答？

考察点：是否能把系统调用和工程问题分开。

30 秒答法：

> 这类题我会先说它们的目标不同：mmap 偏向把文件映射成内存，writev 偏向把多个 buffer 聚合写出。它们都属于底层性能和 I/O 机制，不是这个项目主线，但如果面试官追问，我会按“减少拷贝 / 聚合写入 / 系统边界”来解释。

### Q28: 为什么你不把项目说成 Codex 替代品？

考察点：是否诚实。

30 秒答法：

> 因为不合理。大模型产品在广度、稳定性和能力上都更强，我这个项目只是把它们背后的工程机制复现出来，方便学习和答辩。

### Q29: 如果被问到通用 Agent 组成部分，你怎么回答？

考察点：是否能抽象。

30 秒答法：

> 通常是 session/state、planner、tool router、execution loop、memory、observability、evaluation、safety/approval。这个项目基本把这几层都做了，但范围收束在开源引入评审。

### Q30: 如果被问“为什么还要拆成三个 Agent”，怎么答？

考察点：是否会反问“价值在哪里”。

30 秒答法：

> 因为这个任务有三个稳定子问题：仓库事实、架构适配、运维风险。拆出来以后，路由、证据和评测都更清楚；如果不清楚，我宁愿退回单 Agent。

## 10. 后端协议与工程补充

### Q31: 如何做接口鉴权，如果不想把 token 暴露给 agent，一般怎么处理？

考察点：安全边界是否清楚。

30 秒答法：

> token 不应该进入 agent prompt，也不应该进入 trace。正确做法是把凭证留在后端配置层或 secret manager 里，agent 只看到工具状态和 schema，由后端 tool adapter 代为调用。

对应 artifact：`src/agentic_research_copilot/settings.py`、`src/agentic_research_copilot/agent.py`

### Q32: 怎么理解 MCP 协议，和 function calling 的区别是什么？

考察点：是否理解工具标准化。

30 秒答法：

> function calling 是模型侧的工具调用接口，MCP 是把工具、资源和 prompt 统一暴露给 agent host 的开放协议。前者偏“模型怎么调函数”，后者偏“外部系统怎么标准化接入 agent”。

2 分钟答法：

> 我会把 function calling 理解成模型能力层，而把 MCP 理解成 agent 工具层的通信协议。MCP 的优点是跨模型、跨 host、跨工具更统一，缺点是引入了协议、服务器和权限边界，工程复杂度更高。这个项目里我把 GitHub MCP 当成外部证据源，和本地 web / vector tool 分层处理。

对应 artifact：`src/agentic_research_copilot/agent.py`、`docs/architecture.md`

### Q33: SSE 断联后怎么恢复？

考察点：事件流和断点续传。

30 秒答法：

> 我们项目现在主路径是 SSE + durable event log，polling 只是降级。前端保存最后看到的 event id，断线后用 `after_event_id` 重连 `/events/stream`，后端从 SQLite 事件时间线补发后续事件。

2 分钟答法：

> 设计上不是把事件只放内存里直接推，而是执行节点先写 `AgentEvent`、`AgentRunStep`、`ToolInvocation`、`ApprovalRequest` 等持久化记录。SSE 只负责传输，恢复时按 `Last-Event-ID` 或 `after_event_id` 从 durable ledger 继续发；大 payload 只在事件里放摘要和 artifact id，避免 Redis 或 SSE 消息过大。

对应 artifact：`src/agentic_research_copilot/agent.py`、`src/agentic_research_copilot/server.py`、`apps/web/index.html`

### Q34: MySQL 和 Redis 数据不一致要怎么解决？

考察点：通用后端一致性。

30 秒答法：

> 这是分布式后端题，不是我们项目主线。我们的项目是单机 SQLite workbench，所以不该硬凑 MySQL/Redis。一旦业务真的需要分布式缓存，常见做法是缓存失效、版本号、幂等重试、延迟双删和最终一致性。

### Q35: 如果让你设计一个评测系统，你会怎么设计？

考察点：是否能把 agent 过程指标化。

30 秒答法：

> 我会把评测拆成三层：结果层、过程层、质量门。结果层看报告是否回答问题；过程层看工具调用、路由、记忆和步骤；质量门看 citation precision、constraint coverage、tool success rate、replay fidelity 这些可解释指标。

2 分钟答法：

> 在这个项目里我已经把它拆成了 benchmark summary、route decisions、evidence ledger、memory evaluation 和 constraint coverage。真正可讲的不是“模型输出了答案”，而是“它有没有按预期路由工具、有没有覆盖约束、有没有把证据用对、失败后能不能回放”。如果要继续加强，我会加 labeled fixture、cursor replay、单步回归和异常样本集。

对应 artifact：`src/agentic_research_copilot/pipeline.py`、`src/agentic_research_copilot/agent.py`

### Q36: Claude Code 的记忆机制和长期记忆压缩机制怎么理解？

考察点：是否理解产品级记忆。

30 秒答法：

> 我理解它主要是文件化的持久上下文，比如项目级和用户级记忆文件，再配合会话压缩，让长期偏好和临时上下文分层保存。这个项目里我对应实现的是 user/project/session 三层 memory 和 context_summary。

2 分钟答法：

> Claude Code 的思路更像“把长期知识落成可编辑文件，把短期上下文压成摘要”。我们项目没有直接照搬那个实现，而是自己做了一版 SQLite memory：session memory 管当前对话，project memory 管团队约束，user memory 管长期偏好，context summary 负责长会话压缩。区别是我们更强调 research workbench 的可观测和可评测，而不是个人助理式自动记忆。

对应 artifact：`src/agentic_research_copilot/agent.py`、`docs/memory-and-constraint-eval.zh-CN.md`

## 11. 基础与平台工程追问

这一组题不是全部都要硬套到 Research Desk 上。正确策略是：agent / MCP / trace / evaluation / SSE 恢复可以接回项目；进程线程协程、数据库索引、语言数据结构、Redis/MySQL 一致性属于通用基础题，单独准备，面试时不要强行说“我项目里用了”。

### Q37: 进程、线程、协程有什么区别？

考察点：基础并发模型。

30 秒答法：

> 进程是资源隔离单位，有独立地址空间；线程是 CPU 调度单位，同进程线程共享内存；协程是用户态调度的轻量执行单元，适合 I/O 密集场景，用事件循环在单线程里切换任务。

2 分钟答法：

> 进程隔离强但切换和通信成本高，适合服务隔离和多核并行；线程共享内存，通信方便但要处理锁和竞态；协程靠主动让出执行权，开销小，适合网络请求、数据库访问、LLM 调用这种大量等待 I/O 的场景。Python 里常见是 multiprocessing、threading、asyncio；Java 里是 Thread、线程池，新的虚拟线程也属于轻量并发模型。

诚实边界：

> Research Desk 主链路是 FastAPI + in-process job + LangGraph，不是一个高并发服务项目。这里能讲的是为什么 agent 的长任务适合异步任务、状态持久化和可恢复，而不是吹自己做了复杂并发框架。

### Q38: 数据库索引是什么？为什么能加速查询？

考察点：数据库基础。

30 秒答法：

> 索引是额外的数据结构，常见是 B+Tree。它通过有序结构减少全表扫描，让查询从扫描所有行变成按索引定位范围或主键。

2 分钟答法：

> B+Tree 适合范围查询、排序和前缀匹配；哈希索引适合等值查询但不适合范围。索引不是越多越好，因为会增加写入和维护成本。联合索引要注意最左前缀，查询条件、排序字段和选择性都会影响索引是否有效。

接回项目：

> 项目里本地持久化用 SQLite，核心不是大规模 OLTP，而是保存 session、message、run、trace、memory、tool invocation。面试时可以说：如果要产品化到多用户，我会给 `session_id`、`run_id`、`created_at`、`workspace_id` 这类查询路径加索引，并把 trace/event 的冷热数据分层。

### Q39: Python 或 Java 常见数据结构有哪些？有什么特性？

考察点：语言基础。

30 秒答法：

> Python 常用 list、tuple、dict、set、deque、heapq；Java 常用 ArrayList、LinkedList、HashMap、HashSet、TreeMap、PriorityQueue、ConcurrentHashMap。核心是知道底层复杂度和适用场景。

2 分钟答法：

> Python list 是动态数组，随机访问快，头部插入慢；dict/set 基于哈希，平均 O(1)，但要考虑哈希冲突和内存；deque 适合双端队列。Java ArrayList 也是动态数组，LinkedList 指针开销大，HashMap 平均 O(1)，TreeMap 基于红黑树支持有序查询，ConcurrentHashMap 用于并发访问。

接回项目：

> 这个项目里更常见的是 schema 化对象和 list/dict 组合，比如 `EvidenceItem`、`RunTraceEvent`、`AgentRunStep`。真正工程重点是把非结构化模型输出变成稳定数据结构，方便 trace、evaluation 和 replay。

### Q40: Skill 的协议是什么？是不是只有 Markdown？

考察点：是否理解 skill 不是简单 prompt。

30 秒答法：

> 在这个项目里 skill 是一个轻量协议：`skill.json` 描述元信息、触发词、required inputs、脚本入口；`SKILL.md` 描述操作步骤和策略；可选 `scripts/` 做受控 preflight。agent 先发现 skill，再加载说明，必要时用 JSON stdin/stdout 跑脚本。

2 分钟答法：

> Skill 的价值不是“把提示词写进 md”，而是把某类任务的执行经验固化成可发现、可审计、可复用的 playbook。它和 tool 不一样：skill 负责告诉 agent 这个场景应该怎么做，tool 负责执行一个具体动作。我们项目没有做插件市场，也没有给 skill 开任意 shell，只允许 manifest 声明的脚本在超时和路径限制内运行。

对应 artifact：`src/agentic_research_copilot/skill_registry.py`、`skills/open_source_adoption_review/`

### Q41: HTTP、SSE、WebSocket 有什么区别？

考察点：前后端传输协议。

30 秒答法：

> HTTP 请求响应最简单，适合普通 API；SSE 是服务端到客户端的单向事件流，适合状态推送和日志；WebSocket 是双向长连接，适合实时协作、IM、多人编辑这类双向频繁通信。

2 分钟答法：

> SSE 基于 HTTP，浏览器原生支持 `EventSource`，可以自动重连，也有 `Last-Event-ID` 语义，但主要是单向推送；WebSocket 建立后是双向通道，灵活但心跳、鉴权、扩容和代理处理更复杂。Research Desk 现在选择 polling + durable event log，是为了先保证断线可恢复和实现简单；后续如果要更实时，可以把 `/events` 升级成 SSE。

对应 artifact：`src/agentic_research_copilot/server.py`、`docs/tool-loop-and-hitl.zh-CN.md`

### Q42: Agent 接口里的 SSE 应该怎么实现？断了会不会丢？

考察点：实时过程流设计。

30 秒答法：

> 不应该只把事件写在内存里直接推。每个事件先落库，带 `event_id`、`session_id`、`run_id`、`kind`、`status`、`created_at`。SSE 只负责推送，断线后前端用 last event id 补拉。

2 分钟答法：

> 一个稳妥设计是：执行节点先写 durable event log，再由推送层读取并发送 SSE。前端保存最后收到的 `event_id`，重连时带上 `Last-Event-ID` 或 `after_event_id`，服务端先补发缺失事件，再继续流式发送。Redis 可以做短期 pub/sub 或 stream，但不能替代长期事实库；数据太大时只在事件里放摘要和 artifact id，完整内容放对象存储或数据库。

接回项目：

> Research Desk 已经把这套做成主路径：UI 用 `EventSource` 订阅 `/events/stream`，服务端按 durable event log 推送；连接异常时才回退到轮询，所以既有实时体验，也不会因为前端断线丢执行状态。

### Q43: MCP 有哪几种 transport？你写 MCP 有什么要注意的？

考察点：MCP 实际工程边界。

30 秒答法：

> 常见 MCP transport 有本地 `stdio` 和远程 HTTP 类 transport。新规范更强调 Streamable HTTP，早期很多实现用 HTTP + SSE。写 MCP 时最重要的是工具 schema、权限、超时、错误处理和不要把 token 暴露给模型。

2 分钟答法：

> `stdio` 适合本地工具进程，简单、延迟低，但部署在本机；Streamable HTTP 适合远程服务，方便跨机器和云端部署，也更需要鉴权和网络容错。早期 SSE transport 还会在一些服务里出现，所以项目配置里保留了 `streamable_http / sse`。在 Research Desk 里，MCP 是外部证据边界，不是让外部 agent 替代自己的 planner/reporter；GitHub MCP 只提供 repo、code、issue、PR、release 证据，结果会转成 `EvidenceItem(kind="mcp")`。

对应 artifact：`src/agentic_research_copilot/mcp_tools.py`、`docs/architecture.md`

### Q44: 你们的 agent 结构到底是什么？是 loop 还是 workflow？

考察点：能否讲清“流动结构”。

30 秒答法：

> 外层是 workflow，内层是 bounded tool loop。外层负责 session、memory、skill、plan、confirm、research、report、verify、eval；内层 researcher 针对一个 plan item 决定 web、vector、MCP 或 complete。

2 分钟答法：

> 我把它拆成两层是为了可控。纯 ReAct loop 容易短视、循环和走偏；纯固定 workflow 又不够灵活。所以 Research Desk 用 LangGraph 管外层状态和阶段，用 researcher loop 处理局部证据缺口。三个 specialist worker 不是另一套隐藏流程，而是在 research stage 内按角色执行工具循环，并把 route decision、tool invocation 和 evidence ledger 写回统一 run artifact。

对应 artifact：`src/agentic_research_copilot/graph_runtime.py`、`src/agentic_research_copilot/agents/researcher.py`

### Q45: ReAct、Plan-and-Execute、Spec 这些规划方式怎么理解？

考察点：agent 规划范式。

30 秒答法：

> ReAct 是边想边做，灵活但容易短视和循环；Plan-and-Execute 是先计划再执行，结构清晰但计划错了会传导；Spec 更像先定义目标、约束和验收标准，再执行并校验，适合工程任务和高风险任务。

2 分钟答法：

> Research Desk 更接近 Plan-and-Execute + Spec gate：先把研究问题、团队约束和成功标准写成 plan draft，用户确认后再执行；执行中局部使用 ReAct-like tool loop；最后用 verifier/evaluator 检查 citation、faithfulness 和 constraint coverage。这样比纯 ReAct 更适合秋招项目，因为状态、证据和质量门都能讲清楚。

### Q46: 你怎么了解 agent 前沿？

考察点：学习来源和判断力。

30 秒答法：

> 我会看三类：主流开源项目的架构和 issue，比如 LangGraph、OpenHands、Open Deep Research、Open WebUI；协议和 SDK，比如 MCP、OpenAI Agents SDK、Langfuse/OTel；真实产品的交互和失败案例，比如 Codex、Claude Code、Deep Research。

2 分钟答法：

> 我不会只看概念文章，而是看它们在解决什么工程问题：持久状态、人类确认、工具权限、memory、trace、eval、replay、长任务恢复。Research Desk 也是按这个思路做的，不是把热门名词都塞进去，而是围绕开源引入评审这条闭环选择需要的机制。

### Q47: 简历项目里最难的点是什么？

考察点：是否真做过复杂问题。

30 秒答法：

> 最难的是让开放式研究任务变成可控、可观测、可评测的执行链路。具体包括工具路由不乱选、MCP 失败不伪装、Reporter 结构化输出不被截断、团队约束能进入最终报告、trace 能复盘。

2 分钟答法：

> 一个真实例子是 Reporter 曾经因为上下文太大触发 `finish_reason=length`，表现成 `ReporterContract EOF`。我没有只加兜底，而是定位到首轮请求预算问题，把 Reporter 输入压缩到最多 8 条证据和 4 个章节草稿，增加 `finish_reason` 诊断和回归测试，并用真实 provider 验证首轮 `finish_reason=stop`。这比“失败后重试”更能说明工程问题是怎么被定位和修复的。

对应 artifact：`src/agentic_research_copilot/providers.py`、`tests/test_providers.py`

### Q48: Trace 上报链路怎么做？评测架构怎么做？了解 OTel 吗？

考察点：可观测性和平台化视角。

30 秒答法：

> 项目里执行节点先写本地 SQLite ledger，包含 run、step、tool invocation、trace、evaluation；然后可选把 run-level trace 和 scores 发到 Langfuse。OTel 是更通用的观测标准，核心是 trace/span/metric/log 和 context propagation。

2 分钟答法：

> 我现在的设计是“本地事实库 + 外部观测 sink”。SQLite 是 source of truth，保证离线、replay 和导出；Langfuse 用来看 agent/tool/evaluator 的时间线和分数。如果企业化，我会把 `RunTraceEvent` 映射成 OTel span：session/run 是 root span，planner、worker、tool call、reporter、verifier 是 child spans，route precision、tool success、constraint coverage 是 metrics。ES 更适合存日志和检索事件，OLAP/metrics 系统更适合聚合指标，不能把所有大 payload 都塞进 Redis 或 trace event。

对应 artifact：`src/agentic_research_copilot/observability.py`、`docs/observability-design.zh-CN.md`

### Q49: Benchmark 数据存 ES 合理吗？

考察点：评测数据架构。

30 秒答法：

> ES 适合检索日志和 trace 文本，不一定适合做所有 benchmark 的事实库。更合理的是任务定义、期望标签、运行结果和分数用关系库或对象存储保存，ES 用来查日志、事件和失败样本。

2 分钟答法：

> Agent benchmark 至少有四类数据：task spec、expected labels、run artifact、score summary。task spec 和 score summary 需要稳定结构；trace/event 适合检索；大报告和工具结果适合对象存储或本地 bundle。Research Desk 现在是单机 SQLite + export bundle，面试时可以说这是学习版；如果上生产，会把热事件、冷 artifact、聚合指标分开存。

### Q50: 这些题我们项目到底覆盖到什么程度？

考察点：不要过度包装。

30 秒答法：

> Agent 结构、Skill、MCP、tool loop、SSE 恢复思路、trace、evaluation、memory 压缩这些都能用项目讲；进程线程协程、索引、语言数据结构、MySQL/Redis 一致性、OTel 细节属于通用基础和扩展设计，需要单独背，不能说项目已经完整实现企业级方案。

复习顺序：

1. 先背 Q1-Q22，掌握项目主线。
2. 再背 Q31-Q36，掌握后端协议和 agent 工程。
3. 最后背 Q37-Q50，补基础题和平台化追问。
