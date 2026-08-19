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
