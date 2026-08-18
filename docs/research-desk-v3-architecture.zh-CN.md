# Research Desk v3 架构

这份文档讲的是当前项目的真实数据流，不是理想图。

## 1. 一句话架构

```text
workspace -> session -> message -> memory extraction -> skill selection -> context compaction
-> clarify or plan draft -> user confirmation -> research job -> steps/events/heartbeat
-> tool loop -> report -> constraint coverage -> evaluation -> export bundle
```

## 2. 核心对象

### WorkspaceProfile

workspace 是团队约束的正式载体，负责回答这些问题：

- 这个研究是给哪个团队用的
- 技术栈是什么
- 部署边界是什么
- 风险策略是什么
- 哪些来源偏好、哪些工具禁用

### AgentSession

session 是一次对话 + 一次研究任务的外壳。

关键字段：

- `session_id`
- `session_key`
- `workspace_id`
- `selected_skill_id`
- `context_summary`
- `active_run_id`
- `last_heartbeat_at`

### ResearchSkill

skill 是一个可加载的研究型能力包。它不是无限扩张的插件市场，而是由结构化 manifest、Markdown 操作说明和可选的安全脚本组成。

当前只有 3 个：

- `open_source_adoption_review`
- `architecture_tradeoff_memo`
- `demo_readiness_risk_review`

默认目录结构：

```text
skills/
└── open_source_adoption_review/
    ├── skill.json
    ├── SKILL.md
    └── scripts/
        └── preflight.py
```

运行时会：

1. 扫描 `ARC_SKILL_PATHS` 指定的目录。
2. 读取 `skill.json` 和 `SKILL.md`。
3. 把 skill 注册进 catalog。
4. 根据消息和 workspace 选择 skill。
5. 把 skill 的 instructions excerpt、plan template、evaluation focus 注入 planning。
6. 对声明为 `auto` 的脚本执行一次受控 preflight。

脚本只允许走 manifest 声明的相对路径，使用 JSON stdin/stdout、固定超时和无 shell 的 `subprocess` 调用。它可以做输入规范化、评分、校验和本地计算，但不会获得任意文件写入或破坏性工具权限。

### AgentEvent / AgentRunStep

- `AgentEvent`：统一时间线视图，包含 message / step / approval / invocation
- `AgentRunStep`：session 可见的阶段状态，适合 UI 轮询

## 3. 运行数据流

### 3.1 发消息

用户发消息后，agent 会：

1. 保存 user message
2. 做 memory extraction
3. 绑定 workspace
4. 选择 skill
5. 如果 skill 有 auto script，执行 preflight
6. 必要时做 context compaction
7. 决定是澄清还是出 plan draft

### 3.2 澄清

如果缺少 skill 要求的输入，agent 不直接研究，而是先追问。

这一步的价值是：

- 少浪费 token
- 让 plan 更稳定
- 把“必须先问”的边界显式化

### 3.3 计划确认

当 plan draft 生成后：

- `AgentPlanDraft` 会记录 research brief、plan items、assumptions、success criteria
- 用户确认后，agent 才调用 `ResearchCopilot.submit_job`
- session 进入 `researching`

### 3.4 研究执行

研究 runtime 仍然是原来的 pipeline：

- planner
- supervisor
- researcher
- reporter
- verifier / evaluator

agent 层只负责把 session 可见的 step / event / approval / invocation 记录下来。

### 3.5 Heartbeat

如果 job 还在跑，session 会定期写入 heartbeat step。

这不是为了炫技，而是为了让长任务在 UI 和 export bundle 里一直是“活着的”。

### 3.6 完成后的学习记忆

run 完成后：

- report / trace / evaluation 会回写到 session
- constraint coverage 会重新计算
- 还会写一条 decision memory，用来支持下次复用

## 4. API 面

### Agent

- `POST /v1/agent/sessions`
- `GET /v1/agent/sessions`
- `GET /v1/agent/sessions/{session_id}`
- `POST /v1/agent/sessions/{session_id}/messages`
- `POST /v1/agent/sessions/{session_id}/confirm-plan`
- `GET /v1/agent/sessions/{session_id}/events`
- `GET /v1/agent/sessions/{session_id}/steps`
- `GET /v1/agent/sessions/{session_id}/tool-invocations`
- `GET /v1/agent/sessions/{session_id}/export`

### Workspace / Skill

- `GET /v1/agent/workspaces`
- `POST /v1/agent/workspaces`
- `GET /v1/agent/skills`
- `GET /v1/agent/skills/{skill_id}`
- `POST /v1/agent/skills/{skill_id}/scripts/{script_name}/run`

### Memory / Quality

- `POST /v1/memory`
- `GET /v1/memory`
- `GET /v1/agent/sessions/{session_id}/memory/evaluation`
- `GET /v1/research/runs/{run_id}/constraint-coverage`

## 5. 这版为什么不做得更大

v3 仍然刻意不做这些事：

- 多用户登录
- 通用 Agent OS
- 桌面控制
- destructive tools
- 插件市场
- 真正的 durable interrupt/resume

因为秋招里最重要的不是“什么都能说”，而是你能把一个研究型 agent 的闭环讲清楚。
