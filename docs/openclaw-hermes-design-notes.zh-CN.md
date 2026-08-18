# OpenClaw / Hermes 设计笔记

这份笔记只回答一个问题：**这三个仓库里，哪些设计值得抄，哪些不值得抄，为什么**。

参考仓库：

- [openclaw-mini](https://github.com/voocel/openclaw-mini)
- [OpenClaw](https://github.com/openclaw/openclaw)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent)

## 1. openclaw-mini 值得抄什么

它最像一个“可学习的最小壳”。真正有价值的不是功能多，而是骨架清楚：

- `sessionKey` 贯穿会话、事件、运行和恢复
- 外层 planning loop + 内层 tool loop 的双层结构
- 运行时 event stream，便于看 agent 在干嘛
- heartbeat / activity signal，让长任务不会像死掉一样
- context pruning / summary compression，解决上下文越跑越长的问题
- memory / skills / playbook 这种轻量可复用知识

### 我在本项目里怎么抄

- `session_key`：作为 session 的稳定外部句柄
- `AgentEvent`：把 message / step / approval / invocation 统一成时间线
- `heartbeat`：研究任务运行中写轻量 step，给 UI 一个活性信号
- `context_summary`：会话变长后做压缩，再回填 planning
- `ResearchSkill`：用少量 playbook 收束 demo 场景，而不是做插件市场

### 不抄什么

- 不抄“大而全”的产品外壳
- 不抄多端聊天壳
- 不抄复杂插件生态

因为这个项目要的是“秋招可讲的闭环”，不是另一个 Agent OS。

## 2. OpenClaw 值得抄什么

OpenClaw 更像控制平面视角：它关心 workspace、工具边界、事件、导出、恢复，而不只是一次研究回答。

### 值得抄的点

- workspace / control-plane 思想：团队约束要显式化
- 工具边界要可见：哪些工具可用、哪些要 auth、哪些风险高
- 运行结果要能导出 / 复盘
- session 不是消息列表，而是 workspace 下的工作单元

### 我在本项目里怎么抄

- `WorkspaceProfile`：团队约束、技术栈、部署边界、风险策略、偏好来源
- `GET /v1/agent/workspaces`：让 workspace 成为正式对象
- session 默认绑定 workspace
- export bundle：把 session、workspace、plan、trace、eval 一次性导出

### 不抄什么

- 不抄多用户 SaaS 的复杂权限系统
- 不抄多 channel 聊天 OS
- 不抄桌面控制 / shell 自动化

这个项目现在还应该保持单用户本地研究台的边界。

## 3. Hermes 值得抄什么

Hermes 最有意思的地方不是“更像聊天助手”，而是它在尝试把知识沉淀成可复用的 skills / memory / playbooks。

### 值得抄的点

- bounded memory：只保留能复用的长期偏好或决策
- skills / playbooks：让 agent 有稳定的场景模板
- cross-session reuse：同一类约束下，下次不用重新讲一遍
- learning loop：完成一次 run 后把决策记回去

### 我在本项目里怎么抄

- `ResearchSkill` 只保留 3 个内置场景
- run 完成后写入 decision memory
- planning 时注入 relevant memory + workspace profile + selected skill
- 让 skill 影响追问、plan draft 和 evaluation focus

### 不抄什么

- 不抄无限增长的技能市场
- 不抄“自己变成通用助手”的叙事

Hermes 对我们来说是 memory / playbook 的设计参考，不是终局形态。

## 4. 这三者组合起来，应该怎么讲

可以这样讲：

> 我从 openclaw-mini 抄了 session 和 event 的骨架，从 OpenClaw 抄了 workspace/control-plane 的边界，从 Hermes 抄了 memory 和 playbook 的收束方法。最后把它们都收到了一个面向开源引入评审的 research agent workbench 里。

## 5. 对应代码路径

- `src/agentic_research_copilot/agent.py`：session、workspace、skill、events、heartbeat、export
- `src/agentic_research_copilot/schemas.py`：数据模型
- `src/agentic_research_copilot/storage.py`：SQLite 持久化
- `src/agentic_research_copilot/server.py`：workspace / skills / export API
- `apps/web/index.html`：workbench UI

