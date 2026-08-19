# Research Desk 使用说明

这份说明只讲最基本的使用方式。你可以先把项目当成一个本地的技术研究台：

```text
输入问题和团队约束
-> agent 生成研究计划
-> 你确认计划
-> agent 调用工具并研究
-> 查看报告、证据和质量结果
```

## 1. 启动项目

在项目根目录运行：

```powershell
.\scripts\start_workbench_local.ps1 -Port 8002
```

然后打开：

```text
http://127.0.0.1:8002/
```

API 文档在：

```text
http://127.0.0.1:8002/docs
```

如果页面顶部显示 `API ok`，说明前端已经连上后端。

这个启动脚本是推荐方式。它会使用独立的本地 SQLite/Qdrant 目录，并默认关闭 GitHub MCP，避免旧进程或旧环境变量让页面看起来“不像刚改完的代码”。

## 2. 第一次使用

1. 点击左侧的“新建”。
2. 在输入框写清楚三件事：
   - 想研究什么；
   - 研究对象是什么，例如 `owner/repo`、技术库或架构方案；
   - 团队有哪些约束，例如 Python/FastAPI、单机部署、预算、回滚要求。
3. 点击“发送”。
4. 查看右侧的“计划”。
5. 计划没问题时点击“确认并开始研究”。
6. 等待状态变成 `completed`，然后查看“结果”。

推荐输入：

```text
评估 langchain-ai/langgraph 是否适合一个 5 人 Python/FastAPI 团队。
我们希望单机 Docker Compose 部署，必须支持失败恢复和回滚，
不希望引入过重的分布式基础设施。请给出采用建议、主要风险和替代方案。
```

第一次点击“发送”不会直接开始研究。系统会先澄清问题或生成计划，这是产品的确认门。

## 3. 右侧区域怎么看

默认只需要看四个区域：

- `下一步`：告诉你当前应该补充信息、确认计划、等待运行，还是查看报告。
- `计划`：研究 brief、计划项和确认按钮。
- `记忆`：保存团队约束、偏好和决策，后续 session 会自动使用。
- `结果`：报告正文、来源数量和基本质量指标。

不需要一开始就打开“高级信息”。只有想学习或排错时，再展开：

- 工作区：团队背景、技术栈、部署约束、偏好来源。
- Skill：当前选择的场景 playbook 和可执行脚本。
- 工具：web、vector、GitHub MCP 的状态和调用记录。
- 路由：RepoSignal / ArchitectureFit / OpsRisk 三个 specialist worker 为什么被选中、各自用了哪些工具。
- 质量：引用、忠实度、上下文召回和约束覆盖。

## 4. 记忆怎么用

在“记忆”区域保存长期有效的信息，例如：

```text
团队只有 5 个人，优先 Python/FastAPI，部署在单机 Docker Compose。
```

建议：

- 团队规则放到 `project / constraint`；
- 个人偏好放到 `user / preference`；
- 当前研究临时事实放到 `session / fact`；
- 不要把一次性搜索结果全部保存成长期记忆。

下一次研究时，planner 会自动读取相关记忆，不需要你重复粘贴。

## 5. MCP 怎么看

顶部显示 `MCP 未配置` 不代表项目坏了。基础研究仍然可以使用本地知识库和 web 工具。

只有在需要 GitHub 仓库、代码、Issue、PR 或 Release 的一手证据时，才配置 GitHub MCP。没有 token 时，界面会明确显示不可用，不会伪造 MCP 证据。

## 6. 研究完成后做什么

先看：

1. 报告结论是否回答了原问题；
2. 团队约束是否都被覆盖；
3. 引用和来源是否足够；
4. 是否存在 evidence gap 或 warning。

如果要保存这次研究，点击顶部“导出会话”。导出的 JSON 包含 session、memory、plan、report、trace、route 和 evaluation，适合复盘或面试演示。

## 7. 常见问题

### 页面显示 `API offline`

后端没有启动，重新运行：

```powershell
.\scripts\start_workbench_local.ps1 -Port 8002
```

如果你刚改过代码但页面没有变化，优先怀疑旧进程还在跑。关闭旧的 `uvicorn`/`python` 服务后重新执行上面的启动脚本。

### 一直停在计划阶段

检查是否已经点击“确认并开始研究”。系统设计上不会在计划生成后自动开始。

### MCP 显示不可用

先不用处理，web + local KB 仍可以运行。只有需要 GitHub 一手证据时，才检查：

```text
ARC_MCP_AUTH_TOKEN
GH_TOKEN
GITHUB_TOKEN
GITHUB_PERSONAL_ACCESS_TOKEN
```

### 想学习 agent 是怎么工作的

打开“高级信息”，按这个顺序看：

```text
路由 -> 工具 -> 步骤 -> 质量 -> trace
```

这条顺序对应 agent 的核心链路：

```text
为什么选这个 specialist worker
-> 调用了哪些工具
-> 中间经过哪些步骤
-> 结果质量如何
-> 出问题时如何回放
```

## 8. 最小学习路径

第一次不要同时研究所有技术。建议按下面顺序：

1. 先跑通一次 `发送 -> 确认 -> 报告`。
2. 再保存一条 project constraint，观察下一次计划是否自动使用。
3. 再打开高级信息，理解 route、tool loop 和 trace。
4. 最后运行 benchmark，观察 route precision / recall 和 constraint coverage。
