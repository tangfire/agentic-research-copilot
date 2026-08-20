# Demo Script

这份脚本对应 5 分钟演示。

## 1. 开场

一句话：

> 这是一个面向开源引入评审的 conversational research agent workbench。它会先记住团队约束，再给计划，确认后才开始研究，最后输出可复盘的 memo、trace、evaluation 和 export bundle。

## 2. 启动

```powershell
.\scripts\start_workbench_local.ps1 -Port 8002
```

打开：

```text
http://127.0.0.1:8002/
```

演示前先确认这是刚启动的新进程，不要复用旧端口上的旧服务。

## 3. 演示顺序

### Step 1：输入研究目标

先说明团队约束会自动从输入中提取，并写入 workspace profile：

> 我不想让团队约束每次都手工贴 prompt，所以把它做成 workspace profile。页面不把所有底层字段铺在右侧，运行过程通过 Trace 查看。

推荐输入：

```text
我们团队是 5 人 Python/FastAPI，单机 Docker Compose 部署，必须可回滚。
请评估 langchain-ai/langgraph 是否适合作为研究型 agent 的 workflow runtime，
输出 adoption memo，并关注可观测性、checkpoint、工具循环和秋招展示价值。
```

你可以解释：

- memory 会抽取团队约束
- skill 会选到 `open_source_adoption_review`
- session 会先生成 plan draft，而不是直接跑

### Step 2：看 Plan

重点看：

- research brief
- plan items
- assumptions
- success criteria
- selected skill

讲法：

> 我做的是 interactive planning。研究前先出 plan，再让用户确认，避免一上来就长跑。

如果面试官问 skill，直接说明它是场景 playbook，定义输入要求、计划模板和评测重点；具体 manifest 和脚本在代码与文档中维护，不把右侧做成第二个管理后台。

### Step 3：确认并开始研究

点击 `确认并开始研究`。

这时你可以说：

- session 进入 researching
- job 被创建
- steps 开始流动
- tool policy / approval 会可见

### Step 4：看结果

先看中间报告和右侧运行摘要，再点击 `查看 Langfuse Trace`。没有 key 或连接失败时，打开 `查看本地 Trace JSON`。

在 Langfuse 里看：

- `research_run` 根链路；
- RepoSignal / ArchitectureFit / OpsRisk specialist worker；
- web/vector/GitHub MCP tool observation；
- Verifier/evaluator observation；
- citation precision / context recall / faithfulness / plan coverage 分数。

讲法：

> 这里的 multi-agent 不是为了把 agent 数量堆上去，而是把开源引入评审拆成三个稳定责任边界：仓库事实、架构适配、运维风险。Harness 区能看到每个 worker 负责哪些 plan item、用了哪些工具、拿到多少证据，以及 Verifier 发现了哪些冲突或覆盖缺口。

### Step 5：收尾

讲一句：

> 这不是一个“能聊天的玩具”，而是一个能把研究过程结构化、可复盘、可解释的 agent runtime。

如果演示 replay：

> Replay 不是重新联网再跑一次，而是基于冻结的 run artifact 生成一个新 run id，用来复盘当时的工具结果、报告、trace 和质量指标。

## 4.1 Langfuse 配置

Langfuse 是默认观测层。安装：

```powershell
python -m pip install -e ".[observability]"
```

然后在 `.env` 配置 `ARC_LANGFUSE_PUBLIC_KEY` 和 `ARC_LANGFUSE_SECRET_KEY`，重启服务：

```powershell
.\scripts\start_workbench_local.ps1 -Port 8002 -UseMcp -Restart
```

如果没有 key，演示仍然使用本地 SQLite trace；不要为了演示伪造外部 trace。

## 4. 如果 GitHub MCP 没配置

就直接说：

> 现在 GitHub MCP 是 unavailable，但系统不会假装成功。它会把这件事写进 tool registry 和 approval artifact，保证证据边界诚实。

如果已经配好 token，演示前用这条命令启动：

```powershell
.\scripts\start_workbench_local.ps1 -Port 8002 -UseMcp -Restart
```
