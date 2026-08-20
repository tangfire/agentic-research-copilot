# Research Desk 观测设计

## 1. 为什么需要 Langfuse

Research Desk 的主产品是研究闭环，不是监控平台。研究过程中仍然要回答：

- Planner 为什么选择这些 specialist worker；
- worker 调用了哪些工具，哪些调用成功或失败；
- 哪些证据进入了报告；
- Verifier 发现了什么缺口；
- 最终 citation precision、context recall、faithfulness 和 constraint coverage 是多少。

以前这些信息主要塞在右侧栏，结果页面既重复报告内容，又像一个很长的调试面板。现在采用两层设计：

```text
Workbench 主页面
  -> 只看下一步、运行摘要、失败原因和入口

Langfuse
  -> 看 agent / tool / evaluator 的时间线、输入输出摘要、延迟和分数

SQLite 本地 ledger
  -> 保存完整 run、checkpoint、trace、replay 和 export bundle
```

Langfuse 是可选的外部观测层，不是运行依赖，也不是事实源。没有 Langfuse 时，项目仍然可以正常研究、回放和导出。

## 2. 安装

在项目根目录执行：

```powershell
python -m pip install -e ".[observability]"
```

如果只想使用本地 trace，不需要安装这个 extra。

## 3. 创建 Langfuse 配置

在 Langfuse 创建一个项目，取得 `public key` 和 `secret key`，然后写入本地 `.env`：

```text
ARC_OBSERVABILITY_PROVIDER=langfuse
ARC_LANGFUSE_PUBLIC_KEY=你的_public_key
ARC_LANGFUSE_SECRET_KEY=你的_secret_key
ARC_LANGFUSE_HOST=https://cloud.langfuse.com
ARC_LANGFUSE_ENVIRONMENT=local
ARC_LANGFUSE_RELEASE=research-desk
ARC_LANGFUSE_CAPTURE_CONTENT=false
```

注意：

- `.env` 不能提交到 git；
- 不要把 Langfuse key 放进 prompt、tool arguments、前端或 agent 可见的上下文；
- `ARC_LANGFUSE_CAPTURE_CONTENT=false` 是默认隐私边界；
- publisher 会过滤 token、secret、password、api key 和 authorization 字段，并截断事件文本；
- GitHub token 和模型 provider key 不会被发送到 Langfuse。

如果暂时没有 Langfuse key，保持：

```text
ARC_OBSERVABILITY_PROVIDER=none
```

页面会显示“本地观测已启用”，这不是错误。

## 4. 启动并验证

修改 `.env` 后必须重启本地服务，避免旧进程继续使用旧配置：

```powershell
.\scripts\start_workbench_local.ps1 -Port 8002 -UseMcp -Restart
```

检查外部观测状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8002/v1/observability/status
```

配置成功时，结果中的 `enabled` 应该是 `True`。如果是 `False`，先看 `reason`，常见原因是 key 没有配置、Langfuse extra 没安装或旧服务没有重启。

## 5. 一次完整使用

1. 打开 `http://127.0.0.1:8002/`。
2. 新建会话。
3. 输入 repo、技术问题和团队约束。
4. 点击发送，等待计划生成。
5. 检查计划后点击“确认并开始研究”。
6. 研究完成后，右侧只看运行摘要和质量概览。
7. 点击“查看 Langfuse Trace”进入详细过程。
8. 如果 Langfuse 未启用，点击“查看本地 Trace JSON”或“查看会话事件”。

本地 trace 和 Langfuse 的对应关系：

| 本地对象 | Langfuse 对象 |
| --- | --- |
| `ResearchRun` | `research_run` 根 chain |
| `RunTraceEvent(kind=tool_call)` | tool observation |
| `RunTraceEvent(kind=handoff)` | agent observation |
| `RunTraceEvent(kind=verification/evaluation)` | evaluator observation |
| `citation_precision` 等指标 | numeric score |

## 6. 为什么不让 Langfuse 取代 SQLite

Langfuse 适合查看时间线、筛选 trace、比较运行和观察质量分数；本项目还需要：

- 本地离线运行；
- 失败后 replay；
- 导出完整 session bundle；
- 在没有外网时保留事实；
- 让测试直接断言 trace 和 evaluation。

因此 SQLite 是本地 source of truth，Langfuse 是可选的 read-oriented observability sink。外部观测发送失败不会让报告丢失，也不会改变本地 run 状态。

## 7. 面试时怎么讲

> 我把观测分成两层。运行事实由执行节点直接写入 SQLite ledger，包括 checkpoint、tool invocation、trace、evaluation 和 replay artifact；Langfuse 是可选的外部观测 sink，用来把一次研究 run 展开成 root chain、specialist/tool/evaluator observations 和质量 scores。这样既能本地离线复盘，又能用专业工具看调用时间线。Langfuse 不参与权限判断，也不是 source of truth，token 不进入 agent context，外部观测失败不会影响本地报告。

## 8. 当前边界

这版做的是“可用的 run-level tracing”，不是把整个系统改造成 Langfuse 专属运行时：

- 不把 Langfuse SDK 写进每个业务模块；
- 不发送完整敏感 prompt 和 token；
- 不依赖 Langfuse 才能启动；
- 不用右侧栏复制整条 trace；
- 不把一次真实实验中没有出现的工具调用伪装成成功。

后续如果需要更细的 provider token/cost 统计，可以在模型 provider 边界补充标准化 usage 字段；这属于观测增强，不改变研究主链路。
