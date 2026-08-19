# Project Guide

这是这个仓库的总入口。

## 一句话

这是一个面向开源引入评审的 conversational research agent runtime。

## 主链路

```text
session -> memory -> plan -> confirm -> tool loop -> report -> trace/eval -> export
```

## 三个层次

- `workflow node`：LangGraph 里的控制流节点，负责什么时候该跑下一步。
- `agent`：模型能力封装，负责一次规划、监督、研究、写作或验证。
- `specialist worker`：RepoSignal / ArchitectureFit / OpsRisk 三个责任边界，在 research stage 内真正执行各自的工具循环和证据收集。

你可以把它理解成：

```text
node 负责调度和持久化状态
agent capability 负责规划、监督、写作或验证
specialist worker 负责某个责任边界下的工具循环和证据归属
```

## 先读什么

1. [README.md](../README.md)
2. [Architecture](architecture.md)
3. [OpenClaw / Hermes Design Notes](openclaw-hermes-design-notes.zh-CN.md)
4. [Agent Maturity Pack](agent-maturity-pack.zh-CN.md)
5. [Tool Loop And HITL](tool-loop-and-hitl.zh-CN.md)
6. [Memory And Constraint Evaluation](memory-and-constraint-eval.zh-CN.md)
7. [Autumn Recruiting Playbook](autumn-recruiting-playbook.zh-CN.md)
8. [Interview Question Bank](interview-question-bank.zh-CN.md)
9. [Demo Script](demo-script.zh-CN.md)
10. [Usage Guide](usage-guide.zh-CN.md)

## 怎么启动

```powershell
uvicorn agentic_research_copilot.server:create_app --factory --host 127.0.0.1 --port 8000
```

## 怎么用

1. 建一个 session。
2. 输入 repo、技术问题、团队约束。
3. 等 plan draft。
4. 确认计划。
5. 看 report、trace、evaluation、constraint coverage。

## 读文档时别混的点

- `agent` 不是“节点后再跑一套隐藏 workflow”。
- `specialist worker` 是在线执行单元，但它运行在同一条 LangGraph research stage 内。
- `memory precision` 和 `constraint coverage` 多半是工程 proxy，不是学术 benchmark。
- `MCP unavailable` 不是故障，缺 token 时本来就应该显式降级。
