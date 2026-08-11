# `agents/` 阅读指南

## 先说结论

`agents/` 目录里的大多数文件不是“智能算法本体”，而是 agent wrapper，也就是 agent 封装层。真正的大模型决策在 `providers.py`。

你读这个目录时要分清：

- agent 负责组织输入、调用 provider、做少量规范化。
- provider 负责让模型产出结构化结果。
- pipeline/runtime 负责把 agent 输出接到完整运行图里。

## 文件说明

### `planner.py`

`PlannerAgent` 很薄。

它调用：

```python
model_provider.draft_plan(...)
```

输出：

- `research_brief`
- `PlanItem[]`
- assumptions
- success criteria

### `supervisor.py`

`SupervisorAgent` 负责规范化 supervisor 输出。

重点看：

- `decide`
- `_normalize`
- `_normalize_conduct_call`
- `_fallback_route_fields`

它会保证：

- 至少有 `think_tool`。
- 必要 plan item 会被 `ConductResearch` 覆盖。
- `selected_tools` 只保留合法工具。
- 没有本地文档时，不会强行使用 `vector_retrieval`。
- 缺字段时补 fallback route。

### `researcher.py`

这是 agents 里最重要的文件。

它有两个层次：

- `collect`：单次 web search。
- `collect_iterative`：多轮 researcher loop。

`collect_iterative` 每轮会调用：

```python
model_provider.decide_researcher_action(...)
```

模型可以选择：

- `web_search`
- `mcp_tool`
- `think_tool`
- `ResearchComplete`

然后 researcher 会记录：

- 本轮 query。
- 新增证据数量。
- 总证据数量。
- source 数量。
- gaps。
- reflection。
- stopping reason。

### `reporter.py`

调用：

```python
model_provider.compose_report(...)
```

它把 `ReportSection[]` 和 evidence 交给模型，生成最终 `ResearchReport`。

### `verifier.py`

调用：

```python
model_provider.assess_report(...)
```

它判断报告是否有：

- 引用缺失。
- 覆盖不足。
- 置信度太低。
- 需要 revision。

## 为什么这些 agent 看起来薄

这是合理的。因为这个项目采用的是 provider-centered design：

- agent 不把 prompt 写死在自己内部。
- provider 统一处理结构化 LLM 调用。
- schema 保证模型输出可检查。

所以读 agents 时不要困惑：“怎么没看到大段智能逻辑？”真正的大模型方法在 `providers.py`。

## 当前没有什么

当前 agents 里没有 memory agent，也没有本地 workbench MCP agent。

MCP 只在 `researcher.py` 里作为可选外部工具调用出现。现在推荐的外部 MCP 是 GitHub MCP：provider 先根据 tool catalog 选择 `mcp_tool_name`，再给出 `mcp_tool_args`，researcher 负责执行并把结果转成 evidence。
