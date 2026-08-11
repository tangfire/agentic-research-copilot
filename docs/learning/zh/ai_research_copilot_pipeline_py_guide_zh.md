# `pipeline.py` 阅读指南

## 这个文件是什么

`pipeline.py` 里的 `ResearchCopilot` 是应用总装配层。它把 provider、retriever、agents、storage、job queue、telemetry、evaluation 组在一起。

它不是某一个算法模块，而是 orchestration facade。facade 的意思是“统一门面”：外部 API 和 LangGraph 节点都通过它调用底层能力。

## 初始化看什么

在 `ResearchCopilot.__init__` 里看：

- `build_model_provider`
- `build_embedding_provider`
- `build_mcp_tool`
- `DocumentStore`
- `RAGEvaluator`
- `RetrievalCoordinator`
- `PlannerAgent`
- `ResearchAgent`
- `ReporterAgent`
- `VerifierAgent`
- `SupervisorAgent`

这告诉你系统启动时有哪些核心组件。

## 对外能力

常用方法：

- `add_document`
- `ingest_document_path`
- `clear_documents`
- `clarify`
- `submit_job`
- `run`
- `list_runs`
- `get_run`
- `runtime_config`
- `replay`

现在没有 `add_memory`，也没有 memory store。

## 运行入口

`run()` 很短：

```python
from .graph_runtime import LangGraphResearchRuntime
return LangGraphResearchRuntime(self).run(request, job_id=job_id)
```

这说明真正的流程控制在 `graph_runtime.py`。

## 最核心的 helper

### `_routes_from_supervisor_decision`

把 supervisor 的 `ConductResearch` tool call 变成 `RetrievalRoute`。

也就是把模型说的：

```text
这个研究单元用 web_search + vector_retrieval，至少 2 条证据
```

变成代码能执行的数据对象。

### `_research_plan_items`

并发执行多个 plan item。

### `_research_plan_item`

执行一个研究单元：

- 调 researcher loop。
- 调 document retrieval。
- 合并 evidence。
- 生成 `ResearchNote`。

### `_build_sections`

把 plan、notes、evidence 变成报告章节。这个方法已经修过，不再生成固定 demo 章节，而是按真实 plan item 生成 topic 相关章节。

### `_estimate_confidence`

根据 evidence 数量、document hits、source 数量、plan 数量粗略估计报告置信度。

### `_rank_evidence_for_report`

报告前对 evidence 排序，让更可靠、更像正式来源的证据排前面。

## MCP 在这里怎么走

初始化时：

```text
settings -> build_mcp_tool -> self.mcp_registry -> self.mcp_tool + self.mcp_tool_catalog
```

如果没有配置 URL 和 tool allowlist，`self.mcp_tool` 是 `None`。

所以 router 只有在真正有外部 MCP 工具时，才会把 `mcp_tool` 放进路线。

现在 GitHub MCP 这条链路还会加载 tool catalog，给 provider 判断工具用。模型选择 MCP 时，需要产出 `mcp_tool_name` 和 `mcp_tool_args`，执行结果会进入 evidence 和 trace。

## 你要记住

`pipeline.py` 的价值不是某个算法，而是把研究应用从“几个函数”提升成“可运行、可检查、可持久化、可评估”的系统。
