# Memory And Constraint Evaluation

本文档解释本项目 v2 的 memory 分层、抽取边界、memory quality 和 constraint coverage gate。你可以把它当成学习 agent memory 的主线文档。

参考资料：

- Mem0 overview: https://docs.mem0.ai/platform/overview
- OpenAI Agents SDK sessions: https://openai.github.io/openai-agents-python/sessions/
- CrewAI memory docs: https://docs.crewai.com/concepts/memory
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence

## 1. 为什么 Research Agent 需要 Memory

普通 RAG 多半是：

```text
question -> retrieve docs -> answer
```

但技术采用评审不是孤立问题。用户真正想问的通常是：

```text
在我们这个团队、这个技术栈、这个部署方式、这个风险偏好下，
某个开源项目值不值得引入？
```

所以同一个 repo，对不同团队结论可能完全不同。

例如：

- 5 人团队，Python/FastAPI，单机 Docker Compose，必须可回滚。
- 大公司团队，Kubernetes，多服务，允许引入复杂 observability。
- 学生秋招项目，重点是能讲清 agent runtime，而不是企业级可用性。

如果没有 memory，用户每次都要复制这些约束。v2 的目标是把这些约束保存下来，并在 planning 和 retrieval 时自动注入。

## 2. 三层 Memory

代码路径：

```text
src/agentic_research_copilot/schemas.py
src/agentic_research_copilot/storage.py
src/agentic_research_copilot/agent.py
```

### 2.1 user_memory

长期偏好和个人目标。

例子：

```text
我想用这个项目备战秋招。
我偏向 Python/FastAPI。
我希望文档写详细一点，方便学习。
```

用途：

- 影响解释风格。
- 影响计划侧重点。
- 影响 demo 和面试话术。

### 2.2 project_memory

项目或团队约束，是最重要的一层。

例子：

```text
团队是 5 人 Python/FastAPI。
部署只接受单机 Docker Compose。
必须有 rollback plan。
MCP token 缺失时不能伪造 GitHub evidence。
```

用途：

- 作为 hard constraint 注入 planning request。
- 同步进入本地 DocumentStore，参与 vector/BM25/graph retrieval。
- 被 constraint coverage gate 逐条检查。

### 2.3 session_memory

当前会话里的临时事实。

例子：

```text
这次评估对象是 langchain-ai/langgraph。
这次输出 max_sections=2。
用户希望先确认计划再开跑。
```

用途：

- 保持当前 session 连贯。
- 帮助 follow-up。
- 不一定跨 session 长期有效。

## 3. MemoryItem 数据结构

```text
memory_id
scope: user | project | session
kind: preference | constraint | decision | fact | todo
content
source_message_id
confidence
created_at / updated_at
metadata
```

重要约定：

- `scope=project` 或 `kind=constraint` 会被标记为 `hard_constraint=true`。
- project memory 会同步写入 DocumentStore，document id 是 `memory:{memory_id}`。
- 删除 project memory 时，对应本地 document 也会删除。

## 4. Memory Extraction 的边界

v2 仍使用 heuristic extractor，不引入 Mem0 SDK，也不直接上 LLM extractor。

它会识别：

- 明确团队约束：团队、约束、我们、部署、回滚、FastAPI、Docker 等。
- 明确用户偏好：秋招、面试、我的目标、我想等。
- 足够长的当前会话事实。

每次 user message 后会生成 `MemoryExtractionResult`：

```text
source_message_id
session_id
candidates
accepted
rejected
reason
metadata
```

API：

```text
GET /v1/agent/sessions/{session_id}/memory/evaluation
```

为什么先用 heuristic：

- 面试项目里更容易讲清实现。
- 测试稳定。
- 不把 deterministic/real model 的差异混在 memory 基础能力里。
- 后续可以加 LLM extractor，再拿 labeled fixture 做 precision/recall。

## 5. Memory Quality 指标

v2 的 memory evaluation 是 proxy，不是严格学术评测。

```text
memory_precision = accepted / candidates
memory_recall = 有 accepted memory 的 user message 数 / user message 总数
project_constraint_count = accepted 里 project 或 constraint 的数量
```

它能回答：

- 用户给了约束，系统是否抽出来？
- 抽出来的 memory 是否被接受？
- project constraint 是否足够进入 planning？
- 删除 memory 后，endpoint 是否不再返回？

局限：

- 没有人工标注集，所以 recall 只是代理指标。
- heuristic extractor 可能漏掉隐含约束。
- 重复约束会被 rejected，不代表 extractor 失败。

## 6. Constraint Recall 和 Constraint Coverage 的区别

真实实验里你看到过：

```text
constraint_recall = 0.25
```

这不是“项目没用”的证明，反而是很真实的产品问题：

- memory 可能写进去了。
- planner 可能看到了。
- 但 reporter 最终没有稳定逐条覆盖。

所以需要两个概念：

### 6.1 constraint_recall

实验级指标，问的是：

```text
预期团队约束里，有多少被最终报告提到？
```

它更像 adoption memo 实验的外部评估。

脚本：

```powershell
python scripts/run_adoption_memo_experiment.py --clean --mode real --max-sections 2
```

### 6.2 constraint_coverage

运行时指标，问的是：

```text
本次 run 里的 hard constraints，有多少被 report sections 或 evidence 覆盖？
```

数据结构：

```text
constraint_id
content
covered
matched_sections
matched_evidence
confidence
reason
```

API：

```text
GET /v1/research/runs/{run_id}/constraint-coverage
```

脚本：

```powershell
python scripts/run_memory_constraint_eval.py
```

## 7. Constraint Coverage 怎么算

代码路径：

```text
src/agentic_research_copilot/constraint_evaluation.py
```

流程：

1. 从 `ResearchRequest.topic` 中提取 `[project/constraint]` 行。
2. 或从 session project memory 中提取 hard constraints。
3. 对每条 constraint 做 token overlap。
4. 检查 report sections。
5. 检查 evidence title/source/snippet/content。
6. 生成 `ConstraintCoverage`。
7. 汇总 score。
8. 把结果写入 SQLite。

阈值：

```text
score >= 0.6: pass
0.4 <= score < 0.6: warning
score < 0.4: evaluation failed
```

注意：低 coverage 不会丢弃报告，因为真实产品里坏结果也有复盘价值。

## 8. Planner 如何使用 Memory

确认前，agent 会把最近用户消息和相关 memory 组合成新的 `ResearchRequest`：

```text
Conversation research request:
- 用户最近消息

Hard project constraints that must be addressed in the final memo:
- [project/constraint] 团队是 5 人 Python/FastAPI，单机 Docker Compose，必须可回滚。

Relevant saved memory:
- [user/preference] 我想用这个项目备战秋招。

Deliverable: prepare a citation-backed technical research memo with plan, evidence, trace, evaluation, and explicit coverage of every hard project constraint.
```

这就是为什么用户不需要每次复制长 prompt。

## 9. 删除 Memory 后会发生什么

删除接口：

```text
DELETE /v1/memory/{memory_id}
```

效果：

- SQLite memory item 删除。
- 如果是 project memory，对应 `memory:{memory_id}` 文档也删除。
- session memory endpoint 不再返回。
- 之后新的 planning request 不再注入这条约束。
- 新 run 的 constraint coverage 不再使用这条 memory。

已经完成的旧 run 不会被重写，因为旧报告和旧 trace 是历史事实。

## 10. 面试讲法

一句话：

> 我把 memory 分成 user/project/session 三层，其中 project memory 会进入本地知识库和 planning prompt，并作为 hard constraints 被 constraint coverage gate 检查。

被问“为什么不直接用 Mem0”：

> 我借鉴了 Mem0 的长期 memory 和检索思想，但 v2 没引入 SDK。因为这是学习项目，我希望自己实现最小闭环：抽取、存储、检索、注入 planning、进入本地 KB、最后评估覆盖率。

被问“constraint_recall 低怎么办”：

> 这说明真实产品问题被测出来了。v2 的修复是把 project memory 标成 hard constraint，planner prompt 明确要求覆盖，report 后再做 constraint coverage gate。低于阈值会进入 evaluation notes，严重时标记 evaluation failed，但保留报告便于复盘。

被问“memory 会不会越存越脏”：

> 会，所以 v2 不把所有话都长期保存。它只保存明确偏好、团队约束和会话事实，并记录 MemoryExtractionResult。后续可以引入人工标注集和 LLM extractor 做更严格的 precision/recall。

