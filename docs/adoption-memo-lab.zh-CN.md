# 开源引入评审实验说明

这个项目当前最真实的部署方向是：

> 面向小型研发团队的开源引入评审和技术决策研究台。

输入是 repo、技术问题和团队约束；输出是带公开证据、本地约束、引用、trace 和 evaluation 的技术采用 memo。

## 为什么这不是玩具

真实团队不会每次都把约束重新复制进 prompt。团队会有固定的技术栈、部署边界、风险偏好、预算、运维能力和评审口径。这个实验把这些内容写成私有语料，放在 `examples/adoption-lab/team-context/`，然后让系统在研究过程中自动检索它们。

这让项目有了一个很具体的用途：

- 评审一个 GitHub 仓库是否值得引入
- 比较一个技术方案是否适合当前团队
- 生成可以复盘的技术决策 memo
- 用指标发现报告是否只是“看起来像研究”

## 实验入口

运行：

```powershell
python scripts/run_adoption_memo_experiment.py --clean
```

默认模式是 deterministic，用来稳定复现实验报告、trace 和指标。要走真实模型、搜索、embedding 和 rerank 配置，可以运行：

```powershell
python scripts/run_adoption_memo_experiment.py --clean --mode real
```

真实模式更像 provider 集成测试，可能受网络、模型响应时间和搜索结果波动影响。

默认实验问题是：

> Northstar Platform 这个 5 人 Python/FastAPI 平台团队，是否应该把 `langchain-ai/langgraph` 作为内部开源引入评审和技术决策研究台的工作流运行时？

实验会生成：

- `examples/adoption-lab/outputs/adoption-memo.report.md`
- `examples/adoption-lab/outputs/adoption-memo.trace.json`
- `examples/adoption-lab/outputs/adoption-memo.evaluation.json`
- `examples/adoption-lab/outputs/adoption-memo.summary.json`
- `examples/adoption-lab/outputs/adoption-memo.analysis.md`

## 怎么看指标

- `context_recall`：本地团队约束是否被检索并进入证据链。
- `citation_precision`：报告 section 是否都有引用。
- `faithfulness_proxy`：生成内容和引用证据是否有足够重合。
- `source_diversity`：是否不是单一来源撑完整篇报告。
- `constraint_recall`：报告是否提到团队的关键约束。
- `expected_term_recall`：报告是否覆盖预设评审要点。
- `graph_signal_hits`：图检索是否真的贡献了候选证据，而不是只开了配置。

## 图结构是否硬凑

在这个场景里，图结构是合理的，但边界要讲清楚。

它合理的原因是：开源引入评审不是一次性问答。它有计划、证据搜集、本地约束检索、报告生成、验证、评价和可能的修订。每一步的下一步都依赖状态，例如证据是否足够、引用是否完整、是否触发 revision。

它不该被滥用的地方是：如果只是问“这个 repo 是什么”，或者只总结一个 README，那么 graph 就是过度设计。这个项目应该把图结构用于需要分支、handoff、质量门禁和 replay 的任务。
