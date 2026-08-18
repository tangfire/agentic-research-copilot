# 开源引入评审实验说明

这个项目当前最真实的部署方向是：

> 面向小型研发团队的开源引入评审和技术决策研究台。

输入是 `repo / 技术问题 / 团队约束`；输出是带 GitHub 或 Web 证据、本地团队约束、引用、trace 和 evaluation 的技术采用 memo。

## 为什么这不是玩具

真实团队不会每次都把约束重新复制进 prompt。团队会有固定的技术栈、部署边界、风险偏好、预算、运维能力和评审口径。这个实验把这些内容写成私有语料，放在 `examples/adoption-lab/team-context/`，然后让系统在研究过程中自动检索它们。

这让项目有了一个具体用途：

- 评审一个 GitHub 仓库是否值得引入。
- 判断一个技术方案是否适合当前团队。
- 生成可复盘的技术决策 memo。
- 用指标检查报告是不是只是在“看起来像研究”。

## 实验入口

默认运行真实模式：

```powershell
python scripts/run_adoption_memo_experiment.py --clean
```

真实模式会使用当前 `.env` 里的真实 provider：chat model、Tavily 搜索、embedding、rerank、本地 Qdrant、trace 和 evaluation。它不是假跑。

如果你只是想做离线回归测试，可以显式使用 fixture：

```powershell
python scripts/run_adoption_memo_experiment.py --clean --mode fixture
```

fixture 的作用是稳定测试流程和数据结构，不适合用来证明产品真实有效。

## GitHub MCP 怎么接

推荐用 GitHub MCP 作为外部 MCP，因为它能提供 repository files、code search、issues、pull requests、releases 这类开发者源头证据。当前项目在 `--mode real --use-mcp` 时会强制使用 GitHub read-only endpoint 和 GitHub 工具白名单，避免误把旧本地 MCP 工具当成 GitHub MCP。

配置方式：

```powershell
$env:ARC_MCP_ENABLED="true"
$env:ARC_MCP_SERVER_URL="https://api.githubcopilot.com/mcp/readonly"
$env:ARC_MCP_TOOLS="search_repositories,get_file_contents,search_code,list_issues,issue_read,search_issues,list_pull_requests,pull_request_read,get_latest_release"
$env:ARC_MCP_AUTH_REQUIRED="true"
$env:ARC_MCP_AUTH_TOKEN="<github-token>"
```

也可以不用 `ARC_MCP_AUTH_TOKEN`，改用这些常见变量之一：

```powershell
$env:GH_TOKEN="<github-token>"
$env:GITHUB_TOKEN="<github-token>"
$env:GITHUB_PERSONAL_ACCESS_TOKEN="<github-token>"
```

然后运行：

```powershell
python scripts/run_adoption_memo_experiment.py --clean --mode real --use-mcp
```

完整实验比较慢，所以可以先单独检查 GitHub MCP：

```powershell
python scripts/check_github_mcp.py
```

这个脚本只加载 GitHub MCP 工具目录，不跑完整研究。输出里应该看到 `auth_token_configured: true` 和非 0 的 `loaded_tool_count`。

如果 token 缺失，runner 会快速失败并告诉你缺少 GitHub MCP token。这个行为是故意的：没有 GitHub MCP 证据时，系统不应该假装自己用了 MCP。

## 默认实验问题

默认实验会模拟：

> Northstar Platform 这个 5 人 Python/FastAPI 平台团队，是否应该把 `langchain-ai/langgraph` 作为内部开源引入评审和技术决策研究台的 workflow runtime？

它会结合：

- 本地团队约束文档
- 公开 Web/GitHub 证据
- graph 结构是否真的必要
- 风险
- pilot 计划
- rollback 计划

## 输出文件

实验会写入：

- `examples/adoption-lab/outputs/adoption-memo.report.md`
- `examples/adoption-lab/outputs/adoption-memo.trace.json`
- `examples/adoption-lab/outputs/adoption-memo.evaluation.json`
- `examples/adoption-lab/outputs/adoption-memo.summary.json`
- `examples/adoption-lab/outputs/adoption-memo.analysis.md`

## 怎么看指标

- `context_recall`：本地团队约束是否被检索并进入证据链。
- `citation_precision`：报告 section 是否都有引用。
- `faithfulness_proxy`：生成内容和引用证据是否有足够重合。
- `source_diversity`：报告是否依赖多个来源，而不是单一来源撑完整篇。
- `constraint_recall`：报告是否提到团队关键约束。
- `expected_term_recall`：报告是否覆盖预设评审要点。
- `graph_signal_hits`：图检索是否真的贡献了候选证据，而不是只打开了配置。

## 图结构是不是硬凑

在这个场景里，图结构是合理的，但边界必须讲清楚。

它合理的原因是：开源引入评审不是一次性问答。它有计划、证据搜集、本地约束检索、报告生成、验证、评价和可能的修订。每一步的下一步都依赖状态，例如证据是否足够、引用是否完整、是否触发 revision。

它不应该被滥用的地方是：如果只是问“这个 repo 是什么”，或者只总结一个 README，那 graph 就是过度设计。这个项目应该把图结构用于需要分支、handoff、质量门禁和 replay 的任务。

## 本次真实运行结论样例

最近一次真实主链路运行使用了：

- Chat model: `qwen-plus`
- Search: `tavily`
- Embedding: `qwen3.7-text-embedding`
- Rerank: `dashscope`
- MCP: 未启用，因为本机没有 GitHub token

结果：

- `status`: `completed`
- `evaluation_passed`: `true`
- `source_count`: `5`
- `context_recall`: `1.0`
- `citation_precision`: `1.0`
- `faithfulness_proxy`: `0.816`
- `graph_signal_hits`: `18`

解释：这个结果证明真实模型、真实搜索、本地约束检索、报告合成、验证、评价和 trace 主链路可以跑通。但它还没有证明 GitHub MCP 的价值，因为当前机器缺少 GitHub MCP token。要证明 MCP，需要配置 token 后再跑 `--use-mcp`。
