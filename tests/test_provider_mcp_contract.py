from agentic_research_copilot.providers import (
    _github_repository_hints,
    _mcp_routing_hints,
    _normalize_researcher_action,
)
from agentic_research_copilot.schemas import (
    MCPToolDescriptor,
    PlanItem,
    ResearcherToolDecisionContract,
)


def test_researcher_action_preserves_structured_mcp_args():
    item = PlanItem(
        id="repo-readme",
        question="How is the repository structured?",
        purpose="Inspect GitHub source evidence.",
        search_query="GitHub repository structure",
    )
    contract = ResearcherToolDecisionContract(
        action="mcp_tool",
        query="read README",
        mcp_tool_name="get_file_contents",
        mcp_tool_args={
            "owner": " langchain-ai ",
            "repo": "open_deep_research",
            "path": "README.md",
            "empty": "",
        },
        confidence=1.2,
    )

    normalized = _normalize_researcher_action(
        contract,
        item,
        ["mcp_tool"],
        [],
        [],
        ["need GitHub source evidence"],
        [
            MCPToolDescriptor(
                name="get_file_contents",
                required_args=["owner", "repo", "path"],
            )
        ],
    )

    assert normalized.action == "mcp_tool"
    assert normalized.mcp_tool_name == "get_file_contents"
    assert normalized.mcp_tool_args == {
        "owner": "langchain-ai",
        "repo": "open_deep_research",
        "path": "README.md",
    }
    assert normalized.confidence == 1.0


def test_researcher_action_clears_mcp_args_for_web_search():
    item = PlanItem(
        id="web",
        question="What public context is available?",
        purpose="Use web search.",
    )
    contract = ResearcherToolDecisionContract(
        action="web_search",
        query="public context",
        mcp_tool_name="get_file_contents",
        mcp_tool_args={"owner": "langchain-ai", "repo": "open_deep_research"},
    )

    normalized = _normalize_researcher_action(
        contract,
        item,
        ["web_search", "mcp_tool"],
        [],
        [],
        ["need another source"],
    )

    assert normalized.action == "web_search"
    assert normalized.mcp_tool_name is None
    assert normalized.mcp_tool_args is None


def test_mcp_routing_hints_extract_github_repository_from_url():
    item = PlanItem(
        id="repo-url",
        question="Research https://github.com/langchain-ai/open_deep_research architecture risks.",
        purpose="Use source-of-truth GitHub evidence.",
    )

    hints = _mcp_routing_hints(
        item,
        [
            MCPToolDescriptor(name="get_file_contents", required_args=["owner", "repo", "path"]),
            MCPToolDescriptor(name="search_code", optional_args=["query"]),
            MCPToolDescriptor(name="list_issues", required_args=["owner", "repo"]),
        ],
    )

    assert hints["github_repository"] == {
        "owner": "langchain-ai",
        "repo": "open_deep_research",
    }
    assert hints["suggested_tools"] == ["get_file_contents", "search_code", "list_issues"]


def test_github_repository_hints_extract_owner_repo_from_chinese_repo_request():
    item = PlanItem(
        id="repo-slug",
        question="调研 langchain-ai/open_deep_research 这个仓库的架构和 issue 风险",
        purpose="分析开源项目",
    )

    assert _github_repository_hints(item) == {
        "owner": "langchain-ai",
        "repo": "open_deep_research",
    }
