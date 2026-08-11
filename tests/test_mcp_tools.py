import pytest
from pydantic import BaseModel

from agentic_research_copilot.mcp_tools import MCPToolRegistry, _payload_for_tool, build_mcp_tool
from agentic_research_copilot.settings import AppSettings, GITHUB_MCP_READONLY_URL


class QueryArgs(BaseModel):
    query: str


class FileArgs(BaseModel):
    owner: str
    repo: str
    path: str


class FakeTool:
    def __init__(self, name: str, args_schema: type[BaseModel], description: str = "") -> None:
        self.name = name
        self.args_schema = args_schema
        self.description = description


def test_mcp_registry_preserves_github_readonly_urls():
    registry = MCPToolRegistry(
        server_url="https://api.githubcopilot.com/mcp/readonly",
        tool_names=["search_code"],
    )
    toolset_registry = MCPToolRegistry(
        server_url="https://api.githubcopilot.com/mcp/x/repos/readonly",
        tool_names=["search_code"],
    )
    generic_registry = MCPToolRegistry(
        server_url="https://example.test",
        tool_names=["search_code"],
    )

    assert registry._normalized_server_url() == "https://api.githubcopilot.com/mcp/readonly"
    assert toolset_registry._normalized_server_url() == "https://api.githubcopilot.com/mcp/x/repos/readonly"
    assert generic_registry._normalized_server_url() == "https://example.test/mcp"


def test_mcp_structured_args_take_priority_over_query_fallback():
    tool = FakeTool("get_file_contents", FileArgs)

    payload = _payload_for_tool(
        tool,
        "ignored natural language query",
        {
            "owner": "langchain-ai",
            "repo": "open_deep_research",
            "path": "README.md",
            "unused": "",
        },
    )

    assert payload == {
        "owner": "langchain-ai",
        "repo": "open_deep_research",
        "path": "README.md",
    }


def test_mcp_query_fallback_still_supports_query_only_tools():
    tool = FakeTool("search_code", QueryArgs)

    payload = _payload_for_tool(tool, "LangGraph StateGraph architecture")

    assert payload == {"query": "LangGraph StateGraph architecture"}


def test_mcp_allowlist_blocks_unconfigured_tools():
    allowed = FakeTool("search_code", QueryArgs, "Search repository code")
    blocked = FakeTool("delete_file", FileArgs, "Mutate repository files")
    registry = MCPToolRegistry(
        server_url="https://api.githubcopilot.com/mcp/readonly",
        tool_names=["search_code"],
    )

    selected = registry._select_tools(
        [allowed, blocked],
        query="implementation code",
        tool_name=None,
        tool_args=None,
    )
    blocked_selection = registry._select_tools(
        [allowed, blocked],
        query="delete README",
        tool_name="delete_file",
        tool_args=None,
    )

    assert selected == [allowed]
    assert blocked_selection == []


def test_strict_mcp_auth_required_without_token_fails_fast():
    settings = AppSettings(
        strict_providers=True,
        mcp_enabled=True,
        mcp_server_url=GITHUB_MCP_READONLY_URL,
        mcp_tools=["search_code"],
        mcp_auth_required=True,
        mcp_auth_token="",
    )

    with pytest.raises(RuntimeError, match="ARC_MCP_AUTH_REQUIRED=true"):
        build_mcp_tool(settings)
