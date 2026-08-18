from collections.abc import Sequence
from pathlib import Path

from agentic_research_copilot.dev_fixtures import FixtureResearchModelProvider
from agentic_research_copilot.agents import ResearchAgent
from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.provider_base import ModelUsage
from agentic_research_copilot.schemas import (
    EvidenceItem,
    MCPToolDescriptor,
    PlanItem,
    ResearchRequest,
    ResearcherToolDecisionContract,
)
from agentic_research_copilot.settings import AppSettings


class StructuredMCPDecisionProvider(FixtureResearchModelProvider):
    def decide_researcher_action(
        self,
        *,
        item: PlanItem,
        available_tools: Sequence[str],
        previous_queries: Sequence[str],
        evidence: Sequence[EvidenceItem],
        gaps: Sequence[str],
        iteration: int,
        max_iterations: int,
        mcp_tools: Sequence[MCPToolDescriptor] = (),
    ) -> tuple[ResearcherToolDecisionContract, ModelUsage]:
        if not evidence:
            return (
                ResearcherToolDecisionContract(
                    action="web_search",
                    query="github project architecture overview",
                    rationale="Start with a broad public source.",
                    confidence=0.8,
                ),
                ModelUsage(provider="test", model="structured-mcp-test"),
            )
        selected_tool = mcp_tools[0].name if mcp_tools else "get_file_contents"
        if selected_tool == "search_code":
            tool_args = {"query": "repository architecture implementation"}
        else:
            tool_args = {
                "owner": "langchain-ai",
                "repo": "open_deep_research",
                "path": "README.md",
            }
        return (
            ResearcherToolDecisionContract(
                action="mcp_tool",
                query="read repository README",
                mcp_tool_name=selected_tool,
                mcp_tool_args=tool_args,
                rationale="Read the source-of-truth repository file through GitHub MCP.",
                confidence=0.85,
            ),
            ModelUsage(provider="test", model="structured-mcp-test"),
        )


def test_research_agent_passes_structured_mcp_args_to_tool():
    mcp_calls = []

    def fake_search(query):
        return [
            {
                "title": "Web overview",
                "source": "tavily",
                "url": "https://example.test/overview",
                "snippet": "General project overview.",
                "content": "General project overview.",
                "score": 0.8,
            }
        ]

    def fake_mcp(query, tool_name=None, tool_args=None):
        mcp_calls.append((query, tool_name, tool_args))
        return [
            {
                "title": "GitHub README",
                "source": "mcp:get_file_contents",
                "kind": "mcp",
                "snippet": "README content from GitHub.",
                "content": "README content from GitHub MCP with project architecture details.",
                "score": 0.82,
                "metadata": {"mcp_tool_name": tool_name},
            }
        ]

    agent = ResearchAgent(
        fake_search,
        model_provider=StructuredMCPDecisionProvider(),
        mcp_tool=fake_mcp,
        mcp_tool_catalog=[
            MCPToolDescriptor(
                name="get_file_contents",
                description="Read file contents from a GitHub repository.",
                required_args=["owner", "repo", "path"],
            )
        ],
        source_reader_enabled=False,
        max_iterations=2,
    )
    collection = agent.collect_iterative(
        PlanItem(
            id="github-readme",
            question="How does the GitHub project structure its research workflow?",
            purpose="Verify structured MCP arguments.",
            search_query="GitHub project research workflow",
        ),
        ["github project architecture"],
        min_evidence=2,
        min_sources=2,
    )

    expected_args = {
        "owner": "langchain-ai",
        "repo": "open_deep_research",
        "path": "README.md",
    }
    assert mcp_calls == [("read repository README", "get_file_contents", expected_args)]
    assert collection.completed_reason == "sufficiency_met"
    assert collection.iterations[1]["mcp_tool_name"] == "get_file_contents"
    assert collection.iterations[1]["mcp_tool_args"] == expected_args
    assert collection.iterations[1]["source_channel"] == "mcp"
    assert collection.iterations[1]["result_count"] == 1
    assert collection.evidence[1].metadata["mcp_tool_args"] == expected_args


def test_pipeline_trace_records_structured_github_mcp_tool_call(tmp_path: Path):
    mcp_calls = []

    def fake_search(query):
        return [
            {
                "title": "Project overview",
                "source": "tavily",
                "url": "https://example.test/github-project",
                "snippet": "Public overview of a GitHub project.",
                "content": "Public overview of a GitHub project architecture.",
                "score": 0.82,
            }
        ]

    def fake_mcp(query, tool_name=None, tool_args=None):
        mcp_calls.append((query, tool_name, tool_args))
        return [
            {
                "title": "GitHub code search result",
                "source": "mcp:search_code",
                "kind": "mcp",
                "snippet": "GitHub MCP found repository code evidence.",
                "content": "GitHub MCP found repository code evidence for the architecture.",
                "score": 0.8,
                "metadata": {"mcp_tool_name": tool_name, "mcp_tool_args": tool_args or {}},
            }
        ]

    provider = StructuredMCPDecisionProvider()
    copilot = ResearchCopilot(
        settings=AppSettings(
            storage_path=str(tmp_path / "github-mcp.sqlite"),
            langgraph_checkpoint_path=str(tmp_path / "github-mcp-checkpoints.sqlite"),
            research_max_iterations=2,
            rag_min_evidence_per_item=2,
            rag_min_source_diversity=2,
        ),
        search_tool=fake_search,
        model_provider=provider,
        embedding_provider=provider,
    )
    catalog = [
        MCPToolDescriptor(
            name="search_code",
            description="Search GitHub code for implementation evidence.",
            optional_args=["query"],
        )
    ]
    copilot.mcp_tool = fake_mcp
    copilot.mcp_tool_catalog = catalog
    copilot.router.mcp_enabled = True
    copilot.researcher.mcp_tool = fake_mcp
    copilot.researcher.mcp_tool_catalog = catalog

    result = copilot.run(
        ResearchRequest(
            topic="GitHub project architecture evidence",
            max_sections=1,
        )
    )

    assert result.status == "completed"
    assert mcp_calls
    mcp_trace = [
        event
        for event in result.trace
        if event.kind == "tool_call" and event.provider == "model_context_protocol"
    ]
    assert mcp_trace
    assert mcp_trace[0].tool_name == "search_code"
    assert mcp_trace[0].metadata["source_channel"] == "mcp"
    assert mcp_trace[0].metadata["mcp_tool_args"]["query"]
