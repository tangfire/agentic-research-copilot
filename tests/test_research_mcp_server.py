from pathlib import Path

from agentic_research_copilot.research_mcp_server import (
    check_demo_readiness,
    inspect_runtime_config,
    recall_project_memory,
    recommend_demo_questions,
    search_grounding_corpus,
    search_reference_corpus,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_research_mcp_server_searches_configured_reference_roots(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "odr.md").write_text(
        "Open Deep Research loads MCP tools from mcp_config.url and mcp_config.tools.",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARC_MCP_DEMO_ROOTS", str(tmp_path))

    result = search_reference_corpus("Open Deep Research MCP tools")

    assert result["result_count"] >= 1
    assert "mcp_config" in result["results"][0]["snippet"]


def test_research_mcp_server_recommends_mcp_demo_questions():
    result = recommend_demo_questions("MCP ODR demo")

    assert result["question_count"] >= 1
    assert "search_grounding_corpus" in result["demo_setup"]["default_tools"]
    assert "search_reference_corpus" in result["demo_setup"]["optional_tools"]
    assert any("MCP" in item["question"] for item in result["questions"])


def test_research_mcp_server_runtime_config_handles_unavailable_api(monkeypatch):
    monkeypatch.setenv("ARC_MCP_DEMO_API_BASE", "http://127.0.0.1:9")

    result = inspect_runtime_config("tool readiness")

    assert result["available"] is False
    assert "Start the FastAPI app" in result["hint"]


def test_research_mcp_server_searches_grounding_corpus(monkeypatch):
    def fake_get(url, params=None, timeout=5):
        assert url.endswith("/v1/documents/search")
        assert params["q"] == "contextual retrieval"
        return FakeResponse(
            {
                "result_count": 1,
                "corpus_profile": {"document_count": 2, "keyword_backend": "sqlite_fts5_bm25"},
                "results": [
                    {
                        "title": "Architecture #chunk-1",
                        "source": "docs/architecture.md",
                        "kind": "document-chunk",
                        "score": 0.91,
                        "snippet": "Contextual retrieval uses BM25 and dense fusion.",
                        "metadata": {
                            "retrieval_strategy": "parent_child_dense_bm25_graph_rerank",
                            "keyword_backend": "sqlite_fts5_bm25",
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr("agentic_research_copilot.research_mcp_server.httpx.get", fake_get)

    result = search_grounding_corpus("contextual retrieval")

    assert result["available"] is True
    assert result["result_count"] == 1
    assert result["results"][0]["retrieval"]["keyword_backend"] == "sqlite_fts5_bm25"


def test_research_mcp_server_recalls_project_memory(monkeypatch):
    def fake_get(url, params=None, timeout=5):
        assert url.endswith("/v1/memory/search")
        return FakeResponse(
            {
                "result_count": 1,
                "governance": {"canonical_count": 1},
                "results": [
                    {
                        "key": "project:positioning",
                        "value": "The system is an AI Research Copilot.",
                        "layer": "canonical",
                        "topic": "agentic research",
                        "tags": ["project"],
                        "confidence": 0.86,
                        "metadata": {"governance_status": "active", "last_recall_score": 1.2},
                    }
                ],
            }
        )

    monkeypatch.setattr("agentic_research_copilot.research_mcp_server.httpx.get", fake_get)

    result = recall_project_memory("project positioning")

    assert result["result_count"] == 1
    assert result["results"][0]["layer"] == "canonical"
    assert result["results"][0]["governance_status"] == "active"


def test_research_mcp_server_demo_readiness_handles_unavailable_api(monkeypatch):
    monkeypatch.setenv("ARC_MCP_DEMO_API_BASE", "http://127.0.0.1:9")

    result = check_demo_readiness("demo")

    assert result["available"] is False
    assert any(item["name"] == "api_running" and not item["passed"] for item in result["checks"])


def test_research_mcp_server_demo_readiness_uses_runtime_provider_report(monkeypatch):
    def fake_get(url, params=None, timeout=5):
        if url.endswith("/v1/runtime/config"):
            return FakeResponse(
                {
                    "provider_readiness": {
                        "ready": True,
                        "issues": [],
                        "providers": {
                            "model": {
                                "provider": "openai_compatible",
                                "base_url_configured": True,
                                "api_key_configured": True,
                                "chat_model": "test-chat",
                            },
                            "embedding": {
                                "provider": "openai_compatible",
                                "base_url_configured": True,
                                "api_key_configured": True,
                                "model": "test-embedding",
                            },
                            "search": {
                                "provider": "tavily",
                                "api_key_configured": True,
                                "base_url_configured": False,
                                "model": "",
                            },
                            "rerank": {
                                "provider": "dashscope",
                                "base_url_configured": True,
                                "api_key_configured": True,
                                "model": "qwen3-rerank",
                            },
                        },
                    },
                    "tool_registry": [{"name": "mcp_tool", "loaded": True}],
                    "retrieval": {"keyword_backend": "sqlite_fts5_bm25"},
                }
            )
        if url.endswith("/v1/documents"):
            return FakeResponse([{"id": "doc-1"}])
        if url.endswith("/v1/memory"):
            return FakeResponse([{"key": "memory-1"}])
        if url.endswith("/v1/research/runs"):
            return FakeResponse([{"run_id": "run-1"}])
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("agentic_research_copilot.research_mcp_server.httpx.get", fake_get)

    result = check_demo_readiness("demo")

    assert result["available"] is True
    assert result["passed"] == result["total"]
    assert not [item for item in result["checks"] if not item["passed"]]
