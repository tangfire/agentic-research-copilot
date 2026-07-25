import pytest

from agentic_research_copilot.search import (
    AnthropicWebSearchClient,
    ArxivSearchClient,
    BraveSearchClient,
    ExaSearchClient,
    LinkupSearchClient,
    OpenAIWebSearchClient,
    PerplexitySearchClient,
    PubMedSearchClient,
    SerpAPISearchClient,
    StrictSearchTool,
    TavilySearchClient,
    build_search_tool,
)
from agentic_research_copilot.settings import AppSettings


class FakeResponse:
    def __init__(self, payload, text=None):
        self.payload = payload
        self.text = text if text is not None else ""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, *, post_payload=None, get_payload=None, text=None):
        self.post_payloads = post_payload if isinstance(post_payload, list) else [post_payload]
        self.get_payloads = get_payload if isinstance(get_payload, list) else [get_payload]
        self.texts = text if isinstance(text, list) else [text]
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, endpoint, headers=None, json=None):
        self.requests.append(("post", endpoint, json))
        payload = self.post_payloads.pop(0) if self.post_payloads else None
        return FakeResponse(payload or {})

    def get(self, endpoint, headers=None, params=None):
        self.requests.append(("get", endpoint, headers, params))
        payload = self.get_payloads.pop(0) if self.get_payloads else None
        text = self.texts.pop(0) if self.texts else None
        return FakeResponse(payload or {}, text=text)


def test_build_search_tool_supports_formal_providers():
    for provider in [
        "tavily",
        "brave",
        "serpapi",
        "exa",
        "perplexity",
        "arxiv",
        "pubmed",
        "linkup",
        "openai_web",
        "anthropic_web",
    ]:
        tool = build_search_tool(AppSettings(search_provider=provider, search_api_key="key"))
        assert tool is not None


def test_formal_search_providers_return_empty_without_key():
    assert TavilySearchClient(api_key="").search("agentic rag") == []
    assert BraveSearchClient(api_key="").search("agentic rag") == []
    assert SerpAPISearchClient(api_key="").search("agentic rag") == []
    assert ExaSearchClient(api_key="").search("agentic rag") == []
    assert PerplexitySearchClient(api_key="").search("agentic rag") == []
    assert LinkupSearchClient(api_key="").search("agentic rag") == []
    assert OpenAIWebSearchClient(api_key="").search("agentic rag") == []
    assert AnthropicWebSearchClient(api_key="").search("agentic rag") == []


def test_strict_search_tool_raises_on_empty_results():
    tool = build_search_tool(
        AppSettings(
            strict_providers=True,
            search_provider="tavily",
            search_api_key="",
        )
    )

    assert tool is not None
    with pytest.raises(RuntimeError, match="Strict search provider"):
        tool("agentic rag")


def test_strict_search_tool_compacts_long_queries_before_failing_the_run():
    calls = []

    def fake_search(query):
        calls.append(query)
        if len(calls) == 1:
            return []
        return [
            {
                "title": "LangGraph persistence",
                "source": "tavily",
                "url": "https://docs.langchain.com/oss/python/langgraph/persistence",
                "snippet": "Checkpointing reference.",
                "content": "Checkpointing reference.",
                "score": 0.8,
                "metadata": {"query": query},
            }
        ]

    tool = StrictSearchTool("tavily", fake_search)
    results = tool(
        "What are the core architectural patterns of Open Deep Research-style orchestration, "
        "and how do they compare to LangGraph persistence patterns?"
    )

    assert len(calls) == 2
    assert len(calls[1]) < len(calls[0])
    assert results[0]["metadata"]["query_rewrite_strategy"] == "strict_provider_compacted"
    assert results[0]["metadata"]["original_query"] == calls[0]


def test_tavily_adapter_maps_results(monkeypatch):
    fake = FakeClient(
        post_payload={
            "results": [
                {
                    "title": "Tavily result",
                    "url": "https://example.com/a",
                    "content": "research evidence",
                    "raw_content": "Long webpage content with detailed agentic RAG evidence.",
                    "score": 0.9,
                }
            ]
        }
    )
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = TavilySearchClient(api_key="key").search("agentic rag")

    assert results[0]["source"] == "tavily"
    assert results[0]["url"] == "https://example.com/a"
    assert results[0]["metadata"]["query"] == "agentic rag"
    assert results[0]["metadata"]["depth"] == "basic"
    assert results[0]["metadata"]["raw_content_requested"] is True
    assert results[0]["metadata"]["raw_content_available"] is True
    assert results[0]["raw_content"] == "Long webpage content with detailed agentic RAG evidence."
    assert fake.requests[0][2]["search_depth"] == "basic"
    assert fake.requests[0][2]["include_raw_content"] is True


def test_brave_adapter_maps_results(monkeypatch):
    fake = FakeClient(
        get_payload={
            "web": {
                "results": [
                    {"title": "Brave result", "url": "https://example.com/b", "description": "search evidence"}
                ]
            }
        }
    )
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = BraveSearchClient(api_key="key").search("agentic rag")

    assert results[0]["source"] == "brave"
    assert results[0]["snippet"] == "search evidence"
    assert fake.requests[0][2]["X-Subscription-Token"] == "key"


def test_serpapi_adapter_maps_results(monkeypatch):
    fake = FakeClient(
        get_payload={
            "organic_results": [
                {"title": "SerpAPI result", "link": "https://example.com/c", "snippet": "organic evidence"}
            ]
        }
    )
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = SerpAPISearchClient(api_key="key").search("agentic rag")

    assert results[0]["source"] == "serpapi"
    assert results[0]["content"] == "organic evidence"
    assert fake.requests[0][3]["engine"] == "google"


def test_exa_adapter_maps_results(monkeypatch):
    fake = FakeClient(
        post_payload={
            "results": [
                {"title": "Exa result", "url": "https://example.com/exa", "text": "neural search evidence", "score": 0.88}
            ]
        }
    )
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = ExaSearchClient(api_key="key").search("agentic rag")

    assert results[0]["source"] == "exa"
    assert results[0]["content"] == "neural search evidence"
    assert fake.requests[0][2]["query"] == "agentic rag"


def test_perplexity_adapter_maps_answer_and_sources(monkeypatch):
    fake = FakeClient(
        post_payload={
            "choices": [{"message": {"content": "A sourced answer."}}],
            "search_results": [
                {"title": "Perplexity source", "url": "https://example.com/pplx", "snippet": "source evidence"}
            ],
        }
    )
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = PerplexitySearchClient(api_key="key", model="sonar").search("agentic rag")

    assert results[0]["source"] == "perplexity"
    assert results[0]["kind"] == "web-summary"
    assert results[1]["url"] == "https://example.com/pplx"
    assert fake.requests[0][2]["model"] == "sonar"


def test_linkup_adapter_maps_results(monkeypatch):
    fake = FakeClient(
        post_payload={
            "results": [
                {"title": "Linkup result", "url": "https://example.com/linkup", "content": "fresh web evidence"}
            ]
        }
    )
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = LinkupSearchClient(api_key="key", depth="deep").search("agentic rag")

    assert results[0]["source"] == "linkup"
    assert results[0]["metadata"]["depth"] == "deep"
    assert fake.requests[0][2]["outputType"] == "searchResults"


def test_arxiv_adapter_maps_atom_feed(monkeypatch):
    atom = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2501.00001</id>
        <title>Agentic Retrieval</title>
        <summary>Paper evidence for agentic retrieval.</summary>
        <published>2025-01-01T00:00:00Z</published>
        <author><name>Ada Researcher</name></author>
      </entry>
    </feed>
    """
    fake = FakeClient(text=atom)
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = ArxivSearchClient().search("agentic retrieval")

    assert results[0]["source"] == "arxiv"
    assert results[0]["kind"] == "paper"
    assert results[0]["metadata"]["authors"] == ["Ada Researcher"]


def test_pubmed_adapter_maps_esummary(monkeypatch):
    fake = FakeClient(
        get_payload=[
            {"esearchresult": {"idlist": ["123"]}},
            {
                "result": {
                    "uids": ["123"],
                    "123": {
                        "title": "Clinical RAG Study",
                        "fulljournalname": "Journal of Retrieval",
                        "pubdate": "2025",
                    },
                }
            },
        ]
    )
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = PubMedSearchClient().search("clinical rag")

    assert results[0]["source"] == "pubmed"
    assert results[0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert fake.requests[0][3]["db"] == "pubmed"


def test_openai_web_adapter_maps_response_annotations(monkeypatch):
    fake = FakeClient(
        post_payload={
            "output_text": "OpenAI web answer.",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "OpenAI web answer.",
                            "annotations": [{"url": "https://example.com/openai", "title": "OpenAI source"}],
                        }
                    ]
                }
            ],
        }
    )
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = OpenAIWebSearchClient(api_key="key", model="gpt-4.1-mini").search("agentic rag")

    assert results[0]["source"] == "openai_web"
    assert results[1]["url"] == "https://example.com/openai"


def test_anthropic_web_adapter_maps_text(monkeypatch):
    fake = FakeClient(
        post_payload={
            "content": [
                {
                    "type": "text",
                    "text": "Anthropic web answer.",
                    "annotations": [{"url": "https://example.com/anthropic", "title": "Anthropic source"}],
                }
            ]
        }
    )
    monkeypatch.setattr("agentic_research_copilot.search.httpx.Client", lambda **_: fake)

    results = AnthropicWebSearchClient(api_key="key", model="claude-3-5-haiku-latest").search("agentic rag")

    assert results[0]["source"] == "anthropic_web"
    assert results[1]["url"] == "https://example.com/anthropic"
