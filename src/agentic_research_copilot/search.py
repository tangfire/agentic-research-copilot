from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

import httpx

from .settings import AppSettings


SearchTool = Callable[[str], list[dict[str, object]]]

OPEN_DEEP_RESEARCH_STYLE_PROVIDERS = {
    "tavily",
    "exa",
    "perplexity",
    "arxiv",
    "pubmed",
    "linkup",
    "openai_web",
    "anthropic_web",
}
SEARCH_QUERY_STOPWORDS = {
    "about",
    "after",
    "and",
    "approach",
    "are",
    "between",
    "compare",
    "compares",
    "contrast",
    "core",
    "does",
    "establish",
    "for",
    "from",
    "highlighting",
    "how",
    "into",
    "its",
    "model",
    "of",
    "on",
    "or",
    "patterns",
    "the",
    "their",
    "to",
    "trade",
    "what",
    "with",
}
KEYED_SEARCH_PROVIDERS = {
    "tavily",
    "brave",
    "serpapi",
    "exa",
    "perplexity",
    "linkup",
    "openai_web",
    "anthropic_web",
}


class StrictSearchTool:
    def __init__(self, provider: str, search_tool: SearchTool) -> None:
        self.provider = provider
        self.search_tool = search_tool

    def __call__(self, query: str) -> list[dict[str, object]]:
        attempted: list[str] = []
        for candidate in _search_query_variants(query):
            if candidate in attempted:
                continue
            attempted.append(candidate)
            results = self.search_tool(candidate)
            if results:
                if candidate != query:
                    for item in results:
                        metadata = item.get("metadata")
                        if isinstance(metadata, dict):
                            metadata["original_query"] = query
                            metadata["query_rewrite_strategy"] = "strict_provider_compacted"
                return results
        raise RuntimeError(
            f"Strict search provider '{self.provider}' returned no evidence for query: {query}"
        )


class DuckDuckGoSearchClient:
    """Small HTTP search adapter with no API key requirement."""

    endpoint = "https://api.duckduckgo.com/"

    def __init__(self, timeout_seconds: float = 8.0, max_results: int = 5) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

    def search(self, query: str) -> list[dict[str, object]]:
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.get(
                    self.endpoint,
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": 1,
                        "skip_disambig": 1,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        results: list[dict[str, object]] = []
        abstract = _clean_text(payload.get("AbstractText"))
        abstract_url = _clean_text(payload.get("AbstractURL")) or None
        heading = _clean_text(payload.get("Heading")) or query
        if abstract:
            results.append(
                _search_result(
                    title=heading,
                    source="duckduckgo",
                    url=abstract_url,
                    snippet=abstract,
                    content=abstract,
                    score=0.82,
                    query=query,
                )
            )

        for topic in self._flatten_topics(payload.get("RelatedTopics", [])):
            if len(results) >= self.max_results:
                break
            text = _clean_text(topic.get("Text"))
            first_url = _clean_text(topic.get("FirstURL")) or None
            if not text:
                continue
            results.append(
                _search_result(
                    title=text.split(" - ", 1)[0][:120],
                    source="duckduckgo",
                    url=first_url,
                    snippet=text,
                    content=text,
                    score=0.68,
                    query=query,
                )
            )

        return results[: self.max_results]

    def _flatten_topics(self, topics: list[Any]) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            if "Topics" in topic and isinstance(topic["Topics"], list):
                flattened.extend(self._flatten_topics(topic["Topics"]))
            else:
                flattened.append(topic)
        return flattened


class TavilySearchClient:
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        depth: str = "basic",
        base_url: str = "",
        include_raw_content: bool = True,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.depth = "advanced" if depth == "advanced" else "basic"
        self.endpoint = base_url or self.endpoint
        self.include_raw_content = include_raw_content

    def search(self, query: str) -> list[dict[str, object]]:
        if not self.api_key:
            return []
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.post(
                    self.endpoint,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": self.max_results,
                        "search_depth": self.depth,
                        "include_answer": False,
                        "include_raw_content": self.include_raw_content,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        results: list[dict[str, object]] = []
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title")) or query
            content = _clean_text(item.get("content"))
            raw_content = _clean_text(item.get("raw_content"))
            url = _clean_text(item.get("url")) or None
            if not content and not raw_content and not title:
                continue
            results.append(
                _search_result(
                    title=title[:180],
                    source="tavily",
                    url=url,
                    snippet=(content or raw_content)[:900],
                    content=content or raw_content[:1600],
                    score=float(item.get("score", 0.72) or 0.72),
                    query=query,
                    metadata={
                        "depth": self.depth,
                        "raw_content_requested": self.include_raw_content,
                        "raw_content_available": bool(raw_content),
                        "raw_content_chars": len(raw_content),
                    },
                    raw_content=raw_content,
                )
            )
        return results[: self.max_results]


class BraveSearchClient:
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.endpoint = base_url or self.endpoint

    def search(self, query: str) -> list[dict[str, object]]:
        if not self.api_key:
            return []
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.get(
                    self.endpoint,
                    headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
                    params={"q": query, "count": self.max_results},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        web_payload = payload.get("web") if isinstance(payload.get("web"), dict) else {}
        results: list[dict[str, object]] = []
        for item in web_payload.get("results", []):
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title")) or query
            description = _clean_text(item.get("description"))
            url = _clean_text(item.get("url")) or None
            if not description and not title:
                continue
            results.append(
                _search_result(
                    title=title[:180],
                    source="brave",
                    url=url,
                    snippet=description,
                    content=description,
                    score=0.74,
                    query=query,
                )
            )
        return results[: self.max_results]


class SerpAPISearchClient:
    endpoint = "https://serpapi.com/search.json"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.endpoint = base_url or self.endpoint

    def search(self, query: str) -> list[dict[str, object]]:
        if not self.api_key:
            return []
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.get(
                    self.endpoint,
                    params={
                        "engine": "google",
                        "q": query,
                        "api_key": self.api_key,
                        "num": self.max_results,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        results: list[dict[str, object]] = []
        for item in payload.get("organic_results", []):
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title")) or query
            snippet = _clean_text(item.get("snippet"))
            url = _clean_text(item.get("link")) or None
            if not snippet and not title:
                continue
            results.append(
                _search_result(
                    title=title[:180],
                    source="serpapi",
                    url=url,
                    snippet=snippet,
                    content=snippet,
                    score=0.73,
                    query=query,
                )
            )
        return results[: self.max_results]


class ExaSearchClient:
    endpoint = "https://api.exa.ai/search"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.endpoint = base_url or self.endpoint

    def search(self, query: str) -> list[dict[str, object]]:
        if not self.api_key:
            return []
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.post(
                    self.endpoint,
                    headers={"x-api-key": self.api_key, "Accept": "application/json"},
                    json={
                        "query": query,
                        "numResults": self.max_results,
                        "contents": {"text": True, "highlights": True},
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        results: list[dict[str, object]] = []
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title")) or query
            text = _clean_text(item.get("text")) or _join_text(item.get("highlights"))
            url = _clean_text(item.get("url")) or None
            if not text and not title:
                continue
            results.append(
                _search_result(
                    title=title[:180],
                    source="exa",
                    url=url,
                    snippet=text[:900],
                    content=text,
                    score=float(item.get("score", 0.76) or 0.76),
                    query=query,
                )
            )
        return results[: self.max_results]


class PerplexitySearchClient:
    endpoint = "https://api.perplexity.ai/chat/completions"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        model: str = "sonar",
        base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.model = model or "sonar"
        self.endpoint = _join_base(base_url, "chat/completions") if base_url else self.endpoint

    def search(self, query: str) -> list[dict[str, object]]:
        if not self.api_key:
            return []
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Return concise, source-backed research findings.",
                            },
                            {"role": "user", "content": query},
                        ],
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        content = _extract_openai_chat_content(payload)
        results: list[dict[str, object]] = []
        if content:
            results.append(
                _search_result(
                    title=f"Perplexity answer for {query}",
                    source="perplexity",
                    kind="web-summary",
                    url=None,
                    snippet=content[:900],
                    content=content,
                    score=0.78,
                    query=query,
                    metadata={"model": self.model},
                )
            )

        for index, item in enumerate(_perplexity_sources(payload), start=1):
            if len(results) >= self.max_results:
                break
            title = _clean_text(item.get("title")) or f"Perplexity source {index}"
            url = _clean_text(item.get("url")) or None
            snippet = _clean_text(item.get("snippet")) or _clean_text(item.get("text")) or title
            results.append(
                _search_result(
                    title=title[:180],
                    source="perplexity",
                    url=url,
                    snippet=snippet[:900],
                    content=snippet,
                    score=0.72,
                    query=query,
                    metadata={"model": self.model, "source_index": index},
                )
            )
        return results[: self.max_results]


class ArxivSearchClient:
    endpoint = "https://export.arxiv.org/api/query"

    def __init__(self, timeout_seconds: float = 8.0, max_results: int = 5, base_url: str = "") -> None:
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.endpoint = base_url or self.endpoint

    def search(self, query: str) -> list[dict[str, object]]:
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.get(
                    self.endpoint,
                    params={
                        "search_query": f"all:{query}",
                        "start": 0,
                        "max_results": self.max_results,
                    },
                )
                response.raise_for_status()
                payload = response.text
        except Exception:
            return []

        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results: list[dict[str, object]] = []
        for entry in root.findall("atom:entry", ns):
            title = _clean_text(_xml_text(entry, "atom:title", ns)) or query
            summary = _clean_text(_xml_text(entry, "atom:summary", ns))
            url = _clean_text(_xml_text(entry, "atom:id", ns)) or None
            authors = [
                _clean_text(author.findtext("atom:name", default="", namespaces=ns))
                for author in entry.findall("atom:author", ns)
            ]
            results.append(
                _search_result(
                    title=title[:180],
                    source="arxiv",
                    kind="paper",
                    url=url,
                    snippet=summary[:900],
                    content=summary,
                    score=0.77,
                    query=query,
                    metadata={
                        "published": _xml_text(entry, "atom:published", ns),
                        "updated": _xml_text(entry, "atom:updated", ns),
                        "authors": [author for author in authors if author],
                    },
                )
            )
        return results[: self.max_results]


class PubMedSearchClient:
    endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        api_key: str = "",
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.endpoint = (base_url or self.endpoint).rstrip("/")

    def search(self, query: str) -> list[dict[str, object]]:
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                search_params: dict[str, object] = {
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": self.max_results,
                }
                if self.api_key:
                    search_params["api_key"] = self.api_key
                search_response = client.get(f"{self.endpoint}/esearch.fcgi", params=search_params)
                search_response.raise_for_status()
                search_payload = search_response.json()
                ids = search_payload.get("esearchresult", {}).get("idlist", [])
                if not ids:
                    return []

                summary_params: dict[str, object] = {
                    "db": "pubmed",
                    "id": ",".join(ids[: self.max_results]),
                    "retmode": "json",
                }
                if self.api_key:
                    summary_params["api_key"] = self.api_key
                summary_response = client.get(f"{self.endpoint}/esummary.fcgi", params=summary_params)
                summary_response.raise_for_status()
                summary_payload = summary_response.json()
        except Exception:
            return []

        summary = summary_payload.get("result") if isinstance(summary_payload.get("result"), dict) else {}
        uids = summary.get("uids", ids[: self.max_results])
        results: list[dict[str, object]] = []
        for uid in uids[: self.max_results]:
            item = summary.get(str(uid), {})
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title")) or f"PubMed {uid}"
            journal = _clean_text(item.get("fulljournalname")) or _clean_text(item.get("source")) or "pubmed"
            pubdate = _clean_text(item.get("pubdate"))
            snippet = " ".join(part for part in [title, journal, pubdate] if part)
            results.append(
                _search_result(
                    title=title[:180],
                    source="pubmed",
                    kind="paper",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                    snippet=snippet,
                    content=snippet,
                    score=0.75,
                    query=query,
                    metadata={"uid": str(uid), "journal": journal, "pubdate": pubdate},
                )
            )
        return results[: self.max_results]


class LinkupSearchClient:
    endpoint = "https://api.linkup.so/v1/search"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        depth: str = "standard",
        base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.depth = depth or "standard"
        self.endpoint = _join_base(base_url, "search") if base_url else self.endpoint

    def search(self, query: str) -> list[dict[str, object]]:
        if not self.api_key:
            return []
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "q": query,
                        "depth": self.depth,
                        "outputType": "searchResults",
                        "includeImages": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        raw_results = _first_list(payload, "results", "data", "searchResults", "items")
        results: list[dict[str, object]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title")) or query
            snippet = _clean_text(item.get("snippet")) or _clean_text(item.get("content")) or _clean_text(item.get("text"))
            url = _clean_text(item.get("url")) or _clean_text(item.get("link")) or None
            if not snippet and not title:
                continue
            results.append(
                _search_result(
                    title=title[:180],
                    source="linkup",
                    url=url,
                    snippet=snippet[:900],
                    content=snippet,
                    score=float(item.get("score", 0.74) or 0.74),
                    query=query,
                    metadata={"depth": self.depth},
                )
            )
        return results[: self.max_results]


class OpenAIWebSearchClient:
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        model: str = "gpt-4.1-mini",
        base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.model = model or "gpt-4.1-mini"
        self.endpoint = _join_base(base_url, "responses") if base_url else self.endpoint

    def search(self, query: str) -> list[dict[str, object]]:
        if not self.api_key:
            return []
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "tools": [{"type": "web_search_preview"}],
                        "input": f"Find concise, source-backed evidence for: {query}",
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        content = _extract_response_text(payload)
        results = [
            _search_result(
                title=f"OpenAI web search answer for {query}",
                source="openai_web",
                kind="web-summary",
                url=None,
                snippet=content[:900],
                content=content,
                score=0.78,
                query=query,
                metadata={"model": self.model},
            )
        ] if content else []
        results.extend(_annotation_results(payload, provider="openai_web", query=query))
        return results[: self.max_results]


class AnthropicWebSearchClient:
    endpoint = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 8.0,
        max_results: int = 5,
        model: str = "claude-3-5-haiku-latest",
        base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.model = model or "claude-3-5-haiku-latest"
        self.endpoint = _join_base(base_url, "messages") if base_url else self.endpoint

    def search(self, query: str) -> list[dict[str, object]]:
        if not self.api_key:
            return []
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.post(
                    self.endpoint,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 800,
                        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": self.max_results}],
                        "messages": [{"role": "user", "content": f"Find source-backed evidence for: {query}"}],
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        content = _extract_anthropic_text(payload)
        results = [
            _search_result(
                title=f"Anthropic web search answer for {query}",
                source="anthropic_web",
                kind="web-summary",
                url=None,
                snippet=content[:900],
                content=content,
                score=0.78,
                query=query,
                metadata={"model": self.model},
            )
        ] if content else []
        results.extend(_annotation_results(payload, provider="anthropic_web", query=query))
        return results[: self.max_results]


def build_search_tool(settings: AppSettings) -> SearchTool | None:
    common = {
        "timeout_seconds": settings.search_timeout_seconds,
        "max_results": settings.search_max_results,
    }
    provider = settings.search_provider
    tool: SearchTool | None = None
    if provider == "duckduckgo":
        tool = DuckDuckGoSearchClient(**common).search
    elif provider == "tavily":
        tool = TavilySearchClient(
            api_key=settings.search_api_key,
            depth=settings.search_depth,
            base_url=settings.search_base_url,
            include_raw_content=settings.search_include_raw_content,
            **common,
        ).search
    elif provider == "brave":
        tool = BraveSearchClient(
            api_key=settings.search_api_key,
            base_url=settings.search_base_url,
            **common,
        ).search
    elif provider == "serpapi":
        tool = SerpAPISearchClient(
            api_key=settings.search_api_key,
            base_url=settings.search_base_url,
            **common,
        ).search
    elif provider == "exa":
        tool = ExaSearchClient(
            api_key=settings.search_api_key,
            base_url=settings.search_base_url,
            **common,
        ).search
    elif provider == "perplexity":
        tool = PerplexitySearchClient(
            api_key=settings.search_api_key,
            model=settings.search_model or "sonar",
            base_url=settings.search_base_url,
            **common,
        ).search
    elif provider == "arxiv":
        tool = ArxivSearchClient(base_url=settings.search_base_url, **common).search
    elif provider == "pubmed":
        tool = PubMedSearchClient(
            api_key=settings.search_api_key,
            base_url=settings.search_base_url,
            **common,
        ).search
    elif provider == "linkup":
        tool = LinkupSearchClient(
            api_key=settings.search_api_key,
            depth=settings.search_depth,
            base_url=settings.search_base_url,
            **common,
        ).search
    elif provider == "openai_web":
        tool = OpenAIWebSearchClient(
            api_key=settings.search_api_key,
            model=settings.search_model or "gpt-4.1-mini",
            base_url=settings.search_base_url,
            **common,
        ).search
    elif provider == "anthropic_web":
        tool = AnthropicWebSearchClient(
            api_key=settings.search_api_key,
            model=settings.search_model or "claude-3-5-haiku-latest",
            base_url=settings.search_base_url,
            **common,
        ).search
    if tool is not None and getattr(settings, "strict_providers", False):
        return StrictSearchTool(provider, tool)
    return tool


def search_provider_requires_key(provider: str) -> bool:
    return provider in KEYED_SEARCH_PROVIDERS


def _search_query_variants(query: str) -> list[str]:
    cleaned = _clean_text(query)
    variants = [cleaned] if cleaned else []
    first_sentence = re.split(r"[.?!;:]", cleaned, maxsplit=1)[0].strip()
    compacted = _compact_search_query(cleaned)
    sentence_compacted = _compact_search_query(first_sentence)
    for candidate in (compacted, first_sentence, sentence_compacted):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _compact_search_query(query: str, *, max_terms: int = 14) -> str:
    terms = re.findall(r"[a-zA-Z0-9_+.-]+|[\u4e00-\u9fff]+", query.lower())
    selected: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in SEARCH_QUERY_STOPWORDS or len(term) <= 2:
            continue
        if term in seen:
            continue
        seen.add(term)
        selected.append(term)
        if len(selected) >= max_terms:
            break
    return " ".join(selected)


def _search_result(
    *,
    title: str,
    source: str,
    url: str | None,
    snippet: str,
    content: str,
    score: float,
    query: str,
    kind: str = "web",
    metadata: dict[str, object] | None = None,
    raw_content: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "title": title or query,
        "source": source,
        "kind": kind,
        "url": url,
        "snippet": snippet,
        "content": content,
        "score": score,
        "metadata": {"provider": source, "query": query, **(metadata or {})},
    }
    if raw_content:
        result["raw_content"] = raw_content
    return result


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _join_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(_clean_text(item) for item in value if _clean_text(item))
    return _clean_text(value)


def _first_list(payload: object, *keys: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _join_base(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _xml_text(entry: ET.Element, path: str, ns: dict[str, str]) -> str:
    value = entry.findtext(path, default="", namespaces=ns)
    return _clean_text(value)


def _extract_openai_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") if isinstance(payload.get("choices"), list) else []
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return ""
    return _clean_text(message.get("content"))


def _perplexity_sources(payload: dict[str, Any]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for item in _first_list(payload, "search_results", "sources", "results"):
        if isinstance(item, dict):
            sources.append(item)
    citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
    for citation in citations:
        if isinstance(citation, str):
            sources.append({"title": citation, "url": citation, "snippet": citation})
        elif isinstance(citation, dict):
            sources.append(citation)
    return sources


def _extract_response_text(payload: dict[str, Any]) -> str:
    if _clean_text(payload.get("output_text")):
        return _clean_text(payload.get("output_text"))
    texts: list[str] = []
    for output in _first_list(payload, "output"):
        if not isinstance(output, dict):
            continue
        for item in output.get("content", []) if isinstance(output.get("content"), list) else []:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"output_text", "text"}:
                texts.append(_clean_text(item.get("text")))
    return " ".join(text for text in texts if text)


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in _first_list(payload, "content"):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            texts.append(_clean_text(item.get("text")))
    return " ".join(text for text in texts if text)


def _annotation_results(payload: dict[str, Any], *, provider: str, query: str) -> list[dict[str, object]]:
    annotations: list[dict[str, object]] = []
    _collect_annotations(payload, annotations)
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            continue
        url = _clean_text(annotation.get("url")) or _clean_text(annotation.get("uri")) or None
        title = _clean_text(annotation.get("title")) or _clean_text(annotation.get("text")) or f"{provider} source {index}"
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(
            _search_result(
                title=title[:180],
                source=provider,
                url=url,
                snippet=title,
                content=title,
                score=0.7,
                query=query,
                metadata={"source_index": index},
            )
        )
    return results


def _collect_annotations(value: object, out: list[dict[str, object]]) -> None:
    if isinstance(value, dict):
        annotations = value.get("annotations")
        if isinstance(annotations, list):
            out.extend(item for item in annotations if isinstance(item, dict))
        for child in value.values():
            _collect_annotations(child, out)
    elif isinstance(value, list):
        for child in value:
            _collect_annotations(child, out)
