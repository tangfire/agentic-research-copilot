from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class ChunkLike(Protocol):
    source: str
    chunk_index: int
    contextual_text: str


ScoredChunk = tuple[float, ChunkLike, dict[str, object]]


@dataclass(frozen=True)
class RerankerConfig:
    provider: str = "rule"
    model: str = "qwen3-rerank"
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: float = 15.0
    candidate_limit: int = 24
    allow_fallback: bool = True


class BaseReranker:
    name = "base"

    def rerank(self, query: str, scored_chunks: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
        raise NotImplementedError


class RuleBasedReranker(BaseReranker):
    name = "rule_diversity_chunk_bonus"

    def rerank(self, query: str, scored_chunks: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
        scored_chunks.sort(key=lambda item: (-item[0], item[1].source, item[1].chunk_index))

        reranked: list[ScoredChunk] = []
        source_counts: Counter[str] = Counter()
        for score, chunk, scores in scored_chunks:
            diversity_penalty = min(0.08, source_counts[chunk.source] * 0.025)
            chunk_bonus = 0.03 if chunk.chunk_index == 0 else 0.0
            final_score = max(0.0, min(1.0, score + chunk_bonus - diversity_penalty))
            reranked.append(
                (
                    final_score,
                    chunk,
                    {
                        **scores,
                        "rerank_score": round(final_score, 4),
                        "reranker": self.name,
                        "rerank_provider": "rule",
                    },
                )
            )
            source_counts[chunk.source] += 1

        reranked.sort(key=lambda item: (-item[0], item[1].source, item[1].chunk_index))
        return reranked[:limit]


class DashScopeReranker(BaseReranker):
    """DashScope/Qwen reranker with an optional local rule fallback."""

    name = "dashscope_qwen3_rerank"

    def __init__(
        self,
        config: RerankerConfig,
        fallback: BaseReranker | None = None,
    ) -> None:
        self.config = config
        self.fallback = fallback or RuleBasedReranker()

    def rerank(self, query: str, scored_chunks: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
        if not self.config.api_key or not self.config.base_url:
            return self._fallback(query, scored_chunks, limit, "missing_dashscope_config")

        candidates = scored_chunks[: max(limit, min(len(scored_chunks), self.config.candidate_limit))]
        documents = [self._document_text(chunk) for _, chunk, _ in candidates]
        if not documents:
            return []

        try:
            response = self._post(query=query, documents=documents, top_n=min(limit, len(documents)))
            results = _parse_rerank_results(response)
        except Exception:
            return self._fallback(query, scored_chunks, limit, "dashscope_request_failed")

        if not results:
            return self._fallback(query, scored_chunks, limit, "dashscope_empty_response")

        reranked: list[ScoredChunk] = []
        seen_indexes: set[int] = set()
        for index, relevance_score in results:
            if index < 0 or index >= len(candidates) or index in seen_indexes:
                continue
            seen_indexes.add(index)
            base_score, chunk, scores = candidates[index]
            combined_score = max(0.0, min(1.0, (base_score * 0.35) + (relevance_score * 0.65)))
            reranked.append(
                (
                    combined_score,
                    chunk,
                    {
                        **scores,
                        "rerank_score": round(combined_score, 4),
                        "rerank_relevance_score": round(relevance_score, 4),
                        "reranker": f"dashscope:{self.config.model}",
                        "rerank_provider": "dashscope",
                    },
                )
            )

        if len(reranked) < limit and self.config.allow_fallback:
            fallback_tail = self.fallback.rerank(
                query,
                [item for idx, item in enumerate(candidates) if idx not in seen_indexes],
                limit - len(reranked),
            )
            reranked.extend(
                (score, chunk, {**scores, "rerank_fallback": "dashscope_partial_response"})
                for score, chunk, scores in fallback_tail
            )

        reranked.sort(key=lambda item: (-item[0], item[1].source, item[1].chunk_index))
        return reranked[:limit]

    def _post(self, *, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        url = _rerank_url(self.config.base_url)
        payload: dict[str, Any]
        if "/api/v1/services/rerank/" in url:
            payload = {
                "model": self.config.model,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": top_n, "return_documents": False},
            }
        else:
            payload = {
                "model": self.config.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            }
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def _fallback(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
        limit: int,
        reason: str,
    ) -> list[ScoredChunk]:
        if not self.config.allow_fallback:
            raise RuntimeError(f"DashScope reranker failed in strict provider mode: {reason}")
        return [
            (score, chunk, {**scores, "rerank_fallback": reason})
            for score, chunk, scores in self.fallback.rerank(query, scored_chunks, limit)
        ]

    @staticmethod
    def _document_text(chunk: ChunkLike) -> str:
        return " ".join(chunk.contextual_text.split())[:6000]


def build_reranker(config: RerankerConfig) -> BaseReranker:
    if config.provider.lower() in {"dashscope", "qwen", "qwen3"}:
        return DashScopeReranker(config)
    return RuleBasedReranker()


def _rerank_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/reranks") or base.endswith("/text-rerank"):
        return base
    if base in {
        "https://dashscope.aliyuncs.com/compatible-api/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }:
        return "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    if base.endswith("/compatible-api/v1") or base.endswith("/compatible-mode/v1"):
        return f"{base}/reranks"
    return f"{base}/api/v1/services/rerank/text-rerank/text-rerank"


def _parse_rerank_results(response: dict[str, Any]) -> list[tuple[int, float]]:
    raw_results = (
        response.get("results")
        or response.get("output", {}).get("results")
        or response.get("data", [])
    )
    parsed: list[tuple[int, float]] = []
    for result in raw_results or []:
        if not isinstance(result, dict):
            continue
        raw_index = result.get("index", result.get("document_index", result.get("documentIndex")))
        raw_score = result.get("relevance_score", result.get("score", result.get("relevanceScore")))
        if raw_index is None or raw_score is None:
            continue
        try:
            parsed.append((int(raw_index), max(0.0, min(1.0, float(raw_score)))))
        except (TypeError, ValueError):
            continue
    parsed.sort(key=lambda item: -item[1])
    return parsed
