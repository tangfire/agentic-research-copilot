from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal

from .provider_base import ResearchModelProvider


SourceReaderStrategy = Literal["extract", "model_compress", "chunk_rerank_compress"]


@dataclass
class SourceReadResult:
    content: str
    snippet: str
    metadata: dict[str, object]


class SourceReader:
    """Turn provider-returned raw content into compact, citation-ready evidence."""

    def __init__(
        self,
        *,
        strategy: SourceReaderStrategy = "extract",
        model_provider: ResearchModelProvider | None = None,
        embedding_provider: ResearchModelProvider | None = None,
        raw_content_max_chars: int = 50000,
        excerpt_max_chars: int = 1600,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        top_chunks: int = 5,
        semantic_candidate_chunks: int = 12,
        chunk_context_window: int = 1,
    ) -> None:
        self.strategy = strategy
        self.model_provider = model_provider
        self.embedding_provider = embedding_provider
        self.raw_content_max_chars = max(1000, raw_content_max_chars)
        self.excerpt_max_chars = max(400, excerpt_max_chars)
        self.chunk_size = max(400, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.top_chunks = max(1, top_chunks)
        self.semantic_candidate_chunks = max(self.top_chunks, semantic_candidate_chunks)
        self.chunk_context_window = max(0, chunk_context_window)

    def read(
        self,
        *,
        query: str,
        title: str,
        url: str | None,
        raw_content: str,
        fallback: str = "",
    ) -> SourceReadResult | None:
        raw_content = _clean_text(raw_content)
        if not raw_content:
            return None
        bounded = raw_content[: self.raw_content_max_chars]
        if self.strategy == "chunk_rerank_compress":
            result = self._chunk_rerank_compress(
                query=query,
                title=title,
                url=url,
                raw_content=bounded,
                raw_content_chars=len(raw_content),
            )
            if result is not None:
                return result
        if self.strategy == "model_compress" and self.model_provider is not None:
            result = self._model_compress(
                query=query,
                title=title,
                url=url,
                raw_content=bounded,
                raw_content_chars=len(raw_content),
            )
            if result is not None:
                return result
        return self._extract(
            query=query,
            raw_content=bounded,
            fallback=fallback,
            raw_content_chars=len(raw_content),
        )

    def _chunk_rerank_compress(
        self,
        *,
        query: str,
        title: str,
        url: str | None,
        raw_content: str,
        raw_content_chars: int,
    ) -> SourceReadResult | None:
        chunks = split_text(raw_content, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        if not chunks:
            return None
        ranked_chunks, rerank_method = self._rank_chunks(query, chunks)
        selected_chunks = ranked_chunks[: self.top_chunks]
        expanded_chunks = expand_neighbor_chunks(
            chunks,
            selected_chunks,
            window=self.chunk_context_window,
        )
        stitched = stitch_chunks(expanded_chunks)
        if self.model_provider is not None:
            result = self._model_compress(
                query=query,
                title=title,
                url=url,
                raw_content=stitched,
                raw_content_chars=raw_content_chars,
            )
            if result is not None:
                result.metadata = {
                    **result.metadata,
                    "read_strategy": "provider_raw_content_chunk_rerank_model_compress",
                    "chunk_count": len(chunks),
                    "selected_chunk_count": len(selected_chunks),
                    "expanded_chunk_count": len(expanded_chunks),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunk_context_window": self.chunk_context_window,
                    "chunk_expansion": "selected_chunk_neighbor_window",
                    "chunk_rerank_method": rerank_method,
                    "raw_content_used_chars": len(raw_content),
                    "reranked_content_chars": len(stitched),
                }
                return result
        fallback_excerpt = _trim(stitched, self.excerpt_max_chars)
        if not fallback_excerpt:
            return None
        return SourceReadResult(
            content=fallback_excerpt,
            snippet=_trim(fallback_excerpt, min(900, self.excerpt_max_chars)),
            metadata={
                "read_strategy": "provider_raw_content_chunk_rerank_extract",
                "raw_content_chars": raw_content_chars,
                "raw_content_used_chars": len(raw_content),
                "chunk_count": len(chunks),
                "selected_chunk_count": len(selected_chunks),
                "expanded_chunk_count": len(expanded_chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunk_context_window": self.chunk_context_window,
                "chunk_expansion": "selected_chunk_neighbor_window",
                "chunk_rerank_method": rerank_method,
                "reranked_content_chars": len(stitched),
                "excerpt_chars": len(fallback_excerpt),
            },
        )

    def _rank_chunks(self, query: str, chunks: list["SourceChunk"]) -> tuple[list["SourceChunk"], str]:
        lexical_ranked = rank_chunks_lexical(query, chunks)
        if self.embedding_provider is None:
            return lexical_ranked, "lexical"
        semantic_candidates = lexical_ranked[: self.semantic_candidate_chunks]
        try:
            query_vector, _ = self.embedding_provider.embed_text(query)
            chunk_vectors, _ = self.embedding_provider.embed_texts([chunk.text for chunk in semantic_candidates])
        except Exception:
            return lexical_ranked, "lexical_fallback"
        query_terms = _query_terms(query)
        lexical_scores = {
            chunk.index: _lexical_chunk_score(chunk.text, query_terms)
            for chunk in semantic_candidates
        }
        max_lexical_score = max(lexical_scores.values(), default=0.0) or 1.0
        semantic_ranked = sorted(
            zip(semantic_candidates, chunk_vectors, strict=False),
            key=lambda item: (
                lexical_scores[item[0].index] > 0,
                min(1.0, lexical_scores[item[0].index] / max_lexical_score) * 0.85
                + max(0.0, _cosine_similarity(query_vector, item[1])) * 0.15,
                -item[0].index,
            ),
            reverse=True,
        )
        ranked = [chunk for chunk, _ in semantic_ranked]
        used_indexes = {chunk.index for chunk in ranked}
        ranked.extend(chunk for chunk in lexical_ranked if chunk.index not in used_indexes)
        return ranked, "blended_lexical_embedding"

    def _model_compress(
        self,
        *,
        query: str,
        title: str,
        url: str | None,
        raw_content: str,
        raw_content_chars: int,
    ) -> SourceReadResult | None:
        try:
            contract, usage = self.model_provider.compress_source(  # type: ignore[union-attr]
                query=query,
                title=title,
                url=url,
                raw_content=raw_content,
            )
        except Exception:
            return None
        content = _format_compression(contract.summary, contract.key_excerpts, contract.limitations)
        if not content:
            return None
        metadata = {
            "read_strategy": "provider_raw_content_model_compress",
            "source_reader_model": usage.model,
            "source_reader_provider": usage.provider,
            "source_reader_tokens_in": usage.prompt_tokens,
            "source_reader_tokens_out": usage.completion_tokens,
            "source_reader_latency_ms": usage.latency_ms,
            "source_relevance": contract.relevance,
            "source_limitations": contract.limitations,
            "raw_content_chars": raw_content_chars,
            "raw_content_used_chars": len(raw_content),
            "compressed_chars": len(content),
            "key_excerpt_count": len(contract.key_excerpts),
        }
        return SourceReadResult(
            content=_trim(content, self.excerpt_max_chars),
            snippet=_trim(content, min(900, self.excerpt_max_chars)),
            metadata=metadata,
        )

    def _extract(
        self,
        *,
        query: str,
        raw_content: str,
        fallback: str,
        raw_content_chars: int,
    ) -> SourceReadResult | None:
        excerpt = extract_relevant_excerpt(
            raw_content,
            query,
            fallback=fallback,
            max_chars=self.excerpt_max_chars,
        )
        if not excerpt:
            return None
        return SourceReadResult(
            content=excerpt,
            snippet=_trim(excerpt, min(900, self.excerpt_max_chars)),
            metadata={
                "read_strategy": "provider_raw_content_extract",
                "raw_content_chars": raw_content_chars,
                "raw_content_used_chars": len(raw_content),
                "excerpt_chars": len(excerpt),
            },
        )


def source_reader_strategy_label(strategy: str) -> str:
    if strategy == "chunk_rerank_compress":
        return "provider_raw_content_chunk_rerank_model_compress"
    if strategy == "model_compress":
        return "provider_raw_content_model_compress"
    return "provider_raw_content_extract"


@dataclass(frozen=True)
class SourceChunk:
    index: int
    text: str
    start: int


def split_text(text: str, *, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[SourceChunk]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [SourceChunk(index=0, text=cleaned, start=0)]
    chunks: list[SourceChunk] = []
    start = 0
    index = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk_text = cleaned[start:end].strip()
        if chunk_text:
            chunks.append(SourceChunk(index=index, text=chunk_text, start=start))
            index += 1
        if end >= len(cleaned):
            break
        start += step
    return chunks


def rank_chunks_lexical(query: str, chunks: list[SourceChunk]) -> list[SourceChunk]:
    query_terms = _query_terms(query)
    if not query_terms:
        return chunks
    return sorted(
        chunks,
        key=lambda chunk: (_lexical_chunk_score(chunk.text, query_terms), -chunk.index),
        reverse=True,
    )


def expand_neighbor_chunks(
    chunks: list[SourceChunk],
    selected_chunks: list[SourceChunk],
    *,
    window: int = 1,
) -> list[SourceChunk]:
    if not selected_chunks or window <= 0:
        return sorted(selected_chunks, key=lambda chunk: chunk.index)
    chunk_by_index = {chunk.index: chunk for chunk in chunks}
    expanded_indexes: set[int] = set()
    for chunk in selected_chunks:
        for index in range(chunk.index - window, chunk.index + window + 1):
            if index in chunk_by_index:
                expanded_indexes.add(index)
    return [chunk_by_index[index] for index in sorted(expanded_indexes)]


def stitch_chunks(chunks: list[SourceChunk]) -> str:
    ordered = sorted(chunks, key=lambda chunk: chunk.index)
    seen: set[str] = set()
    parts: list[str] = []
    for chunk in ordered:
        fingerprint = chunk.text
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        parts.append(f"...{chunk.text}...")
    return "\n\n".join(parts)


def extract_relevant_excerpt(raw_content: str, query: str, *, fallback: str = "", max_chars: int = 1600) -> str:
    cleaned = _clean_text(raw_content)
    if not cleaned:
        return _trim(fallback, max_chars)
    query_terms = _query_terms(query)
    if not query_terms:
        return _trim(cleaned, max_chars)

    sentences = _sentences(cleaned)
    if not sentences:
        return _trim(cleaned, max_chars)

    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        score = sum(1 for term in query_terms if term in lower)
        if score:
            scored.append((score, -index, sentence))
    if not scored:
        return _trim(cleaned, max_chars)

    selected = [sentence for _, _, sentence in sorted(scored, reverse=True)[:4]]
    return _trim(" ".join(selected), max_chars)


def _format_compression(summary: str, key_excerpts: list[str], limitations: list[str]) -> str:
    parts: list[str] = []
    summary = _clean_text(summary)
    if summary:
        parts.append(summary)
    cleaned_excerpts = [_clean_text(excerpt) for excerpt in key_excerpts if _clean_text(excerpt)]
    if cleaned_excerpts:
        parts.append("Key excerpts: " + " ".join(f"- {excerpt}" for excerpt in cleaned_excerpts[:5]))
    cleaned_limitations = [_clean_text(item) for item in limitations if _clean_text(item)]
    if cleaned_limitations:
        parts.append("Limitations: " + "; ".join(cleaned_limitations[:3]))
    return "\n\n".join(parts)


def _query_terms(query: str) -> set[str]:
    stopwords = {
        "about",
        "and",
        "are",
        "between",
        "for",
        "from",
        "how",
        "into",
        "the",
        "this",
        "what",
        "with",
    }
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", query)
        if token.lower() not in stopwords
    }


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _lexical_chunk_score(text: str, query_terms: set[str]) -> float:
    lower = text.lower()
    score = 0.0
    for term in query_terms:
        count = lower.count(term)
        if count:
            score += 1.0 + min(3, count) * 0.35
    return score + min(0.5, len(text) / 5000)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _trim(text: str, max_chars: int) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
