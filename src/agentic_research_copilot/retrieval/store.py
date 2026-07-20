from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..providers import DeterministicResearchModelProvider, ResearchModelProvider
from ..schemas import CorpusProfile, EvidenceItem
from .rerank import BaseReranker, RuleBasedReranker

try:  # pragma: no cover - import availability is environment specific
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - exercised when qdrant-client is unavailable
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 160
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    chunk_id: str
    title: str
    source: str
    url: str | None
    text: str
    contextual_text: str
    chunk_index: int
    total_chunks: int
    metadata: dict[str, object]
    tokens: tuple[str, ...]
    embedding: list[float]


class DocumentStore:
    """Contextual retrieval over uploaded documents with a Qdrant-first backend."""

    def __init__(
        self,
        embedding_provider: ResearchModelProvider | None = None,
        *,
        collection_name: str = "arc_documents",
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        qdrant_location: str = ":memory:",
        qdrant_prefer_local: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        hybrid_fusion: str = "rrf",
        reranker: BaseReranker | None = None,
        allow_local_fallback: bool = True,
    ) -> None:
        self._docs: list[EvidenceItem] = []
        self._chunks: list[DocumentChunk] = []
        self.embedding_provider = embedding_provider or DeterministicResearchModelProvider()
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hybrid_fusion = hybrid_fusion if hybrid_fusion in {"rrf", "dbsf"} else "rrf"
        self.reranker = reranker or RuleBasedReranker()
        self.allow_local_fallback = allow_local_fallback
        self._client = self._build_client(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_location=qdrant_location,
            qdrant_prefer_local=qdrant_prefer_local,
        )
        self._vector_backend = "qdrant_dense_sparse" if self._client is not None else "local"
        self._collection_ready = False
        self._ensure_collection()

    def add(
        self,
        title: str,
        source: str,
        url: str | None = None,
        snippet: str | None = None,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> EvidenceItem:
        doc = EvidenceItem(
            title=title,
            source=source,
            kind="document",
            url=url,
            snippet=snippet,
            content=content,
            score=1.0,
            metadata=metadata or {},
        )
        self._docs.append(doc)
        self._index_document(doc)
        return doc

    def extend(self, docs: list[EvidenceItem]) -> None:
        known = {
            doc.url or f"{doc.source}:{doc.title}:{doc.snippet or doc.content or ''}"
            for doc in self._docs
        }
        for doc in docs:
            identity = doc.url or f"{doc.source}:{doc.title}:{doc.snippet or doc.content or ''}"
            if identity not in known:
                self._docs.append(doc)
                self._index_document(doc)
                known.add(identity)

    def search(
        self,
        query: str,
        limit: int = 5,
        context: str | None = None,
        purpose: str | None = None,
    ) -> list[EvidenceItem]:
        query_text = " ".join(part for part in [query, context or "", purpose or ""] if part)
        query_tokens = tuple(_tokenize(query_text))
        if not query_tokens:
            return []

        query_embedding = self._embed(query_text)
        scored_chunks: list[tuple[float, DocumentChunk, dict[str, object]]] = []

        if self._client is not None and qmodels is not None:
            scored_chunks.extend(
                self._search_qdrant(query, query_text, query_tokens, query_embedding, limit)
            )
        else:
            scored_chunks.extend(self._search_local(query, query_text, query_tokens, query_embedding))

        reranked = self._rerank(query_text, scored_chunks, limit)
        return [
            EvidenceItem(
                title=f"{chunk.title} #chunk-{chunk.chunk_index + 1}",
                source=chunk.source,
                kind="document-chunk",
                url=chunk.url,
                snippet=_trim(chunk.text, 320),
                content=chunk.contextual_text,
                score=round(score, 4),
                metadata={
                    **chunk.metadata,
                    **scores,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "matched_query": query,
                    "grounding_query": query_text,
                    "query_context": context,
                    "query_purpose": purpose,
                    "context_used": bool(context),
                    "purpose": purpose,
                    "retrieval_strategy": "contextual_dense_sparse_fusion_rerank",
                    "retrieval_backend": f"{self._vector_backend}_embedding_hybrid",
                    "hybrid_fusion": self.hybrid_fusion,
                    "embedding_dimensions": len(chunk.embedding),
                },
            )
            for score, chunk, scores in reranked
        ]

    def list(self) -> list[EvidenceItem]:
        return sorted(self._docs, key=lambda doc: (doc.source, doc.title))

    def profile(self) -> CorpusProfile:
        source_names = sorted({doc.source for doc in self._docs if doc.source})
        kind_counts: Counter[str] = Counter(doc.kind for doc in self._docs if doc.kind)
        keyword_counts: Counter[str] = Counter()
        for doc in self._docs:
            keyword_counts.update(_tokenize(" ".join([doc.title, doc.snippet or "", doc.content or ""])))

        keyword_signals = [
            token
            for token, count in keyword_counts.most_common(16)
            if len(token) > 2 and count > 1
        ]
        has_reference_docs = any(
            doc.source.startswith(("README", "docs/", "doc/", "notes/"))
            or doc.metadata.get("kind") in {"project_overview", "architecture", "source_map"}
            for doc in self._docs
        )
        return CorpusProfile(
            document_count=len(self._docs),
            source_count=len(source_names),
            source_names=source_names,
            document_kinds=dict(kind_counts),
            keyword_signals=keyword_signals,
            has_private_docs=bool(self._docs),
            has_reference_docs=has_reference_docs,
            vector_backend=self._vector_backend,
            embedding_dimensions=getattr(self.embedding_provider, "embedding_dimensions", 0),
            collection_name=self.collection_name,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    def clear(self) -> None:
        self._docs.clear()
        self._chunks.clear()
        if self._client is not None and qmodels is not None:
            try:
                if self._collection_exists():
                    self._client.delete_collection(self.collection_name)
            except Exception:
                pass
        self._collection_ready = False
        self._ensure_collection()

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.close()
        except Exception:
            pass

    def _index_document(self, doc: EvidenceItem) -> None:
        text = "\n\n".join(
            part for part in [doc.snippet or "", doc.content or ""] if part.strip()
        ).strip()
        if not text:
            return

        document_id = _stable_id(doc.url or f"{doc.source}:{doc.title}:{text[:120]}")
        chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
        total_chunks = len(chunks)
        for index, chunk_text in enumerate(chunks):
            contextual_text = _build_contextual_text(doc, chunk_text, index, total_chunks)
            embedding = self._embed(contextual_text)
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_id=f"{document_id}:{index}",
                title=doc.title,
                source=doc.source,
                url=doc.url,
                text=chunk_text,
                contextual_text=contextual_text,
                chunk_index=index,
                total_chunks=total_chunks,
                metadata=doc.metadata,
                tokens=tuple(_tokenize(contextual_text)),
                embedding=embedding,
            )
            self._chunks.append(chunk)
            self._upsert_chunk(chunk)

    def _search_qdrant(
        self,
        query: str,
        query_text: str,
        query_tokens: tuple[str, ...],
        query_embedding: list[float],
        limit: int,
    ) -> list[tuple[float, DocumentChunk, dict[str, object]]]:
        self._ensure_collection()
        if self._client is None or qmodels is None:
            if not self.allow_local_fallback:
                raise RuntimeError("Qdrant hybrid query failed because the Qdrant client is unavailable.")
            return []
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    qmodels.Prefetch(
                        query=query_embedding,
                        using=DENSE_VECTOR_NAME,
                        limit=max(limit * 4, 12),
                    ),
                    qmodels.Prefetch(
                        query=_sparse_vector(query_tokens),
                        using=SPARSE_VECTOR_NAME,
                        limit=max(limit * 4, 12),
                    ),
                ],
                query=qmodels.FusionQuery(
                    fusion=qmodels.Fusion.RRF if self.hybrid_fusion == "rrf" else qmodels.Fusion.DBSF
                ),
                limit=max(limit * 4, 12),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            if not self.allow_local_fallback:
                raise RuntimeError(f"Qdrant hybrid query failed: {exc}") from exc
            return self._search_local(query, query_text, query_tokens, query_embedding)

        scored_chunks: list[tuple[float, DocumentChunk, dict[str, object]]] = []
        for hit in response.points:
            payload = hit.payload or {}
            chunk = self._chunk_from_payload(payload)
            if chunk is None:
                continue
            sparse_score, matched_terms = _sparse_score(query_tokens, chunk.tokens)
            semantic_score = max(0.0, min(1.0, float(hit.score or 0.0)))
            coverage_score = _coverage_score(query_tokens, chunk.tokens)
            phrase_score = _phrase_score(query, chunk.contextual_text)
            raw_score = (
                sparse_score * 0.34
                + semantic_score * 0.36
                + coverage_score * 0.22
                + phrase_score * 0.08
            )
            scored_chunks.append(
                (
                    raw_score,
                    chunk,
                    {
                        "sparse_score": round(sparse_score, 4),
                        "semantic_score": round(semantic_score, 4),
                        "coverage_score": round(coverage_score, 4),
                        "phrase_score": round(phrase_score, 4),
                        "matched_terms": matched_terms[:12],
                        "qdrant_score": round(semantic_score, 4),
                        "fusion_score": round(semantic_score, 4),
                        "fusion_algorithm": self.hybrid_fusion,
                        "retrieval_stage": "qdrant_dense_sparse_fusion",
                    },
                )
            )
        return scored_chunks

    def _search_local(
        self,
        query: str,
        query_text: str,
        query_tokens: tuple[str, ...],
        query_embedding: list[float],
    ) -> list[tuple[float, DocumentChunk, dict[str, object]]]:
        scored_chunks: list[tuple[float, DocumentChunk, dict[str, object]]] = []
        for chunk in self._chunks:
            sparse_score, matched_terms = _sparse_score(query_tokens, chunk.tokens)
            semantic_score = _cosine_similarity(query_embedding, chunk.embedding)
            coverage_score = _coverage_score(query_tokens, chunk.tokens)
            phrase_score = _phrase_score(query, chunk.contextual_text)
            raw_score = (
                sparse_score * 0.38
                + semantic_score * 0.32
                + coverage_score * 0.22
                + phrase_score * 0.08
            )
            if raw_score <= 0:
                continue
            scored_chunks.append(
                (
                    raw_score,
                    chunk,
                    {
                        "sparse_score": round(sparse_score, 4),
                        "semantic_score": round(semantic_score, 4),
                        "coverage_score": round(coverage_score, 4),
                        "phrase_score": round(phrase_score, 4),
                        "matched_terms": matched_terms[:12],
                        "fusion_algorithm": "local_weighted",
                        "retrieval_stage": "local_dense_sparse_fusion",
                    },
                )
            )
        return scored_chunks

    def _rerank(
        self,
        query: str,
        scored_chunks: list[tuple[float, DocumentChunk, dict[str, object]]],
        limit: int,
    ) -> list[tuple[float, DocumentChunk, dict[str, object]]]:
        return self.reranker.rerank(query, scored_chunks, limit)

    def _upsert_chunk(self, chunk: DocumentChunk) -> None:
        if self._client is None or qmodels is None:
            return
        self._ensure_collection()
        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=_qdrant_point_id(chunk.chunk_id),
                        vector={
                            DENSE_VECTOR_NAME: chunk.embedding,
                            SPARSE_VECTOR_NAME: _sparse_vector(chunk.tokens),
                        },
                        payload=_chunk_payload(chunk),
                    )
                ],
            )
        except Exception as exc:
            if not self.allow_local_fallback:
                raise RuntimeError(f"Qdrant chunk upsert failed: {exc}") from exc
            self._client = None
            self._vector_backend = "local"

    def _chunk_from_payload(self, payload: dict[str, Any]) -> DocumentChunk | None:
        try:
            return DocumentChunk(
                document_id=str(payload["document_id"]),
                chunk_id=str(payload["chunk_id"]),
                title=str(payload["title"]),
                source=str(payload["source"]),
                url=payload.get("url") or None,
                text=str(payload["text"]),
                contextual_text=str(payload["contextual_text"]),
                chunk_index=int(payload["chunk_index"]),
                total_chunks=int(payload["total_chunks"]),
                metadata=dict(payload.get("metadata") or {}),
                tokens=tuple(payload.get("tokens") or _tokenize(str(payload["contextual_text"]))),
                embedding=[float(value) for value in payload.get("embedding") or self._embed(str(payload["contextual_text"]))],
            )
        except Exception:
            return None

    def _ensure_collection(self) -> None:
        if self._collection_ready or self._client is None or qmodels is None:
            if self._client is not None and qmodels is None and not self.allow_local_fallback:
                raise RuntimeError("Qdrant models are unavailable; install qdrant-client.")
            return
        try:
            if not self._collection_exists():
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: qmodels.VectorParams(
                            size=getattr(self.embedding_provider, "embedding_dimensions", 256),
                            distance=qmodels.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
                )
            self._collection_ready = True
        except Exception as exc:
            if not self.allow_local_fallback:
                raise RuntimeError(f"Qdrant collection initialization failed: {exc}") from exc
            self._client = None
            self._vector_backend = "local"

    def _collection_exists(self) -> bool:
        if self._client is None or qmodels is None:
            return False
        try:
            return self._client.collection_exists(self.collection_name)
        except Exception:
            return False

    def _build_client(
        self,
        *,
        qdrant_url: str,
        qdrant_api_key: str,
        qdrant_location: str,
        qdrant_prefer_local: bool,
    ) -> QdrantClient | None:
        if QdrantClient is None:
            if not self.allow_local_fallback:
                raise RuntimeError("qdrant-client is required in strict provider mode.")
            return None
        try:
            if qdrant_url:
                return QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
            if qdrant_prefer_local:
                if qdrant_location and qdrant_location != ":memory:":
                    return QdrantClient(path=qdrant_location)
                return QdrantClient(location=":memory:")
        except Exception as exc:
            if not self.allow_local_fallback:
                raise RuntimeError(f"Qdrant client initialization failed: {exc}") from exc
            return None
        if not self.allow_local_fallback:
            raise RuntimeError("Qdrant strict mode requires ARC_QDRANT_URL or a Qdrant location.")
        return None

    def _embed(self, text: str) -> list[float]:
        vector, _usage = self.embedding_provider.embed_text(text)
        return vector


def _chunk_payload(chunk: DocumentChunk) -> dict[str, object]:
    return {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "source": chunk.source,
        "url": chunk.url,
        "text": chunk.text,
        "contextual_text": chunk.contextual_text,
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "metadata": chunk.metadata,
        "tokens": list(chunk.tokens),
        "embedding": chunk.embedding,
    }


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs or [normalized]:
        paragraph_parts = _split_long_text(paragraph, chunk_size)
        for part in paragraph_parts:
            if not current:
                current = part
                continue
            if len(current) + len(part) + 2 <= chunk_size:
                current = f"{current}\n\n{part}"
                continue
            chunks.append(current)
            prefix = current[-overlap:] if overlap > 0 else ""
            current = f"{prefix}\n\n{part}".strip() if prefix else part

    if current:
        chunks.append(current)
    return chunks


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current:
                parts.append(current)
                current = ""
            parts.extend(sentence[i : i + chunk_size] for i in range(0, len(sentence), chunk_size))
            continue
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}"
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts


def _build_contextual_text(
    doc: EvidenceItem,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    metadata_bits = [
        f"{key}: {value}"
        for key, value in sorted(doc.metadata.items())
        if isinstance(value, (str, int, float, bool))
    ][:6]
    context_lines = [
        f"Document title: {doc.title}",
        f"Source: {doc.source}",
        f"Chunk: {chunk_index + 1}/{total_chunks}",
    ]
    if doc.url:
        context_lines.append(f"URL: {doc.url}")
    if metadata_bits:
        context_lines.append("Metadata: " + "; ".join(metadata_bits))
    context_lines.append("Excerpt:")
    context_lines.append(chunk_text)
    return "\n".join(context_lines)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _sparse_score(query_tokens: tuple[str, ...], chunk_tokens: tuple[str, ...]) -> tuple[float, list[str]]:
    if not query_tokens or not chunk_tokens:
        return 0.0, []
    chunk_counts = Counter(chunk_tokens)
    unique_query = list(dict.fromkeys(query_tokens))
    matched = [token for token in unique_query if token in chunk_counts]
    if not matched:
        return 0.0, []

    score = 0.0
    doc_len = len(chunk_tokens)
    avg_len = 180
    k1 = 1.2
    b = 0.75
    for token in matched:
        tf = chunk_counts[token]
        score += ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len)))
    return min(1.0, score / max(1, len(unique_query))), matched


def _sparse_vector(tokens: tuple[str, ...]):
    if qmodels is None:
        return None
    counts = Counter(tokens)
    weighted: dict[int, float] = defaultdict(float)
    token_count = max(1, len(tokens))
    for token, count in counts.items():
        index = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
        weighted[index] += count / token_count
    sorted_items = sorted(weighted.items())
    return qmodels.SparseVector(
        indices=[index for index, _value in sorted_items],
        values=[value for _index, value in sorted_items],
    )


def _coverage_score(query_tokens: tuple[str, ...], chunk_tokens: tuple[str, ...]) -> float:
    unique_query = set(query_tokens)
    if not unique_query:
        return 0.0
    chunk_token_set = set(chunk_tokens)
    return len(unique_query & chunk_token_set) / len(unique_query)


def _phrase_score(query: str, text: str) -> float:
    query = re.sub(r"\s+", " ", query.strip().lower())
    text = re.sub(r"\s+", " ", text.strip().lower())
    if not query or len(query) < 12:
        return 0.0
    return 1.0 if query in text else 0.0


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        size = min(len(left), len(right))
        left = left[:size]
        right = right[:size]
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _qdrant_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _trim(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
