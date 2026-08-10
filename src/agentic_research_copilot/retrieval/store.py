from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..deterministic_provider import DeterministicResearchModelProvider
from ..provider_base import ModelUsage, ResearchModelProvider
from ..schemas import (
    ChunkContextContract,
    CorpusProfile,
    EvidenceItem,
    KnowledgeGraphEntity,
    KnowledgeGraphExtractionContract,
    KnowledgeGraphQueryContract,
    KnowledgeGraphRelationship,
)
from .fulltext import SQLiteBM25Index
from .rerank import BaseReranker, RuleBasedReranker

try:  # pragma: no cover - import availability is environment specific
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - exercised when qdrant-client is unavailable
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 160
DEFAULT_PARENT_CONTEXT_WINDOW = 1
DEFAULT_PARENT_CONTEXT_MAX_CHARS = 2400
DEFAULT_GRAPH_MAX_ENTITIES_PER_CHUNK = 12
DEFAULT_GRAPH_MAX_RELATIONSHIPS_PER_CHUNK = 16
DEFAULT_GRAPH_NEIGHBOR_LIMIT = 8
DEFAULT_GRAPH_ENTITY_CANDIDATE_LIMIT = 8
DEFAULT_GRAPH_RELATION_CANDIDATE_LIMIT = 8
DENSE_VECTOR_NAME = "dense"
CONTEXTUALIZER_PROMPT_VERSION = "contextual-retrieval-v1"
GRAPH_EXTRACTION_PROMPT_VERSION = "knowledge-graph-extraction-v1"
GRAPH_QUERY_PROMPT_VERSION = "graph-query-keywords-v1"


@dataclass(frozen=True)
class GraphEntityRecord:
    key: str
    label: str
    entity_type: str
    description: str
    aliases: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class GraphRelationshipRecord:
    key: str
    source_key: str
    target_key: str
    relation_type: str
    description: str
    keywords: tuple[str, ...]
    weight: float
    confidence: float


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
        parent_context_window: int = DEFAULT_PARENT_CONTEXT_WINDOW,
        parent_context_max_chars: int = DEFAULT_PARENT_CONTEXT_MAX_CHARS,
        graph_enabled: bool = True,
        graph_max_entities_per_chunk: int = DEFAULT_GRAPH_MAX_ENTITIES_PER_CHUNK,
        graph_max_relationships_per_chunk: int = DEFAULT_GRAPH_MAX_RELATIONSHIPS_PER_CHUNK,
        graph_neighbor_limit: int = DEFAULT_GRAPH_NEIGHBOR_LIMIT,
        graph_entity_candidate_limit: int = DEFAULT_GRAPH_ENTITY_CANDIDATE_LIMIT,
        graph_relation_candidate_limit: int = DEFAULT_GRAPH_RELATION_CANDIDATE_LIMIT,
        hybrid_fusion: str = "rrf",
        reranker: BaseReranker | None = None,
        contextualizer_provider: ResearchModelProvider | None = None,
        graph_provider: ResearchModelProvider | None = None,
        allow_local_fallback: bool = True,
    ) -> None:
        self._docs: list[EvidenceItem] = []
        self._chunks: list[DocumentChunk] = []
        self.embedding_provider = embedding_provider or DeterministicResearchModelProvider()
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_context_window = max(0, parent_context_window)
        self.parent_context_max_chars = max(320, parent_context_max_chars)
        self.graph_enabled = graph_enabled
        self.graph_max_entities_per_chunk = max(2, graph_max_entities_per_chunk)
        self.graph_max_relationships_per_chunk = max(1, graph_max_relationships_per_chunk)
        self.graph_neighbor_limit = max(0, graph_neighbor_limit)
        self.graph_entity_candidate_limit = max(1, graph_entity_candidate_limit)
        self.graph_relation_candidate_limit = max(1, graph_relation_candidate_limit)
        self.hybrid_fusion = hybrid_fusion if hybrid_fusion in {"rrf", "dbsf"} else "rrf"
        self.reranker = reranker or RuleBasedReranker()
        self.contextualizer_provider = contextualizer_provider or self.embedding_provider
        self.graph_provider = graph_provider or self.contextualizer_provider
        self._fallback_contextualizer = DeterministicResearchModelProvider(
            embedding_dimensions=getattr(self.embedding_provider, "embedding_dimensions", 256)
        )
        self._fallback_graph_provider = self._fallback_contextualizer
        self.allow_local_fallback = allow_local_fallback
        self._contextualization_cache: dict[str, tuple[ChunkContextContract, ModelUsage]] = {}
        self._graph_extraction_cache: dict[
            str,
            tuple[KnowledgeGraphExtractionContract, ModelUsage, str | None],
        ] = {}
        self._graph_query_cache: dict[
            str,
            tuple[KnowledgeGraphQueryContract, ModelUsage, str | None],
        ] = {}
        self._entity_chunks: dict[str, set[str]] = defaultdict(set)
        self._entity_profiles: dict[str, GraphEntityRecord] = {}
        self._entity_embeddings: dict[str, list[float]] = {}
        self._entity_embedding_texts: dict[str, str] = {}
        self._entity_neighbors: dict[str, Counter[str]] = defaultdict(Counter)
        self._relationship_profiles: dict[str, GraphRelationshipRecord] = {}
        self._relationship_chunks: dict[str, set[str]] = defaultdict(set)
        self._relationship_embeddings: dict[str, list[float]] = {}
        self._relationship_embedding_texts: dict[str, str] = {}
        self._chunk_entities: dict[str, tuple[str, ...]] = {}
        self._chunk_relationships: dict[str, tuple[str, ...]] = {}
        self._keyword_index = SQLiteBM25Index()
        self._keyword_backend = self._keyword_index.backend
        self._client = self._build_client(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_location=qdrant_location,
            qdrant_prefer_local=qdrant_prefer_local,
        )
        self._vector_backend = "qdrant_dense" if self._client is not None else "local_dense"
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
        doc = self._with_document_id(doc)
        self._docs.append(doc)
        self._index_document(doc)
        return doc

    def extend(self, docs: list[EvidenceItem]) -> None:
        known = {_document_identity(doc) for doc in self._docs}
        for doc in docs:
            doc = self._with_document_id(doc)
            identity = _document_identity(doc)
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

        scored_chunks = self._merge_keyword_candidates(query, query_text, query_tokens, scored_chunks, limit)
        scored_chunks = self._merge_graph_candidates(query_text, query_tokens, query_embedding, scored_chunks)
        reranked = self._rerank(query_text, scored_chunks, limit)
        evidence: list[EvidenceItem] = []
        for score, chunk, scores in reranked:
            parent_context = self._parent_context_for_child(chunk)
            evidence.append(
                EvidenceItem(
                    title=f"{chunk.title} #chunk-{chunk.chunk_index + 1}",
                    source=chunk.source,
                    kind="document-chunk",
                    url=chunk.url,
                    snippet=_trim(chunk.text, 320),
                    content=parent_context,
                    score=round(score, 4),
                    metadata={
                        **chunk.metadata,
                        **scores,
                        "document_id": chunk.document_id,
                        "parent_id": chunk.document_id,
                        "parent_title": chunk.title,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "child_text_chars": len(chunk.text),
                        "parent_context_chars": len(parent_context),
                        "parent_context_window": self.parent_context_window,
                        "matched_query": query,
                        "grounding_query": query_text,
                        "query_context": context,
                        "query_purpose": purpose,
                        "context_used": bool(context),
                        "purpose": purpose,
                        "retrieval_strategy": "light_rag_inspired_parent_child_dense_bm25_graph_rerank",
                        "parent_child_retrieval": True,
                        "child_retrieval": "dense_bm25_fusion_rerank",
                        "parent_expansion": "same-document_neighbor_window",
                        "graph_augmented_retrieval": self.graph_enabled,
                        "graph_strategy": "structured_entity_relation_dual_level",
                        "retrieval_backend": f"{self._vector_backend}_embedding_hybrid",
                        "hybrid_fusion": self.hybrid_fusion,
                        "keyword_backend": self._keyword_backend,
                        "embedding_dimensions": len(chunk.embedding),
                    },
                )
            )
        return evidence

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
            keyword_backend=self._keyword_backend,
            embedding_dimensions=getattr(self.embedding_provider, "embedding_dimensions", 0),
            collection_name=self.collection_name,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    def clear(self) -> None:
        self._docs.clear()
        self._chunks.clear()
        self._contextualization_cache.clear()
        self._graph_extraction_cache.clear()
        self._graph_query_cache.clear()
        self._keyword_index.clear()
        self._clear_graph_index(clear_embedding_cache=True)
        if self._client is not None and qmodels is not None:
            try:
                if self._collection_exists():
                    self._client.delete_collection(self.collection_name)
            except Exception:
                pass
        self._collection_ready = False
        self._ensure_collection()

    def delete(self, document_id: str) -> bool:
        kept_docs = [doc for doc in self._docs if _document_identity(doc) != document_id]
        if len(kept_docs) == len(self._docs):
            return False
        self._docs = kept_docs
        self._chunks = [chunk for chunk in self._chunks if chunk.document_id != document_id]
        self._contextualization_cache = {
            key: value for key, value in self._contextualization_cache.items() if not key.startswith(f"{document_id}:")
        }
        self._graph_extraction_cache = {
            key: value
            for key, value in self._graph_extraction_cache.items()
            if not key.startswith(f"{document_id}:")
        }
        self._keyword_index.delete_document(document_id)
        self._rebuild_graph_index()
        self._delete_qdrant_document(document_id)
        return True

    def close(self) -> None:
        self._keyword_index.close()
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

        doc = self._with_document_id(doc)
        document_id = _document_identity(doc)
        chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
        total_chunks = len(chunks)
        for index, chunk_text in enumerate(chunks):
            chunk_context, context_usage = self._contextualize_chunk(
                doc=doc,
                document_text=text,
                chunk_text=chunk_text,
                chunk_index=index,
                total_chunks=total_chunks,
            )
            graph_contract, graph_usage, graph_fallback = self._extract_knowledge_graph(
                doc=doc,
                document_text=text,
                chunk_text=chunk_text,
                chunk_index=index,
                total_chunks=total_chunks,
            )
            chunk_metadata = {
                **doc.metadata,
                "contextual_retrieval": True,
                "contextualizer_prompt_version": CONTEXTUALIZER_PROMPT_VERSION,
                "context_prefix": chunk_context.context,
                "context_key_terms": list(chunk_context.key_terms),
                "context_provenance_hint": chunk_context.provenance_hint,
                "context_confidence": round(float(chunk_context.confidence), 4),
                "contextualizer_provider": getattr(context_usage, "provider", "unknown"),
                "contextualizer_model": getattr(context_usage, "model", "unknown"),
                "graph_extraction": self.graph_enabled,
                "graph_extraction_prompt_version": GRAPH_EXTRACTION_PROMPT_VERSION,
                "graph_extractor_provider": getattr(graph_usage, "provider", "disabled"),
                "graph_extractor_model": getattr(graph_usage, "model", "disabled"),
                "graph_extraction_fallback": graph_fallback or "",
                "graph_entities": [entity.model_dump() for entity in graph_contract.entities],
                "graph_relationships": [
                    relationship.model_dump()
                    for relationship in graph_contract.relationships
                ],
            }
            contextual_text = _build_contextual_text(
                doc,
                chunk_text,
                index,
                total_chunks,
                context_prefix=chunk_context.context,
                context_key_terms=chunk_context.key_terms,
            )
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
                metadata=chunk_metadata,
                tokens=tuple(_tokenize(contextual_text)),
                embedding=embedding,
            )
            self._chunks.append(chunk)
            self._keyword_index.add_chunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                source=chunk.source,
                text=chunk.contextual_text,
                tokens=chunk.tokens,
            )
            self._index_graph_chunk(chunk)
            self._upsert_chunk(chunk)

    def _contextualize_chunk(
        self,
        *,
        doc: EvidenceItem,
        document_text: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
    ):
        document_id = _document_identity(doc)
        chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
        provider_name = getattr(self.contextualizer_provider, "name", "unknown")
        cache_key = f"{document_id}:{chunk_index}:{chunk_hash}:{provider_name}:{CONTEXTUALIZER_PROMPT_VERSION}"
        cached = self._contextualization_cache.get(cache_key)
        if cached is not None:
            return cached

        document_excerpt = _document_context_excerpt(document_text, chunk_text)
        try:
            result = self.contextualizer_provider.contextualize_chunk(
                document_title=doc.title,
                source=doc.source,
                metadata=doc.metadata,
                document_excerpt=document_excerpt,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
        except Exception as exc:
            if not self.allow_local_fallback:
                raise RuntimeError(f"Chunk contextualization failed: {exc}") from exc
            result = self._fallback_contextualizer.contextualize_chunk(
                document_title=doc.title,
                source=doc.source,
                metadata=doc.metadata,
                document_excerpt=document_excerpt,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
        self._contextualization_cache[cache_key] = result
        return result

    def _extract_knowledge_graph(
        self,
        *,
        doc: EvidenceItem,
        document_text: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> tuple[KnowledgeGraphExtractionContract, ModelUsage, str | None]:
        if not self.graph_enabled:
            return (
                KnowledgeGraphExtractionContract(),
                ModelUsage(provider="disabled", model="graph-disabled"),
                None,
            )

        document_id = _document_identity(doc)
        chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
        provider_name = getattr(self.graph_provider, "name", "unknown")
        cache_key = f"{document_id}:{chunk_index}:{chunk_hash}:{provider_name}:{GRAPH_EXTRACTION_PROMPT_VERSION}"
        cached = self._graph_extraction_cache.get(cache_key)
        if cached is not None:
            return cached

        document_excerpt = _document_context_excerpt(document_text, chunk_text)
        fallback_reason: str | None = None
        try:
            contract, usage = self.graph_provider.extract_knowledge_graph(
                document_title=doc.title,
                source=doc.source,
                metadata=doc.metadata,
                document_excerpt=document_excerpt,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                max_entities=self.graph_max_entities_per_chunk,
                max_relationships=self.graph_max_relationships_per_chunk,
            )
        except Exception as exc:
            if not self.allow_local_fallback:
                raise RuntimeError(f"Knowledge graph extraction failed: {exc}") from exc
            fallback_reason = exc.__class__.__name__
            contract, usage = self._fallback_graph_provider.extract_knowledge_graph(
                document_title=doc.title,
                source=doc.source,
                metadata=doc.metadata,
                document_excerpt=document_excerpt,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                max_entities=self.graph_max_entities_per_chunk,
                max_relationships=self.graph_max_relationships_per_chunk,
            )

        result = (
            _normalize_graph_contract(
                contract,
                max_entities=self.graph_max_entities_per_chunk,
                max_relationships=self.graph_max_relationships_per_chunk,
            ),
            usage,
            fallback_reason,
        )
        self._graph_extraction_cache[cache_key] = result
        return result

    def _with_document_id(self, doc: EvidenceItem) -> EvidenceItem:
        metadata = dict(doc.metadata)
        document_id = _document_identity(doc)
        metadata["document_id"] = document_id
        if metadata == doc.metadata:
            return doc
        return doc.model_copy(update={"metadata": metadata})

    def _delete_qdrant_document(self, document_id: str) -> None:
        if self._client is None or qmodels is None:
            return
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            if not self.allow_local_fallback:
                raise RuntimeError(f"Qdrant document delete failed: {exc}") from exc
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        docs = list(self._docs)
        self._chunks.clear()
        self._graph_query_cache.clear()
        self._keyword_index.clear()
        self._clear_graph_index()
        if self._client is not None and qmodels is not None:
            try:
                if self._collection_exists():
                    self._client.delete_collection(self.collection_name)
            except Exception:
                pass
        self._collection_ready = False
        self._ensure_collection()
        for doc in docs:
            self._index_document(doc)

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
                query=query_embedding,
                using=DENSE_VECTOR_NAME,
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
            semantic_score = max(0.0, min(1.0, float(hit.score or 0.0)))
            coverage_score = _coverage_score(query_tokens, chunk.tokens)
            phrase_score = _phrase_score(query, chunk.contextual_text)
            raw_score = (
                semantic_score * 0.68
                + coverage_score * 0.24
                + phrase_score * 0.08
            )
            scored_chunks.append(
                (
                    raw_score,
                    chunk,
                    {
                        "semantic_score": round(semantic_score, 4),
                        "coverage_score": round(coverage_score, 4),
                        "phrase_score": round(phrase_score, 4),
                        "matched_terms": _matched_terms(query_tokens, chunk.tokens)[:12],
                        "qdrant_score": round(semantic_score, 4),
                        "fusion_score": round(semantic_score, 4),
                        "fusion_algorithm": "dense_prefetch",
                        "retrieval_stage": "qdrant_dense_prefetch",
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
            semantic_score = _cosine_similarity(query_embedding, chunk.embedding)
            coverage_score = _coverage_score(query_tokens, chunk.tokens)
            phrase_score = _phrase_score(query, chunk.contextual_text)
            raw_score = (
                semantic_score * 0.68
                + coverage_score * 0.22
                + phrase_score * 0.10
            )
            if raw_score <= 0:
                continue
            scored_chunks.append(
                (
                    raw_score,
                    chunk,
                    {
                        "semantic_score": round(semantic_score, 4),
                        "coverage_score": round(coverage_score, 4),
                        "phrase_score": round(phrase_score, 4),
                        "matched_terms": _matched_terms(query_tokens, chunk.tokens)[:12],
                        "fusion_algorithm": "local_dense_prefetch",
                        "retrieval_stage": "local_dense_prefetch",
                    },
                )
            )
        return scored_chunks

    def _merge_keyword_candidates(
        self,
        query: str,
        query_text: str,
        query_tokens: tuple[str, ...],
        scored_chunks: list[tuple[float, DocumentChunk, dict[str, object]]],
        limit: int,
    ) -> list[tuple[float, DocumentChunk, dict[str, object]]]:
        keyword_hits = self._keyword_index.search(query_tokens, limit=max(limit * 6, 24))
        if not keyword_hits:
            return scored_chunks

        chunk_lookup = {chunk.chunk_id: chunk for chunk in self._chunks}
        dense_ranks = {
            chunk.chunk_id: rank
            for rank, (_score, chunk, _scores) in enumerate(
                sorted(scored_chunks, key=lambda item: (-item[0], item[1].source, item[1].chunk_index)),
                start=1,
            )
        }
        merged: dict[str, tuple[float, DocumentChunk, dict[str, object]]] = {
            chunk.chunk_id: (score, chunk, dict(scores))
            for score, chunk, scores in scored_chunks
        }

        for chunk_id, keyword_hit in keyword_hits.items():
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            bm25_score = keyword_hit.score
            dense_score = merged.get(chunk_id, (0.0, chunk, {}))[0]
            fused_score = self._fuse_dense_keyword_score(
                dense_score=dense_score,
                bm25_score=bm25_score,
                dense_rank=dense_ranks.get(chunk_id),
                keyword_rank=keyword_hit.rank,
            )
            keyword_meta = {
                "bm25_score": round(bm25_score, 4),
                "bm25_rank": keyword_hit.rank,
                "bm25_raw_rank": keyword_hit.raw_rank,
                "keyword_backend": keyword_hit.backend,
                "keyword_augmented": True,
                "matched_terms": _matched_terms(query_tokens, chunk.tokens)[:12],
                "fusion_algorithm": f"{self.hybrid_fusion}_dense_bm25",
                "fusion_score": round(fused_score, 4),
                "retrieval_stage": f"{self._vector_backend}_bm25_fusion",
                "keyword_query": query_text,
                "matched_query": query,
            }
            if chunk_id in merged:
                _score, existing_chunk, scores = merged[chunk_id]
                merged[chunk_id] = (fused_score, existing_chunk, {**scores, **keyword_meta})
                continue
            merged[chunk_id] = (fused_score, chunk, keyword_meta)

        return sorted(merged.values(), key=lambda item: (-item[0], item[1].source, item[1].chunk_index))

    def _fuse_dense_keyword_score(
        self,
        *,
        dense_score: float,
        bm25_score: float,
        dense_rank: int | None,
        keyword_rank: int,
    ) -> float:
        if self.hybrid_fusion == "rrf":
            dense_rrf = _rrf_score(dense_rank) if dense_rank is not None else 0.0
            keyword_rrf = _rrf_score(keyword_rank)
            max_rrf = 2 * _rrf_score(1)
            rank_score = (dense_rrf + keyword_rrf) / max_rrf
            lexical_score = bm25_score * 0.72 + rank_score * 0.28
            return max(0.0, min(1.0, max(dense_score, lexical_score)))
        return max(0.0, min(1.0, dense_score * 0.55 + bm25_score * 0.45))

    def _rerank(
        self,
        query: str,
        scored_chunks: list[tuple[float, DocumentChunk, dict[str, object]]],
        limit: int,
    ) -> list[tuple[float, DocumentChunk, dict[str, object]]]:
        return self.reranker.rerank(query, scored_chunks, limit)

    def _index_graph_chunk(self, chunk: DocumentChunk) -> None:
        if not self.graph_enabled:
            return
        contract = _graph_contract_from_metadata(chunk.metadata)
        entities = [
            entity
            for entity in contract.entities
            if _graph_key(entity.name)
        ][: self.graph_max_entities_per_chunk]
        if not entities:
            return

        entity_keys: list[str] = []
        for entity in entities:
            key = _graph_key(entity.name)
            if not key or key in entity_keys:
                continue
            entity_keys.append(key)
            self._entity_chunks[key].add(chunk.chunk_id)
            self._upsert_graph_entity_profile(key, entity)

        relationships = [
            relationship
            for relationship in contract.relationships
            if _graph_key(relationship.source) in entity_keys
            and _graph_key(relationship.target) in entity_keys
        ][: self.graph_max_relationships_per_chunk]

        relationship_keys: list[str] = []
        for relationship in relationships:
            source_key = _graph_key(relationship.source)
            target_key = _graph_key(relationship.target)
            if not source_key or not target_key or source_key == target_key:
                continue
            relationship_key = _graph_relationship_key(relationship)
            if relationship_key in relationship_keys:
                continue
            relationship_keys.append(relationship_key)
            self._relationship_chunks[relationship_key].add(chunk.chunk_id)
            self._upsert_graph_relationship_profile(relationship_key, relationship)
            edge_weight = max(
                1,
                round(
                    float(relationship.weight or 1.0)
                    * max(0.1, relationship.confidence or 0.5)
                ),
            )
            self._entity_neighbors[source_key][target_key] += edge_weight
            self._entity_neighbors[target_key][source_key] += edge_weight

        if not relationship_keys and len(entity_keys) > 1:
            for left_index, left_key in enumerate(entity_keys):
                for right_key in entity_keys[left_index + 1 :]:
                    if left_key == right_key:
                        continue
                    self._entity_neighbors[left_key][right_key] += 1
                    self._entity_neighbors[right_key][left_key] += 1

        self._chunk_entities[chunk.chunk_id] = tuple(entity_keys)
        self._chunk_relationships[chunk.chunk_id] = tuple(relationship_keys)

    def _upsert_graph_entity_profile(self, key: str, entity: KnowledgeGraphEntity) -> None:
        current = self._entity_profiles.get(key)
        confidence = max(0.0, min(1.0, float(entity.confidence or 0.0)))
        if current is None or confidence >= current.confidence:
            aliases = tuple(
                alias
                for alias in (_clean_entity_label(value) for value in entity.aliases)
                if alias
            )[:8]
            record = GraphEntityRecord(
                key=key,
                label=_clean_entity_label(entity.name),
                entity_type=_clean_entity_label(entity.entity_type) or "concept",
                description=_trim(entity.description, 600),
                aliases=aliases,
                confidence=confidence,
            )
            self._entity_profiles[key] = record
            embedding_text = _graph_entity_embedding_text(record)
            if self._entity_embedding_texts.get(key) != embedding_text:
                self._entity_embeddings[key] = self._embed(embedding_text)
                self._entity_embedding_texts[key] = embedding_text

    def _upsert_graph_relationship_profile(
        self,
        key: str,
        relationship: KnowledgeGraphRelationship,
    ) -> None:
        current = self._relationship_profiles.get(key)
        confidence = max(0.0, min(1.0, float(relationship.confidence or 0.0)))
        if current is None or confidence >= current.confidence:
            keywords = tuple(
                keyword
                for keyword in (
                    _clean_entity_label(value)
                    for value in relationship.keywords
                )
                if keyword
            )[:10]
            record = GraphRelationshipRecord(
                key=key,
                source_key=_graph_key(relationship.source),
                target_key=_graph_key(relationship.target),
                relation_type=_clean_entity_label(relationship.relation_type)
                or "related_to",
                description=_trim(relationship.description, 600),
                keywords=keywords,
                weight=max(0.05, min(3.0, float(relationship.weight or 1.0))),
                confidence=confidence,
            )
            self._relationship_profiles[key] = record
            embedding_text = _graph_relationship_embedding_text(
                record,
                self._entity_profiles,
            )
            if self._relationship_embedding_texts.get(key) != embedding_text:
                self._relationship_embeddings[key] = self._embed(embedding_text)
                self._relationship_embedding_texts[key] = embedding_text

    def _clear_graph_index(self, *, clear_embedding_cache: bool = False) -> None:
        self._entity_chunks.clear()
        self._entity_profiles.clear()
        self._entity_neighbors.clear()
        self._relationship_profiles.clear()
        self._relationship_chunks.clear()
        self._chunk_entities.clear()
        self._chunk_relationships.clear()
        if clear_embedding_cache:
            self._entity_embeddings.clear()
            self._entity_embedding_texts.clear()
            self._relationship_embeddings.clear()
            self._relationship_embedding_texts.clear()

    def _rebuild_graph_index(self) -> None:
        self._clear_graph_index()
        for chunk in self._chunks:
            self._index_graph_chunk(chunk)

    def _merge_graph_candidates(
        self,
        query_text: str,
        query_tokens: tuple[str, ...],
        query_embedding: list[float],
        scored_chunks: list[tuple[float, DocumentChunk, dict[str, object]]],
    ) -> list[tuple[float, DocumentChunk, dict[str, object]]]:
        if not self.graph_enabled or not self._chunk_entities:
            return scored_chunks

        graph_scores = self._search_graph(query_text, query_tokens, query_embedding)
        if not graph_scores:
            return scored_chunks

        chunk_lookup = {chunk.chunk_id: chunk for chunk in self._chunks}
        merged: dict[str, tuple[float, DocumentChunk, dict[str, object]]] = {
            chunk.chunk_id: (score, chunk, dict(scores))
            for score, chunk, scores in scored_chunks
        }
        for chunk_id, graph_meta in graph_scores.items():
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            graph_score = float(graph_meta["graph_score"])
            if chunk_id in merged:
                score, existing_chunk, scores = merged[chunk_id]
                scores.update(graph_meta)
                scores["graph_augmented"] = True
                merged[chunk_id] = (min(1.0, score + graph_score * 0.18), existing_chunk, scores)
                continue
            scores = {
                **graph_meta,
                "graph_augmented": True,
                "retrieval_stage": "graph_entity_relation_expansion",
                "fusion_algorithm": "graph_plus_dense_bm25",
            }
            merged[chunk_id] = (min(1.0, graph_score * 0.72), chunk, scores)
        return sorted(merged.values(), key=lambda item: (-item[0], item[1].source, item[1].chunk_index))

    def _search_graph(
        self,
        query_text: str,
        query_tokens: tuple[str, ...],
        query_embedding: list[float],
    ) -> dict[str, dict[str, object]]:
        query_contract, query_usage, query_fallback = self._extract_graph_query(
            query_text,
            query_tokens,
        )
        local_keywords = list(query_contract.local_keywords)
        global_keywords = list(query_contract.global_keywords)
        local_query_text = " ".join(local_keywords) or query_text
        global_query_text = " ".join(global_keywords) or local_query_text
        local_embedding = (
            query_embedding
            if local_query_text == query_text
            else self._embed(local_query_text)
        )
        global_embedding = (
            query_embedding
            if global_query_text == query_text
            else self._embed(global_query_text)
        )

        query_keys: list[str] = []
        for keyword in local_keywords:
            key = _graph_key(keyword)
            if key in self._entity_chunks and key not in query_keys:
                query_keys.append(key)
        if not query_keys:
            for token in dict.fromkeys(query_tokens):
                key = _graph_key(token)
                if key in self._entity_chunks and key not in query_keys:
                    query_keys.append(key)
                if len(query_keys) >= self.graph_max_entities_per_chunk:
                    break

        chunk_scores: dict[str, float] = defaultdict(float)
        matched_entities: dict[str, set[str]] = defaultdict(set)
        expanded_entities: dict[str, set[str]] = defaultdict(set)
        matched_relationships: dict[str, set[str]] = defaultdict(set)

        for key in query_keys:
            profile = self._entity_profiles.get(key)
            label = profile.label if profile is not None else key
            for chunk_id in self._entity_chunks.get(key, set()):
                chunk_scores[chunk_id] += 1.0
                matched_entities[chunk_id].add(label)
            for neighbor_key, weight in self._entity_neighbors.get(key, Counter()).most_common(self.graph_neighbor_limit):
                neighbor_profile = self._entity_profiles.get(neighbor_key)
                neighbor_label = (
                    neighbor_profile.label
                    if neighbor_profile is not None
                    else neighbor_key
                )
                relation_score = min(0.45, 0.18 + 0.05 * weight)
                for chunk_id in self._entity_chunks.get(neighbor_key, set()):
                    chunk_scores[chunk_id] += relation_score
                    expanded_entities[chunk_id].add(neighbor_label)

        for key, score in self._semantic_graph_matches(
            local_embedding,
            self._entity_embeddings,
            self.graph_entity_candidate_limit,
        ):
            profile = self._entity_profiles.get(key)
            if profile is None:
                continue
            semantic_entity_score = min(0.82, score * 0.72)
            for chunk_id in self._entity_chunks.get(key, set()):
                chunk_scores[chunk_id] += semantic_entity_score
                matched_entities[chunk_id].add(profile.label)

        for relationship_key, score in self._semantic_graph_matches(
            global_embedding,
            self._relationship_embeddings,
            self.graph_relation_candidate_limit,
        ):
            relationship = self._relationship_profiles.get(relationship_key)
            if relationship is None:
                continue
            relationship_score = min(
                0.9,
                score * 0.65
                + relationship.weight * relationship.confidence * 0.18,
            )
            label = _relationship_label(relationship, self._entity_profiles)
            for chunk_id in self._relationship_chunks.get(
                relationship_key,
                set(),
            ):
                chunk_scores[chunk_id] += relationship_score
                matched_relationships[chunk_id].add(label)

        if not chunk_scores:
            return {}
        max_score = max(chunk_scores.values()) or 1.0
        return {
            chunk_id: {
                "graph_score": round(score / max_score, 4),
                "graph_matched_entities": sorted(matched_entities.get(chunk_id, set()))[:8],
                "graph_expanded_entities": sorted(expanded_entities.get(chunk_id, set()))[:8],
                "graph_matched_relationships": sorted(
                    matched_relationships.get(chunk_id, set())
                )[:8],
                "graph_query_entities": [
                    self._entity_profiles[key].label
                    for key in query_keys
                    if key in self._entity_profiles
                ],
                "graph_query_local_keywords": local_keywords[:8],
                "graph_query_global_keywords": global_keywords[:8],
                "graph_query_prompt_version": GRAPH_QUERY_PROMPT_VERSION,
                "graph_query_provider": getattr(query_usage, "provider", "unknown"),
                "graph_query_model": getattr(query_usage, "model", "unknown"),
                "graph_query_fallback": query_fallback or "",
                "graph_query_confidence": round(float(query_contract.confidence or 0.0), 4),
                "graph_neighbor_limit": self.graph_neighbor_limit,
            }
            for chunk_id, score in chunk_scores.items()
        }

    def _extract_graph_query(
        self,
        query_text: str,
        query_tokens: tuple[str, ...],
    ) -> tuple[KnowledgeGraphQueryContract, ModelUsage, str | None]:
        provider_name = getattr(self.graph_provider, "name", "unknown")
        query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{query_hash}:{provider_name}:{GRAPH_QUERY_PROMPT_VERSION}"
        cached = self._graph_query_cache.get(cache_key)
        if cached is not None:
            return cached

        fallback_reason: str | None = None
        try:
            contract, usage = self.graph_provider.extract_graph_query(
                query=query_text,
                max_local_keywords=self.graph_max_entities_per_chunk,
                max_global_keywords=self.graph_relation_candidate_limit,
            )
        except Exception as exc:
            if not self.allow_local_fallback:
                raise RuntimeError(f"Graph query extraction failed: {exc}") from exc
            fallback_reason = exc.__class__.__name__
            contract, usage = self._fallback_graph_provider.extract_graph_query(
                query=query_text,
                max_local_keywords=self.graph_max_entities_per_chunk,
                max_global_keywords=self.graph_relation_candidate_limit,
            )

        normalized = _normalize_graph_query_contract(
            contract,
            query_tokens=query_tokens,
            max_local_keywords=self.graph_max_entities_per_chunk,
            max_global_keywords=self.graph_relation_candidate_limit,
        )
        result = (normalized, usage, fallback_reason)
        self._graph_query_cache[cache_key] = result
        return result

    @staticmethod
    def _semantic_graph_matches(
        query_embedding: list[float],
        candidate_embeddings: dict[str, list[float]],
        limit: int,
    ) -> list[tuple[str, float]]:
        matches = [
            (key, _cosine_similarity(query_embedding, embedding))
            for key, embedding in candidate_embeddings.items()
        ]
        return [
            (key, score)
            for key, score in sorted(matches, key=lambda item: -item[1])[:limit]
            if score >= 0.18
        ]

    def _parent_context_for_child(self, child: DocumentChunk) -> str:
        siblings = [
            chunk
            for chunk in self._chunks
            if chunk.document_id == child.document_id
            and abs(chunk.chunk_index - child.chunk_index) <= self.parent_context_window
        ]
        siblings.sort(key=lambda chunk: chunk.chunk_index)
        if not siblings:
            return child.contextual_text

        context_lines = [
            f"Parent document: {child.title}",
            f"Source: {child.source}",
            f"Matched child chunk: {child.chunk_index + 1}/{child.total_chunks}",
            f"Parent context window: +/-{self.parent_context_window} child chunk(s)",
        ]
        if child.url:
            context_lines.append(f"URL: {child.url}")
        metadata_bits = [
            f"{key}: {value}"
            for key, value in sorted(child.metadata.items())
            if isinstance(value, (str, int, float, bool))
        ][:8]
        if metadata_bits:
            context_lines.append("Metadata: " + "; ".join(metadata_bits))
        context_lines.append("Parent context:")
        for sibling in siblings:
            marker = "matched child" if sibling.chunk_id == child.chunk_id else "neighbor child"
            context_lines.append(f"[{marker} {sibling.chunk_index + 1}/{sibling.total_chunks}]")
            context_lines.append(sibling.text)
        return _trim("\n".join(context_lines), self.parent_context_max_chars)

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
                        vector={DENSE_VECTOR_NAME: chunk.embedding},
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
    sentences = SENTENCE_SPLIT_PATTERN.split(text)
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
    *,
    context_prefix: str,
    context_key_terms: list[str] | tuple[str, ...] = (),
) -> str:
    metadata_bits = [
        f"{key}: {value}"
        for key, value in sorted(doc.metadata.items())
        if isinstance(value, (str, int, float, bool))
    ][:6]
    context_lines = [
        f"Retrieval context: {context_prefix}",
        f"Document title: {doc.title}",
        f"Source: {doc.source}",
        f"Chunk: {chunk_index + 1}/{total_chunks}",
    ]
    if context_key_terms:
        context_lines.append("Context key terms: " + ", ".join(context_key_terms[:12]))
    if doc.url:
        context_lines.append(f"URL: {doc.url}")
    if metadata_bits:
        context_lines.append("Metadata: " + "; ".join(metadata_bits))
    context_lines.append("Excerpt:")
    context_lines.append(chunk_text)
    return "\n".join(context_lines)


def _document_context_excerpt(document_text: str, chunk_text: str, max_chars: int = 12000) -> str:
    document_text = document_text.strip()
    if len(document_text) <= max_chars:
        return document_text
    chunk_start = document_text.find(chunk_text[: min(80, len(chunk_text))])
    if chunk_start < 0:
        return document_text[:max_chars]
    half = max_chars // 2
    start = max(0, chunk_start - half)
    end = min(len(document_text), chunk_start + len(chunk_text) + half)
    excerpt = document_text[start:end]
    prefix = "[document excerpt continues] " if start > 0 else ""
    suffix = " [document excerpt continues]" if end < len(document_text) else ""
    return f"{prefix}{excerpt}{suffix}"


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _matched_terms(query_tokens: tuple[str, ...], chunk_tokens: tuple[str, ...]) -> list[str]:
    if not query_tokens or not chunk_tokens:
        return []
    chunk_token_set = set(chunk_tokens)
    return [token for token in dict.fromkeys(query_tokens) if token in chunk_token_set]


def _rrf_score(rank: int) -> float:
    return 1.0 / (60.0 + max(1, rank))


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


def _clean_entity_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" -_/.,:;()[]{}")).strip()


def _graph_key(value: str) -> str:
    cleaned = _clean_entity_label(value).casefold()
    cleaned = re.sub(r"[^\w\u4e00-\u9fff+.#/-]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _graph_relationship_key(relationship: KnowledgeGraphRelationship) -> str:
    return "|".join(
        [
            _graph_key(relationship.source),
            _graph_key(relationship.target),
            _graph_key(relationship.relation_type),
        ]
    )


def _graph_contract_from_metadata(
    metadata: dict[str, object],
) -> KnowledgeGraphExtractionContract:
    try:
        entities = [
            KnowledgeGraphEntity.model_validate(value)
            for value in metadata.get("graph_entities", [])
            if isinstance(value, dict)
        ]
        relationships = [
            KnowledgeGraphRelationship.model_validate(value)
            for value in metadata.get("graph_relationships", [])
            if isinstance(value, dict)
        ]
    except Exception:
        return KnowledgeGraphExtractionContract()
    return KnowledgeGraphExtractionContract(
        entities=entities,
        relationships=relationships,
    )


def _normalize_graph_contract(
    contract: KnowledgeGraphExtractionContract,
    *,
    max_entities: int,
    max_relationships: int,
) -> KnowledgeGraphExtractionContract:
    entities: list[KnowledgeGraphEntity] = []
    known: set[str] = set()
    for entity in contract.entities:
        name = _trim(entity.name, 160)
        key = _graph_key(name)
        if not key or key in known:
            continue
        known.add(key)
        aliases = [
            alias
            for alias in dict.fromkeys(
                _trim(value, 120) for value in entity.aliases
            )
            if alias and _graph_key(alias) != key
        ][:8]
        entities.append(
            entity.model_copy(
                update={
                    "name": name,
                    "entity_type": _trim(entity.entity_type, 80) or "concept",
                    "description": _trim(entity.description, 600),
                    "aliases": aliases,
                    "confidence": max(
                        0.0,
                        min(1.0, float(entity.confidence or 0.0)),
                    ),
                }
            )
        )
        if len(entities) >= max(1, max_entities):
            break

    canonical_names = {_graph_key(entity.name): entity.name for entity in entities}
    relationships: list[KnowledgeGraphRelationship] = []
    relation_keys: set[str] = set()
    for relationship in contract.relationships:
        source = canonical_names.get(_graph_key(relationship.source))
        target = canonical_names.get(_graph_key(relationship.target))
        if not source or not target or source == target:
            continue
        relation_type = _trim(relationship.relation_type, 100) or "related_to"
        normalized = relationship.model_copy(
            update={
                "source": source,
                "target": target,
                "relation_type": relation_type,
                "description": _trim(relationship.description, 600),
                "keywords": [
                    keyword
                    for keyword in dict.fromkeys(
                        _trim(value, 100) for value in relationship.keywords
                    )
                    if keyword
                ][:10],
                "weight": max(
                    0.05,
                    min(3.0, float(relationship.weight or 1.0)),
                ),
                "confidence": max(
                    0.0,
                    min(1.0, float(relationship.confidence or 0.0)),
                ),
            }
        )
        relation_key = _graph_relationship_key(normalized)
        if relation_key in relation_keys:
            continue
        relation_keys.add(relation_key)
        relationships.append(normalized)
        if len(relationships) >= max(1, max_relationships):
            break

    return KnowledgeGraphExtractionContract(
        entities=entities,
        relationships=relationships,
        summary=_trim(contract.summary, 600),
        confidence=max(0.0, min(1.0, float(contract.confidence or 0.0))),
    )


def _normalize_graph_query_contract(
    contract: KnowledgeGraphQueryContract,
    *,
    query_tokens: tuple[str, ...],
    max_local_keywords: int,
    max_global_keywords: int,
) -> KnowledgeGraphQueryContract:
    local_keywords = [
        keyword
        for keyword in dict.fromkeys(
            _trim(value, 120) for value in contract.local_keywords
        )
        if keyword
    ][: max(1, max_local_keywords)]
    global_keywords = [
        keyword
        for keyword in dict.fromkeys(
            _trim(value, 120) for value in contract.global_keywords
        )
        if keyword
    ][: max(1, max_global_keywords)]
    if not local_keywords:
        local_keywords = list(dict.fromkeys(query_tokens))[: max(1, max_local_keywords)]
    if not global_keywords:
        global_keywords = local_keywords[: max(1, max_global_keywords)]
    return contract.model_copy(
        update={
            "local_keywords": local_keywords,
            "global_keywords": global_keywords,
            "confidence": max(0.0, min(1.0, float(contract.confidence or 0.0))),
        }
    )


def _graph_entity_embedding_text(record: GraphEntityRecord) -> str:
    aliases = ", ".join(record.aliases)
    return "\n".join(
        [
            f"Entity: {record.label}",
            f"Type: {record.entity_type}",
            f"Aliases: {aliases}",
            f"Description: {record.description}",
        ]
    )


def _graph_relationship_embedding_text(
    record: GraphRelationshipRecord,
    profiles: dict[str, GraphEntityRecord],
) -> str:
    source = profiles.get(record.source_key)
    target = profiles.get(record.target_key)
    source_label = source.label if source is not None else record.source_key
    target_label = target.label if target is not None else record.target_key
    return "\n".join(
        [
            f"Source: {source_label}",
            f"Target: {target_label}",
            f"Relation type: {record.relation_type}",
            f"Keywords: {', '.join(record.keywords)}",
            f"Description: {record.description}",
        ]
    )


def _relationship_label(
    relationship: GraphRelationshipRecord,
    profiles: dict[str, GraphEntityRecord],
) -> str:
    source = profiles.get(relationship.source_key)
    target = profiles.get(relationship.target_key)
    source_label = source.label if source is not None else relationship.source_key
    target_label = target.label if target is not None else relationship.target_key
    return f"{source_label} -[{relationship.relation_type}]-> {target_label}"


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        size = min(len(left), len(right))
        left = left[:size]
        right = right[:size]
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _document_identity(document: EvidenceItem) -> str:
    metadata_id = document.metadata.get("document_id")
    if isinstance(metadata_id, str) and metadata_id:
        return metadata_id
    stable = document.url or f"{document.source}:{document.title}:{document.snippet or document.content or ''}"
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _qdrant_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _trim(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
