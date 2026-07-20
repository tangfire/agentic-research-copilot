import pytest

from agentic_research_copilot.retrieval import (
    BaseReranker,
    DashScopeReranker,
    DocumentStore,
    RerankerConfig,
)


class PreferSecondReranker(BaseReranker):
    name = "prefer_second"

    def rerank(self, query, scored_chunks, limit):
        reordered = sorted(scored_chunks, key=lambda item: 0 if item[1].chunk_index == 1 else 1)
        return [
            (score, chunk, {**scores, "reranker": self.name, "rerank_score": score})
            for score, chunk, scores in reordered[:limit]
        ]


def test_document_store_returns_contextual_chunks():
    store = DocumentStore(chunk_size=120, chunk_overlap=20)
    store.add(
        title="Research Notes",
        source="notes.md",
        content=(
            "Contextual retrieval matters for research systems. "
            "The report should cite sources, not guess. "
            "Hybrid search improves recall when the corpus grows."
        ),
    )

    hits = store.search(
        "How do we improve recall for research systems?",
        context="grounding with project notes",
        purpose="find evidence",
        limit=3,
    )

    assert hits
    assert hits[0].kind == "document-chunk"
    assert hits[0].metadata["retrieval_strategy"] == "contextual_dense_sparse_fusion_rerank"
    assert hits[0].metadata["retrieval_backend"].endswith("_embedding_hybrid")
    assert hits[0].metadata["hybrid_fusion"] in {"rrf", "dbsf"}
    assert hits[0].metadata["reranker"] == "rule_diversity_chunk_bonus"
    assert hits[0].metadata["rerank_provider"] == "rule"
    assert hits[0].metadata["chunk_id"]
    assert "grounding with project notes" in hits[0].metadata["grounding_query"]


def test_document_store_accepts_pluggable_reranker():
    store = DocumentStore(chunk_size=80, chunk_overlap=0, reranker=PreferSecondReranker())
    store.add(
        title="Architecture Notes",
        source="notes.md",
        content=(
            "First chunk covers planner routing and search. "
            "Second chunk covers Qdrant reranking and citation evaluation."
        ),
    )

    hits = store.search("reranking citation evaluation", limit=1)

    assert hits
    assert hits[0].metadata["reranker"] == "prefer_second"


def test_dashscope_reranker_falls_back_without_key():
    store = DocumentStore(
        chunk_size=80,
        chunk_overlap=0,
        reranker=DashScopeReranker(RerankerConfig(provider="dashscope", api_key="", base_url="")),
    )
    store.add(
        title="Grounding Notes",
        source="notes.md",
        content="Qdrant dense and sparse retrieval should still work when rerank credentials are absent.",
    )

    hits = store.search("dense sparse retrieval", limit=1)

    assert hits
    assert hits[0].metadata["reranker"] == "rule_diversity_chunk_bonus"
    assert hits[0].metadata["rerank_fallback"] == "missing_dashscope_config"


def test_dashscope_reranker_strict_mode_raises_without_key():
    store = DocumentStore(
        chunk_size=80,
        chunk_overlap=0,
        reranker=DashScopeReranker(
            RerankerConfig(
                provider="dashscope",
                api_key="",
                base_url="",
                allow_fallback=False,
            )
        ),
    )
    store.add(
        title="Grounding Notes",
        source="notes.md",
        content="Strict reranking should fail instead of silently returning rule-based results.",
    )

    with pytest.raises(RuntimeError, match="strict provider mode"):
        store.search("strict reranking", limit=1)


def test_document_store_profile_summarizes_internal_corpus():
    store = DocumentStore()
    store.add(title="Resume Notes", source="notes.md", content="Private notes about agent routing.")
    store.add(title="Project README", source="README.md", content="Project overview and grounding.")

    profile = store.profile()

    assert profile.document_count == 2
    assert profile.source_count == 2
    assert profile.has_private_docs is True
    assert "README.md" in profile.source_names
    assert profile.vector_backend in {"qdrant_dense_sparse", "local"}
