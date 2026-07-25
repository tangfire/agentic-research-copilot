import pytest

from agentic_research_copilot.providers import DeterministicResearchModelProvider, ModelUsage
from agentic_research_copilot.retrieval import (
    BaseReranker,
    DashScopeReranker,
    DocumentStore,
    RerankerConfig,
)
from agentic_research_copilot.schemas import ChunkContextContract


class PreferSecondReranker(BaseReranker):
    name = "prefer_second"

    def rerank(self, query, scored_chunks, limit):
        reordered = sorted(scored_chunks, key=lambda item: 0 if item[1].chunk_index == 1 else 1)
        return [
            (score, chunk, {**scores, "reranker": self.name, "rerank_score": score})
            for score, chunk, scores in reordered[:limit]
        ]


class StaticContextProvider(DeterministicResearchModelProvider):
    name = "static_context_provider"

    def contextualize_chunk(self, **kwargs):
        return (
            ChunkContextContract(
                context=(
                    "This chunk belongs to the payment reconciliation section and explains PayPal "
                    "callback repair, settlement audit, and amount verification."
                ),
                key_terms=["payment", "paypal", "reconciliation", "settlement", "callback"],
                provenance_hint="payment reconciliation section",
                confidence=0.91,
            ),
            ModelUsage(provider=self.name, model="static-contextualizer", prompt_tokens=24, completion_tokens=24),
        )


def test_document_store_returns_contextual_chunks():
    store = DocumentStore(chunk_size=120, chunk_overlap=20)
    document = store.add(
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
    assert hits[0].metadata["retrieval_strategy"] == "light_rag_inspired_parent_child_dense_bm25_graph_rerank"
    assert hits[0].metadata["parent_child_retrieval"] is True
    assert hits[0].metadata["parent_id"] == document.metadata["document_id"]
    assert hits[0].metadata["retrieval_backend"].endswith("_embedding_hybrid")
    assert hits[0].metadata["hybrid_fusion"] in {"rrf", "dbsf"}
    assert hits[0].metadata["keyword_backend"] == "sqlite_fts5_bm25"
    assert hits[0].metadata["contextual_retrieval"] is True
    assert hits[0].metadata["context_prefix"]
    assert hits[0].metadata["reranker"] == "rule_diversity_chunk_bonus"
    assert hits[0].metadata["rerank_provider"] == "rule"
    assert hits[0].metadata["chunk_id"]
    assert hits[0].metadata["document_id"] == document.metadata["document_id"]
    assert "grounding with project notes" in hits[0].metadata["grounding_query"]


def test_document_store_expands_child_hit_to_parent_context():
    store = DocumentStore(chunk_size=70, chunk_overlap=0, parent_context_window=2, parent_context_max_chars=1200)
    store.add(
        title="Pricing Playbook",
        source="pricing.md",
        content=(
            "Parent opening explains order pricing context and checkout invariants.\n\n"
            "The child chunk mentions coupon stacking, freight discounting, and payment amount validation.\n\n"
            "Parent closing explains settlement reconciliation and callback repair."
        ),
    )

    hits = store.search("coupon stacking payment validation", limit=1)

    assert hits
    assert hits[0].metadata["parent_child_retrieval"] is True
    assert hits[0].metadata["parent_expansion"] == "same-document_neighbor_window"
    assert hits[0].metadata["parent_context_window"] == 2
    assert "coupon stacking" in (hits[0].content or "")
    assert "Parent opening" in (hits[0].content or "")
    assert "Parent closing" in (hits[0].content or "")


def test_document_store_fuses_light_rag_inspired_graph_signal():
    store = DocumentStore(chunk_size=130, chunk_overlap=0, graph_enabled=True, graph_neighbor_limit=4)
    store.add(
        title="Argus Runtime Notes",
        source="argus.md",
        content=(
            "Hermes scheduler dispatches source connections and records checkpoint recovery. "
            "Zeus runtime governance shares lineage metadata with Hermes.\n\n"
            "Dashboard copy discusses screenshots and unrelated release notes."
        ),
    )

    hits = store.search("Zeus scheduler lineage", limit=2)

    assert hits
    assert any(hit.metadata.get("graph_augmented") is True for hit in hits)
    assert any("Zeus" in hit.metadata.get("graph_query_entities", []) for hit in hits)
    assert any(hit.metadata.get("graph_score", 0) > 0 for hit in hits)
    assert all(hit.metadata.get("graph_augmented_retrieval") is True for hit in hits)


def test_document_store_uses_real_bm25_keyword_index_for_exact_terms():
    store = DocumentStore(chunk_size=140, chunk_overlap=0)
    store.add(
        title="Generic Planning Notes",
        source="generic.md",
        content="This note discusses broad research planning, synthesis, and citation review.",
    )
    lexical_doc = store.add(
        title="Protocol Reference",
        source="protocol.md",
        content="The rare protocol codename ZKMERKLE-481 controls checkpoint replay and source audit repair.",
    )

    hits = store.search("ZKMERKLE-481 checkpoint replay", limit=1)

    assert hits
    assert hits[0].metadata["document_id"] == lexical_doc.metadata["document_id"]
    assert hits[0].metadata["keyword_backend"] == "sqlite_fts5_bm25"
    assert hits[0].metadata["keyword_augmented"] is True
    assert hits[0].metadata["bm25_score"] > 0
    assert "zkmerkle" in hits[0].metadata["matched_terms"]


def test_contextual_retrieval_prefix_is_indexed_for_bm25():
    store = DocumentStore(
        chunk_size=160,
        chunk_overlap=0,
        contextualizer_provider=StaticContextProvider(),
    )
    document = store.add(
        title="Payment Ops",
        source="payment.md",
        content="Callback handling is covered here. The operational checklist is concise.",
    )

    hits = store.search("PayPal reconciliation settlement", limit=1)

    assert hits
    assert hits[0].metadata["document_id"] == document.metadata["document_id"]
    assert hits[0].metadata["contextualizer_provider"] == "static_context_provider"
    assert hits[0].metadata["contextualizer_model"] == "static-contextualizer"
    assert hits[0].metadata["contextualizer_prompt_version"] == "contextual-retrieval-v1"
    assert "PayPal callback repair" in hits[0].metadata["context_prefix"]
    assert hits[0].metadata["bm25_score"] > 0
    assert "paypal" in hits[0].metadata["matched_terms"]


def test_document_store_chunks_long_paragraphs_with_sentence_boundaries():
    store = DocumentStore(chunk_size=90, chunk_overlap=12)
    store.add(
        title="Long Parser Notes",
        source="parser.md",
        content=(
            "Parsing quality matters for retrieval. "
            "Chunk boundaries should preserve useful evidence. "
            "Reranking should still find the parser discussion."
        ),
    )

    hits = store.search("parser chunk boundaries retrieval", limit=3)

    assert hits
    assert any("Chunk boundaries" in (hit.content or "") for hit in hits)
    assert all(hit.metadata["total_chunks"] >= 2 for hit in hits)


def test_document_store_can_delete_single_document_from_index():
    store = DocumentStore(chunk_size=120, chunk_overlap=20)
    removed = store.add(
        title="Temporary Notes",
        source="tmp.md",
        content="Temporary indexed evidence about deletion should disappear from retrieval.",
    )
    kept = store.add(
        title="Persistent Notes",
        source="kept.md",
        content="Persistent indexed evidence about citations should remain searchable.",
    )

    assert store.delete(removed.metadata["document_id"]) is True
    assert store.delete("missing-document") is False

    titles = {doc.title for doc in store.list()}
    assert titles == {"Persistent Notes"}

    hits = store.search("deletion evidence", limit=3)
    assert all(hit.metadata["document_id"] != removed.metadata["document_id"] for hit in hits)
    remaining_hits = store.search("citations evidence", limit=3)
    assert any(hit.metadata["document_id"] == kept.metadata["document_id"] for hit in remaining_hits)


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
        content="Qdrant dense retrieval and SQLite BM25 keyword search should still work when rerank credentials are absent.",
    )

    hits = store.search("dense BM25 retrieval", limit=1)

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
    assert profile.vector_backend in {"qdrant_dense", "local_dense"}
    assert profile.keyword_backend == "sqlite_fts5_bm25"
