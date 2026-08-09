from agentic_research_copilot.memory import MemoryStore
from agentic_research_copilot.deterministic_provider import DeterministicResearchModelProvider


def test_memory_store_supports_layered_recall():
    store = MemoryStore()
    store.add_session_note(
        "session:note",
        "Track transient reasoning from the current run.",
        tags=["session"],
        topic="research copilot",
        run_id="run-1",
    )
    store.add_fact(
        "fact:note",
        "Canonical facts should survive across runs.",
        tags=["canonical"],
        topic="research copilot",
        run_id="run-1",
    )
    store.add_summary(
        "summary:note",
        "Topic summaries sit between session notes and canonical facts.",
        tags=["summary"],
        topic="research copilot",
        run_id="run-1",
    )

    canonical = store.list(layer="canonical")
    summary = store.list(layer="summary")
    recalled = store.recall("research copilot", layer="summary", topic="research copilot")

    assert len(canonical) == 1
    assert canonical[0].layer == "canonical"
    assert len(summary) == 1
    assert summary[0].layer == "summary"
    assert recalled
    assert recalled[0].layer == "summary"


def test_canonical_memory_conflicts_are_marked_for_review():
    store = MemoryStore()
    store.add_fact(
        "fact:positioning",
        "The product is an AI research copilot.",
        tags=["canonical"],
        topic="research copilot",
        confidence=0.9,
    )
    conflict = store.add_fact(
        "fact:positioning",
        "The product is a generic agent platform.",
        tags=["canonical"],
        topic="research copilot",
        confidence=0.9,
    )
    report = store.governance_report()

    assert conflict.metadata["governance_status"] == "needs_review"
    assert conflict.metadata["conflict_count"] == 1
    assert report["needs_review_count"] == 1
    assert report["conflict_count"] == 1


def test_memory_recall_uses_embedding_assisted_scoring():
    store = MemoryStore(DeterministicResearchModelProvider())
    store.add_summary(
        "summary:routing",
        "Planner output should drive tool routing, vector retrieval, memory recall, and citation checks.",
        tags=["routing", "agentic-rag"],
        topic="research workflow",
        confidence=0.9,
    )
    store.add_summary(
        "summary:ui",
        "The static console renders compact admin panels and trace lists.",
        tags=["ui"],
        topic="web console",
        confidence=0.7,
    )

    recalled = store.recall("semantic route retrieval planning", layer="summary", limit=2)
    report = store.governance_report()

    assert recalled
    assert recalled[0].key == "summary:routing"
    assert "last_recall_semantic_score" in recalled[0].metadata
    assert report["embedding_enabled"] is True
    assert report["indexed_memory_count"] == 2
