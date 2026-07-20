from agentic_research_copilot.routing import RetrievalCoordinator
from agentic_research_copilot.schemas import CorpusProfile, PlanItem, ResearchRequest


def test_router_uses_hybrid_when_public_and_private_evidence_are_both_relevant():
    router = RetrievalCoordinator()
    request = ResearchRequest(
        topic="latest project architecture from private docs",
        depth="standard",
        include_private_docs=True,
    )
    item = PlanItem(
        id="abc-data",
        question="What sources support the project architecture?",
        purpose="Explain the knowledge layer and context reuse.",
        search_query="latest project architecture private docs",
    )
    profile = CorpusProfile(
        document_count=2,
        source_count=2,
        source_names=["README.md", "docs/architecture.md"],
        keyword_signals=["project", "architecture", "grounding"],
        has_private_docs=True,
    )

    routes = router.build_routes(request, "Research project architecture.", [item], profile)

    assert routes[0].mode == "hybrid"
    assert routes[0].web_query
    assert routes[0].internal_query
    assert "contextual grounding" in routes[0].reason
    assert routes[0].selected_tools == ["web_search", "vector_retrieval", "memory_recall"]
    assert len(routes[0].web_queries) >= 2
    assert len(routes[0].internal_queries) >= 2
    assert routes[0].min_evidence >= 2
    assert routes[0].sufficiency_criteria


def test_router_falls_back_to_external_when_private_corpus_is_unavailable():
    router = RetrievalCoordinator()
    request = ResearchRequest(topic="research copilot benchmarks", include_private_docs=True)
    item = PlanItem(
        id="abc-risk",
        question="What public benchmarks matter?",
        purpose="Compare industry evidence.",
        search_query="research copilot benchmarks",
    )
    profile = CorpusProfile(has_private_docs=False)

    routes = router.build_routes(request, "Compare public evidence.", [item], profile)

    assert routes[0].mode == "external"
    assert routes[0].web_query == "research copilot benchmarks"
    assert routes[0].internal_query is None
    assert routes[0].selected_tools == ["web_search", "memory_recall"]
