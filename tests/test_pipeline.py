from pathlib import Path

from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.deterministic_provider import DeterministicResearchModelProvider
from agentic_research_copilot.providers import (
    OpenAICompatibleResearchModelProvider,
    build_embedding_provider,
    build_model_provider,
)
from agentic_research_copilot.agents import ResearchAgent
from agentic_research_copilot.settings import AppSettings
from agentic_research_copilot.schemas import (
    EvidenceItem,
    PlanItem,
    ResearchJob,
    ResearchNote,
    ResearchRequest,
    RetrievalRoute,
    SearchQuery,
)
from agentic_research_copilot.source_reader import SourceReader


def test_chat_provider_can_use_deterministic_embedding_provider(tmp_path: Path):
    copilot = ResearchCopilot(
        settings=AppSettings(
            storage_path=str(tmp_path / "deepseek.sqlite"),
            model_provider="openai_compatible",
            model_base_url="https://api.deepseek.com",
            model_api_key="test-key",
            model_chat_model="deepseek-chat",
            embedding_provider="deterministic",
        )
    )

    assert isinstance(copilot.model_provider, OpenAICompatibleResearchModelProvider)
    assert isinstance(copilot.embedding_provider, DeterministicResearchModelProvider)
    assert copilot.documents.profile().embedding_dimensions == copilot.settings.embedding_dimensions


def test_chat_provider_can_use_separate_openai_embedding_provider(tmp_path: Path):
    settings = AppSettings(
        storage_path=str(tmp_path / "separate-embedding.sqlite"),
        model_provider="openai_compatible",
        model_base_url="https://relay.example.test/v1",
        model_api_key="chat-key",
        model_chat_model="relay-chat",
        embedding_provider="openai_compatible",
        embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_api_key="embedding-key",
        embedding_model="text-embedding-v4",
        embedding_dimensions=256,
    )
    model_provider = build_model_provider(settings)
    embedding_provider = build_embedding_provider(settings, model_provider)

    assert isinstance(model_provider, OpenAICompatibleResearchModelProvider)
    assert isinstance(embedding_provider, OpenAICompatibleResearchModelProvider)
    assert model_provider.base_url == "https://relay.example.test/v1"
    assert embedding_provider.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert embedding_provider.embedding_model == "text-embedding-v4"
    assert embedding_provider.embedding_dimensions == 256


def test_clarify_request_identifies_vague_topics(tmp_path: Path):
    copilot = ResearchCopilot(
        settings=AppSettings(
            storage_path=str(tmp_path / "clarify.sqlite"),
            langgraph_checkpoint_path=str(tmp_path / "clarify-checkpoints.sqlite"),
        )
    )

    result = copilot.clarify(ResearchRequest(topic="RAG"))

    assert result.need_clarification is True
    assert result.question
    assert "scope" in result.question.lower()
    assert result.verification == ""
    assert result.missing_dimensions


def test_clarify_request_accepts_specific_topics(tmp_path: Path):
    copilot = ResearchCopilot(
        settings=AppSettings(
            storage_path=str(tmp_path / "clarify-specific.sqlite"),
            langgraph_checkpoint_path=str(tmp_path / "clarify-specific-checkpoints.sqlite"),
        )
    )

    result = copilot.clarify(
        ResearchRequest(
            topic="Compare Open Deep Research and local AI research copilot architecture",
            depth="standard",
        )
    )

    assert result.need_clarification is False
    assert result.question == ""
    assert result.verification
    assert "research" in result.verification.lower()


def test_pipeline_returns_report(tmp_path: Path):
    copilot = ResearchCopilot(
        settings=AppSettings(
            storage_path=str(tmp_path / "pipeline.sqlite"),
            langgraph_checkpoint_path=str(tmp_path / "pipeline-checkpoints.sqlite"),
            seed_reference_knowledge=True,
        )
    )
    result = copilot.run(ResearchRequest(topic="multi-agent memory"))

    assert result.status == "completed"
    assert result.report is not None
    assert "multi-agent" in result.report.title.lower()
    assert result.research_brief
    assert result.report.sections
    assert result.report.source_count >= 1
    assert result.search_queries
    assert result.retrieval_routes
    assert result.supervisor_decision is not None
    assert any(call.name == "think_tool" for call in result.supervisor_decision.tool_calls)
    assert any(call.name == "ConductResearch" for call in result.supervisor_decision.tool_calls)
    assert any(call.name == "ResearchComplete" for call in result.supervisor_decision.tool_calls)
    assert any(query.tool in {"web_search", "vector_retrieval", "memory_recall"} for query in result.search_queries)
    assert all(route.selected_tools for route in result.retrieval_routes)
    assert any(len(route.internal_queries) >= 2 for route in result.retrieval_routes)
    assert result.corpus_profile is not None
    assert result.corpus_profile.document_count >= 1
    assert len(result.retrieval_routes) == len(result.plan)
    assert result.checkpoints
    assert result.trace
    assert result.handoffs
    assert result.evaluation is not None
    assert result.evaluation.citation_precision >= 1.0
    assert result.evaluation.plan_coverage >= 0.8
    assert result.evaluation.evidence_sufficiency >= 0.8
    assert result.evaluation.tool_selection_coverage >= 1.0
    assert result.evaluation.query_rewrite_count >= len(result.plan)
    assert result.revision_count >= 0
    assert result.failure_reason is None
    assert any(event.kind == "handoff" for event in result.trace)
    assert any(event.kind == "evaluation" for event in result.trace)
    assert any(event.actor == "planner" and event.model for event in result.trace)
    assert any(event.actor == "research_supervisor" and event.model for event in result.trace)
    assert any(event.tool_name == "ConductResearch" for event in result.trace)
    assert any(event.actor == "reporter" and event.model for event in result.trace)
    assert any(event.kind == "tool_call" and event.metadata.get("parallel") is True for event in result.trace)
    assert any(event.actor == "researcher" and "sufficiency_score" in event.metadata for event in result.trace)
    assert any(note.research_iterations for note in result.notes)
    assert any(event.actor == "researcher" and event.metadata.get("research_iteration_count", 0) >= 1 for event in result.trace)
    assert any(checkpoint.stage == "langgraph.runtime" for checkpoint in result.checkpoints)
    assert any(checkpoint.stage == "supervisor.decision" for checkpoint in result.checkpoints)
    assert any(checkpoint.stage == "research.parallel.started" for checkpoint in result.checkpoints)
    assert any(checkpoint.stage == "rag.evaluated" for checkpoint in result.checkpoints)
    assert result.report.source_index
    assert any(item.kind == "run-artifact" for item in result.evidence)
    legacy_system_headings = {
        "Problem framing",
        "Execution flow",
        "Contextual grounding",
        "Trade-offs and failure modes",
    }
    plan_questions = {item.question for item in result.plan}
    assert not any(section.heading in legacy_system_headings for section in result.report.sections)
    assert all(section.heading in plan_questions for section in result.report.sections)
    assert all(section.citations for section in result.report.sections)
    assert any("multi-agent memory" in section.content.lower() for section in result.report.sections)
    assert result.web_hits is not None
    assert result.document_hits
    assert any(hit.kind == "document-chunk" for hit in result.document_hits)
    assert any(hit.metadata.get("retrieval_strategy") == "light_rag_inspired_parent_child_dense_bm25_graph_rerank" for hit in result.document_hits)
    assert any(hit.metadata.get("parent_child_retrieval") is True for hit in result.document_hits)
    assert any(hit.metadata.get("graph_augmented_retrieval") is True for hit in result.document_hits)
    assert any(hit.metadata.get("hybrid_fusion") in {"rrf", "dbsf"} for hit in result.document_hits)
    assert any(hit.metadata.get("keyword_backend") == "sqlite_fts5_bm25" for hit in result.document_hits)
    assert any(hit.metadata.get("contextual_retrieval") is True for hit in result.document_hits)
    assert any(hit.metadata.get("contextualizer_prompt_version") == "contextual-retrieval-v1" for hit in result.document_hits)
    assert any(record.metadata.get("run_id") == result.run_id for record in copilot.memory.list())
    assert copilot.list_runs()
    assert copilot.get_run(result.run_id) is not None


def test_build_sections_uses_plan_notes_and_topic_evidence(tmp_path: Path):
    copilot = ResearchCopilot(settings=AppSettings(storage_path=str(tmp_path / "sections.sqlite")))
    request = ResearchRequest(topic="battery recycling supply chain", max_sections=1)
    plan = [
        PlanItem(
            id="item_1",
            question="How do closed-loop recycling partnerships affect battery material supply?",
            purpose="Explain whether recycling can reduce supply pressure for critical minerals.",
            search_query="closed loop battery recycling partnerships critical minerals",
        )
    ]
    route = RetrievalRoute(
        plan_item_id="item_1",
        mode="external",
        reason="Need current market evidence.",
        selected_tools=["web_search"],
        web_queries=["closed loop battery recycling partnerships critical minerals"],
        min_evidence=1,
        min_sources=1,
    )
    evidence = [
        EvidenceItem(
            title="Battery recycling partnership report",
            source="market-report",
            kind="web",
            url="https://example.test/battery-recycling",
            snippet="Closed-loop battery recycling partnerships recover lithium, nickel, and cobalt for reuse.",
            content="Closed-loop battery recycling partnerships can reduce critical mineral supply pressure.",
            score=0.91,
            metadata={"plan_item_id": "item_1"},
        )
    ]
    note = ResearchNote(
        plan_item_id="item_1",
        question=plan[0].question,
        finding="Closed-loop recycling partnerships recover lithium, nickel, and cobalt for reuse.",
        evidence_titles=["Battery recycling partnership report"],
        confidence=0.84,
        sufficiency_score=1.0,
    )

    sections = copilot._build_sections(
        request=request,
        research_brief="Assess battery recycling supply chain effects.",
        plan=plan,
        retrieval_routes=[route],
        evidence=evidence,
        web_hits=evidence,
        memory_hits=[],
        document_hits=[],
        notes=[note],
        search_queries=[
            SearchQuery(
                query="closed loop battery recycling partnerships critical minerals",
                intent="Gather evidence for battery recycling supply effects.",
                plan_item_id="item_1",
            )
        ],
    )

    assert len(sections) == 1
    assert sections[0].heading == plan[0].question
    assert "battery recycling supply chain" in sections[0].content.lower()
    assert "closed-loop recycling partnerships" in sections[0].content
    assert "LangGraph StateGraph" not in sections[0].content
    assert [item.title for item in sections[0].citations] == ["Battery recycling partnership report"]


def test_report_evidence_ranking_keeps_weak_sources_visible_but_not_primary(tmp_path: Path):
    copilot = ResearchCopilot(settings=AppSettings(storage_path=str(tmp_path / "ranking.sqlite")))
    ranked = copilot._rank_evidence_for_report(
        [
            EvidenceItem(
                title="Community thread",
                source="tavily",
                kind="web",
                url="https://www.reddit.com/r/example/comments/1",
                snippet="Community discussion.",
                score=0.95,
            ),
            EvidenceItem(
                title="Prior memory",
                source="memory",
                kind="memory",
                snippet="Memory note.",
                score=1.0,
            ),
            EvidenceItem(
                title="Qdrant Hybrid Queries",
                source="qdrant.tech/documentation/search/hybrid-queries",
                kind="document-chunk",
                url="https://qdrant.tech/documentation/search/hybrid-queries/",
                snippet="Official hybrid retrieval reference.",
                score=0.7,
                metadata={"kind": "official_reference"},
            ),
        ]
    )

    assert ranked[0].title == "Qdrant Hybrid Queries"
    assert {item.title for item in ranked} == {"Community thread", "Prior memory", "Qdrant Hybrid Queries"}


def test_research_agent_extracts_raw_content_into_read_evidence():
    def fake_search(query):
        return [
            {
                "title": "Deep Research Source",
                "source": "tavily",
                "url": "https://example.com/research",
                "snippet": "Short search snippet.",
                "content": "Short search result content.",
                "raw_content": (
                    "Introductory unrelated text. "
                    "Agentic research systems should plan searches, read sources, and verify citations. "
                    "They also use RAG as a grounding cache after documents have already been read."
                ),
                "score": 0.9,
                "metadata": {"provider": "tavily"},
            }
    ]

    agent = ResearchAgent(fake_search, source_reader_enabled=True, excerpt_max_chars=180)
    evidence = agent.collect(
        PlanItem(
            id="plan-1",
            question="How should agentic research use RAG?",
            purpose="Explain the read-and-ground path.",
        ),
        query="agentic research RAG citations",
    )

    assert evidence[0].content is not None
    assert "Agentic research systems" in evidence[0].content
    assert evidence[0].metadata["read_strategy"] == "provider_raw_content_extract"
    assert evidence[0].metadata["raw_content_chars"] > len(evidence[0].content)


def test_research_agent_iterates_until_evidence_is_sufficient():
    calls = []

    def fake_search(query):
        calls.append(query)
        if "second" in query:
            return [
                {
                    "title": "Second Source",
                    "source": "official-docs",
                    "url": "https://example.com/second",
                    "snippet": "Second independent source.",
                    "content": "Second independent source with corroborating evidence.",
                    "score": 0.84,
                }
            ]
        return [
            {
                "title": "First Source",
                "source": "paper",
                "url": "https://example.com/first",
                "snippet": "First source.",
                "content": "First source has useful but insufficient evidence.",
                "score": 0.91,
            }
        ]

    agent = ResearchAgent(fake_search, source_reader_enabled=False, max_iterations=3)
    collection = agent.collect_iterative(
        PlanItem(
            id="plan-1",
            question="How should researcher loops work?",
            purpose="Verify iterative collection.",
            search_query="researcher loop",
        ),
        ["first query", "second query"],
        min_evidence=2,
        min_sources=2,
    )

    assert calls == ["first query", "second query"]
    assert collection.completed_reason == "sufficiency_met"
    assert len(collection.evidence) == 2
    assert len(collection.iterations) == 2
    assert collection.iterations[0]["gaps"]
    assert collection.iterations[1]["gaps"] == []
    assert "researcher should continue" in collection.iterations[0]["reflection"]


def test_research_agent_can_call_mcp_when_search_evidence_is_insufficient():
    search_calls = []
    mcp_calls = []

    def fake_search(query):
        search_calls.append(query)
        return [
            {
                "title": "Initial Web Source",
                "source": "official-docs",
                "url": "https://example.com/web",
                "snippet": "Initial source with partial evidence.",
                "content": "Initial source with partial evidence.",
                "score": 0.82,
            }
        ]

    def fake_mcp(query, tool_name=None):
        mcp_calls.append((query, tool_name))
        return [
            {
                "title": "MCP Dataset Lookup",
                "source": "mcp:dataset",
                "kind": "mcp",
                "snippet": "MCP tool returned a second source.",
                "content": "MCP tool returned structured evidence from a configured external tool.",
                "score": 0.78,
                "metadata": {"mcp_tool_name": "dataset_lookup"},
            }
        ]

    agent = ResearchAgent(
        fake_search,
        mcp_tool=fake_mcp,
        source_reader_enabled=False,
        max_iterations=3,
    )
    collection = agent.collect_iterative(
        PlanItem(
            id="plan-mcp",
            question="How should external tool evidence be added?",
            purpose="Verify MCP tool routing.",
            search_query="external tool evidence",
        ),
        ["first query", "mcp query"],
        min_evidence=2,
        min_sources=2,
    )

    assert search_calls == ["first query"]
    assert mcp_calls
    assert mcp_calls[0][1] == "search_grounding_corpus"
    assert collection.completed_reason == "sufficiency_met"
    assert any(item.kind == "mcp" for item in collection.evidence)
    assert [iteration["action"] for iteration in collection.iterations] == ["web_search", "mcp_tool"]


def test_research_agent_can_model_compress_raw_content():
    def fake_search(query):
        return [
            {
                "title": "Source Reader Design",
                "source": "tavily",
                "url": "https://example.com/source-reader",
                "snippet": "Short search snippet.",
                "content": "Short search result content.",
                "raw_content": (
                    "Open Deep Research fetches raw content through search providers. "
                    "A downstream source reader compresses long webpages into summaries and key excerpts. "
                    "The final report should only cite evidence that already exists in the evidence index."
                ),
                "score": 0.9,
            }
        ]

    agent = ResearchAgent(
        fake_search,
        model_provider=DeterministicResearchModelProvider(),
        source_reader_enabled=True,
        source_reader_strategy="model_compress",
        excerpt_max_chars=400,
    )
    evidence = agent.collect(
        PlanItem(
            id="plan-1",
            question="How should source reading work?",
            purpose="Explain source compression.",
        ),
        query="Open Deep Research source reader raw content compression",
    )

    assert evidence[0].metadata["read_strategy"] == "provider_raw_content_model_compress"
    assert evidence[0].metadata["source_reader_model"] == "heuristic-source-compressor"
    assert evidence[0].metadata["key_excerpt_count"] >= 1
    assert "Key excerpts" in (evidence[0].content or "")


def test_research_agent_can_chunk_rerank_then_compress_raw_content():
    irrelevant = "General background without the target design terms. " * 80
    relevant = (
        "Open Deep Research legacy split_and_rerank splits raw webpage content into chunks, "
        "retrieves query-relevant passages, stitches them by source URL, and then feeds compact "
        "evidence into downstream synthesis."
    )

    def fake_search(query):
        return [
            {
                "title": "Split And Rerank Reader",
                "source": "tavily",
                "url": "https://example.com/split-rerank",
                "snippet": "Short search snippet.",
                "content": "Short search result content.",
                "raw_content": f"{irrelevant} {relevant} {irrelevant}",
                "score": 0.9,
            }
        ]

    provider = DeterministicResearchModelProvider()
    agent = ResearchAgent(
        fake_search,
        model_provider=provider,
        embedding_provider=provider,
        source_reader_enabled=True,
        source_reader_strategy="chunk_rerank_compress",
        excerpt_max_chars=600,
    )
    evidence = agent.collect(
        PlanItem(
            id="plan-1",
            question="How should split and rerank source reading work?",
            purpose="Explain chunk reranking before compression.",
        ),
        query="Open Deep Research split_and_rerank raw content chunks synthesis",
    )

    metadata = evidence[0].metadata
    assert metadata["read_strategy"] == "provider_raw_content_chunk_rerank_model_compress"
    assert metadata["chunk_count"] > 1
    assert metadata["selected_chunk_count"] >= 1
    assert metadata["chunk_rerank_method"] in {"blended_lexical_embedding", "lexical_fallback", "lexical"}
    assert "split_and_rerank" in (evidence[0].content or "")


def test_source_reader_expands_neighbor_chunks_before_compression():
    def block(text: str) -> str:
        return text.ljust(400, ".")

    raw_content = "".join(
        [
            block("According to the latest policy, reimbursement evidence continues next."),
            block("The reimbursement upper limit is 5000 yuan for eligible employees."),
            block("Unrelated background about office travel booking workflows."),
        ]
    )
    reader = SourceReader(
        strategy="chunk_rerank_compress",
        chunk_size=400,
        chunk_overlap=0,
        top_chunks=1,
        chunk_context_window=1,
        excerpt_max_chars=1000,
    )

    result = reader.read(
        query="reimbursement upper limit 5000",
        title="Policy",
        url="https://example.com/policy",
        raw_content=raw_content,
    )

    assert result is not None
    assert "latest policy" in result.content
    assert "5000 yuan" in result.content
    assert result.metadata["selected_chunk_count"] == 1
    assert result.metadata["expanded_chunk_count"] == 3
    assert result.metadata["chunk_context_window"] == 1
    assert result.metadata["chunk_expansion"] == "selected_chunk_neighbor_window"


def test_api_process_reads_worker_updates_from_sqlite(tmp_path: Path):
    storage_path = tmp_path / "shared-worker.sqlite"
    request = ResearchRequest(topic="single-node worker visibility")
    api_process = ResearchCopilot(
        settings=AppSettings(
            storage_path=str(storage_path),
            langgraph_checkpoint_path=str(tmp_path / "api-checkpoints.sqlite"),
        )
    )
    queued_job = ResearchJob(
        job_id="job-shared-1",
        request=request,
        status="queued",
    )
    api_process._record_job(queued_job)

    worker_process = ResearchCopilot(
        settings=AppSettings(
            storage_path=str(storage_path),
            langgraph_checkpoint_path=str(tmp_path / "worker-checkpoints.sqlite"),
        )
    )
    run = worker_process.run(request, job_id=queued_job.job_id)
    worker_process._record_job(
        queued_job.model_copy(
            update={
                "status": "completed",
                "run_id": run.run_id,
                "finished_at": run.finished_at,
            }
        )
    )

    refreshed_job = api_process.get_job(queued_job.job_id)
    refreshed_run = api_process.get_run(run.run_id)

    assert refreshed_job is not None
    assert refreshed_job.status == "completed"
    assert refreshed_job.run_id == run.run_id
    assert refreshed_run is not None
    assert refreshed_run.run_id == run.run_id
