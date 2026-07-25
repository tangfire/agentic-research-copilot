from pathlib import Path

from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.providers import (
    DeterministicResearchModelProvider,
    OpenAICompatibleResearchModelProvider,
    build_embedding_provider,
    build_model_provider,
)
from agentic_research_copilot.agents import ResearchAgent
from agentic_research_copilot.settings import AppSettings
from agentic_research_copilot.schemas import EvidenceItem, PlanItem, ResearchJob, ResearchRequest
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
    assert any(checkpoint.stage == "langgraph.runtime" for checkpoint in result.checkpoints)
    assert any(checkpoint.stage == "supervisor.decision" for checkpoint in result.checkpoints)
    assert any(checkpoint.stage == "research.parallel.started" for checkpoint in result.checkpoints)
    assert any(checkpoint.stage == "rag.evaluated" for checkpoint in result.checkpoints)
    assert result.report.source_index
    assert any(item.kind == "run-artifact" for item in result.evidence)
    assert any(
        item.kind == "run-artifact"
        for section in result.report.sections
        if section.heading == "Execution flow"
        for item in section.citations
    )
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
