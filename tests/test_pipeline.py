from pathlib import Path

from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.providers import (
    DeterministicResearchModelProvider,
    OpenAICompatibleResearchModelProvider,
    build_embedding_provider,
    build_model_provider,
)
from agentic_research_copilot.settings import AppSettings
from agentic_research_copilot.schemas import ResearchJob, ResearchRequest


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
        settings=AppSettings(storage_path=str(tmp_path / "pipeline.sqlite"))
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
    assert any(event.actor == "reporter" and event.model for event in result.trace)
    assert any(event.kind == "tool_call" and event.metadata.get("parallel") is True for event in result.trace)
    assert any(event.actor == "researcher" and "sufficiency_score" in event.metadata for event in result.trace)
    assert any(checkpoint.stage == "langgraph.runtime" for checkpoint in result.checkpoints)
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
    assert any(hit.metadata.get("retrieval_strategy") == "contextual_dense_sparse_fusion_rerank" for hit in result.document_hits)
    assert any(hit.metadata.get("hybrid_fusion") in {"rrf", "dbsf"} for hit in result.document_hits)
    assert any(record.metadata.get("run_id") == result.run_id for record in copilot.memory.list())
    assert copilot.list_runs()
    assert copilot.get_run(result.run_id) is not None


def test_api_process_reads_worker_updates_from_sqlite(tmp_path: Path):
    storage_path = tmp_path / "shared-worker.sqlite"
    request = ResearchRequest(topic="single-node worker visibility")
    api_process = ResearchCopilot(settings=AppSettings(storage_path=str(storage_path)))
    queued_job = ResearchJob(
        job_id="job-shared-1",
        request=request,
        status="queued",
    )
    api_process._record_job(queued_job)

    worker_process = ResearchCopilot(settings=AppSettings(storage_path=str(storage_path)))
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
