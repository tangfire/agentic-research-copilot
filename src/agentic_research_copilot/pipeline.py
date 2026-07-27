from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import RLock

from .agents import PlannerAgent, ResearchAgent, ReporterAgent, SupervisorAgent, VerifierAgent
from .document_reader import DocumentReader
from .evaluation import RAGEvaluator
from .ledger import JobLedger, RunLedger
from .mcp_tools import build_mcp_tool
from .providers import build_embedding_provider, build_model_provider
from .provider_validation import provider_runtime_report, require_real_provider_config
from .memory import MemoryStore
from .retrieval import DocumentStore, RerankerConfig, build_reranker
from .schemas import (
    AgentHandoff,
    ClarificationContract,
    CorpusProfile,
    EvidenceItem,
    ResearchNote,
    ResearchJob,
    ResearchRequest,
    ResearchRun,
    ReportSection,
    RetrievalRoute,
    RunCheckpoint,
    RunTraceEvent,
    SearchQuery,
    SupervisorDecisionContract,
)
from .routing import RetrievalCoordinator
from .search import OPEN_DEEP_RESEARCH_STYLE_PROVIDERS, SearchTool, build_search_tool, search_provider_requires_key
from .settings import AppSettings, load_settings, resolve_storage_path
from .source_reader import source_reader_strategy_label
from .storage import SQLiteStore
from .telemetry import TelemetryLog
from .workflow import ResearchWorkflow


@dataclass
class PlanItemResearchResult:
    item_id: str
    web_evidence: list[EvidenceItem] = field(default_factory=list)
    document_evidence: list[EvidenceItem] = field(default_factory=list)
    note: ResearchNote | None = None
    web_latency_ms: int = 0
    document_latency_ms: int = 0


class ResearchCopilot:
    """Research orchestration layer combining upstream-inspired workflow pieces."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        storage: SQLiteStore | None = None,
        search_tool: SearchTool | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        if self.settings.strict_providers:
            require_real_provider_config(self.settings)
        self.model_provider = build_model_provider(self.settings)
        self.embedding_provider = build_embedding_provider(self.settings, self.model_provider)
        self.mcp_tool = build_mcp_tool(self.settings)
        self.reranker = build_reranker(
            RerankerConfig(
                provider=self.settings.rerank_provider,
                model=self.settings.rerank_model,
                base_url=self.settings.rerank_base_url,
                api_key=self.settings.rerank_api_key,
                timeout_seconds=self.settings.rerank_timeout_seconds,
                candidate_limit=self.settings.rerank_candidate_limit,
                allow_fallback=not self.settings.strict_providers,
            )
        )
        self.memory = MemoryStore(self.embedding_provider)
        self.documents = DocumentStore(
            self.embedding_provider,
            collection_name=self.settings.qdrant_collection,
            qdrant_url=self.settings.qdrant_url,
            qdrant_api_key=self.settings.qdrant_api_key,
            qdrant_location=self.settings.qdrant_location,
            qdrant_prefer_local=self.settings.qdrant_prefer_local,
            hybrid_fusion=self.settings.rag_hybrid_fusion,
            graph_enabled=self.settings.rag_graph_enabled,
            graph_max_entities_per_chunk=self.settings.rag_graph_max_entities_per_chunk,
            graph_neighbor_limit=self.settings.rag_graph_neighbor_limit,
            reranker=self.reranker,
            contextualizer_provider=self.model_provider,
            allow_local_fallback=not self.settings.strict_providers,
        )
        self.document_reader = DocumentReader()
        self.telemetry = TelemetryLog()
        self.ledger = RunLedger()
        self.jobs = JobLedger()
        self.storage = storage or SQLiteStore(resolve_storage_path(self.settings.storage_path))
        self._job_lock = RLock()
        self._job_executor: ThreadPoolExecutor | None = None
        self.workflow = ResearchWorkflow()
        self.evaluator = RAGEvaluator()
        self.router = RetrievalCoordinator(
            max_query_rewrites=self.settings.rag_max_query_rewrites,
            min_evidence_per_item=self.settings.rag_min_evidence_per_item,
            min_source_diversity=self.settings.rag_min_source_diversity,
            mcp_enabled=self.settings.mcp_enabled,
        )
        self.planner = PlannerAgent(self.model_provider, self.settings)
        self.researcher = ResearchAgent(
            search_tool or build_search_tool(self.settings),
            model_provider=self.model_provider,
            embedding_provider=self.embedding_provider,
            mcp_tool=self.mcp_tool,
            source_reader_enabled=self.settings.source_reader_enabled,
            source_reader_strategy=self.settings.source_reader_strategy,
            raw_content_max_chars=self.settings.source_reader_max_chars,
            excerpt_max_chars=self.settings.source_reader_excerpt_chars,
            chunk_context_window=self.settings.source_reader_chunk_context_window,
            max_iterations=self.settings.research_max_iterations,
        )
        self.verifier = VerifierAgent(self.model_provider, self.settings)
        self.reporter = ReporterAgent(self.model_provider, self.settings)
        self.supervisor_agent = SupervisorAgent(self.model_provider, self.settings)
        self._restore_state()
        if self.settings.seed_reference_knowledge:
            self._seed_reference_knowledge()

    def add_document(
        self,
        title: str,
        source: str,
        url: str | None = None,
        snippet: str | None = None,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> EvidenceItem:
        document = self.documents.add(
            title=title,
            source=source,
            url=url,
            snippet=snippet,
            content=content,
            metadata=metadata,
        )
        self.storage.save_document(document)
        return document

    def ingest_document_path(
        self,
        path: str,
        *,
        title: str | None = None,
        source: str | None = None,
        url: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> list[EvidenceItem]:
        segments = self.document_reader.read_path(
            path,
            title=title,
            source=source,
            url=url,
            metadata=metadata,
        )
        return [
            self.add_document(
                title=segment.title,
                source=segment.source,
                url=segment.url,
                content=segment.content,
                metadata=segment.metadata,
            )
            for segment in segments
        ]

    def delete_document(self, document_id: str) -> bool:
        deleted_from_index = self.documents.delete(document_id)
        deleted_from_storage = self.storage.delete_document(document_id)
        return deleted_from_index or deleted_from_storage

    def clear_documents(self) -> dict[str, object]:
        storage_deleted = self.storage.clear_documents()
        indexed_deleted = len(self.documents.list())
        self.documents.clear()
        return {
            "deleted": True,
            "documents_deleted": max(storage_deleted, indexed_deleted),
        }

    def clear_history(self, *, include_memory: bool = False) -> dict[str, object]:
        storage_runs_deleted = self.storage.clear_runs()
        storage_jobs_deleted = self.storage.clear_jobs()
        runs_deleted = max(storage_runs_deleted, self.ledger.clear())
        jobs_deleted = max(storage_jobs_deleted, self.jobs.clear())
        telemetry_deleted = self.telemetry.clear()
        memory_deleted = 0
        if include_memory:
            storage_memory_deleted = self.storage.clear_memory()
            memory_deleted = max(storage_memory_deleted, self.memory.clear())
        return {
            "deleted": True,
            "runs_deleted": runs_deleted,
            "jobs_deleted": jobs_deleted,
            "telemetry_deleted": telemetry_deleted,
            "memory_deleted": memory_deleted,
            "memory_cleared": include_memory,
        }

    def clarify(self, request: ResearchRequest) -> ClarificationContract:
        corpus_profile = self.documents.profile()
        memory_records = self._recall_memory_context(request, "clarification") if request.use_memory else []
        contract, usage = self.model_provider.clarify_request(
            request,
            corpus_profile,
            memory_records=memory_records,
        )
        self.telemetry.emit(
            "clarification.checked",
            request.topic,
            actor="clarifier",
            provider=usage.provider,
            model=usage.model,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=usage.latency_ms,
            need_clarification=contract.need_clarification,
            missing_dimensions=contract.missing_dimensions,
        )
        return contract

    def add_memory(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        *,
        layer: str = "session",
        run_id: str | None = None,
        session_id: str | None = None,
        topic: str | None = None,
        confidence: float = 0.0,
    ):
        if layer == "summary":
            record = self.memory.add_summary(
                key=key,
                value=value,
                tags=tags,
                metadata=metadata,
                run_id=run_id,
                session_id=session_id,
                topic=topic,
                confidence=confidence,
            )
        elif layer == "canonical":
            record = self.memory.add_fact(
                key=key,
                value=value,
                tags=tags,
                metadata=metadata,
                run_id=run_id,
                session_id=session_id,
                topic=topic,
                confidence=confidence,
            )
        else:
            record = self.memory.add_session_note(
                key=key,
                value=value,
                tags=tags,
                metadata=metadata,
                run_id=run_id,
                session_id=session_id,
                topic=topic,
                confidence=confidence,
            )
        self.storage.save_memory(record)
        return record

    def list_runs(self) -> list[ResearchRun]:
        self.ledger.extend(self.storage.load_runs())
        return self.ledger.list()

    def get_run(self, run_id: str) -> ResearchRun | None:
        persisted = self.storage.load_run(run_id)
        if persisted is not None:
            return self.ledger.record(persisted)
        return self.ledger.get(run_id)

    def submit_job(self, request: ResearchRequest) -> ResearchJob:
        job = ResearchJob(
            job_id=str(uuid.uuid4()),
            request=request.model_copy(),
            status="queued",
            max_attempts=max(1, self.settings.job_max_attempts),
            timeout_seconds=self.settings.job_timeout_seconds,
        )
        self._record_job(job)
        self.telemetry.emit("job.queued", request.topic, job_id=job.job_id)
        if self.settings.job_queue_backend == "celery" and self._submit_celery_job(job.job_id):
            return job
        if self.settings.job_queue_backend == "celery" and self.settings.strict_providers:
            message = "Celery queue backend is configured but enqueue failed in strict provider mode."
            failed_job = job.model_copy(
                update={
                    "status": "failed",
                    "error": message,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._record_job(failed_job)
            raise RuntimeError(message)
        self._ensure_job_executor().submit(self._execute_job, job.job_id)
        return job

    def list_jobs(self) -> list[ResearchJob]:
        self.jobs.extend(self.storage.load_jobs())
        return self.jobs.list()

    def get_job(self, job_id: str) -> ResearchJob | None:
        persisted = self.storage.load_job(job_id)
        if persisted is not None:
            return self.jobs.record(persisted)
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> ResearchJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.status in {"completed", "failed", "cancelled"}:
            return job
        if job.status == "queued":
            return self._mark_job_cancelled(
                job,
                message="Job was cancelled before execution started.",
            )
        updated = job.model_copy(
            update={
                "cancel_requested": True,
                "error": "Cancellation requested; the in-process worker will stop at the next safe boundary.",
            }
        )
        self._record_job(updated)
        self.telemetry.emit("job.cancel_requested", updated.request.topic, job_id=job_id)
        return updated

    def close(self) -> None:
        with self._job_lock:
            executor = self._job_executor
            self._job_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self.documents.close()

    def refresh_state(self) -> None:
        self._restore_state()

    def runtime_config(self) -> dict[str, object]:
        corpus_profile = self.documents.profile()
        return {
            "product": {
                "name": "AI Research Copilot",
                "positioning": (
                    "A deep research assistant that turns complex questions into planned, cited, "
                    "traceable, and reviewable research reports."
                ),
                "release_shape": "single Python FastAPI service with a background job queue, Qdrant retrieval, and a static local web console",
                "research_path": "plan -> search/read -> synthesize -> verify/evaluate -> replay",
                "seed_reference_knowledge": self.settings.seed_reference_knowledge,
            },
            "clarification": {
                "enabled": True,
                "mode": "ODR-style clarify_with_user front door",
                "purpose": "Ask at most one concise follow-up question when the request is too vague to research safely.",
            },
            "orchestration": {
                "runtime": self.settings.orchestration_runtime,
                "strict_providers": self.settings.strict_providers,
                "active_graph": "supervisor -> memory -> planner -> research_supervisor -> parallel_research -> reporter -> verifier/evaluator -> memory_write",
                "fallback_runtime": "custom",
                "checkpointer": self.settings.langgraph_checkpointer,
                "checkpoint_path": self.settings.langgraph_checkpoint_path,
                "durability_boundary": "Single-node LangGraph sqlite checkpointing is the default graph durability layer; SQLite run traces/replay are always persisted by the app, with MemorySaver used only as a defensive fallback.",
                "reference_pattern": "Open Deep Research uses LangGraph StateGraph/subgraph orchestration; this repo uses LangGraph for the v1 research workflow while keeping product-specific nodes.",
            },
            "modeling": {
                "strict_providers": self.settings.strict_providers,
                "provider": self.settings.model_provider,
                "chat_model": self.settings.model_chat_model,
                "embedding_provider": self.settings.embedding_provider,
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "temperature": self.settings.model_temperature,
                "max_revisions": self.settings.max_revisions,
            },
            "job_execution": {
                "mode": "background",
                "executor": (
                    "Celery over Redis with in-process fallback"
                    if self.settings.job_queue_backend == "celery"
                    else "single-worker thread queue"
                ),
                "queue_backend": self.settings.job_queue_backend,
                "status_contract": ["queued", "running", "completed", "failed", "cancelled"],
                "reason": "Research tasks can be inspected independently from final run artifacts.",
                "research_max_workers": self.settings.research_max_workers,
                "max_attempts": self.settings.job_max_attempts,
                "timeout_seconds": self.settings.job_timeout_seconds,
                "cancel_contract": "queued jobs cancel immediately; running jobs record cancellation and stop at the next safe boundary",
                "celery_broker_configured": bool(self.settings.celery_broker_url),
                "production_boundary": "single-node personal deployment; Celery/Redis is optional for process separation, not a distributed platform claim",
                "state_source": "SQLite is the source of truth for job/run status so an API process can observe updates written by a separate local worker.",
                "fallback_allowed": not self.settings.strict_providers,
            },
            "provider_readiness": provider_runtime_report(self.settings),
            "reference_designs": [
                {
                    "name": "Open Deep Research",
                    "learning_priority": "primary",
                    "used_for": [
                        "clarify_with_user front door and structured research brief generation",
                        "LangGraph-oriented research graph structure",
                        "plan -> research -> compress -> report loop",
                        "LLM supervisor tool loop with think/delegate/complete decisions",
                        "provider raw-content reading and compression",
                        "MCP-compatible external tool layer",
                        "citation-backed final answer contract",
                        "research state shaped around questions, notes, evidence, and sections",
                        "judge/evaluator-style demo artifacts",
                    ],
                    "dependency": False,
                },
                {
                    "name": "PraisonAI",
                    "learning_priority": "secondary",
                    "used_for": [
                        "memory and persistence concepts",
                        "reader registry and document ingestion extension shape",
                        "agent handoff vocabulary",
                        "observability, replay, and run-ledger patterns",
                    ],
                    "dependency": False,
                },
            ],
            "open_deep_research_alignment": {
                "positioning": (
                    "Complex-question AI Research Copilot: plan, search/read, retrieve, "
                    "recall memory, synthesize, verify citations, evaluate, and replay."
                ),
                "matched_boundaries": [
                    "ODR-style clarify_with_user front door before research starts",
                    "LangGraph-style supervisor/research/report graph",
                    "ODR-style supervisor decisions expressed as think_tool, ConductResearch, and ResearchComplete calls",
                    "MCP-compatible tool registry as an additional external evidence channel",
                    "provider raw_content reading plus compression before report synthesis",
                    "citation-backed report sections mapped to existing evidence",
                    "source quality surfaced through evaluation rather than runtime source blocking",
                    "LLM judge style demo artifact outside the hot path",
                ],
                "product_specific_differences": [
                    "Uses an ODR-style LLM research supervisor; ConductResearch calls carry selected tools, query rewrites, grounding mode, and sufficiency criteria",
                    "Keeps deterministic route hints only for offline tests/fallbacks, not as the primary real-provider decision layer",
                    "Adds local document grounding with text/Markdown/HTML/PDF parsing before Qdrant retrieval",
                    "Uses single-node FastAPI/Celery/Redis/SQLite/Qdrant deployment, not a distributed platform",
                ],
            },
            "agents": [
                {"name": "planner", "role": "creates a research brief and decomposed plan"},
                {
                    "name": "clarifier",
                    "role": "decides whether the request is specific enough to start research or needs one concise follow-up question",
                },
                {
                    "name": "research_supervisor",
                    "role": "emits ODR-style think_tool, ConductResearch, and ResearchComplete decisions before research execution",
                },
                {
                    "name": "route_materializer",
                    "role": "turns ConductResearch tool-call arguments into executable web/vector/memory route contracts",
                },
                {"name": "researcher", "role": "collects web evidence through the configured search tool"},
                {
                    "name": "grounding_layer",
                    "role": "retrieves precise child chunks, expands parent/neighbor context, then applies dense/BM25 fusion and reranking",
                },
                {"name": "memory_manager", "role": "recalls and writes structured topic memory"},
                {"name": "supervisor", "role": "coordinates handoffs, retries, and failure states"},
                {
                    "name": "reporter",
                    "role": "synthesizes compressed findings into citation-backed sections while mapping citations to existing evidence",
                },
                {"name": "critic", "role": "checks source diversity, coverage, and confidence"},
            ],
            "tool_registry": [
                {
                    "name": "web_search",
                    "provider": self.settings.search_provider,
                    "enabled": self.settings.search_provider != "none",
                    "max_results": self.settings.search_max_results,
                    "base_url": self.settings.search_base_url,
                    "model": self.settings.search_model,
                    "depth": self.settings.search_depth,
                    "include_raw_content": self.settings.search_include_raw_content,
                    "reader_enabled": self.settings.source_reader_enabled,
                    "reader_strategy": source_reader_strategy_label(self.settings.source_reader_strategy),
                    "reader_max_chars": self.settings.source_reader_max_chars,
                    "reader_excerpt_chars": self.settings.source_reader_excerpt_chars,
                    "reader_chunk_context_window": self.settings.source_reader_chunk_context_window,
                    "research_max_iterations": self.settings.research_max_iterations,
                    "reader_contract": (
                        "split/rerank -> neighbor expansion -> summary/key_excerpts/relevance/limitations"
                        if self.settings.source_reader_strategy == "chunk_rerank_compress"
                        else "summary/key_excerpts/relevance/limitations"
                        if self.settings.source_reader_strategy == "model_compress"
                        else "query_relevant_excerpt"
                    ),
                    "api_key_configured": bool(self.settings.search_api_key)
                    if search_provider_requires_key(self.settings.search_provider)
                    else True,
                    "open_deep_research_style": self.settings.search_provider in OPEN_DEEP_RESEARCH_STYLE_PROVIDERS,
                },
                {
                    "name": "mcp_tool",
                    "provider": "model_context_protocol",
                    "enabled": self.settings.mcp_enabled,
                    "server_url_configured": bool(self.settings.mcp_server_url),
                    "loaded": self.mcp_tool is not None,
                    "tools": self.settings.mcp_tools,
                    "tools_configured": bool(self.settings.mcp_tools),
                    "auth_required": self.settings.mcp_auth_required,
                    "auth_token_configured": bool(self.settings.mcp_auth_token),
                    "mcp_prompt_configured": bool(self.settings.mcp_prompt),
                    "transport": self.settings.mcp_transport,
                    "timeout_seconds": self.settings.mcp_timeout_seconds,
                    "reference_shape": "Open Deep Research loads configured MCP tools from mcp_config.url + mcp_config.tools into the researcher toolkit; this repo exposes the same configured tools as citation-backed evidence.",
                },
                {
                    "name": "model_provider",
                    "provider": self.settings.model_provider,
                    "enabled": True,
                    "chat_model": self.settings.model_chat_model,
                    "embedding_provider": self.settings.embedding_provider,
                    "embedding_model": self.settings.embedding_model,
                    "embedding_api_key_configured": bool(self.settings.embedding_api_key)
                    if self.settings.embedding_provider == "openai_compatible"
                    else True,
                },
                {
                    "name": "document_retrieval",
                    "provider": corpus_profile.vector_backend,
                    "enabled": corpus_profile.has_private_docs,
                    "strategy": "light_rag_inspired_parent_child_dense_bm25_graph_rerank",
                    "parent_child": "child chunks are retrieved precisely, then same-document neighbor context is returned for synthesis",
                    "contextual_retrieval": "indexing-time chunk context prefixes are prepended before dense embedding and BM25 indexing",
                    "graph_augmented": self.settings.rag_graph_enabled,
                    "graph_strategy": "LightRAG-inspired entity co-occurrence graph signal fused with dense/BM25 candidates",
                    "collection": corpus_profile.collection_name,
                    "keyword_backend": corpus_profile.keyword_backend,
                    "reranker": self.reranker.name,
                    "rerank_provider": self.settings.rerank_provider,
                    "rerank_model": self.settings.rerank_model,
                    "rerank_api_key_configured": bool(self.settings.rerank_api_key),
                },
                {
                    "name": "document_reader",
                    "provider": "local_file_reader",
                    "enabled": True,
                    "supported_inputs": ["text", "markdown", "html", "pdf_with_pymupdf"],
                    "parser_boundary": "file parsing is separated from vector indexing and research orchestration",
                    "section_strategy": "Markdown/HTML headings become section segments with section_heading and section_path metadata before chunking",
                    "pdf_strategy": (
                        "PDFs are split into page segments with page_number/page_count, block, "
                        "heading-hint, table, and layout metadata before chunking"
                    ),
                    "chunking_strategy": "DocumentStore performs paragraph-aware child chunking, contextual retrieval prefixing, dense embedding, and BM25 indexing",
                    "parent_child_strategy": "retrieval scores child chunks and returns same-document parent/neighbor context for synthesis",
                    "graph_strategy": "LightRAG-inspired lightweight entity/relation graph; graph hits are fused before rerank",
                },
                {"name": "memory_search", "provider": "layered_structured_memory", "enabled": True},
                {"name": "run_ledger", "provider": "sqlite", "enabled": True},
                {"name": "job_queue", "provider": self.settings.job_queue_backend, "enabled": True},
                {"name": "telemetry", "provider": "in_process_event_log", "enabled": True},
            ],
            "retrieval": {
                "routes": ["external", "internal", "hybrid"],
                "default_strategy": "light_rag_inspired_parent_child_dense_bm25_graph_rerank",
                "hybrid_pipeline": {
                    "parent_child": "child retrieval with parent/neighbor context expansion",
                    "contextual_retrieval": "chunk-specific context prefixes are generated at ingestion and indexed with the chunk",
                    "graph_augmented": self.settings.rag_graph_enabled,
                    "graph_max_entities_per_chunk": self.settings.rag_graph_max_entities_per_chunk,
                    "graph_neighbor_limit": self.settings.rag_graph_neighbor_limit,
                    "dense_vector": "Qdrant named vector 'dense'",
                    "keyword_index": "SQLite FTS5 bm25() over contextual child chunks",
                    "fusion": self.settings.rag_hybrid_fusion,
                    "reranker": self.reranker.name,
                    "rerank_provider": self.settings.rerank_provider,
                    "rerank_model": self.settings.rerank_model,
                    "fallback": (
                        "disabled in strict provider mode"
                        if self.settings.strict_providers
                        else "local dense + SQLite BM25 fusion when Qdrant is unavailable"
                    ),
                },
                "agentic_rag": {
                    "query_rewrite": True,
                    "max_query_rewrites": self.settings.rag_max_query_rewrites,
                    "tool_selection": ["web_search", "vector_retrieval", "memory_recall", "mcp_tool"],
                    "min_evidence_per_item": self.settings.rag_min_evidence_per_item,
                    "min_source_diversity": self.settings.rag_min_source_diversity,
                    "sufficiency_check": "route-level evidence thresholds feed the verifier/evaluator revision loop",
                },
                "vector_backend": corpus_profile.vector_backend,
                "keyword_backend": corpus_profile.keyword_backend,
                "embedding_dimensions": corpus_profile.embedding_dimensions,
                "collection_name": corpus_profile.collection_name,
                "production_upgrade": "swap the model adapter or vector backend without changing the route contract",
                "corpus_profile": corpus_profile.model_dump(),
            },
            "memory": {
                "layers": ["session", "canonical", "summary"],
                "recall_strategy": "lexical + embedding semantic similarity + confidence/layer/governance weighting",
                "write_policy": [
                    "session notes capture the run summary",
                    "summary memory stores topic-level takeaways",
                    "canonical memory stores verified facts from successful runs",
                ],
                "governance": self.memory.governance_report(),
            },
            "observability": {
                "trace_fields": [
                    "handoff",
                    "tool_call",
                    "step",
                    "checkpoint",
                    "memory_write",
                    "verification",
                    "evaluation",
                ],
                "stores": ["telemetry", "run checkpoints", "run trace", "run ledger"],
            },
            "evaluation": {
                "metrics": [
                    "plan_coverage",
                    "retrieval_hit_rate",
                    "private_retrieval_hit_rate",
                    "evidence_sufficiency",
                    "tool_selection_coverage",
                    "query_rewrite_count",
                    "source_quality_score",
                    "context_precision",
                    "context_recall",
                    "faithfulness_proxy",
                    "citation_precision",
                    "citation_source_coverage",
                    "source_diversity",
                    "unsupported_sections",
                ],
                "purpose": "Make RAG, source, and citation quality inspectable without pretending local demo metrics are a labeled benchmark.",
                "source_quality_handling": "Open Deep Research keeps source quality as an evaluator concern; this repo keeps source quality in evaluation instead of adding a standalone runtime search-filtering layer.",
                "llm_judge_artifact": "scripts/run_llm_judge_eval.py provides an optional Open Deep Research-style judge report for saved demo artifacts.",
            },
            "quality_gates": [
                "final reports must attach evidence",
                "plan items are tracked for coverage",
                "citations are assembled into a source index",
                "source diversity and low-confidence reports are flagged",
                "checkpoints and telemetry support replay and inspection",
            ],
            "storage": {
                "backend": "sqlite",
                "path": str(self.storage.path),
                "persisted_objects": ["documents", "memory_records", "research_jobs", "research_runs"],
            },
        }

    def replay(self, run_id: str) -> ResearchRun | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        return self.run(run.request.model_copy())

    def _ensure_job_executor(self) -> ThreadPoolExecutor:
        with self._job_lock:
            if self._job_executor is None:
                self._job_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="arc-job",
                )
            return self._job_executor

    def _submit_celery_job(self, job_id: str) -> bool:
        try:
            from .celery_app import celery_app

            celery_app.send_task("agentic_research_copilot.execute_job", args=[job_id])
        except Exception as exc:
            self.telemetry.emit(
                "job.celery_enqueue_failed",
                "Fell back to in-process queue.",
                job_id=job_id,
                error=str(exc),
            )
            return False
        self.telemetry.emit("job.enqueued_celery", "Submitted job to Celery.", job_id=job_id)
        return True

    def _execute_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        if job.cancel_requested or job.status == "cancelled":
            self._mark_job_cancelled(job, message="Job was cancelled before execution started.")
            return

        started_at = datetime.now(timezone.utc).isoformat()
        running_job = job.model_copy(
            update={
                "status": "running",
                "started_at": started_at,
                "error": None,
            }
        )
        self._record_job(running_job)
        self.telemetry.emit("job.started", running_job.request.topic, job_id=job_id)

        max_attempts = max(1, running_job.max_attempts)
        last_error: str | None = None
        run = None
        for attempt in range(1, max_attempts + 1):
            latest_job = self.get_job(job_id) or running_job
            if latest_job.cancel_requested or latest_job.status == "cancelled":
                self._mark_job_cancelled(
                    latest_job,
                    message="Job was cancelled before the next attempt started.",
                )
                return
            attempt_job = latest_job.model_copy(
                update={
                    "status": "running",
                    "attempts": attempt,
                    "started_at": latest_job.started_at or started_at,
                    "error": None,
                }
            )
            self._record_job(attempt_job)
            try:
                run = self.run(attempt_job.request.model_copy(), job_id=job_id)
                break
            except Exception as exc:  # pragma: no cover - exercised through integration failures
                last_error = str(exc)
                self.telemetry.emit(
                    "job.attempt_failed",
                    last_error,
                    job_id=job_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                if attempt < max_attempts:
                    self._record_job(attempt_job.model_copy(update={"error": last_error}))
                    continue
                finished_at = datetime.now(timezone.utc).isoformat()
                failed_job = attempt_job.model_copy(
                    update={
                        "status": "failed",
                        "error": last_error,
                        "finished_at": finished_at,
                    }
                )
                self._record_job(failed_job)
                self.telemetry.emit("job.failed", last_error, job_id=job_id)
                return

        if run is None:
            finished_at = datetime.now(timezone.utc).isoformat()
            failed_job = running_job.model_copy(
                update={
                    "status": "failed",
                    "error": last_error or "Research job did not produce a run.",
                    "finished_at": finished_at,
                }
            )
            self._record_job(failed_job)
            self.telemetry.emit("job.failed", failed_job.error or "", job_id=job_id)
            return

        finished_at = datetime.now(timezone.utc).isoformat()
        latest_job = self.get_job(job_id) or running_job
        if latest_job.cancel_requested or latest_job.status == "cancelled":
            self._mark_job_cancelled(
                latest_job,
                run_id=run.run_id,
                message="Cancellation was requested; run artifact was saved before the worker stopped.",
            )
            return

        if running_job.timeout_seconds and run.duration_ms and run.duration_ms > running_job.timeout_seconds * 1000:
            self.telemetry.emit(
                "job.timeout_exceeded",
                "Run exceeded the configured timeout after completion.",
                job_id=job_id,
                run_id=run.run_id,
                timeout_seconds=running_job.timeout_seconds,
                duration_ms=run.duration_ms,
            )

        completed_status = "completed" if run.status == "completed" else "failed"
        completed_job = latest_job.model_copy(
            update={
                "status": completed_status,
                "run_id": run.run_id,
                "finished_at": finished_at,
                "error": None if completed_status == "completed" else run.failure_reason,
            }
        )
        self._record_job(completed_job)
        self.telemetry.emit(
            "job.completed" if completed_status == "completed" else "job.failed",
            run.request.topic,
            job_id=job_id,
            run_id=run.run_id,
        )

    def _mark_job_cancelled(
        self,
        job: ResearchJob,
        *,
        message: str,
        run_id: str | None = None,
    ) -> ResearchJob:
        finished_at = datetime.now(timezone.utc).isoformat()
        cancelled = job.model_copy(
            update={
                "status": "cancelled",
                "run_id": run_id or job.run_id,
                "error": message,
                "cancel_requested": True,
                "finished_at": finished_at,
            }
        )
        self._record_job(cancelled)
        self.telemetry.emit("job.cancelled", message, job_id=job.job_id, run_id=run_id)
        return cancelled

    def _record_job(self, job: ResearchJob) -> ResearchJob:
        with self._job_lock:
            self.jobs.record(job)
            self.storage.save_job(job)
        return job

    def _restore_state(self) -> None:
        self.documents.extend(self.storage.load_documents())
        self.memory.extend(self.storage.load_memory())
        restored_jobs: list[ResearchJob] = []
        for job in self.storage.load_jobs():
            if job.cancel_requested and job.status in {"queued", "running"}:
                interrupted = job.model_copy(
                    update={
                        "status": "cancelled",
                        "error": job.error or "Job cancellation was restored after process restart.",
                        "finished_at": job.finished_at or datetime.now(timezone.utc).isoformat(),
                    }
                )
                self.storage.save_job(interrupted)
                restored_jobs.append(interrupted)
            elif self.settings.job_queue_backend == "celery" and job.status in {"queued", "running"}:
                restored_jobs.append(job)
            elif job.status in {"queued", "running"}:
                interrupted = job.model_copy(
                    update={
                        "status": "failed",
                        "error": "Job was interrupted before completion.",
                        "finished_at": job.finished_at or datetime.now(timezone.utc).isoformat(),
                    }
                )
                self.storage.save_job(interrupted)
                restored_jobs.append(interrupted)
            else:
                restored_jobs.append(job)
        self.jobs.extend(restored_jobs)
        self.ledger.extend(self.storage.load_runs())

    def _seed_reference_knowledge(self) -> None:
        self._ensure_seed_document(
            title="Project README",
            source="README.md",
            snippet=(
                "AI Research Copilot turns complex questions into planned, cited, traceable, "
                "and reviewable research reports."
            ),
            content=(
                "The product uses LangGraph + Agentic RAG to plan sub-questions, run an ODR-style "
                "research supervisor, emit ConductResearch calls with selected tools and query rewrites, "
                "route between external search, vector_retrieval, memory_recall, and hybrid evidence, "
                "verify citations, evaluate RAG quality, persist trace and replay artifacts, and support "
                "OpenAI-compatible chat providers, Qwen embeddings, Tavily search, and deterministic test doubles."
            ),
            metadata={"kind": "project_overview"},
        )
        self._ensure_seed_document(
            title="Architecture overview",
            source="docs/architecture.md",
            snippet="Build an AI Research Copilot that can plan, search, ground, remember, verify, evaluate, and report.",
            content=(
                "The architecture centers on a LangGraph StateGraph supervisor, layered memory, "
                "planner, ODR-style research_supervisor, concurrent researcher/retriever workers, Qdrant dense "
                "vectors, indexing-time contextual retrieval prefixes, SQLite FTS5 BM25 keyword retrieval, "
                "RRF/DBSF hybrid fusion, Qwen/DashScope reranking with deterministic fallback, "
                "reporter, Verifier, Evaluator, SQLite checkpoints, trace replay, source quality, "
                "citation precision, evidence sufficiency, context precision, context recall, "
                "faithfulness proxy, queued jobs, retry, and cancelled states."
            ),
            metadata={"kind": "architecture"},
        )
        self._ensure_seed_document(
            title="Source map",
            source="docs/source-map.md",
            snippet=(
                "open_deep_research contributes planning, parallel research, citations, and report generation; "
                "PraisonAI contributes memory, handoff, observability, and workflow ideas."
            ),
            content=(
                "The source map explains which upstream ideas are reused for planning, memory, "
                "contextual grounding, observability, and original API glue."
            ),
            metadata={"kind": "source_map"},
        )
        self._ensure_seed_document(
            title="Hardening roadmap",
            source="docs/hardening-roadmap.md",
            snippet=(
                "Resume-safe hardening priorities: expand evaluation, add a reranker interface, "
                "keep source quality as evaluation, and document checkpoint boundaries."
            ),
            content=(
                "The hardening roadmap says source quality remains evaluation-side rather than a "
                "runtime hard filter because the Open Deep Research reference treats source quality "
                "as evaluator output. The useful v1 boundary is a 12-case regression set, a "
                "Qwen/DashScope reranker with deterministic fallback for offline tests, clear "
                "single-node SQLite checkpoint/replay boundaries, and SQLite-backed job/run status "
                "visibility across the API and local worker. Streaming, auth, and multi-tenancy are "
                "deferred because this is a personal research copilot, not a SaaS platform."
            ),
            metadata={"kind": "hardening_roadmap"},
        )
        if not any(record.key == "project:positioning" for record in self.memory.list()):
            self.memory.add_fact(
                key="project:positioning",
                value=(
                    "This is a clean-room scaffold inspired by open source references, intended "
                    "for learning multi-agent research, contextual grounding, memory, and observability."
                ),
                tags=["project", "positioning", "resume"],
                metadata={"kind": "project_note"},
            )

    def run(self, request: ResearchRequest, *, job_id: str | None = None) -> ResearchRun:
        if self.settings.orchestration_runtime == "langgraph":
            from .graph_runtime import LangGraphResearchRuntime

            return LangGraphResearchRuntime(self).run(request, job_id=job_id)
        return self._run_custom_workflow(request, job_id=job_id)

    def _run_custom_workflow(self, request: ResearchRequest, *, job_id: str | None = None) -> ResearchRun:
        run_id = str(uuid.uuid4())
        start = datetime.now(timezone.utc)
        self.telemetry.emit("run.start", request.topic, run_id=run_id, job_id=job_id, depth=request.depth)

        checkpoints: list[RunCheckpoint] = []
        trace: list[RunTraceEvent] = []
        handoffs: list[AgentHandoff] = []
        revision_notes: list[str] = []
        revision_count = 0
        failure_reason: str | None = None
        final_status = "completed"
        final_research_brief: str | None = None
        final_corpus_profile: CorpusProfile | None = None
        final_supervisor_decision: SupervisorDecisionContract | None = None
        final_plan: list = []
        final_search_queries: list[SearchQuery] = []
        final_retrieval_routes = []
        final_notes: list[ResearchNote] = []
        final_evidence: list[EvidenceItem] = []
        final_web_hits: list[EvidenceItem] = []
        final_document_hits: list[EvidenceItem] = []
        final_report = None
        final_evaluation = None
        final_issues: list[str] = []

        def checkpoint(stage: str, payload: dict[str, object]) -> None:
            checkpoints.append(RunCheckpoint(run_id=run_id, stage=stage, payload=payload))
            self.telemetry.emit("checkpoint.created", stage, run_id=run_id, job_id=job_id, step=stage)
            trace.append(
                RunTraceEvent(
                    kind="checkpoint",
                    actor="supervisor",
                    message=stage,
                    step=stage,
                    status="completed",
                    metadata=payload,
                )
            )

        def append_trace(
            *,
            kind: str,
            actor: str,
            message: str,
            step: str | None = None,
            status: str = "completed",
            from_agent: str | None = None,
            to_agent: str | None = None,
            handoff: AgentHandoff | None = None,
            tool_name: str | None = None,
            provider: str | None = None,
            model: str | None = None,
            tokens_in: int = 0,
            tokens_out: int = 0,
            cost_usd: float = 0.0,
            latency_ms: int = 0,
            **metadata: object,
        ) -> None:
            event = RunTraceEvent(
                kind=kind,
                actor=actor,
                message=message,
                step=step,
                status=status,  # type: ignore[arg-type]
                from_agent=from_agent,
                to_agent=to_agent,
                handoff=handoff,
                tool_name=tool_name,
                provider=provider,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                metadata=metadata,
            )
            trace.append(event)
            self.telemetry.emit(
                kind,
                message,
                run_id=run_id,
                job_id=job_id,
                actor=actor,
                step=step,
                status=status,
                from_agent=from_agent,
                to_agent=to_agent,
                tool_name=tool_name,
                provider=provider,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                **metadata,
            )

        def record_handoff(from_agent: str, to_agent: str, step: str, reason: str, revision: int = 0) -> AgentHandoff:
            handoff = AgentHandoff(
                from_agent=from_agent,
                to_agent=to_agent,
                step=step,
                reason=reason,
                revision=revision,
            )
            handoffs.append(handoff)
            append_trace(
                kind="handoff",
                actor=from_agent,
                message=reason,
                step=step,
                from_agent=from_agent,
                to_agent=to_agent,
                handoff=handoff,
                revision=revision,
            )
            return handoff

        checkpoint(
            "supervisor.start",
            {
                "topic": request.topic,
                "depth": request.depth,
                "max_revisions": request.max_revisions,
            },
        )

        memory_records = self._recall_memory_context(request, run_id) if request.use_memory else []
        memory_hits = self._memory_records_to_evidence(memory_records)
        checkpoint(
            "memory.recalled",
            {
                "memory_hits": len(memory_hits),
                "layers": sorted({record.layer for record in memory_records}),
                "topic": request.topic,
            },
        )

        while True:
            corpus_profile = self.documents.profile()
            final_corpus_profile = corpus_profile
            record_handoff("supervisor", "planner", "planning", "Build the plan and research brief.", revision_count)
            planner_contract = self.planner.draft(
                request,
                corpus_profile=corpus_profile,
                memory_records=memory_records,
                revision_count=revision_count,
                revision_notes=revision_notes,
            )
            planner_usage = self.planner.last_usage
            research_brief = planner_contract.research_brief
            plan = [item.model_copy() for item in planner_contract.plan]
            route_hints = self.router.build_routes(request, research_brief, plan, corpus_profile)

            final_research_brief = research_brief
            final_plan = plan

            append_trace(
                kind="step",
                actor="planner",
                message="Generated the research plan.",
                step="planning",
                provider=getattr(planner_usage, "provider", None),
                model=getattr(planner_usage, "model", None),
                tokens_in=getattr(planner_usage, "prompt_tokens", 0),
                tokens_out=getattr(planner_usage, "completion_tokens", 0),
                latency_ms=getattr(planner_usage, "latency_ms", 0),
                plan_count=len(plan),
                revision_count=revision_count,
            )

            checkpoint(
                "plan.generated",
                {
                    "research_brief": research_brief,
                    "plan_count": len(plan),
                    "document_count": corpus_profile.document_count,
                    "model_provider": getattr(planner_usage, "provider", None),
                    "model_name": getattr(planner_usage, "model", None),
                    "revision_count": revision_count,
                },
            )

            record_handoff(
                "planner",
                "research_supervisor",
                "supervisor.decision",
                "Decide which research units to delegate and which evidence tools each unit should use.",
                revision_count,
            )
            supervisor_decision = self.supervisor_agent.decide(
                request,
                research_brief=research_brief,
                plan=plan,
                retrieval_routes=route_hints,
                corpus_profile=corpus_profile,
                memory_records=memory_records,
                revision_count=revision_count,
                revision_notes=revision_notes,
            )
            supervisor_usage = self.supervisor_agent.last_usage
            retrieval_routes = self._routes_from_supervisor_decision(
                request=request,
                research_brief=research_brief,
                plan=plan,
                supervisor_decision=supervisor_decision,
                route_hints=route_hints,
                corpus_profile=corpus_profile,
            )
            search_queries = self.workflow.build_queries(plan, retrieval_routes, revision_count=revision_count)
            final_supervisor_decision = supervisor_decision
            final_search_queries = search_queries
            final_retrieval_routes = retrieval_routes

            append_trace(
                kind="step",
                actor="research_supervisor",
                message="Issued ODR-style supervisor tool calls.",
                step="supervisor.decision",
                provider=getattr(supervisor_usage, "provider", None),
                model=getattr(supervisor_usage, "model", None),
                tokens_in=getattr(supervisor_usage, "prompt_tokens", 0),
                tokens_out=getattr(supervisor_usage, "completion_tokens", 0),
                latency_ms=getattr(supervisor_usage, "latency_ms", 0),
                tool_calls=[call.model_dump() for call in supervisor_decision.tool_calls],
                completion_criteria=supervisor_decision.completion_criteria,
                max_concurrent_research_units=supervisor_decision.max_concurrent_research_units,
                query_count=len(search_queries),
                revision_count=revision_count,
            )
            checkpoint(
                "supervisor.decision",
                {
                    "reflection": supervisor_decision.reflection,
                    "tool_calls": [call.model_dump() for call in supervisor_decision.tool_calls],
                    "completion_criteria": supervisor_decision.completion_criteria,
                    "max_concurrent_research_units": supervisor_decision.max_concurrent_research_units,
                    "model_provider": getattr(supervisor_usage, "provider", None),
                    "model_name": getattr(supervisor_usage, "model", None),
                    "revision_count": revision_count,
                },
            )
            checkpoint(
                "routing.generated",
                {
                    "route_count": len(retrieval_routes),
                    "external_routes": sum(1 for route in retrieval_routes if route.mode == "external"),
                    "internal_routes": sum(1 for route in retrieval_routes if route.mode == "internal"),
                    "hybrid_routes": sum(1 for route in retrieval_routes if route.mode == "hybrid"),
                    "tool_selections": {
                        route.plan_item_id: route.selected_tools
                        for route in retrieval_routes
                    },
                    "query_rewrite_count": sum(
                        len(route.web_queries) + len(route.internal_queries)
                        for route in retrieval_routes
                    ),
                    "corpus_profile": corpus_profile.model_dump(),
                },
            )

            web_hits: list[EvidenceItem] = []
            document_hits: list[EvidenceItem] = []
            notes: list[ResearchNote] = []
            evidence: list[EvidenceItem] = list(memory_hits)
            route_lookup = {route.plan_item_id: route for route in retrieval_routes}

            for call in supervisor_decision.tool_calls:
                if call.name == "think_tool":
                    append_trace(
                        kind="tool_call",
                        actor="research_supervisor",
                        message=call.reflection or call.rationale,
                        step="supervisor.think",
                        tool_name="think_tool",
                        rationale=call.rationale,
                    )
                    continue
                if call.name == "ResearchComplete":
                    append_trace(
                        kind="tool_call",
                        actor="research_supervisor",
                        message=call.reflection or call.rationale,
                        step="supervisor.complete.criteria",
                        tool_name="ResearchComplete",
                        completion_criteria=supervisor_decision.completion_criteria,
                    )
                    continue
                if call.name != "ConductResearch":
                    continue

                for plan_item_id in call.plan_item_ids:
                    route = route_lookup.get(plan_item_id)
                    item = next((candidate for candidate in plan if candidate.id == plan_item_id), None)
                    if route is None or item is None:
                        continue
                    item.status = "running"
                    record_handoff("research_supervisor", "researcher", f"research.{item.id}", route.reason, revision_count)
                    append_trace(
                        kind="tool_call",
                        actor="research_supervisor",
                        message=call.research_topic or item.question,
                        step=f"supervisor.conduct.{item.id}",
                        tool_name="ConductResearch",
                        plan_item_id=item.id,
                        rationale=call.rationale,
                        retrieval_mode=route.mode,
                        selected_tools=route.selected_tools,
                        web_queries=route.web_queries,
                        internal_queries=route.internal_queries,
                        min_evidence=route.min_evidence,
                        min_sources=route.min_sources,
                    )
                    if route.mode in {"internal", "hybrid"} and request.include_private_docs and corpus_profile.has_private_docs:
                        record_handoff("researcher", "retriever", f"retrieve.{item.id}", route.reason, revision_count)

            checkpoint(
                "research.parallel.started",
                {
                    "worker_count": min(max(1, self.settings.research_max_workers), max(1, len(plan))),
                    "plan_count": len(plan),
                },
            )
            research_results = self._research_plan_items(
                request=request,
                plan=plan,
                route_lookup=route_lookup,
                corpus_profile=corpus_profile,
                research_brief=research_brief,
                supervisor_decision=supervisor_decision,
            )

            for item in plan:
                route = route_lookup.get(item.id)
                result = research_results.get(item.id)
                if route is None or result is None:
                    item.status = "done"
                    continue

                web_evidence = result.web_evidence
                document_evidence = result.document_evidence
                web_hits.extend(web_evidence)
                document_hits.extend(document_evidence)

                if route.mode in {"external", "hybrid"}:
                    append_trace(
                        kind="tool_call",
                        actor="researcher",
                        message=f"Collected {len(web_evidence)} external evidence items.",
                        step=f"research.{item.id}.web",
                        tool_name="web_search",
                        provider=self.settings.search_provider,
                        latency_ms=result.web_latency_ms,
                        result_count=len(web_evidence),
                        query=route.web_query or item.search_query or item.question,
                        queries=route.web_queries,
                        parallel=True,
                    )

                if route.mode in {"internal", "hybrid"} and request.include_private_docs and corpus_profile.has_private_docs:
                    append_trace(
                        kind="tool_call",
                        actor="retriever",
                        message=f"Retrieved {len(document_evidence)} contextual evidence items.",
                        step=f"retrieve.{item.id}",
                        tool_name="qdrant_retrieval",
                        provider=corpus_profile.vector_backend,
                        latency_ms=result.document_latency_ms,
                        result_count=len(document_evidence),
                        query=route.internal_query or item.search_query or item.question,
                        queries=route.internal_queries,
                        backend=corpus_profile.vector_backend,
                        parallel=True,
                    )

                item_evidence = self._dedupe_evidence([*web_evidence, *document_evidence])
                item.evidence_count = len(item_evidence)
                item.status = "done"
                evidence.extend(item_evidence)
                note = result.note or self.workflow.compress_findings(item, item_evidence, route)
                notes.append(note)
                append_trace(
                    kind="step",
                    actor="researcher",
                    message=f"Completed {item.question}",
                    step=f"research.{item.id}",
                    status="completed",
                    evidence_count=item.evidence_count,
                    note_confidence=note.confidence,
                    sufficiency_score=note.sufficiency_score,
                    sufficiency_gaps=note.gaps,
                    follow_up_queries=note.follow_up_queries,
                    research_iteration_count=len(note.research_iterations),
                    research_iterations=note.research_iterations,
                    completed_reason=note.completed_reason,
                    retrieval_mode=route.mode,
                    route_reason=route.reason,
                    selected_tools=route.selected_tools,
                    min_evidence=route.min_evidence,
                    min_sources=route.min_sources,
                )
                checkpoint(
                    f"research.{item.id}",
                    {
                        "question": item.question,
                        "evidence_count": item.evidence_count,
                        "note_confidence": note.confidence,
                        "sufficiency_score": note.sufficiency_score,
                        "sufficiency_gaps": note.gaps,
                        "follow_up_queries": note.follow_up_queries,
                        "research_iteration_count": len(note.research_iterations),
                        "research_iterations": note.research_iterations,
                        "completed_reason": note.completed_reason,
                        "retrieval_mode": route.mode,
                        "route_reason": route.reason,
                        "selected_tools": route.selected_tools,
                        "min_evidence": route.min_evidence,
                        "min_sources": route.min_sources,
                        "web_evidence_count": len(web_evidence),
                        "document_evidence_count": len(document_evidence),
                        "retrieval_strategy": "light_rag_inspired_parent_child_dense_bm25_graph_rerank",
                        "parallel": True,
                    },
                )

            runtime_evidence = self._build_run_artifact_evidence(
                run_id=run_id,
                plan=plan,
                search_queries=search_queries,
                retrieval_routes=retrieval_routes,
                web_hits=web_hits,
                memory_hits=memory_hits,
                document_hits=document_hits,
                notes=notes,
                revision_count=revision_count,
            )
            evidence = self._dedupe_evidence([*evidence, runtime_evidence])
            final_web_hits = web_hits
            final_document_hits = document_hits
            final_notes = notes
            final_evidence = evidence

            sections = self._build_sections(
                request,
                research_brief,
                plan,
                retrieval_routes,
                evidence,
                web_hits,
                memory_hits,
                document_hits,
                notes,
                search_queries,
            )
            confidence = self._estimate_confidence(request, evidence, memory_hits, document_hits, plan)
            record_handoff("supervisor", "reporter", "reporting", "Assemble the citation-backed answer.", revision_count)
            report_evidence = self._rank_evidence_for_report(evidence)
            report = self.reporter.build_report(request.topic, sections, report_evidence, confidence)
            reporter_usage = self.reporter.last_usage
            append_trace(
                kind="step",
                actor="reporter",
                message="Assembled the citation-backed report.",
                step="reporting",
                provider=getattr(reporter_usage, "provider", None),
                model=getattr(reporter_usage, "model", None),
                tokens_in=getattr(reporter_usage, "prompt_tokens", 0),
                tokens_out=getattr(reporter_usage, "completion_tokens", 0),
                latency_ms=getattr(reporter_usage, "latency_ms", 0),
                section_count=len(sections),
                source_count=report.source_count,
            )
            report = report.model_copy(
                update={"source_index": self.workflow.format_sources(self._rank_evidence_for_report(evidence))}
            )
            assessment = self.verifier.assess(
                report,
                evidence,
                plan,
                revision_count=revision_count,
                max_revisions=request.max_revisions,
            )
            evaluation = self.evaluator.evaluate(
                report=report,
                plan=plan,
                evidence=evidence,
                document_hits=document_hits,
                retrieval_routes=retrieval_routes,
            )
            verifier_usage = self.verifier.last_usage
            final_report = report
            final_evaluation = evaluation
            final_issues = assessment.issues

            append_trace(
                kind="verification",
                actor="verifier",
                message="Verified the report.",
                step="verification",
                provider=getattr(verifier_usage, "provider", None),
                model=getattr(verifier_usage, "model", None),
                tokens_in=getattr(verifier_usage, "prompt_tokens", 0),
                tokens_out=getattr(verifier_usage, "completion_tokens", 0),
                latency_ms=getattr(verifier_usage, "latency_ms", 0),
                issue_count=len(assessment.issues),
                critical_issue_count=len(assessment.critical_issues),
                coverage_score=assessment.coverage_score,
                should_revise=assessment.should_revise,
            )
            append_trace(
                kind="evaluation",
                actor="evaluator",
                message="Evaluated RAG and citation quality.",
                step="evaluation",
                plan_coverage=evaluation.plan_coverage,
                retrieval_hit_rate=evaluation.retrieval_hit_rate,
                private_retrieval_hit_rate=evaluation.private_retrieval_hit_rate,
                evidence_sufficiency=evaluation.evidence_sufficiency,
                tool_selection_coverage=evaluation.tool_selection_coverage,
                query_rewrite_count=evaluation.query_rewrite_count,
                source_quality_score=evaluation.source_quality_score,
                context_precision=evaluation.context_precision,
                context_recall=evaluation.context_recall,
                faithfulness_proxy=evaluation.faithfulness_proxy,
                citation_precision=evaluation.citation_precision,
                citation_source_coverage=evaluation.citation_source_coverage,
                source_diversity=evaluation.source_diversity,
                insufficient_plan_items=evaluation.insufficient_plan_items,
                passed=evaluation.passed,
                notes=evaluation.notes,
            )
            checkpoint(
                "report.verified",
                {
                    "confidence": confidence,
                    "issue_count": len(assessment.issues),
                    "critical_issue_count": len(assessment.critical_issues),
                    "source_count": report.source_count,
                    "coverage_score": assessment.coverage_score,
                    "should_revise": assessment.should_revise,
                },
            )
            checkpoint(
                "rag.evaluated",
                {
                    "plan_coverage": evaluation.plan_coverage,
                    "retrieval_hit_rate": evaluation.retrieval_hit_rate,
                    "private_retrieval_hit_rate": evaluation.private_retrieval_hit_rate,
                    "evidence_sufficiency": evaluation.evidence_sufficiency,
                    "tool_selection_coverage": evaluation.tool_selection_coverage,
                    "query_rewrite_count": evaluation.query_rewrite_count,
                    "source_quality_score": evaluation.source_quality_score,
                    "context_precision": evaluation.context_precision,
                    "context_recall": evaluation.context_recall,
                    "faithfulness_proxy": evaluation.faithfulness_proxy,
                    "citation_precision": evaluation.citation_precision,
                    "citation_source_coverage": evaluation.citation_source_coverage,
                    "source_diversity": evaluation.source_diversity,
                    "insufficient_plan_items": evaluation.insufficient_plan_items,
                    "passed": evaluation.passed,
                    "notes": evaluation.notes,
                },
            )

            quality_should_revise = not evaluation.passed and bool(evaluation.notes)
            if (assessment.should_revise or quality_should_revise) and revision_count < request.max_revisions:
                revision_notes = (
                    assessment.critical_issues
                    or assessment.issues
                    or evaluation.notes
                    or ["tighten citations and evidence coverage"]
                )
                revision_count += 1
                record_handoff(
                    "supervisor",
                    "planner",
                    "revision",
                    assessment.revision_reason or "; ".join(evaluation.notes[:2]) or "Verification requested another pass.",
                    revision_count,
                )
                checkpoint(
                    "supervisor.revision_requested",
                    {
                        "revision_count": revision_count,
                        "revision_notes": revision_notes,
                        "revision_reason": assessment.revision_reason or "; ".join(evaluation.notes[:2]),
                    },
                )
                continue

            usable_report = bool(report.citations) and evaluation.passed
            if (assessment.should_revise or quality_should_revise) and revision_count >= request.max_revisions and not usable_report:
                final_status = "failed"
                failure_reason = assessment.revision_reason or "; ".join(evaluation.notes[:2]) or "Maximum revision budget reached."
            elif assessment.critical_issues and not report.citations:
                final_status = "failed"
                failure_reason = assessment.revision_reason or "Citation-backed report could not be produced."
            else:
                final_status = "completed"
                failure_reason = None
            break

        end = datetime.now(timezone.utc)
        duration_ms = int((end - start).total_seconds() * 1000)

        if request.use_memory and final_report is not None:
            memory_artifacts = self._build_memory_artifacts(
                request=request,
                run_id=run_id,
                report=final_report,
                revision_count=revision_count,
                status=final_status,
            )
            for record in memory_artifacts:
                self.storage.save_memory(record)
                append_trace(
                    kind="memory_write",
                    actor="memory_manager",
                    message=f"Stored {record.layer} memory {record.key}.",
                    step=f"memory.{record.layer}",
                    layer=record.layer,
                    memory_key=record.key,
                    topic=request.topic,
                    confidence=record.confidence,
                )

        run = ResearchRun(
            run_id=run_id,
            job_id=job_id,
            request=request,
            research_brief=final_research_brief,
            corpus_profile=final_corpus_profile,
            supervisor_decision=final_supervisor_decision,
            plan=final_plan,
            search_queries=final_search_queries,
            retrieval_routes=final_retrieval_routes,
            notes=final_notes,
            evidence=final_evidence,
            web_hits=final_web_hits,
            memory_hits=memory_hits,
            document_hits=final_document_hits,
            checkpoints=checkpoints,
            trace=trace,
            handoffs=handoffs,
            report=final_report,
            evaluation=final_evaluation,
            issues=final_issues,
            status=final_status,
            revision_count=revision_count,
            failure_reason=failure_reason,
            started_at=start.isoformat(),
            finished_at=end.isoformat(),
            duration_ms=duration_ms,
        )
        self.ledger.record(run)
        self.storage.save_run(run)
        self.telemetry.emit(
            "run.finish",
            request.topic,
            run_id=run_id,
            job_id=job_id,
            duration_ms=duration_ms,
            issues=len(final_issues),
            revision_count=revision_count,
            status=final_status,
            failure_reason=failure_reason,
        )
        return run

    def _research_plan_items(
        self,
        *,
        request: ResearchRequest,
        plan: list,
        route_lookup,
        corpus_profile: CorpusProfile,
        research_brief: str,
        supervisor_decision: SupervisorDecisionContract | None = None,
    ) -> dict[str, PlanItemResearchResult]:
        runnable_items = self._conduct_plan_items(plan, route_lookup, supervisor_decision)
        if not runnable_items:
            return {}

        supervisor_worker_limit = (
            supervisor_decision.max_concurrent_research_units
            if supervisor_decision is not None
            else self.settings.research_max_workers
        )
        max_workers = min(
            max(1, self.settings.research_max_workers),
            max(1, supervisor_worker_limit),
            len(runnable_items),
        )
        if max_workers == 1:
            return {
                item.id: self._research_plan_item(
                    request=request,
                    item=item,
                    route=route_lookup[item.id],
                    corpus_profile=corpus_profile,
                    research_brief=research_brief,
                )
                for item in runnable_items
            }

        results: dict[str, PlanItemResearchResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="arc-research") as executor:
            futures = {
                executor.submit(
                    self._research_plan_item,
                    request=request,
                    item=item,
                    route=route_lookup[item.id],
                    corpus_profile=corpus_profile,
                    research_brief=research_brief,
                ): item
                for item in runnable_items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results[item.id] = future.result()
                except Exception as exc:  # pragma: no cover - defensive guard for provider failures
                    if self.settings.strict_providers:
                        raise
                    results[item.id] = PlanItemResearchResult(
                        item_id=item.id,
                        note=ResearchNote(
                            plan_item_id=item.id,
                            question=item.question,
                            finding=f"Research task failed: {exc}",
                            confidence=0.05,
                        ),
                    )
        return results

    def _conduct_plan_items(
        self,
        plan: list,
        route_lookup,
        supervisor_decision: SupervisorDecisionContract | None,
    ) -> list:
        base_items = [item for item in plan if item.id in route_lookup]
        if supervisor_decision is None:
            return base_items
        delegated_ids: list[str] = []
        seen: set[str] = set()
        for call in supervisor_decision.tool_calls:
            if call.name != "ConductResearch":
                continue
            for plan_item_id in call.plan_item_ids:
                if plan_item_id in route_lookup and plan_item_id not in seen:
                    seen.add(plan_item_id)
                    delegated_ids.append(plan_item_id)
        if not delegated_ids:
            return base_items
        item_lookup = {item.id: item for item in base_items}
        return [item_lookup[item_id] for item_id in delegated_ids if item_id in item_lookup]

    def _research_plan_item(
        self,
        *,
        request: ResearchRequest,
        item,
        route,
        corpus_profile: CorpusProfile,
        research_brief: str,
    ) -> PlanItemResearchResult:
        web_evidence: list[EvidenceItem] = []
        document_evidence: list[EvidenceItem] = []
        web_latency_ms = 0
        document_latency_ms = 0
        research_iterations: list[dict] = []
        researcher_completed_reason: str | None = None
        researcher_follow_up_queries: list[str] = []

        should_run_researcher_loop = route.mode in {"external", "hybrid"} or (
            "mcp_tool" in getattr(route, "selected_tools", []) and self.mcp_tool is not None
        )
        if should_run_researcher_loop:
            start_collect = datetime.now(timezone.utc)
            web_queries = (
                getattr(route, "web_queries", None)
                or getattr(route, "internal_queries", None)
                or [route.web_query or route.internal_query or item.search_query or item.question]
            )
            collection = self.researcher.collect_iterative(
                item,
                web_queries,
                min_evidence=max(1, route.min_evidence),
                min_sources=max(1, route.min_sources),
                max_iterations=self.settings.research_max_iterations,
            )
            web_evidence = self._dedupe_evidence(collection.evidence)
            research_iterations.extend(collection.iterations)
            researcher_completed_reason = collection.completed_reason
            researcher_follow_up_queries.extend(collection.follow_up_queries)
            web_latency_ms = int((datetime.now(timezone.utc) - start_collect).total_seconds() * 1000)

        if route.mode in {"internal", "hybrid"} and request.include_private_docs and corpus_profile.has_private_docs:
            start_collect = datetime.now(timezone.utc)
            for query in (getattr(route, "internal_queries", None) or [route.internal_query or item.search_query or item.question]):
                document_evidence.extend(
                    self.documents.search(
                        query,
                        limit=3,
                        context=research_brief,
                        purpose=item.purpose,
                    )
                )
            document_evidence = self._dedupe_evidence(document_evidence)
            document_latency_ms = int((datetime.now(timezone.utc) - start_collect).total_seconds() * 1000)

        item_evidence = self._dedupe_evidence([*web_evidence, *document_evidence])
        note = self.workflow.compress_findings(item, item_evidence, route)
        if research_iterations or researcher_completed_reason or researcher_follow_up_queries:
            note = note.model_copy(
                update={
                    "research_iterations": research_iterations,
                    "completed_reason": researcher_completed_reason,
                    "follow_up_queries": self._unique_strings(
                        [*note.follow_up_queries, *researcher_follow_up_queries]
                    ),
                }
            )
        return PlanItemResearchResult(
            item_id=item.id,
            web_evidence=web_evidence,
            document_evidence=document_evidence,
            note=note,
            web_latency_ms=web_latency_ms,
            document_latency_ms=document_latency_ms,
        )

    def _recall_memory_context(self, request: ResearchRequest, run_id: str) -> list:
        topic = request.topic.strip()
        records: list = []
        records.extend(self.memory.recall(topic, layer="summary", topic=topic, limit=3))
        records.extend(self.memory.recall(topic, layer="canonical", topic=topic, limit=4))
        records.extend(self.memory.recall(topic, layer="session", run_id=run_id, limit=3))
        records.extend(self.memory.recall(topic, topic=topic, limit=2))
        return self._dedupe_memory_records(records)

    def _build_memory_artifacts(
        self,
        *,
        request: ResearchRequest,
        run_id: str,
        report,
        revision_count: int,
        status: str,
    ) -> list:
        records = [
            self.memory.add_session_note(
                key=f"session:{run_id}:summary",
                value=report.summary,
                tags=["session", request.depth, status],
                run_id=run_id,
                topic=request.topic,
                confidence=report.confidence,
                metadata={
                    "run_id": run_id,
                    "source_count": report.source_count,
                    "revision_count": revision_count,
                    "status": status,
                },
            ),
            self.memory.add_summary(
                key=f"summary:{request.topic}",
                value=report.summary,
                tags=["topic", request.depth, "summary"],
                run_id=run_id,
                topic=request.topic,
                confidence=report.confidence,
                metadata={
                    "run_id": run_id,
                    "source_count": report.source_count,
                    "revision_count": revision_count,
                    "status": status,
                },
            ),
        ]
        if status == "completed" and report.confidence >= 0.6 and report.citations:
            records.append(
                self.memory.add_fact(
                    key=f"fact:{request.topic}",
                    value=self._canonical_fact_from_report(report),
                    tags=["canonical", "fact", request.depth],
                    run_id=run_id,
                    topic=request.topic,
                    confidence=report.confidence,
                    metadata={
                        "run_id": run_id,
                        "source_count": report.source_count,
                        "revision_count": revision_count,
                        "source_index": report.source_index[:5],
                        "write_policy": "completed citation-backed run with confidence >= 0.6",
                        "supporting_source_count": report.source_count,
                    },
                )
            )
        return records

    def _routes_from_supervisor_decision(
        self,
        *,
        request: ResearchRequest,
        research_brief: str,
        plan: list,
        supervisor_decision: SupervisorDecisionContract,
        route_hints: list[RetrievalRoute],
        corpus_profile: CorpusProfile,
    ) -> list[RetrievalRoute]:
        """Materialize ConductResearch calls into executable evidence routes."""
        item_lookup = {item.id: item for item in plan}
        hint_lookup = {route.plan_item_id: route for route in route_hints}
        routes: list[RetrievalRoute] = []
        seen: set[str] = set()

        for call in supervisor_decision.tool_calls:
            if call.name != "ConductResearch":
                continue
            for plan_item_id in call.plan_item_ids:
                item = item_lookup.get(plan_item_id)
                if item is None or plan_item_id in seen:
                    continue
                routes.append(
                    self._route_from_conduct_call(
                        request=request,
                        research_brief=research_brief,
                        item=item,
                        call=call,
                        route_hint=hint_lookup.get(plan_item_id),
                        corpus_profile=corpus_profile,
                    )
                )
                seen.add(plan_item_id)

        for item in plan:
            if not item.requires_research or item.id in seen:
                continue
            route_hint = hint_lookup.get(item.id)
            routes.append(
                route_hint
                or self._route_from_conduct_call(
                    request=request,
                    research_brief=research_brief,
                    item=item,
                    call=None,
                    route_hint=None,
                    corpus_profile=corpus_profile,
                )
            )
        return routes

    def _route_from_conduct_call(
        self,
        *,
        request: ResearchRequest,
        research_brief: str,
        item,
        call,
        route_hint: RetrievalRoute | None,
        corpus_profile: CorpusProfile,
    ) -> RetrievalRoute:
        selected_tools = self._normalize_supervisor_tools(
            getattr(call, "selected_tools", []) if call is not None else [],
            request=request,
            corpus_profile=corpus_profile,
        )
        if not selected_tools and route_hint is not None:
            selected_tools = self._normalize_supervisor_tools(
                route_hint.selected_tools,
                request=request,
                corpus_profile=corpus_profile,
            )
        if not selected_tools:
            selected_tools = ["web_search"]
            if request.use_memory:
                selected_tools.append("memory_recall")
        if "web_search" not in selected_tools and "vector_retrieval" not in selected_tools:
            selected_tools.insert(0, "web_search")

        mode = self._mode_from_tools(selected_tools)
        requested_mode = getattr(call, "mode", None) if call is not None else None
        if requested_mode in {"external", "internal", "hybrid"} and self._mode_is_compatible(requested_mode, selected_tools):
            mode = requested_mode

        web_queries = self._clean_supervisor_queries(getattr(call, "web_queries", []) if call is not None else [])
        internal_queries = self._clean_supervisor_queries(getattr(call, "internal_queries", []) if call is not None else [])
        if not web_queries and route_hint is not None:
            web_queries = list(route_hint.web_queries)
        if not internal_queries and route_hint is not None:
            internal_queries = list(route_hint.internal_queries)

        default_query = self._clean_supervisor_queries(
            [
                getattr(call, "research_topic", None) if call is not None else None,
                item.search_query,
                item.question,
                f"{request.topic} {item.purpose} evidence sources",
            ]
        )
        if mode in {"external", "hybrid"} and not web_queries:
            web_queries = default_query[: self.router.max_query_rewrites]
        if mode in {"internal", "hybrid"} and not internal_queries:
            internal_queries = self._clean_supervisor_queries(
                [
                    f"{request.topic} {research_brief} {item.purpose}",
                    item.search_query,
                    item.question,
                ]
            )[: self.router.max_query_rewrites]
        if mode == "external":
            internal_queries = []
        if mode == "internal":
            web_queries = []

        memory_query = getattr(call, "memory_query", None) if call is not None else None
        if request.use_memory and not memory_query:
            memory_query = route_hint.memory_query if route_hint is not None else f"{request.topic} {item.purpose}"
        if not request.use_memory:
            memory_query = None

        min_evidence = getattr(call, "min_evidence", None) if call is not None else None
        min_sources = getattr(call, "min_sources", None) if call is not None else None
        sufficiency_criteria = list(getattr(call, "sufficiency_criteria", []) if call is not None else [])
        if route_hint is not None:
            min_evidence = min_evidence or route_hint.min_evidence
            min_sources = min_sources or route_hint.min_sources
            sufficiency_criteria = sufficiency_criteria or list(route_hint.sufficiency_criteria)
        min_evidence = int(min_evidence or self.router.min_evidence_per_item)
        min_sources = int(min_sources or (1 if mode == "internal" else self.router.min_source_diversity))
        if not sufficiency_criteria:
            sufficiency_criteria = [
                f"collect at least {min_evidence} evidence items for this delegated research unit",
                f"use at least {min_sources} non-memory source group(s) when available",
                "preserve citations for report assembly",
            ]

        rationale = getattr(call, "rationale", None) if call is not None else None
        if route_hint is not None and not rationale:
            rationale = route_hint.reason
        return RetrievalRoute(
            plan_item_id=item.id,
            mode=mode,
            web_query=web_queries[0] if web_queries else None,
            internal_query=internal_queries[0] if internal_queries else None,
            reason=rationale or "Supervisor delegated this research unit.",
            selected_tools=selected_tools,
            web_queries=web_queries,
            internal_queries=internal_queries,
            memory_query=memory_query,
            min_evidence=min_evidence,
            min_sources=min_sources,
            sufficiency_criteria=sufficiency_criteria,
        )

    def _normalize_supervisor_tools(
        self,
        tools,
        *,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
    ) -> list[str]:
        valid = {"web_search", "vector_retrieval", "memory_recall", "mcp_tool"}
        normalized: list[str] = []
        for tool in tools or []:
            if tool not in valid or tool in normalized:
                continue
            if tool == "memory_recall" and not request.use_memory:
                continue
            if tool == "vector_retrieval" and (not request.include_private_docs or not corpus_profile.has_private_docs):
                continue
            normalized.append(tool)
        return normalized

    def _mode_from_tools(self, tools: list[str]) -> str:
        has_web = "web_search" in tools
        has_vector = "vector_retrieval" in tools
        if has_web and has_vector:
            return "hybrid"
        if has_vector:
            return "internal"
        return "external"

    def _mode_is_compatible(self, mode: str, tools: list[str]) -> bool:
        if mode == "hybrid":
            return "web_search" in tools and "vector_retrieval" in tools
        if mode == "internal":
            return "vector_retrieval" in tools
        return "web_search" in tools

    def _clean_supervisor_queries(self, queries) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if query is None:
                continue
            normalized = " ".join(str(query).split()).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned

    def _build_run_artifact_evidence(
        self,
        *,
        run_id: str,
        plan: list,
        search_queries: list[SearchQuery],
        retrieval_routes,
        web_hits: list[EvidenceItem],
        memory_hits: list[EvidenceItem],
        document_hits: list[EvidenceItem],
        notes: list[ResearchNote],
        revision_count: int,
    ) -> EvidenceItem:
        route_counts = {
            "external": sum(1 for route in retrieval_routes if route.mode == "external"),
            "internal": sum(1 for route in retrieval_routes if route.mode == "internal"),
            "hybrid": sum(1 for route in retrieval_routes if route.mode == "hybrid"),
        }
        tool_counts: dict[str, int] = {}
        for route in retrieval_routes:
            for tool in getattr(route, "selected_tools", []):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        query_rewrite_count = sum(
            len(getattr(route, "web_queries", [])) + len(getattr(route, "internal_queries", []))
            for route in retrieval_routes
        )
        content = (
            f"Run {run_id} generated {len(plan)} plan items, {len(search_queries)} search queries, "
            f"{len(retrieval_routes)} retrieval routes, {len(web_hits)} web hits, {len(memory_hits)} memory hits, "
            f"{len(document_hits)} contextual grounding hits, and {len(notes)} compressed research notes. "
            f"Route mix: external={route_counts['external']}, internal={route_counts['internal']}, "
            f"hybrid={route_counts['hybrid']}. Tool selections={tool_counts}. "
            f"Query rewrites={query_rewrite_count}. Revision count: {revision_count}."
        )
        return EvidenceItem(
            title="Run artifact metrics",
            source="run-ledger",
            kind="run-artifact",
            snippet=content,
            content=content,
            score=1.0,
            metadata={
                "run_id": run_id,
                "plan_count": len(plan),
                "search_query_count": len(search_queries),
                "route_count": len(retrieval_routes),
                "web_hit_count": len(web_hits),
                "memory_hit_count": len(memory_hits),
                "document_hit_count": len(document_hits),
                "note_count": len(notes),
                "revision_count": revision_count,
                "route_counts": route_counts,
                "tool_counts": tool_counts,
                "query_rewrite_count": query_rewrite_count,
            },
        )

    def _canonical_fact_from_report(self, report) -> str:
        if report.highlights:
            return report.highlights[0]
        if report.summary:
            return report.summary
        return report.title

    def _dedupe_memory_records(self, records) -> list:
        deduped = []
        seen: set[tuple[str, str | None, str | None, str | None, str | None]] = set()
        for record in records:
            identity = (record.key, record.run_id, record.session_id, record.topic, record.layer)
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(record)
        return deduped

    def _build_sections(
        self,
        request: ResearchRequest,
        research_brief: str,
        plan: list,
        retrieval_routes,
        evidence: list[EvidenceItem],
        web_hits: list[EvidenceItem],
        memory_hits: list[EvidenceItem],
        document_hits: list[EvidenceItem],
        notes: list[ResearchNote],
        search_queries: list[SearchQuery],
    ) -> list[ReportSection]:
        sections: list[ReportSection] = []
        plan_count = len(plan)
        base_sources = self._source_names(evidence)
        ranked_evidence = self._rank_evidence_for_report(evidence)
        run_artifacts = [item for item in ranked_evidence if item.kind == "run-artifact"]
        external_grounding = [
            item
            for item in ranked_evidence
            if item.kind == "web" and item.source not in {"memory", "internal-note"}
        ]
        project_grounding = self._rank_evidence_for_report(document_hits)[:4] or ranked_evidence[:4]
        non_memory_grounding = [
            item
            for item in ranked_evidence
            if item.kind != "memory" and item.source != "memory"
        ]

        section_specs: list[tuple[str, str, list[EvidenceItem]]] = [
            (
                "Problem framing",
                (
                    f"{research_brief} The current plan splits the topic into {plan_count} steps "
                    "and frames the project as an AI Research Copilot: it starts from a complex question, "
                    "plans the investigation, then uses LangGraph + Agentic RAG to route between public and "
                    "contextual evidence, verify citations, and store reusable memory."
                ),
                project_grounding + external_grounding[:2],
            ),
            (
                "Execution flow",
                (
                    "The LangGraph StateGraph runs as supervisor -> memory -> planner -> research_supervisor -> researcher/retriever "
                    "-> reporter -> verifier/evaluator -> memory. "
                    f"It combines {len(search_queries)} generated queries, {len(retrieval_routes)} retrieval routes, "
                    f"{len(web_hits)} web hits, {len(notes)} compressed notes, "
                    f"{len(memory_hits)} memory hits, and {len(document_hits)} contextual grounding hits "
                    "to keep the result evidence-backed and traceable. The research supervisor emits think_tool, "
                    "ConductResearch, and ResearchComplete decisions; each delegated unit records external search, "
                    "vector_retrieval, memory_recall, tool selection, query rewrite counts, and source-indexed "
                    "checkpoints so the handoff trace can be reviewed after the run. The graph uses single-node "
                    "LangGraph SQLite checkpointing by default, while SQLite also stores durable run traces and replay artifacts; "
                    "the job layer is a single-worker queue with queued, retry, and cancelled states rather than a "
                    "distributed scheduler."
                ),
                run_artifacts + project_grounding[:3],
            ),
            (
                "Contextual grounding",
                (
                    "Memory stores session notes, canonical facts, and topic summaries with confidence metadata, while the "
                    "grounding layer prepends chunk-specific contextual retrieval prefixes before writing to a "
                    "Qdrant-backed dense embedding index plus SQLite FTS5 BM25 keyword search, then uses "
                    "RRF/DBSF hybrid fusion and a Qwen/DashScope reranker that falls back to a "
                    "deterministic rule_diversity_chunk_bonus reranker when no API key is configured. "
                    "Web search stays separate until report assembly, so project context stays traceable and fresh evidence "
                    "remains distinct. The same retrieval contract can still swap in a cross-encoder or another hosted "
                    "reranker later."
                ),
                project_grounding + external_grounding[:2],
            ),
            (
                "Trade-offs and failure modes",
                (
                    "The system does not pretend that every search result is authoritative. Search providers can surface "
                    "tutorials, community discussions, or low-signal pages, so source quality is handled as an evaluation "
                    "and presentation concern: weak sources remain visible in trace artifacts, while the report should "
                    "prefer project documents, official references, papers, and source-backed run artifacts for core "
                    "claims. The single-node queue and SQLite checkpoint design are intentional for a personal research "
                    "assistant; a multi-tenant SaaS would need stronger auth, rate limits, distributed queue operations, "
                    "human review, and production monitoring."
                ),
                run_artifacts + non_memory_grounding[:4],
            ),
            (
                "Verification and next steps",
                (
                    "The verifier checks source diversity, plan coverage, evidence sufficiency, and confidence, while the "
                    "RAG evaluator records retrieval coverage, source quality, citation precision, citation source coverage, "
                    "context precision, context recall, faithfulness proxy, and unsupported sections before the report is "
                    "considered complete. Source quality is evaluation-side instead of a runtime hard filter, matching the "
                    "Open Deep Research reference shape and leaving provider ranking to Tavily, Exa, Perplexity, Brave, "
                    "SerpAPI, arXiv, PubMed, or other configured search providers. Real demos can use an OpenAI-compatible "
                    "chat provider, Qwen embeddings, and Tavily search, while deterministic providers remain the default "
                    "for stable tests. When citations or evidence sufficiency fail, the Verifier and Evaluator trigger the "
                    "revision loop until the revision budget is exhausted. "
                    f"This run recalled {len(memory_hits)} memory items; when memory is thin, the trace makes that gap visible "
                    "instead of hiding it behind fluent output."
                ),
                run_artifacts + non_memory_grounding[:4],
            ),
        ]

        if request.depth == "deep":
            section_specs.append(
                (
                    "Delivery and evaluation",
                    (
                        "A stronger version of the project should expose run replay, evaluation metrics, and provider swaps "
                        "so the copilot can be demonstrated as a usable product instead of a demo. The intended demo path "
                        "uses an OpenAI-compatible chat provider, Qwen embeddings, and Tavily or another real search "
                        "provider, while deterministic providers remain the default for tests."
                    ),
                    evidence[1:7],
                )
            )

        for heading, content, citations in section_specs[: request.max_sections]:
            section_citations = self._dedupe_evidence(citations)
            sections.append(
                ReportSection(
                    heading=heading,
                    content=content,
                    citations=section_citations,
                    evidence_count=len(section_citations),
                    source_summary=self._source_names(section_citations) or base_sources[:3],
                )
            )

        return sections

    def _estimate_confidence(
        self,
        request: ResearchRequest,
        evidence: list[EvidenceItem],
        memory_hits: list[EvidenceItem],
        document_hits: list[EvidenceItem],
        plan: list,
    ) -> float:
        non_internal_sources = {
            item.source
            for item in evidence
            if item.source not in {"internal-note", "memory"}
        }
        score = 0.3
        score += min(0.35, len(evidence) * 0.04)
        score += min(0.12, len(document_hits) * 0.02)
        score += min(0.08, len(memory_hits) * 0.02)
        score += min(0.08, len(non_internal_sources) * 0.03)
        score += min(0.07, len(plan) * 0.01)
        if request.depth == "deep":
            score += 0.03
        if not non_internal_sources:
            score -= 0.08
        return max(0.2, min(score, 0.95))

    def _memory_records_to_evidence(self, records) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for index, record in enumerate(records):
            evidence.append(
                EvidenceItem(
                    title=record.key,
                    source="memory",
                    kind="memory",
                    snippet=record.value,
                    content=record.value,
                    score=max(0.4, 1.0 - index * 0.1),
                    metadata={
                        "tags": record.tags,
                        "layer": record.layer,
                        "run_id": record.run_id,
                        "session_id": record.session_id,
                        "topic": record.topic,
                        "confidence": record.confidence,
                        "created_at": record.created_at,
                        **record.metadata,
                    },
                )
            )
        return evidence

    def _ensure_seed_document(
        self,
        title: str,
        source: str,
        snippet: str,
        content: str,
        metadata: dict[str, object],
    ) -> None:
        if any(doc.title == title and doc.source == source for doc in self.documents.list()):
            return
        self.documents.add(
            title=title,
            source=source,
            snippet=snippet,
            content=content,
            metadata=metadata,
        )

    def _dedupe_evidence(self, items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
        deduped: list[EvidenceItem] = []
        seen: set[str] = set()
        for item in items:
            key = item.url or f"{item.kind}:{item.source}:{item.title}:{item.snippet or item.content or ''}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _unique_strings(self, values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = value.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
        return result

    def _rank_evidence_for_report(self, items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
        return sorted(
            self._dedupe_evidence(items),
            key=lambda item: (-self._report_evidence_weight(item), -(item.score or 0.0), item.title.lower()),
        )

    @staticmethod
    def _report_evidence_weight(item: EvidenceItem) -> float:
        source = (item.source or "").lower()
        url = (item.url or "").lower()
        metadata_kind = str(item.metadata.get("kind", "")).lower()
        score = 0.0
        if item.kind in {"document-chunk", "run-artifact", "paper", "web-summary"}:
            score += 3.0
        if metadata_kind in {"official_reference", "evaluation_reference", "reference_design", "architecture", "source_map"}:
            score += 2.0
        if any(domain in source or domain in url for domain in ("docs.", "github.com", "qdrant.tech", "langchain.com", "langchain-ai", "ragas.io", "arxiv.org", "pubmed")):
            score += 1.5
        if item.url:
            score += 0.5
        if item.kind == "memory" or source == "memory":
            score -= 1.5
        if any(domain in source or domain in url for domain in ("youtube.com", "youtu.be", "reddit.com")):
            score -= 1.0
        return score

    def _source_names(self, items: Iterable[EvidenceItem]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item.source and item.source not in seen:
                seen.add(item.source)
                names.append(item.source)
        return names
