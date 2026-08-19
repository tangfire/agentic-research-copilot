from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import RLock

from .agents import PlannerAgent, ResearchAgent, ReporterAgent, SupervisorAgent, VerifierAgent
from .constraint_evaluation import (
    apply_constraint_coverage_gate,
    extract_constraint_coverage_from_run,
    extract_constraint_texts,
)
from .document_reader import DocumentReader
from .evaluation import RAGEvaluator
from .github_repository import canonical_repository_slug, parse_github_repository_hint
from .ledger import JobLedger, RunLedger
from .mcp_tools import build_mcp_tool
from .multi_agent_harness import (
    SPECIALIST_PROFILES,
    build_role_assignments,
    build_route_decisions,
    enrich_research_run,
    replay_from_frozen_run,
)
from .providers import build_embedding_provider, build_model_provider
from .provider_base import ResearchModelProvider
from .provider_validation import provider_runtime_report, require_real_provider_config
from .retrieval import DocumentStore, RerankerConfig, build_reranker
from .schemas import (
    ClarificationContract,
    CorpusProfile,
    EvidenceItem,
    ResearchNote,
    ResearchJob,
    ResearchRequest,
    ResearchRun,
    ReportSection,
    RetrievalRoute,
    SearchQuery,
    SupervisorDecisionContract,
    AgentSpecialistId,
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
    agent_id: AgentSpecialistId = "architecture_fit"
    agent_name: str = "ArchitectureFitAgent"
    executed_route: RetrievalRoute | None = None
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
        model_provider: ResearchModelProvider | None = None,
        embedding_provider: ResearchModelProvider | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        if self.settings.strict_providers and model_provider is None:
            require_real_provider_config(self.settings)
        self.model_provider = model_provider or build_model_provider(self.settings)
        self.embedding_provider = embedding_provider or build_embedding_provider(self.settings, self.model_provider)
        self.mcp_registry = build_mcp_tool(self.settings)
        self.mcp_tool = self.mcp_registry.search if self.mcp_registry is not None else None
        self.mcp_tool_catalog_error = ""
        self.mcp_tool_catalog = []
        if self.mcp_registry is not None:
            try:
                self.mcp_tool_catalog = self.mcp_registry.describe_tools()
            except Exception as exc:
                self.mcp_tool_catalog_error = str(exc)
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
            graph_max_relationships_per_chunk=self.settings.rag_graph_max_relationships_per_chunk,
            graph_neighbor_limit=self.settings.rag_graph_neighbor_limit,
            graph_entity_candidate_limit=self.settings.rag_graph_entity_candidate_limit,
            graph_relation_candidate_limit=self.settings.rag_graph_relation_candidate_limit,
            reranker=self.reranker,
            contextualizer_provider=self.model_provider,
            graph_provider=self.model_provider,
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
        self.search_tool = search_tool or build_search_tool(self.settings)
        self.router = RetrievalCoordinator(
            max_query_rewrites=self.settings.rag_max_query_rewrites,
            min_evidence_per_item=self.settings.rag_min_evidence_per_item,
            min_source_diversity=self.settings.rag_min_source_diversity,
            mcp_enabled=self.mcp_tool is not None,
        )
        self.planner = PlannerAgent(self.model_provider, self.settings)
        self.researcher = self._build_research_worker()
        self.verifier = VerifierAgent(self.model_provider, self.settings)
        self.reporter = ReporterAgent(self.model_provider, self.settings)
        self.supervisor_agent = SupervisorAgent(self.model_provider, self.settings)
        self._restore_state()
        if self.settings.seed_reference_knowledge:
            self._seed_reference_knowledge()

    def _build_research_worker(self) -> ResearchAgent:
        return ResearchAgent(
            self.search_tool,
            model_provider=self.model_provider,
            embedding_provider=self.embedding_provider,
            mcp_tool=self.mcp_tool,
            mcp_tool_catalog=self.mcp_tool_catalog,
            source_reader_enabled=self.settings.source_reader_enabled,
            source_reader_strategy=self.settings.source_reader_strategy,
            raw_content_max_chars=self.settings.source_reader_max_chars,
            excerpt_max_chars=self.settings.source_reader_excerpt_chars,
            chunk_context_window=self.settings.source_reader_chunk_context_window,
            max_iterations=self.settings.research_max_iterations,
        )

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

    def clear_history(self) -> dict[str, object]:
        storage_runs_deleted = self.storage.clear_runs()
        storage_jobs_deleted = self.storage.clear_jobs()
        runs_deleted = max(storage_runs_deleted, self.ledger.clear())
        jobs_deleted = max(storage_jobs_deleted, self.jobs.clear())
        telemetry_deleted = self.telemetry.clear()
        return {
            "deleted": True,
            "runs_deleted": runs_deleted,
            "jobs_deleted": jobs_deleted,
            "telemetry_deleted": telemetry_deleted,
            "memory_removed_from_core": False,
            "agent_memory_preserved": True,
        }

    def clarify(self, request: ResearchRequest) -> ClarificationContract:
        corpus_profile = self.documents.profile()
        contract, usage = self.model_provider.clarify_request(
            request,
            corpus_profile,
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

    def list_runs(self) -> list[ResearchRun]:
        self.ledger.extend(self.storage.load_runs())
        return self.ledger.list()

    def get_run(self, run_id: str) -> ResearchRun | None:
        persisted = self.storage.load_run(run_id)
        if persisted is not None:
            return self.ledger.record(persisted)
        return self.ledger.get(run_id)

    def submit_job(self, request: ResearchRequest) -> ResearchJob:
        request = self._request_with_repository_context(request)
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
                    "A conversational research agent that turns technical questions, local constraints, "
                    "and saved memory into confirmed research plans, cited reports, trace, and evaluation."
                ),
                "release_shape": "single Python FastAPI service with an agent session layer, SQLite memory, background job queue, Qdrant retrieval, and a static local workbench",
                "research_path": "session -> memory -> interactive plan -> confirm -> research -> report/eval/replay",
                "seed_reference_knowledge": self.settings.seed_reference_knowledge,
            },
            "agent_session": {
                "enabled": True,
                "status_contract": [
                    "collecting",
                    "planning",
                    "awaiting_confirmation",
                    "researching",
                    "completed",
                    "failed",
                ],
                "confirmation_gate": "Research runs are not started from chat until the user confirms the generated plan draft.",
                "handoff_boundary": "ConversationalResearchAgent owns session state, memory injection, clarification, plan confirmation, and job binding; ResearchCopilot remains the research runtime.",
                "ui_shape": "Agent Workbench with session list, chat timeline, plan draft, memory inspector, tool status, report, trace, and evaluation tabs.",
                "session_key": "session_id is mirrored as session_key so export bundles and event streams have a stable external handle.",
                "step_stream": {
                    "mode": "polling",
                    "endpoint": "/v1/agent/sessions/{session_id}/steps",
                    "events_endpoint": "/v1/agent/sessions/{session_id}/events",
                    "notes": "Session-visible AgentRunStep rows are written during chat/planning/confirmation and synchronized from run trace after completion.",
                },
                "context_compaction": {
                    "enabled": True,
                    "trigger": "long conversations are summarized into session.context_summary before planning",
                    "heartbeat": "researching sessions refresh a lightweight heartbeat step while the background job is still active",
                },
            },
            "workspace_control_plane": {
                "enabled": True,
                "endpoint": "/v1/agent/workspaces",
                "purpose": "Single-user workspace profiles carry team context, stack, deployment constraints, source preferences, and disabled tools.",
                "default_workspace": "auto-created on startup if no workspace exists",
            },
            "skills": {
                "enabled": True,
                "endpoint": "/v1/agent/skills",
                "pack_roots": self.settings.skill_paths,
                "script_timeout_seconds": self.settings.skill_script_timeout_seconds,
                "pack_shape": "skill.json + SKILL.md + optional scripts/",
                "execution_boundary": "Only manifest-declared local scripts run through JSON stdin/stdout, with a timeout and no shell command composition.",
                "catalog": [
                    "open_source_adoption_review",
                    "architecture_tradeoff_memo",
                    "demo_readiness_risk_review",
                ],
                "selection_policy": "Registry-loaded skill selection with required-input checks, instruction excerpts, optional preflight hooks, and a readable reason in the plan draft",
            },
            "tool_policy": {
                "enabled": True,
                "registry_endpoint": "/v1/agent/tools",
                "default_tools": ["web_search", "vector_retrieval", "mcp_tool"],
                "approval_model": "observable_hitl_v2",
                "approval_boundary": "v2 records approval requests for unavailable or risky MCP actions without interrupting every research tool call.",
                "destructive_tools_supported": False,
            },
            "clarification": {
                "enabled": True,
                "mode": "ODR-style clarify_with_user front door",
                "purpose": "Ask at most one concise follow-up question when the request is too vague to research safely.",
            },
            "memory": {
                "enabled": True,
                "provider": "sqlite",
                "scopes": ["user", "project", "session"],
                "kinds": ["preference", "constraint", "decision", "fact", "todo"],
                "extraction": "lightweight heuristic extractor records explicit preferences, team/project constraints, and session facts after user turns",
                "planning_injection": "Relevant memory is appended to the ResearchRequest before planner drafting so team constraints do not need to be pasted repeatedly.",
                "knowledge_base_sync": "Project-scope memory is also indexed as a local document for vector/BM25/graph retrieval.",
                "boundary": "Mem0-inspired memory layering without adding the Mem0 SDK.",
            },
            "constraint_coverage": {
                "enabled": True,
                "warning_threshold": 0.6,
                "fail_threshold": 0.4,
                "purpose": "Project constraints saved as memory must be visible in the final memo or evaluation notes.",
                "endpoint": "/v1/research/runs/{run_id}/constraint-coverage",
            },
            "orchestration": {
                "runtime": self.settings.orchestration_runtime,
                "strict_providers": self.settings.strict_providers,
                "active_graph": "supervisor -> planner -> research_supervisor -> specialist_worker stage -> reporter -> verifier/evaluator -> finalize",
                "checkpointer": self.settings.langgraph_checkpointer,
                "checkpoint_path": self.settings.langgraph_checkpoint_path,
                "durability_boundary": "Single-node LangGraph sqlite checkpointing is the default graph durability layer; SQLite run traces/replay are always persisted by the app, with MemorySaver used only as a defensive fallback.",
                "reference_pattern": "Open Deep Research uses LangGraph StateGraph/subgraph orchestration; this repo uses LangGraph for a product-specific research workflow with specialist workers inside the research stage.",
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
                        "persistence and replay concepts",
                        "reader registry and document ingestion extension shape",
                        "agent handoff vocabulary",
                        "observability, replay, and run-ledger patterns",
                    ],
                    "dependency": False,
                },
                {
                    "name": "Mem0",
                    "learning_priority": "reference",
                    "used_for": [
                        "user/project/session memory layering",
                        "preference and constraint persistence",
                        "memory retrieval before planning",
                    ],
                    "dependency": False,
                },
                {
                    "name": "Open WebUI / AnythingLLM",
                    "learning_priority": "reference",
                    "used_for": [
                        "chat-first workspace shell",
                        "local knowledge base entry point",
                        "session-oriented workbench layout",
                    ],
                    "dependency": False,
                },
            ],
            "open_deep_research_alignment": {
                "positioning": (
                    "Complex-question AI Research Copilot: plan, search/read, retrieve, "
                    "synthesize, verify citations, evaluate, and replay."
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
                    "human confirmation gate before launching a long-running research job",
                ],
                "product_specific_differences": [
                    "Uses an ODR-style LLM research supervisor; ConductResearch calls carry selected tools, query rewrites, grounding mode, and sufficiency criteria",
                    "Routes delegated research units to three bounded specialist workers for repo signals, architecture fit, and operations risk",
                    "Adds local document grounding with text/Markdown/HTML/PDF parsing before Qdrant retrieval",
                    "Adds a conversational session and memory layer in front of the research graph instead of turning every chat turn into a run",
                    "Uses single-node FastAPI/Celery/Redis/SQLite/Qdrant deployment, not a distributed platform",
                ],
            },
            "agents": [
                {
                    "name": "conversational_agent",
                    "role": "manages sessions, memory, clarification, plan drafts, confirmation gate, and research job binding",
                },
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
                    "role": "turns ConductResearch tool-call arguments into executable web/vector/MCP route contracts",
                },
                {
                    "name": "repo_signal_agent",
                    "role": "checks repository facts, source authority, maintenance signals, code, issues, pull requests, releases, and licenses",
                },
                {
                    "name": "architecture_fit_agent",
                    "role": "checks architecture fit, API/runtime semantics, integration cost, workflow design, and local KB alignment",
                },
                {
                    "name": "ops_risk_agent",
                    "role": "checks deployment, rollback, dependency, security/compliance, cost, latency, and reliability constraints",
                },
                {
                    "name": "grounding_layer",
                    "role": "serves vector retrieval to specialist workers with precise child chunks, parent/neighbor context, dense/BM25 fusion, and optional graph/rerank enhancement",
                },
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
                    "enabled": self.mcp_tool is not None,
                    "configured_enabled": self.settings.mcp_enabled,
                    "server_url_configured": bool(self.settings.mcp_server_url),
                    "loaded": self.mcp_tool is not None,
                    "available": self.mcp_tool is not None and not self.mcp_tool_catalog_error,
                    "catalog_error": self.mcp_tool_catalog_error,
                    "tools": self.settings.mcp_tools,
                    "tools_configured": bool(self.settings.mcp_tools),
                    "tool_catalog": [tool.model_dump() for tool in self.mcp_tool_catalog],
                    "tool_catalog_count": len(self.mcp_tool_catalog),
                    "auth_required": self.settings.mcp_auth_required,
                    "auth_token_configured": bool(self.settings.mcp_auth_token),
                    "mcp_prompt_configured": bool(self.settings.mcp_prompt),
                    "transport": self.settings.mcp_transport,
                    "timeout_seconds": self.settings.mcp_timeout_seconds,
                    "reference_shape": "Open Deep Research loads configured MCP tools from mcp_config.url + mcp_config.tools into the researcher toolkit; this repo treats MCP as an external tool boundary rather than a local workbench server.",
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
                    "strategy": "parent_child_dense_bm25_optional_graph_rerank",
                    "parent_child": "child chunks are retrieved precisely, then same-document neighbor context is returned for synthesis",
                    "contextual_retrieval": "indexing-time chunk context prefixes are prepended before dense embedding and BM25 indexing",
                    "graph_augmented": self.settings.rag_graph_enabled,
                    "graph_strategy": "Optional LightRAG-inspired structured entity/relation graph fused with dense/BM25 candidates",
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
                    "graph_strategy": "Optional LightRAG-inspired structured entity/relation graph; graph hits are fused before rerank when enabled",
                },
                {"name": "run_ledger", "provider": "sqlite", "enabled": True},
                {"name": "job_queue", "provider": self.settings.job_queue_backend, "enabled": True},
                {"name": "telemetry", "provider": "in_process_event_log", "enabled": True},
                {
                    "name": "agent_memory",
                    "provider": "sqlite + local DocumentStore sync",
                    "enabled": True,
                    "scopes": ["user", "project", "session"],
                },
            ],
            "retrieval": {
                "routes": ["external", "internal", "hybrid"],
                "default_strategy": "parent_child_dense_bm25_optional_graph_rerank",
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
                    "tool_selection": ["web_search", "vector_retrieval", "mcp_tool"],
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
            "observability": {
                "trace_fields": [
                    "handoff",
                    "tool_call",
                    "step",
                    "checkpoint",
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
                "persisted_objects": [
                    "documents",
                    "research_jobs",
                    "research_runs",
                    "agent_workspaces",
                    "agent_sessions",
                    "agent_messages",
                    "agent_plan_drafts",
                    "agent_run_steps",
                    "tool_invocations",
                    "approval_requests",
                    "memory_items",
                    "memory_extraction_results",
                    "constraint_coverage",
                ],
            },
        }

    def replay(self, run_id: str) -> ResearchRun | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        replayed = replay_from_frozen_run(run)
        coverage = extract_constraint_coverage_from_run(replayed)
        if coverage:
            self.storage.save_constraint_coverage(coverage)
        self.ledger.record(replayed)
        self.storage.save_run(replayed)
        self.telemetry.emit(
            "run.replay",
            "Replayed frozen research artifacts without re-calling live tools.",
            run_id=replayed.run_id,
            source_run_id=run.run_id,
            replay_mode="frozen_artifacts",
        )
        return replayed

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
                "route between external search, vector_retrieval, configured MCP tools, and hybrid evidence, "
                "verify citations, evaluate RAG quality, persist trace and replay artifacts, and support "
                "OpenAI-compatible chat providers, Qwen embeddings, Tavily search, and explicit dev fixtures for tests."
            ),
            metadata={"kind": "project_overview"},
        )
        self._ensure_seed_document(
            title="Architecture overview",
            source="docs/architecture.md",
            snippet="Build an AI Research Copilot that can plan, search, ground, verify, evaluate, and report.",
            content=(
                "The architecture centers on a LangGraph StateGraph supervisor, planner, "
                "ODR-style research_supervisor, concurrent researcher/retriever workers, Qdrant dense "
                "vectors, indexing-time contextual retrieval prefixes, SQLite FTS5 BM25 keyword retrieval, "
                "RRF/DBSF hybrid fusion, Qwen/DashScope reranking with local rule fallback, "
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
                "PraisonAI contributes handoff, observability, and workflow ideas."
            ),
            content=(
                "The source map explains which upstream ideas are reused for planning, "
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
                "Qwen/DashScope reranker with local rule fallback, explicit fixture injection for tests, clear "
                "single-node SQLite checkpoint/replay boundaries, and SQLite-backed job/run status "
                "visibility across the API and local worker. Streaming, auth, and multi-tenancy are "
                "deferred because this is a personal research copilot, not a SaaS platform."
            ),
            metadata={"kind": "hardening_roadmap"},
        )
    def run(self, request: ResearchRequest, *, job_id: str | None = None) -> ResearchRun:
        from .graph_runtime import LangGraphResearchRuntime

        request = self._request_with_repository_context(request)
        run = LangGraphResearchRuntime(self).run(request, job_id=job_id)
        run = run.model_copy(
            update={
                "metadata": {
                    **run.metadata,
                    "request_metadata": request.metadata,
                    "job_id": job_id or run.job_id,
                    "harness_version": "research-desk-v4",
                },
            }
        )
        coverage = extract_constraint_coverage_from_run(run)
        if coverage:
            self.storage.save_constraint_coverage(coverage)
            run = apply_constraint_coverage_gate(run, coverage)
        request_metadata = request.metadata or {}
        run = enrich_research_run(
            run,
            session_id=_optional_text(request_metadata.get("session_id")),
            skill_id=_optional_text(request_metadata.get("skill_id")),
            workspace_context=_workspace_context_from_request_metadata(request_metadata),
        )
        self.ledger.record(run)
        self.storage.save_run(run)
        return run

    def _request_with_repository_context(self, request: ResearchRequest) -> ResearchRequest:
        metadata = dict(request.metadata or {})
        hint = parse_github_repository_hint(metadata, request.topic)
        if hint is None:
            return request
        metadata["github_repository"] = hint
        metadata["github_repository_slug"] = canonical_repository_slug(hint)
        return request.model_copy(update={"metadata": metadata})

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

        request_metadata = request.metadata or {}
        assignments = build_role_assignments(
            request,
            plan,
            list(route_lookup.values()),
            [],
            run_id=None,
            session_id=_optional_text(request_metadata.get("session_id")),
            skill_id=_optional_text(request_metadata.get("skill_id")),
            workspace_context=_workspace_context_from_request_metadata(request_metadata),
        )
        route_decisions = build_route_decisions(
            request,
            plan,
            list(route_lookup.values()),
            [],
            assignments,
        )
        worker_by_item = {
            decision.plan_item_id: decision.agent_id
            for decision in route_decisions
            if decision.status == "selected"
        }

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
                    route=self._route_for_specialist(
                        route_lookup[item.id],
                        worker_by_item.get(item.id, "architecture_fit"),
                        request=request,
                        corpus_profile=corpus_profile,
                    ),
                    corpus_profile=corpus_profile,
                    research_brief=research_brief,
                    agent_id=worker_by_item.get(item.id, "architecture_fit"),
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
                    route=self._route_for_specialist(
                        route_lookup[item.id],
                        worker_by_item.get(item.id, "architecture_fit"),
                        request=request,
                        corpus_profile=corpus_profile,
                    ),
                    corpus_profile=corpus_profile,
                    research_brief=research_brief,
                    agent_id=worker_by_item.get(item.id, "architecture_fit"),
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
                        agent_id=worker_by_item.get(item.id, "architecture_fit"),
                        agent_name=SPECIALIST_PROFILES[
                            worker_by_item.get(item.id, "architecture_fit")
                        ].agent_name,
                        executed_route=route_lookup.get(item.id),
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
        agent_id: AgentSpecialistId = "architecture_fit",
    ) -> PlanItemResearchResult:
        profile = SPECIALIST_PROFILES[agent_id]
        worker = self._build_research_worker()
        worker.mcp_tool = self.mcp_tool
        worker.mcp_tool_catalog = list(self.mcp_tool_catalog)
        repository_hint = parse_github_repository_hint(request.metadata, request.topic, item.question, item.search_query)
        repository_slug = canonical_repository_slug(repository_hint)
        repository_context = f" Target GitHub repository: {repository_slug}." if repository_slug else ""
        worker_item = item.model_copy(
            update={
                "purpose": (
                    f"{item.purpose}{repository_context} "
                    f"Specialist mandate: {profile.responsibility} "
                    f"Evidence should be selected and explained from the {profile.agent_name} perspective."
                )
            }
        )
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
            collection = worker.collect_iterative(
                worker_item,
                web_queries,
                min_evidence=max(1, route.min_evidence),
                min_sources=max(1, route.min_sources),
                max_iterations=self.settings.research_max_iterations,
                repository_hint=repository_hint,
                required_tools=tuple(route.selected_tools),
            )
            web_evidence = self._tag_specialist_evidence(
                self._dedupe_evidence(collection.evidence),
                agent_id=agent_id,
                agent_name=profile.agent_name,
            )
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
            document_evidence = self._tag_specialist_evidence(
                document_evidence,
                agent_id=agent_id,
                agent_name=profile.agent_name,
            )
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
            agent_id=agent_id,
            agent_name=profile.agent_name,
            executed_route=route,
            web_evidence=web_evidence,
            document_evidence=document_evidence,
            note=note,
            web_latency_ms=web_latency_ms,
            document_latency_ms=document_latency_ms,
        )

    def _route_for_specialist(
        self,
        route: RetrievalRoute,
        agent_id: AgentSpecialistId,
        *,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
    ) -> RetrievalRoute:
        """Apply the specialist's tool boundary before its worker loop starts."""
        profile = SPECIALIST_PROFILES[agent_id]
        allowed_tools = set(profile.shared_tools) | set(profile.exclusive_tools)
        selected_tools = [
            tool
            for tool in route.selected_tools
            if tool in allowed_tools
            and not (tool == "mcp_tool" and self.mcp_tool is None)
        ]
        if not selected_tools:
            if agent_id == "repo_signal" and self.mcp_tool is not None:
                selected_tools = ["web_search"]
            elif agent_id == "architecture_fit" and request.include_private_docs and corpus_profile.has_private_docs:
                selected_tools = ["vector_retrieval"]
            elif agent_id == "ops_risk" and request.include_private_docs and corpus_profile.has_private_docs:
                selected_tools = ["vector_retrieval"]
            else:
                selected_tools = ["web_search"]

        mode = "hybrid" if {"web_search", "vector_retrieval"} <= set(selected_tools) else (
            "internal" if selected_tools == ["vector_retrieval"] else "external"
        )
        web_queries = list(route.web_queries) if "web_search" in selected_tools else []
        internal_queries = list(route.internal_queries) if "vector_retrieval" in selected_tools else []
        if "web_search" in selected_tools and not web_queries:
            web_queries = [route.web_query or route.internal_query or " ".join(
                [route.reason, request.topic]
            )]
        if "vector_retrieval" in selected_tools and not internal_queries:
            internal_queries = [route.internal_query or route.web_query or request.topic]

        return route.model_copy(
            update={
                "mode": mode,
                "selected_tools": selected_tools,
                "web_query": web_queries[0] if web_queries else None,
                "internal_query": internal_queries[0] if internal_queries else None,
                "web_queries": web_queries,
                "internal_queries": internal_queries,
                "reason": (
                    f"{route.reason} Specialist boundary: {profile.agent_name} "
                    f"owns {', '.join(selected_tools)}."
                ),
            }
        )

    def _tag_specialist_evidence(
        self,
        evidence: list[EvidenceItem],
        *,
        agent_id: AgentSpecialistId,
        agent_name: str,
    ) -> list[EvidenceItem]:
        return [
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "agent_id": agent_id,
                        "agent_name": agent_name,
                        "execution_mode": "specialist_worker",
                    }
                }
            )
            for item in evidence
        ]

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
            if request.include_private_docs and corpus_profile.has_private_docs:
                selected_tools = ["vector_retrieval"]
            elif self.researcher.search_tool is not None:
                selected_tools = ["web_search"]
        if "web_search" not in selected_tools and "vector_retrieval" not in selected_tools:
            if request.include_private_docs and corpus_profile.has_private_docs:
                selected_tools.insert(0, "vector_retrieval")
            elif self.researcher.search_tool is not None:
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
                f"use at least {min_sources} source group(s) when available",
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
        valid = {"web_search", "vector_retrieval", "mcp_tool"}
        normalized: list[str] = []
        for tool in tools or []:
            if tool not in valid or tool in normalized:
                continue
            if tool == "web_search" and self.researcher.search_tool is None:
                continue
            if tool == "vector_retrieval" and (not request.include_private_docs or not corpus_profile.has_private_docs):
                continue
            if tool == "mcp_tool" and self.mcp_tool is None:
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
            f"{len(retrieval_routes)} retrieval routes, {len(web_hits)} web hits, "
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
                "document_hit_count": len(document_hits),
                "note_count": len(notes),
                "revision_count": revision_count,
                "route_counts": route_counts,
                "tool_counts": tool_counts,
                "query_rewrite_count": query_rewrite_count,
            },
        )

    def _build_sections(
        self,
        request: ResearchRequest,
        research_brief: str,
        plan: list,
        retrieval_routes,
        evidence: list[EvidenceItem],
        web_hits: list[EvidenceItem],
        document_hits: list[EvidenceItem],
        notes: list[ResearchNote],
        search_queries: list[SearchQuery],
    ) -> list[ReportSection]:
        sections: list[ReportSection] = []
        base_sources = self._source_names(evidence)
        ranked_evidence = self._rank_evidence_for_report(evidence)
        topic_evidence = [
            item
            for item in ranked_evidence
            if item.kind != "run-artifact" and item.source not in {"run-ledger", "internal-note"}
        ]
        note_lookup = {note.plan_item_id: note for note in notes}
        route_lookup = {route.plan_item_id: route for route in retrieval_routes}
        search_query_lookup: dict[str, list[SearchQuery]] = {}
        evidence_by_plan: dict[str, list[EvidenceItem]] = {}
        evidence_by_title = {item.title: item for item in topic_evidence}

        for query in search_queries:
            if query.plan_item_id:
                search_query_lookup.setdefault(query.plan_item_id, []).append(query)

        for evidence_item in topic_evidence:
            plan_item_id = evidence_item.metadata.get("plan_item_id")
            if plan_item_id:
                evidence_by_plan.setdefault(str(plan_item_id), []).append(evidence_item)

        planned_items = [item for item in plan if getattr(item, "requires_research", True)] or list(plan)
        max_sections = max(1, request.max_sections)
        for index, item in enumerate(planned_items[:max_sections], start=1):
            note = note_lookup.get(item.id)
            route = route_lookup.get(item.id)
            citations = list(evidence_by_plan.get(item.id, []))
            if note is not None:
                citations.extend(
                    evidence_by_title[title]
                    for title in note.evidence_titles
                    if title in evidence_by_title
                )
            if not citations:
                citations = topic_evidence[:4] or ranked_evidence[:4]

            section_citations = self._dedupe_evidence(citations)
            sections.append(
                ReportSection(
                    heading=self._section_heading(item, index),
                    content=self._section_content(
                        request=request,
                        research_brief=research_brief,
                        item=item,
                        note=note,
                        route=route,
                        citations=section_citations,
                        search_queries=search_query_lookup.get(item.id, []),
                    ),
                    citations=section_citations,
                    evidence_count=len(section_citations),
                    source_summary=self._source_names(section_citations) or base_sources[:3],
                )
            )

        if not sections and ranked_evidence:
            section_citations = self._dedupe_evidence(topic_evidence[:4] or ranked_evidence[:4])
            sections.append(
                ReportSection(
                    heading=request.topic,
                    content=self._fallback_section_content(request.topic, research_brief, section_citations),
                    citations=section_citations,
                    evidence_count=len(section_citations),
                    source_summary=self._source_names(section_citations) or base_sources[:3],
                )
            )
        constraint_section = self._build_team_constraint_section(
            request=request,
            evidence=evidence,
            document_hits=document_hits,
        )
        if constraint_section is not None:
            sections.append(constraint_section)
        return sections

    def _build_team_constraint_section(
        self,
        *,
        request: ResearchRequest,
        evidence: list[EvidenceItem],
        document_hits: list[EvidenceItem],
    ) -> ReportSection | None:
        constraints = self._team_constraint_texts(request, document_hits)
        if not constraints:
            return None
        citations = self._dedupe_evidence(document_hits[:4] or evidence[:4])
        if not citations:
            return None
        lines = [
            "这一节由 runtime 自动生成，用来把采用建议和团队硬约束对齐，方便复盘。",
            "团队约束覆盖清单：",
        ]
        for constraint in constraints[:12]:
            lines.append(f"- 已覆盖：{self._compact_report_text(constraint, 220)}")
        lines.append(
            "决策含义：只有当试点计划、回滚路径、证据质量和运维风险都能满足这些约束时，才应该接受最终采用建议。"
        )
        return ReportSection(
            heading="团队约束覆盖",
            content="\n".join(lines),
            citations=citations,
            evidence_count=len(citations),
            source_summary=self._source_names(citations),
        )

    def _team_constraint_texts(
        self,
        request: ResearchRequest,
        document_hits: list[EvidenceItem],
    ) -> list[str]:
        texts: list[str] = []
        metadata = request.metadata or {}
        metadata_values = [
            metadata.get("workspace_context"),
            metadata.get("team_context"),
            metadata.get("default_stack"),
            metadata.get("deployment_constraints"),
            metadata.get("risk_policy"),
            metadata.get("memory_context"),
            metadata.get("hard_constraints"),
            metadata.get("constraints"),
        ]
        for value in [request.topic, *metadata_values]:
            if isinstance(value, str):
                texts.extend(extract_constraint_texts(value))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        texts.extend(extract_constraint_texts(item, fallback_to_full_text=True))
        for item in document_hits[:8]:
            text = "\n".join(
                part
                for part in [item.snippet or "", item.content or ""]
                if part and part.strip()
            )
            texts.extend(extract_constraint_texts(text))
        return self._unique_strings(texts)

    def _section_heading(self, item, index: int) -> str:
        heading = " ".join(str(getattr(item, "question", "")).split())
        if not heading:
            return f"研究问题 {index}"
        return heading[:120].rstrip()

    def _section_content(
        self,
        *,
        request: ResearchRequest,
        research_brief: str,
        item,
        note: ResearchNote | None,
        route: RetrievalRoute | None,
        citations: list[EvidenceItem],
        search_queries: list[SearchQuery],
    ) -> str:
        finding = (note.finding if note is not None else "").strip()
        if not finding:
            finding = self._fallback_section_content(item.question, research_brief, citations)

        evidence_summary = self._evidence_summary(citations)
        parts = [
            f"本节回答：{self._compact_report_text(item.question, 260)}",
            f"研究目标：{self._compact_report_text(request.topic, 420)}。",
            f"为什么重要：{self._compact_report_text(item.purpose, 260)}",
            f"核心发现：{self._compact_report_text(finding, 900)}",
        ]
        if evidence_summary:
            parts.append(f"支撑证据：{evidence_summary}")
        if route is not None:
            tool_summary = ", ".join(route.selected_tools) or route.mode
            parts.append(
                f"检索路由：{route.mode}，使用 {tool_summary}；"
                f"充分性目标是至少 {route.min_evidence} 条证据，"
                f"覆盖 {route.min_sources} 个来源组。"
            )
        if search_queries:
            query_text = "; ".join(
                self._compact_report_text(query.query, 180) for query in search_queries[:2]
            )
            parts.append(f"使用的查询：{query_text}。")
        if note is not None and note.gaps:
            gaps = "; ".join(self._compact_report_text(gap, 180) for gap in note.gaps[:2])
            parts.append(f"剩余注意点：{gaps}。")
        if note is not None and note.follow_up_queries:
            followups = "; ".join(
                self._compact_report_text(query, 180) for query in note.follow_up_queries[:2]
            )
            parts.append(f"建议继续追问：{followups}。")
        return " ".join(part.strip() for part in parts if part and part.strip())

    def _fallback_section_content(
        self,
        topic: str,
        research_brief: str,
        citations: list[EvidenceItem],
    ) -> str:
        evidence_summary = self._evidence_summary(citations)
        if evidence_summary:
            return f"{research_brief} 与 {topic} 相关的证据：{evidence_summary}"
        return f"{research_brief} 暂时没有找到可引用证据来支撑 {topic}。"

    def _evidence_summary(self, citations: list[EvidenceItem]) -> str:
        snippets: list[str] = []
        for item in citations[:3]:
            text = " ".join(
                part.strip()
                for part in [item.title, item.snippet or "", item.content or ""]
                if part and part.strip()
            )
            if text:
                snippets.append(text[:320].rstrip())
        return " ".join(snippets)

    @staticmethod
    def _compact_report_text(value: object, max_chars: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _estimate_confidence(
        self,
        request: ResearchRequest,
        evidence: list[EvidenceItem],
        document_hits: list[EvidenceItem],
        plan: list,
    ) -> float:
        non_internal_sources = {
            item.source
            for item in evidence
            if item.source != "internal-note"
        }
        score = 0.3
        score += min(0.35, len(evidence) * 0.04)
        score += min(0.12, len(document_hits) * 0.02)
        score += min(0.08, len(non_internal_sources) * 0.03)
        score += min(0.07, len(plan) * 0.01)
        if request.depth == "deep":
            score += 0.03
        if not non_internal_sources:
            score -= 0.08
        return max(0.2, min(score, 0.95))

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


def _optional_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _workspace_context_from_request_metadata(metadata: dict[str, object]) -> str:
    fragments: list[str] = []
    for key in ("workspace_name", "workspace_context", "risk_policy", "skill_name", "skill_scenario"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(f"{key}: {value.strip()}")
    for key in ("default_stack", "deployment_constraints", "preferred_sources", "evaluation_focus"):
        value = metadata.get(key)
        if isinstance(value, list) and value:
            fragments.append(f"{key}: {', '.join(str(item) for item in value if str(item).strip())}")
    return " | ".join(fragments)
