from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _none_to_empty_list(value: Any) -> Any:
    return [] if value is None else value


ResearchToolName = Literal["web_search", "vector_retrieval", "memory_recall", "mcp_tool"]


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3)
    depth: Literal["quick", "standard", "deep"] = "standard"
    include_private_docs: bool = True
    use_memory: bool = True
    max_sections: int = 4
    max_revisions: int = 2


class PlanItem(BaseModel):
    id: str
    question: str
    purpose: str
    status: Literal["pending", "running", "done"] = "pending"
    requires_research: bool = True
    search_query: str | None = None
    evidence_count: int = 0
    revision_hint: str | None = None


class SearchQuery(BaseModel):
    query: str
    intent: str
    plan_item_id: str | None = None
    tool: ResearchToolName = "web_search"
    rewrite_index: int = 0
    revision: int = 0


class EvidenceItem(BaseModel):
    title: str
    source: str
    kind: str = "web"
    url: str | None = None
    snippet: str | None = None
    content: str | None = None
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceCompressionContract(BaseModel):
    summary: str
    key_excerpts: list[str] = Field(default_factory=list)
    relevance: float = 0.0
    limitations: list[str] = Field(default_factory=list)


class ChunkContextContract(BaseModel):
    context: str
    key_terms: list[str] = Field(default_factory=list)
    provenance_hint: str = ""
    confidence: float = 0.0


class ResearchNote(BaseModel):
    plan_item_id: str
    question: str
    finding: str
    evidence_titles: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    sufficiency_score: float = 0.0
    gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    research_iterations: list[dict[str, Any]] = Field(default_factory=list)
    completed_reason: str | None = None


class RetrievalRoute(BaseModel):
    plan_item_id: str
    mode: Literal["external", "internal", "hybrid"] = "hybrid"
    web_query: str | None = None
    internal_query: str | None = None
    reason: str
    selected_tools: list[ResearchToolName] = Field(default_factory=list)
    web_queries: list[str] = Field(default_factory=list)
    internal_queries: list[str] = Field(default_factory=list)
    memory_query: str | None = None
    min_evidence: int = 1
    min_sources: int = 1
    sufficiency_criteria: list[str] = Field(default_factory=list)


class CorpusProfile(BaseModel):
    document_count: int = 0
    source_count: int = 0
    source_names: list[str] = Field(default_factory=list)
    document_kinds: dict[str, int] = Field(default_factory=dict)
    keyword_signals: list[str] = Field(default_factory=list)
    has_private_docs: bool = False
    has_reference_docs: bool = False
    vector_backend: str = "qdrant"
    keyword_backend: str = "sqlite_fts5_bm25"
    embedding_dimensions: int = 0
    collection_name: str | None = None
    last_updated: str = Field(default_factory=_utc_now)


class MemoryRecord(BaseModel):
    key: str
    value: str
    layer: Literal["session", "canonical", "summary"] = "session"
    tags: list[str] = Field(default_factory=list)
    run_id: str | None = None
    session_id: str | None = None
    topic: str | None = None
    confidence: float = 0.0
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClarificationContract(BaseModel):
    need_clarification: bool = False
    question: str = ""
    verification: str = ""
    missing_dimensions: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class PlannerContract(BaseModel):
    research_brief: str
    plan: list[PlanItem] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    revision_budget: int = 0
    confidence: float = 0.0


class SupervisorToolCall(BaseModel):
    name: Literal["think_tool", "ConductResearch", "ResearchComplete"]
    rationale: str
    plan_item_ids: list[str] = Field(default_factory=list)
    research_topic: str | None = None
    reflection: str | None = None
    mode: Literal["external", "internal", "hybrid"] | None = None
    selected_tools: list[ResearchToolName] = Field(default_factory=list)
    web_queries: list[str] = Field(default_factory=list)
    internal_queries: list[str] = Field(default_factory=list)
    memory_query: str | None = None
    min_evidence: int | None = None
    min_sources: int | None = None
    sufficiency_criteria: list[str] = Field(default_factory=list)

    @field_validator(
        "plan_item_ids",
        "selected_tools",
        "web_queries",
        "internal_queries",
        "sufficiency_criteria",
        mode="before",
    )
    @classmethod
    def _normalize_optional_lists(cls, value: Any) -> Any:
        return _none_to_empty_list(value)


class SupervisorDecisionContract(BaseModel):
    reflection: str
    tool_calls: list[SupervisorToolCall] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    max_concurrent_research_units: int = 1
    confidence: float = 0.0

    @field_validator("tool_calls", "completion_criteria", mode="before")
    @classmethod
    def _normalize_optional_lists(cls, value: Any) -> Any:
        return _none_to_empty_list(value)


class ResearcherToolDecisionContract(BaseModel):
    action: Literal["think_tool", "web_search", "mcp_tool", "ResearchComplete"] = "web_search"
    query: str | None = None
    mcp_tool_name: str | None = None
    rationale: str = ""
    reflection: str = ""
    completion_reason: str | None = None
    confidence: float = 0.0


class VerificationContract(BaseModel):
    issues: list[str] = Field(default_factory=list)
    critical_issues: list[str] = Field(default_factory=list)
    should_revise: bool = False
    revision_reason: str | None = None
    confidence: float = 0.0
    coverage_score: float = 0.0


class ReporterSectionDraft(BaseModel):
    heading: str
    content: str
    citation_indexes: list[int] = Field(default_factory=list)


class ReporterContract(BaseModel):
    title: str
    summary: str
    sections: list[ReporterSectionDraft] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    source_index: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class AgentHandoff(BaseModel):
    from_agent: str
    to_agent: str
    step: str
    reason: str
    revision: int = 0
    created_at: str = Field(default_factory=_utc_now)


class RunTraceEvent(BaseModel):
    kind: Literal["handoff", "tool_call", "step", "memory_write", "verification", "checkpoint", "failure", "evaluation"]
    actor: str
    message: str
    step: str | None = None
    status: Literal["started", "completed", "failed", "skipped"] = "completed"
    from_agent: str | None = None
    to_agent: str | None = None
    handoff: AgentHandoff | None = None
    tool_name: str | None = None
    provider: str | None = None
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now)


class ReportSection(BaseModel):
    heading: str
    content: str
    citations: list[EvidenceItem] = Field(default_factory=list)
    evidence_count: int = 0
    source_summary: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    title: str
    summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    citations: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = 0.0
    highlights: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    source_index: list[str] = Field(default_factory=list)
    source_count: int = 0


class RAGEvaluation(BaseModel):
    plan_coverage: float = 0.0
    retrieval_hit_rate: float = 0.0
    private_retrieval_hit_rate: float = 0.0
    evidence_sufficiency: float = 0.0
    tool_selection_coverage: float = 0.0
    query_rewrite_count: int = 0
    source_quality_score: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness_proxy: float = 0.0
    citation_precision: float = 0.0
    citation_source_coverage: float = 0.0
    source_diversity: int = 0
    insufficient_plan_items: list[str] = Field(default_factory=list)
    unsupported_sections: list[str] = Field(default_factory=list)
    no_citation_section_count: int = 0
    passed: bool = False
    notes: list[str] = Field(default_factory=list)


class RunCheckpoint(BaseModel):
    run_id: str
    stage: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now)


class ResearchJob(BaseModel):
    job_id: str
    request: ResearchRequest
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    run_id: str | None = None
    error: str | None = None
    attempts: int = 0
    max_attempts: int = 1
    timeout_seconds: float | None = None
    cancel_requested: bool = False
    queued_at: str = Field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None


class ResearchRun(BaseModel):
    run_id: str
    job_id: str | None = None
    request: ResearchRequest
    research_brief: str | None = None
    corpus_profile: CorpusProfile | None = None
    supervisor_decision: SupervisorDecisionContract | None = None
    plan: list[PlanItem] = Field(default_factory=list)
    search_queries: list[SearchQuery] = Field(default_factory=list)
    retrieval_routes: list[RetrievalRoute] = Field(default_factory=list)
    notes: list[ResearchNote] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    web_hits: list[EvidenceItem] = Field(default_factory=list)
    memory_hits: list[EvidenceItem] = Field(default_factory=list)
    document_hits: list[EvidenceItem] = Field(default_factory=list)
    checkpoints: list[RunCheckpoint] = Field(default_factory=list)
    trace: list[RunTraceEvent] = Field(default_factory=list)
    handoffs: list[AgentHandoff] = Field(default_factory=list)
    report: ResearchReport | None = None
    evaluation: RAGEvaluation | None = None
    issues: list[str] = Field(default_factory=list)
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    revision_count: int = 0
    failure_reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int | None = None
