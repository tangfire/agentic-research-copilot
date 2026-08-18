from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _none_to_empty_list(value: Any) -> Any:
    return [] if value is None else value


ResearchToolName = Literal["web_search", "vector_retrieval", "mcp_tool"]


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3)
    depth: Literal["quick", "standard", "deep"] = "standard"
    include_private_docs: bool = True
    max_sections: int = 4
    max_revisions: int = 2
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanItem(BaseModel):
    id: str = Field(description="Unique identifier for this plan item, e.g. 'item_1', 'item_2'.")
    question: str = Field(description="A focused research sub-question that, when answered, contributes a distinct section to the final report. Should be specific and independently researchable.")
    purpose: str = Field(description="Why this sub-question matters in the context of the overall research topic. Explains the role this item plays in the final report.")
    status: Literal["pending", "running", "done"] = Field(default="pending", description="Execution status of this plan item. Always set to 'pending' when creating a new plan.")
    requires_research: bool = Field(default=True, description="Whether this item needs external evidence gathering. Set to False only for items that can be answered from existing context alone.")
    search_query: str | None = Field(default=None, description="An optimized search string for web or vector search. Should be different from the question — shorter, keyword-focused, and tuned for search engines.")
    evidence_count: int = Field(default=0, description="Number of evidence items collected for this item. Leave as 0 when creating the plan.")
    revision_hint: str | None = Field(default=None, description="If this item is being revised, a short note describing what was missing or incorrect in the previous attempt.")


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


class MCPToolDescriptor(BaseModel):
    name: str
    description: str = ""
    required_args: list[str] = Field(default_factory=list)
    optional_args: list[str] = Field(default_factory=list)
    typical_scenarios: list[str] = Field(default_factory=list)


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


class KnowledgeGraphEntity(BaseModel):
    name: str
    entity_type: str = "concept"
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("aliases", mode="before")
    @classmethod
    def _normalize_aliases(cls, value: Any) -> Any:
        return _none_to_empty_list(value)


class KnowledgeGraphRelationship(BaseModel):
    source: str
    target: str
    relation_type: str = "related_to"
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    weight: float = 1.0
    confidence: float = 0.0

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: Any) -> Any:
        return _none_to_empty_list(value)


class KnowledgeGraphExtractionContract(BaseModel):
    entities: list[KnowledgeGraphEntity] = Field(default_factory=list)
    relationships: list[KnowledgeGraphRelationship] = Field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0

    @field_validator("entities", "relationships", mode="before")
    @classmethod
    def _normalize_graph_lists(cls, value: Any) -> Any:
        return _none_to_empty_list(value)


class KnowledgeGraphQueryContract(BaseModel):
    local_keywords: list[str] = Field(default_factory=list)
    global_keywords: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("local_keywords", "global_keywords", mode="before")
    @classmethod
    def _normalize_query_keywords(cls, value: Any) -> Any:
        return _none_to_empty_list(value)


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


class ClarificationContract(BaseModel):
    need_clarification: bool = False
    question: str = ""
    verification: str = ""
    missing_dimensions: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class PlannerContract(BaseModel):
    research_brief: str = Field(description="A 2-3 sentence summary of the research goal, the intended approach, and any important constraints or priorities. This guides all downstream agents.")
    plan: list[PlanItem] = Field(default_factory=list, description="A list of 3-5 focused sub-questions that together cover the research topic. Each item must be independently researchable and map to a distinct section of the final report. Do not overlap questions.")
    assumptions: list[str] = Field(default_factory=list, description="Key assumptions made while constructing this plan, e.g. about available sources, scope limits, or what prior knowledge exists.")
    success_criteria: list[str] = Field(default_factory=list, description="Concrete, verifiable conditions that define a successful research run, e.g. 'Every section has at least one citation' or 'Coverage includes both theoretical and empirical perspectives'.")
    revision_budget: int = Field(default=0, description="How many revision cycles are expected. Set to 0 for a fresh plan; incremented by the verifier if gaps are found.")
    confidence: float = Field(default=0.0, description="Estimated confidence that this plan will produce a high-quality report, from 0.0 (very uncertain) to 1.0 (highly confident). Consider topic clarity and source availability.")


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
    mcp_tool_args: dict[str, Any] | None = None
    rationale: str = ""
    reflection: str = ""
    completion_reason: str | None = None
    confidence: float = 0.0

    @field_validator("rationale", "reflection", mode="before")
    @classmethod
    def _normalize_optional_strings(cls, value: Any) -> Any:
        return "" if value is None else value


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
    kind: Literal["handoff", "tool_call", "step", "verification", "checkpoint", "failure", "evaluation"]
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
    document_hits: list[EvidenceItem] = Field(default_factory=list)
    checkpoints: list[RunCheckpoint] = Field(default_factory=list)
    trace: list[RunTraceEvent] = Field(default_factory=list)
    handoffs: list[AgentHandoff] = Field(default_factory=list)
    role_assignments: list[AgentRoleAssignment] = Field(default_factory=list)
    route_decisions: list[RouteDecision] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    evidence_ledger: EvidenceLedger | None = None
    benchmark_summary: BenchmarkRunSummary | None = None
    report: ResearchReport | None = None
    evaluation: RAGEvaluation | None = None
    issues: list[str] = Field(default_factory=list)
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    revision_count: int = 0
    failure_reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


AgentSessionStatus = Literal[
    "collecting",
    "planning",
    "awaiting_confirmation",
    "researching",
    "completed",
    "failed",
]
AgentMessageRole = Literal["user", "assistant", "system", "tool"]
AgentMessageIntent = Literal["chat", "clarify", "plan", "confirm", "research", "follow_up"]
MemoryScope = Literal["user", "project", "session"]
MemoryKind = Literal["preference", "constraint", "decision", "fact", "todo"]
AgentRunStepKind = Literal[
    "message",
    "planning",
    "routing",
    "approval",
    "tool_call",
    "retrieval",
    "research",
    "report",
    "verification",
    "evaluation",
    "failure",
    "heartbeat",
]
AgentRunStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]
AgentToolChannel = Literal["web", "vector", "mcp", "local"]
AgentToolRiskLevel = Literal["low", "medium", "high"]
ToolInvocationStatus = Literal["pending_approval", "running", "completed", "failed", "skipped"]
ApprovalStatus = Literal["pending", "approved", "rejected", "expired"]
AgentSpecialistId = Literal["repo_signal", "architecture_fit", "ops_risk"]
AgentRoleStatus = Literal["selected", "skipped", "completed", "failed"]
RouteDecisionStatus = Literal["selected", "blocked", "skipped"]
ConflictKind = Literal[
    "agent_overlap",
    "coverage_gap",
    "evidence_gap",
    "constraint_mismatch",
    "tool_gap",
]
ConflictSeverity = Literal["low", "medium", "high"]
AgentEventKind = Literal[
    "message",
    "planning",
    "approval",
    "tool_call",
    "retrieval",
    "routing",
    "research",
    "report",
    "verification",
    "evaluation",
    "failure",
    "heartbeat",
]


class AgentRoleAssignment(BaseModel):
    assignment_id: str
    run_id: str | None = None
    session_id: str | None = None
    agent_id: AgentSpecialistId
    agent_name: str
    status: AgentRoleStatus = "selected"
    reason: str = ""
    plan_item_ids: list[str] = Field(default_factory=list)
    selected_tools: list[ResearchToolName] = Field(default_factory=list)
    exclusive_tools: list[ResearchToolName] = Field(default_factory=list)
    shared_tools: list[ResearchToolName] = Field(default_factory=list)
    evidence_count: int = 0
    confidence: float = 0.0
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    decision_id: str
    run_id: str | None = None
    session_id: str | None = None
    plan_item_id: str
    agent_id: AgentSpecialistId
    agent_name: str
    status: RouteDecisionStatus = "selected"
    mode: Literal["external", "internal", "hybrid"] = "hybrid"
    selected_tools: list[ResearchToolName] = Field(default_factory=list)
    reason: str = ""
    query_count: int = 0
    evidence_count: int = 0
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictRecord(BaseModel):
    conflict_id: str
    run_id: str | None = None
    session_id: str | None = None
    kind: ConflictKind
    severity: ConflictSeverity = "low"
    agent_ids: list[AgentSpecialistId] = Field(default_factory=list)
    plan_item_ids: list[str] = Field(default_factory=list)
    description: str
    resolution: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    resolved: bool = True
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceLedger(BaseModel):
    run_id: str | None = None
    session_id: str | None = None
    total_evidence_count: int = 0
    citation_count: int = 0
    by_agent: dict[str, int] = Field(default_factory=dict)
    by_tool: dict[str, int] = Field(default_factory=dict)
    by_source_kind: dict[str, int] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    utilization_rate: float = 0.0
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkTask(BaseModel):
    task_id: str
    scenario: str = "open_source_adoption_review"
    topic: str
    depth: Literal["quick", "standard", "deep"] = "standard"
    expected_agent_ids: list[AgentSpecialistId] = Field(default_factory=list)
    expected_tools: list[ResearchToolName] = Field(default_factory=list)
    expected_evidence_kinds: list[str] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    min_source_count: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunSummary(BaseModel):
    benchmark_id: str
    run_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    route_precision: float = 0.0
    route_recall: float = 0.0
    specialist_completion_rate: float = 0.0
    tool_success_rate: float = 0.0
    evidence_utilization: float = 0.0
    citation_precision: float = 0.0
    constraint_coverage: float = 0.0
    replay_fidelity: float = 0.0
    latency_ms: int = 0
    cost_usd: float = 0.0
    passed: bool = False
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceProfile(BaseModel):
    workspace_id: str
    name: str
    team_context: str = ""
    default_stack: list[str] = Field(default_factory=list)
    deployment_constraints: list[str] = Field(default_factory=list)
    risk_policy: str = ""
    preferred_sources: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillScript(BaseModel):
    name: str
    path: str
    description: str = ""
    enabled: bool = True
    auto: bool = False
    timeout_seconds: float = 10.0
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchSkill(BaseModel):
    skill_id: str
    name: str
    scenario: str
    trigger_keywords: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    plan_template: list[str] = Field(default_factory=list)
    evaluation_focus: list[str] = Field(default_factory=list)
    skill_type: Literal["builtin", "pack"] = "builtin"
    version: str = "1.0.0"
    source_path: str = ""
    instruction_path: str = ""
    instructions_excerpt: str = ""
    scripts: list[SkillScript] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    memory_id: str
    scope: MemoryScope
    kind: MemoryKind = "fact"
    content: str = Field(min_length=1)
    session_id: str | None = None
    source_message_id: str | None = None
    confidence: float = 0.0
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryExtractionResult(BaseModel):
    source_message_id: str
    session_id: str
    candidates: list[MemoryItem] = Field(default_factory=list)
    accepted: list[MemoryItem] = Field(default_factory=list)
    rejected: list[MemoryItem] = Field(default_factory=list)
    reason: str = ""
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstraintCoverage(BaseModel):
    constraint_id: str
    run_id: str | None = None
    session_id: str | None = None
    content: str
    covered: bool = False
    matched_sections: list[str] = Field(default_factory=list)
    matched_evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentEvent(BaseModel):
    event_id: str
    session_id: str
    run_id: str | None = None
    job_id: str | None = None
    type: Literal["message", "step", "approval", "tool_invocation"]
    kind: AgentEventKind
    status: str = "completed"
    title: str
    summary: str = ""
    actor: str = "agent"
    tool_name: str | None = None
    created_at: str = Field(default_factory=_utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentRunStep(BaseModel):
    step_id: str
    session_id: str
    run_id: str | None = None
    job_id: str | None = None
    kind: AgentRunStepKind
    status: AgentRunStepStatus = "pending"
    title: str
    summary: str = ""
    actor: str = "agent"
    tool_name: str | None = None
    input_preview: str = ""
    output_preview: str = ""
    evidence_count: int = 0
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentToolDefinition(BaseModel):
    name: str
    channel: AgentToolChannel
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    requires_auth: bool = False
    auth_configured: bool = True
    risk_level: AgentToolRiskLevel = "low"
    approval_required: bool = False
    failure_mode: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolInvocation(BaseModel):
    invocation_id: str
    session_id: str
    run_id: str | None = None
    tool_name: str
    status: ToolInvocationStatus = "running"
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_preview: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    error: str = ""
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    approval_id: str
    session_id: str
    invocation_id: str
    reason: str
    requested_action: str
    status: ApprovalStatus = "pending"
    created_at: str = Field(default_factory=_utc_now)
    resolved_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentPlanDraft(BaseModel):
    session_id: str
    workspace_id: str | None = None
    skill_id: str | None = None
    skill_name: str = ""
    skill_reason: str = ""
    research_request: ResearchRequest
    research_brief: str
    plan_items: list[PlanItem] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    required_confirmation: bool = True
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    message_id: str
    session_id: str
    role: AgentMessageRole
    content: str
    intent: AgentMessageIntent = "chat"
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSession(BaseModel):
    session_id: str
    title: str
    session_key: str | None = None
    workspace_id: str | None = None
    selected_skill_id: str | None = None
    context_summary: str = ""
    status: AgentSessionStatus = "collecting"
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    active_run_id: str | None = None
    last_heartbeat_at: str | None = None
    memory_scope_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSessionBundle(BaseModel):
    session: AgentSession
    workspace: WorkspaceProfile | None = None
    selected_skill: ResearchSkill | None = None
    skill_catalog: list[ResearchSkill] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    plan_draft: AgentPlanDraft | None = None
    memory: list[MemoryItem] = Field(default_factory=list)
    steps: list[AgentRunStep] = Field(default_factory=list)
    tool_registry: list[AgentToolDefinition] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    approval_requests: list[ApprovalRequest] = Field(default_factory=list)
    memory_extraction_results: list[MemoryExtractionResult] = Field(default_factory=list)
    memory_evaluation: dict[str, Any] = Field(default_factory=dict)
    constraint_coverage: list[ConstraintCoverage] = Field(default_factory=list)
    role_assignments: list[AgentRoleAssignment] = Field(default_factory=list)
    route_decisions: list[RouteDecision] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    evidence_ledger: EvidenceLedger | None = None
    benchmark_summary: BenchmarkRunSummary | None = None
    active_job: ResearchJob | None = None
    active_run: ResearchRun | None = None
    mcp_status: dict[str, Any] = Field(default_factory=dict)


class AgentTurnResponse(BaseModel):
    session: AgentSession
    workspace: WorkspaceProfile | None = None
    selected_skill: ResearchSkill | None = None
    user_message: AgentMessage | None = None
    assistant_message: AgentMessage | None = None
    plan_draft: AgentPlanDraft | None = None
    memory_updates: list[MemoryItem] = Field(default_factory=list)
    steps: list[AgentRunStep] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    approval_requests: list[ApprovalRequest] = Field(default_factory=list)
    memory_extraction_result: MemoryExtractionResult | None = None
    active_job: ResearchJob | None = None
    active_run: ResearchRun | None = None
    status_url: str | None = None
    result_url: str | None = None
    mcp_status: dict[str, Any] = Field(default_factory=dict)


class SkillExecutionResult(BaseModel):
    skill_id: str
    script_name: str
    status: Literal["completed", "failed"]
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    started_at: str = Field(default_factory=_utc_now)
    finished_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


ResearchRun.model_rebuild()
AgentSessionBundle.model_rebuild()
