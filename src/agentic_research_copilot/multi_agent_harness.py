from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .schemas import (
    AgentRoleAssignment,
    AgentSpecialistId,
    BenchmarkRunSummary,
    BenchmarkTask,
    ConflictRecord,
    EvidenceItem,
    EvidenceLedger,
    PlanItem,
    ResearchRequest,
    ResearchRun,
    RetrievalRoute,
    RouteDecision,
    RunTraceEvent,
)


@dataclass(frozen=True)
class SpecialistProfile:
    agent_id: AgentSpecialistId
    agent_name: str
    responsibility: str
    trigger_keywords: tuple[str, ...]
    preferred_tools: tuple[str, ...]
    exclusive_tools: tuple[str, ...] = ()
    shared_tools: tuple[str, ...] = ("web_search", "vector_retrieval")


SPECIALIST_PROFILES: dict[AgentSpecialistId, SpecialistProfile] = {
    "repo_signal": SpecialistProfile(
        agent_id="repo_signal",
        agent_name="RepoSignalAgent",
        responsibility="Checks repository facts, maintenance signals, code/issue/PR/release evidence, and source authority.",
        trigger_keywords=(
            "repo",
            "github",
            "仓库",
            "开源",
            "代码",
            "源码",
            "readme",
            "issue",
            "issues",
            "pr",
            "pull request",
            "release",
            "maintenance",
            "stars",
            "license",
            "社区",
            "活跃",
        ),
        preferred_tools=("mcp_tool", "web_search"),
        exclusive_tools=("mcp_tool",),
        shared_tools=("web_search",),
    ),
    "architecture_fit": SpecialistProfile(
        agent_id="architecture_fit",
        agent_name="ArchitectureFitAgent",
        responsibility="Checks architecture fit, API/runtime design, integration cost, workflow semantics, and local KB alignment.",
        trigger_keywords=(
            "architecture",
            "架构",
            "workflow",
            "runtime",
            "stategraph",
            "checkpoint",
            "graph",
            "dag",
            "agent",
            "rag",
            "retrieval",
            "fastapi",
            "api",
            "sdk",
            "集成",
            "适配",
            "可维护",
            "可观测",
        ),
        preferred_tools=("vector_retrieval", "web_search"),
        shared_tools=("web_search", "vector_retrieval"),
    ),
    "ops_risk": SpecialistProfile(
        agent_id="ops_risk",
        agent_name="OpsRiskAgent",
        responsibility="Checks deployment, rollback, compliance, dependency, cost, reliability, and operational risk constraints.",
        trigger_keywords=(
            "ops",
            "risk",
            "风险",
            "部署",
            "运维",
            "docker",
            "compose",
            "rollback",
            "回滚",
            "security",
            "安全",
            "compliance",
            "合规",
            "cost",
            "成本",
            "latency",
            "超时",
            "可靠",
            "单机",
            "production",
            "依赖",
        ),
        preferred_tools=("vector_retrieval", "web_search", "mcp_tool"),
        shared_tools=("web_search", "vector_retrieval", "mcp_tool"),
    ),
}


def specialist_catalog() -> list[dict[str, Any]]:
    return [
        {
            "agent_id": profile.agent_id,
            "agent_name": profile.agent_name,
            "responsibility": profile.responsibility,
            "trigger_keywords": list(profile.trigger_keywords),
            "preferred_tools": list(profile.preferred_tools),
            "exclusive_tools": list(profile.exclusive_tools),
            "shared_tools": list(profile.shared_tools),
        }
        for profile in SPECIALIST_PROFILES.values()
    ]


def role_preview_for_plan(
    request: ResearchRequest,
    plan: Sequence[PlanItem],
    *,
    skill_id: str | None = None,
    workspace_context: str = "",
) -> dict[str, Any]:
    selected = select_specialists(request, skill_id=skill_id, workspace_context=workspace_context, plan=plan)
    item_roles = {
        item.id: [
            role_id
            for role_id in selected
            if _score_role_for_text(role_id, _plan_item_text(item)) > 0
        ]
        for item in plan
    }
    for item_id, roles in item_roles.items():
        if not roles and selected:
            item_roles[item_id] = [_best_role_for_text(_plan_item_text(next(item for item in plan if item.id == item_id)), selected)]
    return {
        "selected_agents": [SPECIALIST_PROFILES[role_id].agent_name for role_id in selected],
        "selected_agent_ids": selected,
        "item_roles": item_roles,
        "reason": _selection_reason(selected),
    }


def enrich_research_run(
    run: ResearchRun,
    *,
    task: BenchmarkTask | None = None,
    session_id: str | None = None,
    skill_id: str | None = None,
    workspace_context: str = "",
    replay_source_run_id: str | None = None,
) -> ResearchRun:
    assignments = build_role_assignments(
        run.request,
        run.plan,
        run.retrieval_routes,
        run.evidence,
        run_id=run.run_id,
        session_id=session_id,
        skill_id=skill_id,
        workspace_context=workspace_context,
    )
    route_decisions = build_route_decisions(
        run.request,
        run.plan,
        run.retrieval_routes,
        run.evidence,
        assignments,
        run_id=run.run_id,
        session_id=session_id,
    )
    evidence_ledger = build_evidence_ledger(
        run.evidence,
        assignments,
        run.report.citations if run.report else [],
        run_id=run.run_id,
        session_id=session_id,
    )
    conflicts = detect_conflicts(
        run,
        assignments,
        route_decisions,
        evidence_ledger,
        run_id=run.run_id,
        session_id=session_id,
    )
    summary = summarize_run(
        run,
        task=task,
        assignments=assignments,
        route_decisions=route_decisions,
        conflicts=conflicts,
        evidence_ledger=evidence_ledger,
        replay_source_run_id=replay_source_run_id,
    )
    trace = _append_harness_trace(run, assignments, route_decisions, conflicts, evidence_ledger, summary)
    return run.model_copy(
        update={
            "role_assignments": assignments,
            "route_decisions": route_decisions,
            "conflicts": conflicts,
            "evidence_ledger": evidence_ledger,
            "benchmark_summary": summary,
            "trace": trace,
        }
    )


def select_specialists(
    request: ResearchRequest,
    *,
    skill_id: str | None = None,
    workspace_context: str = "",
    plan: Sequence[PlanItem] = (),
) -> list[AgentSpecialistId]:
    text = " ".join(
        [
            request.topic,
            skill_id or "",
            workspace_context,
            " ".join(_plan_item_text(item) for item in plan),
        ]
    )
    scores = {role_id: _score_role_for_text(role_id, text) for role_id in SPECIALIST_PROFILES}
    lower = text.lower()

    if skill_id == "open_source_adoption_review" or any(word in lower for word in ("adoption", "采用", "引入", "repo", "github")):
        scores["repo_signal"] += 3
        scores["architecture_fit"] += 2
        if any(word in lower for word in ("团队", "约束", "deploy", "部署", "risk", "风险", "rollback", "回滚", "docker")):
            scores["ops_risk"] += 2
    if skill_id == "architecture_tradeoff_memo" or any(word in lower for word in ("tradeoff", "对比", "比较", "选型", "architecture")):
        scores["architecture_fit"] += 3
    if skill_id == "demo_readiness_risk_review" or any(word in lower for word in ("demo", "秋招", "面试", "展示")):
        scores["ops_risk"] += 2
        scores["architecture_fit"] += 1

    selected = [role_id for role_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score > 0]
    if not selected:
        selected = ["architecture_fit"]
    return selected


def build_role_assignments(
    request: ResearchRequest,
    plan: Sequence[PlanItem],
    routes: Sequence[RetrievalRoute],
    evidence: Sequence[EvidenceItem],
    *,
    run_id: str | None = None,
    session_id: str | None = None,
    skill_id: str | None = None,
    workspace_context: str = "",
) -> list[AgentRoleAssignment]:
    selected = select_specialists(request, skill_id=skill_id, workspace_context=workspace_context, plan=plan)
    route_lookup = {route.plan_item_id: route for route in routes}
    evidence_count_by_item = Counter(str(item.metadata.get("plan_item_id") or "") for item in evidence)
    plan_ids_by_role: dict[AgentSpecialistId, list[str]] = {role_id: [] for role_id in selected}
    role_scores_by_item = {
        item.id: {role_id: _score_role_for_text(role_id, _plan_item_text(item)) for role_id in selected}
        for item in plan
    }

    for item in plan:
        scored = role_scores_by_item.get(item.id, {})
        item_roles = [role_id for role_id, score in scored.items() if score > 0]
        if not item_roles and selected:
            item_roles = [_best_role_for_text(_plan_item_text(item), selected)]
        for role_id in item_roles:
            if item.id not in plan_ids_by_role.setdefault(role_id, []):
                plan_ids_by_role[role_id].append(item.id)

    for role_id, item_ids in plan_ids_by_role.items():
        if item_ids or not plan:
            continue
        best_item = max(plan, key=lambda item: role_scores_by_item.get(item.id, {}).get(role_id, 0))
        item_ids.append(best_item.id)

    assignments: list[AgentRoleAssignment] = []
    for role_id in selected:
        profile = SPECIALIST_PROFILES[role_id]
        item_ids = plan_ids_by_role.get(role_id, [])
        tools = _tools_for_items(item_ids, route_lookup, profile)
        evidence_count = sum(evidence_count_by_item[item_id] for item_id in item_ids)
        status = "completed" if item_ids and tools else "selected"
        assignments.append(
            AgentRoleAssignment(
                assignment_id=_stable_id("role_assignment", run_id, session_id, role_id, ",".join(item_ids)),
                run_id=run_id,
                session_id=session_id,
                agent_id=role_id,
                agent_name=profile.agent_name,
                status=status,
                reason=_assignment_reason(role_id, request, item_ids),
                plan_item_ids=item_ids,
                selected_tools=tools,
                exclusive_tools=[tool for tool in profile.exclusive_tools if tool in tools],
                shared_tools=[tool for tool in profile.shared_tools if tool in tools],
                evidence_count=evidence_count,
                confidence=_confidence_for_assignment(item_ids, evidence_count, tools),
                metadata={
                    "responsibility": profile.responsibility,
                    "preferred_tools": list(profile.preferred_tools),
                    "skill_id": skill_id,
                },
            )
        )
    return assignments


def build_route_decisions(
    request: ResearchRequest,
    plan: Sequence[PlanItem],
    routes: Sequence[RetrievalRoute],
    evidence: Sequence[EvidenceItem],
    assignments: Sequence[AgentRoleAssignment],
    *,
    run_id: str | None = None,
    session_id: str | None = None,
) -> list[RouteDecision]:
    assignment_lookup: dict[str, list[AgentRoleAssignment]] = {}
    for assignment in assignments:
        for item_id in assignment.plan_item_ids:
            assignment_lookup.setdefault(item_id, []).append(assignment)
    route_lookup = {route.plan_item_id: route for route in routes}
    evidence_count_by_item = Counter(str(item.metadata.get("plan_item_id") or "") for item in evidence)
    decisions: list[RouteDecision] = []
    for item in plan:
        route = route_lookup.get(item.id)
        assignment = _primary_assignment_for_item(item, assignment_lookup.get(item.id, []), request)
        if assignment is None:
            continue
        query_count = 0
        if route is not None:
            query_count = len(route.web_queries) + len(route.internal_queries)
        decisions.append(
            RouteDecision(
                decision_id=_stable_id("route_decision", run_id, session_id, item.id, assignment.agent_id),
                run_id=run_id,
                session_id=session_id,
                plan_item_id=item.id,
                agent_id=assignment.agent_id,
                agent_name=assignment.agent_name,
                status="selected",
                mode=route.mode if route is not None else "hybrid",
                selected_tools=list(route.selected_tools if route is not None else assignment.selected_tools),
                reason=route.reason if route is not None else assignment.reason,
                query_count=query_count,
                evidence_count=evidence_count_by_item[item.id],
                metadata={
                    "question": item.question,
                    "purpose": item.purpose,
                    "assignment_id": assignment.assignment_id,
                    "requires_research": item.requires_research,
                },
            )
        )
    return decisions


def build_evidence_ledger(
    evidence: Sequence[EvidenceItem],
    assignments: Sequence[AgentRoleAssignment],
    citations: Sequence[EvidenceItem],
    *,
    run_id: str | None = None,
    session_id: str | None = None,
) -> EvidenceLedger:
    item_to_agents: dict[str, list[str]] = {}
    for assignment in assignments:
        for item_id in assignment.plan_item_ids:
            item_to_agents.setdefault(item_id, []).append(assignment.agent_id)
    by_agent: Counter[str] = Counter()
    by_tool: Counter[str] = Counter()
    by_source_kind: Counter[str] = Counter()
    evidence_ids: list[str] = []
    used = 0
    for item in evidence:
        evidence_id = _evidence_id(item)
        evidence_ids.append(evidence_id)
        plan_item_id = str(item.metadata.get("plan_item_id") or "")
        agents = item_to_agents.get(plan_item_id, [])
        if agents:
            used += 1
        for agent_id in agents or ["unassigned"]:
            by_agent[agent_id] += 1
        by_source_kind[item.kind or "unknown"] += 1
        by_tool[_tool_for_evidence(item)] += 1

    citation_keys = {_evidence_id(item) for item in citations}
    utilization = len(citation_keys & set(evidence_ids)) / max(1, len(evidence_ids))
    if not citation_keys and evidence:
        utilization = used / max(1, len(evidence))
    return EvidenceLedger(
        run_id=run_id,
        session_id=session_id,
        total_evidence_count=len(evidence),
        citation_count=len(citations),
        by_agent=dict(by_agent),
        by_tool=dict(by_tool),
        by_source_kind=dict(by_source_kind),
        evidence_ids=evidence_ids,
        utilization_rate=round(utilization, 4),
        metadata={
            "assigned_evidence_count": used,
            "unassigned_evidence_count": max(0, len(evidence) - used),
        },
    )


def detect_conflicts(
    run: ResearchRun,
    assignments: Sequence[AgentRoleAssignment],
    route_decisions: Sequence[RouteDecision],
    evidence_ledger: EvidenceLedger,
    *,
    run_id: str | None = None,
    session_id: str | None = None,
) -> list[ConflictRecord]:
    conflicts: list[ConflictRecord] = []
    agents_by_item: dict[str, list[AgentSpecialistId]] = {}
    for assignment in assignments:
        for item_id in assignment.plan_item_ids:
            agents_by_item.setdefault(item_id, []).append(assignment.agent_id)
    for item_id, agent_ids in sorted(agents_by_item.items()):
        unique_ids = _unique(agent_ids)
        if len(unique_ids) <= 1:
            continue
        conflicts.append(
            ConflictRecord(
                conflict_id=_stable_id("conflict_overlap", run_id, session_id, item_id, ",".join(unique_ids)),
                run_id=run_id,
                session_id=session_id,
                kind="agent_overlap",
                severity="low",
                agent_ids=unique_ids,
                plan_item_ids=[item_id],
                description=f"Plan item {item_id} is relevant to multiple specialist lanes.",
                resolution="Keep the shared ownership visible; Writer synthesizes the final memo and Verifier checks citation coverage.",
                resolved=True,
            )
        )

    missing_roles = [assignment for assignment in assignments if not assignment.plan_item_ids]
    for assignment in missing_roles:
        conflicts.append(
            ConflictRecord(
                conflict_id=_stable_id("conflict_coverage", run_id, session_id, assignment.agent_id),
                run_id=run_id,
                session_id=session_id,
                kind="coverage_gap",
                severity="medium",
                agent_ids=[assignment.agent_id],
                description=f"{assignment.agent_name} was selected but has no plan item ownership.",
                resolution="Planner should add a focused item or skip this specialist in the next revision.",
                resolved=False,
            )
        )

    if run.evaluation is not None:
        if run.evaluation.insufficient_plan_items or run.evaluation.unsupported_sections:
            conflicts.append(
                ConflictRecord(
                    conflict_id=_stable_id("conflict_evidence_gap", run_id, session_id),
                    run_id=run_id,
                    session_id=session_id,
                    kind="evidence_gap",
                    severity="high",
                    agent_ids=[decision.agent_id for decision in route_decisions],
                    plan_item_ids=run.evaluation.insufficient_plan_items,
                    description="Verifier/evaluator found insufficient evidence or unsupported sections.",
                    resolution="Run remains inspectable; add sources, rerun, or lower scope before treating the memo as final.",
                    resolved=run.evaluation.passed,
                    metadata={
                        "unsupported_sections": run.evaluation.unsupported_sections,
                        "notes": run.evaluation.notes,
                    },
                )
            )

    if evidence_ledger.total_evidence_count == 0:
        conflicts.append(
            ConflictRecord(
                conflict_id=_stable_id("conflict_no_evidence", run_id, session_id),
                run_id=run_id,
                session_id=session_id,
                kind="evidence_gap",
                severity="high",
                agent_ids=[assignment.agent_id for assignment in assignments],
                description="No evidence reached the shared evidence ledger.",
                resolution="Do not trust the memo; inspect provider/search/tool configuration and rerun.",
                resolved=False,
            )
        )
    return conflicts


def summarize_run(
    run: ResearchRun,
    *,
    task: BenchmarkTask | None,
    assignments: Sequence[AgentRoleAssignment],
    route_decisions: Sequence[RouteDecision],
    conflicts: Sequence[ConflictRecord],
    evidence_ledger: EvidenceLedger,
    replay_source_run_id: str | None = None,
) -> BenchmarkRunSummary:
    selected_agents = {assignment.agent_id for assignment in assignments}
    completed_agents = {assignment.agent_id for assignment in assignments if assignment.status == "completed"}
    if task is not None and task.expected_agent_ids:
        expected_agents = set(task.expected_agent_ids)
        route_precision = len(selected_agents & expected_agents) / max(1, len(selected_agents))
        route_recall = len(selected_agents & expected_agents) / max(1, len(expected_agents))
    else:
        route_precision = 1.0 if selected_agents else 0.0
        route_recall = 1.0 if selected_agents else 0.0

    tool_calls = [event for event in run.trace if event.kind == "tool_call"]
    successful_tool_calls = [event for event in tool_calls if event.status in {"completed", "started"}]
    tool_success = len(successful_tool_calls) / max(1, len(tool_calls)) if tool_calls else 1.0
    citation_precision = run.evaluation.citation_precision if run.evaluation is not None else 0.0
    constraint_coverage = _constraint_coverage_proxy(run)
    specialist_completion = len(completed_agents) / max(1, len(assignments))
    replay_fidelity = 1.0 if replay_source_run_id else 0.0
    unresolved_conflicts = [conflict for conflict in conflicts if not conflict.resolved]
    notes: list[str] = []
    if route_recall < 1.0:
        notes.append("Route recall missed at least one expected specialist.")
    if specialist_completion < 1.0:
        notes.append("At least one selected specialist did not receive evidence.")
    if unresolved_conflicts:
        notes.append("Unresolved conflicts remain in the harness output.")
    if replay_source_run_id:
        notes.append(f"Replay reused frozen artifacts from {replay_source_run_id}.")

    passed = (
        route_precision >= 0.75
        and route_recall >= 0.75
        and specialist_completion >= 0.66
        and tool_success >= 0.8
        and evidence_ledger.utilization_rate >= 0.15
        and citation_precision >= 0.8
        and not any(conflict.severity == "high" and not conflict.resolved for conflict in conflicts)
    )
    return BenchmarkRunSummary(
        benchmark_id=_stable_id("benchmark_summary", run.run_id, task.task_id if task else "", replay_source_run_id),
        run_id=run.run_id,
        task_id=task.task_id if task else None,
        route_precision=round(route_precision, 4),
        route_recall=round(route_recall, 4),
        specialist_completion_rate=round(specialist_completion, 4),
        tool_success_rate=round(tool_success, 4),
        evidence_utilization=evidence_ledger.utilization_rate,
        citation_precision=round(citation_precision, 4),
        constraint_coverage=constraint_coverage,
        replay_fidelity=replay_fidelity,
        latency_ms=run.duration_ms or 0,
        passed=passed,
        notes=notes,
        metadata={
            "selected_agent_ids": sorted(selected_agents),
            "completed_agent_ids": sorted(completed_agents),
            "conflict_count": len(conflicts),
            "unresolved_conflict_count": len(unresolved_conflicts),
            "route_decision_count": len(route_decisions),
            "replay_source_run_id": replay_source_run_id,
        },
    )


def replay_from_frozen_run(run: ResearchRun) -> ResearchRun:
    replay_run_id = str(uuid.uuid4())
    frozen_trace = [
        event
        for event in run.trace
        if event.step not in {"harness.role_assignment", "harness.evaluation", "replay.frozen"}
    ]
    trace = [
        *frozen_trace,
        RunTraceEvent(
            kind="step",
            actor="replay",
            message="Replayed from frozen run artifacts without re-calling live tools.",
            step="replay.frozen",
            metadata={"source_run_id": run.run_id, "frozen_tool_results": True},
        ),
    ]
    replayed = run.model_copy(
        deep=True,
        update={
            "run_id": replay_run_id,
            "job_id": None,
            "trace": trace,
            "metadata": {
                **run.metadata,
                "replay_source_run_id": run.run_id,
                "replay_mode": "frozen_artifacts",
            },
        },
    )
    return enrich_research_run(
        replayed,
        replay_source_run_id=run.run_id,
    )


def score_task_against_run(task: BenchmarkTask, run: ResearchRun) -> BenchmarkRunSummary:
    return summarize_run(
        run,
        task=task,
        assignments=run.role_assignments,
        route_decisions=run.route_decisions,
        conflicts=run.conflicts,
        evidence_ledger=run.evidence_ledger or EvidenceLedger(run_id=run.run_id),
    )


def _append_harness_trace(
    run: ResearchRun,
    assignments: Sequence[AgentRoleAssignment],
    route_decisions: Sequence[RouteDecision],
    conflicts: Sequence[ConflictRecord],
    evidence_ledger: EvidenceLedger,
    summary: BenchmarkRunSummary,
) -> list[RunTraceEvent]:
    trace = list(run.trace)
    if any(event.step == "harness.role_assignment" for event in trace):
        return trace
    trace.append(
        RunTraceEvent(
            kind="step",
            actor="multi_agent_harness",
            message="Assigned plan items to specialist lanes.",
            step="harness.role_assignment",
            metadata={
                "role_assignments": [assignment.model_dump(mode="json") for assignment in assignments],
                "selected_agent_ids": [assignment.agent_id for assignment in assignments],
            },
        )
    )
    trace.append(
        RunTraceEvent(
            kind="step",
            actor="multi_agent_harness",
            message="Built route decisions, evidence ledger, conflicts, and benchmark proxy metrics.",
            step="harness.evaluation",
            metadata={
                "route_decision_count": len(route_decisions),
                "conflict_count": len(conflicts),
                "evidence_ledger": evidence_ledger.model_dump(mode="json"),
                "benchmark_summary": summary.model_dump(mode="json"),
            },
        )
    )
    return trace


def _selection_reason(selected: Sequence[AgentSpecialistId]) -> str:
    names = ", ".join(SPECIALIST_PROFILES[role_id].agent_name for role_id in selected)
    return f"Selected specialist lanes for this bounded open-source adoption workflow: {names}."


def _assignment_reason(role_id: AgentSpecialistId, request: ResearchRequest, item_ids: Sequence[str]) -> str:
    profile = SPECIALIST_PROFILES[role_id]
    scope = f"{len(item_ids)} plan item(s)" if item_ids else "no plan item"
    return f"{profile.agent_name} owns {scope} because the request needs {profile.responsibility.lower()}"


def _tools_for_items(
    item_ids: Sequence[str],
    route_lookup: dict[str, RetrievalRoute],
    profile: SpecialistProfile,
) -> list[str]:
    tools: list[str] = []
    for item_id in item_ids:
        route = route_lookup.get(item_id)
        if route is None:
            continue
        tools.extend(route.selected_tools)
    if not tools:
        tools.extend(profile.preferred_tools[:1])
    return _unique([tool for tool in tools if tool in {"web_search", "vector_retrieval", "mcp_tool"}])


def _primary_assignment_for_item(
    item: PlanItem,
    assignments: Sequence[AgentRoleAssignment],
    request: ResearchRequest,
) -> AgentRoleAssignment | None:
    if not assignments:
        return None
    text = f"{request.topic} {_plan_item_text(item)}"
    return max(assignments, key=lambda assignment: _score_role_for_text(assignment.agent_id, text))


def _best_role_for_text(text: str, candidates: Sequence[AgentSpecialistId]) -> AgentSpecialistId:
    if not candidates:
        return "architecture_fit"
    return max(candidates, key=lambda role_id: _score_role_for_text(role_id, text))


def _confidence_for_assignment(item_ids: Sequence[str], evidence_count: int, tools: Sequence[str]) -> float:
    score = 0.45
    score += min(0.25, len(item_ids) * 0.08)
    score += min(0.2, evidence_count * 0.03)
    score += min(0.1, len(tools) * 0.03)
    return round(min(0.95, score), 4)


def _score_role_for_text(role_id: AgentSpecialistId, text: str) -> int:
    normalized = text.lower()
    return sum(1 for keyword in SPECIALIST_PROFILES[role_id].trigger_keywords if keyword.lower() in normalized)


def _plan_item_text(item: PlanItem) -> str:
    return " ".join([item.question, item.purpose, item.search_query or ""])


def _tool_for_evidence(item: EvidenceItem) -> str:
    channel = str(item.metadata.get("source_channel") or "")
    if channel == "mcp" or item.kind == "mcp":
        return "mcp_tool"
    if item.kind == "document-chunk" or channel in {"internal", "local"}:
        return "vector_retrieval"
    if item.kind == "run-artifact":
        return "run_artifact"
    return "web_search"


def _constraint_coverage_proxy(run: ResearchRun) -> float:
    if run.evaluation is None:
        return 0.0
    notes = " ".join(run.evaluation.notes).lower()
    if "constraint coverage" in notes and "failed" in notes:
        return 0.0
    if "constraint coverage" in notes:
        return 0.5
    return 1.0


def _evidence_id(item: EvidenceItem) -> str:
    stable = item.url or f"{item.kind}:{item.source}:{item.title}:{item.snippet or item.content or ''}"
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]


def _stable_id(*parts: str | None) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "::".join(part or "" for part in parts)))


def _unique(values: Iterable[str]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
