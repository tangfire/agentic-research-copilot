from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .schemas import (
    AgentHandoff,
    CorpusProfile,
    EvidenceItem,
    ResearchNote,
    ResearchRequest,
    ResearchRun,
    RunCheckpoint,
    RunTraceEvent,
    SearchQuery,
    SupervisorDecisionContract,
)

if TYPE_CHECKING:
    from .pipeline import ResearchCopilot


class ResearchGraphState(TypedDict, total=False):
    request: ResearchRequest
    job_id: str | None
    run_id: str
    start: datetime
    checkpoints: list[RunCheckpoint]
    trace: list[RunTraceEvent]
    handoffs: list[AgentHandoff]
    revision_notes: list[str]
    revision_count: int
    failure_reason: str | None
    final_status: str
    final_research_brief: str | None
    final_corpus_profile: CorpusProfile | None
    final_supervisor_decision: SupervisorDecisionContract | None
    final_route_hints: list[Any]
    final_plan: list[Any]
    final_search_queries: list[SearchQuery]
    final_retrieval_routes: list[Any]
    final_notes: list[ResearchNote]
    final_evidence: list[EvidenceItem]
    final_web_hits: list[EvidenceItem]
    final_document_hits: list[EvidenceItem]
    final_report: Any
    final_evaluation: Any
    final_issues: list[str]
    needs_revision: bool
    revision_reason: str | None
    run: ResearchRun


class LangGraphResearchRuntime:
    """LangGraph orchestration for the research copilot runtime."""

    def __init__(self, copilot: ResearchCopilot) -> None:
        self.copilot = copilot
        self._checkpoint_context = None
        self._checkpoint_connection = None
        self.checkpointer, self.checkpointer_kind, self.checkpointer_path = self._build_checkpointer()
        self.graph = self._build_graph()

    def run(self, request: ResearchRequest, *, job_id: str | None = None) -> ResearchRun:
        run_id = str(uuid.uuid4())
        try:
            state = self.graph.invoke(
                {"request": request, "job_id": job_id, "run_id": run_id},
                config={"configurable": {"thread_id": run_id}},
            )
            return state["run"]
        finally:
            self.close()

    def close(self) -> None:
        if self._checkpoint_context is not None:
            self._checkpoint_context.__exit__(None, None, None)
            self._checkpoint_context = None
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None

    def _build_checkpointer(self):
        settings = self.copilot.settings
        if settings.langgraph_checkpointer != "sqlite":
            return MemorySaver(), "memory", ""
        path = _resolve_checkpoint_path(settings.langgraph_checkpoint_path)
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except Exception as exc:
            if settings.strict_providers:
                raise RuntimeError("LangGraph SQLite checkpointer package is required in strict mode.") from exc
            return MemorySaver(), "memory_fallback_missing_sqlite_package", str(path)

        try:
            if hasattr(SqliteSaver, "from_conn_string"):
                context = SqliteSaver.from_conn_string(str(path))
                self._checkpoint_context = context
                return context.__enter__(), "sqlite", str(path)
            connection = sqlite3.connect(str(path), check_same_thread=False)
            self._checkpoint_connection = connection
            return SqliteSaver(connection), "sqlite", str(path)
        except Exception as exc:
            if settings.strict_providers:
                raise RuntimeError(f"LangGraph SQLite checkpointer could not open {path}: {exc}") from exc
            return MemorySaver(), "memory_fallback_sqlite_open_failed", str(path)

    def _build_graph(self):
        builder = StateGraph(ResearchGraphState)
        builder.add_node("supervisor_start", self._supervisor_start)
        builder.add_node("planner", self._planner)
        builder.add_node("research_supervisor", self._research_supervisor)
        builder.add_node("parallel_research", self._parallel_research)
        builder.add_node("reporter", self._reporter)
        builder.add_node("verifier_evaluator", self._verifier_evaluator)
        builder.add_node("revision_prepare", self._revision_prepare)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "supervisor_start")
        builder.add_edge("supervisor_start", "planner")
        builder.add_edge("planner", "research_supervisor")
        builder.add_edge("research_supervisor", "parallel_research")
        builder.add_edge("parallel_research", "reporter")
        builder.add_edge("reporter", "verifier_evaluator")
        builder.add_conditional_edges(
            "verifier_evaluator",
            self._route_after_verification,
            {"revise": "revision_prepare", "finish": "finalize"},
        )
        builder.add_edge("revision_prepare", "planner")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _supervisor_start(self, state: ResearchGraphState) -> ResearchGraphState:
        request = state["request"]
        job_id = state.get("job_id")
        run_id = state.get("run_id") or str(uuid.uuid4())
        start = datetime.now(timezone.utc)
        self.copilot.telemetry.emit(
            "run.start",
            request.topic,
            run_id=run_id,
            job_id=job_id,
            depth=request.depth,
            orchestration_runtime="langgraph",
        )
        update: ResearchGraphState = {
            "run_id": run_id,
            "start": start,
            "checkpoints": [],
            "trace": [],
            "handoffs": [],
            "revision_notes": [],
            "revision_count": 0,
            "failure_reason": None,
            "final_status": "completed",
            "final_research_brief": None,
            "final_corpus_profile": None,
            "final_supervisor_decision": None,
            "final_route_hints": [],
            "final_plan": [],
            "final_search_queries": [],
            "final_retrieval_routes": [],
            "final_notes": [],
            "final_evidence": [],
            "final_web_hits": [],
            "final_document_hits": [],
            "final_report": None,
            "final_evaluation": None,
            "final_issues": [],
            "needs_revision": False,
            "revision_reason": None,
        }
        state.update(update)
        self._checkpoint(
            state,
            "langgraph.runtime",
            {
                "runtime": "langgraph",
                "checkpointer": self.checkpointer_kind,
                "checkpoint_path": self.checkpointer_path,
                "graph": [
                    "supervisor_start",
                    "planner",
                    "research_supervisor",
                    "parallel_research",
                    "reporter",
                    "verifier_evaluator",
                    "finalize",
                ],
            },
        )
        self._checkpoint(
            state,
            "supervisor.start",
            {
                "topic": request.topic,
                "depth": request.depth,
                "max_revisions": request.max_revisions,
            },
        )
        self._append_trace(
            state,
            kind="step",
            actor="supervisor",
            message="Started the LangGraph research workflow.",
            step="langgraph.start",
            runtime="langgraph",
        )
        return state

    def _planner(self, state: ResearchGraphState) -> ResearchGraphState:
        request = state["request"]
        revision_count = state["revision_count"]
        revision_notes = state.get("revision_notes", [])
        corpus_profile = self.copilot.documents.profile()
        self._record_handoff(
            state,
            "supervisor",
            "planner",
            "planning",
            "Build the plan and research brief.",
            revision_count,
        )
        planner_contract = self.copilot.planner.draft(
            request,
            corpus_profile=corpus_profile,
            revision_count=revision_count,
            revision_notes=revision_notes,
        )
        planner_usage = self.copilot.planner.last_usage
        research_brief = planner_contract.research_brief
        plan = [item.model_copy() for item in planner_contract.plan]
        route_hints = self.copilot.router.build_routes(request, research_brief, plan, corpus_profile)

        self._append_trace(
            state,
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
            runtime="langgraph",
        )
        self._checkpoint(
            state,
            "plan.generated",
            {
                "research_brief": research_brief,
                "plan_count": len(plan),
                "candidate_route_count": len(route_hints),
                "document_count": corpus_profile.document_count,
                "model_provider": getattr(planner_usage, "provider", None),
                "model_name": getattr(planner_usage, "model", None),
                "revision_count": revision_count,
            },
        )

        return {
            "final_research_brief": research_brief,
            "final_corpus_profile": corpus_profile,
            "final_route_hints": route_hints,
            "final_plan": plan,
            "final_search_queries": [],
            "final_retrieval_routes": [],
            "needs_revision": False,
            "revision_reason": None,
        }

    def _research_supervisor(self, state: ResearchGraphState) -> ResearchGraphState:
        request = state["request"]
        revision_count = state["revision_count"]
        revision_notes = state.get("revision_notes", [])
        research_brief = state["final_research_brief"] or request.topic
        corpus_profile = state["final_corpus_profile"] or CorpusProfile()
        plan = state["final_plan"]
        route_hints = state.get("final_route_hints", [])

        self._record_handoff(
            state,
            "planner",
            "research_supervisor",
            "supervisor.decision",
            "Decide which research units to delegate and which evidence tools each unit should use.",
            revision_count,
        )
        supervisor_decision = self.copilot.supervisor_agent.decide(
            request,
            research_brief=research_brief,
            plan=plan,
            retrieval_routes=route_hints,
            corpus_profile=corpus_profile,
            revision_count=revision_count,
            revision_notes=revision_notes,
        )
        supervisor_usage = self.copilot.supervisor_agent.last_usage
        retrieval_routes = self.copilot._routes_from_supervisor_decision(
            request=request,
            research_brief=research_brief,
            plan=plan,
            supervisor_decision=supervisor_decision,
            route_hints=route_hints,
            corpus_profile=corpus_profile,
        )
        search_queries = self.copilot.workflow.build_queries(plan, retrieval_routes, revision_count=revision_count)

        self._append_trace(
            state,
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
            runtime="langgraph",
        )
        self._checkpoint(
            state,
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
        self._checkpoint(
            state,
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

        route_lookup = {route.plan_item_id: route for route in retrieval_routes}
        item_lookup = {item.id: item for item in plan}
        for call in supervisor_decision.tool_calls:
            if call.name == "think_tool":
                self._append_trace(
                    state,
                    kind="tool_call",
                    actor="research_supervisor",
                    message=call.reflection or call.rationale,
                    step="supervisor.think",
                    tool_name="think_tool",
                    rationale=call.rationale,
                    runtime="langgraph",
                )
                continue
            if call.name == "ResearchComplete":
                self._append_trace(
                    state,
                    kind="tool_call",
                    actor="research_supervisor",
                    message=call.reflection or call.rationale,
                    step="supervisor.complete.criteria",
                    tool_name="ResearchComplete",
                    completion_criteria=supervisor_decision.completion_criteria,
                    runtime="langgraph",
                )
                continue
            if call.name != "ConductResearch":
                continue

            for plan_item_id in call.plan_item_ids:
                route = route_lookup.get(plan_item_id)
                item = item_lookup.get(plan_item_id)
                if route is None or item is None:
                    continue
                item.status = "running"
                self._record_handoff(
                    state,
                    "research_supervisor",
                    "researcher",
                    f"research.{item.id}",
                    route.reason,
                    revision_count,
                )
                self._append_trace(
                    state,
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
                    runtime="langgraph",
                )
                if route.mode in {"internal", "hybrid"} and request.include_private_docs and corpus_profile.has_private_docs:
                    self._record_handoff(
                        state,
                        "researcher",
                        "retriever",
                        f"retrieve.{item.id}",
                        route.reason,
                        revision_count,
                    )

        return {
            "final_supervisor_decision": supervisor_decision,
            "final_search_queries": search_queries,
            "final_retrieval_routes": retrieval_routes,
        }

    def _parallel_research(self, state: ResearchGraphState) -> ResearchGraphState:
        request = state["request"]
        plan = state["final_plan"]
        retrieval_routes = state["final_retrieval_routes"]
        corpus_profile = state["final_corpus_profile"] or CorpusProfile()
        research_brief = state["final_research_brief"] or request.topic
        revision_count = state["revision_count"]
        supervisor_decision = state.get("final_supervisor_decision")
        route_lookup = {route.plan_item_id: route for route in retrieval_routes}
        supervisor_worker_limit = (
            supervisor_decision.max_concurrent_research_units
            if supervisor_decision is not None
            else self.copilot.settings.research_max_workers
        )

        self._checkpoint(
            state,
            "research.parallel.started",
            {
                "worker_count": min(
                    max(1, self.copilot.settings.research_max_workers),
                    max(1, supervisor_worker_limit),
                    max(1, len(plan)),
                ),
                "plan_count": len(plan),
                "runtime": "langgraph",
                "execution_mode": "specialist_worker",
            },
        )
        research_results = self.copilot._research_plan_items(
            request=request,
            plan=plan,
            route_lookup=route_lookup,
            corpus_profile=corpus_profile,
            research_brief=research_brief,
            supervisor_decision=supervisor_decision,
        )

        web_hits: list[EvidenceItem] = []
        document_hits: list[EvidenceItem] = []
        notes: list[ResearchNote] = []
        evidence: list[EvidenceItem] = []
        executed_routes = []

        for item in plan:
            result = research_results.get(item.id)
            route = result.executed_route if result is not None and result.executed_route is not None else route_lookup.get(item.id)
            if route is None or result is None:
                item.status = "done"
                continue
            executed_routes.append(route)

            web_evidence = result.web_evidence
            document_evidence = result.document_evidence
            web_hits.extend(web_evidence)
            document_hits.extend(document_evidence)

            if route.mode in {"external", "hybrid"}:
                self._append_trace(
                    state,
                    kind="tool_call",
                    actor=result.agent_name,
                    message=f"{result.agent_name} collected {len(web_evidence)} external evidence items.",
                    step=f"research.{result.agent_id}.{item.id}.web",
                    tool_name="web_search",
                    provider=self.copilot.settings.search_provider,
                    latency_ms=result.web_latency_ms,
                    result_count=len(web_evidence),
                    query=route.web_query or item.search_query or item.question,
                    queries=route.web_queries,
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    execution_mode="specialist_worker",
                    parallel=True,
                    runtime="langgraph",
                )

            if route.mode in {"internal", "hybrid"} and request.include_private_docs and corpus_profile.has_private_docs:
                self._append_trace(
                    state,
                    kind="tool_call",
                    actor=result.agent_name,
                    message=f"{result.agent_name} retrieved {len(document_evidence)} contextual evidence items.",
                    step=f"research.{result.agent_id}.{item.id}.vector",
                    tool_name="vector_retrieval",
                    provider=corpus_profile.vector_backend,
                    latency_ms=result.document_latency_ms,
                    result_count=len(document_evidence),
                    query=route.internal_query or item.search_query or item.question,
                    queries=route.internal_queries,
                    backend=corpus_profile.vector_backend,
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    execution_mode="specialist_worker",
                    parallel=True,
                    runtime="langgraph",
                )

            item_evidence = self.copilot._dedupe_evidence([*web_evidence, *document_evidence])
            item.evidence_count = len(item_evidence)
            item.status = "done"
            evidence.extend(item_evidence)
            note = result.note or self.copilot.workflow.compress_findings(item, item_evidence, route)
            notes.append(note)
            for iteration in note.research_iterations:
                if iteration.get("action") != "mcp_tool":
                    continue
                self._append_trace(
                    state,
                    kind="tool_call",
                    actor=result.agent_name,
                    message=f"Called MCP tool {iteration.get('mcp_tool_name') or 'mcp_tool'}.",
                    step=f"research.{result.agent_id}.{item.id}.mcp.{iteration.get('iteration', 0)}",
                    tool_name=iteration.get("mcp_tool_name") or "mcp_tool",
                    provider="model_context_protocol",
                    latency_ms=int(iteration.get("tool_latency_ms", 0) or 0),
                    result_count=int(iteration.get("result_count", 0) or 0),
                    query=iteration.get("query"),
                    mcp_tool_name=iteration.get("mcp_tool_name"),
                    mcp_tool_args=iteration.get("mcp_tool_args"),
                    source_channel="mcp",
                    plan_item_id=item.id,
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    execution_mode="specialist_worker",
                    runtime="langgraph",
                )
            self._append_trace(
                state,
                kind="step",
                actor=result.agent_name,
                message=f"{result.agent_name} completed {item.question}",
                step=f"research.{result.agent_id}.{item.id}",
                status="completed",
                evidence_count=item.evidence_count,
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                execution_mode="specialist_worker",
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
                runtime="langgraph",
            )
            self._checkpoint(
                state,
                f"research.{result.agent_id}.{item.id}",
                {
                    "question": item.question,
                    "agent_id": result.agent_id,
                    "agent_name": result.agent_name,
                    "execution_mode": "specialist_worker",
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
                    "retrieval_strategy": "parent_child_dense_bm25_optional_graph_rerank",
                    "parallel": True,
                },
            )

        if executed_routes:
            retrieval_routes = executed_routes
            search_queries = self.copilot.workflow.build_queries(plan, retrieval_routes, revision_count=revision_count)
        else:
            search_queries = state["final_search_queries"]
        runtime_evidence = self.copilot._build_run_artifact_evidence(
            run_id=state["run_id"],
            plan=plan,
            search_queries=search_queries,
            retrieval_routes=retrieval_routes,
            web_hits=web_hits,
            document_hits=document_hits,
            notes=notes,
            revision_count=revision_count,
        )
        evidence = self.copilot._dedupe_evidence([*evidence, runtime_evidence])
        return {
            "final_web_hits": web_hits,
            "final_document_hits": document_hits,
            "final_notes": notes,
            "final_evidence": evidence,
            "final_search_queries": search_queries,
            "final_retrieval_routes": retrieval_routes,
        }

    def _reporter(self, state: ResearchGraphState) -> ResearchGraphState:
        request = state["request"]
        research_brief = state["final_research_brief"] or request.topic
        plan = state["final_plan"]
        retrieval_routes = state["final_retrieval_routes"]
        evidence = state["final_evidence"]
        web_hits = state["final_web_hits"]
        document_hits = state["final_document_hits"]
        notes = state["final_notes"]
        search_queries = state["final_search_queries"]
        revision_count = state["revision_count"]

        sections = self.copilot._build_sections(
            request,
            research_brief,
            plan,
            retrieval_routes,
            evidence,
            web_hits,
            document_hits,
            notes,
            search_queries,
        )
        confidence = self.copilot._estimate_confidence(request, evidence, document_hits, plan)
        self._record_handoff(
            state,
            "supervisor",
            "reporter",
            "reporting",
            "Assemble the citation-backed answer.",
            revision_count,
        )
        report_evidence = self.copilot._rank_evidence_for_report(evidence)
        report = self.copilot.reporter.build_report(request.topic, sections, report_evidence, confidence)
        reporter_usage = self.copilot.reporter.last_usage
        self._append_trace(
            state,
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
            runtime="langgraph",
        )
        report = report.model_copy(update={"source_index": self.copilot.workflow.format_sources(report_evidence)})
        return {"final_report": report}

    def _verifier_evaluator(self, state: ResearchGraphState) -> ResearchGraphState:
        request = state["request"]
        report = state["final_report"]
        evidence = state["final_evidence"]
        plan = state["final_plan"]
        revision_count = state["revision_count"]

        assessment = self.copilot.verifier.assess(
            report,
            evidence,
            plan,
            revision_count=revision_count,
            max_revisions=request.max_revisions,
        )
        evaluation = self.copilot.evaluator.evaluate(
            report=report,
            plan=plan,
            evidence=evidence,
            document_hits=state["final_document_hits"],
            retrieval_routes=state["final_retrieval_routes"],
        )
        verifier_usage = self.copilot.verifier.last_usage
        self._append_trace(
            state,
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
            runtime="langgraph",
        )
        self._append_trace(
            state,
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
            runtime="langgraph",
        )
        self._checkpoint(
            state,
            "report.verified",
            {
                "confidence": report.confidence,
                "issue_count": len(assessment.issues),
                "critical_issue_count": len(assessment.critical_issues),
                "source_count": report.source_count,
                "coverage_score": assessment.coverage_score,
                "should_revise": assessment.should_revise,
            },
        )
        self._checkpoint(
            state,
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
        should_revise = assessment.should_revise or quality_should_revise
        revision_notes = (
            assessment.critical_issues
            or assessment.issues
            or evaluation.notes
            or ["tighten citations and evidence coverage"]
        )
        revision_reason = assessment.revision_reason or "; ".join(evaluation.notes[:2]) or None

        usable_report = bool(report.citations) and evaluation.passed
        strict_mode = bool(getattr(self.copilot.settings, "strict_providers", False))
        if should_revise and revision_count < request.max_revisions:
            final_status = "completed"
            failure_reason = None
            needs_revision = True
        elif should_revise and revision_count >= request.max_revisions and not usable_report and strict_mode:
            final_status = "failed"
            failure_reason = revision_reason or "Maximum revision budget reached."
            needs_revision = False
        elif assessment.critical_issues and not report.citations and strict_mode:
            final_status = "failed"
            failure_reason = assessment.revision_reason or "Citation-backed report could not be produced."
            needs_revision = False
        else:
            final_status = "completed"
            failure_reason = None
            needs_revision = False

        return {
            "final_evaluation": evaluation,
            "final_issues": assessment.issues,
            "final_status": final_status,
            "failure_reason": failure_reason,
            "needs_revision": needs_revision,
            "revision_notes": revision_notes if needs_revision else state.get("revision_notes", []),
            "revision_reason": revision_reason,
        }

    def _revision_prepare(self, state: ResearchGraphState) -> ResearchGraphState:
        new_revision_count = state["revision_count"] + 1
        revision_reason = state.get("revision_reason") or "Verification requested another pass."
        self._record_handoff(
            state,
            "supervisor",
            "planner",
            "revision",
            revision_reason,
            new_revision_count,
        )
        self._checkpoint(
            state,
            "supervisor.revision_requested",
            {
                "revision_count": new_revision_count,
                "revision_notes": state.get("revision_notes", []),
                "revision_reason": revision_reason,
            },
        )
        return {"revision_count": new_revision_count, "needs_revision": False}

    def _finalize(self, state: ResearchGraphState) -> ResearchGraphState:
        end = datetime.now(timezone.utc)
        start = state["start"]
        duration_ms = int((end - start).total_seconds() * 1000)
        request = state["request"]
        run = ResearchRun(
            run_id=state["run_id"],
            job_id=state.get("job_id"),
            request=request,
            research_brief=state.get("final_research_brief"),
            corpus_profile=state.get("final_corpus_profile"),
            supervisor_decision=state.get("final_supervisor_decision"),
            plan=state.get("final_plan", []),
            search_queries=state.get("final_search_queries", []),
            retrieval_routes=state.get("final_retrieval_routes", []),
            notes=state.get("final_notes", []),
            evidence=state.get("final_evidence", []),
            web_hits=state.get("final_web_hits", []),
            document_hits=state.get("final_document_hits", []),
            checkpoints=state.get("checkpoints", []),
            trace=state.get("trace", []),
            handoffs=state.get("handoffs", []),
            report=state.get("final_report"),
            evaluation=state.get("final_evaluation"),
            issues=state.get("final_issues", []),
            status=state.get("final_status", "completed"),
            revision_count=state.get("revision_count", 0),
            failure_reason=state.get("failure_reason"),
            started_at=start.isoformat(),
            finished_at=end.isoformat(),
            duration_ms=duration_ms,
            metadata=state.get("metadata", {}),
        )
        self.copilot.ledger.record(run)
        self.copilot.storage.save_run(run)
        observability_result = self.copilot.observability.publish_run(run)
        if observability_result.trace_url or observability_result.error:
            run = run.model_copy(
                update={
                    "metadata": {
                        **run.metadata,
                        "observability": observability_result.as_dict(),
                    }
                }
            )
            self.copilot.ledger.record(run)
            self.copilot.storage.save_run(run)
        self.copilot.telemetry.emit(
            "run.finish",
            request.topic,
            run_id=run.run_id,
            job_id=run.job_id,
            duration_ms=duration_ms,
            issues=len(run.issues),
            revision_count=run.revision_count,
            status=run.status,
            failure_reason=run.failure_reason,
            orchestration_runtime="langgraph",
        )
        return {"run": run}

    def _route_after_verification(self, state: ResearchGraphState) -> Literal["revise", "finish"]:
        return "revise" if state.get("needs_revision") else "finish"

    def _checkpoint(self, state: ResearchGraphState, stage: str, payload: dict[str, object]) -> None:
        checkpoint = RunCheckpoint(run_id=state["run_id"], stage=stage, payload=payload)
        state.setdefault("checkpoints", []).append(checkpoint)
        self.copilot.telemetry.emit("checkpoint.created", stage, run_id=state["run_id"], job_id=state.get("job_id"), step=stage)
        state.setdefault("trace", []).append(
            RunTraceEvent(
                kind="checkpoint",
                actor="supervisor",
                message=stage,
                step=stage,
                status="completed",
                metadata=payload,
            )
        )

    def _append_trace(
        self,
        state: ResearchGraphState,
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
            kind=kind,  # type: ignore[arg-type]
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
        state.setdefault("trace", []).append(event)
        self.copilot.telemetry.emit(
            kind,
            message,
            run_id=state["run_id"],
            job_id=state.get("job_id"),
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

    def _record_handoff(
        self,
        state: ResearchGraphState,
        from_agent: str,
        to_agent: str,
        step: str,
        reason: str,
        revision: int = 0,
    ) -> AgentHandoff:
        handoff = AgentHandoff(
            from_agent=from_agent,
            to_agent=to_agent,
            step=step,
            reason=reason,
            revision=revision,
        )
        state.setdefault("handoffs", []).append(handoff)
        self._append_trace(
            state,
            kind="handoff",
            actor=from_agent,
            message=reason,
            step=step,
            from_agent=from_agent,
            to_agent=to_agent,
            handoff=handoff,
            revision=revision,
            runtime="langgraph",
        )
        return handoff


def _resolve_checkpoint_path(path: str) -> Path:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint_path
