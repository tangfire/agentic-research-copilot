from __future__ import annotations

from datetime import datetime, timezone
import re
import uuid
from typing import Any

from .constraint_evaluation import (
    derive_constraints_from_memories,
    evaluate_constraint_coverage,
    summarise_constraint_coverage,
)
from .github_repository import canonical_repository_slug, parse_github_repository_hint
from .schemas import (
    AgentEvent,
    AgentRunStep,
    AgentMessage,
    AgentPlanDraft,
    AgentSession,
    AgentSessionBundle,
    AgentToolDefinition,
    AgentTurnResponse,
    ClarificationContract,
    ApprovalRequest,
    ConstraintCoverage,
    MemoryExtractionResult,
    MemoryItem,
    ResearchJob,
    ResearchRequest,
    ResearchRun,
    ResearchSkill,
    SkillExecutionResult,
    SkillScript,
    WorkspaceProfile,
    ToolInvocation,
)
from .multi_agent_harness import role_preview_for_plan
from .skill_registry import SkillRegistry


DEFAULT_WORKSPACE_ID = "default-workspace"
HEARTBEAT_INTERVAL_SECONDS = 12
SESSION_COMPACTION_USER_TURNS = 6
SESSION_COMPACTION_MIN_CHARACTERS = 2200


class ConversationalResearchAgent:
    """Conversation facade around the existing research runtime."""

    def __init__(self, copilot: Any) -> None:
        self.copilot = copilot
        self.store = copilot.storage
        self.skill_registry = SkillRegistry(
            getattr(copilot.settings, "skill_paths", ["skills"]),
            fallback_catalog=_default_skill_catalog(),
            script_timeout_seconds=getattr(copilot.settings, "skill_script_timeout_seconds", 10.0),
        )
        self._default_workspace = self._ensure_default_workspace()

    def create_session(
        self,
        *,
        title: str | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentSession:
        workspace = self.get_workspace(workspace_id) if workspace_id else self.default_workspace()
        session_id = str(uuid.uuid4())
        session = AgentSession(
            session_id=session_id,
            session_key=session_id,
            title=title or "New research session",
            workspace_id=workspace.workspace_id if workspace else self._default_workspace.workspace_id,
            metadata=metadata or {},
        )
        self.store.save_agent_session(session)
        return session

    def default_workspace(self) -> WorkspaceProfile:
        workspace = self.store.load_workspace(self._default_workspace.workspace_id)
        if workspace is not None:
            return workspace
        self.store.save_workspace(self._default_workspace)
        return self._default_workspace

    def list_workspaces(self) -> list[WorkspaceProfile]:
        workspaces = self.store.load_workspaces()
        if not workspaces:
            return [self.default_workspace()]
        return sorted(workspaces, key=lambda item: item.updated_at, reverse=True)

    def get_workspace(self, workspace_id: str | None) -> WorkspaceProfile:
        if not workspace_id:
            return self.default_workspace()
        workspace = self.store.load_workspace(workspace_id)
        if workspace is None:
            raise KeyError(workspace_id)
        return workspace

    def save_workspace(self, workspace: WorkspaceProfile) -> WorkspaceProfile:
        saved = workspace.model_copy(
            update={
                "metadata": {**workspace.metadata, "user_configured": True},
                "updated_at": _utc_now(),
            }
        )
        self.store.save_workspace(saved)
        return saved

    def skill_catalog(self) -> list[ResearchSkill]:
        return self.skill_registry.list_skills()

    def get_skill(self, skill_id: str | None) -> ResearchSkill | None:
        return self.skill_registry.get_skill(skill_id)

    def describe_skill(self, skill_id: str) -> dict[str, Any]:
        return self.skill_registry.describe_skill(skill_id)

    def run_skill_script(
        self,
        skill_id: str,
        script_name: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> SkillExecutionResult:
        return self.skill_registry.run_script(
            skill_id,
            script_name,
            payload or {},
            timeout_seconds=timeout_seconds,
        )

    def _ensure_default_workspace(self) -> WorkspaceProfile:
        workspace = self.store.load_workspace(DEFAULT_WORKSPACE_ID)
        if workspace is not None:
            return workspace
        workspace = _default_workspace_profile()
        self.store.save_workspace(workspace)
        return workspace

    def list_sessions(self) -> list[AgentSession]:
        sessions = [self._refresh_session(session) for session in self.store.load_agent_sessions()]
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def get_session_bundle(self, session_id: str) -> AgentSessionBundle | None:
        session = self.store.load_agent_session(session_id)
        if session is None:
            return None
        session = self._refresh_session(session)
        active_job = self._active_job(session)
        active_run = self._active_run(session)
        memory = self.relevant_memory("", session_id=session_id, limit=40)
        return AgentSessionBundle(
            session=session,
            workspace=self.get_workspace(session.workspace_id),
            selected_skill=self.get_skill(session.selected_skill_id),
            skill_catalog=self.skill_catalog(),
            messages=self.store.load_agent_messages(session_id),
            plan_draft=self.store.load_agent_plan_draft(session_id),
            memory=memory,
            steps=self.list_steps(session_id),
            tool_registry=self.tool_definitions(workspace=self.get_workspace(session.workspace_id)),
            tool_invocations=self.list_tool_invocations(session_id),
            approval_requests=self.store.load_approval_requests(session_id),
            memory_extraction_results=self.store.load_memory_extraction_results(session_id),
            memory_evaluation=self.memory_evaluation(session_id),
            constraint_coverage=self.constraint_coverage(session_id),
            role_assignments=active_run.role_assignments if active_run else [],
            route_decisions=active_run.route_decisions if active_run else [],
            conflicts=active_run.conflicts if active_run else [],
            evidence_ledger=active_run.evidence_ledger if active_run else None,
            benchmark_summary=active_run.benchmark_summary if active_run else None,
            active_job=active_job,
            active_run=active_run,
            mcp_status=self.mcp_status(),
            observability=self.copilot.observability.status(),
        )

    def export_session_bundle(self, session_id: str) -> dict[str, Any]:
        bundle = self.get_session_bundle(session_id)
        if bundle is None:
            raise KeyError(session_id)
        return {
            "exported_at": _utc_now(),
            "session_key": bundle.session.session_key or bundle.session.session_id,
            "workspace": bundle.workspace.model_dump(mode="json") if bundle.workspace else None,
            "selected_skill": bundle.selected_skill.model_dump(mode="json") if bundle.selected_skill else None,
            "session": bundle.session.model_dump(mode="json"),
            "messages": [message.model_dump(mode="json") for message in bundle.messages],
            "plan_draft": bundle.plan_draft.model_dump(mode="json") if bundle.plan_draft else None,
            "memory": [memory.model_dump(mode="json") for memory in bundle.memory],
            "steps": [step.model_dump(mode="json") for step in bundle.steps],
            "events": [event.model_dump(mode="json") for event in self.list_events(session_id, limit=500)],
            "tool_registry": [tool.model_dump(mode="json") for tool in bundle.tool_registry],
            "tool_invocations": [item.model_dump(mode="json") for item in bundle.tool_invocations],
            "approval_requests": [item.model_dump(mode="json") for item in bundle.approval_requests],
            "memory_evaluation": bundle.memory_evaluation,
            "constraint_coverage": [item.model_dump(mode="json") for item in bundle.constraint_coverage],
            "role_assignments": [item.model_dump(mode="json") for item in bundle.role_assignments],
            "route_decisions": [item.model_dump(mode="json") for item in bundle.route_decisions],
            "conflicts": [item.model_dump(mode="json") for item in bundle.conflicts],
            "evidence_ledger": bundle.evidence_ledger.model_dump(mode="json") if bundle.evidence_ledger else None,
            "benchmark_summary": bundle.benchmark_summary.model_dump(mode="json") if bundle.benchmark_summary else None,
            "active_job": bundle.active_job.model_dump(mode="json") if bundle.active_job else None,
            "active_run": bundle.active_run.model_dump(mode="json") if bundle.active_run else None,
            "mcp_status": bundle.mcp_status,
            "observability": bundle.observability,
        }

    def receive_message(
        self,
        session_id: str,
        content: str,
        *,
        depth: str = "standard",
        include_private_docs: bool = True,
        max_sections: int = 4,
        max_revisions: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTurnResponse:
        session = self._require_session(session_id)
        user_message = self._save_message(
            session_id=session_id,
            role="user",
            content=content,
            intent="chat",
            metadata=metadata or {},
        )
        message_step = self._save_step(
            session_id=session_id,
            kind="message",
            status="completed",
            title="User message received",
            summary=_trim(content, 180),
            actor="user",
            input_preview=_trim(content, 300),
            metadata={"message_id": user_message.message_id},
        )
        memory_updates, extraction_result = self._extract_and_store_memories(session, user_message)
        session = self.store.load_agent_session(session_id) or session
        workspace = self.get_workspace(session.workspace_id)
        selected_skill, skill_reason, missing_inputs = self._select_skill(content, workspace=workspace)
        skill_preflight, skill_preflight_step = self._run_skill_preflight(
            session_id=session_id,
            content=content,
            workspace=workspace,
            selected_skill=selected_skill,
        )
        if skill_preflight is not None and skill_preflight.status == "completed":
            script_missing = _as_string_list(skill_preflight.output.get("missing_inputs"))
            if script_missing:
                missing_inputs = _merge_unique_strings(missing_inputs, script_missing)
        session = self._save_session_context(
            session,
            workspace=workspace,
            selected_skill=selected_skill,
            metadata_updates={
                "selected_skill_reason": skill_reason,
                "skill_preflight": skill_preflight.model_dump(mode="json") if skill_preflight is not None else None,
            },
        )
        session = self._maybe_compact_session_context(
            session,
            workspace=workspace,
            selected_skill=selected_skill,
        )
        raw_request = ResearchRequest(
            topic=content,
            depth=depth,  # type: ignore[arg-type]
            include_private_docs=include_private_docs,
            max_sections=max(1, max_sections),
            max_revisions=max(0, max_revisions),
            metadata={
                "source": "agent_session",
                "session_id": session_id,
                "session_key": session.session_key or session.session_id,
                "workspace_id": workspace.workspace_id,
                "workspace_name": workspace.name,
                "skill_id": selected_skill.skill_id,
                "skill_name": selected_skill.name,
            },
        )
        session = self._save_session_status(session, "planning")
        planning_step = self._save_step(
            session_id=session_id,
            kind="planning",
            status="running",
            title="正在生成研究计划",
            summary="正在判断信息是否足够，并准备生成需要用户确认的研究计划。",
            actor="planner",
            input_preview=_trim(content, 500),
            metadata={
                "stage": "clarify_or_plan",
                "skill_id": selected_skill.skill_id,
                "workspace_id": workspace.workspace_id,
                "message_id": user_message.message_id,
            },
            step_id=_stable_id("planning_turn", session_id, user_message.message_id),
        )
        try:
            if missing_inputs:
                clarification = ClarificationContract(
                    need_clarification=True,
                    question=self._skill_clarification_question(
                        selected_skill,
                        missing_inputs,
                        workspace=workspace,
                    ),
                    verification="",
                    missing_dimensions=missing_inputs,
                    confidence=0.95,
                )
            else:
                clarification = self.copilot.clarify(raw_request)
        except Exception as exc:  # pragma: no cover - provider/network dependent
            return self._planning_failure_response(
                session=session,
                workspace=workspace,
                selected_skill=selected_skill,
                user_message=user_message,
                memory_updates=memory_updates,
                extraction_result=extraction_result,
                message_step=message_step,
                skill_preflight_step=skill_preflight_step,
                planning_step=planning_step,
                exc=exc,
                stage="clarifier",
            )
        research_request = self._build_research_request(
            session_id=session_id,
            latest_content=content,
            depth=depth,
            include_private_docs=include_private_docs,
            max_sections=max_sections,
            max_revisions=max_revisions,
            workspace=workspace,
            selected_skill=selected_skill,
        )
        if clarification.need_clarification:
            assistant_message = self._save_message(
                session_id=session_id,
                role="assistant",
                content=clarification.question,
                intent="clarify",
                metadata={
                    "missing_dimensions": clarification.missing_dimensions,
                    "confidence": clarification.confidence,
                    "skill_preflight": skill_preflight.model_dump(mode="json") if skill_preflight is not None else None,
                },
            )
            session = self._save_session_status(session, "collecting")
            clarify_step = self._save_step(
                session_id=session_id,
                kind="planning",
                status="skipped",
                title="需要补充研究信息",
                summary=clarification.question,
                actor="clarifier",
                input_preview=_trim(content, 260),
                output_preview=_trim(clarification.question, 260),
                metadata={
                    "missing_dimensions": clarification.missing_dimensions,
                    "skill_id": selected_skill.skill_id,
                    "skill_name": selected_skill.name,
                    "workspace_id": workspace.workspace_id,
                    "skill_preflight": skill_preflight.model_dump(mode="json") if skill_preflight is not None else None,
                },
                step_id=planning_step.step_id,
                created_at=planning_step.created_at,
            )
            return AgentTurnResponse(
                session=session,
                workspace=workspace,
                selected_skill=selected_skill,
                user_message=user_message,
                assistant_message=assistant_message,
                memory_updates=memory_updates,
                steps=[step for step in [message_step, skill_preflight_step, clarify_step] if step is not None],
                memory_extraction_result=extraction_result,
                mcp_status=self.mcp_status(),
            )

        try:
            corpus_profile = self.copilot.documents.profile()
            planner_contract = self.copilot.planner.draft(research_request, corpus_profile=corpus_profile)
        except Exception as exc:  # pragma: no cover - provider/network dependent
            return self._planning_failure_response(
                session=session,
                workspace=workspace,
                selected_skill=selected_skill,
                user_message=user_message,
                memory_updates=memory_updates,
                extraction_result=extraction_result,
                message_step=message_step,
                skill_preflight_step=skill_preflight_step,
                planning_step=planning_step,
                exc=exc,
                stage="planner",
            )
        role_preview = role_preview_for_plan(
            research_request,
            planner_contract.plan,
            skill_id=selected_skill.skill_id,
            workspace_context=_workspace_context_summary(workspace),
        )
        plan_draft = AgentPlanDraft(
            session_id=session_id,
            workspace_id=workspace.workspace_id,
            skill_id=selected_skill.skill_id,
            skill_name=selected_skill.name,
            skill_reason=skill_reason,
            research_request=research_request,
            research_brief=planner_contract.research_brief,
            plan_items=planner_contract.plan,
            assumptions=planner_contract.assumptions,
            success_criteria=planner_contract.success_criteria,
            metadata={
                "planner_confidence": planner_contract.confidence,
                "memory_ids": [memory.memory_id for memory in self.relevant_memory(content, session_id=session_id)],
                "mcp_status": self.mcp_status(),
                "role_preview": role_preview,
                "skill_preflight": skill_preflight.model_dump(mode="json") if skill_preflight is not None else None,
                "skill_id": selected_skill.skill_id,
                "skill_name": selected_skill.name,
                "skill_reason": skill_reason,
                "workspace_id": workspace.workspace_id,
                "workspace_name": workspace.name,
                "context_summary": session.context_summary,
                "required_inputs": selected_skill.required_inputs,
                "evaluation_focus": selected_skill.evaluation_focus,
            },
        )
        self.store.save_agent_plan_draft(plan_draft)
        planning_step = self._save_step(
            session_id=session_id,
            kind="planning",
            status="completed",
            title="研究计划已生成",
            summary=plan_draft.research_brief,
            actor="planner",
            input_preview=_trim(research_request.topic, 500),
            output_preview=_trim(self._render_plan_message(plan_draft), 600),
            evidence_count=len(memory_updates),
            metadata={
                "plan_item_count": len(plan_draft.plan_items),
                "memory_ids": plan_draft.metadata.get("memory_ids", []),
                "required_confirmation": True,
                "skill_id": selected_skill.skill_id,
                "workspace_id": workspace.workspace_id,
                "role_preview": role_preview,
                "skill_preflight": skill_preflight.model_dump(mode="json") if skill_preflight is not None else None,
            },
            step_id=planning_step.step_id,
            created_at=planning_step.created_at,
        )
        assistant_message = self._save_message(
            session_id=session_id,
            role="assistant",
            content=self._render_plan_message(plan_draft),
            intent="plan",
            metadata={
                "required_confirmation": True,
                "plan_item_count": len(plan_draft.plan_items),
                "skill_id": selected_skill.skill_id,
                "role_preview": role_preview,
                "skill_preflight": skill_preflight.model_dump(mode="json") if skill_preflight is not None else None,
            },
        )
        session = self._save_session_status(session, "awaiting_confirmation")
        return AgentTurnResponse(
            session=session,
            workspace=workspace,
            selected_skill=selected_skill,
            user_message=user_message,
            assistant_message=assistant_message,
            plan_draft=plan_draft,
            memory_updates=memory_updates,
            steps=[step for step in [message_step, skill_preflight_step, planning_step] if step is not None],
            memory_extraction_result=extraction_result,
            mcp_status=self.mcp_status(),
        )

    def confirm_plan(self, session_id: str) -> AgentTurnResponse:
        session = self._require_session(session_id)
        plan_draft = self.store.load_agent_plan_draft(session_id)
        if plan_draft is None:
            raise ValueError("No plan draft is available for this session.")
        job = self.copilot.submit_job(plan_draft.research_request)
        research_step = self._save_step(
            session_id=session_id,
            kind="research",
            status="running",
            title="研究任务已启动",
            summary="已把确认后的计划提交到底层 research runtime。",
            actor="conversational_agent",
            job_id=job.job_id,
            run_id=job.run_id,
            input_preview=_trim(plan_draft.research_request.topic, 500),
            metadata={
                "plan_item_count": len(plan_draft.plan_items),
                "workspace_id": session.workspace_id,
                "skill_id": session.selected_skill_id,
            },
        )
        session_metadata = {
            **session.metadata,
            "active_job_id": job.job_id,
            "confirmed_plan_at": _utc_now(),
            "workspace_id": session.workspace_id,
            "workspace_name": self.get_workspace(session.workspace_id).name,
            "skill_id": session.selected_skill_id,
            "skill_name": (self.get_skill(session.selected_skill_id).name if self.get_skill(session.selected_skill_id) else ""),
        }
        session = self._save_session_status(
            session.model_copy(update={"metadata": session_metadata, "active_run_id": job.run_id}),
            "researching",
        )
        assistant_message = self._save_message(
            session_id=session_id,
            role="assistant",
            content="计划已确认，研究任务已经启动。完成后我会把报告、trace 和 evaluation 绑定回这个会话。",
            intent="confirm",
            metadata={
                "job_id": job.job_id,
                "run_id": job.run_id,
                "workspace_id": session.workspace_id,
                "skill_id": session.selected_skill_id,
            },
        )
        approvals = self._ensure_mcp_approval_if_unavailable(session, job=job)
        return AgentTurnResponse(
            session=session,
            workspace=self.get_workspace(session.workspace_id),
            selected_skill=self.get_skill(session.selected_skill_id),
            assistant_message=assistant_message,
            plan_draft=plan_draft,
            steps=[research_step],
            tool_invocations=self.list_tool_invocations(session_id),
            approval_requests=approvals,
            active_job=job,
            status_url=f"/v1/research/jobs/{job.job_id}/status",
            result_url=f"/v1/research/jobs/{job.job_id}/result",
            mcp_status=self.mcp_status(),
        )

    def cancel_session(self, session_id: str) -> AgentSession:
        session = self._require_session(session_id)
        job_id = session.metadata.get("active_job_id")
        if isinstance(job_id, str) and job_id:
            self.copilot.cancel_job(job_id)
        self._save_step(
            session_id=session_id,
            kind="failure",
            status="failed",
            title="Research cancellation requested",
            summary="The active research job was asked to cancel.",
            actor="conversational_agent",
            job_id=job_id if isinstance(job_id, str) else None,
            metadata={"cancel_requested": True},
        )
        self._save_message(
            session_id=session_id,
            role="assistant",
            content="Research cancellation was requested for this session.",
            intent="research",
            metadata={"cancel_requested": True, "job_id": job_id},
        )
        return self._save_session_status(session, "failed")

    def _planning_failure_response(
        self,
        *,
        session: AgentSession,
        workspace: WorkspaceProfile,
        selected_skill: ResearchSkill,
        user_message: AgentMessage,
        memory_updates: list[MemoryItem],
        extraction_result: MemoryExtractionResult,
        message_step: AgentRunStep,
        skill_preflight_step: AgentRunStep | None,
        planning_step: AgentRunStep,
        exc: Exception,
        stage: str,
    ) -> AgentTurnResponse:
        public_error = _public_error(exc)
        failure_text = (
            f"规划阶段失败：{public_error}\n\n"
            "这次没有启动研究任务，也没有伪装成成功。你可以稍后重试；"
            "如果连续失败，优先检查模型中转站、模型超时和结构化 JSON 输出。"
        )
        failed_step = self._save_step(
            session_id=session.session_id,
            kind="planning",
            status="failed",
            title="研究计划生成失败",
            summary=public_error,
            actor=stage,
            input_preview=_trim(user_message.content, 500),
            output_preview=_trim(failure_text, 600),
            metadata={
                "stage": stage,
                "error_type": exc.__class__.__name__,
                "skill_id": selected_skill.skill_id,
                "workspace_id": workspace.workspace_id,
                "message_id": user_message.message_id,
            },
            step_id=planning_step.step_id,
            created_at=planning_step.created_at,
        )
        assistant_message = self._save_message(
            session_id=session.session_id,
            role="assistant",
            content=failure_text,
            intent="plan",
            metadata={
                "planning_failure": True,
                "stage": stage,
                "error_type": exc.__class__.__name__,
                "skill_id": selected_skill.skill_id,
                "workspace_id": workspace.workspace_id,
            },
        )
        session = self._save_session_status(session, "failed")
        return AgentTurnResponse(
            session=session,
            workspace=workspace,
            selected_skill=selected_skill,
            user_message=user_message,
            assistant_message=assistant_message,
            memory_updates=memory_updates,
            steps=[step for step in [message_step, skill_preflight_step, failed_step] if step is not None],
            memory_extraction_result=extraction_result,
            mcp_status=self.mcp_status(),
        )

    def delete_session(self, session_id: str) -> dict[str, Any]:
        session = self.store.load_agent_session(session_id)
        if session is None:
            return {"deleted": False, "session_id": session_id, "counts": {}}
        job_id = session.metadata.get("active_job_id")
        job_cancel_requested = False
        if isinstance(job_id, str) and job_id:
            job = self.copilot.get_job(job_id)
            if job is not None and job.status not in {"completed", "failed", "cancelled"}:
                self.copilot.cancel_job(job_id)
                job_cancel_requested = True
        linked_memories = [
            memory
            for memory in self.store.load_memory_items(session_id=session_id)
            if memory.session_id == session_id
        ]
        promoted_memory_ids: list[str] = []
        session_memory_ids: list[str] = []
        for memory in linked_memories:
            if memory.scope == "session":
                session_memory_ids.append(memory.memory_id)
                continue
            self.store.save_memory_item(
                memory.model_copy(
                    update={
                        "session_id": None,
                        "updated_at": _utc_now(),
                        "metadata": {
                            **memory.metadata,
                            "promoted_from_deleted_session_id": session_id,
                        },
                    }
                )
            )
            promoted_memory_ids.append(memory.memory_id)
        counts = self.store.delete_agent_session(session_id)
        return {
            "deleted": counts.get("sessions", 0) > 0,
            "session_id": session_id,
            "job_cancel_requested": job_cancel_requested,
            "session_memory_ids": session_memory_ids,
            "promoted_memory_ids": promoted_memory_ids,
            "counts": counts,
        }

    def list_steps(self, session_id: str) -> list[AgentRunStep]:
        self._require_session(session_id)
        return self.store.load_agent_steps(session_id)

    def list_events(self, session_id: str, *, limit: int = 80, after_event_id: str | None = None) -> list[AgentEvent]:
        self._require_session(session_id)
        events: list[AgentEvent] = []
        for message in self.store.load_agent_messages(session_id):
            events.append(self._event_from_message(message))
        for step in self.store.load_agent_steps(session_id):
            events.append(self._event_from_step(step))
        for approval in self.store.load_approval_requests(session_id):
            events.append(self._event_from_approval(approval))
        for invocation in self.store.load_tool_invocations(session_id):
            events.append(self._event_from_tool_invocation(invocation))
        events.sort(key=lambda item: (item.created_at, item.event_id))
        if after_event_id:
            for index, event in enumerate(events):
                if event.event_id == after_event_id:
                    return events[index + 1 : index + 1 + max(1, limit)]
        return events[-max(1, limit) :]

    def tool_definitions(self, *, workspace: WorkspaceProfile | None = None) -> list[AgentToolDefinition]:
        settings = self.copilot.settings
        mcp = self.mcp_status()
        search_provider = getattr(settings, "search_provider", "none")
        web_requires_key = search_provider not in {"none", "duckduckgo"}
        web_enabled = self.copilot.researcher.search_tool is not None
        disabled_tools = set(workspace.disabled_tools if workspace is not None else [])
        return [
            AgentToolDefinition(
                name="web_search",
                channel="web",
                description="通过已配置的搜索服务获取公开 Web 证据。",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                enabled=web_enabled and "web_search" not in disabled_tools,
                requires_auth=web_requires_key,
                auth_configured=not web_requires_key or bool(getattr(settings, "search_api_key", "")),
                risk_level="low",
                approval_required=False,
                failure_mode="" if web_enabled else "搜索服务未启用。",
                metadata={
                    "provider": getattr(settings, "search_provider", "none"),
                    "workspace_disabled": "web_search" in disabled_tools,
                },
            ),
            AgentToolDefinition(
                name="vector_retrieval",
                channel="vector",
                description="检索本地知识库、项目文档和 project memory。",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                enabled="vector_retrieval" not in disabled_tools,
                requires_auth=False,
                auth_configured=True,
                risk_level="low",
                approval_required=False,
                metadata={
                    "backend": self.copilot.documents.profile().vector_backend,
                    "workspace_disabled": "vector_retrieval" in disabled_tools,
                },
            ),
            AgentToolDefinition(
                name="mcp_tool",
                channel="mcp",
                description="外部 MCP 工具，通常用于 GitHub 仓库、代码、Issue、PR 和 Release 证据。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "mcp_tool_name": {"type": "string"},
                        "mcp_tool_args": {"type": "object"},
                    },
                },
                enabled=bool(mcp["configured"] and mcp["available"]) and "mcp_tool" not in disabled_tools,
                requires_auth=bool(mcp["auth_required"]),
                auth_configured=not bool(mcp["auth_required"]) or bool(mcp["auth_token_configured"]),
                risk_level="medium",
                approval_required=bool(mcp["configured"] and not mcp["available"]),
                failure_mode=str(mcp.get("reason") or ""),
                metadata={
                    "tools": mcp.get("tools", []),
                    "catalog_error": mcp.get("catalog_error", ""),
                    "workspace_disabled": "mcp_tool" in disabled_tools,
                },
            ),
        ]

    def list_tool_invocations(self, session_id: str) -> list[ToolInvocation]:
        self._require_session(session_id)
        return self.store.load_tool_invocations(session_id)

    def resolve_approval(self, session_id: str, approval_id: str, *, approve: bool) -> ApprovalRequest:
        self._require_session(session_id)
        approval = self.store.load_approval_request(approval_id)
        if approval is None or approval.session_id != session_id:
            raise KeyError(approval_id)
        now = _utc_now()
        status = "approved" if approve else "rejected"
        updated = approval.model_copy(update={"status": status, "resolved_at": now})
        self.store.save_approval_request(updated)
        invocation = next(
            (
                item
                for item in self.store.load_tool_invocations(session_id)
                if item.invocation_id == approval.invocation_id
            ),
            None,
        )
        if invocation is not None:
            invocation_status = "skipped" if not approve or self.mcp_status().get("available") is False else "completed"
            self.store.save_tool_invocation(
                invocation.model_copy(
                    update={
                        "status": invocation_status,
                        "updated_at": now,
                        "error": "" if approve else "用户拒绝了这次审批。",
                    }
                )
            )
        self._save_step(
            session_id=session_id,
            kind="approval",
            status="completed" if approve else "skipped",
            title=f"Approval {status}",
            summary=approval.reason,
            actor="user",
            metadata={"approval_id": approval_id, "invocation_id": approval.invocation_id},
        )
        return updated

    def memory_evaluation(self, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        results = self.store.load_memory_extraction_results(session_id)
        messages = [message for message in self.store.load_agent_messages(session_id) if message.role == "user"]
        candidate_count = sum(len(result.candidates) for result in results)
        accepted_count = sum(len(result.accepted) for result in results)
        rejected_count = sum(len(result.rejected) for result in results)
        project_constraints = [
            memory
            for result in results
            for memory in result.accepted
            if memory.scope == "project" or memory.kind == "constraint"
        ]
        messages_with_memory = len({result.source_message_id for result in results if result.accepted})
        precision = accepted_count / candidate_count if candidate_count else 1.0
        recall_proxy = messages_with_memory / len(messages) if messages else 1.0
        return {
            "source_message_count": len(messages),
            "candidate_count": candidate_count,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "project_constraint_count": len(project_constraints),
            "memory_precision": round(precision, 4),
            "memory_recall": round(recall_proxy, 4),
            "notes": [
                "memory_recall is a proxy over user messages with accepted memories; use a labeled fixture for stricter evaluation."
            ],
        }

    def constraint_coverage(self, session_id: str) -> list[ConstraintCoverage]:
        session = self._require_session(session_id)
        run = self._active_run(session)
        if run is None or run.report is None:
            return []
        stored = self.store.load_constraint_coverage(run.run_id)
        if stored:
            return stored
        memories = self.store.load_memory_items(session_id=session_id)
        coverage = evaluate_constraint_coverage(
            run_id=run.run_id,
            session_id=session_id,
            constraints=derive_constraints_from_memories(memories),
            report=run.report,
            evidence=run.evidence,
        )
        self.store.save_constraint_coverage(coverage)
        return coverage

    def add_memory(
        self,
        *,
        content: str,
        scope: str = "project",
        kind: str = "fact",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        memory_session_id = session_id if scope == "session" else None
        memory = MemoryItem(
            memory_id=str(uuid.uuid4()),
            scope=scope,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            content=content.strip(),
            session_id=memory_session_id,
            confidence=0.9,
            metadata={
                **(metadata or {}),
                "hard_constraint": scope == "project" or kind == "constraint",
                "source": "manual",
            },
        )
        self._save_memory(memory)
        return memory

    def list_memory(self, *, scope: str | None = None, session_id: str | None = None) -> list[MemoryItem]:
        return self.store.load_memory_items(scope=scope, session_id=session_id)

    def delete_memory(self, memory_id: str) -> bool:
        memory = self.store.load_memory_item(memory_id)
        deleted = self.store.delete_memory_item(memory_id)
        if deleted and memory is not None and memory.scope == "project":
            self.copilot.delete_document(f"memory:{memory.memory_id}")
        return deleted

    def relevant_memory(self, query: str, *, session_id: str | None = None, limit: int = 12) -> list[MemoryItem]:
        candidates = self.store.load_memory_items(session_id=session_id)
        if not query.strip():
            return candidates[:limit]
        query_terms = _tokens(query)
        scored: list[tuple[int, str, MemoryItem]] = []
        for memory in candidates:
            memory_terms = _tokens(memory.content)
            overlap = len(query_terms & memory_terms)
            scope_bonus = 2 if memory.scope == "project" else 1 if memory.scope == "user" else 0
            session_bonus = 2 if session_id and memory.session_id == session_id else 0
            score = overlap * 3 + scope_bonus + session_bonus
            if score > 0:
                scored.append((score, memory.updated_at, memory))
        scored.sort(key=lambda item: (-item[0], item[1]), reverse=False)
        return [item[2] for item in scored[:limit]]

    def mcp_status(self) -> dict[str, Any]:
        settings = self.copilot.settings
        auth_required = bool(getattr(settings, "mcp_auth_required", False))
        auth_token_configured = bool(getattr(settings, "mcp_auth_token", ""))
        server_url = str(getattr(settings, "mcp_server_url", "") or "")
        is_github = "githubcopilot.com" in server_url.lower() or "api.github.com" in server_url.lower()
        provider = "github" if is_github else "custom"
        display_name = "GitHub MCP" if is_github else "MCP"
        configured = bool(
            getattr(settings, "mcp_enabled", False)
            and server_url
            and getattr(settings, "mcp_tools", [])
        )
        catalog_error = str(getattr(self.copilot, "mcp_tool_catalog_error", "") or "")
        available = self.copilot.mcp_tool is not None and not catalog_error
        reason = ""
        if not configured:
            reason = f"{display_name} 未配置。"
        elif auth_required and not auth_token_configured:
            reason = f"{display_name} 缺少 auth token。"
        elif catalog_error:
            reason = catalog_error
        elif not available:
            reason = f"{display_name} 客户端不可用。"
        if available:
            label = f"{display_name} 可用"
        elif not configured or (auth_required and not auth_token_configured):
            label = f"{display_name} 未配置"
        else:
            label = f"{display_name} 不可用"
        return {
            "configured": configured,
            "available": available,
            "provider": provider,
            "display_name": display_name,
            "is_github": is_github,
            "label": label,
            "auth_required": auth_required,
            "auth_token_configured": auth_token_configured,
            "server_url_configured": bool(server_url),
            "tools": list(getattr(settings, "mcp_tools", []) or []),
            "reason": reason,
            "catalog_error": catalog_error,
        }

    def _save_step(
        self,
        *,
        session_id: str,
        kind: str,
        status: str,
        title: str,
        summary: str = "",
        actor: str = "agent",
        tool_name: str | None = None,
        job_id: str | None = None,
        run_id: str | None = None,
        input_preview: str = "",
        output_preview: str = "",
        evidence_count: int = 0,
        metadata: dict[str, Any] | None = None,
        step_id: str | None = None,
        created_at: str | None = None,
    ) -> AgentRunStep:
        now = _utc_now()
        step = AgentRunStep(
            step_id=step_id or str(uuid.uuid4()),
            session_id=session_id,
            run_id=run_id,
            job_id=job_id,
            kind=kind,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            title=title,
            summary=_trim(summary, 500),
            actor=actor,
            tool_name=tool_name,
            input_preview=_trim(input_preview, 600),
            output_preview=_trim(output_preview, 600),
            evidence_count=evidence_count,
            created_at=created_at or now,
            updated_at=now,
            metadata=metadata or {},
        )
        self.store.save_agent_step(step)
        if step.run_id is None:
            try:
                self.copilot.observability.publish_step(step)
            except Exception:  # pragma: no cover - observability must never break core flow
                pass
        return step

    def _ensure_mcp_approval_if_unavailable(self, session: AgentSession, *, job: ResearchJob) -> list[ApprovalRequest]:
        mcp = self.mcp_status()
        if not mcp.get("configured") or mcp.get("available"):
            return []
        existing = [
            approval
            for approval in self.store.load_approval_requests(session.session_id)
            if approval.status == "pending" and approval.metadata.get("kind") == "mcp_unavailable"
        ]
        if existing:
            return existing
        invocation = ToolInvocation(
            invocation_id=_stable_id("mcp_unavailable", session.session_id, job.job_id),
            session_id=session.session_id,
            run_id=job.run_id,
            tool_name="mcp_tool",
            status="pending_approval",
            arguments={"configured_tools": mcp.get("tools", [])},
            result_preview="GitHub MCP 已配置但当前不可用。",
            error=str(mcp.get("reason") or ""),
            metadata={"kind": "mcp_unavailable", "job_id": job.job_id},
        )
        self.store.save_tool_invocation(invocation)
        approval = ApprovalRequest(
            approval_id=_stable_id("approval_mcp_unavailable", session.session_id, job.job_id),
            session_id=session.session_id,
            invocation_id=invocation.invocation_id,
            reason=str(mcp.get("reason") or "GitHub MCP 当前不可用。"),
            requested_action="在信任 MCP 证据前检查 GitHub MCP 配置。",
            metadata={"kind": "mcp_unavailable", "job_id": job.job_id},
        )
        self.store.save_approval_request(approval)
        self._save_step(
            session_id=session.session_id,
            kind="approval",
            status="pending",
            title="需要检查 GitHub MCP",
            summary=approval.reason,
            actor="tool_policy",
            tool_name="mcp_tool",
            job_id=job.job_id,
            run_id=job.run_id,
            metadata={"approval_id": approval.approval_id, "invocation_id": invocation.invocation_id},
            step_id=_stable_id("step_mcp_unavailable", session.session_id, job.job_id),
        )
        return [approval]

    def _sync_run_artifacts(self, session: AgentSession, run: ResearchRun, job: ResearchJob | None) -> None:
        job_id = job.job_id if job else run.job_id
        self._save_step(
            session_id=session.session_id,
            kind="research",
            status="completed" if run.status == "completed" else "failed",
            title="研究运行已结束",
            summary=f"Run {run.run_id} 已结束，状态为 {run.status}。",
            actor="research_runtime",
            job_id=job_id,
            run_id=run.run_id,
            evidence_count=len(run.evidence),
            metadata={"source_count": run.report.source_count if run.report else 0},
            step_id=_stable_id("step_run_completed", session.session_id, run.run_id),
        )
        for index, event in enumerate(run.trace):
            step_kind = _step_kind_for_trace(event.kind, event.actor)
            self._save_step(
                session_id=session.session_id,
                kind=step_kind,
                status=_step_status_for_trace(event.status),
                title=_trace_title(event),
                summary=event.message,
                actor=event.actor,
                tool_name=event.tool_name,
                job_id=job_id,
                run_id=run.run_id,
                input_preview=str(event.metadata.get("query") or event.metadata.get("input") or ""),
                output_preview=event.message,
                evidence_count=int(event.metadata.get("evidence_count", 0) or 0),
                metadata={"trace_index": index, **event.metadata},
                step_id=_stable_id("trace_step", session.session_id, run.run_id, str(index), event.created_at),
            )
            if event.kind == "tool_call":
                self.store.save_tool_invocation(
                    ToolInvocation(
                        invocation_id=_stable_id("tool_invocation", session.session_id, run.run_id, str(index), event.created_at),
                        session_id=session.session_id,
                        run_id=run.run_id,
                        tool_name=event.tool_name or str(event.metadata.get("tool_name") or "unknown_tool"),
                        status=_tool_status_for_trace(event.status),
                        arguments=_tool_arguments_from_trace(event.metadata),
                        result_preview=_trim(event.message, 600),
                        evidence_ids=[str(item) for item in event.metadata.get("evidence_ids", [])]
                        if isinstance(event.metadata.get("evidence_ids"), list)
                        else [],
                        latency_ms=event.latency_ms,
                        error=str(event.metadata.get("error") or "") if event.status == "failed" else "",
                        metadata={"trace_index": index, "actor": event.actor, **event.metadata},
                    )
                )
        if run.report is not None:
            memories = self.store.load_memory_items(session_id=session.session_id)
            coverage = evaluate_constraint_coverage(
                run_id=run.run_id,
                session_id=session.session_id,
                constraints=derive_constraints_from_memories(memories),
                report=run.report,
                evidence=run.evidence,
            )
            self.store.save_constraint_coverage(coverage)
            decision_text = _trim(
                "决策记忆："
                f"{run.report.title}。{run.report.summary} "
                f"Recommendations: {'; '.join(run.report.recommendations[:3])}",
                600,
            )
            decision_memory = MemoryItem(
                memory_id=_stable_id("decision_memory", session.session_id, run.run_id),
                scope="user",
                kind="decision",
                content=decision_text,
                session_id=None,
                confidence=0.82,
                metadata={
                    "source": "completed_run",
                    "source_session_id": session.session_id,
                    "source_run_id": run.run_id,
                    "workspace_id": session.workspace_id,
                    "skill_id": session.selected_skill_id,
                    "decision_memory": True,
                },
            )
            self._save_memory(decision_memory)

    def _require_session(self, session_id: str) -> AgentSession:
        session = self.store.load_agent_session(session_id)
        if session is None:
            raise KeyError(session_id)
        return self._refresh_session(session)

    def _save_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        intent: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,  # type: ignore[arg-type]
            content=content.strip(),
            intent=intent,  # type: ignore[arg-type]
            metadata=metadata or {},
        )
        self.store.save_agent_message(message)
        return message

    def _save_session_status(self, session: AgentSession, status: str) -> AgentSession:
        updated = session.model_copy(update={"status": status, "updated_at": _utc_now()})
        self.store.save_agent_session(updated)
        return updated

    def _extract_and_store_memories(
        self,
        session: AgentSession,
        message: AgentMessage,
    ) -> tuple[list[MemoryItem], MemoryExtractionResult]:
        extracted = _extract_memory_candidates(message.content, session.session_id, message.message_id)
        existing = {
            _normalize_memory_text(memory.content)
            for memory in self.store.load_memory_items(session_id=session.session_id)
        }
        saved: list[MemoryItem] = []
        rejected: list[MemoryItem] = []
        scope_ids = set(session.memory_scope_ids)
        for memory in extracted:
            key = _normalize_memory_text(memory.content)
            if key in existing:
                rejected.append(memory)
                continue
            self._save_memory(memory)
            saved.append(memory)
            scope_ids.add(memory.memory_id)
        if saved:
            self.store.save_agent_session(
                session.model_copy(
                    update={
                        "memory_scope_ids": sorted(scope_ids),
                        "updated_at": _utc_now(),
                    }
                )
            )
        result = MemoryExtractionResult(
            source_message_id=message.message_id,
            session_id=session.session_id,
            candidates=extracted,
            accepted=saved,
            rejected=rejected,
            reason=(
                "已接受明确偏好、团队/项目约束和具体会话事实。"
                if saved
                else "没有新的长期记忆候选被接受。"
            ),
            metadata={"extractor": "heuristic-v2", "duplicate_rejections": len(rejected)},
        )
        self.store.save_memory_extraction_result(result)
        return saved, result

    def _save_memory(self, memory: MemoryItem) -> None:
        if memory.scope == "project" or memory.kind == "constraint":
            memory = memory.model_copy(
                update={
                    "metadata": {
                        **memory.metadata,
                        "hard_constraint": True,
                    }
                }
            )
        self.store.save_memory_item(memory)
        if memory.scope == "project":
            self.copilot.add_document(
                title=f"Memory: {memory.kind}",
                source="agent-memory",
                snippet=memory.content,
                content=memory.content,
                metadata={
                    "kind": "agent_memory",
                    "memory_scope": memory.scope,
                    "memory_kind": memory.kind,
                    "memory_id": memory.memory_id,
                    "document_id": f"memory:{memory.memory_id}",
                },
            )

    def _save_session_context(
        self,
        session: AgentSession,
        *,
        workspace: WorkspaceProfile,
        selected_skill: ResearchSkill,
        metadata_updates: dict[str, Any] | None = None,
    ) -> AgentSession:
        now = _utc_now()
        extra_metadata = metadata_updates or {}
        updated = session.model_copy(
            update={
                "session_key": session.session_key or session.session_id,
                "workspace_id": workspace.workspace_id,
                "selected_skill_id": selected_skill.skill_id,
                "metadata": {
                    **session.metadata,
                    "workspace_id": workspace.workspace_id,
                    "workspace_name": workspace.name,
                    "selected_skill_id": selected_skill.skill_id,
                    "selected_skill_name": selected_skill.name,
                    **extra_metadata,
                },
                "updated_at": now,
            }
        )
        self.store.save_agent_session(updated)
        return updated

    def _maybe_compact_session_context(
        self,
        session: AgentSession,
        *,
        workspace: WorkspaceProfile,
        selected_skill: ResearchSkill,
    ) -> AgentSession:
        messages = self.store.load_agent_messages(session.session_id)
        user_turns = [message for message in messages if message.role == "user"]
        total_chars = sum(len(message.content) for message in messages)
        if len(user_turns) <= SESSION_COMPACTION_USER_TURNS and total_chars <= SESSION_COMPACTION_MIN_CHARACTERS:
            return session
        summary = self._summarize_session_context(messages, workspace=workspace, selected_skill=selected_skill)
        if summary == session.context_summary:
            return session
        now = _utc_now()
        updated = session.model_copy(
            update={
                "context_summary": summary,
                "metadata": {
                    **session.metadata,
                    "context_summary": summary,
                    "context_compaction": {
                        "user_turn_count": len(user_turns),
                        "message_count": len(messages),
                        "total_characters": total_chars,
                    },
                },
                "updated_at": now,
            }
        )
        self.store.save_agent_session(updated)
        self._save_step(
            session_id=session.session_id,
            kind="message",
            status="completed",
            title="Context compacted",
            summary=summary,
            actor="session_memory",
            metadata={
                "workspace_id": workspace.workspace_id,
                "skill_id": selected_skill.skill_id,
                "user_turn_count": len(user_turns),
                "message_count": len(messages),
                "total_characters": total_chars,
            },
        )
        return updated

    def _summarize_session_context(
        self,
        messages: list[AgentMessage],
        *,
        workspace: WorkspaceProfile,
        selected_skill: ResearchSkill,
    ) -> str:
        user_turns = [message.content.strip() for message in messages if message.role == "user" and message.content.strip()]
        assistant_turns = [message.content.strip() for message in messages if message.role == "assistant" and message.content.strip()]
        fragments: list[str] = []
        if workspace.team_context:
            fragments.append(f"workspace={workspace.team_context}")
        if workspace.default_stack:
            fragments.append(f"stack={', '.join(workspace.default_stack)}")
        if selected_skill.name:
            fragments.append(f"skill={selected_skill.name}")
        if user_turns:
            fragments.append("recent_user=" + " | ".join(user_turns[-4:]))
        if assistant_turns:
            fragments.append("recent_assistant=" + " | ".join(assistant_turns[-2:]))
        return _trim(" ; ".join(fragments), 700)

    def _select_skill(self, content: str, *, workspace: WorkspaceProfile) -> tuple[ResearchSkill, str, list[str]]:
        lower = content.lower()
        catalog = self.skill_catalog()
        scored: list[tuple[int, int, ResearchSkill, list[str]]] = []
        for index, skill in enumerate(catalog):
            matches = [keyword for keyword in skill.trigger_keywords if _skill_keyword_matches(content, lower, keyword)]
            score = len(matches)
            if skill.skill_id == "open_source_adoption_review" and _has_repo_target(content):
                score += 3
            if skill.skill_id == "architecture_tradeoff_memo" and _has_comparison_signal(content):
                score += 3
            if skill.skill_id == "demo_readiness_risk_review" and _has_demo_signal(content):
                score += 3
            if workspace.team_context and any(keyword in workspace.team_context.lower() for keyword in matches if keyword.isascii()):
                score += 1
            scored.append((score, -index, skill, matches))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_score, _, selected_skill, matches = scored[0]
        if selected_score <= 0:
            selected_skill = next((skill for skill in catalog if skill.skill_id == "open_source_adoption_review"), catalog[0])
            matches = []
        reason = (
            f"Matched keywords: {', '.join(matches)}"
            if matches
            else f"Defaulted to {selected_skill.name} because the request looks like an open-source adoption review."
        )
        missing_inputs = self._missing_skill_inputs(selected_skill, content=content, workspace=workspace)
        return selected_skill, reason, missing_inputs

    def _run_skill_preflight(
        self,
        *,
        session_id: str,
        content: str,
        workspace: WorkspaceProfile,
        selected_skill: ResearchSkill,
    ) -> tuple[SkillExecutionResult | None, AgentRunStep | None]:
        script = _select_auto_skill_script(selected_skill)
        if script is None:
            return None, None
        current_session = self.store.load_agent_session(session_id)
        payload = {
            "session_id": session_id,
            "workspace": workspace.model_dump(mode="json"),
            "content": content,
            "selected_skill": selected_skill.model_dump(mode="json"),
            "session_summary": current_session.context_summary if current_session is not None else "",
        }
        try:
            result = self.run_skill_script(selected_skill.skill_id, script.name, payload=payload)
        except Exception as exc:  # pragma: no cover - defensive for script runtime failure
            result = SkillExecutionResult(
                skill_id=selected_skill.skill_id,
                script_name=script.name,
                status="failed",
                exit_code=1,
                stderr=str(exc),
                output={"error": str(exc)},
                metadata={
                    "skill_root": selected_skill.metadata.get("skill_root", ""),
                    "script_path": script.path,
                },
            )
        step_status = "completed" if result.status == "completed" else "failed"
        step = self._save_step(
            session_id=session_id,
            kind="tool_call",
            status=step_status,
            title=f"Skill preflight: {selected_skill.name}",
            summary=_trim(result.stdout or result.output.get("summary") or result.stderr or script.description or script.name, 240),
            actor="skill_runner",
            tool_name=script.name,
            input_preview=_trim(content, 260),
            output_preview=_trim(result.stdout or result.output.get("summary") or result.stderr or "", 260),
            metadata={
                "skill_id": selected_skill.skill_id,
                "script_name": script.name,
                "status": result.status,
                "exit_code": result.exit_code,
                "output_keys": sorted(result.output.keys()) if isinstance(result.output, dict) else [],
                "script_path": result.metadata.get("script_path", script.path),
            },
        )
        return result, step

    def _missing_skill_inputs(
        self,
        skill: ResearchSkill,
        *,
        content: str,
        workspace: WorkspaceProfile,
    ) -> list[str]:
        missing: list[str] = []
        if skill.skill_id == "open_source_adoption_review":
            if not _has_repo_target(content):
                missing.append("目标 repo / 项目")
            if not (_has_team_constraints(content) or _workspace_has_team_constraints(workspace)):
                missing.append("团队约束")
            if not _has_decision_context(content):
                missing.append("决策问题")
        elif skill.skill_id == "architecture_tradeoff_memo":
            if not _has_comparison_signal(content):
                missing.append("要比较的方案")
            if not _has_decision_context(content):
                missing.append("决策标准")
            if not (_has_team_constraints(content) or _workspace_has_team_constraints(workspace)):
                missing.append("团队约束")
        elif skill.skill_id == "demo_readiness_risk_review":
            if not _has_demo_signal(content):
                missing.append("演示 / 面试场景")
            if not _has_risk_signal(content):
                missing.append("风险点")
            if not _has_decision_context(content):
                missing.append("需要验证的结论")
        return missing

    def _skill_clarification_question(
        self,
        skill: ResearchSkill,
        missing_inputs: list[str],
        *,
        workspace: WorkspaceProfile,
    ) -> str:
        context_hint = workspace.name or "当前 workspace"
        missing_text = "、".join(missing_inputs)
        return f"我会按「{skill.name}」来走。请补充 {missing_text}，尤其说明 {context_hint} 的真实约束，这样我才能先出计划再开始研究。"

    def _build_research_request(
        self,
        *,
        session_id: str,
        latest_content: str,
        depth: str,
        include_private_docs: bool,
        max_sections: int,
        max_revisions: int,
        workspace: WorkspaceProfile,
        selected_skill: ResearchSkill,
    ) -> ResearchRequest:
        session = self._require_session(session_id)
        messages = self.store.load_agent_messages(session_id)
        user_turns = [message.content for message in messages if message.role == "user"][-6:]
        memories = self.relevant_memory(latest_content, session_id=session_id, limit=10)
        hard_constraints = [
            memory
            for memory in memories
            if memory.scope == "project" or memory.kind == "constraint" or memory.metadata.get("hard_constraint") is True
        ]
        parts = ["Conversation research request:"]
        if workspace.name or workspace.team_context:
            parts.extend(
                [
                    "Workspace profile:",
                    f"- name: {workspace.name}",
                ]
            )
            if workspace.team_context:
                parts.append(f"- team_context: {workspace.team_context}")
            if workspace.default_stack:
                parts.append(f"- default_stack: {', '.join(workspace.default_stack)}")
            if workspace.deployment_constraints:
                parts.append(f"- deployment_constraints: {', '.join(workspace.deployment_constraints)}")
            if workspace.preferred_sources:
                parts.append(f"- preferred_sources: {', '.join(workspace.preferred_sources)}")
            if workspace.risk_policy:
                parts.append(f"- risk_policy: {workspace.risk_policy}")
        if session.context_summary:
            parts.extend(["", "Compacted session context:", f"- {session.context_summary}"])
        if selected_skill.skill_id:
            parts.extend(
                [
                    "",
                    "Selected skill:",
                    f"- id: {selected_skill.skill_id}",
                    f"- name: {selected_skill.name}",
                    f"- scenario: {selected_skill.scenario}",
                ]
            )
            if selected_skill.required_inputs:
                parts.append(f"- required_inputs: {', '.join(selected_skill.required_inputs)}")
            if selected_skill.evaluation_focus:
                parts.append(f"- evaluation_focus: {', '.join(selected_skill.evaluation_focus)}")
            if selected_skill.plan_template:
                parts.extend(["- plan_template:"] + [f"  - {item}" for item in selected_skill.plan_template])
            if selected_skill.instructions_excerpt:
                parts.extend(["- instructions_excerpt:", f"  {selected_skill.instructions_excerpt}"])
            if selected_skill.scripts:
                parts.append(f"- scripts: {', '.join(script.name for script in selected_skill.scripts)}")
        skill_preflight = session.metadata.get("skill_preflight")
        skill_preflight_output = skill_preflight.get("output") if isinstance(skill_preflight, dict) else None
        repository_hint = parse_github_repository_hint(skill_preflight_output, latest_content, user_turns)
        repository_slug = canonical_repository_slug(repository_hint)
        if isinstance(skill_preflight, dict) and skill_preflight:
            parts.extend(
                [
                    "",
                    "Skill preflight hints:",
                    f"- status: {skill_preflight.get('status', '')}",
                    f"- output: {_trim(skill_preflight.get('output', {}), 600)}",
                ]
            )
        if repository_slug:
            parts.extend(["", "Target GitHub repository:", f"- {repository_slug}"])
        parts.extend(["", "User turns:"])
        parts.extend(f"- {turn}" for turn in user_turns)
        hard_constraint_texts = list(dict.fromkeys(memory.content for memory in hard_constraints))
        if hard_constraint_texts:
            parts.extend(["", "Hard project constraints that must be addressed in the final memo:"])
            parts.extend(f"- [project/constraint] {content}" for content in hard_constraint_texts)
        other_memories = [memory for memory in memories if memory not in hard_constraints]
        if other_memories:
            parts.extend(["", "Relevant saved memory:"])
            parts.extend(f"- [{memory.scope}/{memory.kind}] {memory.content}" for memory in other_memories)
        parts.extend(
            [
                "",
                "Deliverable: prepare a citation-backed technical research memo with plan, evidence, trace, evaluation, and explicit coverage of every hard project constraint.",
            ]
        )
        metadata = {
            "source": "agent_session",
            "session_id": session_id,
            "session_key": session.session_key or session.session_id,
            "workspace_id": workspace.workspace_id,
            "workspace_name": workspace.name,
            "workspace_context": workspace.team_context,
            "default_stack": workspace.default_stack,
            "deployment_constraints": workspace.deployment_constraints,
            "risk_policy": workspace.risk_policy,
            "preferred_sources": workspace.preferred_sources,
            "skill_id": selected_skill.skill_id,
            "skill_name": selected_skill.name,
            "skill_scenario": selected_skill.scenario,
            "required_inputs": selected_skill.required_inputs,
            "evaluation_focus": selected_skill.evaluation_focus,
            "memory_ids": [memory.memory_id for memory in memories],
            "hard_constraint_memory_ids": [memory.memory_id for memory in hard_constraints],
            "hard_constraints": hard_constraint_texts,
            "user_turns": user_turns,
        }
        if repository_hint is not None:
            metadata["github_repository"] = repository_hint
            metadata["github_repository_slug"] = repository_slug
        return ResearchRequest(
            topic="\n".join(parts),
            depth=depth,  # type: ignore[arg-type]
            include_private_docs=include_private_docs,
            max_sections=max(1, max_sections),
            max_revisions=max(0, max_revisions),
            metadata=metadata,
        )

    def _render_plan_message(self, plan_draft: AgentPlanDraft) -> str:
        lines = [
            "我已经拟好研究计划。请先确认，确认后才会启动正式研究。",
            "",
        ]
        if plan_draft.skill_name:
            lines.extend(
                [
                    f"选中的 skill：{plan_draft.skill_name}",
                    f"选择原因：{plan_draft.skill_reason or '匹配当前研究场景。'}",
                    "",
                ]
            )
        if plan_draft.workspace_id:
            lines.extend(
                [
                    f"工作区：{plan_draft.workspace_id}",
                    "",
                ]
            )
        lines.extend(
            [
            f"研究 brief：{plan_draft.research_brief}",
            "",
            "计划：",
            ]
        )
        for index, item in enumerate(plan_draft.plan_items, start=1):
            lines.append(f"{index}. {item.question}")
        if plan_draft.success_criteria:
            lines.extend(["", "成功标准："])
            lines.extend(f"- {item}" for item in plan_draft.success_criteria[:5])
        return "\n".join(lines)

    def _refresh_session(self, session: AgentSession) -> AgentSession:
        updates: dict[str, Any] = {}
        if not session.session_key:
            updates["session_key"] = session.session_id
        if not session.workspace_id:
            updates["workspace_id"] = self.default_workspace().workspace_id
        if updates:
            session = session.model_copy(update={**updates, "updated_at": _utc_now()})
            self.store.save_agent_session(session)
        if session.status == "planning":
            return self._expire_stale_planning_session(session)
        if session.status != "researching":
            return session
        job = self._active_job(session)
        if job is None:
            return session
        if job.status == "completed" and job.run_id:
            updated = self._save_session_status(session.model_copy(update={"active_run_id": job.run_id}), "completed")
            run = self.copilot.get_run(job.run_id)
            if run is not None:
                self._sync_run_artifacts(updated, run, job)
            return updated
        if job.status in {"failed", "cancelled"}:
            self._save_step(
                session_id=session.session_id,
                kind="failure",
                status="failed" if job.status == "failed" else "skipped",
                title=f"Research job {job.status}",
                summary=job.error or f"Job ended with status {job.status}.",
                actor="research_runtime",
                job_id=job.job_id,
                run_id=job.run_id,
                metadata={"cancel_requested": job.cancel_requested},
                step_id=_stable_id("step_job_failed", session.session_id, job.job_id, job.status),
            )
            return self._save_session_status(session, "failed")
        if self._heartbeat_due(session, job):
            now = _utc_now()
            heartbeat_session = session.model_copy(update={"last_heartbeat_at": now, "updated_at": now})
            self.store.save_agent_session(heartbeat_session)
            self._save_step(
                session_id=session.session_id,
                kind="heartbeat",
                status="running",
                title="研究仍在运行",
                summary="已确认的研究任务仍在后台执行。",
                actor="research_runtime",
                job_id=job.job_id,
                run_id=job.run_id,
                metadata={"heartbeat": True},
                step_id=_stable_id("heartbeat_step", session.session_id, job.job_id, now),
            )
            return heartbeat_session
        return session

    def _expire_stale_planning_session(self, session: AgentSession) -> AgentSession:
        updated_at = _parse_utc_timestamp(session.updated_at)
        if updated_at is None:
            return session
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        timeout_seconds = max(90.0, float(getattr(self.copilot.settings, "model_timeout_seconds", 30.0)) * 3 + 15)
        if (datetime.now(timezone.utc) - updated_at).total_seconds() < timeout_seconds:
            return session

        existing_messages = self.store.load_agent_messages(session.session_id)
        if not any(message.metadata.get("planning_failure") == "stale_timeout" for message in existing_messages):
            self._save_message(
                session_id=session.session_id,
                role="assistant",
                content=(
                    "规划阶段超时：模型规划调用没有在预期时间内完成。\n\n"
                    "这次没有启动研究任务，也没有把失败伪装成成功。请检查模型中转站、模型超时配置，"
                    "然后重新发送这条问题。"
                ),
                intent="plan",
                metadata={"planning_failure": "stale_timeout"},
            )
        self._save_step(
            session_id=session.session_id,
            kind="planning",
            status="failed",
            title="规划阶段超时",
            summary="模型规划调用没有在预期时间内完成。",
            actor="planner",
            metadata={"planning_failure": "stale_timeout", "timeout_seconds": timeout_seconds},
            step_id=_stable_id("planning_stale_timeout", session.session_id),
        )
        return self._save_session_status(session, "failed")

    def _heartbeat_due(self, session: AgentSession, job: ResearchJob) -> bool:
        if job.status not in {"queued", "running"}:
            return False
        if not session.last_heartbeat_at:
            return True
        last = _parse_utc_timestamp(session.last_heartbeat_at)
        if last is None:
            return True
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() >= HEARTBEAT_INTERVAL_SECONDS

    def _event_from_message(self, message: AgentMessage) -> AgentEvent:
        return AgentEvent(
            event_id=message.message_id,
            session_id=message.session_id,
            type="message",
            kind="message",
            status=message.intent,
            title=f"{message.role} message",
            summary=_trim(message.content, 220),
            actor=message.role,
            created_at=message.created_at,
            payload=message.model_dump(mode="json"),
        )

    def _event_from_step(self, step: AgentRunStep) -> AgentEvent:
        kind: str = "heartbeat" if step.kind == "heartbeat" else step.kind
        return AgentEvent(
            event_id=step.step_id,
            session_id=step.session_id,
            run_id=step.run_id,
            job_id=step.job_id,
            type="step",
            kind=kind,  # type: ignore[arg-type]
            status=step.status,
            title=step.title,
            summary=step.summary,
            actor=step.actor,
            tool_name=step.tool_name,
            created_at=step.created_at,
            payload=step.model_dump(mode="json"),
        )

    def _event_from_approval(self, approval: ApprovalRequest) -> AgentEvent:
        return AgentEvent(
            event_id=approval.approval_id,
            session_id=approval.session_id,
            type="approval",
            kind="approval",
            status=approval.status,
            title=approval.requested_action,
            summary=approval.reason,
            actor="user",
            created_at=approval.created_at,
            payload=approval.model_dump(mode="json"),
        )

    def _event_from_tool_invocation(self, invocation: ToolInvocation) -> AgentEvent:
        return AgentEvent(
            event_id=invocation.invocation_id,
            session_id=invocation.session_id,
            run_id=invocation.run_id,
            type="tool_invocation",
            kind="tool_call",
            status=invocation.status,
            title=invocation.tool_name,
            summary=invocation.result_preview or invocation.error or "",
            actor="tool_policy" if invocation.status == "pending_approval" else "research_runtime",
            tool_name=invocation.tool_name,
            created_at=invocation.created_at,
            payload=invocation.model_dump(mode="json"),
        )

    def _active_job(self, session: AgentSession) -> ResearchJob | None:
        job_id = session.metadata.get("active_job_id")
        if not isinstance(job_id, str) or not job_id:
            return None
        return self.copilot.get_job(job_id)

    def _active_run(self, session: AgentSession) -> ResearchRun | None:
        if not session.active_run_id:
            return None
        return self.copilot.get_run(session.active_run_id)


def _extract_memory_candidates(content: str, session_id: str, message_id: str) -> list[MemoryItem]:
    normalized = " ".join(content.split())
    if not normalized:
        return []
    candidates: list[MemoryItem] = []
    lower = normalized.lower()
    project_constraints: list[str] = []
    user_preferences: list[str] = []
    session_facts: list[str] = []

    team_size = re.search(r"(\d+)\s*人", normalized)
    if team_size and ("团队" in normalized or "team" in lower):
        project_constraints.append(f"团队规模约束：{team_size.group(1)} 人")
    if "fastapi" in lower and "python" in lower:
        project_constraints.append("技术栈约束：Python/FastAPI")
    elif "fastapi" in lower:
        project_constraints.append("技术栈约束：FastAPI")
    elif "python" in lower:
        project_constraints.append("技术栈约束：Python")
    if "单机" in normalized and ("docker compose" in lower or ("docker" in lower and "compose" in lower)):
        project_constraints.append("部署约束：单机 Docker Compose")
    elif "单机" in normalized:
        project_constraints.append("部署约束：单机部署")
    elif "docker compose" in lower or ("docker" in lower and "compose" in lower):
        project_constraints.append("部署约束：Docker Compose")
    if "回滚" in normalized or "rollback" in lower:
        project_constraints.append("运维约束：必须支持回滚")
    if "可对话" in normalized or "可记忆" in normalized or "可审批" in normalized or "可观察" in normalized or "可评估" in normalized:
        project_constraints.append("项目目标约束：展示可对话、可记忆、可审批、可观察、可评估的研究型 agent")

    for clause in _memory_clauses(normalized):
        clause_lower = clause.lower()
        if (
            ("约束" in clause or "必须" in clause or "只能" in clause or "不允许" in clause)
            and len(clause) <= 140
            and clause not in project_constraints
        ):
            project_constraints.append(clause)
        if ("秋招" in clause or "面试" in clause or "简历" in clause) and len(clause) <= 120:
            user_preferences.append(f"长期目标：{clause}")

    if "秋招" in normalized and not any("秋招" in item for item in user_preferences):
        user_preferences.append("长期目标：用于秋招/面试展示")
    if "面试" in normalized and not any("面试" in item for item in user_preferences):
        user_preferences.append("长期目标：用于面试展示")
    if "简历" in normalized and not any("简历" in item for item in user_preferences):
        user_preferences.append("长期目标：用于简历展示")

    repo_match = re.search(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", normalized)
    if repo_match:
        session_facts.append(f"当前研究对象：{repo_match.group(1)}")
    if "workflow runtime" in lower:
        session_facts.append("当前研究目标：评估 workflow runtime 适配性")
    elif "是否适合" in normalized or "评估" in normalized:
        for clause in _memory_clauses(normalized):
            if ("评估" in clause or "是否适合" in clause) and len(clause) <= 160:
                session_facts.append(f"当前研究目标：{clause}")
                break

    for item in _dedupe_memory_texts(project_constraints)[:8]:
        candidates.append(
            _new_memory(
                scope="project",
                kind="constraint",
                content=item,
                session_id=None,
                message_id=message_id,
                confidence=0.82,
            )
        )
    for item in _dedupe_memory_texts(user_preferences)[:4]:
        candidates.append(
            _new_memory(
                scope="user",
                kind="preference",
                content=item,
                session_id=None,
                message_id=message_id,
                confidence=0.78,
            )
        )
    for item in _dedupe_memory_texts(session_facts)[:4]:
        candidates.append(
            _new_memory(
                scope="session",
                kind="fact",
                content=item,
                session_id=session_id,
                message_id=message_id,
                confidence=0.68,
            )
        )
    return candidates


def _memory_clauses(value: str) -> list[str]:
    raw_parts = re.split(r"[。；;\n]+|，(?=(?:[^（）()]|[（）()].*[（）()])*$)", value)
    clauses: list[str] = []
    for part in raw_parts:
        clause = " ".join(part.strip(" ，,。.；;").split())
        if len(clause) >= 3:
            clauses.append(clause)
    return clauses


def _dedupe_memory_texts(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = _normalize_memory_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    return result


def _new_memory(
    *,
    scope: str,
    kind: str,
    content: str,
    session_id: str | None,
    message_id: str,
    confidence: float,
) -> MemoryItem:
    return MemoryItem(
        memory_id=str(uuid.uuid4()),
        scope=scope,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        content=content,
        session_id=session_id,
        source_message_id=message_id,
        confidence=confidence,
        metadata={"extractor": "heuristic-v1"},
    )


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", value)}


def _normalize_memory_text(value: str) -> str:
    return " ".join(value.lower().split())


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _skill_keyword_matches(content: str, lower: str, keyword: str) -> bool:
    normalized = keyword.strip().lower()
    if not normalized:
        return False
    return normalized in lower or normalized in content


def _has_repo_target(content: str) -> bool:
    normalized = " ".join(content.split())
    lower = normalized.lower()
    return bool(re.search(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", normalized) or "github.com/" in lower or "repo" in lower)


def _has_team_constraints(content: str) -> bool:
    lower = content.lower()
    return any(keyword in content for keyword in ("团队", "约束", "部署", "回滚", "人数", "技术栈")) or "constraint" in lower


def _workspace_has_team_constraints(workspace: WorkspaceProfile) -> bool:
    if not workspace.metadata.get("user_configured"):
        return False
    return bool(workspace.team_context or workspace.default_stack or workspace.deployment_constraints)


def _has_comparison_signal(content: str) -> bool:
    lower = content.lower()
    return any(keyword in content for keyword in ("对比", "比较", "选型", "tradeoff", "vs")) or "compare" in lower


def _has_demo_signal(content: str) -> bool:
    lower = content.lower()
    return any(keyword in content for keyword in ("秋招", "面试", "展示", "demo")) or "demo" in lower or "interview" in lower


def _has_decision_context(content: str) -> bool:
    lower = content.lower()
    return any(keyword in content for keyword in ("评估", "是否", "适合", "采用", "决定", "recommend")) or "adoption" in lower or "evaluate" in lower


def _has_risk_signal(content: str) -> bool:
    lower = content.lower()
    return any(keyword in content for keyword in ("风险", "risk", "坑", "问题", "失败")) or "unreliable" in lower


def _select_auto_skill_script(skill: ResearchSkill) -> SkillScript | None:
    for script in skill.scripts:
        if script.enabled and script.auto:
            return script
    return None


def _merge_unique_strings(existing: list[str], additions: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in [*existing, *additions]:
        normalized = " ".join(value.split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return result
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _default_workspace_profile() -> WorkspaceProfile:
    now = _utc_now()
    return WorkspaceProfile(
        workspace_id=DEFAULT_WORKSPACE_ID,
        name="研究工作区",
        team_context="单用户本地研究台，主要用于开源引入评审和技术决策 memo。",
        default_stack=["FastAPI", "SQLite", "LangGraph", "Qdrant", "GitHub MCP"],
        deployment_constraints=["单节点", "本地优先", "外部证据只读", "不支持破坏性工具"],
        risk_policy="优先使用 web、vector 和只读 MCP 工具；审批过程必须可观察，避免破坏性动作。",
        preferred_sources=["GitHub", "官方文档", "项目 README", "本地笔记"],
        disabled_tools=[],
        created_at=now,
        updated_at=now,
        metadata={"default": True, "scope": "single-user"},
    )


def _default_skill_catalog() -> list[ResearchSkill]:
    now = _utc_now()
    return [
        ResearchSkill(
            skill_id="open_source_adoption_review",
            name="开源引入评审",
            scenario="评估某个仓库或开源项目是否适合小团队的技术栈、部署方式和风险约束。",
            trigger_keywords=["repo", "github", "开源", "采用", "引入", "适合", "adoption", "review"],
            required_inputs=["目标 repo / 项目", "团队约束", "决策问题"],
            plan_template=[
                "确认目标仓库与团队约束",
                "检查架构、维护活跃度、issue/release 信号",
                "核对部署/运维风险与替代方案",
            ],
            evaluation_focus=["来源质量", "约束覆盖", "引用", "演示可解释性"],
            created_at=now,
            updated_at=now,
            metadata={"default": True},
        ),
        ResearchSkill(
            skill_id="architecture_tradeoff_memo",
            name="架构取舍 memo",
            scenario="比较两个或多个库/架构方案，并说明技术决策取舍。",
            trigger_keywords=["对比", "比较", "选型", "tradeoff", "compare", "vs", "architecture"],
            required_inputs=["要比较的方案", "决策标准", "团队约束"],
            plan_template=[
                "明确候选方案",
                "对齐评估标准与约束",
                "比较收益、复杂度、风险和迁移成本",
            ],
            evaluation_focus=["决策清晰度", "取舍覆盖", "证据平衡"],
            created_at=now,
            updated_at=now,
            metadata={"default": False},
        ),
        ResearchSkill(
            skill_id="demo_readiness_risk_review",
            name="演示风险评审",
            scenario="检查项目是否适合用于秋招展示，以及主要风险在哪里。",
            trigger_keywords=["秋招", "面试", "展示", "demo", "risk", "风险", "presentation"],
            required_inputs=["演示 / 面试场景", "风险点", "需要验证的结论"],
            plan_template=[
                "确定演示目标和受众",
                "梳理最容易被质疑的风险点",
                "准备可复盘的证据和讲法",
            ],
            evaluation_focus=["演示可行性", "风险覆盖", "叙事清晰度"],
            created_at=now,
            updated_at=now,
            metadata={"default": False},
        ),
    ]


def _workspace_context_summary(workspace: WorkspaceProfile) -> str:
    parts: list[str] = []
    if workspace.name:
        parts.append(f"name={workspace.name}")
    if workspace.team_context:
        parts.append(f"team={workspace.team_context}")
    if workspace.default_stack:
        parts.append(f"stack={', '.join(workspace.default_stack)}")
    if workspace.deployment_constraints:
        parts.append(f"deployment={', '.join(workspace.deployment_constraints)}")
    if workspace.risk_policy:
        parts.append(f"risk={workspace.risk_policy}")
    return " ; ".join(parts)


def _stable_id(*parts: str | None) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "::".join(part or "" for part in parts)))


def _trim(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _public_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]+", r"\1***", text)
    text = re.sub(r"(?i)(api[_-]?key|token|secret|password)([=:]\s*)[^\s,;]+", r"\1\2***", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1***", text)
    return _trim(text, 420)


def _step_kind_for_trace(kind: str, actor: str) -> str:
    if kind == "tool_call":
        return "tool_call"
    if actor == "multi_agent_harness":
        return "routing"
    if kind == "evaluation" or actor == "evaluator":
        return "evaluation"
    if kind == "verification" or actor == "verifier":
        return "verification"
    if actor == "retriever":
        return "retrieval"
    if actor == "reporter":
        return "report"
    if kind == "failure":
        return "failure"
    return "research"


def _step_status_for_trace(status: str) -> str:
    if status in {"started", "completed", "failed", "skipped"}:
        return "running" if status == "started" else status
    return "completed"


def _tool_status_for_trace(status: str) -> str:
    if status == "failed":
        return "failed"
    if status == "skipped":
        return "skipped"
    if status == "started":
        return "running"
    return "completed"


def _trace_title(event: Any) -> str:
    if event.tool_name:
        return f"{event.tool_name} tool call"
    return f"{event.actor} {event.kind}"


def _tool_arguments_from_trace(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "query",
        "web_query",
        "internal_query",
        "mcp_tool_name",
        "mcp_tool_args",
        "selected_tools",
        "route_mode",
    ]
    return {key: metadata[key] for key in keys if key in metadata}


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
