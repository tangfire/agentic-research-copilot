from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, TypeVar

import httpx

from .schemas import (
    ChunkContextContract,
    ClarificationContract,
    CorpusProfile,
    EvidenceItem,
    MemoryRecord,
    PlanItem,
    PlannerContract,
    ReporterContract,
    ResearchRequest,
    ResearcherToolDecisionContract,
    ResearchReport,
    ReportSection,
    RetrievalRoute,
    SourceCompressionContract,
    SupervisorDecisionContract,
    SupervisorToolCall,
    VerificationContract,
)


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")
CLARIFICATION_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "for",
    "from",
    "how",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}
VAGUE_RESEARCH_TERMS = {
    "agent",
    "agents",
    "ai",
    "analysis",
    "architecture",
    "copilot",
    "deep",
    "project",
    "rag",
    "research",
    "system",
}

TContract = TypeVar("TContract")


@dataclass
class ModelUsage:
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ResearchModelProvider(Protocol):
    name: str
    embedding_dimensions: int

    def clarify_request(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
    ) -> tuple[ClarificationContract, ModelUsage]: ...

    def decide_researcher_action(
        self,
        *,
        item: PlanItem,
        available_tools: Sequence[str],
        previous_queries: Sequence[str],
        evidence: Sequence[EvidenceItem],
        gaps: Sequence[str],
        iteration: int,
        max_iterations: int,
    ) -> tuple[ResearcherToolDecisionContract, ModelUsage]: ...

    def draft_plan(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]: ...

    def supervise_research(
        self,
        request: ResearchRequest,
        research_brief: str,
        plan: Sequence[PlanItem],
        retrieval_routes: Sequence[RetrievalRoute],
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[SupervisorDecisionContract, ModelUsage]: ...

    def assess_report(
        self,
        report: ResearchReport,
        evidence: Sequence[EvidenceItem],
        plan: Sequence[PlanItem],
        *,
        revision_count: int = 0,
        max_revisions: int = 2,
    ) -> tuple[VerificationContract, ModelUsage]: ...

    def compose_report(
        self,
        topic: str,
        sections: Sequence[ReportSection],
        evidence: Sequence[EvidenceItem],
        confidence: float,
    ) -> tuple[ReporterContract, ModelUsage]: ...

    def compress_source(
        self,
        *,
        query: str,
        title: str,
        url: str | None,
        raw_content: str,
    ) -> tuple[SourceCompressionContract, ModelUsage]: ...

    def contextualize_chunk(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, Any],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> tuple[ChunkContextContract, ModelUsage]: ...

    def embed_text(self, text: str) -> tuple[list[float], ModelUsage]: ...
    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], ModelUsage]: ...


class DeterministicResearchModelProvider:
    name = "deterministic"

    def __init__(self, embedding_dimensions: int = 256) -> None:
        self.embedding_dimensions = max(32, embedding_dimensions)

    def clarify_request(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
    ) -> tuple[ClarificationContract, ModelUsage]:
        topic = request.topic.strip()
        missing_dimensions = _clarification_missing_dimensions(topic)
        if missing_dimensions:
            question = (
                "Before I start the research, please clarify the scope: "
                + "; ".join(missing_dimensions[:3])
                + "."
            )
            contract = ClarificationContract(
                need_clarification=True,
                question=question,
                verification="",
                missing_dimensions=missing_dimensions,
                confidence=0.78,
            )
        else:
            context_bits = [
                f"I have enough information to research '{topic}' at {request.depth} depth.",
                "I will turn it into a concrete research brief, gather citation-backed evidence, and return a traceable report.",
            ]
            if corpus_profile.has_private_docs and request.include_private_docs:
                context_bits.append(
                    f"I will also consider {corpus_profile.document_count} uploaded document segment(s) when they are relevant."
                )
            if memory_records and request.use_memory:
                context_bits.append(f"I will reuse {len(memory_records)} memory item(s) only as supporting context.")
            contract = ClarificationContract(
                need_clarification=False,
                question="",
                verification=" ".join(context_bits),
                missing_dimensions=[],
                confidence=0.86,
            )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-clarifier",
            prompt_tokens=48 + len(topic.split()) * 4,
            completion_tokens=max(16, len((contract.question or contract.verification).split()) * 3),
            latency_ms=1,
        )
        return contract, usage

    def decide_researcher_action(
        self,
        *,
        item: PlanItem,
        available_tools: Sequence[str],
        previous_queries: Sequence[str],
        evidence: Sequence[EvidenceItem],
        gaps: Sequence[str],
        iteration: int,
        max_iterations: int,
    ) -> tuple[ResearcherToolDecisionContract, ModelUsage]:
        if not gaps and evidence:
            action = ResearcherToolDecisionContract(
                action="ResearchComplete",
                rationale="Evidence and source sufficiency criteria are satisfied.",
                reflection=f"Research unit {item.id} can stop with {len(evidence)} evidence item(s).",
                completion_reason="sufficiency_met",
                confidence=0.86,
            )
        elif iteration > 1 and "mcp_tool" in available_tools and not any(
            item.metadata.get("source_channel") == "mcp" for item in evidence
        ):
            action = ResearcherToolDecisionContract(
                action="mcp_tool",
                query=_researcher_follow_up_query(item, previous_queries, evidence, gaps),
                mcp_tool_name="search_grounding_corpus",
                rationale="Use the configured MCP workspace tools as an additional grounding channel.",
                reflection="ODR-style researcher loop can call MCP tools when search evidence is still insufficient.",
                confidence=0.72,
            )
        elif iteration >= max_iterations and evidence:
            action = ResearcherToolDecisionContract(
                action="ResearchComplete",
                rationale="Iteration budget is exhausted; return the best grounded evidence collected so far.",
                reflection="The researcher stops because max_iterations was reached.",
                completion_reason="iteration_limit_reached",
                confidence=0.62,
            )
        else:
            action = ResearcherToolDecisionContract(
                action="web_search",
                query=_researcher_follow_up_query(item, previous_queries, evidence, gaps),
                rationale="Search/read another source because evidence is not sufficient yet.",
                reflection="Continue with web_search before deciding whether the delegated research unit is complete.",
                confidence=0.74,
            )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-researcher-tool-loop",
            prompt_tokens=64 + len(previous_queries) * 8 + len(evidence) * 12,
            completion_tokens=48,
            latency_ms=1,
        )
        return action, usage

    def draft_plan(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]:
        topic = request.topic.strip()
        memory_context = _summarize_memory(memory_records)
        focus = _focus_for_topic(topic)
        brief_bits = [
            f"Research the topic '{topic}'.",
            "Prioritize citation-backed evidence, explicit handoffs, and a verifiable source index.",
            "Prefer official docs, primary papers, or original system sources when they are available.",
            f"Focus on {focus}.",
        ]
        if corpus_profile.has_private_docs:
            brief_bits.append(
                f"Ground the answer with {corpus_profile.document_count} uploaded context documents from {corpus_profile.source_count} sources."
            )
        if memory_context:
            brief_bits.append(f"Reuse memory where it is relevant: {memory_context}.")
        if revision_count > 0 and revision_notes:
            brief_bits.append("Repair the previously flagged gaps: " + "; ".join(revision_notes[:3]))

        plan_items = _build_plan_items(request, topic, revision_count=revision_count)
        assumptions = [
            "The final answer must remain source-backed and inspectable.",
            "The supervisor may revise the plan if verification exposes citation or coverage gaps.",
        ]
        if corpus_profile.has_private_docs:
            assumptions.append("Internal grounding is available and should be preferred for project-specific facts.")
        if memory_records:
            assumptions.append("Session and canonical memory can shorten the evidence search.")

        success_criteria = [
            "Every substantive section has citations.",
            "The run records handoffs and trace events.",
            "Verifier issues are either resolved or surfaced as a failure state.",
        ]
        if revision_notes:
            success_criteria.append("The next pass addresses the previously cited gaps.")

        contract = PlannerContract(
            research_brief=" ".join(brief_bits),
            plan=plan_items[: request.max_sections],
            assumptions=assumptions,
            success_criteria=success_criteria,
            revision_budget=request.max_revisions,
            confidence=min(0.92, 0.68 + 0.03 * len(plan_items) + (0.03 if revision_count else 0.0)),
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-planner",
            prompt_tokens=96 + len(topic.split()) * 4,
            completion_tokens=128 + len(contract.plan) * 16,
            latency_ms=3,
        )
        return contract, usage

    def supervise_research(
        self,
        request: ResearchRequest,
        research_brief: str,
        plan: Sequence[PlanItem],
        retrieval_routes: Sequence[RetrievalRoute],
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[SupervisorDecisionContract, ModelUsage]:
        route_lookup = {route.plan_item_id: route for route in retrieval_routes}
        reflection_bits = [
            "Use an Open Deep Research-style supervisor loop: think, delegate focused research, then complete only after verification.",
            f"The plan has {len(plan)} research unit(s).",
            "Prefer official, primary, or original sources when available, and keep source quality visible in the evaluator instead of hard-filtering it at runtime.",
        ]
        if corpus_profile.has_private_docs:
            reflection_bits.append(
                f"Internal grounding is available from {corpus_profile.document_count} document(s), so hybrid/internal routes should be delegated when relevant."
            )
        if memory_records:
            reflection_bits.append(f"Memory recall returned {len(memory_records)} item(s); reuse it as context but keep final claims citation-backed.")
        if revision_notes:
            reflection_bits.append("Revision notes must be repaired: " + "; ".join(revision_notes[:3]))

        tool_calls: list[SupervisorToolCall] = [
            SupervisorToolCall(
                name="think_tool",
                rationale="Reflect before delegating research units.",
                reflection=" ".join(reflection_bits),
            )
        ]
        for item in plan:
            if not item.requires_research:
                continue
            route = route_lookup.get(item.id)
            route_hint = f" Route mode: {route.mode}. Tools: {', '.join(route.selected_tools)}." if route else ""
            selected_tools = route.selected_tools if route else ["web_search"]
            tool_calls.append(
                SupervisorToolCall(
                    name="ConductResearch",
                    rationale=f"Delegate the plan item so a focused researcher can gather evidence.{route_hint}",
                    plan_item_ids=[item.id],
                    research_topic=f"{item.question} Purpose: {item.purpose}",
                    mode=route.mode if route else "external",
                    selected_tools=selected_tools,
                    web_queries=route.web_queries if route else [item.search_query or item.question],
                    internal_queries=route.internal_queries if route else [],
                    memory_query=route.memory_query if route else f"{request.topic} {item.purpose}",
                    min_evidence=route.min_evidence if route else 1,
                    min_sources=route.min_sources if route else 1,
                    sufficiency_criteria=route.sufficiency_criteria
                    if route
                    else ["preserve citations for report assembly"],
                )
            )
        tool_calls.append(
            SupervisorToolCall(
                name="ResearchComplete",
                rationale="Complete only after delegated research returns enough citation-backed evidence and verifier/evaluator checks pass.",
                reflection="Completion is gated by citation coverage, evidence sufficiency, source diversity, and revision budget.",
            )
        )
        contract = SupervisorDecisionContract(
            reflection=" ".join(reflection_bits),
            tool_calls=tool_calls,
            completion_criteria=[
                "Each required plan item has enough evidence for its route.",
                "Final report citations map only to existing evidence.",
                "Verifier and evaluator either pass the run or trigger a revision loop.",
            ],
            max_concurrent_research_units=min(max(1, request.max_sections), max(1, len(plan)), 4),
            confidence=min(0.92, 0.66 + len(plan) * 0.04 + (0.04 if corpus_profile.has_private_docs else 0.0)),
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-supervisor",
            prompt_tokens=112 + len(plan) * 20,
            completion_tokens=96 + len(tool_calls) * 18,
            latency_ms=3,
        )
        return contract, usage

    def assess_report(
        self,
        report: ResearchReport,
        evidence: Sequence[EvidenceItem],
        plan: Sequence[PlanItem],
        *,
        revision_count: int = 0,
        max_revisions: int = 2,
    ) -> tuple[VerificationContract, ModelUsage]:
        issues: list[str] = []
        critical_issues: list[str] = []

        if not evidence:
            issues.append("No evidence attached to the report.")
            critical_issues.append("No evidence attached to the report.")
        if report.sections and not report.citations:
            issues.append("Report sections exist but no citations were assembled.")
            critical_issues.append("Report sections exist but no citations were assembled.")
        uncited_sections = [
            section.heading
            for section in report.sections
            if section.content.strip() and not section.citations
        ]
        if uncited_sections:
            issues.append(f"Sections missing citations: {len(uncited_sections)}")
        if report.citations and not report.source_index:
            issues.append("Source index is missing despite attached citations.")
            critical_issues.append("Source index is missing despite attached citations.")
        if len(report.sections) < 3:
            issues.append("Report is too thin for interview-grade explanation.")
        if report.confidence < 0.55:
            issues.append("Confidence is too low.")
        unique_sources = {
            item.source
            for item in evidence
            if item.source and item.source not in {"internal-note", "memory"}
        }
        if evidence and len(unique_sources) < 2:
            issues.append("Evidence sources are not diverse enough.")
        uncovered = [item.question for item in plan if item.requires_research and item.evidence_count == 0]
        if uncovered:
            issues.append(f"Uncovered plan items: {len(uncovered)}")
        if report.source_count and report.source_count < min(3, len(evidence)):
            issues.append("Source count is weaker than the evidence volume.")
        coverage_score = 0.0
        if plan:
            covered = len(plan) - len(uncovered)
            coverage_score = max(0.0, min(1.0, covered / len(plan)))

        should_revise = bool(critical_issues or (issues and revision_count < max_revisions))
        revision_reason = "; ".join((critical_issues or issues)[:2]) if (critical_issues or issues) else None
        contract = VerificationContract(
            issues=issues,
            critical_issues=critical_issues,
            should_revise=should_revise,
            revision_reason=revision_reason,
            confidence=max(0.0, min(0.96, 0.56 + coverage_score * 0.2 - (0.1 if critical_issues else 0.0))),
            coverage_score=coverage_score,
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-verifier",
            prompt_tokens=72 + len(report.sections) * 16,
            completion_tokens=64 + len(issues) * 8,
            latency_ms=2,
        )
        return contract, usage

    def compose_report(
        self,
        topic: str,
        sections: Sequence[ReportSection],
        evidence: Sequence[EvidenceItem],
        confidence: float,
    ) -> tuple[ReporterContract, ModelUsage]:
        unique_sources: list[str] = []
        seen_sources: set[str] = set()
        source_index: list[str] = []
        seen_citations: set[str] = set()
        for index, item in enumerate(evidence, start=1):
            source_key = item.url or f"{item.source}:{item.title}"
            if source_key not in seen_citations:
                seen_citations.add(source_key)
                suffix = f" - {item.url}" if item.url else ""
                source_index.append(f"[{index}] {item.title} ({item.source}){suffix}")
            if item.source not in seen_sources:
                seen_sources.add(item.source)
                unique_sources.append(item.source)

        contract = ReporterContract(
            title=f"{topic} Research Brief",
            summary=(
                f"A structured research brief on {topic} with {len(sections)} sections and "
                f"{len(unique_sources)} source groups."
            ),
            highlights=[
                f"{section.heading}: {section.content[:120].rstrip()}"
                for section in list(sections)[:3]
            ],
            recommendations=[
                "Keep citations attached to each substantive claim.",
                "Expose handoffs and trace events in the UI for debugging and interviews.",
                "Use the same contract to swap in a real LLM provider later without changing orchestration.",
            ],
            source_index=source_index,
            confidence=confidence,
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-reporter",
            prompt_tokens=64 + len(sections) * 12,
            completion_tokens=72 + len(evidence) * 8,
            latency_ms=2,
        )
        return contract, usage

    def compress_source(
        self,
        *,
        query: str,
        title: str,
        url: str | None,
        raw_content: str,
    ) -> tuple[SourceCompressionContract, ModelUsage]:
        contract = _heuristic_source_compression(query=query, title=title, raw_content=raw_content)
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-source-compressor",
            prompt_tokens=max(1, len(raw_content) // 4),
            completion_tokens=max(16, len(contract.summary) // 4),
            latency_ms=2,
        )
        return contract, usage

    def contextualize_chunk(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, Any],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> tuple[ChunkContextContract, ModelUsage]:
        contract = _heuristic_chunk_context(
            document_title=document_title,
            source=source,
            metadata=metadata,
            document_excerpt=document_excerpt,
            chunk_text=chunk_text,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-contextual-retrieval",
            prompt_tokens=max(1, (len(document_excerpt) + len(chunk_text)) // 4),
            completion_tokens=max(16, len(contract.context) // 4),
            latency_ms=1,
        )
        return contract, usage

    def embed_text(self, text: str) -> tuple[list[float], ModelUsage]:
        vector = _hashed_dense_vector(text, self.embedding_dimensions)
        usage = ModelUsage(provider=self.name, model="hashed-embedding", prompt_tokens=max(1, len(text) // 4), latency_ms=1)
        return vector, usage

    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], ModelUsage]:
        vectors = [_hashed_dense_vector(text, self.embedding_dimensions) for text in texts]
        usage = ModelUsage(
            provider=self.name,
            model="hashed-embedding",
            prompt_tokens=sum(max(1, len(text) // 4) for text in texts),
            latency_ms=max(1, len(texts)),
        )
        return vectors, usage


class OpenAICompatibleResearchModelProvider:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        chat_model: str,
        embedding_model: str,
        timeout_seconds: float = 30.0,
        temperature: float = 0.2,
        embedding_dimensions: int = 256,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.embedding_dimensions = max(32, embedding_dimensions)

    def clarify_request(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
    ) -> tuple[ClarificationContract, ModelUsage]:
        payload = {
            "request": request.model_dump(),
            "corpus_profile": corpus_profile.model_dump(),
            "memory_records": [record.model_dump() for record in memory_records[:6]],
            "instructions": (
                "Follow the Open Deep Research clarify_with_user phase. Decide whether the "
                "user request is specific enough to start research. Ask at most one concise "
                "clarifying question when the scope, target audience, decision context, or "
                "required source type is genuinely missing. If enough information is present, "
                "return a short verification message summarizing the intended research scope."
            ),
        }
        contract, usage = self._chat_structured(
            system_prompt=(
                "You are the clarification gate for an AI Research Copilot. Return valid JSON "
                "only that conforms to the supplied schema. Do not ask unnecessary questions. "
                "Prefer proceeding when the request already contains a concrete topic, target, "
                "comparison, implementation scope, or deliverable."
            ),
            user_payload=payload,
            schema=ClarificationContract.model_json_schema(),
            response_model=ClarificationContract,
        )
        return _normalize_clarification_contract(contract, request), usage

    def decide_researcher_action(
        self,
        *,
        item: PlanItem,
        available_tools: Sequence[str],
        previous_queries: Sequence[str],
        evidence: Sequence[EvidenceItem],
        gaps: Sequence[str],
        iteration: int,
        max_iterations: int,
    ) -> tuple[ResearcherToolDecisionContract, ModelUsage]:
        payload = {
            "plan_item": item.model_dump(),
            "available_tools": list(available_tools),
            "previous_queries": list(previous_queries),
            "evidence": [
                {
                    "title": item.title,
                    "source": item.source,
                    "kind": item.kind,
                    "url": item.url,
                    "snippet": item.snippet,
                    "score": item.score,
                }
                for item in evidence[:10]
            ],
            "gaps": list(gaps),
            "iteration": iteration,
            "max_iterations": max_iterations,
            "instructions": (
                "Follow the Open Deep Research researcher loop. Choose exactly one next action: "
                "think_tool for reflection, web_search for a new external query, mcp_tool for a "
                "configured MCP tool call, or ResearchComplete when enough evidence has been collected "
                "or the iteration budget is exhausted. Use mcp_tool only when it is listed in "
                "available_tools. When using mcp_tool, set mcp_tool_name when a known workspace tool "
                "matches the need: search_grounding_corpus for ingested documents, recall_project_memory "
                "for prior memory, inspect_research_runs for replay/evaluation, or check_demo_readiness "
                "for runtime demo checks. Keep the query concrete and source-oriented."
            ),
        }
        contract, usage = self._chat_structured(
            system_prompt=(
                "You are a focused researcher inside an AI Research Copilot. Return valid JSON "
                "only that conforms to the supplied schema. Be decisive: search when evidence is "
                "thin, use MCP tools when configured and useful, reflect when a pause is needed, "
                "and complete when evidence is sufficient or the budget is exhausted."
            ),
            user_payload=payload,
            schema=ResearcherToolDecisionContract.model_json_schema(),
            response_model=ResearcherToolDecisionContract,
        )
        return _normalize_researcher_action(contract, item, available_tools, previous_queries, evidence, gaps), usage

    def draft_plan(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]:
        payload = {
            "request": request.model_dump(),
            "corpus_profile": corpus_profile.model_dump(),
            "memory_records": [record.model_dump() for record in memory_records[:8]],
            "revision_count": revision_count,
            "revision_notes": list(revision_notes)[:8],
        }
        schema = PlannerContract.model_json_schema()
        return self._chat_structured(
            system_prompt=(
                "You are the planner for a deep research copilot. Your job is to decompose a research topic "
                "into a structured plan of 3-5 focused sub-questions, each independently researchable and "
                "mapping to a distinct section of the final report.\n\n"
                "Guidelines:\n"
                "- Write a clear research_brief that summarizes the goal, approach, and key constraints.\n"
                "- Each plan item must have a specific question (not vague), a clear purpose, and an "
                "optimized search_query tuned for search engines (shorter and keyword-focused).\n"
                "- Avoid overlapping questions. Cover different angles: background, methods, comparisons, "
                "limitations, and practical implications where relevant.\n"
                "- If revision_count > 0, use revision_notes to address previously identified gaps.\n"
                "- If private documents are available (corpus_profile.has_private_docs), include items "
                "that can be grounded in those documents.\n\n"
                "Return valid JSON only that conforms to the supplied schema."
            ),
            user_payload=payload,
            schema=schema,
            response_model=PlannerContract,
        )

    def supervise_research(
        self,
        request: ResearchRequest,
        research_brief: str,
        plan: Sequence[PlanItem],
        retrieval_routes: Sequence[RetrievalRoute],
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[SupervisorDecisionContract, ModelUsage]:
        payload = {
            "request": request.model_dump(),
            "research_brief": research_brief,
            "plan": [item.model_dump() for item in plan],
            "retrieval_routes": [route.model_dump() for route in retrieval_routes],
            "corpus_profile": corpus_profile.model_dump(),
            "memory_records": [record.model_dump() for record in memory_records[:8]],
            "revision_count": revision_count,
            "revision_notes": list(revision_notes)[:8],
                "instructions": (
                    "Follow the Open Deep Research supervisor pattern. First reflect with a "
                    "think_tool-style call, then use ConductResearch calls to delegate concrete "
                    "research units. Use ResearchComplete only as the completion decision after "
                    "delegation and verification criteria are clear. Preserve the provided "
                    "plan_item_ids; do not invent IDs. Each ConductResearch call must choose "
                    "mode, selected_tools, web_queries/internal_queries, memory_query, min_evidence, "
                    "min_sources, and sufficiency_criteria. Prefer primary or official sources "
                    "when they are available, keep source quality visible in evaluation, and "
                    "treat retrieval_routes as optional candidate hints, not as mandatory final routing decisions."
                ),
            }
        return self._chat_structured(
            system_prompt=(
                "You are the research supervisor for an AI Research Copilot. Return valid JSON only "
                "that conforms to the supplied schema. Emit Open Deep Research-style tool calls: "
                "think_tool for reflection, ConductResearch for delegated research, and "
                "ResearchComplete for completion criteria. ConductResearch must include the evidence "
                "tools and query rewrites needed by the delegated unit. Keep decisions inspectable and "
                "citation-oriented."
            ),
            user_payload=payload,
            schema=SupervisorDecisionContract.model_json_schema(),
            response_model=SupervisorDecisionContract,
        )

    def assess_report(
        self,
        report: ResearchReport,
        evidence: Sequence[EvidenceItem],
        plan: Sequence[PlanItem],
        *,
        revision_count: int = 0,
        max_revisions: int = 2,
    ) -> tuple[VerificationContract, ModelUsage]:
        payload = {
            "report": report.model_dump(),
            "evidence": [item.model_dump() for item in evidence[:20]],
            "plan": [item.model_dump() for item in plan[:12]],
            "revision_count": revision_count,
            "max_revisions": max_revisions,
        }
        return self._chat_structured(
            system_prompt=(
                "You are the verifier for a research copilot. Return valid JSON only "
                "that conforms to the supplied schema."
            ),
            user_payload=payload,
            schema=VerificationContract.model_json_schema(),
            response_model=VerificationContract,
        )

    def compose_report(
        self,
        topic: str,
        sections: Sequence[ReportSection],
        evidence: Sequence[EvidenceItem],
        confidence: float,
    ) -> tuple[ReporterContract, ModelUsage]:
        evidence_index = [
            {
                "index": index,
                "title": item.title,
                "source": item.source,
                "kind": item.kind,
                "url": item.url,
                "snippet": item.snippet,
                "content": (item.content or "")[:1200],
                "score": item.score,
            }
            for index, item in enumerate(evidence[:24], start=1)
        ]
        payload = {
            "topic": topic,
            "sections": [section.model_dump() for section in sections[:8]],
            "evidence_index": evidence_index,
            "confidence": confidence,
            "instructions": (
                "Synthesize the final report sections from the draft sections and evidence. "
                "Use the same language as the topic/request. Each section must be specific, "
                "citation-backed, and balanced. Use citation_indexes to reference only the "
                "provided evidence_index entries. Do not invent sources, URLs, facts, or citations."
            ),
        }
        return self._chat_structured(
            system_prompt=(
                "You are the final report writer for a deep research copilot, following the "
                "Open Deep Research pattern of synthesizing compressed findings into a "
                "comprehensive citation-backed report. Return valid JSON only that conforms "
                "to the supplied schema. Populate sections with rewritten section drafts and "
                "citation_indexes that map to the provided evidence_index."
            ),
            user_payload=payload,
            schema=ReporterContract.model_json_schema(),
            response_model=ReporterContract,
        )

    def compress_source(
        self,
        *,
        query: str,
        title: str,
        url: str | None,
        raw_content: str,
    ) -> tuple[SourceCompressionContract, ModelUsage]:
        payload = {
            "query": query,
            "source": {
                "title": title,
                "url": url,
            },
            "raw_content": raw_content,
            "instructions": (
                "Compress the source for a downstream research agent. Preserve concrete facts, "
                "numbers, dates, named entities, and any caveats relevant to the query. Do not "
                "invent facts. key_excerpts must be short excerpts or close paraphrases grounded "
                "in the provided raw_content. Set relevance between 0 and 1."
            ),
        }
        return self._chat_structured(
            system_prompt=(
                "You are a source reader for an AI Research Copilot. Return valid JSON only "
                "that conforms to the supplied schema. The output will become citation-backed "
                "evidence, so preserve supportable facts and list limitations when the source "
                "is thin or off-topic."
            ),
            user_payload=payload,
            schema=SourceCompressionContract.model_json_schema(),
            response_model=SourceCompressionContract,
        )

    def contextualize_chunk(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, Any],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> tuple[ChunkContextContract, ModelUsage]:
        payload = {
            "document": {
                "title": document_title,
                "source": source,
                "metadata": _scalar_metadata(metadata, limit=12),
                "excerpt": document_excerpt[:12000],
            },
            "chunk": {
                "index": chunk_index + 1,
                "total_chunks": total_chunks,
                "text": chunk_text[:2400],
            },
            "instructions": (
                "Generate an indexing-time contextual retrieval prefix for this chunk. "
                "The context should be 50-100 tokens, grounded only in the document and chunk, "
                "and explain where the chunk sits in the document, what local topic it covers, "
                "and which concrete entities/terms matter for dense retrieval and BM25. "
                "Do not answer a user question and do not invent facts."
            ),
        }
        contract, usage = self._chat_structured(
            system_prompt=(
                "You create Anthropic-style contextual retrieval prefixes for RAG indexing. "
                "Return valid JSON only that conforms to the supplied schema. The context field "
                "must be one concise paragraph suitable to prepend to a chunk before embedding "
                "and BM25 indexing."
            ),
            user_payload=payload,
            schema=ChunkContextContract.model_json_schema(),
            response_model=ChunkContextContract,
        )
        return _normalize_chunk_context_contract(contract), usage

    def embed_text(self, text: str) -> tuple[list[float], ModelUsage]:
        start = time.perf_counter()
        payload = {"model": self.embedding_model, "input": text, "dimensions": self.embedding_dimensions}
        with self._client() as client:
            response = client.post("/embeddings", json=payload)
            response.raise_for_status()
            body = response.json()
        vector = list(body["data"][0]["embedding"])
        usage = self._usage_from_body(body, start, self.embedding_model)
        return vector, usage

    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], ModelUsage]:
        start = time.perf_counter()
        payload = {"model": self.embedding_model, "input": list(texts), "dimensions": self.embedding_dimensions}
        with self._client() as client:
            response = client.post("/embeddings", json=payload)
            response.raise_for_status()
            body = response.json()
        vectors = [list(item["embedding"]) for item in body["data"]]
        usage = self._usage_from_body(body, start, self.embedding_model)
        return vectors, usage

    def _chat_structured(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema: dict[str, Any],
        response_model: type[TContract],
    ) -> tuple[TContract, ModelUsage]:
        start = time.perf_counter()
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "schema": schema,
                            "input": user_payload,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        with self._client() as client:
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
        content = _extract_chat_content(body)
        model = response_model.model_validate_json(_extract_json_object(content))
        usage = self._usage_from_body(body, start, self.chat_model)
        return model, usage

    def _client(self) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout_seconds,
        )

    def _usage_from_body(self, body: dict[str, Any], start: float, model: str) -> ModelUsage:
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ModelUsage(
            provider=self.name,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            cost_usd=0.0,
            latency_ms=latency_ms,
        )


def build_model_provider(settings: Any) -> ResearchModelProvider:
    if getattr(settings, "model_provider", "deterministic") == "openai_compatible" and getattr(settings, "model_base_url", ""):
        return OpenAICompatibleResearchModelProvider(
            base_url=settings.model_base_url,
            api_key=getattr(settings, "model_api_key", ""),
            chat_model=getattr(settings, "model_chat_model", "gpt-4o-mini"),
            embedding_model=getattr(settings, "model_embedding_model", "text-embedding-3-small"),
            timeout_seconds=float(getattr(settings, "model_timeout_seconds", 30.0)),
            temperature=float(getattr(settings, "model_temperature", 0.2)),
            embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
        )
    return DeterministicResearchModelProvider(
        embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
    )


def build_embedding_provider(settings: Any, model_provider: ResearchModelProvider | None = None) -> ResearchModelProvider:
    if getattr(settings, "embedding_provider", "model") == "deterministic":
        return DeterministicResearchModelProvider(
            embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
        )
    if getattr(settings, "embedding_provider", "model") == "openai_compatible":
        return OpenAICompatibleResearchModelProvider(
            base_url=getattr(settings, "embedding_base_url", "") or getattr(settings, "model_base_url", ""),
            api_key=getattr(settings, "embedding_api_key", "") or getattr(settings, "model_api_key", ""),
            chat_model=getattr(settings, "model_chat_model", "gpt-4o-mini"),
            embedding_model=getattr(
                settings,
                "embedding_model",
                getattr(settings, "model_embedding_model", "text-embedding-3-small"),
            ),
            timeout_seconds=float(getattr(settings, "model_timeout_seconds", 30.0)),
            temperature=float(getattr(settings, "model_temperature", 0.2)),
            embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
        )
    return model_provider or build_model_provider(settings)


def _build_plan_items(request: ResearchRequest, topic: str, *, revision_count: int = 0) -> list[PlanItem]:
    base = topic.strip()
    seed = hashlib.md5(base.encode("utf-8")).hexdigest()[:6]
    plan_items = [
        (
            "problem",
            f"What is the core problem behind {base}?",
            "Frame the motivation and scope.",
        ),
        (
            "workflow",
            f"How does the end-to-end workflow for {base} operate?",
            "Map the execution path and orchestration.",
        ),
        (
            "data",
            f"What evidence, memory, or retrieval sources support {base}?",
            "Explain the knowledge layer and context reuse.",
        ),
        (
            "risk",
            f"What are the main failure modes and trade-offs of {base}?",
            "Surface risks and quality constraints.",
        ),
    ]
    if request.depth in {"standard", "deep"}:
        plan_items.append(
            (
                "verification",
                f"How should {base} be verified, evaluated, and replayed?",
                "Connect verification, observability, and replay.",
            )
        )
    if request.depth == "deep":
        plan_items.append(
            (
                "delivery",
                f"What is needed to ship and operate {base} as a usable product?",
                "Outline deployment and operating considerations.",
            )
        )
    if revision_count > 0:
        plan_items.append(
            (
                "repair",
                f"What gaps must be repaired before the answer for {base} can be published?",
                "Force the supervisor to close verification gaps.",
            )
        )

    return [
        PlanItem(
            id=f"{seed}-{name}",
            question=question,
            purpose=purpose,
            search_query=f"{base} {name} {purpose}",
            revision_hint="repair citation gaps" if name == "repair" else None,
        )
        for name, question, purpose in plan_items[: request.max_sections]
    ]


def _focus_for_topic(topic: str) -> str:
    lower = topic.lower()
    if any(word in lower for word in ("agent", "copilot", "workflow")):
        return "agent orchestration, memory, verification, and observability"
    if any(word in lower for word in ("rag", "retrieval", "knowledge")):
        return "retrieval quality, grounding, and context management"
    if any(word in lower for word in ("e-commerce", "order", "payment", "marketing")):
        return "transactional reliability, state consistency, and recovery"
    return "system design, execution flow, and measurable outcomes"


def _clarification_missing_dimensions(topic: str) -> list[str]:
    cleaned = _clean_text(topic)
    tokens = [
        token.lower()
        for token in TOKEN_PATTERN.findall(cleaned)
        if token.lower() not in CLARIFICATION_STOPWORDS
    ]
    latin_tokens = [token for token in tokens if re.search(r"[a-zA-Z0-9]", token)]
    is_short = len(cleaned) < 18 and len(tokens) <= 3
    is_generic = bool(latin_tokens) and all(token in VAGUE_RESEARCH_TERMS for token in latin_tokens)
    if not is_short and not is_generic:
        return []

    missing = [
        "the concrete research target or decision you want the report to support",
        "the expected output shape, such as comparison, implementation plan, risk analysis, or interview notes",
    ]
    if len(tokens) <= 2:
        missing.append("any constraints such as timeframe, domain, preferred sources, or technologies")
    return missing


def _normalize_clarification_contract(
    contract: ClarificationContract,
    request: ResearchRequest,
) -> ClarificationContract:
    missing_dimensions = [
        _trim_text(item, 180)
        for item in contract.missing_dimensions
        if _clean_text(item)
    ][:5]
    question = _trim_text(contract.question, 500)
    verification = _trim_text(contract.verification, 500)
    if contract.need_clarification:
        if not question:
            fallback_missing = missing_dimensions or _clarification_missing_dimensions(request.topic)
            question = (
                "Before I start the research, please clarify the scope: "
                + "; ".join(fallback_missing[:3])
                + "."
            )
        verification = ""
    else:
        question = ""
        if not verification:
            verification = (
                f"I have enough information to research '{request.topic}' at {request.depth} depth. "
                "I will build a concrete research brief and gather citation-backed evidence."
            )
        missing_dimensions = []
    return contract.model_copy(
        update={
            "question": question,
            "verification": verification,
            "missing_dimensions": missing_dimensions,
            "confidence": max(0.0, min(1.0, float(contract.confidence or 0.0))),
        }
    )


def _researcher_follow_up_query(
    item: PlanItem,
    previous_queries: Sequence[str],
    evidence: Sequence[EvidenceItem],
    gaps: Sequence[str],
) -> str:
    base = item.search_query or item.question
    used = {query.strip().lower() for query in previous_queries if query.strip()}
    source_terms = " ".join(evidence_item.title for evidence_item in evidence[:2] if evidence_item.title)
    candidates = [
        base,
        f"{item.question} {item.purpose} official source evidence",
        f"{base} independent source comparison {source_terms}".strip(),
        f"{base} limitations verification {' '.join(gaps[:2])}".strip(),
    ]
    for candidate in candidates:
        normalized = _clean_text(candidate)
        if normalized and normalized.lower() not in used:
            return normalized
    return _clean_text(f"{base} follow up evidence")


def _normalize_researcher_action(
    contract: ResearcherToolDecisionContract,
    item: PlanItem,
    available_tools: Sequence[str],
    previous_queries: Sequence[str],
    evidence: Sequence[EvidenceItem],
    gaps: Sequence[str],
) -> ResearcherToolDecisionContract:
    available = set(available_tools)
    action = contract.action
    if action == "mcp_tool" and "mcp_tool" not in available:
        action = "web_search"
    if action == "web_search" and "web_search" not in available:
        action = "ResearchComplete" if evidence else "think_tool"
    if action == "ResearchComplete" and not evidence and gaps:
        action = "web_search" if "web_search" in available else "think_tool"

    query = _trim_text(contract.query or "", 320)
    if action in {"web_search", "mcp_tool"} and not query:
        query = _researcher_follow_up_query(item, previous_queries, evidence, gaps)
    if action not in {"web_search", "mcp_tool"}:
        query = None

    completion_reason = _trim_text(contract.completion_reason or "", 160) or None
    if action == "ResearchComplete" and completion_reason is None:
        completion_reason = "sufficiency_met" if not gaps and evidence else "research_complete"

    return contract.model_copy(
        update={
            "action": action,
            "query": query,
            "mcp_tool_name": _trim_text(contract.mcp_tool_name or "", 120) or None,
            "rationale": _trim_text(contract.rationale, 500),
            "reflection": _trim_text(contract.reflection, 700),
            "completion_reason": completion_reason,
            "confidence": max(0.0, min(1.0, float(contract.confidence or 0.0))),
        }
    )


def _summarize_memory(records: Sequence[MemoryRecord]) -> str:
    if not records:
        return ""
    summary_bits: list[str] = []
    for record in records[:3]:
        summary_bits.append(f"{record.key}: {record.value[:72].rstrip()}")
    return "; ".join(summary_bits)


def _heuristic_source_compression(
    *,
    query: str,
    title: str,
    raw_content: str,
    max_summary_chars: int = 900,
    max_excerpt_chars: int = 260,
) -> SourceCompressionContract:
    cleaned = _clean_text(raw_content)
    if not cleaned:
        return SourceCompressionContract(summary="", key_excerpts=[], relevance=0.0, limitations=["empty source content"])
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", query)
        if token.lower() not in {"about", "and", "for", "from", "how", "the", "this", "what", "with"}
    }
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if not sentences:
        sentences = [cleaned]
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        score = sum(1 for term in terms if term in lower)
        scored.append((score, -index, sentence))
    selected = [
        sentence
        for score, _, sentence in sorted(scored, reverse=True)[:5]
        if score > 0
    ] or sentences[:3]
    summary = _trim_text(" ".join(selected), max_summary_chars)
    excerpts = [_trim_text(sentence, max_excerpt_chars) for sentence in selected[:5]]
    top_scores = sorted(scored, reverse=True)[:8]
    relevance = min(1.0, max(0.1, sum(score for score, _, _ in top_scores) / max(1, len(terms) * 3)))
    limitations = [] if any(score > 0 for score, _, _ in scored) else [f"source was weakly aligned with query; title: {title}"]
    return SourceCompressionContract(
        summary=summary,
        key_excerpts=excerpts,
        relevance=round(relevance, 4),
        limitations=limitations,
    )


def _heuristic_chunk_context(
    *,
    document_title: str,
    source: str,
    metadata: dict[str, Any],
    document_excerpt: str,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> ChunkContextContract:
    scalar_metadata = _scalar_metadata(metadata, limit=8)
    location_bits = [
        f"document '{document_title}'",
        f"source '{source}'",
        f"chunk {chunk_index + 1} of {max(1, total_chunks)}",
    ]
    for key in ("section_path", "section_heading", "page_number", "page_count", "file_type"):
        value = scalar_metadata.get(key)
        if value not in {None, ""}:
            location_bits.append(f"{key} {value}")

    terms = _context_key_terms(" ".join([document_title, source, document_excerpt[:1000], chunk_text]))
    topic_terms = ", ".join(terms[:8]) if terms else "local evidence and provenance"
    context = (
        f"This chunk comes from {', '.join(location_bits)}. "
        f"It provides local evidence about {topic_terms} and should be retrieved when a query needs "
        f"this document's specific context rather than a generic background summary."
    )
    return ChunkContextContract(
        context=_limit_words(context, max_words=96),
        key_terms=terms[:12],
        provenance_hint="; ".join(location_bits),
        confidence=0.74,
    )


def _normalize_chunk_context_contract(contract: ChunkContextContract) -> ChunkContextContract:
    context = _limit_words(contract.context, max_words=110)
    key_terms = [term for term in (_clean_text(term) for term in contract.key_terms) if term][:12]
    return contract.model_copy(
        update={
            "context": context,
            "key_terms": key_terms,
            "provenance_hint": _trim_text(contract.provenance_hint, 220),
            "confidence": max(0.0, min(1.0, float(contract.confidence or 0.0))),
        }
    )


def _scalar_metadata(metadata: dict[str, Any], *, limit: int) -> dict[str, object]:
    scalar: dict[str, object] = {}
    for key, value in sorted(metadata.items()):
        if isinstance(value, (str, int, float, bool)):
            scalar[key] = value
        if len(scalar) >= limit:
            break
    return scalar


def _context_key_terms(text: str) -> list[str]:
    counts = Counter(
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if len(token) > 2
        and token.lower()
        not in {
            "about",
            "and",
            "are",
            "for",
            "from",
            "how",
            "into",
            "not",
            "the",
            "this",
            "that",
            "with",
        }
    )
    return [token for token, _count in counts.most_common(16)]


def _limit_words(text: str, *, max_words: int) -> str:
    words = _clean_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,.;:") + "..."


def _hashed_dense_vector(text: str, dimensions: int) -> list[float]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    if not tokens:
        return [0.0] * dimensions
    values = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, 32, 4):
            bucket = int.from_bytes(digest[offset : offset + 4], "big") % dimensions
            weight = 1.0 if offset == 0 else 0.35
            values[bucket] += weight
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        return values
    return [value / norm for value in values]


def _extract_chat_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("OpenAI-compatible response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenAI-compatible response did not include message content.")
    return content


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _trim_text(text: str, max_chars: int) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
