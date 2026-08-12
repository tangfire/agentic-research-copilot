from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Sequence

from .provider_base import ModelUsage, ResearchModelProvider
from .schemas import (
    ChunkContextContract,
    ClarificationContract,
    CorpusProfile,
    EvidenceItem,
    KnowledgeGraphEntity,
    KnowledgeGraphExtractionContract,
    KnowledgeGraphQueryContract,
    KnowledgeGraphRelationship,
    MCPToolDescriptor,
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
GRAPH_ENTITY_STOPWORDS = {
    "A",
    "An",
    "And",
    "For",
    "Input",
    "Output",
    "Pilot Candidate",
    "The",
    "This",
    "Use",
    "When",
    "What",
    "Which",
    "Who",
    "Why",
    "How",
    "Should",
    "Could",
    "Would",
}


class DeterministicResearchModelProvider(ResearchModelProvider):
    name = "deterministic"

    def __init__(self, embedding_dimensions: int = 256) -> None:
        self.embedding_dimensions = max(32, embedding_dimensions)

    def clarify_request(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
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
        mcp_tools: Sequence[MCPToolDescriptor] = (),
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
            query = _researcher_follow_up_query(item, previous_queries, evidence, gaps)
            mcp_tool_name, mcp_tool_args = _select_mcp_query_tool(query, mcp_tools)
            action = ResearcherToolDecisionContract(
                action="mcp_tool",
                query=query,
                mcp_tool_name=mcp_tool_name,
                mcp_tool_args=mcp_tool_args,
                rationale="Use the configured external MCP tools as an additional grounding channel.",
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
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]:
        topic = request.topic.strip()
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
        if revision_count > 0 and revision_notes:
            brief_bits.append("Repair the previously flagged gaps: " + "; ".join(revision_notes[:3]))

        plan_items = _build_plan_items(request, topic, revision_count=revision_count)
        assumptions = [
            "The final answer must remain source-backed and inspectable.",
            "The supervisor may revise the plan if verification exposes citation or coverage gaps.",
        ]
        if corpus_profile.has_private_docs:
            assumptions.append("Internal grounding is available and should be preferred for project-specific facts.")

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
            if item.source and item.source != "internal-note"
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

    def extract_knowledge_graph(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, Any],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        max_entities: int,
        max_relationships: int,
    ) -> tuple[KnowledgeGraphExtractionContract, ModelUsage]:
        contract = _heuristic_knowledge_graph(
            document_title=document_title,
            source=source,
            chunk_text=chunk_text,
            max_entities=max_entities,
            max_relationships=max_relationships,
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-knowledge-graph-test-double",
            prompt_tokens=max(1, (len(document_excerpt) + len(chunk_text)) // 4),
            completion_tokens=max(16, (len(contract.entities) + len(contract.relationships)) * 12),
            latency_ms=1,
        )
        return contract, usage

    def extract_graph_query(
        self,
        *,
        query: str,
        max_local_keywords: int,
        max_global_keywords: int,
    ) -> tuple[KnowledgeGraphQueryContract, ModelUsage]:
        contract = _heuristic_graph_query(
            query,
            max_local_keywords=max_local_keywords,
            max_global_keywords=max_global_keywords,
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-graph-query-test-double",
            prompt_tokens=max(1, len(query) // 4),
            completion_tokens=max(8, len(contract.local_keywords) + len(contract.global_keywords)),
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
            f"What evidence, retrieval sources, or tool results support {base}?",
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
        return "agent orchestration, verification, and observability"
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


def _select_mcp_query_tool(
    query: str,
    mcp_tools: Sequence[MCPToolDescriptor],
) -> tuple[str | None, dict[str, object] | None]:
    query_capable = [
        tool
        for tool in mcp_tools
        if "query" in {*tool.required_args, *tool.optional_args} or tool.name.startswith("search")
    ]
    if not query_capable:
        return (mcp_tools[0].name, None) if len(mcp_tools) == 1 else (None, None)

    lower = query.lower()
    keyword_priority = [
        ("search_issues", ("issue", "bug", "failure", "risk")),
        ("search_code", ("code", "source", "implementation", "file", "readme", "architecture")),
        ("search_repositories", ("repo", "repository", "github", "project")),
    ]
    tool_lookup = {tool.name: tool for tool in query_capable}
    for tool_name, keywords in keyword_priority:
        if tool_name in tool_lookup and any(keyword in lower for keyword in keywords):
            return tool_name, {"query": query}

    return query_capable[0].name, {"query": query}


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


def _heuristic_knowledge_graph(
    *,
    document_title: str,
    source: str,
    chunk_text: str,
    max_entities: int,
    max_relationships: int,
) -> KnowledgeGraphExtractionContract:
    candidates: Counter[str] = Counter()
    labels: dict[str, str] = {}
    for phrase in re.findall(r"\b[A-Z][A-Za-z0-9+.#/-]*(?:\s+[A-Z][A-Za-z0-9+.#/-]*){0,3}\b", chunk_text):
        label = _clean_text(phrase)
        key = label.casefold()
        if len(label) < 3 or _is_low_value_graph_entity(label):
            continue
        candidates[key] += 3
        labels.setdefault(key, label)
    for token in _context_key_terms(" ".join([document_title, source, chunk_text])):
        key = token.casefold()
        candidates[key] += 1
        labels.setdefault(key, token)

    entities = [
        KnowledgeGraphEntity(
            name=labels[key],
            entity_type="test_term",
            description=f"Deterministic test entity found in {document_title}.",
            confidence=0.55,
        )
        for key, _count in candidates.most_common(max(1, max_entities))
    ]
    relationships: list[KnowledgeGraphRelationship] = []
    for left, right in zip(entities, entities[1:]):
        relationships.append(
            KnowledgeGraphRelationship(
                source=left.name,
                target=right.name,
                relation_type="co_occurs_with",
                description=f"{left.name} and {right.name} occur in the same deterministic test chunk.",
                keywords=["co-occurrence", "test relation"],
                weight=0.25,
                confidence=0.35,
            )
        )
        if len(relationships) >= max(1, max_relationships):
            break
    return KnowledgeGraphExtractionContract(
        entities=entities,
        relationships=relationships,
        summary="Deterministic graph extraction for tests and offline replay.",
        confidence=0.5 if entities else 0.0,
    )


def _heuristic_graph_query(
    query: str,
    *,
    max_local_keywords: int,
    max_global_keywords: int,
) -> KnowledgeGraphQueryContract:
    phrases = [
        _clean_text(value)
        for value in re.findall(r"\b[A-Z][A-Za-z0-9+.#/-]*(?:\s+[A-Z][A-Za-z0-9+.#/-]*){0,3}\b", query)
    ]
    terms = _context_key_terms(query)
    local_keywords = list(dict.fromkeys([*phrases, *terms]))[: max(1, max_local_keywords)]
    global_keywords = [
        term
        for term in terms
        if term.casefold() not in {value.casefold() for value in local_keywords[:3]}
    ][: max(1, max_global_keywords)]
    if not global_keywords:
        global_keywords = terms[: max(1, max_global_keywords)]
    return KnowledgeGraphQueryContract(
        local_keywords=local_keywords,
        global_keywords=global_keywords,
        confidence=0.5 if local_keywords or global_keywords else 0.0,
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


def _is_low_value_graph_entity(label: str) -> bool:
    normalized = _clean_text(label).strip(" .,:;!?")
    if normalized in GRAPH_ENTITY_STOPWORDS:
        return True
    words = normalized.split()
    return bool(words) and all(word in GRAPH_ENTITY_STOPWORDS for word in words)


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


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _trim_text(text: str, max_chars: int) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
