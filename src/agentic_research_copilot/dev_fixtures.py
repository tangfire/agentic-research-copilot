from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from agentic_research_copilot.provider_base import ModelUsage, ResearchModelProvider
from agentic_research_copilot.schemas import (
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
    ReporterSectionDraft,
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


class FixtureResearchModelProvider(ResearchModelProvider):
    """Small test/dev fixture provider.

    This is intentionally outside the provider factory path. Production code uses
    configured real providers; tests and fixture scripts can inject this class
    explicitly when they need stable local contracts.
    """

    name = "fixture"

    def __init__(self, *, embedding_dimensions: int = 256) -> None:
        self.embedding_dimensions = embedding_dimensions

    def clarify_request(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
    ) -> tuple[ClarificationContract, ModelUsage]:
        tokens = _tokens(request.topic)
        vague = len(tokens) <= 2 and any(token in {"ai", "rag", "agent", "agents"} for token in tokens)
        return (
            ClarificationContract(
                need_clarification=vague,
                question="Please clarify the scope, target repository, decision context, and team constraints." if vague else "",
                verification="" if vague else f"Research target is specific enough: {request.topic}",
                missing_dimensions=["scope", "team constraints"] if vague else [],
                confidence=0.78 if vague else 0.84,
            ),
            self._usage("clarifier"),
        )

    def draft_plan(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]:
        topic = request.topic.strip()
        plan = [
            PlanItem(
                id=f"item_{index}",
                question=question,
                purpose=purpose,
                search_query=query,
            )
            for index, (question, purpose, query) in enumerate(
                [
                    (
                        f"What core problem does {topic} solve?",
                        "Establish repository or technology fit before deeper adoption analysis.",
                        f"{topic} overview architecture",
                    ),
                    (
                        f"How does {topic} fit the current architecture and local constraints?",
                        "Check integration cost, runtime semantics, and team stack alignment.",
                        f"{topic} architecture FastAPI integration",
                    ),
                    (
                        f"What operational risks, rollout limits, and rollback paths matter for {topic}?",
                        "Cover production readiness and reversible pilot planning.",
                        f"{topic} deployment risk rollback",
                    ),
                ][: max(1, min(3, request.max_sections))],
                start=1,
            )
        ]
        return (
            PlannerContract(
                research_brief=f"Evaluate {topic} with repository signals, architecture fit, and operations risk.",
                plan=plan,
                assumptions=["Fixture provider is used only for tests and local regression fixtures."],
                success_criteria=["Each section has evidence.", "Team constraints are reflected in the conclusion."],
                revision_budget=revision_count,
                confidence=0.82,
            ),
            self._usage("planner"),
        )

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
        request_text = f"{request.topic} {' '.join(item.question for item in plan)}".lower()
        needs_stronger_evidence = any(
            token in request_text
            for token in ("github", "repository", "repo", "architecture", "code evidence")
        )
        min_evidence = 2 if needs_stronger_evidence else 1
        min_sources = 2 if needs_stronger_evidence else 1
        tool_calls = [
            SupervisorToolCall(
                name="ConductResearch",
                rationale=f"Collect evidence for {item.question}",
                plan_item_ids=[item.id],
                research_topic=item.question,
                mode="hybrid" if corpus_profile.has_private_docs else "external",
                selected_tools=["web_search", "vector_retrieval"] if corpus_profile.has_private_docs else ["web_search"],
                web_queries=[item.search_query or item.question],
                internal_queries=[
                    item.question,
                    f"{item.question} local team constraints",
                ] if corpus_profile.has_private_docs else [],
                min_evidence=min_evidence,
                min_sources=min_sources,
            )
            for item in plan
        ]
        tool_calls.append(
            SupervisorToolCall(
                name="ResearchComplete",
                rationale="Fixture supervisor finished bounded research delegation.",
            )
        )
        return (
            SupervisorDecisionContract(
                reflection="Fixture supervisor delegated all plan items once.",
                tool_calls=tool_calls,
                completion_criteria=["Evidence collected for each plan item."],
                max_concurrent_research_units=1,
                confidence=0.8,
            ),
            self._usage("supervisor"),
        )

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
        if evidence and not gaps:
            return (
                ResearcherToolDecisionContract(
                    action="ResearchComplete",
                    completion_reason="sufficiency_met",
                    rationale="Existing evidence satisfies the fixture sufficiency gate.",
                    confidence=0.88,
                ),
                self._usage("researcher"),
            )
        if evidence and "mcp_tool" in available_tools:
            tool = mcp_tools[0] if mcp_tools else None
            return (
                ResearcherToolDecisionContract(
                    action="mcp_tool",
                    query=item.search_query or item.question,
                    mcp_tool_name=tool.name if tool else None,
                    mcp_tool_args={"query": item.search_query or item.question} if tool else None,
                    rationale="Use source-of-truth tool evidence after broad web evidence.",
                    confidence=0.82,
                ),
                self._usage("researcher"),
            )
        if "web_search" in available_tools:
            return (
                ResearcherToolDecisionContract(
                    action="web_search",
                    query=item.search_query or item.question,
                    rationale="Collect broad public evidence first.",
                    reflection="Researcher should continue until evidence and source diversity are sufficient.",
                    confidence=0.8,
                ),
                self._usage("researcher"),
            )
        return (
            ResearcherToolDecisionContract(
                action="ResearchComplete",
                completion_reason="no_tools_available",
                rationale="No tools are available.",
                confidence=0.4,
            ),
            self._usage("researcher"),
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
        issues: list[str] = []
        if not evidence:
            issues.append("No evidence attached to the report.")
        if report.confidence < 0.4:
            issues.append("Confidence is too low.")
        if not report.sections:
            issues.append("Report has no sections.")
        if report.sections and any(not section.citations for section in report.sections):
            issues.append("Some sections have no citations.")
        if report.sections and not any(section.citations for section in report.sections):
            issues.append("Report sections exist but no citations were assembled.")
        missing_citation_count = sum(1 for section in report.sections if not section.citations)
        if missing_citation_count:
            issues.append(f"Sections missing citations: {missing_citation_count}")
        return (
            VerificationContract(
                issues=issues,
                critical_issues=[],
                should_revise=bool(issues) and revision_count < max_revisions,
                revision_reason="; ".join(issues) if issues else None,
                confidence=0.78,
                coverage_score=0.9 if not issues else 0.5,
            ),
            self._usage("verifier"),
        )

    def compose_report(
        self,
        topic: str,
        sections: Sequence[ReportSection],
        evidence: Sequence[EvidenceItem],
        confidence: float,
    ) -> tuple[ReporterContract, ModelUsage]:
        drafts = [
            ReporterSectionDraft(
                heading=section.heading,
                content=section.content,
                citation_indexes=list(range(1, min(len(evidence), max(1, len(section.citations))) + 1)),
            )
            for section in sections
        ]
        return (
            ReporterContract(
                title=f"Adoption Memo: {topic}",
                summary=f"Fixture synthesis for {topic} based on {len(evidence)} evidence items.",
                sections=drafts,
                highlights=[f"Covered {len(sections)} plan-backed sections."],
                recommendations=["Run a real-provider pass before using this as a decision memo."],
                source_index=[item.title for item in evidence],
                confidence=max(confidence, 0.72 if evidence else 0.35),
            ),
            self._usage("reporter"),
        )

    def compress_source(
        self,
        *,
        query: str,
        title: str,
        url: str | None,
        raw_content: str,
    ) -> tuple[SourceCompressionContract, ModelUsage]:
        sentences = _sentences(raw_content)
        terms = _tokens(query)
        selected = [sentence for sentence in sentences if any(term in sentence.lower() for term in terms[:6])]
        excerpts = selected[:3] or sentences[:2]
        return (
            SourceCompressionContract(
                summary=_trim(" ".join(excerpts) or raw_content, 700),
                key_excerpts=[_trim(sentence, 240) for sentence in excerpts[:3]],
                relevance=0.78 if excerpts else 0.4,
                limitations=[] if excerpts else ["No query-specific passage found."],
            ),
            self._usage("source-compressor"),
        )

    def contextualize_chunk(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, object],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> tuple[ChunkContextContract, ModelUsage]:
        terms = _tokens(f"{document_title} {chunk_text}")[:8]
        return (
            ChunkContextContract(
                context=(
                    f"This chunk comes from '{document_title}', source '{source}', "
                    f"chunk {chunk_index + 1} of {total_chunks}. It is relevant to {', '.join(terms[:6])}."
                ),
                key_terms=terms,
                provenance_hint=f"{document_title} chunk {chunk_index + 1}",
                confidence=0.74,
            ),
            self._usage("contextualizer"),
        )

    def extract_knowledge_graph(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, object],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        max_entities: int,
        max_relationships: int,
    ) -> tuple[KnowledgeGraphExtractionContract, ModelUsage]:
        names: list[str] = []
        for candidate in [*_entity_candidates(chunk_text), *_entity_candidates(document_title)]:
            if candidate not in names:
                names.append(candidate)
        names = names[:max_entities]
        entities = [
            KnowledgeGraphEntity(
                name=name,
                entity_type="concept",
                description=f"Fixture entity extracted from {document_title}.",
                confidence=0.7,
            )
            for name in names
        ]
        relationships = [
            KnowledgeGraphRelationship(
                source=left.name,
                target=right.name,
                relation_type="co_occurs_with",
                description=f"{left.name} and {right.name} appear in the same fixture chunk.",
                keywords=[left.name, right.name],
                weight=1.0,
                confidence=0.65,
            )
            for left, right in zip(entities, entities[1:], strict=False)
        ][:max_relationships]
        return (
            KnowledgeGraphExtractionContract(
                entities=entities,
                relationships=relationships,
                summary=f"Fixture graph extraction for {document_title}.",
                confidence=0.68,
            ),
            self._usage("graph-extractor"),
        )

    def extract_graph_query(
        self,
        *,
        query: str,
        max_local_keywords: int,
        max_global_keywords: int,
    ) -> tuple[KnowledgeGraphQueryContract, ModelUsage]:
        terms = _tokens(query)
        return (
            KnowledgeGraphQueryContract(
                local_keywords=terms[:max_local_keywords],
                global_keywords=terms[:max_global_keywords],
                confidence=0.7,
            ),
            self._usage("graph-query"),
        )

    def embed_text(self, text: str) -> tuple[list[float], ModelUsage]:
        return (_embedding(text, self.embedding_dimensions), self._usage("embedding"))

    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], ModelUsage]:
        return ([_embedding(text, self.embedding_dimensions) for text in texts], self._usage("embedding"))

    def _usage(self, model: str) -> ModelUsage:
        return ModelUsage(provider=self.name, model=f"fixture-{model}", prompt_tokens=32, completion_tokens=24)


def _embedding(text: str, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        values[index] += 1.0
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [round(value / norm, 6) for value in values]


def _tokens(text: str) -> list[str]:
    stopwords = {
        "about",
        "and",
        "are",
        "for",
        "from",
        "how",
        "into",
        "the",
        "this",
        "what",
        "with",
    }
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", str(text or ""))
        if token.lower() not in stopwords
    ]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]


def _entity_candidates(text: str) -> list[str]:
    normalized = re.sub(r"(?<=[.!?])\s+", "\n", str(text or ""))
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_+.-]*(?: [A-Z][A-Za-z0-9_+.-]*){0,3}\b", normalized)
    if not candidates:
        candidates = _tokens(text)
    seen: dict[str, str] = {}
    for candidate in candidates:
        cleaned = candidate.strip().strip(" -_/.,:;()[]{}")
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen[key] = cleaned
    return list(seen.values())


def _trim(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."
