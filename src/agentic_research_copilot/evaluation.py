from __future__ import annotations

from collections.abc import Iterable
import re

from .schemas import EvidenceItem, PlanItem, RAGEvaluation, ResearchReport, RetrievalRoute


MIN_CONTEXT_PRECISION = 0.22


class RAGEvaluator:
    """Automatic quality checks for retrieval and citation behavior.

    These are proxy metrics for local demos. They do not replace a labeled eval set,
    but they make unsupported sections, thin citations, and weak retrieval visible.
    """

    def evaluate(
        self,
        *,
        report: ResearchReport,
        plan: list[PlanItem],
        evidence: list[EvidenceItem],
        document_hits: list[EvidenceItem],
        retrieval_routes: list[RetrievalRoute] | None = None,
    ) -> RAGEvaluation:
        total_plan_items = max(1, len(plan))
        covered_plan_items = sum(1 for item in plan if item.evidence_count > 0)
        plan_coverage = covered_plan_items / total_plan_items
        retrieval_hit_rate = len([item for item in evidence if item.kind != "memory"]) / max(1, len(plan))
        private_retrieval_hit_rate = len(document_hits) / max(1, len(plan))
        route_lookup = {route.plan_item_id: route for route in retrieval_routes or []}
        sufficient_items = 0
        insufficient_plan_items: list[str] = []
        for item in plan:
            route = route_lookup.get(item.id)
            min_evidence = route.min_evidence if route is not None else 1
            if item.evidence_count >= min_evidence:
                sufficient_items += 1
            else:
                insufficient_plan_items.append(item.id)
        evidence_sufficiency = sufficient_items / total_plan_items
        routes_with_tools = sum(1 for route in route_lookup.values() if route.selected_tools)
        tool_selection_coverage = routes_with_tools / max(1, len(route_lookup) or len(plan))
        query_rewrite_count = sum(
            len(route.web_queries) + len(route.internal_queries)
            for route in route_lookup.values()
        )
        source_quality_score = _source_quality_score(evidence)

        section_count = max(1, len(report.sections))
        cited_sections = [section for section in report.sections if section.citations]
        unsupported_sections = [
            section.heading
            for section in report.sections
            if section.content.strip() and not section.citations
        ]
        citation_precision = len(cited_sections) / section_count
        citation_source_coverage = _citation_source_coverage(report, evidence)
        context_precision = _context_precision(report)
        context_recall = _context_recall(plan_coverage, evidence_sufficiency, citation_source_coverage)
        faithfulness_proxy = _faithfulness_proxy(report, context_precision, citation_precision)
        source_diversity = len(_source_set(evidence))

        notes: list[str] = []
        if plan_coverage < 0.8:
            notes.append("Plan coverage is weak; at least one research unit has no evidence.")
        if evidence_sufficiency < 0.8:
            notes.append("Evidence sufficiency is weak; at least one research unit missed its route threshold.")
        if tool_selection_coverage < 1.0 and retrieval_routes:
            notes.append("Tool selection is incomplete for at least one route.")
        if citation_precision < 1.0:
            notes.append("Some sections are not citation-backed.")
        if source_diversity < 2 and evidence:
            notes.append("Evidence source diversity is thin.")
        if source_quality_score < 0.55 and evidence:
            notes.append("Source quality is thin; prefer more authoritative or citation-rich sources.")
        if context_precision < MIN_CONTEXT_PRECISION and report.sections:
            notes.append("Context precision is weak; cited context has low overlap with report sections.")
        if faithfulness_proxy < 0.7 and report.sections:
            notes.append("Faithfulness proxy is weak; sections need stronger citation alignment.")
        if private_retrieval_hit_rate == 0 and document_hits:
            notes.append("Private retrieval returned hits but did not contribute to plan evidence.")

        passed = (
            plan_coverage >= 0.8
            and evidence_sufficiency >= 0.8
            and tool_selection_coverage >= 1.0
            and citation_precision >= 1.0
            and citation_source_coverage >= 0.65
            and source_quality_score >= 0.5
            and context_precision >= MIN_CONTEXT_PRECISION
            and context_recall >= 0.65
            and faithfulness_proxy >= 0.65
            and not unsupported_sections
        )

        return RAGEvaluation(
            plan_coverage=round(plan_coverage, 4),
            retrieval_hit_rate=round(min(1.0, retrieval_hit_rate), 4),
            private_retrieval_hit_rate=round(min(1.0, private_retrieval_hit_rate), 4),
            evidence_sufficiency=round(evidence_sufficiency, 4),
            tool_selection_coverage=round(tool_selection_coverage, 4),
            query_rewrite_count=query_rewrite_count,
            source_quality_score=round(source_quality_score, 4),
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4),
            faithfulness_proxy=round(faithfulness_proxy, 4),
            citation_precision=round(citation_precision, 4),
            citation_source_coverage=round(citation_source_coverage, 4),
            source_diversity=source_diversity,
            insufficient_plan_items=insufficient_plan_items,
            unsupported_sections=unsupported_sections,
            no_citation_section_count=len(unsupported_sections),
            passed=passed,
            notes=notes,
        )


def _citation_source_coverage(report: ResearchReport, evidence: list[EvidenceItem]) -> float:
    evidence_keys = {_evidence_key(item) for item in evidence}
    citation_keys = {
        _evidence_key(item)
        for section in report.sections
        for item in section.citations
    }
    if not citation_keys:
        return 0.0
    return len(citation_keys & evidence_keys) / len(citation_keys)


def _source_set(items: Iterable[EvidenceItem]) -> set[str]:
    return {
        item.source
        for item in items
        if item.source and item.source not in {"internal-note", "memory"}
    }


def _context_precision(report: ResearchReport) -> float:
    section_scores: list[float] = []
    for section in report.sections:
        section_tokens = _tokens(section.content)
        if not section_tokens or not section.citations:
            section_scores.append(0.0)
            continue
        citation_tokens = set()
        for citation in section.citations:
            citation_tokens.update(_tokens(" ".join([citation.title, citation.snippet or "", citation.content or ""])))
        if not citation_tokens:
            section_scores.append(0.0)
            continue
        section_scores.append(len(section_tokens & citation_tokens) / len(section_tokens))
    if not section_scores:
        return 0.0
    return sum(section_scores) / len(section_scores)


def _context_recall(
    plan_coverage: float,
    evidence_sufficiency: float,
    citation_source_coverage: float,
) -> float:
    return (plan_coverage * 0.35) + (evidence_sufficiency * 0.35) + (citation_source_coverage * 0.3)


def _faithfulness_proxy(
    report: ResearchReport,
    context_precision: float,
    citation_precision: float,
) -> float:
    supported_sections = [
        section
        for section in report.sections
        if section.content.strip() and section.citations
    ]
    support_ratio = len(supported_sections) / max(1, len(report.sections))
    return (support_ratio * 0.4) + (context_precision * 0.35) + (citation_precision * 0.25)


def _source_quality_score(items: list[EvidenceItem]) -> float:
    if not items:
        return 0.0
    scored = []
    for item in items:
        score = 0.45
        if item.url:
            score += 0.12
        if item.kind in {"paper", "document-chunk", "run-artifact", "web-summary"}:
            score += 0.12
        if item.source in {"arxiv", "pubmed", "openai_web", "anthropic_web", "perplexity", "run-ledger"}:
            score += 0.1
        if item.snippet or item.content:
            score += 0.08
        if item.source in {"youtube", "reddit", "duckduckgo"}:
            score -= 0.08
        scored.append(max(0.0, min(1.0, score)))
    return sum(scored) / len(scored)


def _evidence_key(item: EvidenceItem) -> str:
    return item.url or f"{item.kind}:{item.source}:{item.title}:{item.snippet or item.content or ''}"


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", value.lower())
        if len(token) > 2
    }
