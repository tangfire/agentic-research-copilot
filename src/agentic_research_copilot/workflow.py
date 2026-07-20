from __future__ import annotations

from collections.abc import Iterable

from .schemas import EvidenceItem, PlanItem, ResearchNote, RetrievalRoute, SearchQuery


class ResearchWorkflow:
    """Workflow helpers adapted from open_deep_research's plan-research-compress loop.

    This module keeps the research flow close to the upstream planning + research +
    compression pattern while staying lightweight for local learning and demo use.
    """

    def build_queries(
        self,
        plan: list[PlanItem],
        retrieval_routes: list[RetrievalRoute] | None = None,
        *,
        revision_count: int = 0,
    ) -> list[SearchQuery]:
        queries: list[SearchQuery] = []
        route_lookup = {route.plan_item_id: route for route in retrieval_routes or []}
        for item in plan:
            route = route_lookup.get(item.id)
            if route is not None:
                for index, query in enumerate(route.web_queries):
                    queries.append(
                        SearchQuery(
                            query=query,
                            intent=f"Web evidence for: {item.purpose}",
                            plan_item_id=item.id,
                            tool="web_search",
                            rewrite_index=index,
                            revision=revision_count,
                        )
                    )
                for index, query in enumerate(route.internal_queries):
                    queries.append(
                        SearchQuery(
                            query=query,
                            intent=f"Contextual retrieval for: {item.purpose}",
                            plan_item_id=item.id,
                            tool="vector_retrieval",
                            rewrite_index=index,
                            revision=revision_count,
                        )
                    )
                if route.memory_query:
                    queries.append(
                        SearchQuery(
                            query=route.memory_query,
                            intent=f"Memory recall for: {item.purpose}",
                            plan_item_id=item.id,
                            tool="memory_recall",
                            rewrite_index=0,
                            revision=revision_count,
                        )
                    )
                continue
            queries.append(
                SearchQuery(
                    query=item.search_query or item.question,
                    intent=f"Gather evidence for: {item.purpose}",
                    plan_item_id=item.id,
                    revision=revision_count,
                )
            )
        return queries

    def compress_findings(
        self,
        item: PlanItem,
        evidence: list[EvidenceItem],
        route: RetrievalRoute | None = None,
    ) -> ResearchNote:
        sufficiency_score, gaps = self._score_sufficiency(evidence, route)
        follow_up_queries = self._follow_up_queries(item, route, gaps)
        if not evidence:
            return ResearchNote(
                plan_item_id=item.id,
                question=item.question,
                finding="No useful evidence was found for this research unit.",
                confidence=0.15,
                sufficiency_score=sufficiency_score,
                gaps=gaps,
                follow_up_queries=follow_up_queries,
            )

        snippets = [
            evidence_item.snippet or evidence_item.content or evidence_item.title
            for evidence_item in evidence[:3]
        ]
        finding = " ".join(snippet.strip() for snippet in snippets if snippet.strip())
        if not finding:
            finding = f"{len(evidence)} evidence items were collected for this research unit."

        return ResearchNote(
            plan_item_id=item.id,
            question=item.question,
            finding=finding[:800],
            evidence_titles=[item.title for item in evidence[:5]],
            confidence=min(0.95, 0.45 + len(evidence) * 0.08),
            sufficiency_score=sufficiency_score,
            gaps=gaps,
            follow_up_queries=follow_up_queries,
        )

    def format_sources(self, evidence: Iterable[EvidenceItem]) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(evidence, start=1):
            key = item.url or f"{item.source}:{item.title}"
            if key in seen:
                continue
            seen.add(key)
            suffix = f" - {item.url}" if item.url else ""
            lines.append(f"[{index}] {item.title} ({item.source}){suffix}")
        return lines

    def _score_sufficiency(
        self,
        evidence: list[EvidenceItem],
        route: RetrievalRoute | None,
    ) -> tuple[float, list[str]]:
        if route is None:
            return (1.0 if evidence else 0.0), ([] if evidence else ["no evidence returned"])

        non_memory_sources = {
            item.source
            for item in evidence
            if item.source and item.kind != "memory" and item.source != "memory"
        }
        evidence_ratio = min(1.0, len(evidence) / max(1, route.min_evidence))
        source_ratio = min(1.0, len(non_memory_sources) / max(1, route.min_sources))
        score = round(evidence_ratio * 0.7 + source_ratio * 0.3, 4)

        gaps: list[str] = []
        if len(evidence) < route.min_evidence:
            gaps.append(f"needs {route.min_evidence - len(evidence)} more evidence item(s)")
        if len(non_memory_sources) < route.min_sources:
            gaps.append(f"needs {route.min_sources - len(non_memory_sources)} more source group(s)")
        if route.mode == "hybrid":
            kinds = {item.kind for item in evidence}
            if "web" not in kinds:
                gaps.append("missing external search evidence")
            if "document-chunk" not in kinds:
                gaps.append("missing contextual retrieval evidence")
        return score, gaps

    def _follow_up_queries(
        self,
        item: PlanItem,
        route: RetrievalRoute | None,
        gaps: list[str],
    ) -> list[str]:
        if not gaps:
            return []
        follow_ups = []
        if route is not None:
            follow_ups.extend(route.web_queries[1:])
            follow_ups.extend(route.internal_queries[1:])
        follow_ups.append(f"{item.question} missing evidence citations")
        deduped: list[str] = []
        seen: set[str] = set()
        for query in follow_ups:
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
        return deduped[:3]
