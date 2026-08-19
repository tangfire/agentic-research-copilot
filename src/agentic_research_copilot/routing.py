from __future__ import annotations

from .github_repository import canonical_repository_slug, parse_github_repository_hint
from .schemas import CorpusProfile, PlanItem, ResearchRequest, RetrievalRoute


class RetrievalCoordinator:
    """Decide how each plan item should mix external search and internal grounding."""

    PRIVATE_SIGNALS = (
        "private",
        "internal",
        "document",
        "documents",
        "notes",
        "note",
        "repo",
        "repository",
        "code",
        "project",
        "resume",
        "paper",
        "pdf",
        "wiki",
        "artifact",
        "history",
        "grounding",
    )
    PUBLIC_SIGNALS = (
        "latest",
        "current",
        "official",
        "benchmark",
        "research",
        "compare",
        "industry",
        "web",
        "external",
        "state of the art",
        "evaluation",
        "deployment",
        "architecture",
    )
    PRIVATE_FOCUS_ITEMS = {"data", "verification", "delivery", "repair", "revision"}

    def __init__(
        self,
        *,
        max_query_rewrites: int = 2,
        min_evidence_per_item: int = 2,
        min_source_diversity: int = 2,
        mcp_enabled: bool = True,
    ) -> None:
        self.max_query_rewrites = max(1, max_query_rewrites)
        self.min_evidence_per_item = max(1, min_evidence_per_item)
        self.min_source_diversity = max(1, min_source_diversity)
        self.mcp_enabled = mcp_enabled

    def build_routes(
        self,
        request: ResearchRequest,
        research_brief: str,
        plan: list[PlanItem],
        corpus_profile: CorpusProfile,
    ) -> list[RetrievalRoute]:
        return [
            self._route_for_item(request, research_brief, item, corpus_profile)
            for item in plan
        ]

    def _route_for_item(
        self,
        request: ResearchRequest,
        research_brief: str,
        item: PlanItem,
        corpus_profile: CorpusProfile,
    ) -> RetrievalRoute:
        repository_hint = parse_github_repository_hint(
            request.metadata,
            request.topic,
            research_brief,
            item.question,
            item.purpose,
            item.search_query or "",
        )
        blob = " ".join(
            part
            for part in [
                request.topic,
                research_brief,
                item.question,
                item.purpose,
                item.search_query or "",
            ]
            if part
        ).lower()
        has_private_corpus = request.include_private_docs and corpus_profile.has_private_docs
        has_public_pressure = self._contains_any(blob, self.PUBLIC_SIGNALS)
        has_private_pressure = self._contains_any(blob, self.PRIVATE_SIGNALS) or (
            item.id.split("-", 1)[-1] in self.PRIVATE_FOCUS_ITEMS
        )
        has_internal_topic_overlap = self._contains_any(blob, tuple(corpus_profile.keyword_signals[:12]))

        private_focus = has_private_corpus and (
            has_private_pressure or has_internal_topic_overlap or request.depth == "deep"
        )
        public_focus = has_public_pressure or not has_private_corpus

        if private_focus and public_focus:
            mode = "hybrid"
        elif private_focus:
            mode = "internal"
        else:
            mode = "external"

        reason_bits = []
        if public_focus:
            reason_bits.append("public evidence is needed for freshness and context")
        if private_focus:
            reason_bits.append(
                f"contextual grounding is useful for project-specific evidence ({corpus_profile.document_count} docs from "
                f"{corpus_profile.source_count} sources)"
            )
        if has_internal_topic_overlap and corpus_profile.keyword_signals:
            reason_bits.append(
                "internal corpus keywords overlap with the task: "
                + ", ".join(corpus_profile.keyword_signals[:4])
            )
        if not reason_bits:
            reason_bits.append("defaulting to public web evidence")
        if repository_hint is not None:
            reason_bits.append(
                f"explicit repository target detected ({canonical_repository_slug(repository_hint)})"
            )

        selected_tools = self._selected_tools(mode, has_repository_target=repository_hint is not None)
        rewrite_count = self._rewrite_count_for_depth(request.depth)
        web_queries = (
            self._dedupe_queries(
                [
                    item.search_query or item.question,
                    f"{item.question} {item.purpose}",
                    f"{request.topic} {item.purpose} evidence sources",
                ],
                limit=rewrite_count,
            )
            if mode in {"external", "hybrid"}
            else []
        )
        internal_queries = (
            self._dedupe_queries(
                [
                    f"{request.topic} {research_brief} {item.purpose} internal sources: {', '.join(corpus_profile.source_names[:4])}",
                    f"{item.question} {item.purpose} {' '.join(corpus_profile.keyword_signals[:6])}",
                    item.search_query or item.question,
                ],
                limit=rewrite_count,
            )
            if mode in {"internal", "hybrid"} and request.include_private_docs
            else []
        )
        min_evidence = self._min_evidence_for_depth(request.depth, mode)
        min_sources = 1 if mode == "internal" else min(self.min_source_diversity, min_evidence)
        sufficiency_criteria = [
            f"collect at least {min_evidence} evidence items for this plan item",
            f"use at least {min_sources} source group(s) when available",
            "preserve citations for report assembly",
        ]
        if mode == "hybrid":
            sufficiency_criteria.append("prefer evidence from both external search and contextual retrieval")

        return RetrievalRoute(
            plan_item_id=item.id,
            mode=mode,
            web_query=web_queries[0] if web_queries else None,
            internal_query=internal_queries[0] if internal_queries else None,
            reason="; ".join(reason_bits),
            selected_tools=selected_tools,
            web_queries=web_queries,
            internal_queries=internal_queries,
            min_evidence=min_evidence,
            min_sources=min_sources,
            sufficiency_criteria=sufficiency_criteria,
        )

    def _contains_any(self, blob: str, signals: tuple[str, ...]) -> bool:
        return any(signal in blob for signal in signals)

    def _selected_tools(
        self,
        mode: str,
        *,
        has_repository_target: bool,
    ) -> list[str]:
        tools: list[str] = []
        if mode in {"external", "hybrid"}:
            tools.append("web_search")
        if mode in {"internal", "hybrid"}:
            tools.append("vector_retrieval")
        if self.mcp_enabled and has_repository_target:
            tools.append("mcp_tool")
        return tools

    def _rewrite_count_for_depth(self, depth: str) -> int:
        if depth == "quick":
            return 1
        if depth == "deep":
            return min(self.max_query_rewrites, 3)
        return min(self.max_query_rewrites, 2)

    def _min_evidence_for_depth(self, depth: str, mode: str) -> int:
        if depth == "quick":
            return 1
        if depth == "deep":
            return max(3, self.min_evidence_per_item)
        if mode == "hybrid":
            return max(2, self.min_evidence_per_item)
        return self.min_evidence_per_item

    def _dedupe_queries(self, queries: list[str], *, limit: int) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = " ".join(query.split()).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
            if len(deduped) >= limit:
                break
        return deduped
