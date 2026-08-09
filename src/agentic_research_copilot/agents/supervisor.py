from __future__ import annotations

from collections.abc import Sequence

from ..provider_base import ModelUsage, ResearchModelProvider
from ..providers import build_model_provider
from ..schemas import (
    CorpusProfile,
    MemoryRecord,
    PlanItem,
    ResearchRequest,
    RetrievalRoute,
    SupervisorDecisionContract,
    SupervisorToolCall,
)
from ..settings import AppSettings, load_settings


class SupervisorAgent:
    def __init__(
        self,
        model_provider: ResearchModelProvider | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.model_provider = model_provider or build_model_provider(self.settings)
        self.last_usage: ModelUsage | None = None

    def decide(
        self,
        request: ResearchRequest,
        *,
        research_brief: str,
        plan: Sequence[PlanItem],
        retrieval_routes: Sequence[RetrievalRoute],
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> SupervisorDecisionContract:
        contract, usage = self.model_provider.supervise_research(
            request,
            research_brief,
            plan,
            retrieval_routes,
            corpus_profile,
            memory_records,
            revision_count=revision_count,
            revision_notes=revision_notes,
        )
        self.last_usage = usage
        return self._normalize(
            contract,
            plan,
            retrieval_routes=retrieval_routes,
            request=request,
            corpus_profile=corpus_profile,
        )

    def _normalize(
        self,
        contract: SupervisorDecisionContract,
        plan: Sequence[PlanItem],
        *,
        retrieval_routes: Sequence[RetrievalRoute],
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
    ) -> SupervisorDecisionContract:
        plan_ids = {item.id for item in plan}
        route_lookup = {route.plan_item_id: route for route in retrieval_routes}
        normalized_calls: list[SupervisorToolCall] = []
        has_think = False
        conduct_ids: set[str] = set()

        for call in contract.tool_calls:
            if call.name == "think_tool":
                has_think = True
                normalized_calls.append(call)
                continue
            if call.name == "ConductResearch":
                valid_ids = [item_id for item_id in call.plan_item_ids if item_id in plan_ids]
                if not valid_ids:
                    continue
                conduct_ids.update(valid_ids)
                normalized_calls.append(
                    self._normalize_conduct_call(
                        call.model_copy(update={"plan_item_ids": valid_ids}),
                        route_lookup=route_lookup,
                        request=request,
                        corpus_profile=corpus_profile,
                    )
                )
                continue
            normalized_calls.append(call)

        if not has_think:
            normalized_calls.insert(
                0,
                SupervisorToolCall(
                    name="think_tool",
                    rationale="Ensure the ODR-style supervisor records a reflection before delegation.",
                    reflection=contract.reflection or "Reflect on the research plan before delegation.",
                ),
            )

        missing_plan_ids = [
            item.id
            for item in plan
            if item.requires_research and item.id not in conduct_ids
        ]
        for item in plan:
            if item.id not in missing_plan_ids:
                continue
            normalized_calls.append(
                SupervisorToolCall(
                    name="ConductResearch",
                    rationale="Fallback delegation added because required plan item was not delegated by the supervisor output.",
                    plan_item_ids=[item.id],
                    research_topic=f"{item.question} Purpose: {item.purpose}",
                    **self._fallback_route_fields(
                        route_lookup.get(item.id),
                        request=request,
                        corpus_profile=corpus_profile,
                    ),
                )
            )

        if not any(call.name == "ResearchComplete" for call in normalized_calls):
            normalized_calls.append(
                SupervisorToolCall(
                    name="ResearchComplete",
                    rationale="Completion is allowed after delegated research and verifier/evaluator quality gates pass.",
                    reflection="Finish only when citations and evidence sufficiency pass.",
                )
            )

        return contract.model_copy(
            update={
                "tool_calls": normalized_calls,
                "max_concurrent_research_units": max(
                    1,
                    min(contract.max_concurrent_research_units or len(plan) or 1, len(plan) or 1),
                ),
            }
        )

    def _normalize_conduct_call(
        self,
        call: SupervisorToolCall,
        *,
        route_lookup: dict[str, RetrievalRoute],
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
    ) -> SupervisorToolCall:
        fallback_route = route_lookup.get(call.plan_item_ids[0]) if call.plan_item_ids else None
        fallback_fields = self._fallback_route_fields(
            fallback_route,
            request=request,
            corpus_profile=corpus_profile,
        )
        mode = call.mode or fallback_fields["mode"]
        selected_tools = self._valid_tools(call.selected_tools) or fallback_fields["selected_tools"]
        if not corpus_profile.has_private_docs or not request.include_private_docs:
            selected_tools = [tool for tool in selected_tools if tool != "vector_retrieval"]
            if mode in {"internal", "hybrid"}:
                mode = "external" if "web_search" in selected_tools else "external"
        if "web_search" not in selected_tools and "vector_retrieval" not in selected_tools:
            selected_tools.insert(0, "web_search")
            mode = "external"
        if "vector_retrieval" in selected_tools and "web_search" in selected_tools:
            mode = "hybrid"
        elif "vector_retrieval" in selected_tools:
            mode = "internal"
        else:
            mode = "external"

        web_queries = self._clean_queries(call.web_queries) or fallback_fields["web_queries"]
        internal_queries = self._clean_queries(call.internal_queries) or fallback_fields["internal_queries"]
        if mode == "external":
            internal_queries = []
        if mode == "internal":
            web_queries = []
        if mode in {"external", "hybrid"} and not web_queries:
            web_queries = fallback_fields["web_queries"]
        if mode in {"internal", "hybrid"} and not internal_queries:
            internal_queries = fallback_fields["internal_queries"]

        return call.model_copy(
            update={
                "mode": mode,
                "selected_tools": selected_tools,
                "web_queries": web_queries,
                "internal_queries": internal_queries,
                "memory_query": call.memory_query or fallback_fields["memory_query"],
                "min_evidence": call.min_evidence or fallback_fields["min_evidence"],
                "min_sources": call.min_sources or fallback_fields["min_sources"],
                "sufficiency_criteria": call.sufficiency_criteria or fallback_fields["sufficiency_criteria"],
            }
        )

    def _fallback_route_fields(
        self,
        route: RetrievalRoute | None,
        *,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
    ) -> dict[str, object]:
        if route is None:
            selected_tools = ["web_search"]
            if request.use_memory:
                selected_tools.append("memory_recall")
            return {
                "mode": "external",
                "selected_tools": selected_tools,
                "web_queries": [],
                "internal_queries": [],
                "memory_query": request.topic if request.use_memory else None,
                "min_evidence": 1,
                "min_sources": 1,
                "sufficiency_criteria": ["preserve citations for report assembly"],
            }
        selected_tools = list(route.selected_tools)
        if not corpus_profile.has_private_docs or not request.include_private_docs:
            selected_tools = [tool for tool in selected_tools if tool != "vector_retrieval"]
        if not selected_tools:
            selected_tools = ["web_search"]
        return {
            "mode": route.mode,
            "selected_tools": selected_tools,
            "web_queries": list(route.web_queries),
            "internal_queries": list(route.internal_queries),
            "memory_query": route.memory_query,
            "min_evidence": route.min_evidence,
            "min_sources": route.min_sources,
            "sufficiency_criteria": list(route.sufficiency_criteria),
        }

    def _valid_tools(self, tools: Sequence[str]) -> list[str]:
        valid = {"web_search", "vector_retrieval", "memory_recall", "mcp_tool"}
        deduped: list[str] = []
        for tool in tools:
            if tool in valid and tool not in deduped:
                deduped.append(tool)
        return deduped

    def _clean_queries(self, queries: Sequence[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = " ".join(query.split()).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned
