from collections.abc import Sequence

import pytest

from agentic_research_copilot.agents.supervisor import SupervisorAgent
from agentic_research_copilot.dev_fixtures import FixtureResearchModelProvider
from agentic_research_copilot.provider_base import ModelUsage
from agentic_research_copilot.schemas import CorpusProfile, PlanItem, ResearchRequest, RetrievalRoute, SupervisorDecisionContract
from agentic_research_copilot.settings import AppSettings


class StructuredOutputFailureProvider(FixtureResearchModelProvider):
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
        raise ValueError("OpenAI-compatible provider returned invalid structured output for SupervisorDecisionContract")


class ProviderCrashProvider(FixtureResearchModelProvider):
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
        raise RuntimeError("provider boom")


def _sample_context() -> tuple[ResearchRequest, list[PlanItem], list[RetrievalRoute], CorpusProfile]:
    request = ResearchRequest(topic="Evaluate LangGraph adoption")
    plan = [
        PlanItem(
            id="item-1",
            question="What public evidence supports LangGraph adoption?",
            purpose="Gather initial architecture signals.",
            search_query="LangGraph adoption evidence",
        )
    ]
    routes = [
        RetrievalRoute(
            plan_item_id="item-1",
            mode="external",
            reason="public evidence is needed for freshness and context",
            selected_tools=["web_search"],
            web_queries=["LangGraph adoption evidence"],
            internal_queries=[],
            min_evidence=1,
            min_sources=1,
            sufficiency_criteria=["preserve citations for report assembly"],
        )
    ]
    corpus_profile = CorpusProfile(has_private_docs=False)
    return request, plan, routes, corpus_profile


def test_supervisor_fallback_is_only_used_for_structured_output_failures():
    request, plan, routes, corpus_profile = _sample_context()
    supervisor = SupervisorAgent(model_provider=StructuredOutputFailureProvider(), settings=AppSettings())

    contract = supervisor.decide(
        request,
        research_brief="Evaluate adoption fit.",
        plan=plan,
        retrieval_routes=routes,
        corpus_profile=corpus_profile,
    )

    assert contract.tool_calls[0].name == "think_tool"
    assert any(call.name == "ConductResearch" for call in contract.tool_calls)


def test_supervisor_propagates_non_structured_provider_errors():
    request, plan, routes, corpus_profile = _sample_context()
    supervisor = SupervisorAgent(model_provider=ProviderCrashProvider(), settings=AppSettings())

    with pytest.raises(RuntimeError, match="provider boom"):
        supervisor.decide(
            request,
            research_brief="Evaluate adoption fit.",
            plan=plan,
            retrieval_routes=routes,
            corpus_profile=corpus_profile,
        )
