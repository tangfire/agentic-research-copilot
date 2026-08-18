from agentic_research_copilot.schemas import ResearcherToolDecisionContract, SupervisorDecisionContract


def test_supervisor_contract_treats_null_lists_as_empty_lists():
    contract = SupervisorDecisionContract.model_validate(
        {
            "reflection": "Need to research with fallback routes.",
            "tool_calls": [
                {
                    "name": "ConductResearch",
                    "rationale": "Use planned fallback queries.",
                    "plan_item_ids": None,
                    "selected_tools": None,
                    "web_queries": None,
                    "internal_queries": None,
                    "sufficiency_criteria": None,
                }
            ],
            "completion_criteria": None,
        }
    )

    assert contract.completion_criteria == []
    assert contract.tool_calls[0].plan_item_ids == []
    assert contract.tool_calls[0].selected_tools == []
    assert contract.tool_calls[0].web_queries == []
    assert contract.tool_calls[0].internal_queries == []
    assert contract.tool_calls[0].sufficiency_criteria == []


def test_researcher_contract_treats_null_reflection_as_empty_string():
    contract = ResearcherToolDecisionContract.model_validate(
        {
            "action": "web_search",
            "query": "LangGraph checkpoint docs",
            "reflection": None,
            "rationale": None,
        }
    )

    assert contract.reflection == ""
    assert contract.rationale == ""
