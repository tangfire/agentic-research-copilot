import json

import pytest

from agentic_research_copilot.providers import (
    OpenAICompatibleResearchModelProvider,
    StructuredOutputError,
    _extract_json_object,
    _compact_planner_request,
)
from agentic_research_copilot.provider_base import ModelUsage
from agentic_research_copilot.schemas import (
    CorpusProfile,
    EvidenceItem,
    PlanItem,
    PlannerContract,
    ReportSection,
    ReporterContract,
    ResearchReport,
    ResearchRequest,
    RetrievalRoute,
    SupervisorDecisionContract,
    VerificationContract,
)


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class FakeClient:
    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, path, json):
        self.calls.append(json)
        return FakeResponse(self.bodies.pop(0))


def _valid_planner_body():
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "research_brief": "研究技术采用适配性。",
                            "plan": [],
                            "assumptions": [],
                            "success_criteria": [],
                            "revision_budget": 0,
                            "confidence": 0.7,
                        },
                        ensure_ascii=False,
                    )
                },
            }
        ],
        "usage": {},
    }


def test_extract_json_object_accepts_fenced_json():
    content = """```json
{"ok": true, "value": 1}
```"""

    assert _extract_json_object(content) == '{"ok": true, "value": 1}'


def test_extract_json_object_accepts_prefixed_json():
    content = 'Here is the JSON: {"ok": true, "value": 1}'

    assert _extract_json_object(content) == '{"ok": true, "value": 1}'


def test_structured_provider_repairs_empty_response(monkeypatch):
    provider = OpenAICompatibleResearchModelProvider(
        base_url="https://relay.example.test/v1",
        api_key="test-key",
        chat_model="test-model",
        embedding_model="test-embedding",
    )
    fake_client = FakeClient(
        [
            {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
            _valid_planner_body(),
        ]
    )
    monkeypatch.setattr(provider, "_client", lambda: fake_client)

    contract, _ = provider.draft_plan(
        ResearchRequest(topic="评估一个技术方案"),
        CorpusProfile(),
    )

    assert contract.research_brief == "研究技术采用适配性。"
    assert len(fake_client.calls) == 2
    assert fake_client.calls[1]["temperature"] == 0.0
    assert fake_client.calls[1]["response_format"] == {"type": "json_object"}


def test_structured_provider_accepts_reasoning_content_json(monkeypatch):
    provider = OpenAICompatibleResearchModelProvider(
        base_url="https://relay.example.test/v1",
        api_key="test-key",
        chat_model="test-model",
        embedding_model="test-embedding",
    )
    fake_client = FakeClient(
        [
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "",
                            "reasoning_content": "最终结果：" + json.dumps(
                                {
                                    "research_brief": "从 reasoning 内容提取。",
                                    "plan": [],
                                    "assumptions": [],
                                    "success_criteria": [],
                                    "revision_budget": 0,
                                    "confidence": 0.6,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
                "usage": {},
            }
        ]
    )
    monkeypatch.setattr(provider, "_client", lambda: fake_client)

    contract, _ = provider.draft_plan(
        ResearchRequest(topic="评估一个技术方案"),
        CorpusProfile(),
    )

    assert contract.research_brief == "从 reasoning 内容提取。"
    assert len(fake_client.calls) == 1


def test_structured_provider_exposes_truncation_diagnostics(monkeypatch):
    provider = OpenAICompatibleResearchModelProvider(
        base_url="https://relay.example.test/v1",
        api_key="test-key",
        chat_model="test-model",
        embedding_model="test-embedding",
    )
    truncated_body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": '{"research_brief":"截断'},
            }
        ],
        "usage": {"completion_tokens": 4096},
    }
    fake_client = FakeClient([truncated_body, truncated_body, truncated_body])
    monkeypatch.setattr(provider, "_client", lambda: fake_client)

    with pytest.raises(StructuredOutputError) as error_info:
        provider.draft_plan(
            ResearchRequest(topic="评估一个技术方案"),
            CorpusProfile(),
        )

    error = error_info.value
    assert error.diagnostics["finish_reason"] == "length"
    assert "truncated" in error.reason
    assert len(fake_client.calls) == 3


def test_compact_planner_request_keeps_only_planning_context():
    request = ResearchRequest(
        topic="评估 langchain-ai/langgraph 是否适合 5 人 Python/FastAPI 团队。" + "x" * 2000,
        metadata={
            "github_repository_slug": "langchain-ai/langgraph",
            "hard_constraints": ["必须支持失败恢复和回滚", "必须单机 Docker Compose 部署"],
            "request_context": "context" * 1000,
            "debug_dump": "drop" * 1000,
            "skill_instructions_excerpt": "drop" * 1000,
        },
    )

    compact = _compact_planner_request(request)

    assert len(compact["topic"]) <= 700
    assert compact["metadata"]["github_repository_slug"] == "langchain-ai/langgraph"
    assert compact["metadata"]["hard_constraints"] == ["必须支持失败恢复和回滚", "必须单机 Docker Compose 部署"]
    assert "request_context_summary" in compact["metadata"]
    assert "debug_dump" not in compact["metadata"]
    assert "skill_instructions_excerpt" not in compact["metadata"]


def test_planner_compacts_agent_session_context_before_model_call(monkeypatch):
    provider = OpenAICompatibleResearchModelProvider(
        base_url="https://relay.example.test/v1",
        api_key="test-key",
        chat_model="test-model",
        embedding_model="test-embedding",
    )
    captured = {}

    def fake_chat_structured(*, system_prompt, user_payload, schema, response_model):
        captured["payload"] = user_payload
        captured["system_prompt"] = system_prompt
        return PlannerContract(
            research_brief="评估 LangGraph 对小团队的采用适配性。",
            plan=[
                PlanItem(
                    id=f"item-{index}",
                    question=f"问题 {index}",
                    purpose="验证一个独立采用维度",
                    search_query="langchain-ai langgraph adoption risk",
                )
                for index in range(3)
            ],
            assumptions=["GitHub repo 是主要证据源"],
            success_criteria=["报告覆盖团队硬约束"],
            revision_budget=0,
            confidence=0.7,
        ), ModelUsage(provider="fake", model="fake")

    monkeypatch.setattr(provider, "_chat_structured", fake_chat_structured)
    request = ResearchRequest(
        topic="Conversation research request:\n评估 langchain-ai/langgraph 是否适合 5 人 Python/FastAPI 团队。" + "x" * 12000,
        metadata={
            "source": "agent_session",
            "session_id": "session-1",
            "workspace_id": "default-workspace",
            "workspace_context": "5 人 Python/FastAPI 团队，单机 Docker Compose 部署。" * 60,
            "github_repository": {"owner": "langchain-ai", "repo": "langgraph"},
            "github_repository_slug": "langchain-ai/langgraph",
            "hard_constraints": ["必须支持失败恢复和回滚" for _ in range(20)],
            "user_turns": ["评估 langchain-ai/langgraph" + "y" * 2000 for _ in range(5)],
            "request_context": "完整会话上下文" * 2000,
            "skill_instructions_excerpt": "不应该塞给 Planner 的完整 skill 文档" * 1000,
            "large_blob": "z" * 20000,
        },
        max_sections=8,
    )

    provider.draft_plan(
        request,
        CorpusProfile(has_private_docs=True, source_names=["source" * 40 for _ in range(10)]),
    )

    payload = captured["payload"]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert len(serialized) < 8000
    assert len(payload["request"]["topic"]) <= 700
    assert payload["request"]["metadata"]["github_repository_slug"] == "langchain-ai/langgraph"
    assert payload["request"]["max_sections"] == 4
    assert "large_blob" not in payload["request"]["metadata"]
    assert "skill_instructions_excerpt" not in payload["request"]["metadata"]
    assert "output_limits" in payload
    assert "compact valid JSON" in captured["system_prompt"]


def test_supervisor_compacts_context_before_model_call(monkeypatch):
    provider = OpenAICompatibleResearchModelProvider(
        base_url="https://relay.example.test/v1",
        api_key="test-key",
        chat_model="test-model",
        embedding_model="test-embedding",
    )
    captured = {}

    def fake_chat_structured(*, system_prompt, user_payload, schema, response_model):
        captured["payload"] = user_payload
        captured["system_prompt"] = system_prompt
        return SupervisorDecisionContract(
            reflection="route compactly",
            tool_calls=[],
            completion_criteria=["citations are preserved"],
            max_concurrent_research_units=1,
            confidence=0.7,
        ), ModelUsage(provider="fake", model="fake")

    monkeypatch.setattr(provider, "_chat_structured", fake_chat_structured)
    request = ResearchRequest(
        topic="Conversation research request:\n" + "x" * 20000,
        metadata={
            "source": "agent_session",
            "workspace_id": "default-workspace",
            "github_repository": {"owner": "langchain-ai", "repo": "langgraph"},
            "github_repository_slug": "langchain-ai/langgraph",
            "hard_constraints": ["必须支持失败恢复和回滚" for _ in range(20)],
            "user_turns": ["评估 langchain-ai/langgraph" + "y" * 2000],
            "large_blob": "z" * 20000,
        },
    )
    plan = [
        PlanItem(
            id=f"item-{index}",
            question="q" * 1000,
            purpose="p" * 1000,
            search_query="s" * 1000,
        )
        for index in range(8)
    ]
    routes = [
        RetrievalRoute(
            plan_item_id=f"item-{index}",
            mode="external",
            reason="r" * 800,
            selected_tools=["web_search", "mcp_tool"],
            web_queries=["web" * 200, "extra" * 200, "drop" * 200],
            internal_queries=["internal" * 200],
            min_evidence=2,
            min_sources=1,
            sufficiency_criteria=["criteria" * 80 for _ in range(6)],
        )
        for index in range(8)
    ]

    provider.supervise_research(
        request,
        "brief" * 1000,
        plan,
        routes,
        CorpusProfile(has_private_docs=True, source_names=["source" * 40 for _ in range(10)]),
    )

    payload = captured["payload"]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert len(serialized) < 14000
    assert len(payload["request"]["topic"]) <= 1200
    assert payload["request"]["metadata"]["github_repository_slug"] == "langchain-ai/langgraph"
    assert len(payload["plan"]) == 5
    assert len(payload["plan"][0]["question"]) <= 320
    assert len(payload["retrieval_routes"]) == 5
    assert len(payload["retrieval_routes"][0]["web_queries"]) == 2
    assert "large_blob" not in payload["request"]["metadata"]
    assert "This is a routing decision" in captured["system_prompt"]


def test_reporter_compacts_context_before_model_call(monkeypatch):
    provider = OpenAICompatibleResearchModelProvider(
        base_url="https://relay.example.test/v1",
        api_key="test-key",
        chat_model="test-model",
        embedding_model="test-embedding",
    )
    captured = {}

    def fake_chat_structured(*, system_prompt, user_payload, schema, response_model, max_tokens=None):
        captured["payload"] = user_payload
        captured["max_tokens"] = max_tokens
        return ReporterContract(title="ok", summary="ok"), ModelUsage(provider="fake", model="fake")

    monkeypatch.setattr(provider, "_chat_structured", fake_chat_structured)
    evidence = [
        EvidenceItem(
            title="e" * 400,
            source="s" * 300,
            kind="web",
            url="https://example.com/" + "u" * 400,
            snippet="n" * 600,
            content="c" * 1200,
        )
        for _ in range(12)
    ]
    sections = [
        ReportSection(
            heading="h" * 300,
            content="x" * 1800,
            citations=evidence,
            evidence_count=12,
            source_summary=["source-1", "source-2", "source-3"],
        )
        for _ in range(6)
    ]

    provider.compose_report("topic" * 500, sections, evidence, 0.8)

    payload = captured["payload"]
    assert len(payload["evidence_index"]) == 6
    assert len(payload["sections"]) == 4
    assert len(payload["topic"]) == 800
    assert len(payload["evidence_index"][0]["content"]) == 260
    assert len(payload["evidence_index"][0]["snippet"]) == 180
    assert len(payload["sections"][0]["content"]) == 420
    assert len(payload["sections"][0]["source_summary"]) == 2
    assert len(payload["sections"][0]["citation_titles"]) == 3
    assert captured["max_tokens"] == provider.max_tokens


def test_verifier_compacts_context_before_model_call(monkeypatch):
    provider = OpenAICompatibleResearchModelProvider(
        base_url="https://relay.example.test/v1",
        api_key="test-key",
        chat_model="test-model",
        embedding_model="test-embedding",
    )
    captured = {}

    def fake_chat_structured(*, system_prompt, user_payload, schema, response_model):
        captured["payload"] = user_payload
        captured["system_prompt"] = system_prompt
        return VerificationContract(
            issues=[],
            critical_issues=[],
            should_revise=False,
            confidence=0.8,
            coverage_score=0.75,
        ), ModelUsage(provider="fake", model="fake")

    monkeypatch.setattr(provider, "_chat_structured", fake_chat_structured)
    evidence = [
        EvidenceItem(
            title="e" * 500,
            source="s" * 300,
            kind="web",
            url="https://example.com/" + "u" * 400,
            snippet="n" * 800,
            content="c" * 2000,
        )
        for _ in range(14)
    ]
    report = ResearchReport(
        title="t" * 300,
        summary="summary" * 400,
        sections=[
            ReportSection(
                heading="h" * 300,
                content="x" * 2400,
                citations=evidence,
                evidence_count=14,
                source_summary=["source-1", "source-2", "source-3", "source-4"],
            )
            for _ in range(7)
        ],
        citations=evidence,
        confidence=0.7,
        highlights=["highlight" * 100 for _ in range(6)],
        recommendations=["recommendation" * 100 for _ in range(6)],
        source_index=["source-index" * 100 for _ in range(12)],
        source_count=6,
    )
    plan = [PlanItem(id=f"item-{index}", question="q" * 600, purpose="p" * 400) for index in range(8)]

    provider.assess_report(report, evidence, plan)

    payload = captured["payload"]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert len(serialized) < 8000
    assert len(payload["report"]["summary"]) <= 240
    assert len(payload["report"]["sections"]) == 3
    assert "citations" not in payload["report"]["sections"][0]
    assert len(payload["report"]["sections"][0]["content_preview"]) <= 180
    assert len(payload["evidence"]) == 4
    assert "content" not in payload["evidence"][0]
    assert len(payload["plan"]) == 5
    assert "quality gate" in captured["system_prompt"]
