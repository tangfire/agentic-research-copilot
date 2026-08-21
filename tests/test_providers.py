import json

import pytest

from agentic_research_copilot.providers import (
    OpenAICompatibleResearchModelProvider,
    StructuredOutputError,
    _extract_json_object,
)
from agentic_research_copilot.provider_base import ModelUsage
from agentic_research_copilot.schemas import (
    CorpusProfile,
    EvidenceItem,
    ReportSection,
    ReporterContract,
    ResearchRequest,
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


def test_reporter_compacts_context_before_model_call(monkeypatch):
    provider = OpenAICompatibleResearchModelProvider(
        base_url="https://relay.example.test/v1",
        api_key="test-key",
        chat_model="test-model",
        embedding_model="test-embedding",
    )
    captured = {}

    def fake_chat_structured(*, system_prompt, user_payload, schema, response_model):
        captured["payload"] = user_payload
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
    assert len(payload["evidence_index"]) == 8
    assert len(payload["sections"]) == 4
    assert len(payload["topic"]) == 800
    assert len(payload["evidence_index"][0]["content"]) == 260
    assert len(payload["evidence_index"][0]["snippet"]) == 180
    assert len(payload["sections"][0]["content"]) == 700
    assert len(payload["sections"][0]["source_summary"]) == 2
    assert len(payload["sections"][0]["citation_titles"]) == 3
