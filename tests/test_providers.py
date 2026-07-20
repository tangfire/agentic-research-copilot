from agentic_research_copilot.providers import _extract_json_object


def test_extract_json_object_accepts_fenced_json():
    content = """```json
{"ok": true, "value": 1}
```"""

    assert _extract_json_object(content) == '{"ok": true, "value": 1}'


def test_extract_json_object_accepts_prefixed_json():
    content = 'Here is the JSON: {"ok": true, "value": 1}'

    assert _extract_json_object(content) == '{"ok": true, "value": 1}'
