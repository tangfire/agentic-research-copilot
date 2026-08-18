from __future__ import annotations

import json
import re
import sys
from typing import Any


def main() -> int:
    payload = _read_payload()
    request = payload.get("input") if isinstance(payload.get("input"), dict) else payload
    content = str((request or {}).get("content") or "")
    workspace = request.get("workspace") if isinstance(request, dict) else {}

    normalized = " ".join(content.split())
    lower = normalized.lower()
    repo = _find_repo(normalized)
    hints: list[str] = []
    missing_inputs: list[str] = []

    if repo:
        hints.append(f"target repo detected: {repo}")
    else:
        missing_inputs.append("目标 repo / 项目")

    if _has_team_constraints(normalized, workspace):
        hints.append("team constraints detected in message or workspace")
    else:
        missing_inputs.append("团队约束")

    if _has_decision_context(normalized):
        hints.append("decision context detected")
    else:
        missing_inputs.append("决策问题")

    if "github" in lower or "repo" in lower:
        hints.append("treat repository metadata, releases, issues, and license as primary evidence")
    if "秋招" in normalized or "面试" in normalized or "demo" in lower:
        hints.append("keep the output demo-ready and interview-friendly")

    output = {
        "recognized_repo": repo or "",
        "missing_inputs": missing_inputs,
        "plan_hints": hints,
        "coverage_checklist": [
            "repo metadata",
            "maintenance health",
            "deployment fit",
            "security or compliance notes",
            "demo readiness",
        ],
        "summary": "Open source adoption preflight completed.",
    }
    sys.stdout.write(json.dumps(output, ensure_ascii=False))
    return 0


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _find_repo(content: str) -> str:
    match = re.search(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", content)
    if match:
        return match.group(1)
    match = re.search(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", content, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _has_team_constraints(content: str, workspace: Any) -> bool:
    lower = content.lower()
    workspace_text = ""
    if isinstance(workspace, dict):
        workspace_text = " ".join(
            str(workspace.get(key) or "")
            for key in ("team_context", "risk_policy", "deployment_constraints", "default_stack")
        )
    return any(keyword in content for keyword in ("团队", "约束", "部署", "回滚", "人数", "技术栈")) or "constraint" in lower or bool(workspace_text.strip())


def _has_decision_context(content: str) -> bool:
    lower = content.lower()
    return any(keyword in content for keyword in ("评估", "是否", "适合", "采用", "决定")) or "adoption" in lower or "evaluate" in lower


if __name__ == "__main__":
    raise SystemExit(main())
