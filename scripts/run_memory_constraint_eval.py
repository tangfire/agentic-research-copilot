from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TEAM_CONTEXT_DIR = Path("examples/adoption-lab/team-context")
OUTPUT_DIR = Path("examples/adoption-lab/outputs")
REPORT_PATH = OUTPUT_DIR / "adoption-memo.report.md"
SUMMARY_PATH = OUTPUT_DIR / "adoption-memo.summary.json"


def main() -> None:
    constraints = _team_constraints()
    report_text = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    covered = [_constraint_covered(constraint, report_text) for constraint in constraints]
    covered_count = sum(1 for item in covered if item["covered"])
    total = len(covered)
    constraint_coverage = covered_count / total if total else 1.0
    summary = _load_json(SUMMARY_PATH)
    memory_precision = 1.0 if total else 0.0
    memory_recall = constraint_coverage
    result = {
        "memory_precision": round(memory_precision, 4),
        "memory_recall": round(memory_recall, 4),
        "constraint_coverage": round(constraint_coverage, 4),
        "constraint_coverage_passed": constraint_coverage >= 0.6,
        "constraint_count": total,
        "covered_constraint_count": covered_count,
        "latest_run_id": (summary.get("headline") or {}).get("run_id"),
        "constraints": covered,
        "notes": [
            "This script is a lightweight labeled fixture for the adoption lab.",
            "memory_precision is 1.0 because team-context constraints are curated input, not auto-extracted memory.",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _team_constraints() -> list[str]:
    constraints: list[str] = []
    for path in sorted(TEAM_CONTEXT_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip(" -\t")
            if not stripped:
                continue
            lower = stripped.lower()
            if any(
                keyword in lower
                for keyword in [
                    "team",
                    "python",
                    "fastapi",
                    "docker",
                    "rollback",
                    "checkpoint",
                    "trace",
                    "evaluation",
                    "risk",
                    "constraint",
                    "single-node",
                    "one machine",
                ]
            ):
                constraints.append(stripped)
    return _dedupe(constraints)


def _constraint_covered(constraint: str, report_text: str) -> dict[str, Any]:
    constraint_tokens = _tokens(constraint)
    report_tokens = _tokens(report_text)
    overlap = sorted(constraint_tokens & report_tokens)
    score = len(overlap) / max(1, min(len(constraint_tokens), 8))
    direct = _normalize(constraint[:24]) in _normalize(report_text)
    return {
        "content": constraint,
        "covered": direct or score >= 0.35,
        "overlap_terms": overlap[:12],
        "score": round(min(1.0, score), 4),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", value.lower()) if len(token) > 1}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


if __name__ == "__main__":
    main()
