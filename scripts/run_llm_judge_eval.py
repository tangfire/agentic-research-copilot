from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_research_copilot.providers import build_model_provider  # noqa: E402
from agentic_research_copilot.settings import load_settings  # noqa: E402


class JudgeScores(BaseModel):
    research_depth: int = Field(ge=1, le=5)
    source_quality: int = Field(ge=1, le=5)
    analytical_rigor: int = Field(ge=1, le=5)
    structure: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    reasoning: str
    weaknesses: list[str] = Field(default_factory=list)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an Open Deep Research-style LLM-as-judge evaluation for a saved report."
    )
    parser.add_argument("--report", default="examples/demo-report.md")
    parser.add_argument("--context", default="examples/demo-trace.json")
    parser.add_argument("--output", default="examples/llm-judge-report.json")
    args = parser.parse_args()

    report_path = Path(args.report)
    context_path = Path(args.context)
    output_path = Path(args.output)
    report_text = report_path.read_text(encoding="utf-8")
    context_text = context_path.read_text(encoding="utf-8")[:20000] if context_path.exists() else ""

    settings = load_settings()
    provider = build_model_provider(settings)
    if not hasattr(provider, "_chat_structured"):
        raise RuntimeError("LLM judge eval requires ARC_MODEL_PROVIDER=openai_compatible.")

    scores, usage = provider._chat_structured(  # type: ignore[attr-defined]
        system_prompt=(
            "You are evaluating a deep research report, following the style of Open Deep "
            "Research evaluators. Score the report from 1 to 5 on research depth, source "
            "quality, analytical rigor, structure, groundedness, and completeness. Penalize "
            "unsupported claims, missing citations, weak sources, and shallow analysis. "
            "Return valid JSON only."
        ),
        user_payload={
            "report": report_text[:30000],
            "context_or_trace": context_text,
        },
        schema=JudgeScores.model_json_schema(),
        response_model=JudgeScores,
    )
    payload = {
        "report": str(report_path),
        "context": str(context_path) if context_path.exists() else None,
        "model": usage.model,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "latency_ms": usage.latency_ms,
        },
        "scores": scores.model_dump(),
        "normalized": {
            "research_depth_score": round(scores.research_depth / 5, 4),
            "source_quality_score": round(scores.source_quality / 5, 4),
            "analytical_rigor_score": round(scores.analytical_rigor / 5, 4),
            "structure_score": round(scores.structure / 5, 4),
            "groundedness_score": round(scores.groundedness / 5, 4),
            "completeness_score": round(scores.completeness / 5, 4),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["normalized"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
