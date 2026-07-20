from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_research_copilot.provider_validation import provider_runtime_report  # noqa: E402
from agentic_research_copilot.providers import (  # noqa: E402
    build_embedding_provider,
    build_model_provider,
)
from agentic_research_copilot.retrieval import RerankerConfig, build_reranker  # noqa: E402
from agentic_research_copilot.search import build_search_tool  # noqa: E402
from agentic_research_copilot.settings import load_settings  # noqa: E402


@dataclass(frozen=True)
class SmokeChunk:
    source: str
    chunk_index: int
    contextual_text: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Check real-provider readiness without printing secrets.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run minimal remote embedding, search, and rerank calls after config validation.",
    )
    args = parser.parse_args()

    settings = load_settings()
    report = provider_runtime_report(settings)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not report["ready"]:
        return 1

    if args.smoke:
        smoke = _run_smoke(settings)
        print(json.dumps({"smoke": smoke}, indent=2, ensure_ascii=False))

    return 0


def _run_smoke(settings) -> dict[str, object]:
    model_provider = build_model_provider(settings)
    embedding_provider = build_embedding_provider(settings, model_provider)
    embedding, embedding_usage = embedding_provider.embed_text("provider readiness check")

    search_tool = build_search_tool(settings)
    search_results = search_tool("LangGraph deep research citation workflow") if search_tool else []

    reranker = build_reranker(
        RerankerConfig(
            provider=settings.rerank_provider,
            model=settings.rerank_model,
            base_url=settings.rerank_base_url,
            api_key=settings.rerank_api_key,
            timeout_seconds=settings.rerank_timeout_seconds,
            candidate_limit=settings.rerank_candidate_limit,
            allow_fallback=False,
        )
    )
    reranked = reranker.rerank(
        "Which document discusses citation-backed research?",
        [
            (
                0.45,
                SmokeChunk(
                    source="doc-a",
                    chunk_index=0,
                    contextual_text="This note is about generic project setup.",
                ),
                {},
            ),
            (
                0.52,
                SmokeChunk(
                    source="doc-b",
                    chunk_index=0,
                    contextual_text="This note discusses citation-backed deep research workflows.",
                ),
                {},
            ),
        ],
        1,
    )
    return {
        "embedding": {
            "provider": getattr(embedding_provider, "name", "unknown"),
            "dimensions": len(embedding),
            "model": embedding_usage.model,
            "latency_ms": embedding_usage.latency_ms,
        },
        "search": {
            "provider": settings.search_provider,
            "result_count": len(search_results),
            "first_source": search_results[0].get("source") if search_results else None,
        },
        "rerank": {
            "reranker": getattr(reranker, "name", "unknown"),
            "result_count": len(reranked),
            "provider": reranked[0][2].get("rerank_provider") if reranked else None,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
