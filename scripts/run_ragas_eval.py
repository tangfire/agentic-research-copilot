from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_research_copilot.settings import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an optional Ragas evaluation over a saved demo report and trace artifact."
    )
    parser.add_argument("--report", default="examples/demo-report.md")
    parser.add_argument("--trace", default="examples/demo-trace.json")
    parser.add_argument("--reference", default="")
    parser.add_argument("--output", default="examples/ragas-report.json")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a skipped artifact instead of failing when the optional Ragas extra is not installed.",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    trace_path = Path(args.trace)
    output_path = Path(args.output)
    report_text = report_path.read_text(encoding="utf-8")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    question = trace.get("request", {}).get("topic") or _first_markdown_heading(report_text)
    contexts = _contexts_from_trace(trace)
    reference = _read_reference(args.reference)

    try:
        result = _run_ragas(question=question, response=report_text, contexts=contexts, reference=reference)
    except ModuleNotFoundError as exc:
        if not args.allow_missing:
            raise SystemExit(
                "Ragas dependencies are not installed. Run `pip install -e .[eval]` "
                "or rerun with `--allow-missing` to write a skipped artifact."
            ) from exc
        result = {
            "status": "skipped",
            "reason": "optional Ragas dependencies are not installed",
            "install": "pip install -e .[eval]",
        }

    payload = _json_safe(
        {
        "report": str(report_path),
        "trace": str(trace_path),
        "question": question,
        "context_count": len(contexts),
        "reference_configured": bool(reference),
        "result": result,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "context_count": len(contexts),
                "metric_keys": list((payload.get("result") or {}).keys()),
            },
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


def _run_ragas(*, question: str, response: str, contexts: list[str], reference: str | None) -> dict[str, Any]:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    import ragas.metrics as ragas_metrics

    settings = load_settings()
    row: dict[str, Any] = {
        "user_input": question,
        "response": response,
        "retrieved_contexts": contexts,
    }
    if reference:
        row["reference"] = reference

    dataset = Dataset.from_list([row])
    chat = ChatOpenAI(
        model=settings.model_chat_model,
        base_url=settings.model_base_url or None,
        api_key=settings.model_api_key or None,
        temperature=0,
        timeout=settings.model_timeout_seconds,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=(settings.embedding_base_url or settings.model_base_url or None),
        api_key=(settings.embedding_api_key or settings.model_api_key or None),
        dimensions=settings.embedding_dimensions,
    )
    metrics = _build_metrics(ragas_metrics, include_reference=bool(reference))
    score = evaluate(
        dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(chat),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        raise_exceptions=False,
    )
    return _score_to_dict(score)


def _build_metrics(ragas_metrics: Any, *, include_reference: bool) -> list[Any]:
    metric_names = [
        "Faithfulness",
        "LLMContextPrecisionWithoutReference",
        "ResponseRelevancy",
    ]
    if include_reference:
        metric_names.extend(["LLMContextRecall", "FactualCorrectness"])
    metrics: list[Any] = []
    for metric_name in metric_names:
        metric_cls = getattr(ragas_metrics, metric_name, None)
        if metric_cls is not None:
            metrics.append(metric_cls())
    if not metrics:
        raise RuntimeError("No compatible Ragas metrics were found in the installed version.")
    return metrics


def _score_to_dict(score: Any) -> dict[str, Any]:
    if hasattr(score, "to_pandas"):
        frame = score.to_pandas()
        records = frame.to_dict(orient="records")
        return records[0] if records else {}
    if hasattr(score, "scores"):
        scores = getattr(score, "scores")
        if isinstance(scores, list) and scores:
            return dict(scores[0])
        if isinstance(scores, dict):
            return scores
    try:
        return dict(score)
    except Exception:
        return {"raw": str(score)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _contexts_from_trace(trace: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    for evidence in trace.get("evidence_contexts", []):
        text = " ".join(
            str(evidence.get(key, "") or "")
            for key in ("title", "source", "url", "snippet", "content")
        ).strip()
        if text:
            contexts.append(text[:3500])
    for source in trace.get("report", {}).get("source_index", []):
        if source:
            contexts.append(str(source)[:1000])
    seen: set[str] = set()
    unique: list[str] = []
    for context in contexts:
        signature = context[:160]
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(context)
    return unique[:12]


def _read_reference(path: str) -> str | None:
    if not path:
        return None
    reference_path = Path(path)
    if not reference_path.exists():
        raise FileNotFoundError(path)
    return reference_path.read_text(encoding="utf-8")


def _first_markdown_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return "AI Research Copilot demo report"


if __name__ == "__main__":
    raise SystemExit(main())
