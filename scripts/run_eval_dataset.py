from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.schemas import ResearchRequest
from agentic_research_copilot.settings import load_settings


DATASET_PATH = Path("examples/eval-dataset.jsonl")
REPORT_PATH = Path("examples/eval-report.json")


def main() -> None:
    settings = load_settings()
    copilot = ResearchCopilot(settings=settings)
    cases = _load_cases(DATASET_PATH)
    results = []
    try:
        for case in cases:
            request = ResearchRequest(
                topic=case["topic"],
                audience=case.get("audience", "technical reviewer"),
                depth=case.get("depth", "standard"),
                max_sections=4,
                max_revisions=1,
            )
            run = copilot.run(request)
            results.append(_score_case(case, run))
    finally:
        copilot.close()

    summary = {
        "case_count": len(results),
        "pass_rate": _avg(results, "passed"),
        "avg_relevance": _avg(results, "term_recall"),
        "avg_source_expectation": _avg(results, "expected_source_recall"),
        "avg_source_quality": _avg(results, "source_quality_score"),
        "avg_context_precision": _avg(results, "context_precision"),
        "avg_context_recall": _avg(results, "context_recall"),
        "avg_faithfulness": _avg(results, "faithfulness_proxy"),
        "avg_groundedness": _avg(results, "groundedness_proxy"),
        "avg_citation_precision": _avg(results, "citation_precision"),
    }
    payload = {"summary": summary, "results": results}
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _score_case(case: dict, run) -> dict:
    report_text = _report_text(run)
    source_index = "\n".join(run.report.source_index if run.report else [])
    expected_terms = case.get("expected_terms", [])
    expected_sources = case.get("expected_sources", [])
    matched_terms = [term for term in expected_terms if term.lower() in report_text.lower()]
    matched_sources = [source for source in expected_sources if source.lower() in source_index.lower()]
    evaluation = run.evaluation
    citation_precision = evaluation.citation_precision if evaluation else 0.0
    source_quality_score = evaluation.source_quality_score if evaluation else 0.0
    context_precision = evaluation.context_precision if evaluation else 0.0
    context_recall = evaluation.context_recall if evaluation else 0.0
    faithfulness_proxy = evaluation.faithfulness_proxy if evaluation else 0.0
    groundedness_proxy = 1.0 if run.report and all(section.citations for section in run.report.sections) else 0.0
    source_count = run.report.source_count if run.report else 0
    min_source_count = int(case.get("min_source_count", 1))
    term_recall = len(matched_terms) / max(1, len(expected_terms))
    expected_source_recall = len(matched_sources) / max(1, len(expected_sources))
    passed = (
        run.status == "completed"
        and term_recall >= 0.6
        and source_count >= min_source_count
        and citation_precision >= 0.9
        and context_recall >= 0.65
        and faithfulness_proxy >= 0.65
        and groundedness_proxy >= 1.0
    )
    return {
        "id": case["id"],
        "topic": case["topic"],
        "run_id": run.run_id,
        "status": run.status,
        "passed": passed,
        "term_recall": round(term_recall, 4),
        "matched_terms": matched_terms,
        "expected_source_recall": round(expected_source_recall, 4),
        "matched_sources": matched_sources,
        "source_count": source_count,
        "source_quality_score": source_quality_score,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness_proxy": faithfulness_proxy,
        "groundedness_proxy": groundedness_proxy,
        "citation_precision": citation_precision,
        "notes": evaluation.notes if evaluation else [],
    }


def _report_text(run) -> str:
    if run.report is None:
        return ""
    return "\n".join(
        [
            run.report.title,
            run.report.summary,
            *(section.heading + "\n" + section.content for section in run.report.sections),
        ]
    )


def _avg(results: list[dict], key: str) -> float:
    if not results:
        return 0.0
    values = [float(result.get(key, 0.0) or 0.0) for result in results]
    return round(mean(values), 4)


if __name__ == "__main__":
    main()
