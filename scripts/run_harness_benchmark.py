from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from agentic_research_copilot.multi_agent_harness import score_task_against_run
from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.schemas import BenchmarkTask, ResearchRequest
from agentic_research_copilot.settings import AppSettings, load_settings, resolve_storage_path
from agentic_research_copilot.storage import SQLiteStore


DEFAULT_DATASET = Path("examples/research-desk-v4-benchmark.jsonl")
DEFAULT_REPORT = Path("examples/research-desk-v4-benchmark-report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Research Desk v4 harness benchmark.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--mode", choices=["deterministic", "real"], default="deterministic")
    parser.add_argument("--max-tasks", type=int, default=24)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    settings = _benchmark_settings(args.mode)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    if args.clean:
        _clear_benchmark_store(settings)
        if output_path.exists():
            output_path.unlink()

    cases = _load_cases(dataset_path)[: max(1, args.max_tasks)]
    copilot = ResearchCopilot(settings=settings)
    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            request = ResearchRequest(
                topic=case.topic,
                depth=case.depth,
                max_sections=3,
                max_revisions=0,
                metadata={
                    "benchmark_task_id": case.task_id,
                    "benchmark_scenario": case.scenario,
                    "expected_agent_ids": case.expected_agent_ids,
                    "expected_tools": case.expected_tools,
                    "hard_constraints": case.hard_constraints,
                },
            )
            run = copilot.run(request)
            summary = score_task_against_run(case, run)
            results.append(_summarize_case(case, run, summary))
    finally:
        copilot.close()

    summary = _aggregate_results(results)
    payload = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def _benchmark_settings(mode: str) -> AppSettings:
    base = load_settings()
    updates: dict[str, Any] = {
        "storage_path": ".arc/research-desk-v4-benchmark.db",
        "langgraph_checkpoint_path": ".arc/research-desk-v4-benchmark-checkpoints.sqlite",
        "qdrant_url": "",
        "qdrant_location": ".arc/research-desk-v4-benchmark-qdrant",
        "qdrant_prefer_local": True,
        "qdrant_collection": "arc_research_desk_v4_benchmark",
        "job_queue_backend": "in_process",
        "seed_reference_knowledge": True,
        "research_max_workers": 1,
        "research_max_iterations": 1 if mode == "real" else min(2, max(1, base.research_max_iterations)),
        "search_max_results": min(3, max(1, base.search_max_results)),
        "search_include_raw_content": False,
        "source_reader_enabled": False,
        "mcp_enabled": False,
    }
    if mode == "deterministic":
        updates.update(
            {
                "strict_providers": False,
                "model_provider": "deterministic",
                "embedding_provider": "deterministic",
                "search_provider": "none",
                "rerank_provider": "rule",
                "model_chat_model": "heuristic-chat",
                "embedding_model": "hashed-embedding",
            }
        )
    return base.model_copy(update=updates)


def _clear_benchmark_store(settings: AppSettings) -> None:
    store = SQLiteStore(resolve_storage_path(settings.storage_path))
    store.clear_documents()
    store.clear_runs()
    store.clear_jobs()


def _load_cases(path: Path) -> list[BenchmarkTask]:
    return [
        BenchmarkTask.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summarize_case(case: BenchmarkTask, run, summary) -> dict[str, Any]:
    evaluation = run.evaluation
    return {
        "task_id": case.task_id,
        "scenario": case.scenario,
        "topic": case.topic,
        "run_id": run.run_id,
        "status": run.status,
        "passed": summary.passed,
        "route_precision": summary.route_precision,
        "route_recall": summary.route_recall,
        "specialist_completion_rate": summary.specialist_completion_rate,
        "tool_success_rate": summary.tool_success_rate,
        "evidence_utilization": summary.evidence_utilization,
        "citation_precision": summary.citation_precision,
        "constraint_coverage": summary.constraint_coverage,
        "replay_fidelity": summary.replay_fidelity,
        "latency_ms": summary.latency_ms,
        "expected_agent_ids": case.expected_agent_ids,
        "selected_agent_ids": [assignment.agent_id for assignment in run.role_assignments],
        "expected_tools": case.expected_tools,
        "selected_tools": _selected_tools(run),
        "expected_terms": case.expected_terms,
        "matched_terms": _matched_terms(case.expected_terms, run),
        "source_count": run.report.source_count if run.report else 0,
        "notes": (evaluation.notes if evaluation else []) + summary.notes,
    }


def _selected_tools(run) -> list[str]:
    tools: list[str] = []
    for assignment in run.role_assignments:
        tools.extend(assignment.selected_tools)
    return sorted({tool for tool in tools if tool})


def _matched_terms(terms: list[str], run) -> list[str]:
    text = "\n".join(
        [
            run.request.topic,
            run.report.title if run.report else "",
            run.report.summary if run.report else "",
            "\n".join(section.heading + "\n" + section.content for section in (run.report.sections if run.report else [])),
        ]
    ).lower()
    return [term for term in terms if term.lower() in text]


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(results),
        "pass_rate": _avg(results, "passed"),
        "avg_route_precision": _avg(results, "route_precision"),
        "avg_route_recall": _avg(results, "route_recall"),
        "avg_specialist_completion_rate": _avg(results, "specialist_completion_rate"),
        "avg_tool_success_rate": _avg(results, "tool_success_rate"),
        "avg_evidence_utilization": _avg(results, "evidence_utilization"),
        "avg_citation_precision": _avg(results, "citation_precision"),
        "avg_constraint_coverage": _avg(results, "constraint_coverage"),
        "avg_replay_fidelity": _avg(results, "replay_fidelity"),
        "avg_source_count": _avg(results, "source_count"),
        "avg_latency_ms": _avg(results, "latency_ms"),
    }


def _avg(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    values = [float(result.get(key, 0.0) or 0.0) for result in results]
    return round(mean(values), 4)


if __name__ == "__main__":
    main()
