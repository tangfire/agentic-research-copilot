from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from agentic_research_copilot.mcp_tools import build_mcp_tool
from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.schemas import EvidenceItem, ResearchRequest, ResearchRun
from agentic_research_copilot.settings import (
    DASHSCOPE_COMPATIBLE_BASE_URL,
    GITHUB_MCP_READONLY_TOOLS,
    GITHUB_MCP_READONLY_URL,
    AppSettings,
    load_settings,
    resolve_storage_path,
)
from agentic_research_copilot.storage import SQLiteStore
from agentic_research_copilot.dev_fixtures import FixtureResearchModelProvider


LAB_DIR = Path("examples/adoption-lab")
TEAM_CONTEXT_DIR = LAB_DIR / "team-context"
OUTPUT_DIR = LAB_DIR / "outputs"
DEFAULT_TOPIC = (
    "Please simulate Northstar Platform, a 5-engineer Python/FastAPI platform team, "
    "and evaluate whether the GitHub repository langchain-ai/langgraph should be piloted "
    "as the workflow runtime for an internal open-source adoption memo and technical "
    "decision research desk. Combine local team constraints, public documentation or "
    "GitHub/Web evidence, whether graph structure is actually necessary, risks, a pilot "
    "plan, and a rollback plan."
)

EXPECTED_TERMS = [
    "LangGraph",
    "langchain-ai/langgraph",
    "StateGraph",
    "checkpoint",
    "trace",
    "revision",
    "FastAPI",
    "single-node",
    "citation",
    "evaluation",
    "pilot",
    "rollback",
]
EXPECTED_CONSTRAINTS = [
    "5-engineer",
    "Python 3.11",
    "FastAPI",
    "Docker Compose",
    "one machine",
    "private team constraints",
    "replayable runs",
    "graph-based orchestration only when",
]
EXPECTED_SOURCE_PATTERNS = [
    "github.com/langchain-ai/langgraph",
    "docs.langchain.com",
    "langgraph",
]
LEGACY_DEMO_ARTIFACTS = [
    Path("examples/resume-demo"),
    Path("examples/demo-report.md"),
    Path("examples/demo-trace.json"),
    Path(".arc/langgraph_resume_demo.sqlite"),
    Path(".arc/real_provider_demo.db"),
    Path(".arc/resume_demo_run.db"),
    Path(".arc/resume_demo_run.db-shm"),
    Path(".arc/resume_demo_run.db-wal"),
    Path(".arc/seed_resume_demo.py"),
]
LAB_STATE_ARTIFACTS = [
    Path(".arc/adoption_memo_lab.db"),
    Path(".arc/adoption_memo_lab.db-shm"),
    Path(".arc/adoption_memo_lab.db-wal"),
    Path(".arc/langgraph_adoption_memo_lab.sqlite"),
    Path(".arc/langgraph_adoption_memo_lab.sqlite-shm"),
    Path(".arc/langgraph_adoption_memo_lab.sqlite-wal"),
    Path(".arc/qdrant-adoption-lab"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a realistic open-source adoption memo experiment."
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--depth", default="standard", choices=["quick", "standard", "deep"])
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear previous lab outputs and old resume demo artifacts.",
    )
    parser.add_argument(
        "--mode",
        default="real",
        choices=["real", "fixture"],
        help="Use real configured providers, or explicit fixture injection for local regression checks.",
    )
    parser.add_argument("--use-mcp", action="store_true", help="Use configured MCP tools for this run.")
    parser.add_argument(
        "--full-real-indexing",
        action="store_true",
        help="Use the configured chat provider for document contextualization and graph extraction.",
    )
    parser.add_argument("--max-sections", type=int, default=3)
    args = parser.parse_args()

    if args.clean:
        _clean_lab_artifacts()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = _experiment_settings(use_mcp=args.use_mcp, mode=args.mode)
    _validate_experiment_settings(settings, mode=args.mode, use_mcp=args.use_mcp)
    _clear_experiment_store(settings)

    fixture_provider = (
        FixtureResearchModelProvider(embedding_dimensions=settings.embedding_dimensions)
        if args.mode == "fixture"
        else None
    )
    copilot = ResearchCopilot(
        settings=settings,
        model_provider=fixture_provider,
        embedding_provider=fixture_provider,
    )
    try:
        if not args.full_real_indexing:
            _use_budgeted_indexing(copilot, settings)
        _seed_team_context(copilot)
        request = ResearchRequest(
            topic=args.topic,
            depth=args.depth,
            include_private_docs=True,
            max_sections=max(1, args.max_sections),
            max_revisions=1,
        )
        run = copilot.run(request)
    finally:
        copilot.close()

    summary = _build_summary(run, settings, mode=args.mode)
    _write_outputs(run, settings, summary)
    print(json.dumps(summary["headline"], ensure_ascii=False))


def _experiment_settings(*, use_mcp: bool, mode: str) -> AppSettings:
    settings = load_settings()
    updates: dict[str, Any] = {
        "storage_path": ".arc/adoption_memo_lab.db",
        "langgraph_checkpoint_path": ".arc/langgraph_adoption_memo_lab.sqlite",
        "strict_providers": mode == "real",
        "qdrant_url": "",
        "qdrant_location": ".arc/qdrant-adoption-lab",
        "qdrant_prefer_local": True,
        "qdrant_collection": "arc_adoption_memo_lab",
        "job_queue_backend": "in_process",
        "mcp_enabled": use_mcp,
        "seed_reference_knowledge": False,
        "research_max_iterations": 1 if mode == "real" else min(2, max(1, settings.research_max_iterations)),
        "research_max_workers": 1,
        "rag_min_evidence_per_item": 1,
        "rag_min_source_diversity": 1,
        "model_timeout_seconds": min(90.0, max(45.0, settings.model_timeout_seconds)),
        "search_timeout_seconds": max(12.0, settings.search_timeout_seconds),
        "search_max_results": min(3, max(1, settings.search_max_results)),
        "search_include_raw_content": False,
        "source_reader_enabled": False,
        "mcp_timeout_seconds": min(15.0, max(5.0, settings.mcp_timeout_seconds)),
        "max_revisions": 1,
    }
    if mode == "real":
        updates.update(_stable_real_chat_provider(settings))
    if use_mcp and mode == "real":
        updates.update(
            {
                "mcp_enabled": True,
                "mcp_server_url": GITHUB_MCP_READONLY_URL,
                "mcp_tools": GITHUB_MCP_READONLY_TOOLS,
                "mcp_auth_required": True,
                "mcp_prompt": (
                    "Use GitHub MCP for source-of-truth repository evidence: README files, "
                    "code search, issues, pull requests, and releases. Use Tavily only for "
                    "broader web context outside GitHub."
                ),
            }
        )
    if mode == "fixture":
        updates.update(
            {
                "search_provider": "none",
                "rerank_provider": "rule",
                "mcp_enabled": False,
                "model_chat_model": "fixture-chat",
                "embedding_model": "fixture-embedding",
            }
        )
    return settings.model_copy(
        update={
            **updates,
        }
    )


def _stable_real_chat_provider(settings: AppSettings) -> dict[str, Any]:
    preferred_model = os.getenv("ARC_REAL_LAB_CHAT_MODEL", "qwen-plus")
    qwen_key = os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("QWEN_API_KEY", "").strip()
    current_url = settings.model_base_url.lower()
    current_model = settings.model_chat_model.lower()
    relay_like = "relay.novelcat" in current_url or "deepseek" in current_model
    if not qwen_key or not relay_like:
        return {}
    return {
        "model_base_url": DASHSCOPE_COMPATIBLE_BASE_URL,
        "model_api_key": qwen_key,
        "model_chat_model": preferred_model,
    }


def _validate_experiment_settings(settings: AppSettings, *, mode: str, use_mcp: bool) -> None:
    if mode != "real":
        return
    missing: list[str] = []
    if settings.model_provider != "openai_compatible" or not settings.model_base_url or not settings.model_api_key:
        missing.append("ARC_MODEL_PROVIDER=openai_compatible plus ARC_MODEL_BASE_URL/ARC_MODEL_API_KEY")
    if settings.embedding_provider != "openai_compatible" or not settings.embedding_api_key:
        missing.append("ARC_EMBEDDING_PROVIDER=openai_compatible plus ARC_EMBEDDING_API_KEY")
    if settings.search_provider == "none" or not settings.search_api_key:
        missing.append("ARC_SEARCH_PROVIDER plus ARC_SEARCH_API_KEY, for example Tavily")
    if use_mcp and not settings.mcp_auth_token:
        missing.append(
            "ARC_MCP_AUTH_TOKEN or GH_TOKEN/GITHUB_TOKEN/GITHUB_PERSONAL_ACCESS_TOKEN for GitHub MCP"
        )
    if missing:
        formatted = "\n".join(f"- {item}" for item in missing)
        raise RuntimeError(f"Real adoption memo experiment is not fully configured:\n{formatted}")
    if use_mcp:
        registry = build_mcp_tool(settings)
        if registry is None:
            raise RuntimeError("GitHub MCP is enabled but the MCP registry was not created.")
        descriptors = registry.describe_tools()
        if not descriptors:
            raise RuntimeError("GitHub MCP is enabled but no allowlisted tools were loaded.")


def _clear_experiment_store(settings: AppSettings) -> None:
    store = SQLiteStore(resolve_storage_path(settings.storage_path))
    store.clear_documents()
    store.clear_runs()
    store.clear_jobs()


def _seed_team_context(copilot: ResearchCopilot) -> None:
    for path in sorted(TEAM_CONTEXT_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        title = _title_from_markdown(content) or path.stem.replace("-", " ").title()
        copilot.add_document(
            title=title,
            source=str(path).replace("\\", "/"),
            content=content,
            metadata={
                "kind": "team_constraint",
                "scenario": "open_source_adoption_memo",
                "document_id": f"adoption-lab:{path.stem}",
            },
        )


def _use_budgeted_indexing(copilot: ResearchCopilot, settings: AppSettings) -> None:
    """Keep the experiment bounded while still using real search/model generation.

    Document contextualization and graph extraction call the chat provider once per
    chunk. In strict real-provider mode that can dominate a small demo run. This
    lab keeps embeddings/search/research decisions/reporting real by default, but
    uses fixture indexing helpers so the local context pack remains repeatable
    and fast.
    """
    helper = FixtureResearchModelProvider(
        embedding_dimensions=settings.embedding_dimensions
    )
    copilot.documents.contextualizer_provider = helper
    copilot.documents.graph_provider = helper


def _build_summary(run: ResearchRun, settings: AppSettings, *, mode: str = "real") -> dict[str, Any]:
    report_text = _report_text(run)
    source_blob = "\n".join(
        [
            *(run.report.source_index if run.report else []),
            *[item.url or item.source for item in run.evidence],
        ]
    ).lower()
    expected_term_hits = _matched(EXPECTED_TERMS, report_text)
    constraint_hits = _matched(EXPECTED_CONSTRAINTS, report_text)
    source_hits = _matched(EXPECTED_SOURCE_PATTERNS, source_blob)
    route_counts = Counter(route.mode for route in run.retrieval_routes)
    tool_counts = Counter(
        tool for route in run.retrieval_routes for tool in route.selected_tools
    )
    evidence_channels = Counter(
        str(item.metadata.get("source_channel", item.kind)) for item in run.evidence
    )
    graph_signal_hits = [
        item for item in run.document_hits if item.metadata.get("graph_augmented") is True
    ]
    graph_enabled_hits = [
        item for item in run.document_hits if item.metadata.get("graph_augmented_retrieval") is True
    ]
    model_events = [event for event in run.trace if event.model]
    handoff_events = [event for event in run.trace if event.kind == "handoff"]
    evaluation = run.evaluation
    labeled = {
        "expected_term_recall": _ratio(len(expected_term_hits), len(EXPECTED_TERMS)),
        "constraint_coverage": _ratio(len(constraint_hits), len(EXPECTED_CONSTRAINTS)),
        "constraint_coverage_passed": _ratio(len(constraint_hits), len(EXPECTED_CONSTRAINTS)) >= 0.6,
        "expected_source_recall": _ratio(len(source_hits), len(EXPECTED_SOURCE_PATTERNS)),
        "matched_terms": expected_term_hits,
        "matched_constraints": constraint_hits,
        "matched_source_patterns": source_hits,
    }
    headline = {
        "run_id": run.run_id,
        "status": run.status,
        "evaluation_passed": evaluation.passed if evaluation else False,
        "source_count": run.report.source_count if run.report else 0,
        "context_recall": evaluation.context_recall if evaluation else 0.0,
        "faithfulness_proxy": evaluation.faithfulness_proxy if evaluation else 0.0,
        "citation_precision": evaluation.citation_precision if evaluation else 0.0,
        "expected_term_recall": labeled["expected_term_recall"],
        "constraint_coverage": labeled["constraint_coverage"],
        "constraint_coverage_passed": labeled["constraint_coverage_passed"],
        "graph_signal_hits": len(graph_signal_hits),
    }
    return {
        "headline": headline,
        "labeled_expectations": labeled,
        "route_counts": dict(route_counts),
        "tool_counts": dict(tool_counts),
        "evidence_channels": dict(evidence_channels),
        "graph": {
            "document_hits": len(run.document_hits),
            "graph_enabled_hits": len(graph_enabled_hits),
            "graph_signal_hits": len(graph_signal_hits),
            "matched_entities_sample": _metadata_values(graph_signal_hits, "graph_matched_entities"),
            "matched_relationships_sample": _metadata_values(graph_signal_hits, "graph_matched_relationships"),
        },
        "trace": {
            "event_count": len(run.trace),
            "checkpoint_count": len(run.checkpoints),
            "handoff_count": len(handoff_events),
            "model_event_count": len(model_events),
            "actors": sorted({event.actor for event in run.trace}),
        },
        "providers": {
            "model_provider": settings.model_provider,
            "chat_model": settings.model_chat_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "search_provider": settings.search_provider,
            "mcp_enabled": settings.mcp_enabled,
            "mcp_auth_token_configured": bool(settings.mcp_auth_token),
            "mcp_tools": settings.mcp_tools if settings.mcp_enabled else [],
            "rerank_provider": settings.rerank_provider,
            "qdrant": "local-path",
            "experiment_mode": mode,
        },
        "evaluation": run.evaluation.model_dump(mode="json") if run.evaluation else None,
        "findings": [
            "Real report synthesis must use compact section drafts and evidence indexes; unbounded prompts make timeout failures likely.",
            "Budgeted local indexing keeps the team context pack fast while preserving real model/search/report execution.",
            "GitHub MCP is only counted as enabled when an auth token is configured and GitHub MCP tools are actually allowlisted.",
        ],
    }


def _write_outputs(run: ResearchRun, settings: AppSettings, summary: dict[str, Any]) -> None:
    (OUTPUT_DIR / "adoption-memo.report.md").write_text(_render_report(run), encoding="utf-8")
    (OUTPUT_DIR / "adoption-memo.run.json").write_text(
        run.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "adoption-memo.trace.json").write_text(
        json.dumps([event.model_dump(mode="json") for event in run.trace], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "adoption-memo.evaluation.json").write_text(
        json.dumps(
            run.evaluation.model_dump(mode="json") if run.evaluation else {},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "adoption-memo.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "adoption-memo.analysis.md").write_text(
        _render_analysis(run, settings, summary),
        encoding="utf-8",
    )


def _render_report(run: ResearchRun) -> str:
    report = run.report
    lines = [
        "# Adoption Memo Report",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Status: `{run.status}`",
        f"- Topic: {run.request.topic}",
        f"- Source count: {report.source_count if report else 0}",
        f"- Revision count: {run.revision_count}",
        "",
    ]
    if run.evaluation is not None:
        lines.extend(
            [
                "## Metrics",
                "",
                f"- Evaluation passed: `{run.evaluation.passed}`",
                f"- Context recall: {run.evaluation.context_recall}",
                f"- Citation precision: {run.evaluation.citation_precision}",
                f"- Faithfulness proxy: {run.evaluation.faithfulness_proxy}",
                f"- Source diversity: {run.evaluation.source_diversity}",
                "",
            ]
        )
    if report is None:
        lines.append("No report was generated.")
        return "\n".join(lines)

    lines.extend([f"## {report.title}", "", report.summary, ""])
    for section in report.sections:
        lines.extend([f"### {section.heading}", "", section.content, ""])
        if section.citations:
            lines.append("Citations:")
            for citation in section.citations:
                suffix = f" - {citation.url}" if citation.url else ""
                lines.append(f"- {citation.title} ({citation.source}){suffix}")
            lines.append("")
    lines.extend(["## Source Index", ""])
    for source in report.source_index:
        lines.append(f"- {source}")
    lines.append("")
    return "\n".join(lines)


def _render_analysis(run: ResearchRun, settings: AppSettings, summary: dict[str, Any]) -> str:
    headline = summary["headline"]
    graph = summary["graph"]
    labeled = summary["labeled_expectations"]
    eval_data = summary.get("evaluation") or {}
    lines = [
        "# Adoption Memo Experiment Analysis",
        "",
        "## Simulated User Input",
        "",
        run.request.topic,
        "",
        "## Headline Metrics",
        "",
        f"- Status: `{headline['status']}`",
        f"- Evaluation passed: `{headline['evaluation_passed']}`",
        f"- Source count: {headline['source_count']}",
        f"- Context recall: {headline['context_recall']}",
        f"- Citation precision: {headline['citation_precision']}",
        f"- Faithfulness proxy: {headline['faithfulness_proxy']}",
        f"- Expected term recall: {headline['expected_term_recall']}",
        f"- Team constraint coverage: {headline['constraint_coverage']}",
        f"- Constraint coverage passed: `{headline['constraint_coverage_passed']}`",
        f"- Expected source recall: {labeled['expected_source_recall']}",
        "",
        "## Routing And Trace",
        "",
        f"- Route counts: `{summary['route_counts']}`",
        f"- Tool counts: `{summary['tool_counts']}`",
        f"- Evidence channels: `{summary['evidence_channels']}`",
        f"- Trace events: {summary['trace']['event_count']}",
        f"- Checkpoints: {summary['trace']['checkpoint_count']}",
        f"- Handoffs: {summary['trace']['handoff_count']}",
        f"- Actors: `{summary['trace']['actors']}`",
        "",
        "## Graph Retrieval Check",
        "",
        f"- Document hits: {graph['document_hits']}",
        f"- Graph-enabled document hits: {graph['graph_enabled_hits']}",
        f"- Graph signal hits: {graph['graph_signal_hits']}",
        f"- Matched graph entities sample: `{graph['matched_entities_sample']}`",
        f"- Matched graph relationships sample: `{graph['matched_relationships_sample']}`",
        "",
        "Interpretation: graph design is justified only when the retrieved team context contains workflow stages, dependencies, quality gates, or revision paths. If graph_signal_hits is 0, the system still used graph-enabled indexing, but this run did not prove that graph signal improved retrieval.",
        "",
        "## Matched Expectations",
        "",
        f"- Terms: `{labeled['matched_terms']}`",
        f"- Constraints: `{labeled['matched_constraints']}`",
        f"- Source patterns: `{labeled['matched_source_patterns']}`",
        "",
        "## Evaluation Notes",
        "",
    ]
    notes = eval_data.get("notes", [])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No evaluator notes.")
    lines.extend(
        [
            "",
        "## Product Findings",
        "",
        "1. The strongest real use case is a repeatable adoption memo, not a generic chatbot. The local team context pack removes the need to paste the same constraints into every prompt.",
        "2. The current runtime already has a real graph: planner, research supervisor, parallel research, reporter, verifier/evaluator, revision, and finalize. In this scenario the graph is conceptually appropriate because evidence sufficiency and citation gates can change the path.",
        "3. GitHub MCP is a separate source-of-truth evidence channel. In real MCP mode the lab now forces the GitHub read-only endpoint and fails fast when auth is missing, so web-only evidence cannot be mistaken for MCP evidence.",
        "4. The next product surface should be a first-class adoption memo preset: repo, decision question, team context pack, generated report, trace, and metrics in one bundle.",
        "",
        "## Issues Found And Fixed",
        "",
            "1. Real provider timeouts could abort the run during long reporter/verifier calls. The reporter input is now compacted and the real lab is the default run mode.",
        "2. Budgeted indexing graph extraction was over-collecting structure words such as `The` and `Input`. A small stopword filter makes graph signal more trustworthy while keeping the real research/report path on real providers.",
        "3. MCP configuration used to inherit stale local workbench tools. Real MCP lab runs now force the GitHub read-only endpoint and GitHub tool allowlist.",
        "",
        "## Provider Snapshot",
        "",
            f"- Chat: `{settings.model_provider}` / `{settings.model_chat_model}`",
            f"- Embedding: `{settings.embedding_provider}` / `{settings.embedding_model}`",
            f"- Search: `{settings.search_provider}`",
            f"- MCP enabled for this run: `{settings.mcp_enabled}`",
            f"- MCP auth token configured: `{bool(settings.mcp_auth_token)}`",
            f"- Rerank: `{settings.rerank_provider}`",
        ]
    )
    return "\n".join(lines)


def _clean_lab_artifacts() -> None:
    root = Path.cwd().resolve()
    for path in [OUTPUT_DIR, *LEGACY_DEMO_ARTIFACTS, *LAB_STATE_ARTIFACTS]:
        resolved = (root / path).resolve()
        if not _is_relative_to(resolved, root):
            raise RuntimeError(f"Refusing to clean outside workspace: {resolved}")
        if not resolved.exists():
            continue
        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
        except PermissionError as exc:
            print(
                json.dumps(
                    {
                        "warning": "skip_locked_artifact",
                        "path": str(resolved),
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                )
            )
            continue
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / ".gitkeep").touch()


def _title_from_markdown(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return ""


def _report_text(run: ResearchRun) -> str:
    if run.report is None:
        return ""
    return "\n".join(
        [
            run.report.title,
            run.report.summary,
            *(section.heading + "\n" + section.content for section in run.report.sections),
        ]
    )


def _matched(expected: list[str], blob: str) -> list[str]:
    lower_blob = blob.lower()
    return [item for item in expected if item.lower() in lower_blob]


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / max(1, denominator), 4)


def _metadata_values(items: list[EvidenceItem], key: str) -> list[Any]:
    values: list[Any] = []
    for item in items[:4]:
        value = item.metadata.get(key)
        if value:
            values.append(value)
    return values


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
