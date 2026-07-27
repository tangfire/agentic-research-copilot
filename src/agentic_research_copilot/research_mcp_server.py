from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import httpx


TEXT_EXTENSIONS = {
    ".md",
    ".py",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".rst",
}
IGNORED_DIRS = {
    ".arc",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
}
MAX_FILE_BYTES = 512_000
DEFAULT_SERVER_PORT = 8765
DEFAULT_API_BASE = "http://127.0.0.1:8010"
DEFAULT_RESULT_LIMIT = 5


def search_grounding_corpus(query: str, max_results: int = DEFAULT_RESULT_LIMIT) -> dict[str, Any]:
    """Search the running copilot's ingested grounding corpus."""
    api_base = _api_base()
    max_results = _clamp_limit(max_results)
    response = _get_json(
        f"{api_base}/v1/documents/search",
        params={"q": query, "limit": max_results},
        unavailable_hint="Start the FastAPI app and ingest documents before calling this MCP tool.",
    )
    if not response["available"]:
        return {"query": query, **response}

    payload = response["payload"]
    return {
        "query": query,
        "api_base": api_base,
        "available": True,
        "result_count": payload.get("result_count", 0),
        "corpus_profile": payload.get("corpus_profile", {}),
        "results": _compact_evidence_items(payload.get("results", []), max_results),
        "how_to_use": (
            "Use these hits as already-ingested grounding evidence. Retrieval metadata exposes "
            "parent-child retrieval, dense/BM25 fusion, graph augmentation, and rerank signals."
        ),
    }


def recall_project_memory(query: str, max_results: int = DEFAULT_RESULT_LIMIT) -> dict[str, Any]:
    """Recall session, summary, and canonical memory from the running copilot."""
    api_base = _api_base()
    max_results = _clamp_limit(max_results)
    response = _get_json(
        f"{api_base}/v1/memory/search",
        params={"q": query, "limit": max_results},
        unavailable_hint="Start the FastAPI app and create memory through a completed run or POST /v1/memory.",
    )
    if not response["available"]:
        return {"query": query, **response}

    payload = response["payload"]
    return {
        "query": query,
        "api_base": api_base,
        "available": True,
        "result_count": payload.get("result_count", 0),
        "governance": payload.get("governance", {}),
        "results": _compact_memory_items(payload.get("results", []), max_results),
        "how_to_use": (
            "Use memory for continuity, user/project preferences, and prior run summaries. Canonical "
            "memories with low confidence or conflicts should be treated as review candidates."
        ),
    }


def inspect_research_runs(query: str, max_results: int = DEFAULT_RESULT_LIMIT) -> dict[str, Any]:
    """Inspect recent research runs, trace events, citations, and evaluation signals."""
    api_base = _api_base()
    max_results = _clamp_limit(max_results)
    runs_response = _get_json(
        f"{api_base}/v1/research/runs",
        unavailable_hint="Start the FastAPI app and complete at least one research job first.",
    )
    if not runs_response["available"]:
        return {"query": query, **runs_response}

    runs = runs_response["payload"]
    if not isinstance(runs, list):
        runs = []
    summaries: list[dict[str, Any]] = []
    for run in _rank_runs(query, runs)[:max_results]:
        summary = _compact_run(run)
        run_id = run.get("run_id")
        if isinstance(run_id, str) and run_id:
            trace_response = _get_json(f"{api_base}/v1/research/runs/{run_id}/trace")
            evaluation_response = _get_json(f"{api_base}/v1/research/runs/{run_id}/evaluation")
            if trace_response["available"]:
                summary["trace_summary"] = _summarize_trace(trace_response["payload"])
            if evaluation_response["available"]:
                summary["evaluation"] = _compact_evaluation(evaluation_response["payload"])
        summaries.append(summary)

    return {
        "query": query,
        "api_base": api_base,
        "available": True,
        "run_count": len(runs),
        "result_count": len(summaries),
        "runs": summaries,
        "how_to_use": (
            "Use this for demo replay: it shows whether a run exercised planning, MCP, retrieval, "
            "memory, citation verification, and evaluator feedback."
        ),
    }


def check_demo_readiness(query: str) -> dict[str, Any]:
    """Check whether the local demo can exercise the important interview paths."""
    api_base = _api_base()
    runtime = inspect_runtime_config(query)
    docs = _get_json(f"{api_base}/v1/documents")
    memory = _get_json(f"{api_base}/v1/memory", params={"limit": 25})
    runs = _get_json(f"{api_base}/v1/research/runs")

    checks: list[dict[str, Any]] = []
    runtime_available = bool(runtime.get("available"))
    checks.append(_check("api_running", runtime_available, "FastAPI runtime is reachable."))
    if runtime_available:
        readiness = runtime.get("provider_readiness", {})
        tool = runtime.get("mcp_tool", {})
        retrieval = runtime.get("retrieval", {})
        checks.extend(
            [
                _check("real_chat_provider", _provider_ready(readiness, "model"), "Chat model is configured."),
                _check("real_search_provider", _provider_ready(readiness, "search"), "Search provider is configured."),
                _check(
                    "real_embedding_provider",
                    _provider_ready(readiness, "embedding"),
                    "Embedding provider is configured.",
                ),
                _check("reranker_ready", _provider_ready(readiness, "rerank"), "Reranker is configured."),
                _check("mcp_loaded", bool(tool.get("loaded")), "MCP registry loaded configured tools."),
                _check("hybrid_retrieval", bool(retrieval.get("keyword_backend")), "Hybrid dense/BM25 retrieval is visible."),
            ]
        )

    document_count = len(docs["payload"]) if docs["available"] and isinstance(docs["payload"], list) else 0
    memory_count = len(memory["payload"]) if memory["available"] and isinstance(memory["payload"], list) else 0
    run_count = len(runs["payload"]) if runs["available"] and isinstance(runs["payload"], list) else 0
    checks.extend(
        [
            _check("grounding_documents", document_count > 0, f"{document_count} document(s) available."),
            _check("memory_records", memory_count > 0, f"{memory_count} memory record(s) available."),
            _check("completed_runs", run_count > 0, f"{run_count} run(s) available for replay/evaluation."),
        ]
    )
    passed = sum(1 for item in checks if item["passed"])
    return {
        "query": query,
        "available": runtime_available,
        "score": round(passed / max(1, len(checks)), 3),
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "next_actions": _demo_next_actions(checks),
    }


def search_reference_corpus(query: str, max_results: int = 6) -> dict[str, Any]:
    """Search local project/reference files for architecture grounding."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return {"query": query, "roots": _root_labels(), "result_count": 0, "results": []}

    candidates: list[dict[str, Any]] = []
    file_budget = int(os.getenv("ARC_MCP_DEMO_MAX_FILES", "1600"))
    scanned = 0
    for root in _allowed_roots():
        for path in _iter_text_files(root):
            scanned += 1
            if scanned > file_budget:
                break
            text = _read_text(path)
            if not text:
                continue
            score = _score_document(query_tokens, path, text)
            if score <= 0:
                continue
            candidates.append(
                {
                    "path": str(path),
                    "root": str(root),
                    "score": round(score, 4),
                    "snippet": _snippet(text, query_tokens),
                }
            )
        if scanned > file_budget:
            break

    candidates.sort(key=lambda item: item["score"], reverse=True)
    results = candidates[: max(1, min(max_results, 12))]
    return {
        "query": query,
        "roots": _root_labels(),
        "scanned_files": scanned,
        "result_count": len(results),
        "results": results,
        "how_to_use": "Use this only for architecture/reference inspection, not as the primary research corpus.",
    }


def inspect_runtime_config(query: str) -> dict[str, Any]:
    """Inspect the running copilot's runtime config without exposing secrets."""
    api_base = _api_base()
    config_response = _get_json(
        f"{api_base}/v1/runtime/config",
        unavailable_hint="Start the FastAPI app before calling this MCP tool.",
    )
    if not config_response["available"]:
        return {"query": query, **config_response}
    config = config_response["payload"]

    tool_registry = config.get("tool_registry", [])
    return {
        "query": query,
        "api_base": api_base,
        "available": True,
        "product": config.get("product", {}),
        "orchestration": config.get("orchestration", {}),
        "provider_readiness": config.get("provider_readiness", {}),
        "open_deep_research_alignment": config.get("open_deep_research_alignment", {}),
        "mcp_tool": _first_tool(tool_registry, "mcp_tool"),
        "tools": [
            {
                "name": tool.get("name"),
                "provider": tool.get("provider"),
                "enabled": tool.get("enabled"),
                "loaded": tool.get("loaded"),
            }
            for tool in tool_registry
        ],
        "retrieval": config.get("retrieval", {}),
    }


def recommend_demo_questions(query: str) -> dict[str, Any]:
    """Return demo tasks that exercise web search, RAG, MCP, memory, eval, and replay."""
    questions = [
        {
            "title": "ODR alignment and MCP tool loop",
            "question": (
                "Compare Open Deep Research's supervisor/researcher/MCP tool loop with this AI Research "
                "Copilot implementation. Which parts are directly aligned and which parts are product-specific?"
            ),
            "uses": ["web_search", "mcp_tool", "memory_recall", "trace_replay"],
            "why": "Good for showing MCP as a configured tool channel plus run replay.",
        },
        {
            "title": "Agentic RAG boundary",
            "question": (
                "For a deep research assistant, why should RAG be a grounding/cache layer rather than the "
                "primary path? Use ODR-style search/read/verify workflow evidence."
            ),
            "uses": ["web_search", "vector_retrieval", "mcp_tool", "evaluation"],
            "why": "Good for explaining planning-search-reading versus plain top-k RAG.",
        },
        {
            "title": "Local grounding quality",
            "question": (
                "Evaluate how contextual retrieval prefixes, SQLite BM25, Qdrant dense search, "
                "LightRAG-inspired graph signal, and rerank improve local document grounding."
            ),
            "uses": ["vector_retrieval", "mcp_tool", "ragas_artifact", "source_index"],
            "why": "Good for showing advanced RAG engineering instead of a toy vector search.",
        },
        {
            "title": "Reader boundary and citation safety",
            "question": (
                "Assess whether provider raw_content reading, local PDF/page metadata, and citation-locked "
                "report synthesis are enough for a v1 research copilot demo."
            ),
            "uses": ["web_search", "document_reader", "mcp_tool", "llm_judge_artifact"],
            "why": "Good for honestly discussing ODR v1 boundaries without overselling OCR/browser automation.",
        },
    ]
    tokens = set(_tokenize(query))
    if tokens:
        filtered = [
            item
            for item in questions
            if tokens & set(_tokenize(" ".join([item["title"], item["question"], item["why"], " ".join(item["uses"])])))
        ]
        if filtered:
            questions = filtered
    return {
        "query": query,
        "question_count": len(questions),
        "questions": questions,
        "demo_setup": {
            "mcp_server_url": f"http://127.0.0.1:{_server_port()}/mcp",
            "default_tools": _default_tool_names(),
            "optional_tools": ["search_reference_corpus", "inspect_runtime_config", "recommend_demo_questions"],
        },
    }


def build_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "arc-research-workbench",
        instructions=(
            "Research workspace MCP server for AI Research Copilot. Use these tools to search ingested "
            "grounding documents, recall project memory, inspect run traces/evaluation, check demo readiness, "
            "and inspect local Open Deep Research/PraisonAI references."
        ),
        host=os.getenv("ARC_MCP_DEMO_HOST", "127.0.0.1"),
        port=_server_port(),
        streamable_http_path="/mcp",
        log_level=os.getenv("ARC_MCP_DEMO_LOG_LEVEL", "INFO"),
    )

    server.tool(
        name="search_grounding_corpus",
        description="Search ingested project documents through the running copilot's hybrid retrieval API.",
    )(search_grounding_corpus)
    server.tool(
        name="recall_project_memory",
        description="Recall session, summary, and canonical memory from the running copilot.",
    )(recall_project_memory)
    server.tool(
        name="inspect_research_runs",
        description="Inspect recent research runs, trace events, citations, and evaluation signals.",
    )(inspect_research_runs)
    server.tool(
        name="check_demo_readiness",
        description="Check whether providers, retrieval, MCP, memory, and replay data are ready for a demo.",
    )(check_demo_readiness)
    server.tool(
        name="search_reference_corpus",
        description="Search local project and reference files for ODR/PraisonAI/RAG/MCP architecture evidence.",
    )(search_reference_corpus)
    server.tool(
        name="inspect_runtime_config",
        description="Inspect the running copilot runtime config and provider/tool readiness.",
    )(inspect_runtime_config)
    server.tool(
        name="recommend_demo_questions",
        description="Recommend demo questions that exercise research, RAG, MCP, memory, eval, and replay.",
    )(recommend_demo_questions)
    return server


def main() -> None:
    build_server().run(transport="streamable-http")


def _default_tool_names() -> list[str]:
    return [
        "search_grounding_corpus",
        "recall_project_memory",
        "inspect_research_runs",
        "check_demo_readiness",
    ]


def _api_base() -> str:
    return os.getenv("ARC_MCP_DEMO_API_BASE", DEFAULT_API_BASE).rstrip("/")


def _get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    unavailable_hint: str = "Start the FastAPI app before calling this MCP tool.",
) -> dict[str, Any]:
    try:
        response = httpx.get(url, params=params, timeout=5)
        response.raise_for_status()
        return {"available": True, "payload": response.json()}
    except Exception as exc:
        return {
            "available": False,
            "api_base": _api_base(),
            "error": str(exc),
            "hint": unavailable_hint,
            "payload": None,
        }


def _compact_evidence_items(items: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    compacted = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        compacted.append(
            {
                "title": item.get("title"),
                "source": item.get("source"),
                "kind": item.get("kind"),
                "url": item.get("url"),
                "score": item.get("score"),
                "snippet": _trim_text(item.get("snippet") or item.get("content") or "", 700),
                "retrieval": {
                    key: metadata.get(key)
                    for key in (
                        "retrieval_strategy",
                        "retrieval_stage",
                        "retrieval_backend",
                        "hybrid_fusion",
                        "keyword_backend",
                        "rerank_score",
                        "dense_score",
                        "bm25_score",
                        "graph_score",
                        "chunk_id",
                        "parent_id",
                    )
                    if key in metadata
                },
            }
        )
    return compacted


def _compact_memory_items(items: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    compacted = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        compacted.append(
            {
                "key": item.get("key"),
                "layer": item.get("layer"),
                "topic": item.get("topic"),
                "tags": item.get("tags", []),
                "confidence": item.get("confidence"),
                "value": _trim_text(item.get("value") or "", 700),
                "recall_score": metadata.get("last_recall_score"),
                "governance_status": metadata.get("governance_status"),
                "conflict_count": metadata.get("conflict_count"),
            }
        )
    return compacted


def _rank_runs(query: str, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = _tokenize(query)

    def score(run: dict[str, Any]) -> tuple[float, str]:
        request = run.get("request") if isinstance(run.get("request"), dict) else {}
        topic = str(request.get("topic") or run.get("topic") or "")
        text = " ".join([topic, str(run.get("research_brief") or ""), str(run.get("status") or "")]).lower()
        lexical = sum(text.count(token) for token in tokens)
        return (float(lexical), str(run.get("started_at") or ""))

    return sorted(runs, key=score, reverse=True)


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    report = run.get("report") if isinstance(run.get("report"), dict) else {}
    evaluation = run.get("evaluation") if isinstance(run.get("evaluation"), dict) else {}
    request = run.get("request") if isinstance(run.get("request"), dict) else {}
    return {
        "run_id": run.get("run_id"),
        "job_id": run.get("job_id"),
        "status": run.get("status"),
        "topic": request.get("topic"),
        "depth": request.get("depth"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "duration_ms": run.get("duration_ms"),
        "plan_count": len(run.get("plan") or []),
        "route_count": len(run.get("retrieval_routes") or []),
        "evidence_count": len(run.get("evidence") or []),
        "web_hit_count": len(run.get("web_hits") or []),
        "document_hit_count": len(run.get("document_hits") or []),
        "memory_hit_count": len(run.get("memory_hits") or []),
        "source_count": report.get("source_count"),
        "evaluation_passed": evaluation.get("passed"),
        "issue_count": len(run.get("issues") or []),
        "highlights": (report.get("highlights") or [])[:3],
    }


def _summarize_trace(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, list):
        return {}
    tool_counts: dict[str, int] = {}
    actors: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for event in trace:
        if not isinstance(event, dict):
            continue
        actor = str(event.get("actor") or "unknown")
        status = str(event.get("status") or "unknown")
        actors[actor] = actors.get(actor, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
        tool_name = event.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    return {
        "event_count": len(trace),
        "actors": actors,
        "statuses": statuses,
        "tool_counts": tool_counts,
        "last_events": [
            {
                "kind": event.get("kind"),
                "actor": event.get("actor"),
                "message": _trim_text(event.get("message") or "", 220),
                "status": event.get("status"),
            }
            for event in trace[-5:]
            if isinstance(event, dict)
        ],
    }


def _compact_evaluation(evaluation: Any) -> dict[str, Any]:
    if not isinstance(evaluation, dict):
        return {}
    keys = (
        "passed",
        "plan_coverage",
        "retrieval_hit_rate",
        "private_retrieval_hit_rate",
        "evidence_sufficiency",
        "tool_selection_coverage",
        "source_quality_score",
        "context_precision",
        "context_recall",
        "faithfulness_proxy",
        "citation_precision",
        "citation_source_coverage",
        "source_diversity",
        "query_rewrite_count",
    )
    return {key: evaluation.get(key) for key in keys if key in evaluation}


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _provider_ready(readiness: Any, provider_name: str) -> bool:
    if not isinstance(readiness, dict):
        return False

    legacy = readiness.get(provider_name)
    if isinstance(legacy, dict) and "ready" in legacy:
        return bool(legacy.get("ready"))

    providers = readiness.get("providers", {})
    details = providers.get(provider_name, {}) if isinstance(providers, dict) else {}
    if not isinstance(details, dict):
        return False

    issue_prefixes = {
        "model": ("ARC_MODEL_",),
        "embedding": ("ARC_EMBEDDING_",),
        "search": ("ARC_SEARCH_",),
        "rerank": ("ARC_RERANK_",),
    }
    for issue in readiness.get("issues", []):
        if not isinstance(issue, dict):
            continue
        field = str(issue.get("field", ""))
        if any(field.startswith(prefix) for prefix in issue_prefixes.get(provider_name, ())):
            return False

    provider = str(details.get("provider", ""))
    if provider_name == "model":
        return (
            provider == "openai_compatible"
            and bool(details.get("base_url_configured"))
            and bool(details.get("api_key_configured"))
            and bool(details.get("chat_model"))
        )
    if provider_name == "embedding":
        return (
            provider not in {"", "deterministic"}
            and bool(details.get("base_url_configured"))
            and bool(details.get("api_key_configured"))
            and bool(details.get("model"))
        )
    if provider_name == "search":
        return provider not in {"", "none", "duckduckgo"}
    if provider_name == "rerank":
        return (
            provider not in {"", "rule"}
            and bool(details.get("base_url_configured"))
            and bool(details.get("api_key_configured"))
            and bool(details.get("model"))
        )
    return bool(readiness.get("ready"))


def _demo_next_actions(checks: list[dict[str, Any]]) -> list[str]:
    failed = {item["name"] for item in checks if not item["passed"]}
    actions = []
    if "api_running" in failed:
        actions.append("Start the app with scripts/start_real.ps1 so MCP can inspect runtime state.")
    if "grounding_documents" in failed:
        actions.append("Ingest one PDF/Markdown/HTML source through POST /v1/documents/ingest.")
    if "memory_records" in failed:
        actions.append("Run one research task or add a canonical memory through POST /v1/memory.")
    if "completed_runs" in failed:
        actions.append("Complete one deep research job, then inspect trace and evaluation artifacts.")
    if not actions:
        actions.append("Run a complex ODR/RAG comparison question and inspect MCP evidence in the trace.")
    return actions


def _allowed_roots() -> list[Path]:
    configured = os.getenv("ARC_MCP_DEMO_ROOTS", "")
    roots: list[Path] = []
    if configured:
        separators = ";" if ";" in configured else ","
        roots.extend(Path(part.strip()) for part in configured.split(separators) if part.strip())
    else:
        repo_root = Path(__file__).resolve().parents[2]
        roots.append(repo_root)
        for ref in (
            Path("D:/kn/_agent_refs/open_deep_research"),
            Path("D:/kn/_agent_refs/PraisonAI"),
        ):
            if ref.exists():
                roots.append(ref)

    resolved: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            candidate = root.expanduser().resolve()
        except OSError:
            continue
        if candidate.exists() and candidate.is_dir() and candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return resolved


def _root_labels() -> list[str]:
    return [str(root) for root in _allowed_roots()]


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _score_document(tokens: list[str], path: Path, text: str) -> float:
    lower_text = text.lower()
    lower_path = str(path).lower()
    score = 0.0
    for token in tokens:
        score += lower_path.count(token) * 3.0
        score += min(lower_text.count(token), 20) * 1.0
    if "mcp" in lower_path or "open_deep_research" in lower_path:
        score += 1.5
    if path.name.lower() in {"readme.md", "configuration.py", "utils.py", "deep_researcher.py"}:
        score += 1.0
    return score


def _snippet(text: str, tokens: list[str], max_chars: int = 900) -> str:
    lower_text = text.lower()
    starts = [lower_text.find(token) for token in tokens if lower_text.find(token) >= 0]
    start = min(starts) if starts else 0
    start = max(0, start - 180)
    snippet = re.sub(r"\s+", " ", text[start : start + max_chars]).strip()
    return snippet


def _tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_\-]+|[\u4e00-\u9fff]{2,}", value.lower())
    stopwords = {"the", "and", "for", "with", "this", "that", "what", "which", "how"}
    return [token for token in tokens if len(token) >= 2 and token not in stopwords]


def _first_tool(tool_registry: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for tool in tool_registry:
        if tool.get("name") == name:
            return tool
    return {}


def _server_port() -> int:
    try:
        return int(os.getenv("ARC_MCP_DEMO_PORT", str(DEFAULT_SERVER_PORT)))
    except ValueError:
        return DEFAULT_SERVER_PORT


def _clamp_limit(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_RESULT_LIMIT
    return max(1, min(parsed, 12))


def _trim_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


if __name__ == "__main__":
    main()
