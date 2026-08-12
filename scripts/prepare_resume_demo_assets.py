from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_API_BASE = "http://127.0.0.1:8010"
DEFAULT_SOURCE_DIR = Path(os.getenv("ARC_RESUME_DEMO_SOURCE_DIR", "examples/resume-demo/source-papers"))
DEFAULT_OUTPUT_DIR = Path("examples/resume-demo")
MAX_DOC_CHARS = 3600
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 900

PAPER_SPECS = [
    {
        "file": "mcmahan17a.pdf",
        "title": "FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data",
        "topic": "federated-learning-baseline",
    },
    {
        "file": "NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf",
        "title": "pFedMe: Personalized Federated Learning with Moreau Envelopes",
        "topic": "personalized-federated-learning",
    },
    {
        "file": "NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf",
        "title": "FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction",
        "topic": "model-heterogeneous-federated-learning",
    },
    {
        "file": "NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf",
        "title": "Personalized Federated Learning via Heterogeneous Model Reassembly",
        "topic": "heterogeneous-model-reassembly",
    },
    {
        "file": "FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf",
        "title": "FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning",
        "topic": "federated-knowledge-transfer",
    },
]

DEMO_REQUESTS = [
    {
        "slug": "fl-heterogeneity-comparison",
        "topic": (
            "Compare FedAvg, pFedMe, FedRolex, heterogeneous model reassembly, and FedAUX "
            "for federated learning under statistical and model heterogeneity. Use the local "
            "paper corpus as grounding, explain trade-offs, and cite evidence."
        ),
        "depth": "quick",
        "max_sections": 2,
        "max_revisions": 0,
    },
    {
        "slug": "fl-personalization-design-memo",
        "topic": (
            "Using the local federated learning corpus, write a design memo for choosing "
            "between pFedMe, FedRolex, heterogeneous model reassembly, and FedAUX when clients "
            "have non-IID data and heterogeneous model capacity. Cite local evidence and note "
            "trade-offs."
        ),
        "depth": "quick",
        "max_sections": 2,
        "max_revisions": 0,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare resume-ready AI Research Copilot demo assets.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=900) as client:
        _assert_api_ready(client, args.api_base)
        corpus_assets = _seed_corpus(client, args.api_base, args.source_dir)
        runtime_config = _get_json(client, f"{args.api_base}/v1/runtime/config")
        run_assets = [] if args.skip_runs else _run_demo_requests(client, args.api_base, args.output_dir)
        mcp_readiness = _mcp_readiness(runtime_config)
        final_profile = _get_json(client, f"{args.api_base}/v1/runtime/config")["retrieval"]["corpus_profile"]

    summary = {
        "api_base": args.api_base,
        "source_dir": "<local reference-paper folder>",
        "source_files": [spec["file"] for spec in PAPER_SPECS],
        "output_dir": str(args.output_dir),
        "corpus_assets": corpus_assets,
        "runtime": {
            "strict_ready": runtime_config["provider_readiness"]["ready"],
            "runtime": runtime_config["orchestration"]["runtime"],
            "queue": runtime_config["job_execution"]["queue_backend"],
            "vector_backend": runtime_config["retrieval"]["vector_backend"],
            "keyword_backend": runtime_config["retrieval"]["keyword_backend"],
            "reranker": runtime_config["retrieval"]["hybrid_pipeline"]["reranker"],
            "mcp_loaded": _mcp_loaded(runtime_config),
        },
        "mcp_readiness": mcp_readiness,
        "final_corpus_profile": final_profile,
        "run_assets": run_assets,
        "demo_boundary": (
            "This script seeds controlled excerpts from PDFs for a reliable resume demo. "
            "Full PDF ingestion remains available through /v1/documents/ingest, but large "
            "PDFs can be slow because strict mode uses real contextualization and embeddings."
        ),
    }
    _write_json(args.output_dir / "demo-summary.json", summary)
    _write_markdown_summary(args.output_dir / "demo-summary.md", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def _assert_api_ready(client: httpx.Client, api_base: str) -> None:
    response = client.get(f"{api_base}/health")
    response.raise_for_status()
    runtime = _get_json(client, f"{api_base}/v1/runtime/provider-check")
    if not runtime.get("ready"):
        raise RuntimeError(f"Runtime is not strict-provider ready: {runtime.get('issues')}")


def _seed_corpus(client: httpx.Client, api_base: str, source_dir: Path) -> list[dict[str, Any]]:
    existing = {
        item.get("source")
        for item in _get_json(client, f"{api_base}/v1/documents")
        if isinstance(item, dict)
    }
    assets: list[dict[str, Any]] = []
    for spec in PAPER_SPECS:
        path = source_dir / spec["file"]
        source = f"resume-demo-papers/{spec['file']}"
        legacy_source = str(path)
        if source in existing or legacy_source in existing:
            assets.append({"title": spec["title"], "source": source, "status": "skipped_existing"})
            continue
        content = _extract_pdf_demo_text(path)
        response = client.post(
            f"{api_base}/v1/documents",
            json={
                "title": spec["title"],
                "source": source,
                "snippet": _trim(content, 700),
                "content": content,
                "metadata": {
                    "demo_asset": "resume_demo_2026_07",
                    "domain": "federated_learning",
                    "topic": spec["topic"],
                    "source_type": "paper_pdf_excerpt",
                    "original_pdf_name": spec["file"],
                    "max_doc_chars": MAX_DOC_CHARS,
                },
            },
        )
        response.raise_for_status()
        document = response.json()
        assets.append(
            {
                "title": spec["title"],
                "source": source,
                "status": "ingested_excerpt",
                "document_id": document.get("metadata", {}).get("document_id"),
                "content_chars": len(content),
            }
        )
    return assets


def _run_demo_requests(client: httpx.Client, api_base: str, output_dir: Path) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for spec in DEMO_REQUESTS:
        existing_asset = _load_existing_completed_run_asset(output_dir, spec["slug"])
        if existing_asset is not None:
            assets.append(existing_asset)
            continue
        response = client.post(
            f"{api_base}/v1/research/jobs",
            json={
                "topic": spec["topic"],
                "depth": spec["depth"],
                "include_private_docs": True,
                "max_sections": spec["max_sections"],
                "max_revisions": spec["max_revisions"],
            },
        )
        response.raise_for_status()
        job = response.json()
        job_id = job["job_id"]
        completed_job = _poll_job(client, api_base, job_id)
        if completed_job["status"] != "completed":
            assets.append({"slug": spec["slug"], "job_id": job_id, "status": completed_job["status"], "error": completed_job.get("error")})
            continue
        run_id = completed_job["run_id"]
        run = _get_json(client, f"{api_base}/v1/research/jobs/{job_id}/result")
        trace = _get_json(client, f"{api_base}/v1/research/runs/{run_id}/trace")
        evaluation = _get_json(client, f"{api_base}/v1/research/runs/{run_id}/evaluation")
        prefix = output_dir / spec["slug"]
        _write_json(prefix.with_suffix(".run.json"), run)
        _write_json(prefix.with_suffix(".trace.json"), trace)
        _write_json(prefix.with_suffix(".evaluation.json"), evaluation)
        report_path = prefix.with_suffix(".report.md")
        report_path.write_text(_render_run_markdown(run, evaluation), encoding="utf-8")
        assets.append(
            {
                "slug": spec["slug"],
                "job_id": job_id,
                "run_id": run_id,
                "status": run.get("status"),
                "source_count": (run.get("report") or {}).get("source_count", 0),
                "evaluation_passed": evaluation.get("passed", False),
                "trace_events": len(trace),
                "report": str(report_path),
            }
        )
    return assets


def _load_existing_completed_run_asset(output_dir: Path, slug: str) -> dict[str, Any] | None:
    prefix = output_dir / slug
    run_path = prefix.with_suffix(".run.json")
    trace_path = prefix.with_suffix(".trace.json")
    evaluation_path = prefix.with_suffix(".evaluation.json")
    report_path = prefix.with_suffix(".report.md")
    if not all(path.exists() for path in [run_path, trace_path, evaluation_path, report_path]):
        return None
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if run.get("status") != "completed":
        return None
    return {
        "slug": slug,
        "job_id": run.get("job_id"),
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "source_count": (run.get("report") or {}).get("source_count", 0),
        "evaluation_passed": evaluation.get("passed", False),
        "trace_events": len(trace),
        "report": str(report_path),
        "reused_existing": True,
    }


def _poll_job(client: httpx.Client, api_base: str, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        job = _get_json(client, f"{api_base}/v1/research/jobs/{job_id}/status")
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Timed out waiting for job {job_id}")


def _mcp_readiness(runtime_config: dict[str, Any]) -> dict[str, Any]:
    for tool in runtime_config.get("tool_registry", []):
        if tool.get("name") == "mcp_tool":
            return {
                "configured_enabled": bool(tool.get("configured_enabled")),
                "loaded": bool(tool.get("loaded")),
                "server_url_configured": bool(tool.get("server_url_configured")),
                "tool_catalog_count": int(tool.get("tool_catalog_count") or 0),
                "tools": tool.get("tools") or [],
            }
    return {
        "configured_enabled": False,
        "loaded": False,
        "server_url_configured": False,
        "tool_catalog_count": 0,
        "tools": [],
    }


def _mcp_loaded(runtime_config: dict[str, Any]) -> bool:
    for tool in runtime_config.get("tool_registry", []):
        if tool.get("name") == "mcp_tool":
            return bool(tool.get("loaded"))
    return False


def _extract_pdf_demo_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import fitz
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install the documents extra for PyMuPDF PDF extraction: pip install -e .[documents]") from exc

    page_texts: list[str] = []
    with fitz.open(path) as doc:
        for page_index in range(min(len(doc), 4)):
            text = doc[page_index].get_text("text") or ""
            if text.strip():
                page_texts.append(f"[PDF page {page_index + 1}]\n{text.strip()}")
    cleaned = _normalize_text("\n\n".join(page_texts))
    return _trim(cleaned, MAX_DOC_CHARS)


def _render_run_markdown(run: dict[str, Any], evaluation: dict[str, Any]) -> str:
    report = run.get("report") or {}
    lines = [
        f"# {report.get('title') or run.get('topic') or 'Research Run'}",
        "",
        "## Run",
        "",
        f"- Run ID: `{run.get('run_id')}`",
        f"- Status: `{run.get('status')}`",
        f"- Topic: {run.get('topic') or (run.get('request') or {}).get('topic')}",
        f"- Source count: {report.get('source_count', 0)}",
        f"- Evaluation passed: `{evaluation.get('passed', False)}`",
        f"- Context precision: {evaluation.get('context_precision', 0)}",
        f"- Context recall: {evaluation.get('context_recall', 0)}",
        f"- Faithfulness proxy: {evaluation.get('faithfulness_proxy', 0)}",
        f"- Citation precision: {evaluation.get('citation_precision', 0)}",
        "",
        "## Summary",
        "",
        report.get("summary") or "",
        "",
    ]
    for section in report.get("sections", []) or []:
        lines.extend([f"## {section.get('heading', 'Section')}", "", section.get("content") or "", ""])
        citations = section.get("citations") or []
        if citations:
            lines.append("Citations:")
            for citation in citations:
                title = citation.get("title", "")
                source = citation.get("source", "")
                url = citation.get("url")
                suffix = f" - {url}" if url else ""
                lines.append(f"- {title} ({source}){suffix}")
            lines.append("")
    source_index = report.get("source_index") or []
    if source_index:
        lines.extend(["## Source Index", ""])
        for source in source_index:
            lines.append(f"- {source}")
        lines.append("")
    notes = evaluation.get("notes") or []
    if notes:
        lines.extend(["## Evaluation Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def _write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Resume Demo Summary",
        "",
        "## Runtime",
        "",
        f"- API base: `{summary['api_base']}`",
        f"- Strict provider ready: `{summary['runtime']['strict_ready']}`",
        f"- Runtime: `{summary['runtime']['runtime']}`",
        f"- Queue: `{summary['runtime']['queue']}`",
        f"- Retrieval: `{summary['runtime']['vector_backend']} + {summary['runtime']['keyword_backend']}`",
        f"- Reranker: `{summary['runtime']['reranker']}`",
        f"- MCP loaded: `{summary['runtime']['mcp_loaded']}`",
        "",
        "## Corpus",
        "",
        f"- Documents: {summary['final_corpus_profile']['document_count']}",
        f"- Sources: {summary['final_corpus_profile']['source_count']}",
        f"- Vector backend: `{summary['final_corpus_profile']['vector_backend']}`",
        f"- Keyword backend: `{summary['final_corpus_profile']['keyword_backend']}`",
        "",
        "## MCP Readiness",
        "",
        f"- Configured enabled: `{summary['mcp_readiness']['configured_enabled']}`",
        f"- Loaded: `{summary['mcp_readiness']['loaded']}`",
        f"- Server URL configured: `{summary['mcp_readiness']['server_url_configured']}`",
        f"- Tool catalog count: {summary['mcp_readiness']['tool_catalog_count']}",
        "",
        "## Runs",
        "",
    ]
    for asset in summary["run_assets"]:
        lines.extend(
            [
                f"- `{asset['slug']}`: status `{asset.get('status')}`, sources {asset.get('source_count')}, "
                f"evaluation `{asset.get('evaluation_passed')}`, trace events {asset.get('trace_events')}",
            ]
        )
    lines.extend(["", "## Boundary", "", summary["demo_boundary"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _get_json(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> Any:
    response = client.get(url, params=params)
    response.raise_for_status()
    return response.json()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _trim(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
