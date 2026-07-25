from __future__ import annotations

import json
import os
from pathlib import Path

from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.schemas import ResearchRequest
from agentic_research_copilot.settings import load_settings


DEMO_TOPIC = (
    "Design a resume-ready AI Research Copilot by comparing Open Deep Research-style "
    "orchestration with LangGraph persistence, Qdrant hybrid retrieval, Ragas-style "
    "evaluation, and PraisonAI-inspired memory and trace design. Include trade-offs, "
    "failure modes, and single-node deployment boundaries."
)


def main() -> None:
    settings = load_settings()
    copilot = ResearchCopilot(settings=settings)
    try:
        _seed_demo_context(copilot)
        run = copilot.run(_demo_request())
    finally:
        copilot.close()

    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    (examples_dir / "demo-report.md").write_text(_render_report(run, settings), encoding="utf-8")
    (examples_dir / "demo-trace.json").write_text(
        json.dumps(_trace_payload(run, settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "status": run.status,
                "source_count": run.report.source_count if run.report else 0,
                "evaluation_passed": run.evaluation.passed if run.evaluation else False,
                "report_path": "examples/demo-report.md",
                "trace_path": "examples/demo-trace.json",
            },
            ensure_ascii=False,
        )
    )


def _seed_demo_context(copilot: ResearchCopilot) -> None:
    _ensure_demo_doc(
        copilot,
        title="AI Engineering Interview Positioning Notes",
        source="examples/interview-positioning-notes.md",
        url=None,
        snippet=(
            "Interviewers should understand the project as a deep research copilot "
            "that turns complex questions into citation-backed reports."
        ),
        content=(
            "A strong interview narrative should lead with the user problem, then show the agent handoff flow, "
            "retrieval routing, Qdrant grounding, layered memory, citation verification, evaluation, and trace replay. "
            "The strongest resume angle is practical experience with multi-agent orchestration, RAG quality gates, "
            "OpenAI-compatible providers, contextual document grounding, LangGraph orchestration, and observable "
            "failure handling. The demo should explain trade-offs, failure modes, single-node boundaries, and why "
            "this is a focused AI Research Copilot rather than a generic agent platform."
        ),
        metadata={"kind": "demo_context_note", "topic": "interview positioning"},
    )
    _ensure_demo_doc(
        copilot,
        title="LangGraph Persistence and Checkpointing Notes",
        source="docs.langchain.com/langgraph/persistence",
        url="https://docs.langchain.com/oss/python/langgraph/persistence",
        snippet="LangGraph persistence uses checkpointers to save graph state and support replay, recovery, and long-running workflows.",
        content=(
            "LangGraph persistence is relevant because a research copilot needs inspectable state across planner, "
            "researcher, reporter, verifier, and memory-write nodes. The demo should describe SQLite checkpointing as "
            "a single-node durability choice: it is appropriate for a personal project and local deployment, while "
            "distributed production would need stronger queueing, cancellation, retry, and operational controls."
        ),
        metadata={"kind": "official_reference", "topic": "langgraph persistence"},
    )
    _ensure_demo_doc(
        copilot,
        title="Qdrant Hybrid Retrieval Notes",
        source="qdrant.tech/documentation/search/hybrid-queries",
        url="https://qdrant.tech/documentation/search/hybrid-queries/",
        snippet="Qdrant hybrid queries combine dense and sparse signals and can fuse candidates with strategies such as RRF or DBSF.",
        content=(
            "Qdrant hybrid retrieval is the right evidence source for explaining why this project is stronger than "
            "plain top-k RAG. Dense vectors capture semantic similarity, sparse vectors preserve lexical matches, and "
            "RRF or DBSF fusion makes retrieval behavior easier to reason about before reranking."
        ),
        metadata={"kind": "official_reference", "topic": "hybrid retrieval"},
    )
    _ensure_demo_doc(
        copilot,
        title="Ragas Metric Notes",
        source="docs.ragas.io/metrics",
        url="https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/",
        snippet="Ragas-style metrics evaluate faithfulness, context precision, context recall, and answer relevancy for RAG systems.",
        content=(
            "Ragas-style evaluation is useful as a demo artifact because it separates retrieval quality from generation "
            "quality. The project should describe local proxy metrics as lightweight regression gates and optional Ragas "
            "or LLM-as-judge artifacts as presentation evidence, not as a full benchmark claim."
        ),
        metadata={"kind": "evaluation_reference", "topic": "rag evaluation"},
    )
    _ensure_demo_doc(
        copilot,
        title="Open Deep Research Interaction Notes",
        source="github.com/langchain-ai/open_deep_research",
        url="https://github.com/langchain-ai/open_deep_research",
        snippet="Open Deep Research uses a LangGraph research workflow with planning, delegated research, compressed findings, and final report generation.",
        content=(
            "The useful reference is the research-loop shape: receive a question, clarify or write a research brief, "
            "delegate focused research work, compress findings while preserving citations, then synthesize a final report. "
            "This project adapts that loop for a focused AI Research Copilot with contextual retrieval, memory governance, "
            "trace replay, and single-node job execution."
        ),
        metadata={"kind": "reference_design", "topic": "open deep research"},
    )
    if not any(record.key == "demo:interview_goal" for record in copilot.memory.list()):
        copilot.add_memory(
            key="demo:interview_goal",
            value=(
                "Position the project as a LangGraph-based agentic research copilot that demonstrates planning, "
                "RAG, memory, tool use, citation verification, evaluation, and traceability for AI engineering interviews."
            ),
            tags=["demo", "interview", "positioning"],
            layer="summary",
            topic=DEMO_TOPIC,
            confidence=0.9,
        )


def _ensure_demo_doc(
    copilot: ResearchCopilot,
    *,
    title: str,
    source: str,
    url: str | None,
    snippet: str,
    content: str,
    metadata: dict[str, object],
) -> None:
    if any(document.title == title for document in copilot.documents.list()):
        return
    copilot.add_document(
        title=title,
        source=source,
        url=url,
        snippet=snippet,
        content=content,
        metadata=metadata,
    )


def _demo_request() -> ResearchRequest:
    return ResearchRequest(
        topic=os.getenv("ARC_DEMO_TOPIC", DEMO_TOPIC),
        depth=os.getenv("ARC_DEMO_DEPTH", "standard"),
        include_private_docs=_env_bool("ARC_DEMO_INCLUDE_PRIVATE_DOCS", True),
        use_memory=_env_bool("ARC_DEMO_USE_MEMORY", False),
        max_sections=int(os.getenv("ARC_DEMO_MAX_SECTIONS", "5")),
        max_revisions=int(os.getenv("ARC_DEMO_MAX_REVISIONS", "0")),
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _render_report(run, settings) -> str:
    report = run.report
    evaluation = run.evaluation
    lines = [
        "# Demo Report: AI Research Copilot",
        "",
        "This artifact is generated by `scripts/capture_demo.py` from a real research run.",
        "",
        "## Run Summary",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Status: `{run.status}`",
        f"- Topic: {run.request.topic}",
        f"- Chat model: `{settings.model_chat_model}`",
        f"- Embedding model: `{settings.embedding_model}`",
        f"- Search provider: `{settings.search_provider}`",
        f"- Source count: {report.source_count if report else 0}",
        f"- Revision count: {run.revision_count}",
        f"- Failure reason: {run.failure_reason or 'none'}",
    ]
    if evaluation is not None:
        lines.extend(
            [
                f"- Evaluation passed: `{evaluation.passed}`",
                f"- Plan coverage: {evaluation.plan_coverage}",
                f"- Context precision: {evaluation.context_precision}",
                f"- Context recall: {evaluation.context_recall}",
                f"- Faithfulness proxy: {evaluation.faithfulness_proxy}",
                f"- Citation precision: {evaluation.citation_precision}",
                f"- Citation source coverage: {evaluation.citation_source_coverage}",
            ]
        )
    lines.append("")

    if report is None:
        lines.extend(["## Report", "", "No report was generated."])
        return "\n".join(lines)

    lines.extend(["## Report", "", f"## {_clean_demo_text(report.title)}", "", _clean_demo_text(report.summary), ""])
    for section in report.sections:
        lines.extend([f"### {_clean_demo_text(section.heading)}", "", _clean_demo_text(section.content), ""])
        if section.citations:
            lines.append("Citations:")
            for citation in section.citations:
                suffix = f" - {citation.url}" if citation.url else ""
                lines.append(f"- {_clean_demo_text(citation.title)} ({citation.source}){suffix}")
            lines.append("")

    lines.extend(["## Source Index", ""])
    for source in report.source_index:
        lines.append(f"- {_clean_demo_text(source)}")
    lines.append("")

    if run.issues:
        lines.extend(["## Non-Blocking Verifier Notes", ""])
        for issue in run.issues:
            lines.append(f"- {_clean_demo_text(issue)}")
        lines.append("")

    return "\n".join(lines)


def _clean_demo_text(value: str) -> str:
    replacements = {
        "—": " - ",
        "→": " -> ",
        "鈫抪": " -> p",
        "鈫抮": " -> r",
        "鈫抳": " -> v",
        "鈫抦": " -> m",
        "鈥攕": " - s",
        "鈥攎": " - m",
        "鈥攖": " - t",
        "鈥攚": " - w",
        "鈥攏": " - n",
        "鈥": "'",
    }
    cleaned = value
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def _trace_payload(run, settings) -> dict[str, object]:
    model_events = [
        {
            "actor": event.actor,
            "kind": event.kind,
            "step": event.step,
            "provider": event.provider,
            "model": event.model,
            "tokens_in": event.tokens_in,
            "tokens_out": event.tokens_out,
            "latency_ms": event.latency_ms,
        }
        for event in run.trace
        if event.model
    ]
    tool_calls = [
        {
            "actor": event.actor,
            "tool_name": event.tool_name,
            "provider": event.provider,
            "step": event.step,
            "latency_ms": event.latency_ms,
            "metadata": event.metadata,
        }
        for event in run.trace
        if event.kind == "tool_call"
    ]
    return {
        "run_id": run.run_id,
        "status": run.status,
        "request": run.request.model_dump(mode="json"),
        "report": run.report.model_dump(mode="json") if run.report else None,
        "evidence_contexts": [
            {
                "id": f"{evidence.kind}:{evidence.source}:{evidence.title}",
                "kind": evidence.kind,
                "title": evidence.title,
                "source": evidence.source,
                "url": evidence.url,
                "snippet": evidence.snippet,
                "content": evidence.content,
                "score": evidence.score,
            }
            for evidence in run.evidence
        ],
        "provider_summary": {
            "model_provider": settings.model_provider,
            "chat_model": settings.model_chat_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "search_provider": settings.search_provider,
            "search_model": settings.search_model,
        },
        "evaluation": run.evaluation.model_dump(mode="json") if run.evaluation else None,
        "corpus_profile": run.corpus_profile.model_dump(mode="json") if run.corpus_profile else None,
        "handoffs": [handoff.model_dump(mode="json") for handoff in run.handoffs],
        "model_events": model_events,
        "tool_calls": tool_calls,
        "checkpoints": [checkpoint.model_dump(mode="json") for checkpoint in run.checkpoints],
        "trace_events": [event.model_dump(mode="json") for event in run.trace],
    }


if __name__ == "__main__":
    main()
