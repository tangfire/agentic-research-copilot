from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from .schemas import ConstraintCoverage, EvidenceItem, MemoryItem, ResearchRun, ResearchReport


@dataclass(frozen=True)
class ConstraintTextSource:
    content: str
    source_id: str


_CONSTRAINT_KEYWORDS = (
    "constraint",
    "constraints",
    "team",
    "部署",
    "回滚",
    "fastapi",
    "docker",
    "mcp",
    "checkpoint",
    "memory",
)
_IGNORED_LINE_PREFIXES = ("已覆盖：", "未覆盖：", "弱覆盖：")
_IGNORED_LINE_PREFIXES_EN = ("covered:", "missing:", "weak:")
_IGNORED_LINE_MARKERS = (
    "parent document:",
    "matched child chunk:",
    "metadata:",
    "source:",
    "context_confid",
)


def _clean_constraint_line(line: str) -> str:
    cleaned = " ".join(line.split()).strip(" -\t")
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if cleaned.startswith(_IGNORED_LINE_PREFIXES) or lowered.startswith(_IGNORED_LINE_PREFIXES_EN):
        return ""
    if any(marker in lowered for marker in _IGNORED_LINE_MARKERS):
        return ""
    return cleaned


def extract_constraint_texts(value: str, *, fallback_to_full_text: bool = False) -> list[str]:
    normalized = " ".join(value.split())
    if not normalized:
        return []
    texts: list[str] = []
    for line in value.splitlines():
        line = _clean_constraint_line(line)
        if not line:
            continue
        if "[project/constraint]" in line or "[session/constraint]" in line or "[user/preference]" in line:
            extracted = _clean_constraint_line(line.split("]", 1)[-1])
            if extracted:
                texts.append(extracted)
            continue
        if any(keyword in line.lower() for keyword in _CONSTRAINT_KEYWORDS):
            texts.append(line)
    if fallback_to_full_text and not texts and len(normalized) > 18:
        cleaned = _clean_constraint_line(normalized)
        if cleaned:
            texts.append(cleaned)
    return _dedupe(texts)


def derive_constraints_from_memories(memories: Iterable[MemoryItem]) -> list[ConstraintTextSource]:
    items: list[ConstraintTextSource] = []
    for memory in memories:
        if memory.scope == "project" or memory.kind == "constraint":
            for text in extract_constraint_texts(memory.content, fallback_to_full_text=True):
                items.append(ConstraintTextSource(content=text, source_id=memory.memory_id))
    return items


def extract_constraints_from_run_topic(topic: str) -> list[ConstraintTextSource]:
    extracted = []
    for index, text in enumerate(extract_constraint_texts(topic)):
        extracted.append(ConstraintTextSource(content=text, source_id=f"topic:{index}"))
    return extracted


def evaluate_constraint_coverage(
    *,
    run_id: str,
    session_id: str | None,
    constraints: Iterable[ConstraintTextSource],
    report: ResearchReport,
    evidence: Iterable[EvidenceItem],
) -> list[ConstraintCoverage]:
    report_sections = report.sections or []
    coverage: list[ConstraintCoverage] = []
    for constraint in constraints:
        content_tokens = _tokens(constraint.content)
        matched_sections: list[str] = []
        matched_evidence: list[str] = []
        if content_tokens:
            for section in report_sections:
                section_tokens = _tokens(" ".join([section.heading, section.content]))
                if len(content_tokens & section_tokens) >= max(1, min(3, len(content_tokens) // 3 or 1)):
                    matched_sections.append(section.heading)
            for item in evidence:
                evidence_tokens = _tokens(" ".join([item.title, item.source, item.snippet or "", item.content or ""]))
                if content_tokens & evidence_tokens:
                    matched_evidence.append(item.title)
        else:
            for section in report_sections:
                if constraint.content[:12] and constraint.content[:12].lower() in section.content.lower():
                    matched_sections.append(section.heading)
            for item in evidence:
                haystack = " ".join([item.title, item.source, item.snippet or "", item.content or ""]).lower()
                if constraint.content[:12].lower() in haystack:
                    matched_evidence.append(item.title)

        covered = bool(matched_sections)
        confidence = 0.0
        if covered:
            confidence = min(1.0, 0.45 + (0.2 * len(matched_sections)) + (0.15 * len(matched_evidence)))
        reason = (
            "Covered explicitly by report sections; evidence matches are supporting context."
            if covered
            else "No strong explicit match found in report sections."
        )
        coverage.append(
            ConstraintCoverage(
                constraint_id=_stable_constraint_id(run_id, session_id, constraint.source_id, constraint.content),
                run_id=run_id,
                session_id=session_id,
                content=constraint.content,
                covered=covered,
                matched_sections=_dedupe(matched_sections),
                matched_evidence=_dedupe(matched_evidence),
                confidence=round(confidence, 4),
                reason=reason,
                metadata={"source_id": constraint.source_id},
            )
        )
    return coverage


def summarise_constraint_coverage(coverage: list[ConstraintCoverage]) -> dict[str, object]:
    total = len(coverage)
    covered = sum(1 for item in coverage if item.covered)
    score = covered / total if total else 0.0
    passed = score >= 0.6 if total else True
    warnings: list[str] = []
    if total and score < 0.6:
        warnings.append("Constraint coverage is weak; some project constraints are not reflected in the report.")
    if total and score < 0.4:
        warnings.append("Constraint coverage is critically low; treat the run as evaluation failed.")
    return {
        "score": round(score, 4),
        "covered": covered,
        "total": total,
        "passed": passed,
        "warnings": warnings,
    }


def apply_constraint_coverage_gate(run: ResearchRun, coverage: list[ConstraintCoverage]) -> ResearchRun:
    if not coverage:
        return run
    summary = summarise_constraint_coverage(coverage)
    evaluation = run.evaluation
    if evaluation is None:
        return run
    notes = list(evaluation.notes)
    notes.extend(summary["warnings"])
    passed = evaluation.passed and bool(summary["passed"])
    if not summary["passed"] and summary["score"] < 0.4:
        passed = False
    return run.model_copy(
        update={
            "evaluation": evaluation.model_copy(update={"passed": passed, "notes": _dedupe(notes)}),
        }
    )


def extract_constraint_coverage_from_run(run: ResearchRun) -> list[ConstraintCoverage]:
    constraints: list[ConstraintTextSource] = [
        ConstraintTextSource(content=text, source_id=f"topic:{index}")
        for index, text in enumerate(extract_constraint_texts(run.request.topic))
    ]
    metadata = run.request.metadata or {}
    metadata_values = [
        metadata.get("workspace_context"),
        metadata.get("team_context"),
        metadata.get("default_stack"),
        metadata.get("deployment_constraints"),
        metadata.get("risk_policy"),
        metadata.get("memory_context"),
        metadata.get("hard_constraints"),
        metadata.get("constraints"),
    ]
    for value in metadata_values:
        if isinstance(value, str):
            for index, text in enumerate(extract_constraint_texts(value, fallback_to_full_text=True)):
                constraints.append(ConstraintTextSource(content=text, source_id=f"metadata:{index}"))
        elif isinstance(value, (list, tuple)):
            for entry_index, item in enumerate(value):
                if not isinstance(item, str):
                    continue
                for text_index, text in enumerate(extract_constraint_texts(item, fallback_to_full_text=True)):
                    constraints.append(
                        ConstraintTextSource(content=text, source_id=f"metadata:{entry_index}:{text_index}")
                    )
    return evaluate_constraint_coverage(
        run_id=run.run_id,
        session_id=None,
        constraints=_dedupe_constraint_sources(constraints),
        report=run.report or ResearchReport(title="", summary=""),
        evidence=run.evidence,
    )


def _soft_match(needle: str, haystack: str) -> bool:
    needle = " ".join(needle.split()).lower()
    if not needle:
        return False
    head = needle[:20]
    return head in haystack.lower()


def _stable_constraint_id(*parts: str | None) -> str:
    basis = "::".join(part or "" for part in parts)
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()
    return f"constraint_{digest[:16]}"


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", value.lower()) if len(token) > 1}


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_constraint_sources(values: Iterable[ConstraintTextSource]) -> list[ConstraintTextSource]:
    seen: set[tuple[str, str]] = set()
    result: list[ConstraintTextSource] = []
    for value in values:
        key = (value.source_id, " ".join(value.content.split()))
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
