from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, TypeVar

import httpx

from .schemas import (
    CorpusProfile,
    EvidenceItem,
    MemoryRecord,
    PlanItem,
    PlannerContract,
    ReporterContract,
    ResearchRequest,
    ResearchReport,
    ReportSection,
    VerificationContract,
)


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")

TContract = TypeVar("TContract")


@dataclass
class ModelUsage:
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ResearchModelProvider(Protocol):
    name: str
    embedding_dimensions: int

    def draft_plan(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]: ...

    def assess_report(
        self,
        report: ResearchReport,
        evidence: Sequence[EvidenceItem],
        plan: Sequence[PlanItem],
        *,
        revision_count: int = 0,
        max_revisions: int = 2,
    ) -> tuple[VerificationContract, ModelUsage]: ...

    def compose_report(
        self,
        topic: str,
        sections: Sequence[ReportSection],
        evidence: Sequence[EvidenceItem],
        confidence: float,
    ) -> tuple[ReporterContract, ModelUsage]: ...

    def embed_text(self, text: str) -> tuple[list[float], ModelUsage]: ...
    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], ModelUsage]: ...


class DeterministicResearchModelProvider:
    name = "deterministic"

    def __init__(self, embedding_dimensions: int = 256) -> None:
        self.embedding_dimensions = max(32, embedding_dimensions)

    def draft_plan(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]:
        topic = request.topic.strip()
        memory_context = _summarize_memory(memory_records)
        focus = _focus_for_topic(topic)
        brief_bits = [
            f"Research the topic '{topic}' for a {request.audience} audience.",
            "Prioritize citation-backed evidence, explicit handoffs, and a verifiable source index.",
            f"Focus on {focus}.",
        ]
        if corpus_profile.has_private_docs:
            brief_bits.append(
                f"Ground the answer with {corpus_profile.document_count} uploaded context documents from {corpus_profile.source_count} sources."
            )
        if memory_context:
            brief_bits.append(f"Reuse memory where it is relevant: {memory_context}.")
        if revision_count > 0 and revision_notes:
            brief_bits.append("Repair the previously flagged gaps: " + "; ".join(revision_notes[:3]))

        plan_items = _build_plan_items(request, topic, revision_count=revision_count)
        assumptions = [
            "The final answer must remain source-backed and inspectable.",
            "The supervisor may revise the plan if verification exposes citation or coverage gaps.",
        ]
        if corpus_profile.has_private_docs:
            assumptions.append("Internal grounding is available and should be preferred for project-specific facts.")
        if memory_records:
            assumptions.append("Session and canonical memory can shorten the evidence search.")

        success_criteria = [
            "Every substantive section has citations.",
            "The run records handoffs and trace events.",
            "Verifier issues are either resolved or surfaced as a failure state.",
        ]
        if revision_notes:
            success_criteria.append("The next pass addresses the previously cited gaps.")

        contract = PlannerContract(
            research_brief=" ".join(brief_bits),
            plan=plan_items[: request.max_sections],
            assumptions=assumptions,
            success_criteria=success_criteria,
            revision_budget=request.max_revisions,
            confidence=min(0.92, 0.68 + 0.03 * len(plan_items) + (0.03 if revision_count else 0.0)),
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-planner",
            prompt_tokens=96 + len(topic.split()) * 4,
            completion_tokens=128 + len(contract.plan) * 16,
            latency_ms=3,
        )
        return contract, usage

    def assess_report(
        self,
        report: ResearchReport,
        evidence: Sequence[EvidenceItem],
        plan: Sequence[PlanItem],
        *,
        revision_count: int = 0,
        max_revisions: int = 2,
    ) -> tuple[VerificationContract, ModelUsage]:
        issues: list[str] = []
        critical_issues: list[str] = []

        if not evidence:
            issues.append("No evidence attached to the report.")
            critical_issues.append("No evidence attached to the report.")
        if report.sections and not report.citations:
            issues.append("Report sections exist but no citations were assembled.")
            critical_issues.append("Report sections exist but no citations were assembled.")
        uncited_sections = [
            section.heading
            for section in report.sections
            if section.content.strip() and not section.citations
        ]
        if uncited_sections:
            issues.append(f"Sections missing citations: {len(uncited_sections)}")
        if report.citations and not report.source_index:
            issues.append("Source index is missing despite attached citations.")
            critical_issues.append("Source index is missing despite attached citations.")
        if len(report.sections) < 3:
            issues.append("Report is too thin for interview-grade explanation.")
        if report.confidence < 0.55:
            issues.append("Confidence is too low.")
        unique_sources = {
            item.source
            for item in evidence
            if item.source and item.source not in {"internal-note", "memory"}
        }
        if evidence and len(unique_sources) < 2:
            issues.append("Evidence sources are not diverse enough.")
        uncovered = [item.question for item in plan if item.requires_research and item.evidence_count == 0]
        if uncovered:
            issues.append(f"Uncovered plan items: {len(uncovered)}")
        if report.source_count and report.source_count < min(3, len(evidence)):
            issues.append("Source count is weaker than the evidence volume.")
        coverage_score = 0.0
        if plan:
            covered = len(plan) - len(uncovered)
            coverage_score = max(0.0, min(1.0, covered / len(plan)))

        should_revise = bool(critical_issues or (issues and revision_count < max_revisions))
        revision_reason = "; ".join((critical_issues or issues)[:2]) if (critical_issues or issues) else None
        contract = VerificationContract(
            issues=issues,
            critical_issues=critical_issues,
            should_revise=should_revise,
            revision_reason=revision_reason,
            confidence=max(0.0, min(0.96, 0.56 + coverage_score * 0.2 - (0.1 if critical_issues else 0.0))),
            coverage_score=coverage_score,
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-verifier",
            prompt_tokens=72 + len(report.sections) * 16,
            completion_tokens=64 + len(issues) * 8,
            latency_ms=2,
        )
        return contract, usage

    def compose_report(
        self,
        topic: str,
        sections: Sequence[ReportSection],
        evidence: Sequence[EvidenceItem],
        confidence: float,
    ) -> tuple[ReporterContract, ModelUsage]:
        unique_sources: list[str] = []
        seen_sources: set[str] = set()
        source_index: list[str] = []
        seen_citations: set[str] = set()
        for index, item in enumerate(evidence, start=1):
            source_key = item.url or f"{item.source}:{item.title}"
            if source_key not in seen_citations:
                seen_citations.add(source_key)
                suffix = f" - {item.url}" if item.url else ""
                source_index.append(f"[{index}] {item.title} ({item.source}){suffix}")
            if item.source not in seen_sources:
                seen_sources.add(item.source)
                unique_sources.append(item.source)

        contract = ReporterContract(
            title=f"{topic} Research Brief",
            summary=(
                f"A structured research brief on {topic} with {len(sections)} sections and "
                f"{len(unique_sources)} source groups."
            ),
            highlights=[
                f"{section.heading}: {section.content[:120].rstrip()}"
                for section in list(sections)[:3]
            ],
            recommendations=[
                "Keep citations attached to each substantive claim.",
                "Expose handoffs and trace events in the UI for debugging and interviews.",
                "Use the same contract to swap in a real LLM provider later without changing orchestration.",
            ],
            source_index=source_index,
            confidence=confidence,
        )
        usage = ModelUsage(
            provider=self.name,
            model="heuristic-reporter",
            prompt_tokens=64 + len(sections) * 12,
            completion_tokens=72 + len(evidence) * 8,
            latency_ms=2,
        )
        return contract, usage

    def embed_text(self, text: str) -> tuple[list[float], ModelUsage]:
        vector = _hashed_dense_vector(text, self.embedding_dimensions)
        usage = ModelUsage(provider=self.name, model="hashed-embedding", prompt_tokens=max(1, len(text) // 4), latency_ms=1)
        return vector, usage

    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], ModelUsage]:
        vectors = [_hashed_dense_vector(text, self.embedding_dimensions) for text in texts]
        usage = ModelUsage(
            provider=self.name,
            model="hashed-embedding",
            prompt_tokens=sum(max(1, len(text) // 4) for text in texts),
            latency_ms=max(1, len(texts)),
        )
        return vectors, usage


class OpenAICompatibleResearchModelProvider:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        chat_model: str,
        embedding_model: str,
        timeout_seconds: float = 30.0,
        temperature: float = 0.2,
        embedding_dimensions: int = 256,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.embedding_dimensions = max(32, embedding_dimensions)

    def draft_plan(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        memory_records: Sequence[MemoryRecord] = (),
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]:
        payload = {
            "request": request.model_dump(),
            "corpus_profile": corpus_profile.model_dump(),
            "memory_records": [record.model_dump() for record in memory_records[:8]],
            "revision_count": revision_count,
            "revision_notes": list(revision_notes)[:8],
        }
        schema = PlannerContract.model_json_schema()
        return self._chat_structured(
            system_prompt=(
                "You are the planner for a deep research copilot. Return valid JSON only "
                "that conforms to the supplied schema."
            ),
            user_payload=payload,
            schema=schema,
            response_model=PlannerContract,
        )

    def assess_report(
        self,
        report: ResearchReport,
        evidence: Sequence[EvidenceItem],
        plan: Sequence[PlanItem],
        *,
        revision_count: int = 0,
        max_revisions: int = 2,
    ) -> tuple[VerificationContract, ModelUsage]:
        payload = {
            "report": report.model_dump(),
            "evidence": [item.model_dump() for item in evidence[:20]],
            "plan": [item.model_dump() for item in plan[:12]],
            "revision_count": revision_count,
            "max_revisions": max_revisions,
        }
        return self._chat_structured(
            system_prompt=(
                "You are the verifier for a research copilot. Return valid JSON only "
                "that conforms to the supplied schema."
            ),
            user_payload=payload,
            schema=VerificationContract.model_json_schema(),
            response_model=VerificationContract,
        )

    def compose_report(
        self,
        topic: str,
        sections: Sequence[ReportSection],
        evidence: Sequence[EvidenceItem],
        confidence: float,
    ) -> tuple[ReporterContract, ModelUsage]:
        evidence_index = [
            {
                "index": index,
                "title": item.title,
                "source": item.source,
                "kind": item.kind,
                "url": item.url,
                "snippet": item.snippet,
                "content": (item.content or "")[:1200],
                "score": item.score,
            }
            for index, item in enumerate(evidence[:24], start=1)
        ]
        payload = {
            "topic": topic,
            "sections": [section.model_dump() for section in sections[:8]],
            "evidence_index": evidence_index,
            "confidence": confidence,
            "instructions": (
                "Synthesize the final report sections from the draft sections and evidence. "
                "Use the same language as the topic/request. Each section must be specific, "
                "citation-backed, and balanced. Use citation_indexes to reference only the "
                "provided evidence_index entries. Do not invent sources, URLs, facts, or citations."
            ),
        }
        return self._chat_structured(
            system_prompt=(
                "You are the final report writer for a deep research copilot, following the "
                "Open Deep Research pattern of synthesizing compressed findings into a "
                "comprehensive citation-backed report. Return valid JSON only that conforms "
                "to the supplied schema. Populate sections with rewritten section drafts and "
                "citation_indexes that map to the provided evidence_index."
            ),
            user_payload=payload,
            schema=ReporterContract.model_json_schema(),
            response_model=ReporterContract,
        )

    def embed_text(self, text: str) -> tuple[list[float], ModelUsage]:
        start = time.perf_counter()
        payload = {"model": self.embedding_model, "input": text, "dimensions": self.embedding_dimensions}
        with self._client() as client:
            response = client.post("/embeddings", json=payload)
            response.raise_for_status()
            body = response.json()
        vector = list(body["data"][0]["embedding"])
        usage = self._usage_from_body(body, start, self.embedding_model)
        return vector, usage

    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], ModelUsage]:
        start = time.perf_counter()
        payload = {"model": self.embedding_model, "input": list(texts), "dimensions": self.embedding_dimensions}
        with self._client() as client:
            response = client.post("/embeddings", json=payload)
            response.raise_for_status()
            body = response.json()
        vectors = [list(item["embedding"]) for item in body["data"]]
        usage = self._usage_from_body(body, start, self.embedding_model)
        return vectors, usage

    def _chat_structured(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema: dict[str, Any],
        response_model: type[TContract],
    ) -> tuple[TContract, ModelUsage]:
        start = time.perf_counter()
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "schema": schema,
                            "input": user_payload,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        with self._client() as client:
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
        content = _extract_chat_content(body)
        model = response_model.model_validate_json(_extract_json_object(content))
        usage = self._usage_from_body(body, start, self.chat_model)
        return model, usage

    def _client(self) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout_seconds,
        )

    def _usage_from_body(self, body: dict[str, Any], start: float, model: str) -> ModelUsage:
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ModelUsage(
            provider=self.name,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            cost_usd=0.0,
            latency_ms=latency_ms,
        )


def build_model_provider(settings: Any) -> ResearchModelProvider:
    if getattr(settings, "model_provider", "deterministic") == "openai_compatible" and getattr(settings, "model_base_url", ""):
        return OpenAICompatibleResearchModelProvider(
            base_url=settings.model_base_url,
            api_key=getattr(settings, "model_api_key", ""),
            chat_model=getattr(settings, "model_chat_model", "gpt-4o-mini"),
            embedding_model=getattr(settings, "model_embedding_model", "text-embedding-3-small"),
            timeout_seconds=float(getattr(settings, "model_timeout_seconds", 30.0)),
            temperature=float(getattr(settings, "model_temperature", 0.2)),
            embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
        )
    return DeterministicResearchModelProvider(
        embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
    )


def build_embedding_provider(settings: Any, model_provider: ResearchModelProvider | None = None) -> ResearchModelProvider:
    if getattr(settings, "embedding_provider", "model") == "deterministic":
        return DeterministicResearchModelProvider(
            embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
        )
    if getattr(settings, "embedding_provider", "model") == "openai_compatible":
        return OpenAICompatibleResearchModelProvider(
            base_url=getattr(settings, "embedding_base_url", "") or getattr(settings, "model_base_url", ""),
            api_key=getattr(settings, "embedding_api_key", "") or getattr(settings, "model_api_key", ""),
            chat_model=getattr(settings, "model_chat_model", "gpt-4o-mini"),
            embedding_model=getattr(
                settings,
                "embedding_model",
                getattr(settings, "model_embedding_model", "text-embedding-3-small"),
            ),
            timeout_seconds=float(getattr(settings, "model_timeout_seconds", 30.0)),
            temperature=float(getattr(settings, "model_temperature", 0.2)),
            embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
        )
    return model_provider or build_model_provider(settings)


def _build_plan_items(request: ResearchRequest, topic: str, *, revision_count: int = 0) -> list[PlanItem]:
    base = topic.strip()
    seed = hashlib.md5(base.encode("utf-8")).hexdigest()[:6]
    plan_items = [
        (
            "problem",
            f"What is the core problem behind {base}?",
            "Frame the motivation and scope.",
        ),
        (
            "workflow",
            f"How does the end-to-end workflow for {base} operate?",
            "Map the execution path and orchestration.",
        ),
        (
            "data",
            f"What evidence, memory, or retrieval sources support {base}?",
            "Explain the knowledge layer and context reuse.",
        ),
        (
            "risk",
            f"What are the main failure modes and trade-offs of {base}?",
            "Surface risks and quality constraints.",
        ),
    ]
    if request.depth in {"standard", "deep"}:
        plan_items.append(
            (
                "verification",
                f"How should {base} be verified, evaluated, and replayed?",
                "Connect verification, observability, and replay.",
            )
        )
    if request.depth == "deep":
        plan_items.append(
            (
                "delivery",
                f"What is needed to ship and operate {base} as a usable product?",
                "Outline deployment and operating considerations.",
            )
        )
    if revision_count > 0:
        plan_items.append(
            (
                "repair",
                f"What gaps must be repaired before the answer for {base} can be published?",
                "Force the supervisor to close verification gaps.",
            )
        )

    return [
        PlanItem(
            id=f"{seed}-{name}",
            question=question,
            purpose=purpose,
            search_query=f"{base} {name} {purpose}",
            revision_hint="repair citation gaps" if name == "repair" else None,
        )
        for name, question, purpose in plan_items[: request.max_sections]
    ]


def _focus_for_topic(topic: str) -> str:
    lower = topic.lower()
    if any(word in lower for word in ("agent", "copilot", "workflow")):
        return "agent orchestration, memory, verification, and observability"
    if any(word in lower for word in ("rag", "retrieval", "knowledge")):
        return "retrieval quality, grounding, and context management"
    if any(word in lower for word in ("e-commerce", "order", "payment", "marketing")):
        return "transactional reliability, state consistency, and recovery"
    return "system design, execution flow, and measurable outcomes"


def _summarize_memory(records: Sequence[MemoryRecord]) -> str:
    if not records:
        return ""
    summary_bits: list[str] = []
    for record in records[:3]:
        summary_bits.append(f"{record.key}: {record.value[:72].rstrip()}")
    return "; ".join(summary_bits)


def _hashed_dense_vector(text: str, dimensions: int) -> list[float]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    if not tokens:
        return [0.0] * dimensions
    values = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, 32, 4):
            bucket = int.from_bytes(digest[offset : offset + 4], "big") % dimensions
            weight = 1.0 if offset == 0 else 0.35
            values[bucket] += weight
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        return values
    return [value / norm for value in values]


def _extract_chat_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("OpenAI-compatible response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenAI-compatible response did not include message content.")
    return content


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped
