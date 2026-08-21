from __future__ import annotations

from ..provider_base import ResearchModelProvider
from ..providers import StructuredOutputError, build_model_provider
from ..schemas import EvidenceItem, ResearchReport, ReportSection, ReporterContract
from ..settings import AppSettings, load_settings


class ReporterAgent:
    def __init__(
        self,
        model_provider: ResearchModelProvider | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.model_provider = model_provider or build_model_provider(self.settings)
        self.last_usage = None
        self.last_degraded_reason: str | None = None

    def compose(
        self,
        topic: str,
        sections: list[ReportSection],
        citations: list[EvidenceItem],
        confidence: float,
    ) -> ReporterContract:
        self.last_degraded_reason = None
        try:
            contract, usage = self.model_provider.compose_report(topic, sections, citations, confidence)
            self.last_usage = usage
            return contract
        except StructuredOutputError as exc:
            self.last_usage = None
            self.last_degraded_reason = str(exc)
            return ReporterContract(
                title=f"研究报告：{topic[:160]}",
                summary=(
                    "报告整理阶段的模型结构化输出暂时不可用，已保留研究阶段生成的证据和章节草稿。"
                    "请重点检查下方引用和证据；这不是重新生成的模型结论。"
                ),
                confidence=max(0.0, min(1.0, confidence)),
                source_index=[
                    f"[{index}] {item.title}"
                    for index, item in enumerate(citations[:12], start=1)
                ],
            )

    def build_report(
        self,
        topic: str,
        sections: list[ReportSection],
        citations: list[EvidenceItem],
        confidence: float,
    ) -> ResearchReport:
        contract = self.compose(topic, sections, citations, confidence)
        unique_sources: list[str] = []
        seen_sources: set[str] = set()
        unique_citations: list[EvidenceItem] = []
        seen_keys: set[str] = set()

        for citation in citations:
            source_key = citation.url or f"{citation.source}:{citation.title}"
            if source_key not in seen_keys:
                seen_keys.add(source_key)
                unique_citations.append(citation)
            if citation.source not in seen_sources:
                seen_sources.add(citation.source)
                unique_sources.append(citation.source)
        report_sections = self._build_synthesized_sections(contract, sections, list(citations))

        return ResearchReport(
            title=contract.title,
            summary=contract.summary,
            sections=report_sections,
            citations=unique_citations,
            confidence=contract.confidence,
            highlights=contract.highlights,
            recommendations=contract.recommendations,
            source_index=contract.source_index,
            source_count=len(unique_sources),
        )

    def _build_synthesized_sections(
        self,
        contract: ReporterContract,
        fallback_sections: list[ReportSection],
        citations: list[EvidenceItem],
    ) -> list[ReportSection]:
        if not contract.sections:
            return fallback_sections

        synthesized: list[ReportSection] = []
        for index, draft in enumerate(contract.sections):
            selected_citations = [
                citations[citation_index - 1]
                for citation_index in draft.citation_indexes
                if 1 <= citation_index <= len(citations)
            ]
            if not selected_citations and index < len(fallback_sections):
                selected_citations = fallback_sections[index].citations
            if not selected_citations:
                continue
            synthesized.append(
                ReportSection(
                    heading=draft.heading,
                    content=draft.content,
                    citations=selected_citations,
                    evidence_count=len(selected_citations),
                    source_summary=self._source_names(selected_citations),
                )
            )

        synthesized = self._preserve_runtime_contract_sections(synthesized, fallback_sections)
        return synthesized or fallback_sections

    def _preserve_runtime_contract_sections(
        self,
        synthesized: list[ReportSection],
        fallback_sections: list[ReportSection],
    ) -> list[ReportSection]:
        existing_headings = {section.heading.strip().lower() for section in synthesized}
        preserved = list(synthesized)
        constraint_headings = {"team constraint coverage", "团队约束覆盖"}
        for section in fallback_sections:
            heading = section.heading.strip().lower()
            if heading in constraint_headings and heading not in existing_headings:
                preserved.append(section)
                existing_headings.add(heading)
        return preserved

    @staticmethod
    def _source_names(items: list[EvidenceItem]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item.source and item.source not in seen:
                seen.add(item.source)
                names.append(item.source)
        return names
