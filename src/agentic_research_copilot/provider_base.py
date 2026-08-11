from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .schemas import (
    ChunkContextContract,
    ClarificationContract,
    CorpusProfile,
    EvidenceItem,
    KnowledgeGraphExtractionContract,
    KnowledgeGraphQueryContract,
    MCPToolDescriptor,
    PlanItem,
    PlannerContract,
    ReporterContract,
    ResearchRequest,
    ResearcherToolDecisionContract,
    ResearchReport,
    ReportSection,
    RetrievalRoute,
    SourceCompressionContract,
    SupervisorDecisionContract,
    VerificationContract,
)


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

    def clarify_request(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
    ) -> tuple[ClarificationContract, ModelUsage]: ...

    def decide_researcher_action(
        self,
        *,
        item: PlanItem,
        available_tools: Sequence[str],
        previous_queries: Sequence[str],
        evidence: Sequence[EvidenceItem],
        gaps: Sequence[str],
        iteration: int,
        max_iterations: int,
        mcp_tools: Sequence[MCPToolDescriptor] = (),
    ) -> tuple[ResearcherToolDecisionContract, ModelUsage]: ...

    def draft_plan(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]: ...

    def supervise_research(
        self,
        request: ResearchRequest,
        research_brief: str,
        plan: Sequence[PlanItem],
        retrieval_routes: Sequence[RetrievalRoute],
        corpus_profile: CorpusProfile,
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[SupervisorDecisionContract, ModelUsage]: ...

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

    def compress_source(
        self,
        *,
        query: str,
        title: str,
        url: str | None,
        raw_content: str,
    ) -> tuple[SourceCompressionContract, ModelUsage]: ...

    def contextualize_chunk(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, object],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> tuple[ChunkContextContract, ModelUsage]: ...

    def extract_knowledge_graph(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, object],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        max_entities: int,
        max_relationships: int,
    ) -> tuple[KnowledgeGraphExtractionContract, ModelUsage]: ...

    def extract_graph_query(
        self,
        *,
        query: str,
        max_local_keywords: int,
        max_global_keywords: int,
    ) -> tuple[KnowledgeGraphQueryContract, ModelUsage]: ...

    def embed_text(self, text: str) -> tuple[list[float], ModelUsage]: ...
    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], ModelUsage]: ...
