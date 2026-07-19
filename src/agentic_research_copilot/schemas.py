from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3)
    audience: str = "recruiter"
    depth: Literal["quick", "standard", "deep"] = "standard"
    include_private_docs: bool = True
    use_memory: bool = True


class PlanItem(BaseModel):
    id: str
    question: str
    purpose: str
    status: Literal["pending", "running", "done"] = "pending"


class EvidenceItem(BaseModel):
    title: str
    source: str
    url: str | None = None
    snippet: str | None = None
    score: float = 0.0


class MemoryRecord(BaseModel):
    key: str
    value: str
    tags: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    heading: str
    content: str
    citations: list[EvidenceItem] = Field(default_factory=list)


class ResearchReport(BaseModel):
    title: str
    summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    citations: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = 0.0


class ResearchRun(BaseModel):
    run_id: str
    request: ResearchRequest
    plan: list[PlanItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    report: ResearchReport | None = None
    issues: list[str] = Field(default_factory=list)
    status: Literal["queued", "running", "completed", "failed"] = "queued"
