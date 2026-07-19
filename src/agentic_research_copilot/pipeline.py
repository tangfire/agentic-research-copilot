from __future__ import annotations

import uuid

from .agents import PlannerAgent, ResearchAgent, ReporterAgent, VerifierAgent
from .memory import MemoryStore
from .retrieval import DocumentStore
from .schemas import EvidenceItem, ResearchRequest, ResearchRun, ReportSection
from .telemetry import TelemetryLog


class ResearchCopilot:
    def __init__(self) -> None:
        self.memory = MemoryStore()
        self.documents = DocumentStore()
        self.telemetry = TelemetryLog()
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.verifier = VerifierAgent()
        self.reporter = ReporterAgent()

    def run(self, request: ResearchRequest) -> ResearchRun:
        run_id = str(uuid.uuid4())
        self.telemetry.emit("run.start", request.topic)

        plan = self.planner.create_plan(request)
        evidence: list[EvidenceItem] = []

        for item in plan:
            evidence.extend(self.researcher.collect(item))

        if request.include_private_docs:
            evidence.extend(self.documents.search(request.topic))

        sections = [
            ReportSection(
                heading="Problem framing",
                content=f"Break down the topic: {request.topic}.",
                citations=evidence[:2],
            ),
            ReportSection(
                heading="Agent design",
                content="Planner, researcher, verifier, reporter, and memory work as a loop.",
                citations=evidence[2:4],
            ),
            ReportSection(
                heading="What to learn",
                content="Use this project to learn tool calling, RAG, memory, evaluation, and observability.",
                citations=evidence[4:6],
            ),
        ]

        confidence = 0.55 if evidence else 0.2
        report = self.reporter.build_report(request.topic, sections, evidence, confidence)
        issues = self.verifier.verify(report, evidence)

        if request.use_memory:
            self.memory.add(
                key=f"topic:{request.topic}",
                value=report.summary,
                tags=["topic", "summary"],
            )

        self.telemetry.emit("run.finish", request.topic)

        return ResearchRun(
            run_id=run_id,
            request=request,
            plan=plan,
            evidence=evidence,
            report=report,
            issues=issues,
            status="completed",
        )

