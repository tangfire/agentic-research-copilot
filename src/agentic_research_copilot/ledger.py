from __future__ import annotations

from collections.abc import Iterable

from .schemas import ResearchJob, ResearchRun


class RunLedger:
    """In-memory ledger for research runs and replay.

    This keeps the run history easy to inspect while staying close to the
    checkpoint / replay ideas used in larger agent platforms.
    """

    def __init__(self) -> None:
        self._runs: dict[str, ResearchRun] = {}

    def record(self, run: ResearchRun) -> ResearchRun:
        self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> ResearchRun | None:
        return self._runs.get(run_id)

    def list(self) -> list[ResearchRun]:
        return sorted(
            self._runs.values(),
            key=lambda run: run.started_at or "",
            reverse=True,
        )

    def extend(self, runs: Iterable[ResearchRun]) -> None:
        for run in runs:
            self.record(run)


class JobLedger:
    """In-memory index for queued and completed research jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, ResearchJob] = {}

    def record(self, job: ResearchJob) -> ResearchJob:
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> ResearchJob | None:
        return self._jobs.get(job_id)

    def list(self) -> list[ResearchJob]:
        return sorted(
            self._jobs.values(),
            key=lambda job: job.queued_at,
            reverse=True,
        )

    def extend(self, jobs: Iterable[ResearchJob]) -> None:
        for job in jobs:
            self.record(job)
