from __future__ import annotations

import importlib
from pathlib import Path

from agentic_research_copilot.schemas import ResearchJob, ResearchRequest
from agentic_research_copilot.settings import AppSettings
from agentic_research_copilot.storage import SQLiteStore


def test_mark_job_starting_updates_sqlite_before_worker_warmup(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    celery_app = importlib.import_module("agentic_research_copilot.celery_app")

    storage_path = tmp_path / "jobs.sqlite"
    store = SQLiteStore(storage_path)
    store.save_job(
        ResearchJob(
            job_id="job-starting-1",
            request=ResearchRequest(topic="worker startup visibility"),
            status="queued",
        )
    )

    celery_app._mark_job_starting(
        "job-starting-1",
        AppSettings(storage_path=str(storage_path)),
    )

    refreshed = store.load_job("job-starting-1")
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.started_at is not None
    assert refreshed.attempts == 0


def test_worker_copilot_is_reused_and_refreshed(monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    celery_app = importlib.import_module("agentic_research_copilot.celery_app")

    class FakeCopilot:
        instances: list["FakeCopilot"] = []

        def __init__(self, settings: AppSettings) -> None:
            self.settings = settings
            self.refresh_count = 0
            self.closed = False
            self.instances.append(self)

        def refresh_state(self) -> None:
            self.refresh_count += 1

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(celery_app, "ResearchCopilot", FakeCopilot)
    monkeypatch.setattr(celery_app, "_worker_copilot", None)

    settings = AppSettings()
    first = celery_app._get_worker_copilot(settings)
    second = celery_app._get_worker_copilot(settings)

    assert first is second
    assert len(FakeCopilot.instances) == 1
    assert first.refresh_count == 1
