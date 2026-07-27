from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock

from celery import Celery
from celery.signals import worker_shutdown

from .pipeline import ResearchCopilot
from .settings import AppSettings, load_settings, resolve_storage_path
from .storage import SQLiteStore


settings = load_settings()
_worker_copilot: ResearchCopilot | None = None
_worker_lock = RLock()

celery_app = Celery(
    "agentic_research_copilot",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_default_queue="agentic_research_copilot",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="agentic_research_copilot.execute_job")
def execute_job(job_id: str) -> dict[str, str]:
    current_settings = load_settings()
    _mark_job_starting(job_id, current_settings)
    copilot = _get_worker_copilot(current_settings)
    copilot._execute_job(job_id)
    return {"job_id": job_id, "status": "processed"}


def _get_worker_copilot(current_settings: AppSettings) -> ResearchCopilot:
    global _worker_copilot
    with _worker_lock:
        if _worker_copilot is None:
            _worker_copilot = ResearchCopilot(settings=current_settings)
        else:
            _worker_copilot.refresh_state()
        return _worker_copilot


def _mark_job_starting(job_id: str, current_settings: AppSettings) -> None:
    store = SQLiteStore(resolve_storage_path(current_settings.storage_path))
    job = store.load_job(job_id)
    if job is None or job.status != "queued":
        return
    store.save_job(
        job.model_copy(
            update={
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
        )
    )


@worker_shutdown.connect
def _close_worker_copilot(**_: object) -> None:
    global _worker_copilot
    with _worker_lock:
        if _worker_copilot is not None:
            _worker_copilot.close()
            _worker_copilot = None
