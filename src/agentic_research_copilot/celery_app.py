from __future__ import annotations

from celery import Celery

from .pipeline import ResearchCopilot
from .settings import load_settings


settings = load_settings()

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
    copilot = ResearchCopilot(settings=load_settings())
    try:
        copilot._execute_job(job_id)
        return {"job_id": job_id, "status": "processed"}
    finally:
        copilot.close()
