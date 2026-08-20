from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schemas import ResearchRun, RunTraceEvent


@dataclass
class ObservabilityPublishResult:
    provider: str = "none"
    enabled: bool = False
    configured: bool = False
    installed: bool = False
    published: bool = False
    trace_id: str | None = None
    trace_url: str | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservabilityPublisher:
    """Optional Langfuse sink; SQLite remains the local source of truth."""

    def __init__(self, settings: Any) -> None:
        self.provider = str(getattr(settings, "observability_provider", "none") or "none").lower()
        self.host = str(getattr(settings, "langfuse_host", "") or "").rstrip("/")
        self.environment = str(getattr(settings, "langfuse_environment", "local") or "local")
        self.release = str(getattr(settings, "langfuse_release", "research-desk") or "research-desk")
        self.capture_content = bool(getattr(settings, "langfuse_capture_content", False))
        self._client: Any | None = None
        self._installed = False
        self._import_error = ""
        public_key = str(getattr(settings, "langfuse_public_key", "") or "").strip()
        secret_key = str(getattr(settings, "langfuse_secret_key", "") or "").strip()
        self._configured = bool(public_key and secret_key)

        if self.provider != "langfuse":
            return
        try:
            from langfuse import Langfuse

            self._installed = True
        except Exception as exc:  # pragma: no cover - depends on optional extra
            self._import_error = str(exc)
            return

        if not self._configured:
            return
        try:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=self.host or "https://cloud.langfuse.com",
                environment=self.environment,
                release=self.release,
                additional_headers={"x-langfuse-ingestion-version": "4"},
            )
        except Exception as exc:  # pragma: no cover - provider SDK/runtime dependent
            self._import_error = str(exc)

    def status(self) -> dict[str, Any]:
        if self.provider == "none":
            reason = "未启用外部观测；本地 SQLite trace 和 replay 仍然可用。"
        elif not self._installed:
            reason = "未安装 Langfuse 可选依赖，请执行 pip install -e .[observability]。"
        elif not self._configured:
            reason = "Langfuse 未配置 public key / secret key；不会发送外部 trace。"
        elif self._client is None:
            reason = self._import_error or "Langfuse 客户端初始化失败。"
        else:
            reason = "Langfuse 已配置，运行完成后会发送 trace 和 evaluation 分数。"
        return {
            "provider": self.provider,
            "enabled": self._client is not None,
            "configured": self._configured,
            "installed": self._installed,
            "host": self.host or ("https://cloud.langfuse.com" if self.provider == "langfuse" else ""),
            "environment": self.environment,
            "release": self.release,
            "capture_content": self.capture_content,
            "reason": reason,
        }

    def publish_run(self, run: ResearchRun) -> ObservabilityPublishResult:
        result = ObservabilityPublishResult(
            provider=self.provider,
            enabled=self._client is not None,
            configured=self._configured,
            installed=self._installed,
        )
        if self._client is None:
            return result

        trace_id: str | None = None
        try:
            trace_context = {"trace_id": self._client.create_trace_id(seed=run.run_id)}
            with self._client.start_as_current_observation(
                trace_context=trace_context,
                name="research_run",
                as_type="chain",
                input=self._run_input(run),
                output=self._run_output(run),
                metadata={
                    "run_id": run.run_id,
                    "job_id": run.job_id,
                    "execution_mode": "specialist_worker",
                    "status": run.status,
                    "revision_count": run.revision_count,
                    "capture_content": self.capture_content,
                },
                level="ERROR" if run.status == "failed" else "DEFAULT",
                status_message=run.failure_reason or None,
            ):
                trace_id = self._client.get_current_trace_id()
                for event in run.trace:
                    self._publish_event(event)
                self._publish_scores(run, trace_id)
            self._client.flush()
            result.published = True
            result.trace_id = trace_id
            result.trace_url = self._client.get_trace_url(trace_id=trace_id) if trace_id else None
            return result
        except Exception as exc:  # pragma: no cover - provider/network dependent
            result.trace_id = trace_id
            result.error = str(exc)
            return result

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
            self._client.shutdown()
        except Exception:
            return

    def _publish_event(self, event: RunTraceEvent) -> None:
        kind = {
            "tool_call": "tool",
            "evaluation": "evaluator",
            "verification": "evaluator",
            "step": "span",
            "handoff": "agent",
            "checkpoint": "chain",
            "failure": "guardrail",
        }.get(event.kind, "span")
        with self._client.start_as_current_observation(
            name=f"{event.actor}.{event.kind}",
            as_type=kind,
            input=self._event_input(event),
            output=self._event_output(event),
            metadata={
                "actor": event.actor,
                "provider": event.provider,
                "model": event.model,
                "latency_ms": event.latency_ms,
                "tokens_in": event.tokens_in,
                "tokens_out": event.tokens_out,
                "cost_usd": event.cost_usd,
                "from_agent": event.from_agent,
                "to_agent": event.to_agent,
            },
            level="ERROR" if event.status == "failed" else "DEFAULT",
            status_message=event.message if event.status == "failed" else None,
        ):
            pass

    def _publish_scores(self, run: ResearchRun, trace_id: str | None) -> None:
        evaluation = run.evaluation
        if evaluation is None or not trace_id:
            return
        values = {
            "citation_precision": evaluation.citation_precision,
            "context_recall": evaluation.context_recall,
            "faithfulness_proxy": evaluation.faithfulness_proxy,
            "source_quality_score": evaluation.source_quality_score,
            "plan_coverage": evaluation.plan_coverage,
        }
        for name, value in values.items():
            self._client.create_score(
                name=name,
                value=float(value),
                trace_id=trace_id,
                data_type="NUMERIC",
            )

    @staticmethod
    def _run_input(run: ResearchRun) -> dict[str, Any]:
        return {
            "depth": run.request.depth,
            "plan_count": len(run.plan),
            "metadata": _safe_value(run.request.metadata),
        }

    @staticmethod
    def _run_output(run: ResearchRun) -> dict[str, Any]:
        report = run.report
        evaluation = run.evaluation
        return {
            "status": run.status,
            "source_count": report.source_count if report else 0,
            "evaluation": (
                {
                    "passed": evaluation.passed,
                    "citation_precision": evaluation.citation_precision,
                    "context_recall": evaluation.context_recall,
                    "faithfulness_proxy": evaluation.faithfulness_proxy,
                }
                if evaluation
                else None
            ),
        }

    def _event_input(self, event: RunTraceEvent) -> dict[str, Any]:
        payload = {
            "step": event.step,
            "tool_name": event.tool_name,
            "metadata": _safe_value(event.metadata),
        }
        if self.capture_content:
            payload["message"] = _trim(event.message, 500)
        return payload

    def _event_output(self, event: RunTraceEvent) -> dict[str, Any]:
        if not self.capture_content:
            return {"status": event.status}
        return {"message": _trim(event.message, 500), "status": event.status}


def _trim(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if not any(token in str(key).lower() for token in ("token", "secret", "password", "api_key", "authorization"))
        }
    if isinstance(value, list):
        return [_safe_value(item) for item in value[:20]]
    if isinstance(value, tuple):
        return [_safe_value(item) for item in value[:20]]
    if isinstance(value, str):
        return _trim(value, 800)
    return value
