from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TelemetryEvent:
    kind: str
    message: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str | None = None
    job_id: str | None = None
    actor: str | None = None
    step: str | None = None
    status: str | None = None
    from_agent: str | None = None
    to_agent: str | None = None
    tool_name: str | None = None
    provider: str | None = None
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryLog:
    def __init__(self) -> None:
        self._events: list[TelemetryEvent] = []

    def emit(self, kind: str, message: str, **metadata: Any) -> None:
        event = TelemetryEvent(
            kind=kind,
            message=message,
            run_id=metadata.pop("run_id", None),
            job_id=metadata.pop("job_id", None),
            actor=metadata.pop("actor", None),
            step=metadata.pop("step", None),
            status=metadata.pop("status", None),
            from_agent=metadata.pop("from_agent", None),
            to_agent=metadata.pop("to_agent", None),
            tool_name=metadata.pop("tool_name", None),
            provider=metadata.pop("provider", None),
            model=metadata.pop("model", None),
            tokens_in=int(metadata.pop("tokens_in", 0) or 0),
            tokens_out=int(metadata.pop("tokens_out", 0) or 0),
            cost_usd=float(metadata.pop("cost_usd", 0.0) or 0.0),
            latency_ms=int(metadata.pop("latency_ms", 0) or 0),
            metadata=metadata,
        )
        self._events.append(event)

    def all(self) -> list[TelemetryEvent]:
        return list(self._events)

    def by_kind(self, kind: str) -> list[TelemetryEvent]:
        return [event for event in self._events if event.kind == kind]
