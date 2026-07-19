from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TelemetryEvent:
    kind: str
    message: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TelemetryLog:
    def __init__(self) -> None:
        self._events: list[TelemetryEvent] = []

    def emit(self, kind: str, message: str) -> None:
        self._events.append(TelemetryEvent(kind=kind, message=message))

    def all(self) -> list[TelemetryEvent]:
        return list(self._events)

