from __future__ import annotations

from collections import defaultdict

from ..schemas import MemoryRecord


class MemoryStore:
    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    def add(self, key: str, value: str, tags: list[str] | None = None) -> MemoryRecord:
        record = MemoryRecord(key=key, value=value, tags=tags or [])
        self._records.append(record)
        return record

    def list(self) -> list[MemoryRecord]:
        return list(self._records)

    def search(self, query: str) -> list[MemoryRecord]:
        query_lower = query.lower()
        return [
            record
            for record in self._records
            if query_lower in record.key.lower() or query_lower in record.value.lower()
        ]

    def grouped_by_tag(self) -> dict[str, list[MemoryRecord]]:
        groups: dict[str, list[MemoryRecord]] = defaultdict(list)
        for record in self._records:
            for tag in record.tags:
                groups[tag].append(record)
        return dict(groups)

