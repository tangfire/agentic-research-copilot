from __future__ import annotations

from ..schemas import EvidenceItem


class DocumentStore:
    def __init__(self) -> None:
        self._docs: list[EvidenceItem] = []

    def add(self, title: str, source: str, url: str | None = None, snippet: str | None = None) -> EvidenceItem:
        doc = EvidenceItem(title=title, source=source, url=url, snippet=snippet, score=1.0)
        self._docs.append(doc)
        return doc

    def search(self, query: str, limit: int = 5) -> list[EvidenceItem]:
        query_lower = query.lower()
        matches = [
            doc
            for doc in self._docs
            if query_lower in doc.title.lower()
            or query_lower in doc.source.lower()
            or (doc.snippet and query_lower in doc.snippet.lower())
        ]
        return matches[:limit]

