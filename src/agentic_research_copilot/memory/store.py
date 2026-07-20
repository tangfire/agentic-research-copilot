from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Protocol

from ..schemas import MemoryRecord


class EmbeddingProviderLike(Protocol):
    name: str
    embedding_dimensions: int

    def embed_text(self, text: str): ...


class MemoryStore:
    def __init__(self, embedding_provider: EmbeddingProviderLike | None = None) -> None:
        self._records: list[MemoryRecord] = []
        self.embedding_provider = embedding_provider
        self._embeddings: dict[str, list[float]] = {}

    def add(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
        *,
        layer: str = "session",
        run_id: str | None = None,
        session_id: str | None = None,
        topic: str | None = None,
        confidence: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> MemoryRecord:
        governed_metadata = self._governance_metadata(
            key=key,
            value=value,
            layer=layer,
            topic=topic,
            confidence=confidence,
            metadata=metadata or {},
        )
        record = MemoryRecord(
            key=key,
            value=value,
            layer=layer,  # type: ignore[arg-type]
            tags=tags or [],
            run_id=run_id,
            session_id=session_id,
            topic=topic,
            confidence=confidence,
            metadata=governed_metadata,
        )
        self._records.append(record)
        self._index_record(record)
        return record

    def add_session_note(
        self,
        key: str,
        value: str,
        *,
        tags: list[str] | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        topic: str | None = None,
        confidence: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> MemoryRecord:
        return self.add(
            key,
            value,
            tags,
            layer="session",
            run_id=run_id,
            session_id=session_id,
            topic=topic,
            confidence=confidence,
            metadata=metadata,
        )

    def add_fact(
        self,
        key: str,
        value: str,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        topic: str | None = None,
        confidence: float = 0.0,
    ) -> MemoryRecord:
        return self.add(
            key,
            value,
            tags,
            layer="canonical",
            run_id=run_id,
            session_id=session_id,
            topic=topic,
            confidence=confidence,
            metadata=metadata,
        )

    def add_summary(
        self,
        key: str,
        value: str,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        topic: str | None = None,
        confidence: float = 0.0,
    ) -> MemoryRecord:
        return self.add(
            key,
            value,
            tags,
            layer="summary",
            run_id=run_id,
            session_id=session_id,
            topic=topic,
            confidence=confidence,
            metadata=metadata,
        )

    def list(
        self,
        *,
        layer: str | None = None,
        topic: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        records = self._records
        if layer is not None:
            records = [record for record in records if record.layer == layer]
        if topic is not None:
            topic_lower = topic.lower()
            records = [
                record
                for record in records
                if topic_lower in (record.topic or "").lower()
                or topic_lower in record.key.lower()
                or topic_lower in record.value.lower()
            ]
        if run_id is not None:
            records = [record for record in records if record.run_id == run_id]
        if session_id is not None:
            records = [record for record in records if record.session_id == session_id]
        if tag is not None:
            tag_lower = tag.lower()
            records = [
                record
                for record in records
                if any(tag_lower == record_tag.lower() for record_tag in record.tags)
            ]
        records = sorted(records, key=lambda record: record.created_at, reverse=True)
        return records[:limit] if limit is not None else records

    def extend(self, records: list[MemoryRecord]) -> None:
        known = {
            (record.key, record.value, record.layer, record.run_id, record.session_id, record.topic, record.created_at)
            for record in self._records
        }
        for record in records:
            identity = (
                record.key,
                record.value,
                record.layer,
                record.run_id,
                record.session_id,
                record.topic,
                record.created_at,
            )
            if identity not in known:
                self._records.append(record)
                self._index_record(record)
                known.add(identity)

    def recall(
        self,
        query: str,
        *,
        layer: str | None = None,
        topic: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        query_lower = query.lower().strip()
        tokens = [token for token in query_lower.split() if token]
        query_embedding = self._embed_text(query) if query_lower else None
        scored_records: list[tuple[float, MemoryRecord, dict[str, float]]] = []

        for record in self.list(layer=layer, topic=topic, run_id=run_id, session_id=session_id):
            haystack = " ".join([record.key, record.value, " ".join(record.tags), record.topic or ""]).lower()
            lexical_score = 0.0
            if query_lower and query_lower in haystack:
                lexical_score += 2.5
            for token in tokens:
                lexical_score += haystack.count(token) * 0.45
            if tags:
                match_count = sum(1 for tag in tags if tag.lower() in (tag_value.lower() for tag_value in record.tags))
                lexical_score += match_count * 0.5
            semantic_score = self._semantic_score(record, query_embedding)
            quality_score = self._quality_score(record)
            layer_bonus = 0.0
            if record.layer == "canonical":
                layer_bonus += 0.35
            elif record.layer == "summary":
                layer_bonus += 0.25
            else:
                layer_bonus += 0.15
            governance_adjustment = self._governance_adjustment(record)
            if record.topic and topic and record.topic.lower() == topic.lower():
                layer_bonus += 0.4
            if record.run_id and run_id and record.run_id == run_id:
                layer_bonus += 0.5
            if record.session_id and session_id and record.session_id == session_id:
                layer_bonus += 0.5
            score = (
                lexical_score
                + semantic_score * 1.25
                + quality_score * 0.35
                + layer_bonus
                + governance_adjustment
            )
            if score > 0:
                scored_records.append(
                    (
                        score,
                        record,
                        {
                            "lexical": lexical_score,
                            "semantic": semantic_score,
                            "quality": quality_score,
                            "layer": layer_bonus,
                            "governance": governance_adjustment,
                        },
                    )
                )

        if not scored_records:
            fallback = self.list(layer=layer, topic=topic, run_id=run_id, session_id=session_id)
            return self._touch_recalled(fallback[:limit])

        scored_records.sort(key=lambda item: (-item[0], item[1].created_at))
        recalled = []
        for score, record, components in scored_records[:limit]:
            record.metadata["last_recall_score"] = round(score, 4)
            record.metadata["last_recall_lexical_score"] = round(components["lexical"], 4)
            record.metadata["last_recall_semantic_score"] = round(components["semantic"], 4)
            record.metadata["last_recall_quality_score"] = round(components["quality"], 4)
            record.metadata["last_recalled_at"] = _utc_now()
            record.metadata["recall_count"] = int(record.metadata.get("recall_count", 0) or 0) + 1
            recalled.append(record)
        return recalled

    def search(self, query: str) -> list[MemoryRecord]:
        return self.recall(query)

    def grouped_by_tag(self) -> dict[str, list[MemoryRecord]]:
        groups: dict[str, list[MemoryRecord]] = defaultdict(list)
        for record in self._records:
            for tag in record.tags:
                groups[tag].append(record)
        return dict(groups)

    def grouped_by_layer(self) -> dict[str, list[MemoryRecord]]:
        groups: dict[str, list[MemoryRecord]] = defaultdict(list)
        for record in self._records:
            groups[record.layer].append(record)
        return dict(groups)

    def governance_report(self) -> dict[str, object]:
        by_layer = {layer: len(records) for layer, records in self.grouped_by_layer().items()}
        review_records = [
            record
            for record in self._records
            if record.metadata.get("governance_status") == "needs_review"
        ]
        conflict_records = [
            record
            for record in self._records
            if int(record.metadata.get("conflict_count", 0) or 0) > 0
        ]
        return {
            "total_records": len(self._records),
            "by_layer": by_layer,
            "canonical_count": by_layer.get("canonical", 0),
            "embedding_enabled": self.embedding_provider is not None,
            "embedding_provider": getattr(self.embedding_provider, "name", None),
            "embedding_dimensions": getattr(self.embedding_provider, "embedding_dimensions", 0),
            "indexed_memory_count": len(self._embeddings),
            "needs_review_count": len(review_records),
            "conflict_record_count": len(conflict_records),
            "conflict_count": sum(
                int(record.metadata.get("conflict_count", 0) or 0)
                for record in conflict_records
            ),
            "review_required": [
                {
                    "key": record.key,
                    "layer": record.layer,
                    "topic": record.topic,
                    "confidence": record.confidence,
                    "governance_status": record.metadata.get("governance_status"),
                    "conflicts_with": record.metadata.get("conflicts_with", []),
                    "write_policy": record.metadata.get("write_policy"),
                }
                for record in review_records
            ],
            "write_rules": [
                "session memory captures run-scoped notes",
                "summary memory captures topic-level takeaways",
                "canonical memory is active only when confidence is sufficient and no conflict is detected",
                "conflicting canonical facts are retained but marked needs_review instead of overwritten",
            ],
            "recall_strategy": [
                "lexical token matching",
                "embedding-assisted semantic similarity when an embedding provider is configured",
                "confidence and layer weighting",
                "governance penalty for needs_review canonical records",
            ],
        }

    def _governance_metadata(
        self,
        *,
        key: str,
        value: str,
        layer: str,
        topic: str | None,
        confidence: float,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        governed = dict(metadata)
        if layer != "canonical":
            return governed

        governed.setdefault(
            "write_policy",
            "canonical memories require citation-backed runs or explicit API/user writes",
        )
        conflicts = [
            self._record_identity(record)
            for record in self._records
            if record.layer == "canonical"
            and self._same_canonical_scope(record, key=key, topic=topic)
            and self._normalize(record.value) != self._normalize(value)
        ]
        governed["conflicts_with"] = conflicts
        governed["conflict_count"] = len(conflicts)
        human_confirmed = bool(governed.get("human_confirmed"))
        if conflicts and not human_confirmed:
            governed["governance_status"] = "needs_review"
            governed["governance_reason"] = "canonical_conflict"
        elif confidence and confidence < 0.6 and not human_confirmed:
            governed["governance_status"] = "needs_review"
            governed["governance_reason"] = "low_confidence"
        else:
            governed.setdefault("governance_status", "active")
            governed.setdefault("governance_reason", "auto_write")
        return governed

    def _same_canonical_scope(self, record: MemoryRecord, *, key: str, topic: str | None) -> bool:
        if record.key == key:
            return True
        if topic and record.topic and record.topic.strip().lower() == topic.strip().lower():
            return True
        return False

    def _record_identity(self, record: MemoryRecord) -> str:
        return ":".join(
            part
            for part in [record.key, record.topic or "", record.run_id or "", record.created_at]
            if part
        )

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().split())

    def _index_record(self, record: MemoryRecord) -> None:
        embedding = self._embed_text(self._record_text(record))
        if embedding:
            self._embeddings[self._record_identity(record)] = embedding

    def _embed_text(self, text: str) -> list[float] | None:
        if self.embedding_provider is None or not text.strip():
            return None
        try:
            embedding, _usage = self.embedding_provider.embed_text(text)
            return [float(value) for value in embedding]
        except Exception:
            return None

    def _semantic_score(self, record: MemoryRecord, query_embedding: list[float] | None) -> float:
        if not query_embedding:
            return 0.0
        record_embedding = self._embeddings.get(self._record_identity(record))
        if not record_embedding:
            record_embedding = self._embed_text(self._record_text(record))
            if record_embedding:
                self._embeddings[self._record_identity(record)] = record_embedding
        return _cosine_similarity(query_embedding, record_embedding or [])

    def _quality_score(self, record: MemoryRecord) -> float:
        scores = [record.confidence]
        for key in ("quality_score", "evaluator_quality", "faithfulness", "accuracy"):
            value = record.metadata.get(key)
            if isinstance(value, (int, float)):
                scores.append(float(value))
        return max(0.0, min(1.0, max(scores or [0.0])))

    def _governance_adjustment(self, record: MemoryRecord) -> float:
        if record.metadata.get("governance_status") == "needs_review":
            return -0.35
        if record.layer == "canonical" and record.metadata.get("governance_status") == "active":
            return 0.1
        return 0.0

    def _record_text(self, record: MemoryRecord) -> str:
        return " ".join(
            part
            for part in [
                record.key,
                record.value,
                record.layer,
                record.topic or "",
                " ".join(record.tags),
            ]
            if part
        )

    def _touch_recalled(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        for record in records:
            record.metadata["last_recalled_at"] = _utc_now()
            record.metadata["recall_count"] = int(record.metadata.get("recall_count", 0) or 0) + 1
        return records


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        size = min(len(left), len(right))
        left = left[:size]
        right = right[:size]
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(0.0, min(1.0, sum(lv * rv for lv, rv in zip(left, right)) / (left_norm * right_norm)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
