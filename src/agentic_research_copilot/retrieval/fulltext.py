from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from threading import RLock


FTS_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "before",
    "between",
    "but",
    "can",
    "for",
    "from",
    "how",
    "into",
    "not",
    "that",
    "the",
    "this",
    "through",
    "with",
    "when",
    "where",
    "which",
    "while",
    "will",
}


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: str
    score: float
    rank: int
    raw_rank: float
    backend: str


class SQLiteBM25Index:
    """Local FTS5/BM25 keyword index for single-node hybrid retrieval."""

    backend = "sqlite_fts5_bm25"

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = RLock()
        with self._lock:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE chunks USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    title,
                    source,
                    text,
                    tokens,
                    tokenize='unicode61'
                )
                """
            )

    def add_chunk(
        self,
        *,
        chunk_id: str,
        document_id: str,
        title: str,
        source: str,
        text: str,
        tokens: tuple[str, ...],
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO chunks(chunk_id, document_id, title, source, text, tokens)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, title, source, text, " ".join(tokens)),
            )
            self._conn.commit()

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.commit()

    def search(self, query_tokens: tuple[str, ...], *, limit: int) -> dict[str, KeywordHit]:
        query = _fts5_query(query_tokens)
        if not query:
            return {}

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT chunk_id, bm25(chunks) AS bm25_rank
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY bm25_rank
                LIMIT ?
                """,
                (query, max(1, limit)),
            ).fetchall()
        if not rows:
            return {}

        raw_scores = [max(0.0, -float(rank)) for _chunk_id, rank in rows]
        max_score = max(raw_scores) or 1.0
        hits: dict[str, KeywordHit] = {}
        for index, ((chunk_id, raw_rank), raw_score) in enumerate(zip(rows, raw_scores), start=1):
            hits[str(chunk_id)] = KeywordHit(
                chunk_id=str(chunk_id),
                score=round(raw_score / max_score, 4),
                rank=index,
                raw_rank=round(float(raw_rank), 8),
                backend=self.backend,
            )
        return hits

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _fts5_query(tokens: tuple[str, ...]) -> str:
    terms: list[str] = []
    for token in tokens:
        if token in FTS_STOPWORDS:
            continue
        if len(token) < 2 and not _is_cjk_token(token):
            continue
        escaped = _escape_fts5_term(token)
        if not escaped or escaped in terms:
            continue
        terms.append(escaped)
        if len(terms) >= 24:
            break
    return " OR ".join(f'"{term}"' for term in terms)


def _escape_fts5_term(value: str) -> str:
    return re.sub(r'\s+', ' ', value.replace('"', '""').strip())


def _is_cjk_token(value: str) -> bool:
    return len(value) == 1 and "\u4e00" <= value <= "\u9fff"
