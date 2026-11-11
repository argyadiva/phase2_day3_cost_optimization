from __future__ import annotations

import re
from pathlib import Path

from src.config import APP_DB
from src.db import connect
from src.models import KnowledgeBaseVersion, RetrievalRecord


STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "if",
    "in", "is", "it", "me", "my", "of", "on", "the", "to", "what", "when",
    "with", "you", "your",
}


def build_fts_query(query: str) -> str:
    terms = [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", query)
        if token.casefold() not in STOPWORDS and len(token) > 1
    ]
    return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms[:12]))


def search(
    query: str,
    kb_version: KnowledgeBaseVersion,
    top_k: int = 3,
    expected_document_ids: list[str] | None = None,
    db_path: Path = APP_DB,
) -> tuple[list[dict[str, object]], RetrievalRecord]:
    if top_k < 1 or top_k > 5:
        raise ValueError("top_k must be between 1 and 5")
    fts_query = build_fts_query(query)
    if not fts_query:
        rows = []
    else:
        with connect(db_path) as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, title, category, content,
                       bm25(kb_chunks_fts) AS rank
                FROM kb_chunks_fts
                WHERE kb_chunks_fts MATCH ? AND kb_version = ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, kb_version, top_k),
            ).fetchall()

    results = [
        {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "title": row["title"],
            "category": row["category"],
            "text": row["content"],
            "score": round(1.0 / (1.0 + abs(float(row["rank"]))), 6),
        }
        for row in rows
    ]
    record = RetrievalRecord(
        query=query,
        kb_version=kb_version,
        chunk_ids=[str(item["chunk_id"]) for item in results],
        document_ids=[str(item["document_id"]) for item in results],
        scores=[float(item["score"]) for item in results],
        expected_document_ids=expected_document_ids or [],
    )
    return results, record

