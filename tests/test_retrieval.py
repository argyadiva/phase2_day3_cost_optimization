from __future__ import annotations

from src.config import APP_DB
from src.db import connect
from src.retrieval import search


def test_setup_creates_fts_index_and_documents() -> None:
    with connect(APP_DB) as connection:
        document_count = connection.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM kb_chunks_fts").fetchone()[0]
    assert document_count == 20
    assert fts_count > 10


def test_stale_return_version_produces_repeatable_retrieval_miss() -> None:
    _, healthy = search(
        "current return window unused item",
        "healthy",
        expected_document_ids=["KB-RET-001"],
    )
    _, stale = search(
        "current return window unused item",
        "stale",
        expected_document_ids=["KB-RET-001"],
    )
    assert healthy.required_document_hit == 1.0
    assert stale.required_document_hit == 0.0


def test_search_rejects_invalid_top_k() -> None:
    for top_k in (0, 6):
        try:
            search("return policy", "healthy", top_k=top_k)
        except ValueError as exc:
            assert "between 1 and 5" in str(exc)
        else:
            raise AssertionError("Invalid top-k should be rejected")
