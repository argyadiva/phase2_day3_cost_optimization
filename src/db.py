from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.config import APP_DB


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def create_app_schema(path: Path = APP_DB) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                segment TEXT NOT NULL,
                authenticated INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            );
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                summary TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            );
            CREATE TABLE IF NOT EXISTS kb_documents (
                document_id TEXT NOT NULL,
                kb_version TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                effective_date TEXT NOT NULL,
                status TEXT NOT NULL,
                source_path TEXT NOT NULL,
                content TEXT NOT NULL,
                PRIMARY KEY(document_id, kb_version)
            );
            CREATE TABLE IF NOT EXISTS kb_chunks (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL UNIQUE,
                document_id TEXT NOT NULL,
                kb_version TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                kb_version UNINDEXED,
                title,
                category,
                content,
                tokenize='porter unicode61'
            );
            """
        )
