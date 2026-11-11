from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import APP_DB, DATA_DIR
from src.db import connect, create_app_schema


KB_ROOT = DATA_DIR / "kb"
MANIFEST = KB_ROOT / "manifest.json"


def _title_and_content(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8").strip()
    first, _, rest = content.partition("\n")
    title = first.removeprefix("# ").strip()
    return title, rest.strip()


def _chunks(content: str, size: int = 700) -> list[str]:
    paragraphs = [item.strip() for item in content.split("\n\n") if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > size:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _version_entries(manifest: dict[str, object], version: str) -> list[dict[str, str]]:
    versions = manifest["versions"]
    config = versions[version]
    if version == "healthy":
        return [dict(item, status="active") for item in config["files"]]

    inherited = {
        item["document_id"]: dict(item, status="active")
        for item in versions[config["inherits"]]["files"]
    }
    for override in config["overrides"]:
        inherited[override["document_id"]] = dict(override)
    return list(inherited.values())


def seed_knowledge_base() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with connect(APP_DB) as connection:
        for version in ("healthy", "stale"):
            effective_date = manifest["versions"][version]["effective_date"]
            for entry in _version_entries(manifest, version):
                source = KB_ROOT / entry["path"]
                title, content = _title_and_content(source)
                status = entry.get("status", "active")
                connection.execute(
                    """
                    INSERT INTO kb_documents(
                        document_id, kb_version, title, category, effective_date,
                        status, source_path, content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry["document_id"],
                        version,
                        title,
                        entry["category"],
                        effective_date,
                        status,
                        entry["path"],
                        content,
                    ),
                )
                # The stale return policy remains auditable in kb_documents but is
                # intentionally excluded from the active retrieval index.
                if status != "active":
                    continue
                for index, chunk in enumerate(_chunks(content), start=1):
                    chunk_id = f"{entry['document_id']}:{version}:{index:02d}"
                    row = (
                        chunk_id,
                        entry["document_id"],
                        version,
                        title,
                        entry["category"],
                        chunk,
                    )
                    connection.execute(
                        """
                        INSERT INTO kb_chunks(
                            chunk_id, document_id, kb_version, title, category, content
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        row,
                    )
                    connection.execute(
                        """
                        INSERT INTO kb_chunks_fts(
                            chunk_id, document_id, kb_version, title, category, content
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        row,
                    )


def seed_customer_data() -> None:
    customers = [
        ("CUST-100", "standard", 1),
        ("CUST-200", "priority", 1),
        ("CUST-300", "standard", 0),
    ]
    orders = [
        ("ORD-1001", "CUST-100", "in transit", "2026-08-19T09:00:00Z"),
        ("ORD-2002", "CUST-200", "delivered", "2026-08-18T13:30:00Z"),
        ("ORD-3003", "CUST-100", "processing", "2026-08-20T02:00:00Z"),
    ]
    with connect(APP_DB) as connection:
        connection.executemany(
            "INSERT INTO customers(customer_id, segment, authenticated) VALUES (?, ?, ?)",
            customers,
        )
        connection.executemany(
            "INSERT INTO orders(order_id, customer_id, status, updated_at) VALUES (?, ?, ?, ?)",
            orders,
        )


def setup(reset: bool) -> None:
    if reset:
        if APP_DB.exists():
            APP_DB.unlink()
    create_app_schema()
    with connect(APP_DB) as connection:
        count = connection.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0]
    if count == 0:
        seed_knowledge_base()
        seed_customer_data()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the Module 13 manual optimization lab")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Recreate the local synthetic application database under data/.",
    )
    args = parser.parse_args()
    setup(reset=args.reset)
    print(f"Application database: {APP_DB}")
    print("Synthetic customers, orders, and the healthy knowledge base are ready.")


if __name__ == "__main__":
    main()

