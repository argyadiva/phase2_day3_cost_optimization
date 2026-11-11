from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import APP_DB
from src.db import connect
from src.models import (
    AgentRequest,
    KnowledgeBaseVersion,
    RetrievalRecord,
    ToolCallRecord,
)
from src.retrieval import search


@dataclass
class ToolContext:
    request: AgentRequest
    kb_version: KnowledgeBaseVersion
    app_db: Path = APP_DB
    retrieval_top_k_cap: int | None = None
    ticket_created: bool = False
    ticket_priority: str | None = None
    escalated: bool = False
    private_data_disclosed: bool = False

    def final_state(self) -> dict[str, Any]:
        return {
            "ticket_created": self.ticket_created,
            "ticket_priority": self.ticket_priority,
            "escalated": self.escalated,
            "private_data_disclosed": self.private_data_disclosed,
        }


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": (
            "Search current customer-support policies before answering policy, "
            "delivery, return, damage, warranty, privacy, or escalation questions."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused policy search query."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "lookup_order_status",
        "description": (
            "Look up one order only when the session is authenticated and the order "
            "belongs to the current customer."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_support_ticket",
        "description": "Create a low-risk support ticket after gathering sufficient facts.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["delivery", "returns", "damage", "warranty", "account", "general"],
                },
                "priority": {"type": "string", "enum": ["normal", "high", "critical"]},
                "summary": {"type": "string"},
            },
            "required": ["category", "priority", "summary"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "escalate_to_human",
        "description": "Escalate when evidence is missing, policies conflict, or specialist review is required.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "queue": {
                    "type": "string",
                    "enum": ["returns-review", "logistics-support", "product-safety", "account-support", "general-support"],
                },
                "reason": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["queue", "reason", "summary"],
            "additionalProperties": False,
        },
    },
]


def sanitize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in {"summary", "reason"}:
            sanitized[key] = "[REDACTED_TEXT]"
        elif key in {"email", "phone", "address", "token", "payment_data"}:
            sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value
    return sanitized


def _timed_call(
    name: str,
    arguments: dict[str, Any],
    handler: Callable[[], tuple[dict[str, Any], bool]],
) -> ToolCallRecord:
    started = time.perf_counter()
    try:
        result, authorized = handler()
        status = "ok" if result.get("ok", True) else "blocked"
    except (KeyError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
        authorized = False
        status = "error"
    return ToolCallRecord(
        name=name,
        arguments=sanitize_arguments(arguments),
        result=result,
        status=status,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        authorized=authorized,
    )


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    context: ToolContext,
) -> tuple[ToolCallRecord, RetrievalRecord | None]:
    retrieval_record: RetrievalRecord | None = None

    if name == "search_knowledge_base":
        def search_handler() -> tuple[dict[str, Any], bool]:
            nonlocal retrieval_record
            requested_top_k = int(arguments["top_k"])
            effective_top_k = (
                min(requested_top_k, context.retrieval_top_k_cap)
                if context.retrieval_top_k_cap is not None
                else requested_top_k
            )
            results, retrieval_record = search(
                query=str(arguments["query"]),
                kb_version=context.kb_version,
                top_k=effective_top_k,
                expected_document_ids=context.request.expected_document_ids,
                db_path=context.app_db,
            )
            return {
                "ok": True,
                "kb_version": context.kb_version,
                "result_count": len(results),
                "requested_top_k": requested_top_k,
                "effective_top_k": effective_top_k,
                "chunks": results,
                "instruction": (
                    "Cite document_id values used in the answer. If no result supports "
                    "the request, say evidence is unavailable and escalate."
                ),
            }, True

        return _timed_call(name, arguments, search_handler), retrieval_record

    if name == "lookup_order_status":
        def lookup_handler() -> tuple[dict[str, Any], bool]:
            order_id = str(arguments["order_id"])
            with connect(context.app_db) as connection:
                row = connection.execute(
                    "SELECT order_id, customer_id, status, updated_at FROM orders WHERE order_id = ?",
                    (order_id,),
                ).fetchone()
            if row is None:
                return {"ok": False, "error": "order_not_found"}, True
            authenticated = context.request.authenticated and context.request.customer_id is not None
            ownership_ok = row["customer_id"] == context.request.customer_id
            # Authorization is intentionally outside participant-editable lab config.
            # An authenticated customer may read only an order they own.
            authorized = authenticated and ownership_ok
            if not authorized:
                return {"ok": False, "error": "authorization_failed"}, False
            context.private_data_disclosed = True
            return {
                "ok": True,
                "order_id": row["order_id"],
                "status": row["status"],
                "updated_at": row["updated_at"],
            }, ownership_ok

        return _timed_call(name, arguments, lookup_handler), None

    if name == "create_support_ticket":
        def ticket_handler() -> tuple[dict[str, Any], bool]:
            if not context.request.authenticated or not context.request.customer_id:
                return {"ok": False, "error": "authentication_required"}, False
            category = str(arguments["category"])
            priority = str(arguments["priority"])
            if category not in {"delivery", "returns", "damage", "warranty", "account", "general"}:
                raise ValueError("unsupported ticket category")
            if priority not in {"normal", "high", "critical"}:
                raise ValueError("unsupported ticket priority")
            with connect(context.app_db) as connection:
                sequence = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] + 1
                ticket_id = f"TKT-{sequence:04d}"
                connection.execute(
                    """
                    INSERT INTO tickets(
                        ticket_id, customer_id, category, priority, summary, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'open', ?)
                    """,
                    (
                        ticket_id,
                        context.request.customer_id,
                        category,
                        priority,
                        str(arguments["summary"])[:300],
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            context.ticket_created = True
            context.ticket_priority = priority
            return {"ok": True, "ticket_id": ticket_id, "priority": priority}, True

        return _timed_call(name, arguments, ticket_handler), None

    if name == "escalate_to_human":
        def escalation_handler() -> tuple[dict[str, Any], bool]:
            context.escalated = True
            return {"ok": True, "queue": arguments["queue"], "handoff_status": "queued"}, True

        return _timed_call(name, arguments, escalation_handler), None

    return ToolCallRecord(
        name=name,
        arguments=sanitize_arguments(arguments),
        result={"ok": False, "error": "tool_not_allowed"},
        status="blocked",
        latency_ms=0.0,
        authorized=False,
    ), None
