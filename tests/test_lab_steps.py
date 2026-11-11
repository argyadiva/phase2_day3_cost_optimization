from __future__ import annotations

import pytest

from src.config import LabSettings
from student.lab_config import (
    ACTIVE_INSTRUCTIONS,
    PROMPT_VERSION,
    RETRIEVAL_TOP_K,
    select_model,
)
from src.prompt_contract import REFERENCE_INSTRUCTIONS
from src.models import AgentRequest
from src.retrieval import search


pytestmark = pytest.mark.lab_step


def test_prompt_compaction_is_shorter_and_preserves_contract() -> None:
    assert PROMPT_VERSION != "manual-baseline-v1"
    assert len(ACTIVE_INSTRUCTIONS) <= len(REFERENCE_INSTRUCTIONS) * 0.75
    lowered = ACTIVE_INSTRUCTIONS.casefold()
    required_concepts = {
        "tool boundary": ("only", "tool"),
        "retrieve before claim": ("search", "policy"),
        "citation": ("cite", "document"),
        "no invention": ("never invent",),
        "missing evidence": ("evidence", "could not find", "escalat"),
        "authorization": ("authenticat", "ownership"),
        "forbidden actions": ("refund", "credit", "replacement", "cancel", "modif", "payment", "identity"),
        "hidden reasoning": ("hidden reasoning",),
    }
    missing = [
        name
        for name, phrases in required_concepts.items()
        if not all(phrase in lowered for phrase in phrases)
    ]
    assert not missing, f"Compact prompt lost required contracts: {missing}"


def test_routing_uses_fast_only_for_anonymous_public_requests() -> None:
    settings = LabSettings("test", "fast-model", "capable-model")
    for category in ("returns", "delivery", "warranty", "support_hours"):
        public = AgentRequest(message="Public policy?", category=category)
        assert select_model(public, settings) == settings.fast_model
        assert select_model(AgentRequest(message="Public policy?", category=category,
                                         authenticated=True), settings) == settings.capable_model
        assert select_model(AgentRequest(message="Public policy?", category=category,
                                         customer_id="CUST-100"), settings) == settings.capable_model

    protected = [
        AgentRequest(
            message="Where is my order?",
            category="order_status",
            authenticated=True,
            customer_id="CUST-100",
        ),
        AgentRequest(
            message="Create a damage ticket",
            category="damage",
            authenticated=True,
            customer_id="CUST-100",
        ),
        AgentRequest(message="Change my order", category="order_change"),
        AgentRequest(message="Explain an account privacy issue", category="account"),
        AgentRequest(message="Handle this multi-policy case", category="complex"),
        AgentRequest(message="Unknown request", category="unknown"),
        AgentRequest(
            message="Public topic with a supplied identity",
            category="returns",
            customer_id="CUST-100",
        ),
        AgentRequest(
            message="Public category but authenticated",
            category="returns",
            authenticated=True,
            customer_id="CUST-100",
        ),
    ]
    assert all(select_model(request, settings) == settings.capable_model for request in protected)


def test_top_k_is_two_and_required_documents_survive_fixed_queries() -> None:
    assert RETRIEVAL_TOP_K == 2
    checks = [
        ("current return window unused item", "KB-RET-001"),
        ("standard delivery business days", "KB-DEL-001"),
        ("warranty review information order", "KB-WAR-001"),
    ]
    for query, document_id in checks:
        _, record = search(
            query,
            "healthy",
            top_k=RETRIEVAL_TOP_K,
            expected_document_ids=[document_id],
        )
        assert record.required_document_hit == 1.0, document_id
