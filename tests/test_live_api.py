from __future__ import annotations

import pytest

from src.agent import run_customer_service_agent
from src.config import load_rate_card, load_settings, validate_rate_models
from src.models import AgentRequest


@pytest.mark.live
def test_live_api_smoke() -> None:
    settings = load_settings()
    rates = load_rate_card()
    validate_rate_models(settings, rates)
    run = run_customer_service_agent(
        AgentRequest(
            message="How long does standard delivery normally take?",
            category="delivery",
            expected_document_ids=["KB-DEL-001"],
        ),
        settings=settings,
        cost_estimator=rates.estimate,
    )
    assert run.output
    assert run.error is None
