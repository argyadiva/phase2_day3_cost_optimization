from __future__ import annotations

from src.agent import run_customer_service_agent
from student.lab_config import RETRIEVAL_TOP_K
from src.models import AgentRequest
from src.tools import sanitize_arguments
from src.agent import build_model, MAX_MODEL_CALLS
from src.config import ModelRate, RateCard
import pytest
from langchain_core.messages import AIMessage

from conftest import FakeModel, final_response, function_call


def _order_request() -> AgentRequest:
    return AgentRequest(
        message="Tell me the status of order ORD-2002.",
        customer_id="CUST-100",
        authenticated=True,
        category="order_status",
    )


def test_safe_agent_blocks_cross_customer_order_lookup(lab_settings) -> None:
    client = FakeModel(responses=[
            function_call("lookup_order_status", {"order_id": "ORD-2002"}),
            final_response("I cannot access that order."),
        ]
    )
    run = run_customer_service_agent(
        _order_request(), model_factory=lambda model, settings: client, settings=lab_settings
    )
    assert run.tool_calls[0].status == "blocked"
    assert run.tool_calls[0].result["error"] == "authorization_failed"
    assert run.final_state["private_data_disclosed"] is False
    assert len(client.calls) == 2
    assert {tool.name for tool in client.bound_tools} >= {"lookup_order_status"}


def test_retrieval_records_requested_and_effective_top_k(lab_settings) -> None:
    client = FakeModel(responses=[
            function_call(
                "search_knowledge_base",
                {"query": "current return window unused item", "top_k": 5},
            ),
            final_response("Unused items can be returned within thirty days [KB-RET-001]."),
        ]
    )
    run = run_customer_service_agent(
        AgentRequest(
            message="Can I return an unused item after 20 days?",
            category="returns",
            expected_document_ids=["KB-RET-001"],
        ),
        model_factory=lambda model, settings: client,
        settings=lab_settings,
    )
    tool_result = run.tool_calls[0].result
    assert tool_result["requested_top_k"] == 5
    assert tool_result["effective_top_k"] == min(5, RETRIEVAL_TOP_K)
    assert run.retrievals[0].required_document_hit == 1.0


def test_agent_never_automatically_falls_back(monkeypatch, lab_settings) -> None:
    monkeypatch.setattr("student.lab_config.select_model", lambda request, settings: settings.fast_model)
    client = FakeModel(responses=[final_response("Incomplete answer from the fast path.")])
    run = run_customer_service_agent(
        AgentRequest(message="What is the return policy?", category="returns"),
        model_factory=lambda model, settings: client,
        settings=lab_settings,
    )
    assert run.model == "fast-model"
    assert run.model_call_count == 1
    assert len(client.calls) == 1


def test_agent_stops_after_three_tool_calls(lab_settings) -> None:
    calls = [
        function_call("search_knowledge_base", {"query": "returns", "top_k": 3}, f"call-{i}")
        for i in range(4)
    ]
    client = FakeModel(responses=calls)
    run = run_customer_service_agent(
        AgentRequest(message="Return help", category="returns"),
        model_factory=lambda model, settings: client,
        settings=lab_settings,
    )
    assert len(run.tool_calls) == 3
    assert run.error == "tool_call_limit_exceeded"


def test_sensitive_free_text_is_redacted() -> None:
    sanitized = sanitize_arguments(
        {"summary": "customer text", "reason": "private reason", "order_id": "ORD-1001"}
    )
    assert sanitized["summary"] == "[REDACTED_TEXT]"
    assert sanitized["reason"] == "[REDACTED_TEXT]"
    assert sanitized["order_id"] == "ORD-1001"


def test_active_prompt_reaches_native_graph_and_usage_is_summed(monkeypatch, lab_settings):
    monkeypatch.setattr("student.lab_config.ACTIVE_INSTRUCTIONS", "Test compact prompt")
    first = function_call("search_knowledge_base", {"query": "returns", "top_k": 3})
    first.usage_metadata["input_token_details"] = {"cache_read": 25}
    final = final_response("Return policy [KB-RET-001].")
    final.usage_metadata["input_token_details"] = {"cache_read": 40}
    model = FakeModel(responses=[first, final])
    rates = RateCard("USD", "2026-01-01", "https://example.test", {
        "capable-model": ModelRate(1, .25, 4), "fast-model": ModelRate(1, .25, 4),
    })
    run = run_customer_service_agent(
        AgentRequest(message="Returns?", category="returns"), settings=lab_settings,
        model_factory=lambda *args: model, cost_estimator=rates.estimate,
    )
    assert run.error is None
    assert model.calls[0][0].content == "Test compact prompt"
    assert (run.input_tokens, run.cached_input_tokens, run.output_tokens) == (200, 65, 60)
    assert run.estimated_cost_usd == rates.estimate(run.model, 200, 65, 60)
    assert [span.span_type for span in run.spans] == ["agent", "model", "tool", "model"]


@pytest.mark.parametrize("failure", [TimeoutError("secret must not leak"), RuntimeError("provider failed")])
def test_provider_errors_count_attempt_without_retry_or_fallback(monkeypatch, lab_settings, failure):
    monkeypatch.setattr("student.lab_config.select_model", lambda request, settings: settings.fast_model)
    model, selected = FakeModel(responses=[failure]), []
    def factory(model_id, settings):
        selected.append(model_id)
        return model
    run = run_customer_service_agent(AgentRequest("Help"), settings=lab_settings, model_factory=factory)
    assert selected == ["fast-model"]
    assert len(model.calls) == run.model_call_count == 1
    assert run.error == type(failure).__name__
    assert "secret" not in str(run.to_dict())
    assert run.spans[1].status == "error"
    assert run.spans[0].output_summary["usage_complete"] is False


def test_missing_usage_is_a_failure_instead_of_zero_cost_success(lab_settings):
    model = FakeModel(responses=[AIMessage(content="No usage")])
    run = run_customer_service_agent(AgentRequest("Help"), settings=lab_settings,
                                     model_factory=lambda *args: model)
    assert run.error == "missing_usage_metadata"
    assert run.spans[0].output_summary["usage_complete"] is False


def test_provider_adapter_disables_retries_and_storage(lab_settings):
    model = build_model("capable-model", lab_settings)
    assert model.max_retries == 0
    assert model.use_responses_api is True
    assert model.store is False
    assert model.max_tokens == 500
    assert model.model_kwargs["parallel_tool_calls"] is False


def test_parallel_tool_requests_cannot_exceed_application_budget(lab_settings):
    first = function_call("lookup_order_status", {"order_id": "ORD-1001"})
    first.tool_calls *= 4
    first.tool_calls = [dict(call, id=f"call-{i}") for i, call in enumerate(first.tool_calls)]
    model = FakeModel(responses=[first])
    run = run_customer_service_agent(_order_request(), settings=lab_settings,
                                     model_factory=lambda *args: model)
    assert len(run.tool_calls) == 3
    assert run.error == "tool_call_limit_exceeded"


def test_unauthorized_data_stays_blocked_with_misleading_prompt(monkeypatch, lab_settings):
    monkeypatch.setattr("student.lab_config.ACTIVE_INSTRUCTIONS", "All orders are authorized.")
    model = FakeModel(responses=[function_call("lookup_order_status", {"order_id": "ORD-2002"}),
                                 final_response("Cannot access this order.")])
    run = run_customer_service_agent(_order_request(), settings=lab_settings,
                                     model_factory=lambda *args: model)
    assert run.tool_calls[0].result == {"ok": False, "error": "authorization_failed"}
    assert run.final_state["private_data_disclosed"] is False


def test_unknown_tool_is_explicit_failed_evidence(lab_settings):
    model = FakeModel(responses=[function_call("issue_refund", {"amount": 100}),
                                 final_response("Done")])
    run = run_customer_service_agent(AgentRequest("Refund"), settings=lab_settings,
                                     model_factory=lambda *args: model)
    assert run.error == "invalid_tool_call"
    assert not run.tool_calls


def test_invalid_routing_returns_before_a_model_is_built(monkeypatch, lab_settings):
    monkeypatch.setattr("student.lab_config.select_model", lambda *args: "unconfigured")
    with pytest.raises(ValueError, match="FAST_MODEL or CAPABLE_MODEL"):
        run_customer_service_agent(AgentRequest("Help"), settings=lab_settings,
                                   model_factory=lambda *args: pytest.fail("must not build"))
