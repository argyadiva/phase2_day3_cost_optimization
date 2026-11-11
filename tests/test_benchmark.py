from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.config import LabSettings, ModelRate, RateCard, load_settings
from src.models import AgentRun, TraceSpan
from src.run_benchmark import (
    CASE_FIELDS,
    RUN_LABELS,
    execute_benchmark,
    load_cases,
    write_outputs,
    summarize,
)
from src.config import load_rate_card, validate_rate_models


def _rates() -> RateCard:
    rate = ModelRate(input_per_1m=1.0, cached_input_per_1m=0.25, output_per_1m=4.0)
    return RateCard(
        currency="USD",
        as_of="2026-01-01",
        source_url="https://provider.example/pricing",
        models={"fast-model": rate, "capable-model": rate},
    )


def _run(request, *, profile: str, settings: LabSettings, **kwargs) -> AgentRun:
    span = TraceSpan(
        span_id="model-span",
        run_id="run-1",
        parent_span_id="root",
        span_type="model",
        name="responses.create.turn_1",
        status="error" if "API failure" in request.message else "ok",
        latency_ms=25.0,
        input_summary={},
        output_summary={},
    )
    error = "RuntimeError: provider unavailable" if "API failure" in request.message else None
    return AgentRun(
        run_id="run-1",
        conversation_id="conversation-1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        profile=profile,
        kb_version="healthy",
        prompt_version="manual-baseline-v1",
        request=request,
        output="Synthetic response for an offline runner test.",
        tool_calls=[],
        retrievals=[],
        final_state={
            "ticket_created": False,
            "ticket_priority": None,
            "escalated": False,
            "private_data_disclosed": False,
        },
        spans=[span],
        latency_ms=50.0,
        input_tokens=100,
        output_tokens=20,
        estimated_cost_usd=0.00018,
        error=error,
        model=settings.capable_model,
        model_call_count=1,
    )


def test_fixed_workload_has_ten_unique_ordered_cases() -> None:
    cases = load_cases()
    assert len(cases) == 10
    assert len({case.case_id for case in cases}) == 10
    assert cases[0].case_id == "return-window-first"
    assert cases[-1].case_id == "unknown-policy"


def test_all_labels_use_the_same_ordered_workload(lab_settings) -> None:
    cases = load_cases()
    expected = [case.case_id for case in cases]
    for label in RUN_LABELS:
        _, summary = execute_benchmark(
            label=label,
            cases=cases,
            settings=lab_settings,
            rates=_rates(),
            model_factory=lambda *args: None,
            runner=_run,
        )
        assert summary["workload"]["ordered_case_ids"] == expected


def test_outputs_are_complete_and_protected_from_accidental_overwrite(
    tmp_path, lab_settings
) -> None:
    cases = load_cases()
    rows, summary = execute_benchmark(
        label="baseline",
        cases=cases,
        settings=lab_settings,
        rates=_rates(),
        model_factory=lambda *args: None,
        runner=_run,
    )
    output_dir = write_outputs(
        "baseline", rows, summary, output_root=tmp_path
    )
    assert {path.name for path in output_dir.iterdir()} == {
        "cases.csv",
        "summary.json",
        "summary.md",
    }
    saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["workload"]["case_count"] == 10
    assert {
        "p50_latency_ms",
        "p95_latency_ms",
        "total_estimated_cost",
        "cost_per_successful_task",
        "total_input_tokens",
        "total_cached_input_tokens",
        "total_output_tokens",
        "quality_pass_rate",
        "authorization_pass_rate",
        "retrieval_hit_rate",
        "model_distribution",
        "api_error_count",
    }.issubset(saved["metrics"])
    assert set(rows[0]) == set(CASE_FIELDS)
    with pytest.raises(FileExistsError):
        write_outputs("baseline", rows, summary, output_root=tmp_path)
    write_outputs("baseline", rows, summary, output_root=tmp_path, overwrite=True)


def test_api_errors_are_explicit_failed_rows(lab_settings) -> None:
    case = load_cases()[0]
    failed_request = type(case.request)(
        message="API failure",
        customer_id=case.request.customer_id,
        authenticated=case.request.authenticated,
        category=case.request.category,
        expected_document_ids=case.request.expected_document_ids,
    )
    failed_case = type(case)(
        case_id=case.case_id,
        category=case.category,
        request=failed_request,
        expected=case.expected,
    )
    rows, summary = execute_benchmark(
        label="baseline",
        cases=[failed_case],
        settings=lab_settings,
        rates=_rates(),
        model_factory=lambda *args: None,
        runner=_run,
    )
    assert rows[0]["quality_passed"] is False
    assert "run_error" in rows[0]["failure_reasons"]
    assert rows[0]["error"]
    assert summary["metrics"]["api_error_count"] == 1


def test_settings_reject_identical_model_ids(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.setenv("FAST_MODEL", "same-model")
    monkeypatch.setenv("CAPABLE_MODEL", "same-model")
    with pytest.raises(RuntimeError, match="different model IDs"):
        load_settings()


def test_percentiles_and_cost_per_success_are_calculated_from_all_rows(lab_settings):
    rows = [{
        "quality_passed": i < 8, "estimated_cost": .5, "expected_document_ids": ["doc"],
        "latency_ms": i + 1, "input_tokens": 100, "cached_input_tokens": 25,
        "output_tokens": 20, "authorization_passed": True, "retrieval_hit": 1,
        "model": "capable-model", "error": "",
    } for i in range(10)]
    summary = summarize("baseline", rows, lab_settings, _rates(), [str(i) for i in range(10)])
    metrics = summary["metrics"]
    assert metrics["p50_latency_ms"] == 5.5
    assert metrics["p95_latency_ms"] == 10
    assert metrics["total_estimated_cost"] == 5
    assert metrics["cost_per_successful_task"] == .625
    assert metrics["quality_pass_rate"] == .8
    assert metrics["total_cached_input_tokens"] == 250
    for row in rows:
        row["quality_passed"] = False
    assert summarize("baseline", rows, lab_settings, _rates(), [])['metrics']['cost_per_successful_task'] is None


def test_credentials_are_validated_without_using_a_parent_env(monkeypatch):
    monkeypatch.setattr("src.config.find_dotenv", lambda **kwargs: "")
    for key in ("API_KEY", "OPENAI_API_KEY", "FAST_MODEL", "CAPABLE_MODEL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="Missing lab configuration"):
        load_settings()


def test_rate_card_rejects_missing_and_placeholder_data(tmp_path, lab_settings):
    path = tmp_path / "rates.json"
    with pytest.raises(RuntimeError, match="Missing"):
        load_rate_card(path)
    path.write_text(json.dumps({"currency": "USD", "as_of": "YYYY-MM-DD", "source_url": "replace"}))
    with pytest.raises(ValueError, match="Replace"):
        load_rate_card(path)
    rates = _rates()
    missing = RateCard(rates.currency, rates.as_of, rates.source_url, {"capable-model": rates.models["capable-model"]})
    with pytest.raises(RuntimeError, match="missing configured models"):
        validate_rate_models(lab_settings, missing)
    zeros = RateCard(rates.currency, rates.as_of, rates.source_url, {name: ModelRate(0, 0, 0) for name in rates.models})
    with pytest.raises(RuntimeError, match="non-placeholder"):
        validate_rate_models(lab_settings, zeros)


def test_cli_rejects_missing_rates_before_executing_any_case(monkeypatch, lab_settings):
    from src import run_benchmark
    monkeypatch.setattr("sys.argv", ["benchmark", "--label", "baseline"])
    monkeypatch.setattr(run_benchmark, "ensure_output_available", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_benchmark, "load_settings", lambda: lab_settings)
    monkeypatch.setattr(run_benchmark, "load_rate_card", lambda: RateCard("USD", "2026-01-01", "https://example.test", {}))
    monkeypatch.setattr(run_benchmark, "execute_benchmark", lambda **kwargs: pytest.fail("must not execute"))
    with pytest.raises(RuntimeError, match="missing configured models"):
        run_benchmark.main()
