from __future__ import annotations

import argparse
import csv
import json
import hashlib
import inspect
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable
from importlib.metadata import version
from dataclasses import asdict

from src.agent import ModelFactory, build_model, run_customer_service_agent
from src.config import (
    BENCHMARK_WORKLOAD,
    OUTPUT_DIR,
    LabSettings,
    RateCard,
    load_rate_card,
    load_settings,
    validate_rate_models,
)
from student import lab_config
from src.models import AgentRequest, AgentRun, BenchmarkCase
from src.setup_lab import setup


RUN_LABELS = ("baseline", "prompt_compact", "model_routing", "top_k_2")
NUMBER_EQUIVALENTS = {"three": "3", "five": "5", "thirty": "30"}
CASE_FIELDS = [
    "case_id",
    "category",
    "model",
    "prompt_version",
    "configured_top_k",
    "requested_top_k",
    "effective_top_k",
    "latency_ms",
    "model_latency_ms",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "estimated_cost",
    "model_calls",
    "tool_calls",
    "tool_names",
    "expected_document_ids",
    "retrieved_document_ids",
    "retrieval_hit",
    "quality_passed",
    "authorization_passed",
    "output",
    "failure_reasons",
    "error",
]


def _normalize(value: str) -> str:
    normalized = value.casefold()
    for word, number in NUMBER_EQUIVALENTS.items():
        normalized = normalized.replace(word, number)
    return " ".join(normalized.split())


def load_cases(path: Path = BENCHMARK_WORKLOAD) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        BenchmarkCase(
            case_id=str(item["case_id"]),
            category=str(item["category"]),
            request=AgentRequest(category=item["category"], **item["request"]),
            expected=dict(item["expected"]),
        )
        for item in raw
    ]
    if len(cases) != 10:
        raise ValueError(f"The classroom benchmark requires exactly 10 cases; found {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Benchmark case IDs must be unique")
    return cases


def _evaluate(case: BenchmarkCase, run: AgentRun) -> dict[str, Any]:
    expected = case.expected
    normalized = _normalize(run.output)
    failures: list[str] = []

    if any(_normalize(str(fact)) not in normalized for fact in expected.get("required_facts", [])):
        failures.append("missing_required_fact")
    if any(
        str(citation).casefold() not in normalized
        for citation in expected.get("required_citations", [])
    ):
        failures.append("missing_required_citation")

    document_ids = [doc for item in run.retrievals for doc in item.document_ids]
    expected_docs = set(case.request.expected_document_ids)
    retrieval_hit = (
        1.0 if not expected_docs else float(expected_docs.issubset(document_ids))
    )
    if retrieval_hit < 1:
        failures.append("retrieval_miss")

    tool_names = [call.name for call in run.tool_calls]
    actual_tools = set(tool_names)
    if set(expected.get("required_tools", [])) - actual_tools:
        failures.append("missing_required_tool")
    if set(expected.get("forbidden_tools", [])) & actual_tools:
        failures.append("forbidden_tool_selected")
    if any(
        run.final_state.get(key) != value
        for key, value in expected.get("final_state", {}).items()
    ):
        failures.append("incorrect_final_state")
    if run.error:
        failures.append("run_error")

    authorization_passed = not any(
        call.result.get("ok") and not call.authorized for call in run.tool_calls
    )
    if not authorization_passed:
        failures.append("unauthorized_action")

    retrieval_calls = [
        call for call in run.tool_calls if call.name == "search_knowledge_base"
    ]
    requested_top_k = [call.result.get("requested_top_k") for call in retrieval_calls]
    effective_top_k = [call.result.get("effective_top_k") for call in retrieval_calls]
    model_latency = sum(
        span.latency_ms for span in run.spans if span.span_type == "model"
    )
    unique_failures = sorted(set(failures))
    return {
        "case_id": case.case_id,
        "category": case.category,
        "model": run.model,
        "prompt_version": run.prompt_version,
        "configured_top_k": lab_config.RETRIEVAL_TOP_K,
        "requested_top_k": requested_top_k,
        "effective_top_k": effective_top_k,
        "latency_ms": round(run.latency_ms, 2),
        "model_latency_ms": round(model_latency, 2),
        "input_tokens": run.input_tokens,
        "cached_input_tokens": run.cached_input_tokens,
        "output_tokens": run.output_tokens,
        "estimated_cost": round(run.estimated_cost_usd, 8),
        "model_calls": run.model_call_count,
        "tool_calls": len(run.tool_calls),
        "tool_names": tool_names,
        "expected_document_ids": list(case.request.expected_document_ids),
        "retrieved_document_ids": document_ids,
        "retrieval_hit": retrieval_hit,
        "quality_passed": not unique_failures,
        "authorization_passed": authorization_passed,
        "output": run.output,
        "failure_reasons": unique_failures,
        "error": run.error or "",
    }


def _nearest_rank(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 2)


def summarize(
    label: str,
    rows: list[dict[str, Any]],
    settings: LabSettings,
    rates: RateCard,
    case_ids: list[str],
) -> dict[str, Any]:
    successes = sum(bool(row["quality_passed"]) for row in rows)
    relevant_retrieval = [row for row in rows if row["expected_document_ids"]]
    total_cost = round(sum(float(row["estimated_cost"]) for row in rows), 8)
    return {
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workload": {
            "case_count": len(rows),
            "ordered_case_ids": case_ids,
        },
        "configuration": {
            "runtime": "langchain.create_agent",
            "langchain_version": version("langchain"),
            "langchain_openai_version": version("langchain-openai"),
            "prompt_version": lab_config.PROMPT_VERSION,
            "prompt_character_count": len(lab_config.ACTIVE_INSTRUCTIONS),
            "active_prompt": lab_config.ACTIVE_INSTRUCTIONS,
            "routing_source": inspect.getsource(lab_config.select_model),
            "lab_config_sha256": hashlib.sha256(Path(lab_config.__file__).read_bytes()).hexdigest(),
            "retrieval_top_k": lab_config.RETRIEVAL_TOP_K,
            "fast_model": settings.fast_model,
            "capable_model": settings.capable_model,
            "rate_card_currency": rates.currency,
            "rate_card_as_of": rates.as_of,
            "rate_card_source_url": rates.source_url,
            "rate_card_models": {name: asdict(rate) for name, rate in rates.models.items()},
            "automatic_fallback": False,
            "sdk_max_retries": 0,
        },
        "metrics": {
            "p50_latency_ms": round(median([row["latency_ms"] for row in rows]), 2),
            "p95_latency_ms": _nearest_rank(
                [row["latency_ms"] for row in rows], 0.95
            ),
            "total_estimated_cost": total_cost,
            "cost_per_successful_task": (
                round(total_cost / successes, 8) if successes else None
            ),
            "total_input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "total_cached_input_tokens": sum(
                int(row["cached_input_tokens"]) for row in rows
            ),
            "total_output_tokens": sum(int(row["output_tokens"]) for row in rows),
            "quality_pass_rate": round(successes / len(rows), 4),
            "authorization_pass_rate": round(
                sum(bool(row["authorization_passed"]) for row in rows) / len(rows),
                4,
            ),
            "retrieval_hit_rate": round(
                sum(float(row["retrieval_hit"]) for row in relevant_retrieval)
                / len(relevant_retrieval),
                4,
            )
            if relevant_retrieval
            else 1.0,
            "model_distribution": dict(Counter(str(row["model"]) for row in rows)),
            "api_error_count": sum(bool(row["error"]) for row in rows),
        },
    }


def _markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    config = summary["configuration"]
    models = ", ".join(
        f"{name}: {count}" for name, count in metrics["model_distribution"].items()
    )
    cost_per_success = metrics["cost_per_successful_task"]
    cost_text = "n/a" if cost_per_success is None else f"{cost_per_success:.8f}"
    return f"""# Benchmark Run: {summary['label']}

This file reports the active code. It does not compare runs or make a release decision.

## Active configuration

| Field | Value |
|---|---|
| Runtime | `{config['runtime']}` |
| LangChain / OpenAI integration | {config['langchain_version']} / {config['langchain_openai_version']} |
| Config SHA256 | `{config['lab_config_sha256']}` |
| Prompt version | `{config['prompt_version']}` |
| Prompt characters | {config['prompt_character_count']} |
| Retrieval top-k | {config['retrieval_top_k']} |
| Fast model | `{config['fast_model']}` |
| Capable model | `{config['capable_model']}` |
| Automatic fallback | `{config['automatic_fallback']}` |
| SDK retries | {config['sdk_max_retries']} |
| Rate card | {config['rate_card_currency']} • {config['rate_card_as_of']} |

## Metrics

| Metric | Value |
|---|---:|
| Cases | {summary['workload']['case_count']} |
| p50 latency | {metrics['p50_latency_ms']:.2f} ms |
| p95 latency | {metrics['p95_latency_ms']:.2f} ms |
| Total estimated cost | {metrics['total_estimated_cost']:.8f} |
| Cost per successful task | {cost_text} |
| Input tokens | {metrics['total_input_tokens']} |
| Cached-input tokens (subset of input) | {metrics['total_cached_input_tokens']} |
| Output tokens | {metrics['total_output_tokens']} |
| Quality pass rate | {metrics['quality_pass_rate']:.1%} |
| Authorization pass rate | {metrics['authorization_pass_rate']:.1%} |
| Retrieval hit rate | {metrics['retrieval_hit_rate']:.1%} |
| API errors | {metrics['api_error_count']} |
| Cost is a lower bound due to failed calls | {metrics.get('cost_is_lower_bound', False)} |
| Model distribution | {models} |

Read `cases.csv` before drawing a conclusion. A lower cost or p95 does not excuse a new quality, authorization, or retrieval failure.
"""


def ensure_output_available(
    label: str,
    *,
    output_root: Path = OUTPUT_DIR,
    overwrite: bool = False,
) -> Path:
    run_dir = output_root / label
    expected = [run_dir / "cases.csv", run_dir / "summary.json", run_dir / "summary.md"]
    if any(path.exists() for path in expected) and not overwrite:
        raise FileExistsError(
            f"Evidence already exists for label '{label}'. Choose the correct next label "
            "or rerun explicitly with --overwrite."
        )
    return run_dir


def write_outputs(
    label: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_root: Path = OUTPUT_DIR,
    overwrite: bool = False,
) -> Path:
    run_dir = ensure_output_available(
        label,
        output_root=output_root,
        overwrite=overwrite,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for field in (
                "requested_top_k",
                "effective_top_k",
                "tool_names",
                "expected_document_ids",
                "retrieved_document_ids",
                "failure_reasons",
            ):
                serialized[field] = json.dumps(serialized[field], ensure_ascii=False)
            writer.writerow(serialized)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    return run_dir


def execute_benchmark(
    *,
    label: str,
    cases: list[BenchmarkCase],
    settings: LabSettings,
    rates: RateCard,
    model_factory: ModelFactory = build_model,
    runner: Callable[..., AgentRun] = run_customer_service_agent,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for case in cases:
        run = runner(
            request=case.request,
            profile=f"manual-{label}",
            model_factory=model_factory,
            settings=settings,
            cost_estimator=rates.estimate,
        )
        rows.append(_evaluate(case, run))
        evidence.append({"case_id": case.case_id, **run.to_dict()})
    summary = summarize(
        label,
        rows,
        settings,
        rates,
        [case.case_id for case in cases],
    )
    summary["workload"]["sha256"] = hashlib.sha256(json.dumps(
        [asdict(case) for case in cases], sort_keys=True
    ).encode()).hexdigest()
    summary["case_evidence"] = evidence
    summary["metrics"]["cost_is_lower_bound"] = any(row["error"] for row in rows)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the currently active manual optimization code"
    )
    parser.add_argument("--label", required=True, choices=RUN_LABELS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite only the three generated files for this label.",
    )
    args = parser.parse_args()

    # Protect existing evidence before loading a client or making any paid call.
    ensure_output_available(args.label, overwrite=args.overwrite)
    settings = load_settings()
    rates = load_rate_card()
    validate_rate_models(settings, rates)
    cases = load_cases()
    setup(reset=False)
    rows, summary = execute_benchmark(
        label=args.label,
        cases=cases,
        settings=settings,
        rates=rates,
    )
    output_dir = write_outputs(
        args.label,
        rows,
        summary,
        overwrite=args.overwrite,
    )
    print(f"label={args.label}")
    print(f"cases={len(rows)}")
    print(f"api_errors={summary['metrics']['api_error_count']}")
    print(f"report={output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
