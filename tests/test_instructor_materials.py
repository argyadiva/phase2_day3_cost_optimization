"""Instructor proof of exercise wiring. Fixture values are not live measurements."""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FakeModel, final_response, function_call
from instructor import prepare_workspace
from src.agent import run_customer_service_agent
from src.config import ModelRate, RateCard
from src.models import AgentRequest
from src.run_benchmark import execute_benchmark, load_cases, write_outputs
from student import lab_config


def _script(request):
    if request.category in {"returns", "delivery", "warranty"}:
        query, answer = {
            "returns": ("current return window unused item", "Unused items within thirty days [KB-RET-001]."),
            "delivery": ("standard delivery business days", "Three to five business days [KB-DEL-001]."),
            "warranty": ("warranty review information order", "Supply order information [KB-WAR-001]."),
        }[request.category]
        return [function_call("search_knowledge_base", {"query": query, "top_k": 5}), final_response(answer)]
    if request.category == "order_status":
        order = "ORD-2002" if "ORD-2002" in request.message else "ORD-1001"
        return [function_call("lookup_order_status", {"order_id": order}), final_response("In transit" if order == "ORD-1001" else "Access denied")]
    if request.category == "damage":
        return [function_call("search_knowledge_base", {"query": "unusable damaged item", "top_k": 5}),
                function_call("search_knowledge_base", {"query": "ticket priority high damaged", "top_k": 5}, "call-2"),
                function_call("create_support_ticket", {"category": "damage", "priority": "high", "summary": "Synthetic damaged item"}, "call-3"),
                final_response("Opened a ticket [KB-DMG-001].")]
    return [function_call("search_knowledge_base", {"query": "teleportation", "top_k": 5}),
            function_call("escalate_to_human", {"queue": "general-support", "reason": "No evidence", "summary": "Synthetic request"}, "call-2"),
            final_response("Could not find the policy; escalated to human support.")]


def test_four_manual_snapshots_execute_identical_cases_through_native_graph(tmp_path, monkeypatch, lab_settings):
    cases = load_cases()
    rate = ModelRate(1, .25, 4)  # Synthetic, not commercial rates.
    rates = RateCard("USD", "2026-01-01", "https://example.test/fixture", {"fast-model": rate, "capable-model": rate})
    workloads = []
    for label in ("baseline", "prompt_compact", "model_routing", "top_k_2"):
        stage = importlib.import_module(f"instructor.stages.{label}")
        for key in ("ACTIVE_INSTRUCTIONS", "PROMPT_VERSION", "RETRIEVAL_TOP_K", "select_model"):
            monkeypatch.setattr(lab_config, key, getattr(stage, key))
        def runner(request, **kwargs):
            model = FakeModel(responses=_script(request))
            kwargs["model_factory"] = lambda *args: model
            return run_customer_service_agent(request, **kwargs)
        rows, summary = execute_benchmark(label=label, cases=cases, settings=lab_settings,
                                         rates=rates, runner=runner)
        assert summary["configuration"]["runtime"] == "langchain.create_agent"
        assert summary["metrics"]["authorization_pass_rate"] == 1
        assert summary["metrics"]["retrieval_hit_rate"] == 1
        assert summary["metrics"]["quality_pass_rate"] == 1
        assert not summary["metrics"]["cost_is_lower_bound"]
        assert len(summary["case_evidence"]) == 10
        assert all(item["spans"] for item in summary["case_evidence"])
        expected_distribution = {"capable-model": 10} if label in {"baseline", "prompt_compact"} else {"fast-model": 6, "capable-model": 4}
        assert summary["metrics"]["model_distribution"] == expected_distribution
        assert all(k <= stage.RETRIEVAL_TOP_K for row in rows for k in row["effective_top_k"])
        workloads.append(summary["workload"])
        saved = write_outputs(label, rows, summary, output_root=tmp_path)
        assert json.loads((saved / "summary.json").read_text())["configuration"]["active_prompt"] == stage.ACTIVE_INSTRUCTIONS
    assert all(workload == workloads[0] for workload in workloads)


def test_reference_snapshots_change_only_one_optimization_at_a_time():
    from instructor.stages import baseline, prompt_compact, model_routing, top_k_2
    assert prompt_compact.ACTIVE_INSTRUCTIONS != baseline.ACTIVE_INSTRUCTIONS
    assert len(prompt_compact.ACTIVE_INSTRUCTIONS) <= .75 * len(baseline.REFERENCE_INSTRUCTIONS)
    assert model_routing.ACTIVE_INSTRUCTIONS == top_k_2.ACTIVE_INSTRUCTIONS == prompt_compact.ACTIVE_INSTRUCTIONS
    assert baseline.RETRIEVAL_TOP_K == prompt_compact.RETRIEVAL_TOP_K == model_routing.RETRIEVAL_TOP_K == 3
    assert top_k_2.RETRIEVAL_TOP_K == 2
    from src.config import LabSettings
    settings = LabSettings("test", "fast", "capable")
    for category in lab_config.PUBLIC_FAST_CATEGORIES:
        public = AgentRequest("Public FAQ", category=category)
        assert baseline.select_model(public, settings) == prompt_compact.select_model(public, settings) == "capable"
        assert model_routing.select_model(public, settings) == top_k_2.select_model(public, settings) == "fast"
        assert top_k_2.select_model(AgentRequest("Public FAQ", category=category, authenticated=True), settings) == "capable"
        assert top_k_2.select_model(AgentRequest("Public FAQ", category=category, customer_id="CUST-100"), settings) == "capable"
    for category in ("order_status", "damage", "order_change", "private", "complex", "unknown"):
        assert top_k_2.select_model(AgentRequest("Protected", category=category), settings) == "capable"


def test_workspace_copy_excludes_private_files_and_preserves_existing_work(tmp_path, monkeypatch):
    source = tmp_path / "project"
    source.mkdir()
    (source / "student").mkdir()
    (source / "student/lab_config.py").write_text("# starter")
    (source / ".env").write_text("PRIVATE")
    (source / ".env.example").write_text("API_KEY=")
    (source / "data").mkdir()
    (source / "data/model_rates.json").write_text("private rates")
    (source / "outputs").mkdir()
    (source / "outputs/old.txt").write_text("evidence")
    monkeypatch.setattr(prepare_workspace, "PROJECT_ROOT", source)
    monkeypatch.setattr(prepare_workspace, "OUTPUT_DIR", source / "outputs")
    target = prepare_workspace.prepare(source / "outputs/playground")
    assert (target / "student/lab_config.py").read_text() == "# starter"
    assert (target / ".env.example").exists()
    assert not (target / ".env").exists()
    assert not (target / "data/model_rates.json").exists()
    assert not (target / "outputs").exists()
    with pytest.raises(FileExistsError):
        prepare_workspace.prepare(target)
    with pytest.raises(ValueError):
        prepare_workspace.prepare(source)
    assert (source / "outputs/old.txt").read_text() == "evidence"


def test_documented_copy_and_compile_loads_current_snapshot(tmp_path):
    """Copy can retain timestamps; the documented compile step must defeat stale pyc."""
    project = Path(__file__).resolve().parents[1]
    student = tmp_path / "student"
    student.mkdir()
    shutil.copy2(project / "student/__init__.py", student / "__init__.py")
    env = {**os.environ, "PYTHONPATH": str(project)}
    for stage, expected in (("model_routing", "3"), ("top_k_2", "2")):
        shutil.copy2(project / f"instructor/stages/{stage}.py", student / "lab_config.py")
        subprocess.run([sys.executable, "-m", "compileall", "-q", "-f", "student"],
                       cwd=tmp_path, env=env, check=True, capture_output=True)
        result = subprocess.run([sys.executable, "-c",
                                 "from student.lab_config import RETRIEVAL_TOP_K; print(RETRIEVAL_TOP_K)"],
                                cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
        assert result.stdout.strip() == expected
