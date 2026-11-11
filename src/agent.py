"""Safe Northstar runtime. LangChain owns the model/tool execution loop."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from student import lab_config
from src.config import LabSettings, load_settings
from src.models import (
    AgentRequest, AgentRun, KnowledgeBaseVersion, RetrievalRecord, ToolCallRecord, TraceSpan,
)
from src.tools import TOOL_DEFINITIONS, ToolContext, execute_tool

MAX_TOOL_CALLS = 3
MAX_MODEL_CALLS = 4
MAX_OUTPUT_TOKENS = 500
ModelFactory = Callable[[str, LabSettings], BaseChatModel]


class LabExecutionError(RuntimeError):
    """A bounded run stopped; record the reason as failed benchmark evidence."""


def build_model(model_id: str, settings: LabSettings) -> BaseChatModel:
    """No fallback or hidden retry. Each request keeps its selected model."""
    return ChatOpenAI(
        model=model_id,
        api_key=settings.api_key,
        base_url=settings.base_url,
        use_responses_api=True,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=60,
        max_retries=0,
        store=False,
        model_kwargs={"parallel_tool_calls": False},
    )


class RunEvidence(AgentMiddleware):
    """Observe model calls without changing prompts or choosing models."""

    def __init__(self, model_id: str, run_id: str, root_id: str) -> None:
        self.model_id, self.run_id, self.root_id = model_id, run_id, root_id
        self.spans: list[TraceSpan] = []
        self.input_tokens = self.cached_input_tokens = self.output_tokens = 0
        self.model_calls = 0
        self.usage_complete = True

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        if self.model_calls >= MAX_MODEL_CALLS:
            raise LabExecutionError("model_call_limit_exceeded")
        self.model_calls += 1
        started = time.perf_counter()
        status, usage = "error", {}
        try:
            response = handler(request)
            for message in response.result:
                if isinstance(message, AIMessage):
                    if message.usage_metadata is None:
                        self.usage_complete = False
                    else:
                        item = message.usage_metadata
                        self.input_tokens += item["input_tokens"]
                        self.output_tokens += item["output_tokens"]
                        self.cached_input_tokens += item.get("input_token_details", {}).get("cache_read", 0)
                        usage = dict(item)
            if not self.usage_complete:
                raise LabExecutionError("missing_usage_metadata")
            status = "ok"
            return response
        except Exception:
            self.usage_complete = False
            raise
        finally:
            self.spans.append(TraceSpan(
                span_id=str(uuid.uuid4()), run_id=self.run_id,
                parent_span_id=self.root_id, span_type="model",
                name=f"langchain.model.turn_{self.model_calls}", status=status,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                input_summary={"model": self.model_id, "message_count": len(request.messages)},
                output_summary=usage,
            ))


def _tools(
    context: ToolContext, evidence: RunEvidence,
) -> tuple[list[StructuredTool], list[ToolCallRecord], list[RetrievalRecord]]:
    records: list[ToolCallRecord] = []
    retrievals: list[RetrievalRecord] = []
    # ToolNode may schedule multiple tools. Serialize state changes and enforce the
    # application budget even when a model ignores parallel_tool_calls=False.
    lock = Lock()

    def bind(definition: dict[str, Any]) -> StructuredTool:
        def invoke(**arguments: Any) -> str:
            with lock:
                if len(records) >= MAX_TOOL_CALLS:
                    raise LabExecutionError("tool_call_limit_exceeded")
                record, retrieval = execute_tool(definition["name"], arguments, context)
                records.append(record)
                if retrieval is not None:
                    retrievals.append(retrieval)
                evidence.spans.append(TraceSpan(
                    span_id=str(uuid.uuid4()), run_id=evidence.run_id,
                    parent_span_id=evidence.root_id, span_type="tool",
                    name=record.name, status=record.status, latency_ms=record.latency_ms,
                    input_summary=record.arguments,
                    output_summary={
                        "ok": record.result.get("ok", False),
                        "authorized": record.authorized,
                        "error": record.result.get("error"),
                        "requested_top_k": record.result.get("requested_top_k"),
                        "effective_top_k": record.result.get("effective_top_k"),
                    },
                ))
                return json.dumps(record.result, ensure_ascii=True)
        return StructuredTool.from_function(
            func=invoke, name=definition["name"], description=definition["description"],
            args_schema=definition["parameters"],
        )

    return [bind(definition) for definition in TOOL_DEFINITIONS], records, retrievals


def run_customer_service_agent(
    request: AgentRequest,
    *,
    kb_version: KnowledgeBaseVersion = "healthy",
    profile: str = "manual-benchmark",
    settings: LabSettings | None = None,
    model_factory: ModelFactory = build_model,
    cost_estimator: Callable[[str, int, int, int], float] | None = None,
) -> AgentRun:
    settings = settings or load_settings()
    selected_model = lab_config.select_model(request, settings)
    if selected_model not in {settings.fast_model, settings.capable_model}:
        raise ValueError("select_model must return FAST_MODEL or CAPABLE_MODEL")
    if lab_config.RETRIEVAL_TOP_K not in {1, 2, 3, 4, 5}:
        raise ValueError("RETRIEVAL_TOP_K must be between 1 and 5")

    run_id, root_id = str(uuid.uuid4()), str(uuid.uuid4())
    evidence = RunEvidence(selected_model, run_id, root_id)
    context = ToolContext(request=request, kb_version=kb_version,
                          retrieval_top_k_cap=lab_config.RETRIEVAL_TOP_K)
    tools, records, retrievals = _tools(context, evidence)
    started = time.perf_counter()
    error = None
    output = "The support service could not complete this request safely. Please contact human support."
    try:
        agent = create_agent(
            model=model_factory(selected_model, settings),
            tools=tools,
            system_prompt=lab_config.ACTIVE_INSTRUCTIONS,
            middleware=[evidence],
        )
        result = agent.invoke({"messages": [{"role": "user", "content": json.dumps({
            "message": request.message,
            "session": {"authenticated": request.authenticated, "customer_id": request.customer_id},
        })}]}, config={"recursion_limit": 32})
        final = result["messages"][-1]
        if any(isinstance(message, ToolMessage) and message.status == "error"
               for message in result["messages"]):
            raise LabExecutionError("invalid_tool_call")
        if any(record.status == "error" for record in records):
            raise LabExecutionError("tool_execution_error")
        if not isinstance(final, AIMessage) or final.tool_calls or not final.text.strip():
            raise LabExecutionError("missing_final_answer")
        output = final.text.strip()
    except Exception as exc:
        # SDK exception text may contain request data or credentials. Preserve a
        # safe error identifier, and count every failed attempt in the evidence.
        error = str(exc) if isinstance(exc, LabExecutionError) else type(exc).__name__
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    spans = [TraceSpan(
        span_id=root_id, run_id=run_id, parent_span_id=None, span_type="agent",
        name="langchain.customer_service_agent", status="error" if error else "ok",
        latency_ms=latency_ms,
        input_summary={"category": request.category, "authenticated": request.authenticated},
        output_summary={"tool_call_count": len(records), "error": error,
                        "usage_complete": evidence.usage_complete},
    ), *evidence.spans]
    return AgentRun(
        run_id=run_id, conversation_id=request.conversation_id or str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(), profile=profile,
        kb_version=kb_version, prompt_version=lab_config.PROMPT_VERSION, request=request,
        output=output, tool_calls=records, retrievals=retrievals,
        final_state=context.final_state(), spans=spans, latency_ms=latency_ms,
        input_tokens=evidence.input_tokens, cached_input_tokens=evidence.cached_input_tokens,
        output_tokens=evidence.output_tokens,
        estimated_cost_usd=cost_estimator(selected_model, evidence.input_tokens,
            evidence.cached_input_tokens, evidence.output_tokens) if cost_estimator else 0.0,
        error=error, model=selected_model, model_call_count=evidence.model_calls,
    )
