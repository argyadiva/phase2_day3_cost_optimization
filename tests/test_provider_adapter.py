"""Exercise the real LangChain/OpenAI adapter over mock HTTP, without API access."""
import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from src import agent
from src.models import AgentRequest


@pytest.mark.parametrize("timeout", [False, True])
def test_responses_payload_usage_and_retry_policy(monkeypatch, lab_settings, timeout):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        if timeout:
            raise httpx.ReadTimeout("sensitive upstream error", request=request)
        return httpx.Response(200, json={
            "id": "resp_fixture", "object": "response", "created_at": 0,
            "model": "capable-model", "status": "completed", "error": None,
            "incomplete_details": None,
            "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text", "text": "Hello", "annotations": []}]}],
            "usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130,
                      "input_tokens_details": {"cached_tokens": 20},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        })
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(agent, "ChatOpenAI", lambda **kwargs: ChatOpenAI(http_client=client, **kwargs))
        run = agent.run_customer_service_agent(AgentRequest("Hello"), settings=lab_settings)
    assert len(calls) == 1
    assert calls[0]["store"] is False
    assert calls[0]["parallel_tool_calls"] is False
    assert calls[0]["max_output_tokens"] == 500
    assert run.model_call_count == 1
    if timeout:
        assert run.error.endswith("TimeoutError")
        assert "sensitive" not in str(run.to_dict())
    else:
        assert run.error is None
        assert run.output == "Hello"
        assert (run.input_tokens, run.cached_input_tokens, run.output_tokens) == (100, 20, 30)
