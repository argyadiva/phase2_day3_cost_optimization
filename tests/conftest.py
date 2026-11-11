from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.config import LabSettings
from src.setup_lab import setup


class FakeModel(BaseChatModel):
    """Scripted provider boundary; tests still execute the real create_agent graph."""
    responses: list[Any]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "offline-scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeModel:
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("Unexpected extra model call")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return ChatResult(generations=[ChatGeneration(message=value)])


def function_call(name: str, arguments: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}
    ], usage_metadata={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130})


def final_response(text: str) -> AIMessage:
    return AIMessage(content=text, usage_metadata={
        "input_tokens": 100, "output_tokens": 30, "total_tokens": 130,
    })


@pytest.fixture(scope="session", autouse=True)
def initialized_lab() -> None:
    setup(reset=True)


@pytest.fixture(autouse=True)
def network_free_by_default(request, monkeypatch) -> None:
    if request.node.get_closest_marker("live") is None:
        def denied(*args, **kwargs):
            raise AssertionError("Offline tests must not open a network connection")
        monkeypatch.setattr("socket.socket.connect", denied)


@pytest.fixture
def lab_settings() -> LabSettings:
    return LabSettings(api_key="test-key", fast_model="fast-model", capable_model="capable-model")
