"""One small chat-model interface over two SDKs.

- provider "openai": the Chat Completions API. Works with OpenAI and every
  OpenAI-compatible server (LiteLLM proxy, vLLM, Ollama, ...) via OPENAI_BASE_URL.
- provider "anthropic": the Claude Messages API.

Messages are plain dicts: {"role": "user" | "assistant", "content": str}.
The system prompt is passed separately.
"""

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeVar

import anthropic
import openai
from django.conf import settings
from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Tool:
    """A function the model may call. `run` gets the parsed arguments and returns text."""

    name: str
    description: str
    parameters: dict  # JSON schema of the arguments
    run: Callable[[dict], str]


class LLMError(RuntimeError):
    pass


class _AnswerText:
    """The answer of a tool loop, as the user sees it.

    Text the model writes in any round (also "Let me search..." before a tool
    call) is streamed and kept; rounds are separated by a blank line, in the
    stream and in the returned answer alike.
    """

    def __init__(self, on_text: Callable[[str], None] | None):
        self.on_text = on_text or (lambda text: None)
        self.rounds = [""]

    def add(self, delta: str) -> None:
        if not delta:
            return
        if not self.rounds[-1] and any(self.rounds):
            self.on_text("\n\n")
        self.rounds[-1] += delta
        self.on_text(delta)

    def next_round(self) -> None:
        self.rounds.append("")

    def text(self) -> str:
        return "\n\n".join(r for r in self.rounds if r)


@lru_cache
def _openai() -> openai.OpenAI:
    return openai.OpenAI(max_retries=3)


@lru_cache
def _anthropic() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=3)


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    effort: str | None = None,
    max_tokens: int | None = None,
) -> "LLM":
    provider = provider or settings.LLM_PROVIDER
    effort = effort or settings.LLM_EFFORT or None
    if provider == "openai":
        return OpenAIChat(model or settings.OPENAI_MODEL, temperature, effort, max_tokens)
    if provider == "anthropic":
        return AnthropicChat(model or settings.ANTHROPIC_MODEL, temperature, effort, max_tokens)
    raise ValueError(f"Unknown LLM provider {provider!r}; use 'openai' or 'anthropic'")


class LLM:
    provider = ""

    def __init__(self, model: str, temperature: float | None, effort: str | None, max_tokens: int | None = None):
        self.model = model
        # Left unset by default: current reasoning models reject a custom temperature.
        self.temperature = temperature
        self.effort = effort
        self.max_tokens = max_tokens  # None: LLM_MAX_TOKENS for Claude, the model's limit for OpenAI

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        """Yield the answer text as it is generated."""
        raise NotImplementedError

    def complete(self, system: str, messages: list[dict]) -> str:
        return "".join(self.stream(system, messages))

    def parse(self, system: str, messages: list[dict], schema: type[T]) -> T:
        """Return the answer as an instance of `schema` (structured output)."""
        raise NotImplementedError

    def run_tools(
        self,
        system: str,
        messages: list[dict],
        tools: list[Tool],
        max_rounds: int = 5,
        on_text: Callable[[str], None] | None = None,
    ) -> str:
        """Let the model call tools until it answers. Returns the final answer.

        `on_text` receives answer text as it streams. After `max_rounds` rounds
        of tool calls the model must answer with what it has.
        """
        raise NotImplementedError


class OpenAIChat(LLM):
    provider = "openai"

    def _params(self, system: str, messages: list[dict]) -> dict:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages] if system else list(messages),
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.effort:
            params["reasoning_effort"] = self.effort
        if self.max_tokens:
            params["max_completion_tokens"] = self.max_tokens
        return params

    def stream(self, system, messages):
        response = _openai().chat.completions.create(**self._params(system, messages), stream=True)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def parse(self, system, messages, schema):
        completion = _openai().chat.completions.parse(**self._params(system, messages), response_format=schema)
        message = completion.choices[0].message
        if message.parsed is None:
            raise LLMError(f"No structured output (refusal: {message.refusal!r})")
        return message.parsed

    def run_tools(self, system, messages, tools, max_rounds=5, on_text=None):
        by_name = {t.name: t for t in tools}
        specs = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]
        params = self._params(system, messages)
        history = params.pop("messages")
        answer = _AnswerText(on_text)

        for round_no in range(max_rounds + 1):
            last_round = round_no == max_rounds
            response = _openai().chat.completions.create(
                **params,
                messages=history,
                tools=specs,
                tool_choice="none" if last_round else "auto",
                stream=True,
            )
            text, calls = "", {}
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text += delta.content
                    answer.add(delta.content)
                for call in delta.tool_calls or []:
                    # Tool calls arrive in pieces: an id and name first, then argument fragments.
                    entry = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                    entry["id"] = call.id or entry["id"]
                    if call.function:
                        entry["name"] += call.function.name or ""
                        entry["arguments"] += call.function.arguments or ""

            if not calls:
                return answer.text()

            history.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in calls.values()
                    ],
                }
            )
            for c in calls.values():
                history.append(
                    {"role": "tool", "tool_call_id": c["id"], "content": _call_tool(by_name, c["name"], c["arguments"])}
                )
            answer.next_round()
        return answer.text()


class AnthropicChat(LLM):
    provider = "anthropic"

    def _params(self, system: str, messages: list[dict], max_tokens: int | None = None) -> dict:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens or settings.LLM_MAX_TOKENS,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if settings.ANTHROPIC_FALLBACKS:
            # If the model declines a request, the API retries it on a fallback model.
            params["betas"] = [ANTHROPIC_FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def stream(self, system, messages):
        with _anthropic().beta.messages.stream(**self._params(system, messages)) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            raise LLMError("The model declined to answer")

    def parse(self, system, messages, schema):
        # Not streamed, so keep max_tokens below the SDK's limit for non-streaming requests.
        params = self._params(system, messages, max_tokens=min(self.max_tokens or settings.LLM_MAX_TOKENS, 16000))
        response = _anthropic().beta.messages.parse(**params, output_format=schema)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise LLMError(f"No structured output (stop_reason: {response.stop_reason})")
        return response.parsed_output

    def run_tools(self, system, messages, tools, max_rounds=5, on_text=None):
        by_name = {t.name: t for t in tools}
        specs = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
                "eager_input_streaming": True,
            }
            for t in tools
        ]
        history = list(messages)
        answer = _AnswerText(on_text)

        for round_no in range(max_rounds + 1):
            last_round = round_no == max_rounds
            params = self._params(system, history)
            with _anthropic().beta.messages.stream(
                **params,
                tools=specs,
                tool_choice={"type": "none"} if last_round else {"type": "auto"},
            ) as stream:
                for event in stream:
                    if event.type == "text":
                        answer.add(event.text)
                response = stream.get_final_message()

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "refusal":
                raise LLMError("The model declined to answer")
            if not tool_uses:
                return answer.text()
            if response.stop_reason == "max_tokens":
                raise LLMError("A tool call was cut off by max_tokens; raise LLM_MAX_TOKENS")

            # Send the whole assistant turn back unchanged (it may contain thinking blocks).
            history.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_uses:
                output = _call_tool(by_name, block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            history.append({"role": "user", "content": results})
            answer.next_round()
        return answer.text()


def _call_tool(tools: dict[str, Tool], name: str, arguments: str | dict) -> str:
    """Run one tool call. Errors are returned to the model as text, so it can recover."""
    tool = tools.get(name)
    if tool is None:
        return f"Error: unknown tool {name!r}"
    try:
        args = json.loads(arguments or "{}") if isinstance(arguments, str) else arguments
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        if missing:
            raise ValueError(f"missing argument(s): {', '.join(missing)}")
        return tool.run(args)
    except Exception as exc:  # noqa: BLE001 - the model gets the error and can retry
        log.warning("tool %s failed: %s", name, exc)
        return f"Error: {exc}"
