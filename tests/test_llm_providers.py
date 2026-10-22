"""core/llm.py and core/embeddings.py against local fake OpenAI / Anthropic servers.

The real SDKs make real HTTP requests; the servers replay scripted responses in
each API's wire format and record what they received.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

import pytest
from pydantic import BaseModel

from core import embeddings as embeddings_module
from core import llm as llm_module
from core.llm import Tool, get_llm


class Choice(BaseModel):
    category: Literal["a", "b"]


class FakeAPI:
    """Replays `script` (a list of responses) and records requests."""

    def __init__(self, render):
        self.render = render  # (scripted response, request body) -> (content type, body bytes)
        self.script: list = []
        self.requests: list[dict] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                content_type, payload = api.render(api.script.pop(0), body)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


def sse(events: list[tuple[str | None, dict | str]]) -> bytes:
    out = []
    for name, data in events:
        line = f"event: {name}\n" if name else ""
        out.append(line + f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n")
    return "".join(out).encode()


# --- OpenAI-compatible ------------------------------------------------------------------


def openai_render(item, body):
    if "embedding" in item:
        data = [{"object": "embedding", "index": i, "embedding": item["embedding"]} for i in range(len(body["input"]))]
        return "application/json", json.dumps(
            {"object": "list", "data": data, "model": "e", "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        ).encode()
    if not body.get("stream"):
        message = {"role": "assistant", "content": item["text"]}
        return "application/json", json.dumps(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()

    def chunk(delta, finish=None):
        return (
            None,
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            },
        )

    events = [chunk({"role": "assistant", "content": ""})]
    if item.get("before_tools"):
        events.append(chunk({"content": item["before_tools"]}))
    if "tool_calls" in item:
        for i, call in enumerate(item["tool_calls"]):
            args = json.dumps(call["args"])
            events.append(
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {"name": call["name"], "arguments": ""},
                            }
                        ]
                    }
                )
            )
            events.append(chunk({"tool_calls": [{"index": i, "function": {"arguments": args[:5]}}]}))
            events.append(chunk({"tool_calls": [{"index": i, "function": {"arguments": args[5:]}}]}))
        events.append(chunk({}, "tool_calls"))
    else:
        text = item["text"]
        events += [chunk({"content": text[:4]}), chunk({"content": text[4:]}), chunk({}, "stop")]
    events.append((None, "[DONE]"))
    return "text/event-stream", sse(events)


@pytest.fixture
def openai_api(monkeypatch, settings):
    api = FakeAPI(openai_render)
    monkeypatch.setenv("OPENAI_BASE_URL", api.url + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    settings.LLM_EFFORT = ""
    settings.EMBEDDING_BASE_URL, settings.EMBEDDING_API_KEY = None, None
    llm_module._openai.cache_clear()
    embeddings_module._client.cache_clear()
    yield api
    api.close()
    llm_module._openai.cache_clear()
    embeddings_module._client.cache_clear()


def test_openai_stream(openai_api):
    openai_api.script = [{"text": "Hello world"}]
    llm = get_llm("openai", "gpt-test")
    assert list(llm.stream("Be nice.", [{"role": "user", "content": "hi"}])) == ["Hell", "o world"]
    body = openai_api.requests[0]["body"]
    assert body["model"] == "gpt-test" and body["stream"] is True
    assert body["messages"] == [{"role": "system", "content": "Be nice."}, {"role": "user", "content": "hi"}]
    assert "temperature" not in body and "reasoning_effort" not in body


def test_openai_parse(openai_api):
    openai_api.script = [{"text": '{"category": "b"}'}]
    choice = get_llm("openai", effort="low").parse("Pick one.", [{"role": "user", "content": "x"}], Choice)
    assert choice == Choice(category="b")
    body = openai_api.requests[0]["body"]
    assert body["response_format"]["type"] == "json_schema" and body["reasoning_effort"] == "low"


def test_openai_tool_loop(openai_api):
    openai_api.script = [
        {"tool_calls": [{"name": "search", "args": {"query": "red shoes"}}, {"name": "nope", "args": {}}]},
        {"text": "Found them [1]."},
    ]
    seen = []
    tool = Tool(
        "search",
        "Search products",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        run=lambda args: seen.append(args) or "[1] Red shoes",
    )
    streamed = []
    answer = get_llm("openai").run_tools(
        "sys", [{"role": "user", "content": "shoes?"}], [tool], on_text=streamed.append
    )

    assert answer == "Found them [1]." == "".join(streamed)
    assert seen == [{"query": "red shoes"}]
    second = openai_api.requests[1]["body"]["messages"]
    assert second[2]["tool_calls"][0]["function"] == {"name": "search", "arguments": '{"query": "red shoes"}'}
    assert second[3] == {"role": "tool", "tool_call_id": "call_0", "content": "[1] Red shoes"}
    assert second[4]["content"] == "Error: unknown tool 'nope'"


def test_openai_tool_loop_stops_after_max_rounds(openai_api):
    call = {"tool_calls": [{"name": "search", "args": {"query": "q"}}]}
    openai_api.script = [call, call, {"text": "Best I can do."}]
    tool = Tool("search", "Search", {"type": "object", "properties": {}}, run=lambda args: "nothing")
    answer = get_llm("openai").run_tools("sys", [{"role": "user", "content": "q"}], [tool], max_rounds=2)
    assert answer == "Best I can do."
    assert [r["body"]["tool_choice"] for r in openai_api.requests] == ["auto", "auto", "none"]


def test_embeddings(openai_api, settings):
    settings.EMBEDDING_BATCH_SIZE = 2
    vector = [0.1] * settings.EMBEDDING_DIMENSIONS
    openai_api.script = [{"embedding": vector}, {"embedding": vector}]
    assert embeddings_module.embed_texts(["a", "b", "c"]) == [vector] * 3
    assert [r["body"]["input"] for r in openai_api.requests] == [["a", "b"], ["c"]]
    assert openai_api.requests[0]["body"]["encoding_format"] == "float"

    openai_api.script = [{"embedding": [0.1] * 8}]
    with pytest.raises(ValueError, match="returned 8 dimensions"):
        embeddings_module.embed_texts(["a"])


# --- Anthropic ------------------------------------------------------------------------


def anthropic_render(item, body):
    usage = {"input_tokens": 1, "output_tokens": 1}
    if not body.get("stream"):
        return "application/json", json.dumps(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": item["text"]}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": usage,
            }
        ).encode()

    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage,
                },
            },
        )
    ]
    if "tool_calls" in item:
        for i, call in enumerate(item["tool_calls"]):
            args = json.dumps(call["args"])
            events += [
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "tool_use", "id": f"toolu_{i}", "name": call["name"], "input": {}},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": args[:5]},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": args[5:]},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": i}),
            ]
        stop = "tool_use"
    else:
        text = item["text"]
        events += [
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[:4]}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[4:]}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ]
        stop = item.get("stop_reason", "end_turn")
    events += [
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "text/event-stream", sse(events)


@pytest.fixture
def anthropic_api(monkeypatch, settings):
    api = FakeAPI(anthropic_render)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", api.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    settings.ANTHROPIC_FALLBACKS = True
    settings.LLM_EFFORT = ""
    llm_module._anthropic.cache_clear()
    yield api
    api.close()
    llm_module._anthropic.cache_clear()


def test_anthropic_stream(anthropic_api):
    anthropic_api.script = [{"text": "Hello world"}]
    llm = get_llm("anthropic")
    assert "".join(llm.stream("Be nice.", [{"role": "user", "content": "hi"}])) == "Hello world"
    request = anthropic_api.requests[0]
    assert request["body"]["model"] == "claude-opus-5-5"
    assert request["body"]["system"] == "Be nice." and request["body"]["max_tokens"] == 16000
    assert request["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request["headers"]["anthropic-beta"]
    assert "output_config" not in request["body"] and "temperature" not in request["body"]


def test_anthropic_refusal_raises(anthropic_api):
    anthropic_api.script = [{"text": "Sorry", "stop_reason": "refusal"}]
    with pytest.raises(llm_module.LLMError, match="declined"):
        list(get_llm("anthropic").stream("", [{"role": "user", "content": "x"}]))


def test_anthropic_parse_and_effort(anthropic_api, settings):
    settings.ANTHROPIC_FALLBACKS = False
    anthropic_api.script = [{"text": '{"category": "a"}'}]
    choice = get_llm("anthropic", effort="low").parse("Pick one.", [{"role": "user", "content": "x"}], Choice)
    assert choice == Choice(category="a")
    body = anthropic_api.requests[0]["body"]
    assert body["output_config"]["effort"] == "low" and "fallbacks" not in body


def test_anthropic_tool_loop(anthropic_api):
    anthropic_api.script = [
        {"tool_calls": [{"name": "search", "args": {"query": "red shoes"}}]},
        {"text": "Found them [1]."},
    ]
    seen = []
    tool = Tool(
        "search",
        "Search products",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        run=lambda args: seen.append(args) or "[1] Red shoes",
    )
    streamed = []
    answer = get_llm("anthropic").run_tools(
        "sys", [{"role": "user", "content": "shoes?"}], [tool], on_text=streamed.append
    )

    assert answer == "Found them [1]." == "".join(streamed)
    assert seen == [{"query": "red shoes"}]
    first, second = (r["body"] for r in anthropic_api.requests)
    assert first["tools"][0]["eager_input_streaming"] is True and first["tool_choice"] == {"type": "auto"}
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][1]["content"][0]["type"] == "tool_use"
    assert second["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_0", "content": "[1] Red shoes"}],
    }


def test_text_before_a_tool_call_is_kept(openai_api):
    openai_api.script = [
        {"before_tools": "Let me search.", "tool_calls": [{"name": "search", "args": {"query": "q"}}]},
        {"text": "Found it."},
    ]
    tool = Tool("search", "Search", {"type": "object", "properties": {}}, run=lambda args: "[1] x")
    streamed = []
    answer = get_llm("openai").run_tools("", [{"role": "user", "content": "q"}], [tool], on_text=streamed.append)
    assert answer == "Let me search.\n\nFound it." == "".join(streamed)
    first = openai_api.requests[0]["body"]
    assert first["messages"] == [{"role": "user", "content": "q"}]  # no empty system message


def test_anthropic_parse_stays_under_the_non_streaming_limit(anthropic_api, settings):
    settings.LLM_MAX_TOKENS = 64000
    anthropic_api.script = [{"text": '{"category": "a"}'}, {"text": "hi"}]
    get_llm("anthropic").parse("", [{"role": "user", "content": "x"}], Choice)
    assert anthropic_api.requests[0]["body"]["max_tokens"] == 16000
    assert "system" not in anthropic_api.requests[0]["body"]
    "".join(get_llm("anthropic").stream("", [{"role": "user", "content": "x"}]))
    assert anthropic_api.requests[1]["body"]["max_tokens"] == 64000  # streaming keeps the full budget
