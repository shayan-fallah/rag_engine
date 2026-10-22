"""What each node type does when it runs.

A node function gets the run state and returns an update:
  {"outputs": {node_id: {...}}, "answer": [text shown to the user], "references": [sources]}
It reports progress with stream events (see engine.py).
"""

import time
from typing import Literal

from django.db import connection
from langgraph.config import get_stream_writer
from pydantic import create_model

from core.llm import Tool, get_llm
from knowledge.search import as_reference, format_chunks, search
from workflows.dsl import Node
from workflows.variables import render, resolve

CATEGORIZE_PROMPT = """You sort the user's message into exactly one of these categories.

{categories}

Answer with the name of the best matching category."""


class Emitter:
    """Sends events to the client and writes the user-visible answer, separating answer parts."""

    def __init__(self, state: dict):
        self.write = get_stream_writer()
        self.needs_separator = any(state["answer"])
        self.started = False

    def event(self, name: str, **data) -> None:
        self.write({"event": name, **data})

    def text(self, delta: str) -> None:
        if not delta:
            return
        if not self.started and self.needs_separator:
            self.write({"event": "message", "delta": "\n\n"})
        self.started = True
        self.write({"event": "message", "delta": delta})


def chat_messages(state: dict, history: int, prompt: str) -> list[dict]:
    messages = state["sys"]["history"][-history:] if history else []
    messages = [m for m in messages if m["content"].strip()]  # e.g. a run that ended without an answer
    while messages and messages[0]["role"] != "user":
        messages = messages[1:]  # a conversation must start with the user
    return [*messages, {"role": "user", "content": prompt}]


def llm_for(params):
    return get_llm(params.provider, params.model, params.temperature, params.effort)


# --- Node types -------------------------------------------------------------------


def run_begin(node_id: str, node: Node, state: dict, emit: Emitter, streamed: bool) -> dict:
    inputs = {**node.parsed.inputs, **(state.get("inputs") or {})}
    return {"outputs": {node_id: inputs}}


def run_retrieval(node_id, node, state, emit, streamed):
    p = node.parsed
    query = render(p.query, state)
    chunks = search(query, p.kb_ids, p.top_k, p.metadata_filter, p.max_distance)
    first = len(state["references"]) + 1
    refs = [as_reference(c, first + i) for i, c in enumerate(chunks)]
    content = format_chunks(chunks, start=first) if chunks else p.empty_response
    return {
        "outputs": {node_id: {"formalized_content": content, "chunks": refs, "count": len(refs), "query": query}},
        "references": refs,
    }


def run_llm(node_id, node, state, emit, streamed):
    p = node.parsed
    llm = llm_for(p)
    system = render(p.system_prompt, state)
    messages = chat_messages(state, p.history, render(p.prompt, state))
    if streamed:
        parts = []
        for delta in llm.stream(system, messages):
            parts.append(delta)
            emit.text(delta)
        content = "".join(parts)
    else:
        content = llm.complete(system, messages)
    update = {"outputs": {node_id: {"content": content}}}
    if streamed:
        update["answer"] = [content]
    return update


def run_agent(node_id, node, state, emit, streamed):
    p = node.parsed
    refs: list[dict] = []
    first = len(state["references"]) + 1

    def make_tool(spec) -> Tool:
        def run(args: dict) -> str:
            query = str(args.get("query", "")).strip()
            chunks = search(query, spec.kb_ids, spec.top_k, spec.metadata_filter, spec.max_distance)
            start = first + len(refs)
            refs.extend(as_reference(c, start + i) for i, c in enumerate(chunks))
            emit.event("tool_call", node=node_id, tool=spec.name, query=query, results=len(chunks))
            return format_chunks(chunks, start=start) if chunks else "No results."

        return Tool(
            name=spec.name,
            description=spec.description,
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            run=run,
        )

    content = llm_for(p).run_tools(
        render(p.system_prompt, state),
        chat_messages(state, p.history, render(p.prompt, state)),
        [make_tool(t) for t in p.tools],
        max_rounds=p.max_rounds,
        on_text=emit.text if streamed else None,
    )
    update = {"outputs": {node_id: {"content": content}}, "references": refs}
    if streamed:
        update["answer"] = [content]
    return update


def run_categorize(node_id, node, state, emit, streamed):
    p = node.parsed
    lines = []
    for c in p.categories:
        lines.append(f"- {c.name}: {c.description}".rstrip(": "))
        lines.extend(f"    example: {e}" for e in c.examples)
    names = tuple(c.name for c in p.categories)
    choice_model = create_model("Choice", category=(Literal[names], ...))
    choice = llm_for(p).parse(
        CATEGORIZE_PROMPT.format(categories="\n".join(lines)),
        chat_messages(state, p.history, render(p.input, state)),
        choice_model,
    )
    target = next(c.next for c in p.categories if c.name == choice.category)
    return {"outputs": {node_id: {"category": choice.category, "_next": target}}}


def run_switch(node_id, node, state, emit, streamed):
    p = node.parsed
    for i, case in enumerate(p.cases):
        results = [
            compare(resolve(c.var, state), c.op, resolve(c.value, state) if isinstance(c.value, str) else c.value)
            for c in case.conditions
        ]
        if all(results) if case.logic == "and" else any(results):
            return {"outputs": {node_id: {"case": i, "_next": case.next}}}
    return {"outputs": {node_id: {"case": "default", "_next": p.default}}}


def run_message(node_id, node, state, emit, streamed):
    if streamed:
        # The text is an LLM/agent output that was already streamed to the user.
        ref = node.parsed.text.strip()[1:-1]
        source_id, _, key = ref.partition("@")
        return {"outputs": {node_id: {"content": state["outputs"].get(source_id, {}).get(key, "")}}}
    text = render(node.parsed.text, state)
    emit.text(text)
    return {"outputs": {node_id: {"content": text}}, "answer": [text]}


RUNNERS = {
    "begin": run_begin,
    "retrieval": run_retrieval,
    "llm": run_llm,
    "agent": run_agent,
    "categorize": run_categorize,
    "switch": run_switch,
    "message": run_message,
}


def compare(left, op: str, right) -> bool:
    if op == "empty":
        return (
            left is None
            or (hasattr(left, "__len__") and len(left) == 0)
            or (isinstance(left, str) and not left.strip())
        )
    if op == "not_empty":
        return not compare(left, "empty", right)
    if op in ("contains", "not_contains"):
        if isinstance(left, (list, tuple, dict)):
            found = right in left
        else:
            found = str(right).lower() in ("" if left is None else str(left)).lower()
        return found if op == "contains" else not found
    a, b = _number(left), _number(right)
    if a is None or b is None:  # not both numbers: compare as text
        a, b = ("" if left is None else str(left)), ("" if right is None else str(right))
    return {
        "==": a == b,
        "!=": a != b,
        ">": a > b,
        ">=": a >= b,
        "<": a < b,
        "<=": a <= b,
    }[op]


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def make_node(node_id: str, node: Node, streamed: bool):
    """Wrap a node type's runner with start/finish events for the LangGraph graph."""
    runner = RUNNERS[node.type]

    def run(state: dict) -> dict:
        # LangGraph runs nodes on its own threads while it streams events, and Django
        # opens one database connection per thread. Close the ones opened here.
        had_connection = connection.connection is not None
        try:
            return run_node(state)
        finally:
            if not had_connection:
                connection.close()

    def run_node(state: dict) -> dict:
        emit = Emitter(state)
        emit.event("node_started", node=node_id, type=node.type)
        start = time.monotonic()
        update = runner(node_id, node, state, emit, streamed)
        outputs = {k: v for k, v in update.get("outputs", {}).get(node_id, {}).items() if not k.startswith("_")}
        emit.event(
            "node_finished",
            node=node_id,
            type=node.type,
            elapsed_ms=round((time.monotonic() - start) * 1000),
            outputs=_preview(outputs),
        )
        return update

    return run


def _preview(outputs: dict) -> dict:
    preview = {}
    for key, value in outputs.items():
        if isinstance(value, str) and len(value) > 300:
            value = value[:300] + "…"
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            value = f"[{len(value)} items]"
        preview[key] = value
    return preview
