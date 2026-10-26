"""What each node type does when it runs.

A node function gets the run state and returns an update:
  {"outputs": {node_id: {...}}, "answer": [text shown to the user], "references": [sources]}
It reports progress with stream events (see engine.py), and records what it worked
with (the prompts it sent, the query it searched...) with emit.inputs() for the run log.
"""

import time
from typing import Literal

from django.db import connection
from langgraph.config import get_stream_writer
from pydantic import create_model

from core.llm import Tool, get_llm
from knowledge.search import (
    FILTER_OPS,
    SORT_ORDERS,
    as_reference,
    check_filters,
    format_chunks,
    metadata_search,
    search,
)
from workflows.dsl import VAR_PATTERN, Node
from workflows.variables import render, resolve


class Emitter:
    """Sends events to the client and writes the user-visible answer, separating answer parts.

    In a debug run (the studio's test run) node events also carry the node's inputs and
    full outputs, and tool calls their results.
    """

    def __init__(self, state: dict):
        self.write = get_stream_writer()
        self.debug = bool(state.get("debug"))
        self.needs_separator = any(state["answer"])
        self.started = False
        self.recorded: dict = {}

    def event(self, name: str, **data) -> None:
        self.write({"event": name, **data})

    def inputs(self, **values) -> None:
        """Record what the node works with, for the run log."""
        self.recorded.update({k: v for k, v in values.items() if v is not None})

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
    return get_llm(params.provider, params.model, params.temperature, params.effort, params.max_tokens)


def model_info(llm) -> dict:
    """The model a node uses, with the server defaults filled in."""
    info = {
        "provider": llm.provider,
        "model": llm.model,
        "temperature": llm.temperature,
        "effort": llm.effort,
        "max_tokens": llm.max_tokens,
    }
    return {k: v for k, v in info.items() if v not in (None, "")}


def resolve_filters(filters, state: dict) -> tuple[list[dict], list[str]]:
    """Fill variables in filter values -> (filters, skipped).

    A filter whose value is empty when the node runs is skipped, e.g. a category from
    {begin@category} when the app sent none: then every category matches.
    """
    resolved, skipped = [], []
    for f in filters:
        value = f.value
        if isinstance(value, str) and VAR_PATTERN.search(value):
            value = resolve(value, state)
            if value is None or value == [] or (isinstance(value, str) and not value.strip()):
                skipped.append(f"{f.field} {f.op} {f.value}")
                continue
        resolved.append({"field": f.field, "op": f.op, "value": value})
    return check_filters(resolved), skipped


# --- Node types -------------------------------------------------------------------


def run_begin(node_id: str, node: Node, state: dict, emit: Emitter, streamed: bool) -> dict:
    emit.inputs(
        sys={k: v for k, v in state["sys"].items() if k != "history"},
        history=state["sys"]["history"],
        request_inputs=state.get("inputs") or {},
        defaults=node.parsed.inputs,
    )
    inputs = {**node.parsed.inputs, **(state.get("inputs") or {})}
    return {"outputs": {node_id: inputs}}


def run_retrieval(node_id, node, state, emit, streamed):
    p = node.parsed
    filters, skipped = resolve_filters(p.filters, state)
    emit.inputs(skipped_filters=skipped or None)
    sort = {"sort_by": p.sort_by, "sort_order": p.sort_order if p.sort_by else None}
    if p.mode == "metadata":
        query = ""
        emit.inputs(mode="metadata", kb_ids=p.kb_ids, filters=filters, **sort, top_k=p.top_k)
        chunks = metadata_search(p.kb_ids, p.top_k, filters=filters, sort_by=p.sort_by, sort_order=p.sort_order)
    else:
        query = render(p.query, state)
        emit.inputs(
            mode="search",
            query=query,
            kb_ids=p.kb_ids,
            filters=filters,
            **sort,
            sort_pool=p.sort_pool if p.sort_by else None,
            top_k=p.top_k,
            max_distance=p.max_distance,
        )
        chunks = search(
            query,
            p.kb_ids,
            p.top_k,
            max_distance=p.max_distance,
            filters=filters,
            sort_by=p.sort_by,
            sort_order=p.sort_order,
            sort_pool=p.sort_pool,
        )
    first = len(state["references"]) + 1
    refs = [as_reference(c, first + i) for i, c in enumerate(chunks)]
    content = format_chunks(chunks, start=first, template=p.chunk_template) if chunks else p.empty_response
    return {
        "outputs": {node_id: {"formalized_content": content, "chunks": refs, "count": len(refs), "query": query}},
        "references": refs,
    }


def run_llm(node_id, node, state, emit, streamed):
    p = node.parsed
    llm = llm_for(p)
    system = render(p.system_prompt, state)
    messages = chat_messages(state, p.history, render(p.prompt, state))
    emit.inputs(model=model_info(llm), system_prompt=system, messages=messages)
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


# --- Agent search tools ---------------------------------------------------------------


def tool_description(spec, state: dict) -> str:
    """The tool's description, plus the metadata instructions when the model can filter and sort."""
    text = render(spec.description, state)
    if spec.metadata_fields:
        fields = "\n".join(
            f"- {f.name} ({f.type})" + (f": {f.description}" if f.description else "") for f in spec.metadata_fields
        )
        text += "\n\n" + render(spec.metadata_prompt, state).replace("{fields}", fields)
    return text


def tool_parameters(spec) -> dict:
    """The JSON schema of the tool's arguments. With metadata fields, the query is optional."""
    properties = {"query": {"type": "string", "description": spec.query_description}}
    if not spec.metadata_fields:
        return {"type": "object", "properties": properties, "required": ["query"], "additionalProperties": False}
    names = [f.name for f in spec.metadata_fields]
    properties["filters"] = {
        "type": "array",
        "description": spec.filters_description,
        "items": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": names},
                "op": {"type": "string", "enum": list(FILTER_OPS)},
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}},
                    ]
                },
            },
            "required": ["field", "op", "value"],
            "additionalProperties": False,
        },
    }
    properties["sort_by"] = {"type": "string", "enum": names, "description": spec.sort_description}
    properties["order"] = {"type": "string", "enum": list(SORT_ORDERS), "description": spec.order_description}
    return {"type": "object", "properties": properties, "required": [], "additionalProperties": False}


def tool_arguments(spec, args: dict) -> tuple[str, list[dict], str | None, str]:
    """(query, filters, sort_by, order) from a tool call. Mistakes raise ValueError: the model sees them."""
    query = str(args.get("query") or "").strip()
    names = [f.name for f in spec.metadata_fields]
    if not names:
        if not query:
            raise ValueError("query is required")
        return query, [], None, "desc"
    known = ", ".join(names)
    filters = args.get("filters") or []
    if not isinstance(filters, list):
        raise ValueError("filters must be a list of {field, op, value}")
    for f in filters:
        if isinstance(f, dict) and f.get("field") not in names:
            raise ValueError(f"unknown field {f.get('field')!r}; use one of: {known}")
    sort_by = args.get("sort_by") or None
    if sort_by is not None and sort_by not in names:
        raise ValueError(f"unknown sort_by field {sort_by!r}; use one of: {known}")
    order = args.get("order") or "desc"
    if order not in SORT_ORDERS:
        raise ValueError('order must be "desc" or "asc"')
    return query, check_filters(filters), sort_by, order


def run_agent(node_id, node, state, emit, streamed):
    p = node.parsed
    refs: list[dict] = []
    first = len(state["references"]) + 1

    fixed_filters = {}  # tool name -> (filters, skipped), for the run log

    def make_tool(spec) -> Tool:
        fixed_filters[spec.name] = resolve_filters(spec.filters, state)
        fixed = fixed_filters[spec.name][0]

        def run(args: dict) -> str:
            started = time.monotonic()
            call = {"tool": spec.name, "arguments": args, "query": str(args.get("query") or "").strip()}
            try:
                query, filters, sort_by, order = tool_arguments(spec, args)
                if query:
                    chunks = search(
                        query,
                        spec.kb_ids,
                        spec.top_k,
                        max_distance=spec.max_distance,
                        filters=fixed + filters,
                        sort_by=sort_by,
                        sort_order=order,
                        sort_pool=spec.sort_pool,
                    )
                else:  # the model asked for metadata only, e.g. the newest products
                    chunks = metadata_search(
                        spec.kb_ids, spec.top_k, filters=fixed + filters, sort_by=sort_by, sort_order=order
                    )
                start = first + len(refs)
                refs.extend(as_reference(c, start + i) for i, c in enumerate(chunks))
                result = (
                    format_chunks(chunks, start=start, template=spec.chunk_template) if chunks else spec.empty_response
                )
                call["results"] = len(chunks)
                if emit.debug:
                    call["result"] = result
                return result
            except Exception as exc:
                call["error"] = str(exc)  # the model gets it as the tool result and can retry
                raise
            finally:
                call["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                emit.event("tool_call", node=node_id, **call)

        return Tool(
            name=spec.name, description=tool_description(spec, state), parameters=tool_parameters(spec), run=run
        )

    llm = llm_for(p)
    system = render(p.system_prompt, state)
    messages = chat_messages(state, p.history, render(p.prompt, state))
    tools = [make_tool(t) for t in p.tools]
    emit.inputs(
        model=model_info(llm),
        system_prompt=system,
        messages=messages,
        tools=[
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                **({"fixed_filters": fixed_filters[t.name][0]} if fixed_filters[t.name][0] else {}),
                **({"skipped_filters": fixed_filters[t.name][1]} if fixed_filters[t.name][1] else {}),
            }
            for t in tools
        ],
        max_rounds=p.max_rounds,
    )
    content = llm.run_tools(system, messages, tools, max_rounds=p.max_rounds, on_text=emit.text if streamed else None)
    update = {"outputs": {node_id: {"content": content}}, "references": refs}
    if streamed:
        update["answer"] = [content]
    return update


# --- Routing and messages ---------------------------------------------------------------


def run_categorize(node_id, node, state, emit, streamed):
    p = node.parsed
    lines = []
    for c in p.categories:
        lines.append(f"- {c.name}: {c.description}".rstrip(": "))
        lines.extend(f"    example: {e}" for e in c.examples)
    names = tuple(c.name for c in p.categories)
    choice_model = create_model("Choice", category=(Literal[names], ...))
    llm = llm_for(p)
    system = render(p.system_prompt, state).replace("{categories}", "\n".join(lines))
    messages = chat_messages(state, p.history, render(p.input, state))
    emit.inputs(model=model_info(llm), system_prompt=system, messages=messages, categories=list(names))
    choice = llm.parse(system, messages, choice_model)
    target = next(c.next for c in p.categories if c.name == choice.category)
    return {"outputs": {node_id: {"category": choice.category, "_next": target}}}


def run_switch(node_id, node, state, emit, streamed):
    p = node.parsed
    checked = []
    try:
        for i, case in enumerate(p.cases):
            conditions = []
            for c in case.conditions:
                left = resolve(c.var, state)
                right = resolve(c.value, state) if isinstance(c.value, str) else c.value
                conditions.append({"var": c.var, "value": left, "op": c.op, "compared_to": right})
                conditions[-1]["result"] = compare(left, c.op, right)
            results = [c["result"] for c in conditions]
            matched = all(results) if case.logic == "and" else any(results)
            checked.append({"case": i, "logic": case.logic, "conditions": conditions, "matched": matched})
            if matched:
                return {"outputs": {node_id: {"case": i, "_next": case.next}}}
        return {"outputs": {node_id: {"case": "default", "_next": p.default}}}
    finally:
        emit.inputs(cases=checked)


def run_message(node_id, node, state, emit, streamed):
    emit.inputs(text=node.parsed.text)
    if streamed:
        # The text is an LLM/agent output that was already streamed to the user.
        ref = node.parsed.text.strip()[1:-1]
        source_id, _, key = ref.partition("@")
        emit.inputs(streamed_from=source_id)
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

        def details() -> dict:
            return {"inputs": emit.recorded} if emit.debug else {}

        try:
            update = runner(node_id, node, state, emit, streamed)
        except Exception as exc:
            emit.event(
                "node_failed",
                node=node_id,
                type=node.type,
                elapsed_ms=round((time.monotonic() - start) * 1000),
                error=f"{type(exc).__name__}: {exc}",
                **details(),
            )
            raise
        result = update.get("outputs", {}).get(node_id, {})
        outputs = {k: v for k, v in result.items() if not k.startswith("_")}
        emit.event(
            "node_finished",
            node=node_id,
            type=node.type,
            elapsed_ms=round((time.monotonic() - start) * 1000),
            outputs=outputs if emit.debug else _preview(outputs),
            next=result["_next"] if "_next" in result else node.next,
            **details(),
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
