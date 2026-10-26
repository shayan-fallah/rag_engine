"""Run a workflow with LangGraph.

Each node becomes a LangGraph node, "next" becomes an edge, and categorize/switch
nodes become conditional edges. The run state is:

    sys         {query, user_id, conversation_id, date, history, turns}
    inputs      values for the begin node, from the request
    debug       true: events carry each node's inputs and full outputs (the studio's run log)
    outputs     {node_id: {output: value}}        (merged as nodes finish)
    answer      [text shown to the user]           (appended)
    references  [chunks the answer is based on]    (appended)

run_workflow() yields events as dicts:
    node_started   {node, type}
    node_finished  {node, type, elapsed_ms, outputs, next}     (+ inputs when debug)
    node_failed    {node, type, elapsed_ms, error}             (+ inputs when debug)
    tool_call      {node, tool, arguments, query, results, elapsed_ms, error?}   (agent nodes; + result when debug)
    message        {delta}                         (answer text, as it is written)
    done           {answer, references}           (references[].cited: the answer cites it as [n])

Without debug, long outputs are shortened (lists of results become "[n items]").
"""

import operator
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from workflows.dsl import VAR_PATTERN, WorkflowSpec, validate_dsl
from workflows.nodes import make_node


def merge(old: dict, new: dict) -> dict:
    return {**old, **new}


class RunState(TypedDict):
    sys: dict
    inputs: dict
    debug: bool
    outputs: Annotated[dict, merge]
    answer: Annotated[list[str], operator.add]
    references: Annotated[list[dict], operator.add]


def passthrough_messages(spec: WorkflowSpec) -> dict[str, str]:
    """Message nodes that show, as-is, the output of the LLM/agent node right before them -> that node's id.

    e.g. answer (llm, next: reply) -> reply (message, text: "{answer@content}").
    The LLM/agent node then streams its tokens to the user as they are generated,
    and the message node doesn't send the text again. Only the node directly before
    counts, so a branch that doesn't show the text never streams it.
    """
    found = {}
    for node_id, node in spec.nodes.items():
        if node.type == "message":
            match = VAR_PATTERN.fullmatch(node.parsed.text.strip())
            if match and "@" in match.group(1):
                source_id, _, key = match.group(1).partition("@")
                source = spec.nodes[source_id]
                if key == "content" and source.type in ("llm", "agent") and source.next == node_id:
                    found[node_id] = source_id
    return found


def build_graph(spec: WorkflowSpec):
    passthrough = passthrough_messages(spec)
    streamed = set(passthrough) | set(passthrough.values())
    graph = StateGraph(RunState)
    for node_id, node in spec.nodes.items():
        graph.add_node(node_id, make_node(node_id, node, streamed=node_id in streamed))

    graph.add_edge(START, spec.begin_id)
    for node_id, node in spec.nodes.items():
        if node.type in ("categorize", "switch"):
            targets = {t or END for t in node.targets()}
            graph.add_conditional_edges(
                node_id,
                lambda state, nid=node_id: state["outputs"][nid]["_next"] or END,
                {t: t for t in targets},
            )
        else:
            graph.add_edge(node_id, node.next or END)
    return graph.compile()


_graphs: dict[tuple, object] = {}


def get_graph(workflow) -> tuple[WorkflowSpec, object]:
    """Compiled graphs are cached until the workflow changes."""
    key = (workflow.pk, workflow.updated_at)
    if key not in _graphs:
        spec = validate_dsl(workflow.dsl, check_kbs=False)
        if len(_graphs) > 200:
            _graphs.clear()
        _graphs[key] = (spec, build_graph(spec))
    return _graphs[key]


class WorkflowError(RuntimeError):
    pass


def run_workflow(
    workflow,
    query: str,
    user_id: str = "",
    conversation_id: str = "",
    history: list[dict] | None = None,
    inputs: dict | None = None,
) -> Iterator[dict]:
    spec, graph = get_graph(workflow)
    yield from run_graph(spec, graph, query, user_id, conversation_id, history, inputs)


def run_draft(
    dsl: dict, query: str, user_id: str = "", history: list[dict] | None = None, inputs: dict | None = None
) -> Iterator[dict]:
    """Run a workflow that isn't saved (the studio's test panel), with debug events.

    An invalid workflow raises DSLError right away.
    """
    spec = validate_dsl(dsl)
    return run_graph(spec, build_graph(spec), query, user_id, "", history, inputs, debug=True)


def run_graph(
    spec: WorkflowSpec,
    graph,
    query: str,
    user_id: str = "",
    conversation_id: str = "",
    history: list[dict] | None = None,
    inputs: dict | None = None,
    debug: bool = False,
) -> Iterator[dict]:
    history = history or []
    state = {
        "sys": {
            "query": query,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "history": history,
            "turns": sum(1 for m in history if m["role"] == "user"),
        },
        "inputs": inputs or {},
        "debug": debug,
        "outputs": {},
        "answer": [],
        "references": [],
    }
    final = state
    try:
        for mode, chunk in graph.stream(
            state, stream_mode=["custom", "values"], config={"recursion_limit": spec.max_steps}
        ):
            if mode == "custom":
                yield chunk
            else:
                final = chunk
    except GraphRecursionError as exc:
        raise WorkflowError(f"The workflow ran more than {spec.max_steps} steps (a loop that never ends?)") from exc

    answer = "\n\n".join(part for part in final["answer"] if part)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    yield {
        "event": "done",
        "answer": answer,
        # Everything that was retrieved; "cited" marks the ones the answer refers to as [n].
        "references": [{**ref, "cited": ref["ref"] in cited} for ref in final["references"]],
    }
