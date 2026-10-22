import copy
import json
from pathlib import Path

import pytest

from knowledge.ingest import DocInput, ingest_documents
from knowledge.models import KnowledgeBase
from tests.conftest import PRODUCTS
from tests.fakes import ScriptedLLM
from workflows.dsl import validate_dsl
from workflows.engine import WorkflowError, run_workflow
from workflows.models import Workflow
from workflows.nodes import compare

TEMPLATES = Path(__file__).resolve().parent.parent / "workflows" / "templates"


def template(name: str, kb_id: int) -> dict:
    """A shipped template, pointed at the test knowledge base."""
    text = (TEMPLATES / f"{name}.json").read_text().replace('"kb_ids": [1]', f'"kb_ids": [{kb_id}]')
    return json.loads(text)["dsl"]


def make_workflow(dsl, name="wf") -> Workflow:
    validate_dsl(dsl)
    return Workflow.objects.create(name=name, dsl=dsl)


def run(workflow, query, **kwargs) -> tuple[list[dict], dict]:
    events = list(run_workflow(workflow, query, user_id="u1", **kwargs))
    return events, events[-1]


def streamed_text(events) -> str:
    return "".join(e["delta"] for e in events if e["event"] == "message")


@pytest.fixture
def products(transactional_db, embeddings):
    # Nodes run on LangGraph's threads, with their own database connection,
    # so the test data must be committed (transactional_db), not in the test's transaction.
    kb = KnowledgeBase.objects.create(name="products")
    ingest_documents(kb, [DocInput(pid, desc, title, meta) for pid, title, desc, meta in PRODUCTS])
    return kb


# --- Validation -------------------------------------------------------------------


@pytest.mark.parametrize(
    "change, error",
    [
        (lambda d: d["nodes"]["begin"].update(next="nope"), "next node 'nope' does not exist"),
        (
            lambda d: d["nodes"]["answer"]["params"].update(prompt="{ghost@content}"),
            "refers to a node that does not exist",
        ),
        (lambda d: d["nodes"]["answer"]["params"].update(prompt="{sys.weather}"), "unknown variable {sys.weather}"),
        (lambda d: d["nodes"]["reply"].update(type="begin", params={}), "exactly one begin node"),
        (lambda d: d["nodes"]["search"]["params"].update(top_k=500), "top_k"),
        (lambda d: d["nodes"]["search"]["params"].update(kb_ids=[9999]), "[9999] do not exist"),
        (lambda d: d["nodes"]["answer"].update(colour="red"), "colour"),
    ],
)
def test_validation_errors(kb, change, error):
    dsl = template("rag_chat", kb.id)
    change(dsl)
    with pytest.raises(ValueError, match=error.replace("[", r"\[").replace("{", r"\{")):
        validate_dsl(dsl)


def test_shipped_templates_are_valid(kb):
    for path in TEMPLATES.glob("*.json"):
        validate_dsl(template(path.stem, kb.id))


# --- Running ----------------------------------------------------------------------


def test_rag_chat_streams_answer_with_references(products, use_llm):
    llm = use_llm(ScriptedLLM(reply=lambda system, messages: "The Pixel 9 has a great camera [2]."))
    wf = make_workflow(template("rag_chat", products.id))

    events, done = run(wf, "which phone has a great camera?")

    assert done["event"] == "done"
    assert done["answer"] == "The Pixel 9 has a great camera [2]."
    assert streamed_text(events) == done["answer"]  # streamed token by token, sent once
    assert {r["external_id"] for r in done["references"]} >= {"p1", "p2"}
    assert [r["ref"] for r in done["references"]] == list(range(1, len(done["references"]) + 1))
    assert [r["ref"] for r in done["references"] if r["cited"]] == [2]
    # the retrieved products were in the prompt
    assert "[1]" in llm.calls[0]["system"] and "Pixel 9" in llm.calls[0]["system"]
    started = [e["node"] for e in events if e["event"] == "node_started"]
    assert started == ["begin", "search", "answer", "reply"]


def test_history_reaches_the_model(products, use_llm):
    llm = use_llm(ScriptedLLM(reply=lambda s, m: "ok"))
    wf = make_workflow(template("rag_chat", products.id))
    history = [
        {"role": "assistant", "content": "Hi! How can I help?"},  # dropped: must start with the user
        {"role": "user", "content": "I need a phone"},
        {"role": "assistant", "content": "We have Galaxy and Pixel."},
    ]
    run(wf, "which one is cheaper?", history=history)
    assert llm.calls[0]["messages"] == [*history[1:], {"role": "user", "content": "which one is cheaper?"}]


def test_product_assistant_routes(products, use_llm):
    def choose(system, messages):
        text = messages[-1]["content"]
        return "smalltalk" if "hello" in text else "order_status" if "order" in text else "product_question"

    def reply(system, messages):
        if "Rewrite" in system:
            return "trail running shoes"
        if "friendly" in system:
            return "Hello! Ask me about our products."
        return "Try the trail running shoes [1]."

    llm = use_llm(ScriptedLLM(reply=reply, choose=choose))
    wf = make_workflow(template("product_assistant", products.id))

    _, done = run(wf, "hello there")
    assert done["answer"] == "Hello! Ask me about our products." and done["references"] == []

    calls_before = len(llm.calls)
    _, done = run(wf, "where is my order 123?")
    assert "support" in done["answer"]
    assert [c["kind"] for c in llm.calls[calls_before:]] == ["parse"]  # only the classifier ran

    events, done = run(wf, "shoes for mountains")
    assert done["answer"] == "Try the trail running shoes [1]."
    search = next(e for e in events if e["event"] == "node_finished" and e["node"] == "search")
    assert search["outputs"]["query"] == "trail running shoes"  # the rewritten question was searched
    assert done["references"][0]["external_id"] == "p4"


def test_switch_sends_empty_results_to_a_fixed_message(products, use_llm):
    use_llm(ScriptedLLM(reply=lambda s, m: "zebra xylophone", choose=lambda s, m: "product_question"))
    dsl = template("product_assistant", products.id)
    dsl["nodes"]["search"]["params"]["max_distance"] = 0.3
    wf = make_workflow(dsl)
    _, done = run(wf, "zebra xylophone")
    assert done["answer"] == dsl["nodes"]["not_found"]["params"]["text"]


def test_agent_searches_with_tools(products, use_llm):
    llm = use_llm(
        ScriptedLLM(
            tool_plan=lambda s, m: [
                ("search_products", {"query": "noise cancelling headphones"}),
                ("search_products", {"query": "running shoes"}),
            ],
            reply=lambda s, m: "Bose QuietComfort [1] and trail shoes [3].",
        )
    )
    wf = make_workflow(template("product_agent", products.id))
    events, done = run(wf, "gift ideas for a runner who likes music?")

    tool_calls = [e for e in events if e["event"] == "tool_call"]
    assert [t["query"] for t in tool_calls] == ["noise cancelling headphones", "running shoes"]
    refs = done["references"]
    assert refs[0]["external_id"] == "p3" and [r["ref"] for r in refs] == list(range(1, len(refs) + 1))
    assert done["answer"] == "Bose QuietComfort [1] and trail shoes [3]." == streamed_text(events)
    assert "[1] QuietComfort" in llm.calls[-1]["messages"][-1]["content"]  # numbered results went back to the model


def test_two_message_parts_are_separated(kb, use_llm):
    use_llm(ScriptedLLM(reply=lambda s, m: "Answer."))
    wf = make_workflow(
        {
            "nodes": {
                "begin": {"type": "begin", "next": "hello"},
                "hello": {"type": "message", "params": {"text": "Hi {sys.user_id}!"}, "next": "llm"},
                "llm": {"type": "llm", "next": "out"},
                "out": {"type": "message", "params": {"text": "{llm@content}"}},
            }
        }
    )
    events, done = run(wf, "q")
    assert done["answer"] == "Hi u1!\n\nAnswer." == streamed_text(events)


def test_begin_inputs_and_loops_are_limited(kb, use_llm):
    use_llm(ScriptedLLM())
    wf = make_workflow(
        {
            "max_steps": 10,
            "nodes": {
                "begin": {"type": "begin", "params": {"inputs": {"lang": "English"}}, "next": "loop"},
                "loop": {
                    "type": "switch",
                    "params": {
                        "cases": [
                            {"conditions": [{"var": "{begin@lang}", "op": "==", "value": "English"}], "next": "say"}
                        ]
                    },
                },
                "say": {"type": "message", "params": {"text": "in {begin@lang}"}, "next": "loop"},
            },
        }
    )
    with pytest.raises(WorkflowError, match="more than 10 steps"):
        run(wf, "q")
    events, done = run(wf, "q", inputs={"lang": "Persian"})  # switch default -> end
    assert done["answer"] == ""


def test_workflow_changes_take_effect(kb, use_llm):
    use_llm(ScriptedLLM())
    dsl = {"nodes": {"begin": {"type": "begin", "next": "m"}, "m": {"type": "message", "params": {"text": "v1"}}}}
    wf = make_workflow(dsl)
    assert run(wf, "q")[1]["answer"] == "v1"
    dsl2 = copy.deepcopy(dsl)
    dsl2["nodes"]["m"]["params"]["text"] = "v2"
    wf.dsl = dsl2
    wf.save()
    assert run(wf, "q")[1]["answer"] == "v2"


@pytest.mark.parametrize(
    "left, op, right, expected",
    [
        (0, "==", 0, True),
        ("3", ">", 2, True),
        (10, "<=", "9", False),
        ("abc", "==", "abc", True),
        ("Hello World", "contains", "world", True),
        (["a"], "not_contains", "a", False),
        ("", "empty", None, True),
        ([], "empty", None, True),
        (None, "not_empty", None, False),
        ("x", "!=", "y", True),
    ],
)
def test_compare(left, op, right, expected):
    assert compare(left, op, right) is expected


def test_branch_not_taken_is_not_streamed(kb, use_llm):
    use_llm(ScriptedLLM(reply=lambda s, m: "DRAFT"))
    wf = make_workflow(
        {
            "nodes": {
                "begin": {"type": "begin", "next": "draft"},
                "draft": {"type": "llm", "next": "check"},
                "check": {
                    "type": "switch",
                    "params": {
                        "cases": [
                            {"conditions": [{"var": "{sys.query}", "op": "==", "value": "show"}], "next": "show"}
                        ],
                        "default": "sorry",
                    },
                },
                "show": {"type": "message", "params": {"text": "{draft@content}"}},
                "sorry": {"type": "message", "params": {"text": "Sorry."}},
            }
        }
    )
    events, done = run(wf, "hide")
    assert done["answer"] == "Sorry." == streamed_text(events)
    events, done = run(wf, "show")
    assert done["answer"] == "DRAFT" == streamed_text(events)


def test_switch_value_can_be_a_variable(kb, use_llm):
    use_llm(ScriptedLLM())
    wf = make_workflow(
        {
            "nodes": {
                "begin": {"type": "begin", "params": {"inputs": {"limit": 3, "n": 5}}, "next": "check"},
                "check": {
                    "type": "switch",
                    "params": {
                        "cases": [
                            {"conditions": [{"var": "{begin@n}", "op": ">", "value": "{begin@limit}"}], "next": "big"}
                        ],
                        "default": "small",
                    },
                },
                "big": {"type": "message", "params": {"text": "big"}},
                "small": {"type": "message", "params": {"text": "small"}},
            }
        }
    )
    assert run(wf, "q")[1]["answer"] == "big"
    assert run(wf, "q", inputs={"n": 2})[1]["answer"] == "small"


def test_empty_messages_are_left_out_of_history(kb, embeddings, use_llm):
    llm = use_llm(ScriptedLLM(reply=lambda s, m: "ok"))
    wf = make_workflow(template("rag_chat", kb.id))
    history = [{"role": "user", "content": "one"}, {"role": "assistant", "content": ""}]
    run(wf, "two", history=history)
    assert llm.calls[0]["messages"] == [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
