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


# --- Settings that used to be internal ----------------------------------------------


def test_categorize_system_prompt_is_a_setting(kb, use_llm):
    llm = use_llm(ScriptedLLM(choose=lambda s, m: "a"))
    dsl = {
        "nodes": {
            "begin": {"type": "begin", "next": "pick"},
            "pick": {
                "type": "categorize",
                "params": {
                    "system_prompt": "Choose for user {sys.user_id}:\n{categories}",
                    "categories": [
                        {"name": "a", "description": "first", "examples": ["x"], "next": "out"},
                        {"name": "b"},
                    ],
                },
            },
            "out": {"type": "message", "params": {"text": "{pick@category}"}},
        }
    }
    _, done = run(make_workflow(dsl), "q")
    assert done["answer"] == "a"
    assert llm.calls[0]["system"] == "Choose for user u1:\n- a: first\n    example: x\n- b"


def test_retrieval_result_format(products, use_llm):
    use_llm(ScriptedLLM())
    dsl = {
        "nodes": {
            "begin": {"type": "begin", "next": "search"},
            "search": {
                "type": "retrieval",
                "params": {"kb_ids": [products.id], "top_k": 1, "chunk_template": "#{ref} {title}: {metadata.price}"},
                "next": "out",
            },
            "out": {"type": "message", "params": {"text": "{search@formalized_content}"}},
        }
    }
    _, done = run(make_workflow(dsl), "noise cancelling headphones")
    assert done["answer"] == "#1 QuietComfort headphones: 349"


def test_agent_tool_texts_are_settings(products, use_llm):
    class Capture(ScriptedLLM):
        def run_tools(self, system, messages, tools, max_rounds=5, on_text=None):
            self.tools = tools
            return super().run_tools(system, messages, tools, max_rounds, on_text)

    llm = use_llm(Capture(tool_plan=lambda s, m: [("find", {"query": "zebra xylophone"})]))
    tool = {
        "name": "find",
        "description": "Find products",
        "kb_ids": [products.id],
        "max_distance": 0.01,
        "query_description": "Product words in English",
        "empty_response": "Nothing matched; try other words.",
    }
    dsl = {
        "nodes": {
            "begin": {"type": "begin", "next": "agent"},
            "agent": {"type": "agent", "params": {"tools": [tool], "max_tokens": 900}},
        }
    }
    run(make_workflow(dsl), "q")
    assert llm.tools[0].parameters["properties"]["query"]["description"] == "Product words in English"
    assert llm.calls[-1]["messages"][-1]["content"] == "Nothing matched; try other words."


def test_layout_is_part_of_the_workflow(kb):
    dsl = template("rag_chat", kb.id)
    dsl["layout"] = {"begin": {"x": 0, "y": 0}, "search": {"x": 300, "y": 10.5}}
    validate_dsl(dsl)
    dsl["layout"]["search"] = {"x": "left"}
    with pytest.raises(ValueError, match="layout"):
        validate_dsl(dsl)


def test_errors_name_their_node(kb):
    from workflows.dsl import DSLError

    dsl = template("rag_chat", kb.id)
    dsl["nodes"]["search"]["params"]["top_k"] = 0
    dsl["nodes"]["answer"]["params"]["prompt"] = "{nope@content}"
    dsl["nodes"]["reply"]["colour"] = "red"
    with pytest.raises(DSLError) as exc:
        validate_dsl(dsl)
    assert {node for node, _ in exc.value.errors} == {"reply"}  # the shape is checked first
    del dsl["nodes"]["reply"]["colour"]
    with pytest.raises(DSLError) as exc:
        validate_dsl(dsl)
    assert sorted(node for node, _ in exc.value.errors) == ["answer", "search"]


# --- Metadata search --------------------------------------------------------------


@pytest.fixture
def catalog(transactional_db, embeddings):
    from tests.test_knowledge import CATALOG

    kb = KnowledgeBase.objects.create(name="catalog")
    ingest_documents(kb, [DocInput(pid, text, title, meta) for pid, title, text, meta in CATALOG])
    return kb


def test_retrieval_by_metadata_only(catalog, use_llm):
    use_llm(ScriptedLLM())
    latest = {
        "kb_ids": [catalog.id],
        "mode": "metadata",
        "sort_by": "date_added",
        "top_k": 2,
        "filters": [{"field": "category", "op": "==", "value": "{begin@category}"}],
        "chunk_template": "{title}",
    }
    dsl = {
        "nodes": {
            "begin": {"type": "begin", "params": {"inputs": {"category": ""}}, "next": "latest"},
            "latest": {"type": "retrieval", "params": latest, "next": "out"},
            "out": {"type": "message", "params": {"text": "{latest@formalized_content}"}},
        }
    }
    wf = make_workflow(dsl)
    _, done = run(wf, "what's new?")
    assert done["answer"] == "QuietComfort headphones\n\nPixel 9 phone"  # empty variable: the filter is skipped
    assert done["references"][0]["score"] is None
    _, done = run(wf, "what's new?", inputs={"category": "phones"})
    assert done["answer"] == "Pixel 9 phone\n\nGalaxy S24 phone"

    from workflows.engine import run_draft

    events = list(run_draft(dsl, "what's new?"))
    latest_log = next(e for e in events if e["event"] == "node_finished" and e["node"] == "latest")["inputs"]
    assert latest_log["skipped_filters"] == ["category == {begin@category}"] and latest_log["filters"] == []


def test_old_metadata_filter_still_works(products, use_llm):
    use_llm(ScriptedLLM())
    dsl = template("rag_chat", products.id)
    dsl["nodes"]["search"]["params"]["metadata_filter"] = {"category": "audio"}
    events, _ = run(make_workflow(dsl), "phone")
    assert [r["external_id"] for r in _["references"]] == ["p3"]


class ToolRunner(ScriptedLLM):
    """Runs the planned tool calls like the real models do: errors come back to the model as text."""

    def run_tools(self, system, messages, tools, max_rounds=5, on_text=None):
        from core.llm import _call_tool

        self.tools = tools
        by_name = {t.name: t for t in tools}
        self.results = [_call_tool(by_name, name, args) for name, args in self.tool_plan(system, messages)]
        return "done"


def test_agent_chooses_metadata_or_query_search(catalog, use_llm):
    llm = use_llm(
        ToolRunner(
            tool_plan=lambda s, m: [
                ("products", {"sort_by": "date_added"}),  # "the newest product": metadata only
                ("products", {"query": "smartphone", "sort_by": "date_added", "order": "desc"}),  # the newest phones
                (
                    "products",
                    {"filters": [{"field": "price", "op": "<", "value": 500}], "sort_by": "price", "order": "asc"},
                ),
                ("products", {"filters": [{"field": "colour", "op": "==", "value": "red"}]}),
                ("products", {"query": "x", "order": "newest"}),
            ]
        )
    )
    tool = {
        "name": "products",
        "description": "Our products",
        "kb_ids": [catalog.id],
        "top_k": 2,
        "sort_pool": 3,
        "chunk_template": "{title}",
        "filters": [{"field": "category", "op": "!=", "value": "Shoes"}],  # always applied
        "metadata_fields": [
            {"name": "date_added", "type": "date", "description": "When it was added"},
            {"name": "price", "type": "number"},
        ],
    }
    dsl = {
        "nodes": {
            "begin": {"type": "begin", "next": "agent"},
            "agent": {"type": "agent", "params": {"tools": [tool]}},
        }
    }
    events, _ = run(make_workflow(dsl), "q")

    spec = llm.tools[0]
    assert spec.parameters["required"] == [] and spec.parameters["properties"]["sort_by"]["enum"] == [
        "date_added",
        "price",
    ]
    assert spec.description.startswith("Our products\n\nYou can also filter and sort by these metadata fields:\n")
    assert "- date_added (date): When it was added\n- price (number)\n" in spec.description
    assert "Today is 20" in spec.description  # {sys.date} is filled in

    newest, newest_phones, cheapest, unknown, bad_order = llm.results
    assert newest == "QuietComfort headphones\n\nPixel 9 phone"
    assert newest_phones == "Pixel 9 phone\n\nGalaxy S24 phone"  # the 3 best matches, newest first
    assert cheapest == "Redmi Note phone\n\nQuietComfort headphones"
    assert unknown == "Error: unknown field 'colour'; use one of: date_added, price"
    assert bad_order == 'Error: order must be "desc" or "asc"'

    calls = [e for e in events if e["event"] == "tool_call"]
    assert calls[0]["query"] == "" and calls[0]["arguments"] == {"sort_by": "date_added"} and calls[0]["results"] == 2
    assert "result" not in calls[0]  # tool results only in debug runs
    assert calls[3]["error"].startswith("unknown field")


def test_tool_without_fields_needs_a_query(products, use_llm):
    llm = use_llm(ToolRunner(tool_plan=lambda s, m: [("find", {"query": " "})]))
    tool = {"name": "find", "description": "Find products", "kb_ids": [products.id]}
    dsl = {"nodes": {"begin": {"type": "begin", "next": "a"}, "a": {"type": "agent", "params": {"tools": [tool]}}}}
    run(make_workflow(dsl), "q")
    assert llm.tools[0].parameters["required"] == ["query"] and list(llm.tools[0].parameters["properties"]) == ["query"]
    assert llm.results == ["Error: query is required"]


def test_filter_settings_are_checked(kb):
    dsl = template("rag_chat", kb.id)
    params = dsl["nodes"]["search"]["params"]
    params["filters"] = [{"field": "price", "op": ">", "value": True}, {"field": "brand", "op": "=="}]
    with pytest.raises(ValueError) as exc:
        validate_dsl(dsl)
    assert "filters.0: true/false only works with == and !=" in str(exc.value)
    assert "filters.1: enter a value" in str(exc.value)
    params["filters"] = [{"field": "brand", "op": "in", "value": "{ghost@brands}"}]
    with pytest.raises(ValueError, match="refers to a node that does not exist"):
        validate_dsl(dsl)


# --- Run log (debug runs) -----------------------------------------------------------


def test_debug_run_logs_inputs_and_full_outputs(products, use_llm):
    from workflows.engine import run_draft

    use_llm(ScriptedLLM(reply=lambda s, m: "x" * 400))
    dsl = template("rag_chat", products.id)
    dsl["nodes"]["answer"]["params"]["system_prompt"] = "Use:\n{search@formalized_content}"
    dsl["nodes"]["search"]["next"] = "check"
    dsl["nodes"]["check"] = {
        "type": "switch",
        "params": {"cases": [{"conditions": [{"var": "{search@count}", "op": ">", "value": 0}], "next": "answer"}]},
    }
    events = list(run_draft(dsl, "smartphone", user_id="t"))
    finished = {e["node"]: e for e in events if e["event"] == "node_finished"}

    assert finished["begin"]["inputs"]["sys"]["query"] == "smartphone" and finished["begin"]["next"] == "search"
    search_log = finished["search"]
    assert search_log["inputs"] == {
        "mode": "search",
        "query": "smartphone",
        "kb_ids": [products.id],
        "filters": [],
        "top_k": 6,
    }
    assert len(search_log["outputs"]["chunks"]) == 4  # the full list
    condition = finished["check"]["inputs"]["cases"][0]["conditions"][0]
    assert condition == {"var": "{search@count}", "value": 4, "op": ">", "compared_to": 0, "result": True}
    assert finished["check"]["next"] == "answer"
    answer = finished["answer"]["inputs"]
    assert answer["system_prompt"].startswith("Use:\n[1] ") and answer["model"] == {"model": "scripted"}
    assert answer["messages"] == [{"role": "user", "content": "smartphone"}]
    assert len(finished["answer"]["outputs"]["content"]) == 400  # not shortened

    # Saved workflows run without the log: no inputs, shortened outputs.
    events, _ = run(make_workflow(dsl), "smartphone")
    finished = {e["node"]: e for e in events if e["event"] == "node_finished"}
    assert "inputs" not in finished["answer"] and finished["search"]["outputs"]["chunks"] == "[4 items]"


def test_failed_node_is_logged(products, use_llm):
    from workflows.api import guarded
    from workflows.engine import run_draft

    class Down(ScriptedLLM):
        def stream(self, system, messages):
            raise RuntimeError("the model is down")

    use_llm(Down())
    events = list(guarded(run_draft(template("rag_chat", products.id), "phone"), "test"))
    failed = [e for e in events if e["event"] == "node_failed"]
    assert len(failed) == 1 and failed[0]["node"] == "answer"
    assert failed[0]["error"] == "RuntimeError: the model is down"
    assert "[1] " in failed[0]["inputs"]["system_prompt"]  # what it was about to send
    assert events[-1] == {"event": "error", "message": "RuntimeError: the model is down"}
