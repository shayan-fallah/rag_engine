"""The studio (visual editor) and the API endpoints it uses."""

import json
from importlib import reload

import pytest
from django.contrib.auth.models import User
from django.test import Client
from django.urls import clear_url_caches

from tests.fakes import ScriptedLLM
from tests.test_api import TEMPLATES, sse_events
from workflows.models import Conversation, Message


@pytest.fixture
def api(transactional_db, embeddings):
    client = Client()

    def call(method, path, body=None):
        kwargs = {"content_type": "application/json", "data": json.dumps(body)} if body is not None else {}
        return getattr(client, method)(f"/api{path}", **kwargs)

    return call


def rag_chat(kb_id: int) -> dict:
    return json.loads((TEMPLATES / "rag_chat.json").read_text().replace('"kb_ids": [1]', f'"kb_ids": [{kb_id}]'))


@pytest.fixture
def with_api_key(settings):
    """API_KEY set: the API needs the key or a staff session."""
    import config.api
    import config.urls

    settings.API_KEY = "secret"
    reload(config.api)
    reload(config.urls)
    clear_url_caches()
    yield
    settings.API_KEY = ""
    reload(config.api)
    reload(config.urls)
    clear_url_caches()


# --- Endpoints for the editor ------------------------------------------------------------


def test_node_types_describe_every_setting(api):
    data = api("get", "/node-types").json()
    types = {t["type"]: t for t in data["types"]}
    assert set(types) == {"begin", "retrieval", "llm", "agent", "categorize", "switch", "message"}
    retrieval = types["retrieval"]["schema"]["properties"]
    assert retrieval["mode"]["labels"]["metadata"] and retrieval["sort_order"]["hide_when"] == {"sort_by": "empty"}
    assert retrieval["sort_by"]["widget"] == "field" and "metadata_filter" not in retrieval
    categorize = types["categorize"]["schema"]["properties"]
    assert "{categories}" in categorize["system_prompt"]["default"]  # the internal prompt is a setting
    assert categorize["system_prompt"]["group"] == "advanced"
    tool = types["agent"]["schema"]["$defs"]["SearchTool"]["properties"]
    assert {"query_description", "empty_response", "chunk_template"} <= set(tool)
    assert types["retrieval"]["schema"]["properties"]["kb_ids"]["widget"] == "kbs"
    assert types["retrieval"]["outputs"]["count"]
    assert types["switch"]["router"] and not types["llm"]["router"]
    assert "query" in data["sys_vars"] and data["model_defaults"]["openai_model"]


def test_templates(api):
    templates = api("get", "/workflow-templates").json()
    assert {t["key"] for t in templates} == {"rag_chat", "product_assistant", "product_agent"}
    assert all(t["name"] and t["dsl"]["nodes"] for t in templates)


def test_validate_names_the_nodes(api):
    kb = api("post", "/kbs", {"name": "shop"}).json()
    body = rag_chat(kb["id"])
    assert api("post", "/workflows/validate", {"dsl": body["dsl"]}).json() == {"valid": True, "errors": []}

    body["dsl"]["nodes"]["search"]["params"]["kb_ids"] = [kb["id"] + 1000]
    body["dsl"]["nodes"]["reply"]["params"]["text"] = "{ghost@content}"
    result = api("post", "/workflows/validate", {"dsl": body["dsl"]}).json()
    assert not result["valid"]
    assert [e["node"] for e in result["errors"]] == ["reply"]  # kbs are checked once the rest is valid

    r = api("post", "/workflows", body)
    assert r.status_code == 400 and r.json()["errors"][0]["node"] == "reply"


def test_test_run_streams_without_saving(api, use_llm):
    llm = use_llm(ScriptedLLM(reply=lambda s, m: "Draft answer [1]"))
    kb = api("post", "/kbs", {"name": "shop"}).json()
    api("post", f"/kbs/{kb['id']}/documents", {"documents": [{"external_id": "a", "content": "Blue mugs"}]})
    body = {
        "dsl": rag_chat(kb["id"])["dsl"],
        "query": "mugs?",
        "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}],
        "inputs": {},
    }
    events = sse_events(api("post", "/workflows/test-run", body))
    assert [e["node"] for e in events if e["event"] == "node_started"] == ["begin", "search", "answer", "reply"]
    assert events[-1]["event"] == "done" and events[-1]["answer"] == "Draft answer [1]"
    assert events[-1]["references"][0]["cited"]
    assert llm.calls[0]["messages"][:2] == body["history"]
    assert not Conversation.objects.exists() and not Message.objects.exists()

    body["dsl"]["nodes"]["begin"]["next"] = "nowhere"
    r = api("post", "/workflows/test-run", body)
    assert r.status_code == 400 and r.json()["errors"][0]["node"] == "begin"


def test_save_refuses_to_overwrite_newer_changes(api):
    dsl = {"nodes": {"begin": {"type": "begin", "next": "m"}, "m": {"type": "message", "params": {"text": "v1"}}}}
    wf = api("post", "/workflows", {"name": "w", "dsl": dsl}).json()
    loaded = api("get", f"/workflows/{wf['id']}").json()

    dsl["nodes"]["m"]["params"]["text"] = "v2"
    r = api("put", f"/workflows/{wf['id']}", {"name": "w", "dsl": dsl, "base_updated_at": loaded["updated_at"]})
    assert r.status_code == 200  # nobody saved in between (the JSON time has milliseconds only)

    r = api("put", f"/workflows/{wf['id']}", {"name": "w", "dsl": dsl, "base_updated_at": loaded["updated_at"]})
    assert r.status_code == 409  # based on an old version
    r = api("put", f"/workflows/{wf['id']}", {"name": "w", "dsl": dsl})
    assert r.status_code == 200  # without base_updated_at: always saved


def test_layout_round_trips(api):
    dsl = {
        "nodes": {"begin": {"type": "begin", "next": "m"}, "m": {"type": "message", "params": {"text": "hi"}}},
        "layout": {"begin": {"x": 0, "y": 0}, "m": {"x": 300, "y": -20}},
    }
    wf = api("post", "/workflows", {"name": "w", "dsl": dsl}).json()
    assert api("get", f"/workflows/{wf['id']}").json()["dsl"]["layout"] == dsl["layout"]


# --- The page and logging in ----------------------------------------------------------------


def test_studio_page_needs_a_superuser(transactional_db):
    client = Client()
    r = client.get("/studio/")
    assert r.status_code == 302 and r["Location"].startswith("/admin/login/")
    assert client.get("/").status_code == 302  # / goes to the studio

    User.objects.create_user("bob", password="pw", is_staff=True)
    client.login(username="bob", password="pw")
    assert client.get("/studio/").status_code == 302  # staff, but not a superuser

    User.objects.create_superuser("ann", password="pw")
    client.login(username="ann", password="pw")
    r = client.get("/studio/")
    assert r.status_code == 200
    page = r.content.decode()
    assert 'id="studio-config"' in page and "studio/js/main.js" in page
    assert '"apiRoot": "/api/"' in page and "csrftoken" in r.cookies


def test_superuser_session_uses_the_api(transactional_db, with_api_key):
    User.objects.create_superuser("ann", password="pw")
    User.objects.create_user("bob", password="pw", is_staff=True)
    client = Client(enforce_csrf_checks=True)
    body = {"content_type": "application/json", "data": json.dumps({"name": "shop"})}

    assert client.get("/api/kbs").status_code == 401
    assert client.post("/api/kbs", **body).status_code == 401  # no key, no session: 401, not a CSRF error

    client.login(username="bob", password="pw")
    assert client.get("/api/kbs").status_code == 401  # staff accounts don't get the API

    client.login(username="ann", password="pw")
    assert client.get("/api/kbs").status_code == 200
    assert client.post("/api/kbs", **body).status_code == 403  # writes need the CSRF token
    client.get("/studio/")  # sets the CSRF cookie, like opening the page
    token = client.cookies["csrftoken"].value
    assert client.post("/api/kbs", **body, HTTP_X_CSRFTOKEN=token).status_code == 201

    apps = Client()  # apps keep using the key
    assert apps.get("/api/kbs", HTTP_AUTHORIZATION="Bearer secret").status_code == 200


def test_open_api_only_takes_json_bodies(api):
    client = Client()
    body = json.dumps({"name": "shop"})
    # What another website can send from your browser without a CORS check:
    assert client.post("/api/kbs", data=body, content_type="text/plain").status_code == 415
    assert client.post("/api/kbs", data=body, content_type="application/json").status_code == 201


def test_update_edge_cases(api):
    dsl = {"nodes": {"begin": {"type": "begin", "next": "m"}, "m": {"type": "message", "params": {"text": "hi"}}}}
    wf = api("post", "/workflows", {"name": "w", "dsl": dsl}).json()
    r = api("put", f"/workflows/{wf['id']}", {"name": "w", "dsl": dsl, "base_updated_at": "2020-01-01T00:00:00"})
    assert r.status_code == 409  # a time without a timezone is read as UTC, not a crash
    assert api("put", f"/workflows/{wf['id']}", {"name": "x" * 201, "dsl": dsl}).status_code == 422


# --- Metadata search ---------------------------------------------------------------------


def test_search_api_by_metadata_and_field_list(api):
    kb = api("post", "/kbs", {"name": "shop"}).json()
    docs = [
        {
            "external_id": "a",
            "content": "Blue mug",
            "metadata": {"color": "blue", "price": 12.5, "added": "2025-02-01"},
        },
        {
            "external_id": "b",
            "content": "Red mug",
            "metadata": {"color": "red", "price": "9.00", "added": "2025-03-01", "tags": ["sale"]},
        },
    ]
    api("post", f"/kbs/{kb['id']}/documents", {"documents": docs})

    def found(body):
        return [x["external_id"] for x in api("post", "/search", {"kb_ids": [kb["id"]], **body}).json()["results"]]

    assert found({"sort_by": "added"}) == ["b", "a"]  # no query: by metadata alone
    assert found({"sort_by": "price", "sort_order": "asc"}) == ["b", "a"]  # "9.00" is a number
    assert found({"query": "mug", "filters": [{"field": "price", "op": ">", "value": 10}]}) == ["a"]
    assert found({"filters": [{"field": "tags", "op": "contains", "value": "SALE"}]}) == ["b"]
    r = api("post", "/search", {"kb_ids": [kb["id"]], "filters": [{"field": "price", "op": ">", "value": True}]})
    assert r.status_code == 400 and r.json()["detail"] == "filter 1: true/false only works with == and !="

    assert api("get", f"/kbs/{kb['id']}/metadata-fields").json() == [
        {"name": "added", "type": "date", "count": 2, "examples": ["2025-03-01", "2025-02-01"]},
        {"name": "color", "type": "text", "count": 2, "examples": ["red", "blue"]},
        {"name": "price", "type": "number", "count": 2, "examples": ["9.00", 12.5]},
        {"name": "tags", "type": "list", "count": 1, "examples": []},
    ]
    assert api("get", "/kbs/9999/metadata-fields").status_code == 404


def test_test_run_sends_the_run_log(api, use_llm):
    use_llm(ScriptedLLM(reply=lambda s, m: "Answer"))
    kb = api("post", "/kbs", {"name": "shop"}).json()
    api("post", f"/kbs/{kb['id']}/documents", {"documents": [{"external_id": "a", "content": "Blue mugs"}]})
    events = sse_events(api("post", "/workflows/test-run", {"dsl": rag_chat(kb["id"])["dsl"], "query": "mugs?"}))
    answer = next(e for e in events if e["event"] == "node_finished" and e["node"] == "answer")
    assert "Blue mugs" in answer["inputs"]["system_prompt"] and answer["outputs"] == {"content": "Answer"}
