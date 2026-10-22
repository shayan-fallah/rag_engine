import json
from pathlib import Path

import pytest
from django.test import Client

from tests.conftest import PRODUCTS
from tests.fakes import ScriptedLLM

TEMPLATES = Path(__file__).resolve().parent.parent / "workflows" / "templates"


@pytest.fixture
def api(transactional_db, embeddings):
    client = Client()

    def call(method, path, body=None, **headers):
        kwargs = {"content_type": "application/json", "data": json.dumps(body)} if body is not None else {}
        return getattr(client, method)(f"/api{path}", **kwargs, **headers)

    return call


def sse_events(response) -> list:
    body = b"".join(response.streaming_content).decode()
    lines = [line[len("data: ") :] for line in body.split("\n\n") if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def test_knowledge_api(api):
    r = api("post", "/kbs", {"name": "shop", "chunk_size": 1000})
    assert r.status_code == 201
    kb = r.json()
    assert api("post", "/kbs", {"name": "shop"}).status_code == 409

    docs = [{"external_id": pid, "title": t, "content": c, "metadata": m} for pid, t, c, m in PRODUCTS]
    r = api("post", f"/kbs/{kb['id']}/documents", {"documents": docs})
    assert r.json()["stats"] == {"created": 4, "chunks": 4}
    r = api(
        "post", f"/kbs/{kb['id']}/documents", {"documents": [{"title": "FAQ", "content": "Shipping takes 2 days."}]}
    )
    faq_id = r.json()["external_ids"][0]

    assert api("get", f"/kbs/{kb['id']}").json()["document_count"] == 5
    assert len(api("get", f"/kbs/{kb['id']}/documents?limit=2").json()) == 2
    assert api("get", f"/kbs/{kb['id']}/documents/p3").json()["content"].startswith("Bose")

    r = api("post", "/search", {"query": "noise cancelling", "kb_ids": [kb["id"]], "top_k": 2})
    assert r.json()["results"][0]["external_id"] == "p3"

    status = api("get", "/status").json()
    assert status["indexes"]["vector"] in ("diskann", "hnsw") and status["chunks"] == 5

    assert api("delete", f"/kbs/{kb['id']}/documents/{faq_id}").status_code == 204
    assert api("delete", f"/kbs/{kb['id']}/documents/{faq_id}").status_code == 404


def test_source_api_masks_password(api):
    kb = api("post", "/kbs", {"name": "shop"}).json()
    body = {
        "name": "products",
        "url": "mysql://shop:s3cret@db:3306/shop",
        "query": "SELECT * FROM products",
        "title_column": "name",
        "content_template": "{description}",
        "metadata_columns": ["price"],
    }
    source = api("post", f"/kbs/{kb['id']}/sources", body).json()
    assert source["url"] == "mysql://shop:***@db:3306/shop"

    # Sending the masked URL back keeps the stored password.
    api("put", f"/sources/{source['id']}", {**body, "url": source["url"], "sync_interval_minutes": 60})
    from knowledge.models import DataSource

    assert DataSource.objects.get(pk=source["id"]).url == body["url"]

    job = api("post", f"/sources/{source['id']}/sync").json()
    assert job["status"] == "queued"
    assert api("post", f"/sources/{source['id']}/sync").json()["id"] == job["id"]
    assert api("get", f"/jobs/{job['id']}").json()["status"] == "queued"


def test_workflow_api_and_conversation(api, use_llm):
    llm = use_llm(ScriptedLLM(reply=lambda s, m: f"Answer {len(m)}"))
    kb = api("post", "/kbs", {"name": "shop"}).json()
    docs = [{"external_id": pid, "title": t, "content": c, "metadata": m} for pid, t, c, m in PRODUCTS]
    api("post", f"/kbs/{kb['id']}/documents", {"documents": docs})

    wf_body = json.loads((TEMPLATES / "rag_chat.json").read_text().replace('"kb_ids": [1]', f'"kb_ids": [{kb["id"]}]'))
    r = api("post", "/workflows", wf_body)
    assert r.status_code == 201, r.json()
    wf = r.json()

    bad = {
        **wf_body,
        "name": "bad",
        "dsl": {"nodes": {"begin": {"type": "begin", "next": "x"}, "m": {"type": "message", "params": {"text": "hi"}}}},
    }
    r = api("post", "/workflows", bad)
    assert r.status_code == 400 and "next node 'x' does not exist" in r.json()["detail"]

    # Turn 1, streamed
    r = api("post", f"/workflows/{wf['id']}/run", {"query": "a phone with a good camera?", "user_id": "u1"})
    assert r["Content-Type"] == "text/event-stream"
    events = sse_events(r)
    assert events[0]["event"] == "conversation"
    done = events[-1]
    assert done["event"] == "done" and done["answer"] == "Answer 1" and done["references"]
    conversation_id = done["conversation_id"]

    # Turn 2, not streamed: the model now also sees turn 1
    r = api(
        "post",
        f"/workflows/{wf['id']}/run",
        {"query": "and cheaper?", "user_id": "u1", "conversation_id": conversation_id, "stream": False},
    )
    assert r.json()["answer"] == "Answer 3"
    assert llm.calls[-1]["messages"][0] == {"role": "user", "content": "a phone with a good camera?"}

    # Another user can't continue or read it
    r = api("post", f"/workflows/{wf['id']}/run", {"query": "hi", "user_id": "u2", "conversation_id": conversation_id})
    assert r.status_code == 404
    assert api("get", f"/conversations/{conversation_id}?user_id=u2").status_code == 404

    convo = api("get", f"/conversations/{conversation_id}?user_id=u1").json()
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant", "user", "assistant"]
    assert convo["messages"][1]["references"][0]["ref"] == 1
    assert len(api("get", f"/workflows/{wf['id']}/conversations?user_id=u1").json()) == 1


def test_errors_become_error_events(api, use_llm):
    class Broken(ScriptedLLM):
        def stream(self, system, messages):
            raise RuntimeError("model is down")
            yield

    use_llm(Broken())
    wf = api(
        "post",
        "/workflows",
        {
            "name": "w",
            "dsl": {
                "nodes": {
                    "begin": {"type": "begin", "next": "llm"},
                    "llm": {"type": "llm", "next": "out"},
                    "out": {"type": "message", "params": {"text": "{llm@content}"}},
                }
            },
        },
    ).json()
    events = sse_events(api("post", f"/workflows/{wf['id']}/run", {"query": "q", "user_id": "u"}))
    assert events[-1] == {"event": "error", "message": "RuntimeError: model is down"}
    r = api("post", f"/workflows/{wf['id']}/run", {"query": "q", "user_id": "u", "stream": False})
    assert r.status_code == 500


def test_api_key(api, settings):
    settings.API_KEY = "secret"
    from importlib import reload

    import config.api
    import config.urls

    reload(config.api)
    reload(config.urls)
    from django.urls import clear_url_caches

    clear_url_caches()
    try:
        assert api("get", "/kbs").status_code == 401
        assert api("get", "/kbs", HTTP_AUTHORIZATION="Bearer wrong").status_code == 401
        assert api("get", "/kbs", HTTP_AUTHORIZATION="Bearer secret").status_code == 200
    finally:
        settings.API_KEY = ""
        reload(config.api)
        reload(config.urls)
        clear_url_caches()


def test_document_ids_with_slashes_and_duplicates(api):
    from knowledge.models import DataSource, Document, KnowledgeBase

    kb = api("post", "/kbs", {"name": "shop"}).json()
    api("post", f"/kbs/{kb['id']}/documents", {"documents": [{"external_id": "a/b", "content": "posted"}]})
    assert api("get", f"/kbs/{kb['id']}/documents/a/b").json()["content"] == "posted"

    source = DataSource.objects.create(kb_id=kb["id"], name="s", url="x", query="x")
    Document.objects.create(kb_id=kb["id"], source=source, external_id="a/b", content="synced", content_hash="h")
    assert api("get", f"/kbs/{kb['id']}/documents/a/b").json()["content"] == "posted"
    assert api("get", f"/kbs/{kb['id']}/documents/a/b?source_id={source.id}").json()["content"] == "synced"
    assert api("delete", f"/kbs/{kb['id']}/documents/a/b").status_code == 204
    assert KnowledgeBase.objects.get(pk=kb["id"]).documents.count() == 1


def test_rename_workflow_to_existing_name(api):
    dsl = {"nodes": {"begin": {"type": "begin", "next": "m"}, "m": {"type": "message", "params": {"text": "hi"}}}}
    api("post", "/workflows", {"name": "a", "dsl": dsl})
    b = api("post", "/workflows", {"name": "b", "dsl": dsl}).json()
    assert api("put", f"/workflows/{b['id']}", {"name": "a", "dsl": dsl}).status_code == 409
    assert api("put", f"/workflows/{b['id']}", {"name": "b", "description": "new", "dsl": dsl}).status_code == 200
