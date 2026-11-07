"""LLM profiles: checking workflows that use them, the API, and the admin (keys stay hidden).

Calls to the profiles' servers are tested in test_llm_providers.py.
"""

import pytest
from django.db.models import ProtectedError

from core.models import LLMProfile
from workflows.dsl import DSLError, validate_dsl
from workflows.models import Workflow

SECRET = "sk-profile-secret-0123456789"


def workflow_dsl(**params) -> dict:
    return {
        "nodes": {
            "begin": {"type": "begin", "next": "answer"},
            "answer": {"type": "llm", "params": params, "next": "reply"},
            "reply": {"type": "message", "params": {"text": "{answer@content}"}},
        }
    }


@pytest.fixture
def profile(db):
    return LLMProfile.objects.create(
        name="liara", base_url="https://ai.example.com/v1", api_key=SECRET, default_model="gpt-4.1"
    )


def errors_of(dsl) -> list[str]:
    with pytest.raises(DSLError) as exc:
        validate_dsl(dsl)
    return [message for _, message in exc.value.errors]


def test_workflows_are_checked_against_the_profiles(profile):
    validate_dsl(workflow_dsl(profile="liara"))
    validate_dsl(workflow_dsl(profile=""))  # empty: the .env settings
    assert errors_of(workflow_dsl(profile="nope")) == [
        "profile: LLM profile 'nope' does not exist; add it in the admin"
    ]
    assert errors_of(workflow_dsl(profile="liara", provider="openai")) == [
        "set a profile or a provider, not both (the profile's kind is its provider)"
    ]
    profile.default_model = ""
    profile.save()
    assert "has no default model" in errors_of(workflow_dsl(profile="liara"))[0]
    validate_dsl(workflow_dsl(profile="liara", model="openai/gpt-4.1"))
    validate_dsl(workflow_dsl(profile="nope"), check_db=False)  # running a saved workflow skips the database


def test_the_api_never_returns_a_key(profile, client):
    LLMProfile.objects.create(name="claude", kind="anthropic")
    response = client.get("/api/llm-profiles")
    assert response.json() == [
        {"name": "claude", "kind": "anthropic", "default_model": ""},
        {"name": "liara", "kind": "openai", "default_model": "gpt-4.1"},
    ]

    created = client.post(
        "/api/workflows", {"name": "wf", "dsl": workflow_dsl(profile="liara")}, content_type="application/json"
    )
    assert created.status_code == 201
    texts = [
        response.content,
        created.content,
        client.get(f"/api/workflows/{created.json()['id']}").content,
        client.get("/api/node-types").content,
    ]
    for text in texts:
        assert SECRET.encode() not in text and b"ai.example.com" not in text

    # The studio's forms: a profile dropdown in the Model section, which replaces the provider.
    types = {t["type"]: t["schema"]["properties"] for t in client.get("/api/node-types").json()["types"]}
    for node_type in ("llm", "agent", "categorize"):
        assert types[node_type]["profile"]["widget"] == "profile" and types[node_type]["profile"]["clears"] == [
            "provider"
        ]
        assert types[node_type]["provider"]["hide_when"] == {"profile": "not_empty"}

    missing = client.post(
        "/api/workflows/validate", {"dsl": workflow_dsl(profile="nope")}, content_type="application/json"
    ).json()
    assert missing["errors"] == [
        {"node": "answer", "message": "profile: LLM profile 'nope' does not exist; add it in the admin"}
    ]


def test_admin_hides_the_key_and_keeps_it_when_left_empty(profile, admin_client):
    url = f"/admin/core/llmprofile/{profile.id}/change/"
    page = admin_client.get(url).content.decode()
    assert SECRET not in page and "Saved (…6789). Leave empty to keep it." in page
    assert admin_client.get("/admin/core/llmprofile/").content.decode().count("…6789") == 1

    form = {"name": "liara", "kind": "openai", "base_url": "https://ai.example.com/v1", "default_model": "gpt-4.1"}
    assert admin_client.post(url, {**form, "api_key": ""}).status_code == 302
    profile.refresh_from_db()
    assert profile.api_key == SECRET

    admin_client.post(url, {**form, "api_key": "new-key-0123456789"})
    profile.refresh_from_db()
    assert profile.api_key == "new-key-0123456789"

    admin_client.post(url, {**form, "api_key": "", "remove_api_key": "on"})
    profile.refresh_from_db()
    assert profile.api_key == ""

    bad = admin_client.post(url, {**form, "base_url": "ai.example.com"})
    assert bad.status_code == 200 and "starting with http:// or https://" in bad.content.decode()


def test_a_profile_in_use_cannot_be_deleted(profile, admin_client):
    workflow = Workflow.objects.create(name="Shop", dsl=workflow_dsl(profile="liara"))
    url = f"/admin/core/llmprofile/{profile.id}/delete/"
    page = admin_client.get(url).content.decode()
    assert "Workflow “Shop” uses “liara” in nodes answer" in page
    admin_client.post(url, {"post": "yes"})
    assert LLMProfile.objects.filter(name="liara").exists()

    bulk = admin_client.post(
        "/admin/core/llmprofile/", {"action": "delete_selected", "_selected_action": [profile.id], "post": "yes"}
    )
    assert "Workflow “Shop”" in bulk.content.decode() and LLMProfile.objects.filter(name="liara").exists()

    with pytest.raises(ProtectedError, match="used by the workflow"):
        profile.delete()
    with pytest.raises(ProtectedError):
        LLMProfile.objects.filter(name="liara").delete()

    workflow.delete()
    profile.delete()
    assert not LLMProfile.objects.exists()


def test_renaming_a_profile_updates_the_workflows(profile):
    other = Workflow.objects.create(name="Other", dsl=workflow_dsl(model="gpt-test"))
    shop = Workflow.objects.create(name="Shop", dsl=workflow_dsl(profile="liara"))
    profile.name = "liara-gpt"
    profile.save()
    shop.refresh_from_db()
    other.refresh_from_db()
    assert shop.dsl["nodes"]["answer"]["params"]["profile"] == "liara-gpt"
    assert other.dsl == workflow_dsl(model="gpt-test")
    validate_dsl(shop.dsl)
