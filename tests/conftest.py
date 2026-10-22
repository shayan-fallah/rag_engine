import pytest

from knowledge import search_indexes
from knowledge.models import KnowledgeBase
from tests.fakes import EmbeddingCounter, ScriptedLLM


@pytest.fixture(autouse=True)
def no_api_key(settings):
    settings.API_KEY = ""


@pytest.fixture(autouse=True)
def fresh_index_cache():
    search_indexes._backends.clear()
    yield
    search_indexes._backends.clear()


@pytest.fixture
def embeddings(monkeypatch):
    """Fake embedding model. `embeddings.texts` lists everything that was embedded."""
    counter = EmbeddingCounter()
    monkeypatch.setattr("knowledge.ingest.embed_texts", counter.embed_texts)
    monkeypatch.setattr("knowledge.search.embed_text", counter.embed_text)
    return counter


@pytest.fixture
def use_llm(monkeypatch):
    """use_llm(ScriptedLLM(...)) makes every workflow node use that model."""

    def install(llm: ScriptedLLM) -> ScriptedLLM:
        monkeypatch.setattr("workflows.nodes.get_llm", lambda *args, **kwargs: llm)
        return llm

    return install


@pytest.fixture
def kb(db):
    return KnowledgeBase.objects.create(name="products")


PRODUCTS = [
    (
        "p1",
        "Galaxy S24 phone",
        "Samsung smartphone with a 50MP camera and 8GB RAM. Model SM-S921B.",
        {"category": "phones", "price": 799},
    ),
    (
        "p2",
        "Pixel 9 phone",
        "Google smartphone with a great camera and seven years of updates.",
        {"category": "phones", "price": 699},
    ),
    (
        "p3",
        "QuietComfort headphones",
        "Bose wireless noise cancelling headphones with 24 hour battery.",
        {"category": "audio", "price": 349},
    ),
    (
        "p4",
        "Trail running shoes",
        "Lightweight waterproof shoes for running on mountain trails.",
        {"category": "shoes", "price": 129},
    ),
]
