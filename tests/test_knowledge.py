import pytest
from django.db import connection

from knowledge.ingest import DocInput, ingest_documents
from knowledge.jobs import claim_job, enqueue_sync, work
from knowledge.models import Chunk, DataSource, Document, KnowledgeBase, SyncJob
from knowledge.search import as_reference, check_filters, format_chunks, metadata_search, search
from knowledge.search_indexes import backends
from knowledge.sources import SyncError, page_query, row_to_document, sync_source
from tests.conftest import PRODUCTS


def add_products(kb):
    return ingest_documents(kb, [DocInput(pid, desc, title, meta) for pid, title, desc, meta in PRODUCTS])


# --- Ingest -----------------------------------------------------------------------


def test_ingest_creates_documents_and_chunks(kb, embeddings):
    stats = add_products(kb)
    assert stats["created"] == 4 and stats["chunks"] == 4
    chunk = Chunk.objects.get(document__external_id="p1")
    assert chunk.content.startswith("Galaxy S24 phone\nSamsung")  # title + text
    assert chunk.labels == [kb.id] and chunk.metadata == {"category": "phones", "price": 799}


def test_reingest_only_embeds_changes(kb, embeddings):
    add_products(kb)
    embeddings.texts.clear()

    docs = [DocInput(pid, desc, title, meta) for pid, title, desc, meta in PRODUCTS]
    docs[0].content += " Now with a 256GB option."  # text changed -> re-embed
    docs[1].metadata = {**docs[1].metadata, "price": 599}  # price changed -> no embedding
    stats = ingest_documents(kb, docs)

    assert stats == {"updated": 1, "metadata_updated": 1, "unchanged": 2, "chunks": 1}
    assert len(embeddings.texts) == 1 and "256GB" in embeddings.texts[0]
    assert Chunk.objects.get(document__external_id="p2").metadata["price"] == 599
    assert Chunk.objects.filter(document__external_id="p1").count() == 1


def test_long_document_gets_several_chunks(embeddings, db):
    kb = KnowledgeBase.objects.create(name="small chunks", chunk_size=300, chunk_overlap=50)
    text = "\n".join(f"Fact {i}: the warranty covers part number {i} for two years." for i in range(30))
    stats = ingest_documents(kb, [DocInput("manual", text, "Warranty")])
    assert stats["chunks"] > 3
    assert all(c.content.startswith("Warranty\n") for c in Chunk.objects.filter(kb=kb))


# --- Search -----------------------------------------------------------------------


def test_hybrid_search_finds_exact_terms_and_meaning(kb, embeddings):
    add_products(kb)
    results = search("SM-S921B", [kb.id], top_k=2)
    assert results[0]["external_id"] == "p1"
    assert results[0]["keyword_rank"] == 1  # an exact model number is found by keyword search

    results = search("wireless headphones noise cancelling", [kb.id], top_k=2)
    assert results[0]["external_id"] == "p3"
    assert results[0]["vector_rank"] == 1 and results[0]["keyword_rank"] == 1


def test_search_filters(kb, embeddings):
    add_products(kb)
    other = KnowledgeBase.objects.create(name="other")
    ingest_documents(other, [DocInput("x", "Smartphone camera guide", "Guide")])

    ids = {r["external_id"] for r in search("smartphone camera", [kb.id], top_k=10)}
    assert "x" not in ids and {"p1", "p2"} <= ids

    results = search("smartphone camera", [kb.id], top_k=10, metadata_filter={"category": "audio"})
    assert {r["external_id"] for r in results} == {"p3"}
    assert results[0]["metadata"] == {"category": "audio", "price": 349}

    both = {r["external_id"] for r in search("smartphone camera", [kb.id, other.id], top_k=10)}
    assert "x" in both


def test_max_distance_drops_unrelated_vector_matches(kb, embeddings):
    add_products(kb)
    assert search("zebra xylophone", [kb.id], top_k=5, max_distance=0.5) == []


CATALOG = [
    # id, title, text, metadata (dates as a SQL source stores them: ISO text)
    (
        "a",
        "Galaxy S24 phone",
        "Samsung smartphone with a great camera.",
        {"category": "Phones", "price": 799.0, "date_added": "2025-01-10T09:00:00", "tags": ["new"], "active": 1},
    ),
    (
        "b",
        "Pixel 9 phone",
        "Google smartphone with a great camera.",
        {"category": "Phones", "price": "699.00", "date_added": "2025-03-05T12:30:00", "active": 1},
    ),
    (
        "c",
        "Redmi Note phone",
        "Budget smartphone with a big battery.",
        {"category": "Phones", "price": 229, "date_added": "2024-11-20 08:00:00", "active": 0},
    ),
    (
        "d",
        "QuietComfort headphones",
        "Bose noise cancelling headphones.",
        {"category": "Audio", "price": 429, "date_added": "2025-03-20T18:00:00", "tags": ["sale", "New"]},
    ),
    ("e", "Trail shoes", "Running shoes for mountain trails.", {"category": "Shoes"}),
]


def add_catalog(kb):
    return ingest_documents(kb, [DocInput(pid, text, title, meta) for pid, title, text, meta in CATALOG])


def ids(results):
    return [r["external_id"] for r in results]


def test_metadata_search_sorts_and_filters_without_a_query(kb, embeddings):
    add_catalog(kb)
    embeddings.texts.clear()
    newest = metadata_search([kb.id], top_k=2, sort_by="date_added")
    assert ids(newest) == ["d", "b"] and not embeddings.texts  # no embedding call
    assert newest[0]["score"] is None and newest[0]["content"].startswith("QuietComfort headphones")
    assert as_reference(newest[0], 1)["score"] is None
    assert ids(metadata_search([kb.id], top_k=10, sort_by="date_added", sort_order="asc")) == ["c", "a", "b", "d", "e"]

    def find(*filters, **kwargs):
        return ids(metadata_search([kb.id], top_k=10, filters=list(filters), **kwargs))

    phones = {"field": "category", "op": "==", "value": "phones"}  # text ignores case
    assert find(phones, sort_by="date_added") == ["b", "a", "c"]
    assert find(phones, {"field": "price", "op": "<", "value": 700}, sort_by="price") == [
        "b",
        "c",
    ]  # "699.00" is a number
    assert find({"field": "price", "op": ">=", "value": "699"}, sort_by="price", sort_order="asc") == ["b", "a"]
    assert find({"field": "date_added", "op": ">=", "value": "2025-03"}, sort_by="date_added") == [
        "d",
        "b",
    ]  # the month
    assert find({"field": "date_added", "op": "<=", "value": "2025-01-10"}, sort_by="date_added") == [
        "a",
        "c",
    ]  # the day
    assert find({"field": "date_added", "op": "==", "value": "2024-11-20T08:00"}) == ["c"]  # " " or "T"
    assert find({"field": "category", "op": "in", "value": "Audio, shoes"}) == ["d", "e"]
    assert find({"field": "price", "op": "in", "value": [229, 429]}) == ["c", "d"]
    assert find({"field": "tags", "op": "contains", "value": "new"}) == ["a", "d"]  # a list item, any case
    assert find({"field": "title", "op": "contains", "value": "x"}) == []  # title isn't metadata
    assert find({"field": "category", "op": "contains", "value": "hon"}) == ["a", "b", "c"]
    assert find({"field": "active", "op": "==", "value": True}) == ["a", "b"]  # 1 is true
    assert find({"field": "active", "op": "!=", "value": True}) == ["c", "d", "e"]  # missing counts as "not equal"
    assert find(phones, metadata_filter={"active": 0}) == ["c"]


def test_search_with_filters_and_sort(kb, embeddings):
    add_catalog(kb)
    results = search("smartphone camera", [kb.id], top_k=5)
    assert ids(results)[:2] in (["a", "b"], ["b", "a"])

    # The newest of the best matches: the matches are sorted by the field.
    newest = search("smartphone", [kb.id], top_k=2, sort_by="date_added", sort_pool=3)
    assert ids(newest) == ["b", "a"] and newest[0]["score"] > 0

    cheap = search("smartphone", [kb.id], top_k=5, filters=[{"field": "price", "op": "<", "value": 500}])
    assert ids(cheap)[0] == "c" and set(ids(cheap)) == {"c", "d"}  # d (429) is no phone, but cheap


def test_check_filters_explains_mistakes():
    assert check_filters([{"field": " price ", "op": "in", "value": "1, 2"}]) == [
        {"field": "price", "op": "in", "value": ["1", "2"]}
    ]
    for bad, message in [
        ([{"op": "=="}], "field"),
        ([{"field": "x", "op": "~"}], "op must be one of"),
        ([{"field": "x", "op": "in", "value": []}], "list of values"),
        ([{"field": "x", "op": ">", "value": True}], "true/false only"),
        ([{"field": "x", "op": "==", "value": None}], "value must be"),
        (["price < 3"], "filter 1: a filter must be an object"),
    ]:
        with pytest.raises(ValueError, match=message):
            check_filters(bad)
    with pytest.raises(ValueError, match="order must be"):
        metadata_search([1], sort_by="x", sort_order="newest")


def test_format_chunks_numbers_and_metadata():
    text = format_chunks([{"content": "Pixel 9\nA phone", "metadata": {"price": 699}}], start=3)
    assert text == "[3] Pixel 9\nA phone\n(price: 699)"


def test_fulltext_fallback_when_pg_textsearch_is_missing(kb, embeddings):
    add_products(kb)
    with connection.cursor() as cur:
        cur.execute("SET CONSTRAINTS ALL IMMEDIATE")  # DDL can't run with deferred FK checks pending
        cur.execute("DROP INDEX knowledge_chunk_keyword_idx")
        cur.execute(
            "CREATE INDEX knowledge_chunk_keyword_idx ON knowledge_chunk USING gin (to_tsvector('english'::regconfig, content))"
        )
    assert backends()["keyword"] == "fts"
    results = search("waterproof running shoes", [kb.id], top_k=2)
    assert results[0]["external_id"] == "p4" and results[0]["keyword_rank"] == 1
    results = search("camera", [kb.id], top_k=5, metadata_filter={"category": "phones"})
    assert {r["external_id"] for r in results} == {"p1", "p2"}
    results = search("camera", [kb.id], top_k=5, filters=[{"field": "price", "op": ">", "value": 700}], sort_by="price")
    assert [r["external_id"] for r in results] == ["p1"]


# --- Data sources -----------------------------------------------------------------


def test_page_query_quotes_and_paginates():
    sql = page_query("SELECT * FROM products;", "id", "`", after=True)
    assert sql == "SELECT * FROM (SELECT * FROM products) AS src WHERE src.`id` > %s ORDER BY src.`id` LIMIT 500"
    with pytest.raises(SyncError):
        page_query("SELECT 1", "id; DROP TABLE x", "`", after=False)


def test_row_to_document_template_html_and_metadata(kb):
    source = DataSource(
        kb=kb, id_column="id", title_column="name", content_template="{name}\n{description}", metadata_columns=["price"]
    )
    from decimal import Decimal

    doc = row_to_document(
        source, {"id": 7, "name": "Mug", "description": "<p>Big &amp; <b>blue</b></p>", "price": Decimal("9.50")}
    )
    assert (doc.external_id, doc.title, doc.metadata) == ("7", "Mug", {"price": 9.5})
    assert doc.content == "Mug\nBig &  blue"

    source.content_template = ""  # default: every other column as "column: value"
    assert (
        row_to_document(source, {"id": 7, "name": "Mug", "description": "Blue", "stock": None}).content
        == "description: Blue"
    )


@pytest.fixture
def products_table(transactional_db):
    """A products table in the test database, read through a postgresql:// source URL."""
    with connection.cursor() as cur:
        cur.execute(
            "CREATE TABLE demo_products (id int PRIMARY KEY, name text, description text, category text, price numeric)"
        )
        cur.executemany(
            "INSERT INTO demo_products VALUES (%s, %s, %s, %s, %s)",
            [(i, title, desc, meta["category"], meta["price"]) for i, (_, title, desc, meta) in enumerate(PRODUCTS, 1)],
        )
    s = connection.settings_dict
    yield f"postgresql://{s['USER']}:{s['PASSWORD']}@{s['HOST']}:{s['PORT']}/{s['NAME']}"
    with connection.cursor() as cur:
        cur.execute("DROP TABLE demo_products")


def test_sync_source(products_table, embeddings, monkeypatch):
    monkeypatch.setattr("knowledge.sources.PAGE_SIZE", 3)  # force two pages
    kb = KnowledgeBase.objects.create(name="shop")
    source = DataSource.objects.create(
        kb=kb,
        name="products",
        url=products_table,
        query="SELECT id, name, description, category, price FROM demo_products WHERE price > 0",
        id_column="id",
        title_column="name",
        content_template="{description}",
        metadata_columns=["category", "price"],
    )
    stats = sync_source(source)
    assert stats["rows"] == 4 and stats["created"] == 4 and stats["deleted"] == 0
    assert Document.objects.get(source=source, external_id="3").metadata == {"category": "audio", "price": 349.0}

    with connection.cursor() as cur:
        cur.execute("UPDATE demo_products SET description = 'Brand new description' WHERE id = 1")
        cur.execute("UPDATE demo_products SET price = 100 WHERE id = 2")
        cur.execute("DELETE FROM demo_products WHERE id = 4")
    embeddings.texts.clear()
    stats = sync_source(source)
    assert (stats["updated"], stats["metadata_updated"], stats["unchanged"], stats["deleted"]) == (1, 1, 1, 1)
    assert len(embeddings.texts) == 1
    assert not Document.objects.filter(external_id="4").exists()

    # A query that suddenly returns nothing must not wipe the knowledge base.
    source.query = "SELECT * FROM demo_products WHERE price < 0"
    with pytest.raises(SyncError, match="refusing to delete"):
        sync_source(source)
    assert Document.objects.filter(source=source).count() == 3


def test_jobs_queue(products_table, embeddings):
    kb = KnowledgeBase.objects.create(name="shop")
    source = DataSource.objects.create(
        kb=kb,
        name="products",
        url=products_table,
        query="SELECT * FROM demo_products",
        title_column="name",
        content_template="{description}",
    )
    job = enqueue_sync(source)
    assert enqueue_sync(source).id == job.id  # no duplicate while queued

    work(once=True)
    job.refresh_from_db()
    assert job.status == SyncJob.DONE and job.stats["created"] == 4
    assert claim_job() is None

    source.url = "env:MISSING_URL_VAR"
    source.save()
    job = enqueue_sync(source)
    work(once=True)
    job.refresh_from_db()
    assert job.status == SyncJob.FAILED and "MISSING_URL_VAR" in job.error


def test_updated_document_gets_new_updated_at(kb, embeddings):
    add_products(kb)
    before = Document.objects.get(external_id="p1").updated_at
    ingest_documents(kb, [DocInput("p1", "Completely new text", "Galaxy S24 phone")])
    assert Document.objects.get(external_id="p1").updated_at > before


def test_sync_with_percent_in_query_over_several_pages(products_table, embeddings, monkeypatch):
    monkeypatch.setattr("knowledge.sources.PAGE_SIZE", 1)
    kb = KnowledgeBase.objects.create(name="shop")
    source = DataSource.objects.create(
        kb=kb,
        name="p",
        url=products_table,
        title_column="name",
        content_template="{description}",
        query="SELECT * FROM demo_products WHERE description LIKE '%e%'",
    )
    assert sync_source(source)["created"] == 4


def test_sync_refuses_to_empty_the_kb_on_a_bad_template(products_table, embeddings):
    kb = KnowledgeBase.objects.create(name="shop")
    source = DataSource.objects.create(
        kb=kb,
        name="p",
        url=products_table,
        query="SELECT * FROM demo_products",
        title_column="name",
        content_template="{description}",
    )
    sync_source(source)
    source.content_template = "{descripton}"  # typo
    with pytest.raises(SyncError, match="no column.*descripton"):
        sync_source(source)
    source.content_template = "{category}"
    with connection.cursor() as cur:
        cur.execute("UPDATE demo_products SET category = ''")
    with pytest.raises(SyncError, match="refusing to delete all 4"):
        sync_source(source)
    assert Document.objects.filter(source=source).count() == 4


def test_failing_scheduled_source_is_not_retried_in_a_loop(products_table, embeddings):
    kb = KnowledgeBase.objects.create(name="shop")
    DataSource.objects.create(kb=kb, name="p", url="env:MISSING_URL_VAR", query="SELECT 1", sync_interval_minutes=60)
    work(once=True)  # returns: the failed source isn't due again for 60 minutes
    assert list(SyncJob.objects.values_list("status", flat=True)) == [SyncJob.FAILED]


def test_job_deleted_during_sync_does_not_crash_the_worker(products_table, embeddings, monkeypatch):
    kb = KnowledgeBase.objects.create(name="shop")
    source = DataSource.objects.create(kb=kb, name="p", url=products_table, query="SELECT * FROM demo_products")
    enqueue_sync(source)

    def sync_then_delete(src, job=None):
        src.delete()  # e.g. someone deletes the source while it syncs
        return {}

    monkeypatch.setattr("knowledge.jobs.sync_source", sync_then_delete)
    work(once=True)
    assert not SyncJob.objects.exists()


def test_format_chunks_with_a_template():
    chunks = [
        {"content": "Pixel 9\nA phone", "title": "Pixel 9", "external_id": "p2", "metadata": {"price": 699}},
        {"content": "FAQ text", "title": "FAQ", "external_id": "f1", "metadata": {}},
    ]
    template = "<{ref}> {title} ({external_id})\nPrice: {metadata.price}\n{content}"
    assert format_chunks(chunks, start=1, template=template) == (
        "<1> Pixel 9 (p2)\nPrice: 699\nPixel 9\nA phone\n\n<2> FAQ (f1)\nFAQ text"  # no price: line left out
    )
    # The default drops the "(metadata)" line when there is none.
    assert format_chunks(chunks[1:], start=5) == "[5] FAQ text"
    # A line with other placeholders is kept, even when its metadata is empty.
    assert format_chunks(chunks[1:], template="[{ref}] {content} ({metadata.price})") == "[1] FAQ text ()"
