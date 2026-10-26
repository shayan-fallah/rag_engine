import re
from collections import Counter
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlparse

from django.db import IntegrityError
from django.db.models import F
from django.shortcuts import get_object_or_404
from ninja import Field, Router, Schema
from ninja.errors import HttpError

from knowledge.ingest import DocInput, default_external_id, ingest_documents
from knowledge.jobs import enqueue_sync
from knowledge.models import DataSource, Document, KnowledgeBase, SyncJob
from knowledge.search import DATE_TEXT, FILTER_OPS, metadata_search, search

router = Router()


# --- Knowledge bases ------------------------------------------------------------


class KBIn(Schema):
    name: str
    description: str = ""
    chunk_size: int | None = Field(None, ge=200, le=20000)
    chunk_overlap: int | None = Field(None, ge=0, le=5000)


class KBOut(Schema):
    id: int
    name: str
    description: str
    chunk_size: int
    chunk_overlap: int
    document_count: int
    created_at: datetime


def kb_out(kb: KnowledgeBase) -> dict:
    return {
        **{f: getattr(kb, f) for f in ("id", "name", "description", "chunk_size", "chunk_overlap", "created_at")},
        "document_count": kb.documents.count(),
    }


@router.post("/kbs", response={201: KBOut})
def create_kb(request, data: KBIn):
    fields = data.dict(exclude_none=True)
    if KnowledgeBase.objects.filter(name=data.name).exists():
        raise HttpError(409, f"A knowledge base named {data.name!r} exists")
    return 201, kb_out(KnowledgeBase.objects.create(**fields))


@router.get("/kbs", response=list[KBOut])
def list_kbs(request):
    return [kb_out(kb) for kb in KnowledgeBase.objects.order_by("id")]


@router.get("/kbs/{kb_id}", response=KBOut)
def get_kb(request, kb_id: int):
    return kb_out(get_object_or_404(KnowledgeBase, pk=kb_id))


@router.delete("/kbs/{kb_id}", response={204: None})
def delete_kb(request, kb_id: int):
    get_object_or_404(KnowledgeBase, pk=kb_id).delete()
    return 204, None


# --- Documents ------------------------------------------------------------------


class DocumentIn(Schema):
    external_id: str | None = Field(None, description="Your id. Re-sending the same id updates the document.")
    title: str = ""
    content: str
    metadata: dict = {}


class DocumentsIn(Schema):
    documents: list[DocumentIn] = Field(..., max_length=1000)


class DocumentOut(Schema):
    id: int
    external_id: str
    title: str
    metadata: dict
    chunk_count: int
    source_id: int | None
    updated_at: datetime


class DocumentDetail(DocumentOut):
    content: str


@router.post("/kbs/{kb_id}/documents")
def upsert_documents(request, kb_id: int, data: DocumentsIn):
    """Add or update documents (text you already have, e.g. FAQs). Unchanged documents are skipped."""
    kb = get_object_or_404(KnowledgeBase, pk=kb_id)
    docs = [
        DocInput(
            external_id=d.external_id or default_external_id(d.title, d.content),
            title=d.title,
            content=d.content,
            metadata=d.metadata,
        )
        for d in data.documents
    ]
    try:
        stats = ingest_documents(kb, docs)
    except IntegrityError as exc:
        raise HttpError(409, "The same documents are being written by another request; retry") from exc
    return {"stats": dict(stats), "external_ids": [d.external_id for d in docs]}


@router.get("/kbs/{kb_id}/documents", response=list[DocumentOut])
def list_documents(request, kb_id: int, offset: int = 0, limit: int = 50):
    get_object_or_404(KnowledgeBase, pk=kb_id)
    offset, limit = max(offset, 0), min(max(limit, 1), 500)
    return Document.objects.filter(kb_id=kb_id).order_by("id")[offset : offset + limit]


@router.get("/kbs/{kb_id}/documents/{path:external_id}", response=DocumentDetail)
def get_document(request, kb_id: int, external_id: str, source_id: int | None = None):
    """A document by your id. The same id can come from a data source and from the API: pass source_id to pick."""
    documents = Document.objects.filter(kb_id=kb_id, external_id=external_id)
    if source_id is not None:
        documents = documents.filter(source_id=source_id)
    document = documents.order_by(F("source_id").asc(nulls_first=True)).first()  # posted documents first
    if document is None:
        raise HttpError(404, "No such document")
    return document


@router.delete("/kbs/{kb_id}/documents/{path:external_id}", response={204: None})
def delete_document(request, kb_id: int, external_id: str):
    deleted, _ = Document.objects.filter(kb_id=kb_id, external_id=external_id, source__isnull=True).delete()
    if not deleted:
        raise HttpError(404, "No such document (documents from a data source are removed by syncing)")
    return 204, None


# --- Search ---------------------------------------------------------------------


class FilterIn(Schema):
    field: str = Field(..., min_length=1, max_length=100)
    op: Literal[FILTER_OPS] = "=="
    value: Any = None


class SearchIn(Schema):
    query: str = Field("", description="Empty: find by metadata alone (filters and sort_by), no embedding")
    kb_ids: list[int]
    top_k: int = Field(5, ge=1, le=50)
    metadata_filter: dict | None = Field(None, description='Exact matches, e.g. {"brand": "Samsung"}')
    filters: list[FilterIn] = Field(default_factory=list, description="Conditions that must all match")
    sort_by: str | None = Field(None, max_length=100, description="A metadata field to sort by, e.g. date_added")
    sort_order: Literal["desc", "asc"] = "desc"
    sort_pool: int | None = Field(
        None, ge=1, le=200, description="With a query and sort_by: how many best matches to sort (default: top_k)"
    )
    max_distance: float | None = Field(None, ge=0, le=2)


@router.post("/search")
def search_chunks(request, data: SearchIn):
    """Hybrid (vector + keyword) search, or by metadata alone with an empty query. No LLM involved.

    Use it to tune your data and settings. A filter is {"field", "op", "value"}; ops: == != > >= < <= contains in.
    """
    filters = [f.dict() for f in data.filters]
    sort = {"sort_by": data.sort_by or None, "sort_order": data.sort_order}
    if data.query.strip():
        results = search(
            data.query,
            data.kb_ids,
            data.top_k,
            data.metadata_filter,
            data.max_distance,
            filters=filters,
            sort_pool=data.sort_pool,
            **sort,
        )
    else:
        results = metadata_search(
            data.kb_ids, data.top_k, filters=filters, metadata_filter=data.metadata_filter, **sort
        )
    return {"results": results}


@router.get("/kbs/{kb_id}/metadata-fields")
def metadata_fields(request, kb_id: int):
    """The metadata fields of a knowledge base (from its latest 2000 documents), with a type and examples.

    The studio offers them when you pick fields to filter and sort by.
    """
    kb = get_object_or_404(KnowledgeBase, pk=kb_id)
    kinds: dict[str, Counter] = {}
    examples: dict[str, list] = {}
    for metadata in Document.objects.filter(kb=kb).order_by("-id").values_list("metadata", flat=True)[:2000]:
        for name, value in (metadata or {}).items():
            if value is None or value == "":
                continue
            kinds.setdefault(name, Counter())[field_type(value)] += 1
            seen = examples.setdefault(name, [])
            if len(seen) < 3 and value not in seen and not isinstance(value, (dict, list)):
                seen.append(value if not isinstance(value, str) else value[:60])
    return [
        {
            "name": name,
            "type": kinds[name].most_common(1)[0][0],
            "count": sum(kinds[name].values()),
            "examples": examples[name],
        }
        for name in sorted(kinds)
    ]


def field_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str) and DATE_TEXT.match(value):
        return "date"
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
        return "number"
    return "text"


# --- Data sources ---------------------------------------------------------------


class SourceIn(Schema):
    name: str
    url: str = Field(..., description="mysql://user:pass@host:3306/db, postgresql://..., or env:VAR_NAME")
    query: str = Field(..., description="e.g. SELECT id, name, description, category, price FROM products")
    id_column: str = "id"
    title_column: str = ""
    content_template: str = Field("", description='e.g. "{description}". Empty: all columns as "column: value" lines')
    metadata_columns: list[str] = []
    sync_interval_minutes: int = Field(0, ge=0)


class SourceOut(Schema):
    id: int
    kb_id: int
    name: str
    url: str
    query: str
    id_column: str
    title_column: str
    content_template: str
    metadata_columns: list[str]
    sync_interval_minutes: int
    last_synced_at: datetime | None
    last_sync_stats: dict


class JobOut(Schema):
    id: int
    source_id: int
    status: str
    stats: dict
    error: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


def source_out(source: DataSource) -> DataSource:
    source.url = mask_password(source.url)  # never return credentials
    return source


def mask_password(url: str) -> str:
    u = urlparse(url)
    if u.password:
        return url.replace(f":{u.password}@", ":***@", 1)
    return url


@router.post("/kbs/{kb_id}/sources", response={201: SourceOut})
def create_source(request, kb_id: int, data: SourceIn):
    """Connect a SQL table, e.g. your MySQL products table. Then call POST /sources/{id}/sync."""
    kb = get_object_or_404(KnowledgeBase, pk=kb_id)
    return 201, source_out(DataSource.objects.create(kb=kb, **data.dict()))


@router.get("/sources", response=list[SourceOut])
def list_sources(request):
    return [source_out(s) for s in DataSource.objects.order_by("id")]


@router.put("/sources/{source_id}", response=SourceOut)
def update_source(request, source_id: int, data: SourceIn):
    source = get_object_or_404(DataSource, pk=source_id)
    fields = data.dict()
    if fields["url"] == mask_password(source.url):
        fields["url"] = source.url  # the client sent back the masked URL: keep the password
    for key, value in fields.items():
        setattr(source, key, value)
    source.save()
    return source_out(source)


@router.delete("/sources/{source_id}", response={204: None})
def delete_source(request, source_id: int):
    """Deletes the source and all documents that came from it."""
    get_object_or_404(DataSource, pk=source_id).delete()
    return 204, None


@router.post("/sources/{source_id}/sync", response={202: JobOut})
def sync(request, source_id: int):
    """Queue a sync. The worker (python manage.py worker) runs it; poll GET /jobs/{id}."""
    return 202, enqueue_sync(get_object_or_404(DataSource, pk=source_id))


@router.get("/jobs/{job_id}", response=JobOut)
def get_job(request, job_id: int):
    return get_object_or_404(SyncJob, pk=job_id)


@router.get("/sources/{source_id}/jobs", response=list[JobOut])
def list_jobs(request, source_id: int, limit: int = 20):
    return SyncJob.objects.filter(source_id=source_id)[: min(max(limit, 1), 100)]


# --- Status -----------------------------------------------------------------------


@router.get("/status")
def status(request):
    """Which search indexes this database has (diskann/hnsw, bm25/fts) and how much is indexed."""
    from django.conf import settings

    from knowledge.models import Chunk
    from knowledge.search_indexes import backends, text_search_config

    return {
        "indexes": {**backends(), "text_search_config": text_search_config()},
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
        "llm_provider": settings.LLM_PROVIDER,
        "knowledge_bases": KnowledgeBase.objects.count(),
        "documents": Document.objects.count(),
        "chunks": Chunk.objects.count(),
        "queued_jobs": SyncJob.objects.filter(status=SyncJob.QUEUED).count(),
    }
