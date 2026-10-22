"""Documents -> chunks -> embeddings -> Postgres."""

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from django.db import transaction

from core.embeddings import embed_texts
from knowledge.chunking import split_text
from knowledge.models import Chunk, DataSource, Document, KnowledgeBase


@dataclass
class DocInput:
    external_id: str
    content: str
    title: str = ""
    metadata: dict = field(default_factory=dict)


def json_safe(value):
    """Make a value storable in a JSON column (database rows contain Decimal, datetime, bytes...)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def content_hash(doc: DocInput) -> str:
    return hashlib.sha256(f"{doc.title}\x00{doc.content}".encode()).hexdigest()


def default_external_id(title: str, content: str) -> str:
    return hashlib.sha1(f"{title}\x00{content}".encode()).hexdigest()[:20]


def chunk_document(kb: KnowledgeBase, doc: DocInput) -> list[str]:
    """Each chunk starts with the document title: a chunk alone often doesn't say what it is about."""
    parts = split_text(doc.content, kb.chunk_size, kb.chunk_overlap)
    return [f"{doc.title}\n{part}" if doc.title else part for part in parts]


def ingest_documents(kb: KnowledgeBase, docs: list[DocInput], source: DataSource | None = None) -> Counter:
    """Add new documents, update changed ones, skip unchanged ones.

    Only new or changed text is embedded. A metadata-only change (e.g. a new
    price) updates the metadata without calling the embedding model.
    """
    stats: Counter = Counter()
    unique = {}
    for doc in docs:
        doc.metadata = json_safe(doc.metadata or {})
        unique[doc.external_id] = doc  # the last one wins

    existing = {
        d.external_id: d
        for d in Document.objects.filter(kb=kb, source=source, external_id__in=list(unique)).only(
            "id", "external_id", "content_hash", "metadata"
        )
    }

    to_embed: list[tuple[DocInput, str, Document | None]] = []
    metadata_only: list[tuple[DocInput, Document]] = []
    for doc in unique.values():
        if not doc.content.strip():
            stats["skipped_empty"] += 1
            continue
        digest = content_hash(doc)
        old = existing.get(doc.external_id)
        if old is None:
            to_embed.append((doc, digest, None))
            stats["created"] += 1
        elif old.content_hash != digest:
            to_embed.append((doc, digest, old))
            stats["updated"] += 1
        elif old.metadata != doc.metadata:
            metadata_only.append((doc, old))
            stats["metadata_updated"] += 1
        else:
            stats["unchanged"] += 1

    # Embed before opening the transaction: API calls can be slow.
    chunk_lists = [chunk_document(kb, doc) for doc, _, _ in to_embed]
    vectors = iter(embed_texts([c for chunks in chunk_lists for c in chunks]) if chunk_lists else [])

    with transaction.atomic():
        for (doc, digest, old), chunks in zip(to_embed, chunk_lists, strict=True):
            if old is None:
                document = Document.objects.create(
                    kb=kb,
                    source=source,
                    external_id=doc.external_id,
                    title=doc.title[:500],
                    content=doc.content,
                    metadata=doc.metadata,
                    content_hash=digest,
                    chunk_count=len(chunks),
                )
            else:
                document = old
                document.title, document.content, document.metadata = doc.title[:500], doc.content, doc.metadata
                document.content_hash, document.chunk_count = digest, len(chunks)
                document.save(
                    update_fields=["title", "content", "metadata", "content_hash", "chunk_count", "updated_at"]
                )
                document.chunks.all().delete()
            Chunk.objects.bulk_create(
                Chunk(
                    kb=kb,
                    document=document,
                    position=i,
                    content=text,
                    metadata=doc.metadata,
                    labels=[kb.id],
                    embedding=next(vectors),
                )
                for i, text in enumerate(chunks)
            )
            stats["chunks"] += len(chunks)

        for doc, old in metadata_only:
            old.metadata = doc.metadata
            old.save(update_fields=["metadata", "updated_at"])
            Chunk.objects.filter(document=old).update(metadata=doc.metadata)
    return stats
