"""The two search indexes on the chunk table.

Best case (the timescale/timescaledb-ha image has both extensions):
  - vector:  pgvectorscale StreamingDiskANN, with the kb label inside the index
  - keyword: pg_textsearch BM25

Fallback for a Postgres that only has pgvector (e.g. many managed services):
  - vector:  pgvector HNSW (exact search above 2000 dimensions)
  - keyword: Postgres built-in full-text search (GIN index on to_tsvector)

The search query works with either; it checks which keyword index exists.
"""

import logging
import re

from django.conf import settings
from django.db import DatabaseError, connection, transaction

log = logging.getLogger(__name__)

TABLE = "knowledge_chunk"
VECTOR_INDEX = "knowledge_chunk_embedding_idx"
KEYWORD_INDEX = "knowledge_chunk_keyword_idx"


def text_search_config() -> str:
    config = settings.TEXT_SEARCH_CONFIG
    if not re.fullmatch(r"[a-z_]+", config):
        raise ValueError(f"Invalid TEXT_SEARCH_CONFIG {config!r}")
    return config


def _try_extension(conn, cursor, name: str) -> bool:
    cursor.execute("SELECT 1 FROM pg_available_extensions WHERE name = %s", [name])
    if not cursor.fetchone():
        return False
    try:
        with transaction.atomic(using=conn.alias):  # a savepoint, so a failure doesn't abort the migration
            cursor.execute(f"CREATE EXTENSION IF NOT EXISTS {name} CASCADE")
        return True
    except DatabaseError as exc:  # e.g. pg_textsearch missing from shared_preload_libraries
        log.warning("Extension %s is available but could not be created: %s", name, exc)
        return False


def create_search_indexes(schema_editor=None) -> dict:
    conn = schema_editor.connection if schema_editor else connection
    config = text_search_config()
    with conn.cursor() as cur:
        if _try_extension(conn, cur, "vectorscale"):
            vector = "diskann"
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {VECTOR_INDEX} ON {TABLE} "
                "USING diskann (embedding vector_cosine_ops, labels)"
            )
        elif settings.EMBEDDING_DIMENSIONS <= 2000:
            vector = "hnsw"
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {VECTOR_INDEX} ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
            )
        else:
            # pgvector's HNSW indexes at most 2000 dimensions. Without pgvectorscale,
            # search then compares every chunk (fine for tens of thousands of chunks).
            vector = "exact"
            log.warning(
                "No vector index: install pgvectorscale for %d-dimension embeddings", settings.EMBEDDING_DIMENSIONS
            )

        if _try_extension(conn, cur, "pg_textsearch"):
            keyword = "bm25"
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {KEYWORD_INDEX} ON {TABLE} "
                f"USING bm25 (content) WITH (text_config = '{config}')"
            )
        else:
            keyword = "fts"
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {KEYWORD_INDEX} ON {TABLE} "
                f"USING gin (to_tsvector('{config}'::regconfig, content))"
            )
    log.info("Search indexes: vector=%s keyword=%s (%s)", vector, keyword, config)
    _backends.clear()
    return {"vector": vector, "keyword": keyword, "text_search_config": config}


def drop_search_indexes(schema_editor=None) -> None:
    conn = schema_editor.connection if schema_editor else connection
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {VECTOR_INDEX}")
        cur.execute(f"DROP INDEX IF EXISTS {KEYWORD_INDEX}")
    _backends.clear()


_backends: dict[str, str] = {}


def backends() -> dict[str, str]:
    """Which index types exist: {"vector": "diskann"|"hnsw"|"exact", "keyword": "bm25"|"fts"}. Cached."""
    if not _backends:
        with connection.cursor() as cur:
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s AND indexname IN (%s, %s)",
                [TABLE, VECTOR_INDEX, KEYWORD_INDEX],
            )
            defs = dict(cur.fetchall())
        vector_def = defs.get(VECTOR_INDEX, "")
        # One update() call, so a concurrent thread never sees a half-filled cache.
        _backends.update(
            vector="diskann" if "diskann" in vector_def else "hnsw" if "hnsw" in vector_def else "exact",
            keyword="bm25" if "bm25" in defs.get(KEYWORD_INDEX, "") else "fts",
        )
    return _backends
