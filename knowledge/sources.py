"""Sync a SQL table (e.g. your MySQL products table) into a knowledge base.

A sync reads every row, page by page (keyset pagination on the id column, so
MySQL never holds a long-running cursor while we wait for embeddings):
  - new rows are added, rows whose text changed are re-embedded,
  - rows whose metadata changed (e.g. price) are updated without re-embedding,
  - unchanged rows cost nothing,
  - documents whose row disappeared are deleted.
"""

import html
import logging
import os
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import parse_qs, unquote, urlparse

from django.db import connection
from django.utils import timezone

from knowledge.ingest import DocInput, ingest_documents, json_safe
from knowledge.models import DataSource, Document, SyncJob

log = logging.getLogger(__name__)

PAGE_SIZE = 500
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
SYNC_LOCK = 7341  # advisory lock namespace: (SYNC_LOCK, source id)


class SyncError(RuntimeError):
    pass


# --- Reading rows ---------------------------------------------------------------


def resolve_url(url: str) -> str:
    if url.startswith("env:"):
        name = url[4:]
        value = os.getenv(name)
        if not value:
            raise SyncError(f"Environment variable {name} is not set")
        return value
    return url


def _connect(url: str):
    """Open a connection to the source database. Returns (connection, identifier quote char)."""
    u = urlparse(resolve_url(url))
    if u.scheme.startswith("mysql"):
        import pymysql

        options = {k: v[0] for k, v in parse_qs(u.query).items()}
        conn = pymysql.connect(
            host=u.hostname or "localhost",
            port=u.port or 3306,
            user=unquote(u.username or ""),
            password=unquote(u.password or ""),
            database=u.path.lstrip("/"),
            charset=options.get("charset", "utf8mb4"),
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
            read_timeout=300,
        )
        return conn, "`"
    if u.scheme.startswith("postgres"):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(resolve_url(url), row_factory=dict_row, autocommit=True), '"'
    raise SyncError(f"Unsupported source URL scheme {u.scheme!r}; use mysql:// or postgresql://")


def page_query(query: str, id_column: str, quote: str, after: bool) -> str:
    """Wrap the source query for keyset pagination: rows after the last id, ordered by id."""
    if not IDENTIFIER.match(id_column):
        raise SyncError(f"Invalid id column {id_column!r}")
    query = query.strip().rstrip(";").replace("%", "%%")  # we always pass parameters, so escape "%"
    col = f"src.{quote}{id_column}{quote}"
    where = f" WHERE {col} > %s" if after else ""
    return f"SELECT * FROM ({query}) AS src{where} ORDER BY {col} LIMIT {PAGE_SIZE}"


def fetch_pages(source: DataSource) -> Iterator[list[dict]]:
    conn, quote = _connect(source.url)
    try:
        last_id = None
        while True:
            with conn.cursor() as cur:
                if last_id is None:
                    cur.execute(page_query(source.query, source.id_column, quote, after=False), [])
                else:
                    cur.execute(page_query(source.query, source.id_column, quote, after=True), [last_id])
                rows = list(cur.fetchall())
            if not rows:
                return
            if source.id_column not in rows[0]:
                raise SyncError(f"The query has no column {source.id_column!r}")
            yield rows
            if len(rows) < PAGE_SIZE:
                return
            last_id = rows[-1][source.id_column]
    finally:
        conn.close()


# --- Row -> document ------------------------------------------------------------


def clean(value) -> str:
    """Text of one column: no HTML tags or entities (common in product descriptions)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    if "<" in text:
        text = HTML_TAG.sub(" ", text)
    return html.unescape(text).strip()


def render_row(template: str, row: dict) -> str:
    return PLACEHOLDER.sub(lambda m: clean(row.get(m.group(1))), template)


def check_columns(source: DataSource, row: dict) -> None:
    """Fail early when the settings name columns the query doesn't return."""
    used = set(PLACEHOLDER.findall(source.content_template)) | set(source.metadata_columns or [])
    used |= {c for c in (source.id_column, source.title_column) if c}
    missing = sorted(used - set(row))
    if missing:
        raise SyncError(f"The query has no column(s) {', '.join(missing)}; it returns {', '.join(row)}")


def row_to_document(source: DataSource, row: dict) -> DocInput:
    title = clean(row.get(source.title_column)) if source.title_column else ""
    if source.content_template:
        content = render_row(source.content_template, row)
    else:
        skip = {source.id_column, source.title_column}
        content = "\n".join(f"{col}: {clean(val)}" for col, val in row.items() if col not in skip and clean(val))
    return DocInput(
        external_id=str(row[source.id_column]),
        title=title,
        content=content.strip(),
        metadata={col: json_safe(row.get(col)) for col in source.metadata_columns or []},
    )


# --- Sync -----------------------------------------------------------------------


@contextmanager
def source_lock(source_id: int):
    """Postgres advisory lock: one sync per source at a time, across all workers."""
    with connection.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s, %s)", [SYNC_LOCK, source_id])
        if not cur.fetchone()[0]:
            raise SyncError("A sync of this source is already running")
    try:
        yield
    finally:
        with connection.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s, %s)", [SYNC_LOCK, source_id])


def sync_source(source: DataSource, job: SyncJob | None = None) -> dict:
    with source_lock(source.id):
        started = timezone.now()
        existing = set(Document.objects.filter(source=source).values_list("external_id", flat=True))
        seen: set[str] = set()
        stats: Counter = Counter()

        for rows in fetch_pages(source):
            if stats["rows"] == 0:
                check_columns(source, rows[0])
            docs = []
            for row in rows:
                stats["rows"] += 1
                doc = row_to_document(source, row)
                if doc.content:
                    seen.add(doc.external_id)
                    docs.append(doc)
            stats.update(ingest_documents(source.kb, docs, source=source))
            if job:  # progress, visible in GET /jobs/{id}
                SyncJob.objects.filter(pk=job.pk).update(stats=dict(stats))
            log.info("source %s: %s", source.id, dict(stats))

        stale = existing - seen
        if existing and not seen:
            # Most likely a broken query, template or database, not an empty catalog.
            raise SyncError(
                f"The source gave no documents ({stats['rows']} rows, all empty); "
                f"refusing to delete all {len(existing)} documents"
            )
        stale_ids = list(stale)
        for i in range(0, len(stale_ids), 1000):
            Document.objects.filter(source=source, external_id__in=stale_ids[i : i + 1000]).delete()
        stats["deleted"] = len(stale_ids)

        result = dict(stats, seconds=round((timezone.now() - started).total_seconds(), 1))
        source.last_synced_at = started
        source.last_sync_stats = result
        source.save(update_fields=["last_synced_at", "last_sync_stats"])
        return result
