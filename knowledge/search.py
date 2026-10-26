"""Search: hybrid (vector + keyword, merged with Reciprocal Rank Fusion) or by metadata alone.

Same idea as pgvectorscale-rag-solution's VectorStore.hybrid_search:
  - vector search finds similar meaning ("cheap phone" ~ "budget smartphone")
  - keyword search finds exact terms that embeddings blur (model numbers, brand names)
  - RRF: score = 1/(k + vector rank) + 1/(k + keyword rank). Found by both = ranked highest.

Both kinds take metadata filters ({field, op, value}) and an optional sort by a
metadata field, e.g. the newest products: metadata_search(kb_ids, sort_by="date_added").
"""

import json
import logging
import re
import time
from decimal import Decimal

from django.db import connection, transaction
from pgvector import Vector

from core.embeddings import embed_text
from knowledge.search_indexes import KEYWORD_INDEX, backends, text_search_config

log = logging.getLogger(__name__)

RRF_K = 60

VECTOR_CTE = """
semantic AS (
    SELECT id, distance, row_number() OVER (ORDER BY distance) AS rank
    FROM (
        SELECT id, embedding <=> %(embedding)s::vector AS distance
        FROM knowledge_chunk
        WHERE labels && %(labels)s::smallint[] {metadata_filter}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(candidates)s
    ) AS s
    {max_distance_filter}
)"""

BM25_CTE = """
keyword AS (
    -- <@> returns negative BM25 scores: lower is better, 0 = no shared terms
    SELECT id, row_number() OVER (ORDER BY score) AS rank
    FROM (
        SELECT id, content <@> to_bm25query(%(query)s, '{index}') AS score
        FROM knowledge_chunk
        WHERE kb_id = ANY(%(kb_ids)s) {metadata_filter}
        ORDER BY content <@> to_bm25query(%(query)s, '{index}')
        LIMIT %(candidates)s
    ) AS k
    WHERE score < 0
)"""

# Postgres built-in full-text search, for databases without pg_textsearch.
# Terms are OR-ed (like BM25), so a question doesn't need every word to match.
FTS_CTE = """
keyword AS (
    SELECT id, row_number() OVER (ORDER BY score DESC) AS rank
    FROM (
        SELECT c.id, ts_rank_cd(to_tsvector('{config}'::regconfig, c.content), q.query) AS score
        FROM knowledge_chunk AS c,
             (SELECT replace(plainto_tsquery('{config}'::regconfig, %(query)s)::text, '&', '|')::tsquery AS query) AS q
        WHERE c.kb_id = ANY(%(kb_ids)s) {metadata_filter}
          AND to_tsvector('{config}'::regconfig, c.content) @@ q.query
        ORDER BY score DESC
        LIMIT %(candidates)s
    ) AS k
)"""

MATCHES = """
SELECT c.id, c.document_id, c.kb_id, d.external_id, d.title, c.content, c.metadata,
       (COALESCE(1.0 / (%(rrf_k)s + semantic.rank), 0)
        + COALESCE(1.0 / (%(rrf_k)s + keyword.rank), 0))::float8 AS score,
       semantic.rank AS vector_rank, keyword.rank AS keyword_rank, semantic.distance
FROM semantic
FULL OUTER JOIN keyword USING (id)
JOIN knowledge_chunk AS c USING (id)
JOIN knowledge_document AS d ON d.id = c.document_id
ORDER BY score DESC, c.id
LIMIT %(pool)s
"""

# Metadata only: one result per document (its first chunk), no query. The documents are
# filtered, sorted and cut first, so only the returned ones look up their chunk.
BY_METADATA = """
SELECT c.id, d.id, d.kb_id, d.external_id, d.title, c.content, d.metadata,
       NULL::float8, NULL::bigint, NULL::bigint, NULL::float8
FROM (
    SELECT d.id, d.kb_id, d.external_id, d.title, d.metadata
    FROM knowledge_document AS d
    WHERE d.kb_id = ANY(%(kb_ids)s) AND d.chunk_count > 0 {metadata_filter}
    ORDER BY {order} d.id
    LIMIT %(top_k)s
) AS d
JOIN LATERAL (
    SELECT id, content FROM knowledge_chunk WHERE document_id = d.id ORDER BY position LIMIT 1
) AS c ON true
ORDER BY {order} d.id
"""

COLUMNS = [
    "chunk_id",
    "document_id",
    "kb_id",
    "external_id",
    "title",
    "content",
    "metadata",
    "score",
    "vector_rank",
    "keyword_rank",
    "distance",
]

# --- Metadata filters and sorting -------------------------------------------------------
#
# A filter is {"field": "price", "op": "<", "value": 500}. How values compare depends on the value:
#   numbers (also "500")      as numbers; metadata text like "799.00" counts as a number
#   dates ("2025-03-01")      as dates, to the precision written: "2025-03" is the whole month
#   true / false              matches true/false, 1/0 and "yes"/"no"
#   other text                == and != ignore case; > and < compare alphabetically
#   contains                  text contains the value (ignoring case), or a list has it as an item
#   in                        equals one of a list of values (a text value is split on commas)
# A missing field never matches, except with != (it is "not equal").

FILTER_OPS = ("==", "!=", ">", ">=", "<", "<=", "contains", "in")
SORT_ORDERS = ("desc", "asc")
NUMBER_TEXT = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")
DATE_TEXT = re.compile(r"^\d{4}-\d{2}(-\d{2}([T ]\d{2}(:\d{2}(:\d{2}(\.\d+)?)?)?)?)?$")
SQL_NUMBER = r"'^\s*[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?\s*$'"
TRUE_TEXTS, FALSE_TEXTS = ["true", "1", "yes"], ["false", "0", "no"]


def check_filter(f: dict) -> dict:
    """One filter {field, op, value}, checked and cleaned. Raises ValueError with a readable message."""
    if not isinstance(f, dict):
        raise ValueError("a filter must be an object with field, op and value")
    field, op, value = f.get("field"), f.get("op", "=="), f.get("value")
    if not isinstance(field, str) or not field.strip() or len(field) > 100:
        raise ValueError("field must be a metadata field name")
    if op not in FILTER_OPS:
        raise ValueError(f"op must be one of {', '.join(FILTER_OPS)}")
    if op == "in":
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, list) or not value or len(value) > 100:
            raise ValueError("'in' needs a list of values")
        if any(v is None or isinstance(v, (list, dict)) for v in value):
            raise ValueError("'in' values must be text, numbers or true/false")
    elif value is None or isinstance(value, (list, dict)):
        raise ValueError("value must be text, a number, true/false or a date")
    elif isinstance(value, bool) and op not in ("==", "!="):
        raise ValueError("true/false only works with == and !=")
    return {"field": field.strip(), "op": op, "value": value}


def check_filters(filters: list[dict] | None) -> list[dict]:
    """check_filter() for each filter; errors say which one ("filter 2: ...")."""
    checked = []
    for i, f in enumerate(filters or []):
        try:
            checked.append(check_filter(f))
        except ValueError as exc:
            raise ValueError(f"filter {i + 1}: {exc}") from None
    return checked


def _number(text: str) -> str:
    return f"CASE WHEN {text} ~ {SQL_NUMBER} THEN ({text})::numeric END"


def _equals(text: str, value, name: str, params: dict) -> str:
    """SQL that is true when the metadata text equals value (see the rules above)."""
    if isinstance(value, bool):
        params[name] = TRUE_TEXTS if value else FALSE_TEXTS
        return f"lower({text}) = ANY(%({name})s)"
    return _compare(text, "==", value, name, params)


def _compare(text: str, op: str, value, name: str, params: dict) -> str:
    sql_op = "=" if op == "==" else op
    if isinstance(value, (int, float)) or (isinstance(value, str) and NUMBER_TEXT.match(value)):
        params[name] = Decimal(str(value).strip())
        return f"{_number(text)} {sql_op} %({name})s"
    value = str(value)
    if DATE_TEXT.match(value):
        value = value.replace(" ", "T")
        params[name], params[name + "_len"] = value, len(value)
        return f"left(replace({text}, ' ', 'T'), %({name}_len)s) COLLATE \"C\" {sql_op} %({name})s"
    params[name] = value
    if op == "==":
        return f"lower({text}) = lower(%({name})s)"
    return f'{text} COLLATE "C" {sql_op} %({name})s'


def filter_sql(filters: list[dict], column: str, params: dict, prefix: str = "f") -> str:
    """ "AND ..." conditions on a jsonb column for check_filters() output. Values go into params."""
    parts = []
    for i, f in enumerate(filters):
        name = f"{prefix}{i}"
        params[name + "_field"] = f["field"]
        json_value = f"({column} -> %({name}_field)s)"
        text = f"({column} ->> %({name}_field)s)"
        op, value = f["op"], f["value"]
        if op == "contains":
            params[name] = str(value)
            parts.append(
                f"CASE WHEN jsonb_typeof({json_value}) = 'array' THEN EXISTS "
                f"(SELECT 1 FROM jsonb_array_elements_text({json_value}) AS e(v) WHERE lower(e.v) = lower(%({name})s)) "
                f"ELSE strpos(lower({text}), lower(%({name})s)) > 0 END"
            )
        elif op == "in":
            parts.append("(" + " OR ".join(_equals(text, v, f"{name}_{j}", params) for j, v in enumerate(value)) + ")")
        elif op == "!=":
            parts.append(f"NOT coalesce({_equals(text, value, name, params)}, false)")
        elif op == "==":
            parts.append(_equals(text, value, name, params))
        else:
            parts.append(_compare(text, op, value, name, params))
    return "".join(f" AND {p}" for p in parts)


def order_sql(sort_by: str, order: str, column: str, params: dict) -> str:
    """Sort by a metadata field: numbers as numbers, then text (ISO dates sort right). Missing last."""
    if order not in SORT_ORDERS:
        raise ValueError(f"order must be one of {', '.join(SORT_ORDERS)}")
    params["sort_field"] = sort_by
    text = f"({column} ->> %(sort_field)s)"
    direction = "DESC" if order == "desc" else "ASC"
    return f"{_number(text)} {direction} NULLS LAST, replace({text}, ' ', 'T') COLLATE \"C\" {direction} NULLS LAST"


def _metadata_sql(metadata_filter: dict | None, filters: list[dict], column: str, params: dict) -> str:
    sql = f" AND {column} @> %(metadata)s::jsonb" if metadata_filter else ""
    params["metadata"] = json.dumps(metadata_filter or {})
    return sql + filter_sql(filters, column, params)


def _rows(sql: str, params: dict, hnsw: bool = False) -> list[dict]:
    with transaction.atomic(), connection.cursor() as cur:
        if hnsw:
            # Keep scanning the HNSW index until enough rows pass the filters (pgvector >= 0.8).
            cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        cur.execute(sql, params)
        rows = [dict(zip(COLUMNS, row, strict=True)) for row in cur.fetchall()]
    for row in rows:
        if isinstance(row["metadata"], str):  # Django's psycopg setup returns jsonb as text
            row["metadata"] = json.loads(row["metadata"])
    return rows


def search(
    query: str,
    kb_ids: list[int],
    top_k: int = 5,
    metadata_filter: dict | None = None,
    max_distance: float | None = None,
    *,
    filters: list[dict] | None = None,
    sort_by: str | None = None,
    sort_order: str = "desc",
    sort_pool: int | None = None,
) -> list[dict]:
    """Find the chunks that best match `query` in the given knowledge bases.

    metadata_filter: {"category": "Phones"} keeps chunks whose metadata contains exactly these values.
    filters:         [{"field": "price", "op": "<", "value": 500}, ...]: all must match (see above).
    max_distance:    drop vector matches with a cosine distance above this (0 = same, 1 = unrelated).
                     Keyword matches are kept regardless.
    sort_by:         a metadata field: the `sort_pool` best matches (default: top_k) are sorted by
                     it, e.g. the newest of the matching products, and the first top_k returned.
    """
    query = (query or "").strip()
    if not query or not kb_ids:
        return []
    filters = check_filters(filters)

    embedding = Vector(embed_text(query)).to_text()
    start = time.monotonic()
    kinds = backends()
    pool = max(sort_pool or top_k, top_k) if sort_by else top_k
    params = {
        "embedding": embedding,
        "query": query,
        "labels": list(kb_ids),
        "kb_ids": list(kb_ids),
        "max_distance": max_distance,
        "candidates": max(pool * 4, 20),
        "rrf_k": RRF_K,
        "pool": pool,
        "top_k": top_k,
    }

    vector_cte = VECTOR_CTE.format(
        metadata_filter=_metadata_sql(metadata_filter, filters, "metadata", params),
        max_distance_filter="WHERE distance <= %(max_distance)s" if max_distance is not None else "",
    )
    if kinds["keyword"] == "bm25":
        keyword_cte = BM25_CTE.format(
            index=KEYWORD_INDEX, metadata_filter=_metadata_sql(metadata_filter, filters, "metadata", params)
        )
    else:
        keyword_cte = FTS_CTE.format(
            config=text_search_config(), metadata_filter=_metadata_sql(metadata_filter, filters, "c.metadata", params)
        )
    if sort_by:
        order = order_sql(sort_by, sort_order, "metadata", params)
        sql = (
            f"WITH {vector_cte},\n{keyword_cte},\nmatches AS ({MATCHES})\n"
            f"SELECT * FROM matches ORDER BY {order}, score DESC, id LIMIT %(top_k)s"
        )
    else:
        sql = f"WITH {vector_cte},\n{keyword_cte}\n{MATCHES}"
    rows = _rows(sql, params, hnsw=kinds["vector"] == "hnsw")

    log.info("search %r in kb %s: %d result(s) in %.3fs", query[:80], kb_ids, len(rows), time.monotonic() - start)
    return rows


def metadata_search(
    kb_ids: list[int],
    top_k: int = 5,
    *,
    filters: list[dict] | None = None,
    sort_by: str | None = None,
    sort_order: str = "desc",
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Documents found by metadata alone, no query: e.g. the 5 newest products in a category.

    One result per document (its first chunk). Without sort_by, in the order they were added.
    score, vector_rank, keyword_rank and distance are None.
    """
    if not kb_ids:
        return []
    filters = check_filters(filters)
    start = time.monotonic()
    params = {"kb_ids": list(kb_ids), "top_k": top_k}
    sql = BY_METADATA.format(
        metadata_filter=_metadata_sql(metadata_filter, filters, "d.metadata", params),
        order=order_sql(sort_by, sort_order, "d.metadata", params) + "," if sort_by else "",
    )
    rows = _rows(sql, params)
    log.info(
        "metadata search in kb %s (%d filters, sort %s): %d result(s) in %.3fs",
        kb_ids,
        len(filters),
        sort_by,
        len(rows),
        time.monotonic() - start,
    )
    return rows


# How format_chunks writes each result. Nodes can override it ("Result format" in the studio).
CHUNK_TEMPLATE = "[{ref}] {content}\n({metadata})"
CHUNK_FIELD = re.compile(r"\{(ref|content|title|external_id|metadata(?:\.[^{}\s]+)?)\}")


def format_chunks(chunks: list[dict], start: int = 1, template: str = CHUNK_TEMPLATE) -> str:
    """Number the chunks for a prompt: [1] ..., [2] ... The LLM can cite these numbers.

    Template placeholders: {ref} {content} {title} {external_id} {metadata} (all fields
    as "name: value") and {metadata.NAME} (one field). A line with only metadata
    placeholders, all empty, is left out: e.g. "({metadata})" for a chunk without metadata.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=start):
        metadata = chunk.get("metadata") or {}
        values = {
            "ref": str(i),
            "content": chunk["content"],
            "title": chunk.get("title") or "",
            "external_id": chunk.get("external_id") or "",
            "metadata": ", ".join(f"{k}: {v}" for k, v in metadata.items() if v not in (None, "")),
        }

        lines = []
        for line in template.split("\n"):
            fields = CHUNK_FIELD.findall(line)
            only_metadata = fields and all(f.startswith("metadata") for f in fields)
            if only_metadata and not any(_chunk_value(f, values, metadata) for f in fields):
                continue
            lines.append(CHUNK_FIELD.sub(lambda m, v=values, md=metadata: _chunk_value(m.group(1), v, md), line))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _chunk_value(name: str, values: dict, metadata: dict) -> str:
    if name.startswith("metadata."):
        field = metadata.get(name[len("metadata.") :])
        return "" if field is None else str(field)
    return values[name]


def as_reference(chunk: dict, number: int) -> dict:
    """The part of a search result that is returned to API clients as a source."""
    return {
        "ref": number,
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "kb_id": chunk["kb_id"],
        "external_id": chunk["external_id"],
        "title": chunk["title"],
        "content": chunk["content"],
        "metadata": chunk["metadata"],
        "score": None if chunk["score"] is None else round(chunk["score"], 5),
    }
