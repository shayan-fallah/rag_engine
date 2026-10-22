"""Hybrid search: vector similarity + keyword (BM25) search, merged with Reciprocal Rank Fusion.

Same idea as pgvectorscale-rag-solution's VectorStore.hybrid_search:
  - vector search finds similar meaning ("cheap phone" ~ "budget smartphone")
  - keyword search finds exact terms that embeddings blur (model numbers, brand names)
  - RRF: score = 1/(k + vector rank) + 1/(k + keyword rank). Found by both = ranked highest.
"""

import json
import logging
import time

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

SELECT = """
SELECT c.id, c.document_id, c.kb_id, d.external_id, d.title, c.content, c.metadata,
       (COALESCE(1.0 / (%(rrf_k)s + semantic.rank), 0)
        + COALESCE(1.0 / (%(rrf_k)s + keyword.rank), 0))::float8 AS score,
       semantic.rank AS vector_rank, keyword.rank AS keyword_rank, semantic.distance
FROM semantic
FULL OUTER JOIN keyword USING (id)
JOIN knowledge_chunk AS c USING (id)
JOIN knowledge_document AS d ON d.id = c.document_id
ORDER BY score DESC, c.id
LIMIT %(top_k)s
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


def search(
    query: str,
    kb_ids: list[int],
    top_k: int = 5,
    metadata_filter: dict | None = None,
    max_distance: float | None = None,
) -> list[dict]:
    """Find the chunks that best match `query` in the given knowledge bases.

    metadata_filter: {"category": "Phones"} keeps chunks whose metadata contains these values.
    max_distance:    drop vector matches with a cosine distance above this (0 = same, 1 = unrelated).
                     Keyword matches are kept regardless.
    """
    query = (query or "").strip()
    if not query or not kb_ids:
        return []

    embedding = Vector(embed_text(query)).to_text()
    start = time.monotonic()

    kinds = backends()

    def filter_sql(column: str) -> str:
        return f"AND {column} @> %(metadata)s::jsonb" if metadata_filter else ""

    vector_cte = VECTOR_CTE.format(
        metadata_filter=filter_sql("metadata"),
        max_distance_filter="WHERE distance <= %(max_distance)s" if max_distance is not None else "",
    )
    if kinds["keyword"] == "bm25":
        keyword_cte = BM25_CTE.format(index=KEYWORD_INDEX, metadata_filter=filter_sql("metadata"))
    else:
        keyword_cte = FTS_CTE.format(config=text_search_config(), metadata_filter=filter_sql("c.metadata"))
    sql = f"WITH {vector_cte},\n{keyword_cte}\n{SELECT}"

    params = {
        "embedding": embedding,
        "query": query,
        "labels": list(kb_ids),
        "kb_ids": list(kb_ids),
        "metadata": json.dumps(metadata_filter or {}),
        "max_distance": max_distance,
        "candidates": max(top_k * 4, 20),
        "rrf_k": RRF_K,
        "top_k": top_k,
    }
    with transaction.atomic(), connection.cursor() as cur:
        if kinds["vector"] == "hnsw":
            # Keep scanning the HNSW index until enough rows pass the filters (pgvector >= 0.8).
            cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        cur.execute(sql, params)
        rows = [dict(zip(COLUMNS, row, strict=True)) for row in cur.fetchall()]
    for row in rows:
        if isinstance(row["metadata"], str):  # Django's psycopg setup returns jsonb as text
            row["metadata"] = json.loads(row["metadata"])

    log.info("search %r in kb %s: %d result(s) in %.3fs", query[:80], kb_ids, len(rows), time.monotonic() - start)
    return rows


def format_chunks(chunks: list[dict], start: int = 1) -> str:
    """Number the chunks for a prompt: [1] ..., [2] ... The LLM can cite these numbers."""
    blocks = []
    for i, chunk in enumerate(chunks, start=start):
        text = f"[{i}] {chunk['content']}"
        if chunk.get("metadata"):
            details = ", ".join(f"{k}: {v}" for k, v in chunk["metadata"].items() if v not in (None, ""))
            if details:
                text += f"\n({details})"
        blocks.append(text)
    return "\n\n".join(blocks)


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
        "score": round(chunk["score"], 5),
    }
