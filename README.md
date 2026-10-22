# rag_engine

A small RAG engine with a workflow (agent) builder, in Django. **Postgres is the only service it needs**: app tables, vectors and keyword search all live in one database.

- **Knowledge bases** filled from your **MySQL products table** (or any SQL query), or from documents you post to the API.
- **Hybrid search**: vector search (pgvectorscale DiskANN) + keyword search (pg_textsearch BM25), merged with Reciprocal Rank Fusion. Same approach and libraries as `pgvectorscale-rag-solution`: `psycopg` + `pgvector` + plain SQL, the `openai` and `anthropic` SDKs.
- **Workflows** like RAGFlow's agent canvas, run with **LangGraph**: begin, retrieval, llm, agent (with search tools), categorize, switch, message. Same `{sys.query}` / `{node@output}` variables.
- **Conversations** per user of your app, with streamed answers (Server-Sent Events) and sources.

Not included (on purpose, for now): PDF/Word parsing, other chunking methods, reranking, GraphRAG, RAPTOR, long-term memory, a visual editor, multi-tenancy.

```
  your backend ── HTTP (Bearer API_KEY) ──►  Django + django-ninja  (gunicorn, threads)
                                               │  /api/kbs, /api/search, /api/sources, /api/workflows/{id}/run ...
                                               │
                 ┌─────────────────────────────┼──────────────────────────────┐
                 ▼                             ▼                              ▼
         knowledge/                      workflows/                      core/
         sources.py  MySQL rows → docs   dsl.py     JSON → validated graph   llm.py         OpenAI-compatible / Claude
         ingest.py   docs → chunks →     engine.py  graph → LangGraph        embeddings.py  OpenAI-compatible
                     embeddings          nodes.py   what each node does
         search.py   hybrid search
         jobs.py     sync queue (worker)
                 │                             │
                 └──────────────► Postgres ◄───┘   pgvector · pgvectorscale (DiskANN) · pg_textsearch (BM25)
                                     ▲
   python manage.py worker ──────────┘  (syncs data sources; reads your MySQL)
```

| RAGFlow | here |
|---|---|
| Dataset | Knowledge base (`/api/kbs`) |
| Elasticsearch / Infinity | Postgres table `knowledge_chunk` with a DiskANN index and a BM25 index |
| MySQL (metadata) | the same Postgres |
| Redis + task executor | `sync_job` table + `python manage.py worker` |
| MinIO | not needed (no file uploads) |
| Agent (canvas) | Workflow (`/api/workflows`), JSON in the same spirit |
| Begin / Retrieval / LLM / Agent / Categorize / Switch / Message | `begin` / `retrieval` / `llm` / `agent` / `categorize` / `switch` / `message` |
| `{sys.query}`, `{Retrieval:Foo@formalized_content}` | `{sys.query}`, `{search@formalized_content}` |

---

## Run it

### With Docker

```bash
cp .env.example .env        # set OPENAI_API_KEY (or OPENAI_BASE_URL for LiteLLM), API_KEY, DJANGO_SECRET_KEY
docker compose up -d        # postgres + api (:8000) + worker
docker compose exec api python manage.py createsuperuser   # optional: admin at http://localhost:8000/admin/
```

API docs (Swagger): http://localhost:8000/api/docs

To try a data source with sample products: `docker compose --profile demo up -d` also starts a MySQL with `examples/products_mysql.sql` (URL `mysql://shop:shop@mysql-demo:3306/shop`).

### Without Docker

You need Postgres 17+ with pgvector, pgvectorscale and pg_textsearch (in `shared_preload_libraries`). The `timescale/timescaledb-ha:pg18` image has all three.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
gunicorn config.wsgi:application -k gthread --workers 2 --threads 16 --timeout 300 -b 0.0.0.0:8000
python manage.py worker      # in another terminal
```

**Only plain pgvector?** (e.g. a managed Postgres without pgvectorscale/pg_textsearch): it still works. The migration falls back to an HNSW vector index (no index above 2000 dimensions: exact search) and Postgres' built-in full-text search. `GET /api/status` shows which one you got. After installing the extensions later, run `python manage.py rebuild_search_indexes`.

---

## Use it

```bash
B=http://localhost:8000/api
H=(-H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json")
```

### 1. A knowledge base, filled from your MySQL products table

```bash
curl "${H[@]}" -X POST $B/kbs -d '{"name": "products"}'
# -> {"id": 1, ...}

curl "${H[@]}" -X POST $B/kbs/1/sources -d '{
  "name": "shop products",
  "url": "env:PRODUCTS_DB_URL",
  "query": "SELECT id, name, description, category, brand, price FROM products WHERE is_active = 1",
  "id_column": "id",
  "title_column": "name",
  "content_template": "Brand: {brand}\nCategory: {category}\n{description}",
  "metadata_columns": ["category", "brand", "price"],
  "sync_interval_minutes": 60
}'
# -> {"id": 1, ...}

curl "${H[@]}" -X POST $B/sources/1/sync      # queued; the worker runs it
curl "${H[@]}" $B/jobs/1                       # {"status": "done", "stats": {"rows": 11, "created": 11, ...}}
```

How a row becomes a document:

| field | meaning |
|---|---|
| `url` | `mysql://user:pass@host:3306/db`, `postgresql://...`, or `env:VAR_NAME` to keep the password in an environment variable. API responses mask the password. Use a read-only database user. |
| `query` | Any `SELECT`. Filter here (active products only, one language...). |
| `id_column` | Unique and sortable (the sync pages through rows ordered by it). Becomes the document's `external_id`. |
| `title_column` | Optional. The title is put at the top of every chunk, so a chunk always says which product it is about. |
| `content_template` | The text that is embedded and searched. `{column}` is replaced by the row's value; HTML tags and entities are removed. Empty: every column as `column: value` lines. |
| `metadata_columns` | Stored with each chunk: shown to the model next to the text (e.g. the price), returned with results, and usable as filters. |
| `sync_interval_minutes` | The worker re-syncs this often (0 = only when you call `/sync`). |

A sync reads the whole query result page by page and:
- **embeds only new or changed text** (it compares a hash), so re-syncing 100k unchanged products costs no embedding calls,
- **updates metadata without re-embedding** (a new price is cheap),
- **deletes** documents whose row is gone,
- **refuses to delete everything** if the query suddenly returns no rows or only empty text (a broken query or a typo in the template shouldn't wipe the knowledge base), and stops early if the settings name a column the query doesn't return.

You can also run a sync directly, e.g. from cron: `python manage.py sync_source 1`.

### 2. Or post documents directly (FAQs, policies...)

```bash
curl "${H[@]}" -X POST $B/kbs/1/documents -d '{"documents": [
  {"external_id": "faq-shipping", "title": "Shipping", "content": "Orders ship within 2 days...", "metadata": {"type": "faq"}}
]}'
```

Re-posting the same `external_id` updates it (only if it changed). `GET` / `DELETE /api/kbs/1/documents/faq-shipping` reads / removes it (`GET` takes `?source_id=` when a synced row has the same id).

### 3. Check what search finds (no LLM involved)

```bash
curl "${H[@]}" -X POST $B/search -d '{"query": "wireless headphones", "kb_ids": [1], "top_k": 5}'
curl "${H[@]}" -X POST $B/search -d '{"query": "phone", "kb_ids": [1], "metadata_filter": {"brand": "Samsung"}}'
```

Each result has `score` (RRF), `vector_rank`, `keyword_rank` (null = not found by that search) and `distance` (cosine distance, 0 = identical, ~1 = unrelated). Use this to tune `max_distance`, your `content_template` and `TEXT_SEARCH_CONFIG`.

### 4. A workflow, and a conversation with it

Three templates are in `workflows/templates/` (they use knowledge base 1):

| template | what it does |
|---|---|
| `rag_chat.json` | search → answer from the results. The basic RAG chat. |
| `product_assistant.json` | categorize the message → product question: rewrite it into a search query using the conversation, search, answer (or a fixed reply when nothing is found); order question: fixed reply; small talk: short answer. |
| `product_agent.json` | the model gets a `search_products` tool and decides what (and how often) to search. |

```bash
curl "${H[@]}" -X POST $B/workflows -d @workflows/templates/product_assistant.json
# -> {"id": 1, ...}

# Streamed (Server-Sent Events). Without conversation_id a new conversation starts.
curl -N "${H[@]}" -X POST $B/workflows/1/run -d '{"query": "a phone with a good camera?", "user_id": "user-42"}'

# Next turn: send the conversation_id back. "stream": false returns one JSON object.
curl "${H[@]}" -X POST $B/workflows/1/run \
  -d '{"query": "is there a cheaper one?", "user_id": "user-42", "conversation_id": "<id>", "stream": false}'
```

The stream is one `data: {json}` line per event, then `data: [DONE]`:

```
data: {"event": "conversation", "conversation_id": "3f9d…"}
data: {"event": "node_started", "node": "search", "type": "retrieval"}
data: {"event": "node_finished", "node": "search", "type": "retrieval", "elapsed_ms": 83, "outputs": {...}}
data: {"event": "tool_call", "node": "agent", "tool": "search_products", "query": "...", "results": 5}   (agent nodes)
data: {"event": "message", "delta": "The Pixel 9 "}          ← the answer, as it is written
data: {"event": "done", "answer": "...", "references": [{"ref": 1, "title": "Pixel 9", "external_id": "2", "cited": true, ...}], "conversation_id": "…", "message_id": 17}
data: {"event": "error", "message": "..."}                    (instead of done, if something failed)
```

`references` are the chunks the answer was based on; `ref` is the number the model cites (`[1]`), `cited` says whether the answer cites it, and `external_id` is your product id.

Conversations: `GET /api/workflows/1/conversations?user_id=user-42`, `GET /api/conversations/<id>?user_id=user-42` (with messages and references), `DELETE` the same URL. A conversation can only be used and read with the `user_id` that started it.

---

## Workflows

A workflow is JSON: nodes, each with a `type`, `params` and (except routing nodes) a `next` node. A node without `next` ends the run.

```json
{
  "nodes": {
    "begin":  {"type": "begin", "next": "search"},
    "search": {"type": "retrieval", "params": {"kb_ids": [1], "query": "{sys.query}"}, "next": "answer"},
    "answer": {"type": "llm", "params": {"system_prompt": "Answer from:\n{search@formalized_content}", "history": 6}, "next": "reply"},
    "reply":  {"type": "message", "params": {"text": "{answer@content}"}}
  }
}
```

It is checked when you save it (unknown nodes, bad references, missing knowledge bases, wrong params all return a 400 with the reason).

**Variables** can be used in any text param:
- `{sys.query}` the user's message, `{sys.user_id}`, `{sys.conversation_id}`, `{sys.date}`, `{sys.turns}`, `{sys.history}` (the conversation as text)
- `{node_id@output}` an output of a node that ran earlier, e.g. `{search@formalized_content}`, `{classify@category}`, `{begin@language}`

| type | params | outputs |
|---|---|---|
| `begin` | `inputs`: names and default values, overridden by the request's `"inputs"` | each input |
| `retrieval` | `kb_ids`, `query` (default `{sys.query}`), `top_k` (5), `metadata_filter`, `max_distance`, `empty_response` | `formalized_content` (numbered results for a prompt), `chunks`, `count`, `query` |
| `llm` | `system_prompt`, `prompt` (default `{sys.query}`), `history` (previous messages to include, 0), `provider`, `model`, `temperature`, `effort` | `content` |
| `agent` | like `llm`, plus `tools`: `[{name, description, kb_ids, top_k, metadata_filter, max_distance}]` and `max_rounds` (4) | `content` |
| `categorize` | `input` (default `{sys.query}`), `history`, `categories`: `[{name, description, examples, next}]`, model options | `category` |
| `switch` | `cases`: `[{conditions: [{var, op, value}], logic: "and"/"or", next}]`, `default`. Ops: `== != > >= < <= contains not_contains empty not_empty` | `case` |
| `message` | `text` | `content` |

The answer is the text of the `message` nodes that ran. **Streaming:** when an `llm`/`agent` node's `next` is a message node whose text is exactly `{that_node@content}`, its tokens are streamed to the user as they are generated (like RAGFlow). Any other message is sent whole when the node runs. If an agent writes text before a tool call ("Let me look that up"), it is part of the answer too. Loops are allowed (a switch can lead back); `max_steps` (default 50) stops runaway runs.

Retrieval results are numbered across the whole run (`[1]`, `[2]`... continuing over several retrieval nodes and agent searches), so the model's citations always match `references[].ref`.

Adding a node type: a function in `workflows/nodes.py` (`RUNNERS`), its params model in `workflows/dsl.py` (`PARAMS`).

---

## Models

| setting | |
|---|---|
| `LLM_PROVIDER=openai` | OpenAI, or any OpenAI-compatible server: set `OPENAI_BASE_URL` (LiteLLM proxy, vLLM, Ollama...) and `OPENAI_MODEL`. Uses Chat Completions. |
| `LLM_PROVIDER=anthropic` | Claude via the `anthropic` SDK, `ANTHROPIC_MODEL` (default `claude-opus-5-5`). Declined requests are retried on a fallback model server-side (`ANTHROPIC_FALLBACKS`, Claude API only). |
| `EMBEDDING_*` | Always an OpenAI-compatible endpoint (Anthropic has no embeddings API). Defaults to `text-embedding-3-small`, 1536 dimensions. |

A node can override `provider`, `model`, `temperature` and `effort`, e.g. a cheaper model for `categorize` and `rewrite`. `temperature` is unset by default because current reasoning models reject it.

`EMBEDDING_DIMENSIONS` is fixed by the first migration (the column is `vector(1536)`). To change the embedding model, start with a new database (or write a migration that changes the column and re-sync).

## How search works

```
query ──► embedding ──► DiskANN index, filtered by knowledge base inside the index ──► top 20 by cosine distance
      └─────────────► BM25 index (pg_textsearch) ──────────────────────────────────► top 20 by BM25
                      merge: score = 1/(60 + vector rank) + 1/(60 + keyword rank) → top_k
```

- **Vector search** finds meaning ("cheap phone" ≈ "budget smartphone"). **Keyword search** finds exact terms embeddings blur: model numbers (`SM-S921B`), brands, sizes.
- Every chunk stores `labels = [knowledge base id]`, so pgvectorscale filters by knowledge base *during* the index search (no lost results when one knowledge base is small).
- `TEXT_SEARCH_CONFIG` decides how words are normalized for keyword search. `english` stems English. **For Persian/Arabic and mixed-language catalogs use `simple`**, then `python manage.py rebuild_search_indexes`.

## Operations

- `python manage.py worker`: runs queued syncs and scheduled ones. Run one or several; a job runs once (`FOR UPDATE SKIP LOCKED`), and a Postgres advisory lock makes sure a source is never synced twice at the same time. A failing source is retried once per `sync_interval_minutes`. Jobs of a worker that died are marked failed.
- Behind an HTTPS reverse proxy, set `CSRF_TRUSTED_ORIGINS` and `BEHIND_HTTPS_PROXY=true` (needed for the admin login).
- `GET /api/status`: index types, counts, queued jobs.
- `/admin/`: browse knowledge bases, documents, chunks, sources (with a "Sync now" action), jobs, workflows and conversations.
- `python manage.py rebuild_search_indexes`: after installing extensions or changing `TEXT_SEARCH_CONFIG`.

## Code, in reading order

1. `knowledge/models.py`: knowledge base, data source, document, chunk (the vector store), sync job.
2. `knowledge/sources.py`: MySQL rows → documents; the sync.
3. `knowledge/ingest.py` + `chunking.py`: documents → chunks → embeddings, skipping what didn't change.
4. `knowledge/search.py` + `search_indexes.py`: the hybrid search SQL and its indexes.
5. `core/llm.py`: one small interface (`stream`, `parse`, `run_tools`) over the OpenAI and Anthropic SDKs.
6. `workflows/dsl.py` → `engine.py` → `nodes.py`: the workflow format, how it becomes a LangGraph graph, what each node does.
7. `knowledge/api.py`, `workflows/api.py`: the HTTP API.

## Tests

```bash
pip install -r requirements-dev.txt
DATABASE_URL=postgresql://postgres:password@localhost:5432/postgres pytest
```

They need a Postgres with the extensions (the compose one works) and no API keys: embeddings and chat models are replaced by fakes, and `tests/test_llm_providers.py` runs the real OpenAI/Anthropic SDK code against small local fake servers.
