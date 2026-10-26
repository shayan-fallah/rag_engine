# rag_engine

A small RAG engine with a workflow (agent) builder, in Django. **Postgres is the only service it needs**: app tables, vectors and keyword search all live in one database.

- **Knowledge bases** filled from your **MySQL products table** (or any SQL query), or from documents you post to the API.
- **Hybrid search**: vector search (pgvectorscale DiskANN) + keyword search (pg_textsearch BM25), merged with Reciprocal Rank Fusion. Same approach and libraries as `pgvectorscale-rag-solution`: `psycopg` + `pgvector` + plain SQL, the `openai` and `anthropic` SDKs.
- **Metadata search**: filter and sort by metadata (`price < 500`, `date_added >= 2025-01`, newest first), with or without a query. A retrieval node can be set to it, and an agent's search tool lets the model decide per call: metadata alone ("the newest product"), a query, or both ("the newest phone with a good camera").
- **Workflows** like RAGFlow's agent canvas, run with **LangGraph**: begin, retrieval, llm, agent (with search tools), categorize, switch, message. Same `{sys.query}` / `{node@output}` variables.
- **Studio**: a visual editor for workflows at `/studio/`: drag nodes onto a canvas, connect them, edit every setting of every node (prompts, the internal prompts too, models, search settings) and test-run before saving, with a run log of every node's inputs and outputs.
- **Conversations** per user of your app, with streamed answers (Server-Sent Events) and sources.

Not included (on purpose, for now): PDF/Word parsing, other chunking methods, reranking, GraphRAG, RAPTOR, long-term memory, multi-tenancy.

```
  your backend ── HTTP (Bearer API_KEY) ──►  Django + django-ninja  (gunicorn, threads)
  your browser ── /studio/ (admin login) ──►   │  /api/kbs, /api/search, /api/sources, /api/workflows/{id}/run ...
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
| Agent (canvas) | Workflow (`/api/workflows`), JSON in the same spirit, edited on a canvas in the studio (`/studio/`) |
| Begin / Retrieval / LLM / Agent / Categorize / Switch / Message | `begin` / `retrieval` / `llm` / `agent` / `categorize` / `switch` / `message` |
| `{sys.query}`, `{Retrieval:Foo@formalized_content}` | `{sys.query}`, `{search@formalized_content}` |

---

## Run it

### With Docker

```bash
cp .env.example .env        # set OPENAI_API_KEY (or OPENAI_BASE_URL for LiteLLM), API_KEY, DJANGO_SECRET_KEY
docker compose up -d        # postgres + api (:8000) + worker
docker compose exec api python manage.py createsuperuser   # your login for the studio and the admin
```

Studio (workflow editor): http://localhost:8000/studio/ · API docs (Swagger): http://localhost:8000/api/docs · Admin: http://localhost:8000/admin/

To try a data source with sample products: `docker compose --profile demo up -d` also starts a MySQL with `examples/products_mysql.sql` (URL `mysql://shop:shop@mysql-demo:3306/shop`).

### Without Docker

You need Postgres 17+ with pgvector, pgvectorscale and pg_textsearch (in `shared_preload_libraries`). The `timescale/timescaledb-ha:pg18` image has all three.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py createsuperuser   # your login for the studio and the admin
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
  "query": "SELECT id, name, description, category, brand, price, date_added FROM products WHERE is_active = 1",
  "id_column": "id",
  "title_column": "name",
  "content_template": "Brand: {brand}\nCategory: {category}\n{description}",
  "metadata_columns": ["category", "brand", "price", "date_added"],
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
| `metadata_columns` | Stored with each chunk: shown to the model next to the text (e.g. the price), returned with results, and usable to filter and sort (e.g. the newest products by `date_added`). Dates are stored as ISO text (`2025-03-01T10:00:00`), which sorts and compares correctly. |
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

# Metadata only (no query, no embedding): the 3 newest phones under $800
curl "${H[@]}" -X POST $B/search -d '{"kb_ids": [1], "top_k": 3, "sort_by": "date_added",
  "filters": [{"field": "category", "op": "==", "value": "phones"}, {"field": "price", "op": "<", "value": 800}]}'

# Both: the best matches for a query, newest first
curl "${H[@]}" -X POST $B/search -d '{"query": "phone with a good camera", "kb_ids": [1], "sort_by": "date_added"}'

curl "${H[@]}" $B/kbs/1/metadata-fields    # the fields, their type and example values
```

Each result has `score` (RRF), `vector_rank`, `keyword_rank` (null = not found by that search) and `distance` (cosine distance, 0 = identical, ~1 = unrelated). Use this to tune `max_distance`, your `content_template` and `TEXT_SEARCH_CONFIG`. Results found by metadata alone have none of these (null), and are one per document.

**Filters** are `{"field", "op", "value"}`, all must match. How values compare depends on the value you give:

| value | compares | example |
|---|---|---|
| a number (also `"500"`) | as numbers; metadata text like `"799.00"` counts as a number | `price < 500` |
| a date (`YYYY-MM-DD…`) | as dates, **to the precision you write** | `date_added >= "2025-03"` (from March 2025), `<= "2025-03-01"` (that day included) |
| `true` / `false` | matches true/false, 1/0, "yes"/"no" | `active == true` |
| other text | `==` / `!=` ignore case | `category == "phones"` |

`contains` finds part of a text (ignoring case) or an item of a list (`tags contains "sale"`); `in` is one of a list (`["Phones", "Audio"]` or `"Phones, Audio"`). A missing field never matches, except with `!=`.

**Sorting** (`sort_by`, `sort_order` `desc`/`asc`): numbers as numbers, dates (ISO text) by date, documents without the field last. With a query, the `sort_pool` best matches are sorted (default: as many as `top_k`, so sorting only reorders the best matches; a larger pool lets e.g. newer but less relevant matches in).

### 4. A workflow, and a conversation with it

Three templates are in `workflows/templates/` (they use knowledge base 1):

| template | what it does |
|---|---|
| `rag_chat.json` | search → answer from the results. The basic RAG chat. |
| `product_assistant.json` | categorize the message → product question: rewrite it into a search query using the conversation, search, answer (or a fixed reply when nothing is found); order question: fixed reply; small talk: short answer. |
| `product_agent.json` | the model gets a `search_products` tool and decides what (and how often) to search: by meaning, by metadata (newest, cheapest, a category) or both. |

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
data: {"event": "node_finished", "node": "search", "type": "retrieval", "elapsed_ms": 83, "outputs": {...}, "next": "answer"}
data: {"event": "tool_call", "node": "agent", "tool": "search_products", "arguments": {...}, "query": "...", "results": 5, "elapsed_ms": 40}   (agent nodes)
data: {"event": "node_failed", "node": "answer", "type": "llm", "elapsed_ms": 900, "error": "..."}   (then an error event)
data: {"event": "message", "delta": "The Pixel 9 "}          ← the answer, as it is written
data: {"event": "done", "answer": "...", "references": [{"ref": 1, "title": "Pixel 9", "external_id": "2", "cited": true, ...}], "conversation_id": "…", "message_id": 17}
data: {"event": "error", "message": "..."}                    (instead of done, if something failed)
```

`outputs` are shortened here (long text cut, result lists as `"[5 items]"`); the studio's test runs get them in full, with each node's inputs. `references` are the chunks the answer was based on; `ref` is the number the model cites (`[1]`), `cited` says whether the answer cites it, and `external_id` is your product id.

Conversations: `GET /api/workflows/1/conversations?user_id=user-42`, `GET /api/conversations/<id>?user_id=user-42` (with messages and references), `DELETE` the same URL. A conversation can only be used and read with the `user_id` that started it.

---

## Studio (visual workflow editor)

Open **http://localhost:8000/studio/** and log in with a superuser (`python manage.py createsuperuser`). The studio can do everything the API key can, so other admin accounts (staff without superuser) can't use it.

- **Workflows page**: create a workflow (blank, or from a template, picking the knowledge bases it searches), import/export JSON, duplicate, delete.
- **Canvas**: drag a node type from the left onto the canvas, or click it to add it after the selected node. Connect nodes by dragging from the dot on the right of a node to another node; drop on empty space to add a connected node. Categorize and switch nodes have one dot per category/case (and `else`). Select a connection and press Delete to remove it. Wheel zooms, dragging the background pans, the buttons bottom-left fit the view and arrange the nodes.
- **Settings panel**: every setting of the selected node, with help text: prompts (with a variable picker listing `{sys.*}` and the outputs of the nodes before it, and a large editor), knowledge bases, search settings and metadata filters (field names are suggested from the knowledge base), categories, switch cases and conditions, agent tools (with "Add from the knowledge bases" to list the metadata fields the model may use), begin inputs. **Model** (provider, model, temperature, reasoning effort, max output tokens) and **Advanced** sections hold the settings that used to be built in: the categorize system prompt, the format of search results given to the model, the text when nothing is found, how a tool's query is described to the model. Settings left at their default aren't stored, so the workflow JSON stays short; changed ones get a "Reset" link. Renaming a node updates every `{old@...}` reference.
- **Checks as you edit**: problems are shown on the nodes and in the panel; a workflow with problems can't be saved.
- **Test run** tab: chat with the workflow as it is in the editor, unsaved changes included. Nothing is stored. Nodes light up as they run, and the **run log** opens under the canvas: every node that ran, in order, with what went in (the final prompts and messages sent to the model, the model and its settings, the query, filters and sort a search used, the tools the agent was given, each switch condition and its result) and what came out (every output in full, the results with their metadata), the agent's searches (its arguments, whether it searched by query, metadata or both, and the exact text it got back), errors, and the time each step took. Click a step to open it and select its node; copy or download the log as JSON. Each answer also lists its references.
- **Save** (Ctrl+S): apps using the workflow get the new version right away. If someone else saved it since you opened it, you are asked before overwriting. Undo/redo (Ctrl+Z / Ctrl+Shift+Z) covers every change, "Edit JSON" edits the raw workflow.

It is plain JavaScript modules and CSS in `studio/static/` served by Django (no Node build, no CDN, works offline). It uses the same API as your apps, logged in with the Django session (superusers only; writes need the CSRF token), plus a few endpoints for editors: `GET /api/node-types` (every node type with the JSON schema of its params: the forms are generated from the pydantic models in `workflows/dsl.py`, so a new param shows up in the studio by itself), `GET /api/workflow-templates`, `POST /api/workflows/validate`, `POST /api/workflows/test-run` (streams the run with each node's inputs and full outputs) and `GET /api/kbs/{id}/metadata-fields`. Node positions are stored in the workflow as `"layout"` (ignored when running).

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
| `retrieval` | `kb_ids`, `mode` (`search`, or `metadata`: filters and sort only, no query), `query` (default `{sys.query}`), `filters` (`[{field, op, value}]`), `sort_by`, `sort_order` (`desc`), `top_k` (5), `sort_pool`, `max_distance`, `empty_response`, `chunk_template` | `formalized_content` (numbered results for a prompt), `chunks`, `count`, `query` |
| `llm` | `system_prompt`, `prompt` (default `{sys.query}`), `history` (previous messages to include, 0), model options: `provider`, `model`, `temperature`, `effort`, `max_tokens` | `content` |
| `agent` | like `llm`, plus `tools`: `[{name, description, kb_ids, top_k, metadata_fields, filters, max_distance, sort_pool, metadata_prompt, query_description, filters_description, sort_description, order_description, empty_response, chunk_template}]` and `max_rounds` (4) | `content` |
| `categorize` | `input` (default `{sys.query}`), `history`, `categories`: `[{name, description, examples, next}]`, `system_prompt` (`{categories}` is replaced with the list), model options | `category` |
| `switch` | `cases`: `[{conditions: [{var, op, value}], logic: "and"/"or", next}]`, `default`. Ops: `== != > >= < <= contains not_contains empty not_empty` | `case` |
| `message` | `text` | `content` |

The answer is the text of the `message` nodes that ran. **Streaming:** when an `llm`/`agent` node's `next` is a message node whose text is exactly `{that_node@content}`, its tokens are streamed to the user as they are generated (like RAGFlow). Any other message is sent whole when the node runs. If an agent writes text before a tool call ("Let me look that up"), it is part of the answer too. Loops are allowed (a switch can lead back); `max_steps` (default 50) stops runaway runs.

`chunk_template` is how each search result is written for the model, by default `[{ref}] {content}` and a `({metadata})` line. Placeholders: `{ref}` (the number to cite), `{content}`, `{title}`, `{external_id}`, `{metadata}` (all fields as `name: value`), `{metadata.price}` (one field); a line whose metadata is empty is left out.

**Metadata search in workflows.** A `retrieval` node with `"mode": "metadata"` doesn't search a query: it returns what its filters and sort pick, e.g. the 5 newest products (`"sort_by": "date_added"`). Filter values can be variables, e.g. `{"field": "category", "op": "==", "value": "{begin@category}"}`; a filter whose variable is empty when the node runs is skipped (no category sent: all categories). With `"mode": "search"` the same filters and sort apply to the query's matches.

An `agent` tool with `metadata_fields` (`[{name, type, description}]`) lets the model decide on each call: leave the query empty and filter/sort by these fields ("what's your latest product?" → `sort_by: date_added`), search with a query ("a phone with a good camera"), or both ("the newest phone with a good camera" → query + `sort_by`). The fields, with their descriptions, are listed for the model in the tool description (`metadata_prompt`, where `{fields}` is replaced; `{sys.date}` tells it today's date); a field the model invents gets an error back that lists the right ones. The tool's `filters` are always applied, whatever the model asks (e.g. `active == 1`). Without `metadata_fields` a tool searches by query only, as before. (Workflows saved with the old `metadata_filter: {"brand": "Sony"}` keep working: it is read as `==` filters.)

Retrieval results are numbered across the whole run (`[1]`, `[2]`... continuing over several retrieval nodes and agent searches), so the model's citations always match `references[].ref`.

Adding a node type: a function in `workflows/nodes.py` (`RUNNERS`), its params model in `workflows/dsl.py` (`PARAMS`) and a line in `NODE_INFO` (title, description, outputs) for the studio.

---

## Models

| setting | |
|---|---|
| `LLM_PROVIDER=openai` | OpenAI, or any OpenAI-compatible server: set `OPENAI_BASE_URL` (LiteLLM proxy, vLLM, Ollama...) and `OPENAI_MODEL`. Uses Chat Completions. |
| `LLM_PROVIDER=anthropic` | Claude via the `anthropic` SDK, `ANTHROPIC_MODEL` (default `claude-opus-5-5`). Declined requests are retried on a fallback model server-side (`ANTHROPIC_FALLBACKS`, Claude API only). |
| `EMBEDDING_*` | Always an OpenAI-compatible endpoint (Anthropic has no embeddings API). Defaults to `text-embedding-3-small`, 1536 dimensions. |

A node can override `provider`, `model`, `temperature`, `effort` and `max_tokens`, e.g. a cheaper model for `categorize` and `rewrite`. `temperature` is unset by default because current reasoning models reject it.

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
- Behind an HTTPS reverse proxy, set `CSRF_TRUSTED_ORIGINS` and `BEHIND_HTTPS_PROXY=true` (needed to log in to the studio and the admin).
- `GET /api/status`: index types, counts, queued jobs.
- `/admin/`: browse knowledge bases, documents, chunks, sources (with a "Sync now" action), jobs, workflows and conversations.
- `python manage.py rebuild_search_indexes`: after installing extensions or changing `TEXT_SEARCH_CONFIG`.

## Code, in reading order

1. `knowledge/models.py`: knowledge base, data source, document, chunk (the vector store), sync job.
2. `knowledge/sources.py`: MySQL rows → documents; the sync.
3. `knowledge/ingest.py` + `chunking.py`: documents → chunks → embeddings, skipping what didn't change.
4. `knowledge/search.py` + `search_indexes.py`: the hybrid and metadata search SQL, filters and sorting, and the indexes.
5. `core/llm.py`: one small interface (`stream`, `parse`, `run_tools`) over the OpenAI and Anthropic SDKs.
6. `workflows/dsl.py` → `engine.py` → `nodes.py`: the workflow format, how it becomes a LangGraph graph, what each node does.
7. `knowledge/api.py`, `workflows/api.py`: the HTTP API.
8. `studio/static/studio/js/`: the editor. `editor.js` (the page: changes, undo, save), `canvas.js` (drawing and dragging), `form.js` (forms from the param schemas), `testrun.js` (the test chat) and `runlog.js` (the run log), `graph.js` (pure helpers).

## Tests

```bash
pip install -r requirements-dev.txt
DATABASE_URL=postgresql://postgres:password@localhost:5432/postgres pytest
```

They need a Postgres with the extensions (the compose one works) and no API keys: embeddings and chat models are replaced by fakes, and `tests/test_llm_providers.py` runs the real OpenAI/Anthropic SDK code against small local fake servers.
