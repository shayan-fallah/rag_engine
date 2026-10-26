"""The workflow format: a JSON graph of nodes.

    {
      "nodes": {
        "begin":  {"type": "begin", "next": "search"},
        "search": {"type": "retrieval", "params": {"kb_ids": [1]}, "next": "answer"},
        "answer": {"type": "llm", "params": {"system_prompt": "Answer from:\\n{search@formalized_content}"}, "next": "reply"},
        "reply":  {"type": "message", "params": {"text": "{answer@content}"}}
      }
    }

Text params are templates. Like RAGFlow, {sys.query} is the user's message and
{node_id@output} is an output of an earlier node. A node without "next" ends the
run. "categorize" and "switch" nodes pick the next node themselves.

Node types and their outputs are listed in NODE_INFO below. Every param is
described by its pydantic model; the studio editor builds its forms from them
(node_types()), so a new param shows up there by itself. json_schema_extra
hints how to show it: widget (prompt, template, kbs, node, kv, value, model,
textarea, field: a metadata field name), group (model, metadata, advanced:
collapsed sections), placeholders, variables (false: no variable picker), labels
(for choices), fill_from_kbs and hide_when ({param: [values]} or {param: "empty"}).

"layout" (optional) holds node positions for the editor and is not used to run.
"""

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from knowledge.search import CHUNK_TEMPLATE, FILTER_OPS, check_filter

SYS_VAR_INFO = {
    "query": "The user's message",
    "user_id": "Your app's user id",
    "conversation_id": "The conversation's id",
    "date": "The current date and time (UTC)",
    "history": 'The conversation before this message, as "role: text" lines',
    "turns": "How many messages the user sent before this one",
}
SYS_VARS = tuple(SYS_VAR_INFO)
VAR_PATTERN = re.compile(r"\{(sys\.[a-z_]+|[A-Za-z0-9_\-]+@[A-Za-z0-9_]+)\}")
NAME = r"^[A-Za-z][A-Za-z0-9_\-]*$"

CATEGORIZE_PROMPT = """You sort the user's message into exactly one of these categories.

{categories}

Answer with the name of the best matching category."""

CHUNK_PLACEHOLDERS = ["{ref}", "{content}", "{title}", "{external_id}", "{metadata}", "{metadata.NAME}"]
CHUNK_HELP = (
    "How each result is written: {ref} is its number (cite it as [1]), {metadata} all metadata as "
    '"name: value", {metadata.NAME} one field. A line with only metadata, all empty, is left out. '
    "Results are separated by a blank line."
)

FILTER_HELP = (
    "Numbers compare as numbers. Dates compare to the precision you write: 2025-03 is the whole month. "
    "Text: == ignores case. contains: part of the text, or an item of a list. in: one of a list, e.g. "
    "Phones, Audio. A variable that is empty when the node runs skips the filter."
)

SORT_POOL_HELP = (
    "How many of the best matches are sorted by the field; the first Results of them are returned. "
    "Empty: as many as Results, so sorting only reorders the best matches. Larger: e.g. newer but less "
    "relevant matches can make it in."
)

METADATA_PROMPT = """You can also filter and sort by these metadata fields:
{fields}

Leave the query empty to find results by metadata alone, e.g. the newest ones: sort_by a date field with order "desc". \
Add a query to search by meaning and keywords among the results the filters allow; with sort_by, the best matches \
are then sorted by that field. When the request names something a field holds (a category, a brand), filter by that \
field rather than only putting it in the query. Today is {sys.date}."""


def ui(widget: str | None = None, group: str | None = None, **extra) -> dict:
    """Editor hints, stored in the JSON schema of a param."""
    return {k: v for k, v in {"widget": widget, "group": group, **extra}.items() if v is not None}


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelOptions(Params):
    provider: Literal["openai", "anthropic"] | None = Field(
        None, title="Provider", description="Empty: LLM_PROVIDER of the server", json_schema_extra=ui(group="model")
    )
    model: str | None = Field(
        None,
        title="Model",
        description="Empty: OPENAI_MODEL or ANTHROPIC_MODEL of the server",
        json_schema_extra=ui("model", group="model"),
    )
    temperature: float | None = Field(
        None,
        ge=0,
        le=2,
        title="Temperature",
        description="Empty: the model's default (reasoning models only accept their default)",
        json_schema_extra=ui(group="model"),
    )
    effort: Literal["low", "medium", "high"] | None = Field(
        None,
        title="Reasoning effort",
        description="Empty: LLM_EFFORT of the server, or the model's default",
        json_schema_extra=ui(group="model"),
    )
    max_tokens: int | None = Field(
        None,
        ge=1,
        title="Max output tokens",
        description="Empty: LLM_MAX_TOKENS for Claude, the model's limit for OpenAI-compatible models "
        "(sent to them as max_completion_tokens)",
        json_schema_extra=ui(group="model"),
    )


FilterOp = Literal[FILTER_OPS]


class MetadataFilter(Params):
    field: str = Field(
        min_length=1,
        max_length=100,
        title="Field",
        description="A metadata field, e.g. category or date_added",
        json_schema_extra=ui("field"),
    )
    op: FilterOp = Field("==", title="Operator")
    value: Any = Field(
        None,
        title="Value",
        description=FILTER_HELP,
        json_schema_extra=ui("value", placeholder="e.g. Phones, 500, 2025-03 or {begin@category}"),
    )

    @model_validator(mode="after")
    def check_value(self):
        if isinstance(self.value, str) and VAR_PATTERN.search(self.value):
            return self  # filled in when the node runs
        if self.value is None or self.value == "" or self.value == []:
            raise ValueError("enter a value")
        check_filter(self.model_dump())
        return self


class MetadataField(Params):
    name: str = Field(min_length=1, max_length=100, title="Field", json_schema_extra=ui("field"))
    type: Literal["text", "number", "date", "boolean", "list"] = Field(
        "text", title="Type", description="Tells the model how to write values"
    )
    description: str = Field(
        "", title="Description", description="What it means, for the model, e.g. when the product was added"
    )


def legacy_metadata_filter(data):
    """Workflows saved before "filters" have metadata_filter: {"field": value}. Becomes == filters."""
    if isinstance(data, dict) and isinstance(data.get("metadata_filter"), dict | None) and "metadata_filter" in data:
        data = dict(data)
        old = data.pop("metadata_filter") or {}
        data["filters"] = [{"field": k, "op": "==", "value": v} for k, v in old.items()] + list(
            data.get("filters") or []
        )
    return data


class BeginParams(Params):
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        title="Inputs",
        description='Values your app can send with each run ("inputs" in the run request) and their defaults. '
        "Use them as {begin@name}.",
        json_schema_extra=ui("kv"),
    )


class RetrievalParams(Params):
    kb_ids: list[int] = Field(min_length=1, title="Knowledge bases", json_schema_extra=ui("kbs"))
    mode: Literal["search", "metadata"] = Field(
        "search",
        title="Find by",
        description="Metadata only: no query and no embedding, e.g. the 5 newest products (sort by date_added) "
        "or every product of a brand.",
        json_schema_extra=ui(
            labels={"search": "Query: meaning and keywords", "metadata": "Metadata only: filters and sort"}
        ),
    )
    query: str = Field(
        "{sys.query}",
        title="Search query",
        description="What to search for: usually {sys.query}, or a rewritten question like {rewrite@content}",
        json_schema_extra=ui("template", hide_when={"mode": ["metadata"]}),
    )
    filters: list[MetadataFilter] = Field(
        default_factory=list,
        title="Metadata filters",
        description="Only results whose metadata matches all of these, e.g. category == Phones or "
        "date_added >= 2025-01-01",
    )
    sort_by: str | None = Field(
        None,
        max_length=100,
        title="Sort by",
        description="A metadata field, e.g. date_added or price. Empty: the best matches first "
        "(Metadata only: in the order the documents were added).",
        json_schema_extra=ui("field", placeholder="None"),
    )
    sort_order: Literal["desc", "asc"] = Field(
        "desc",
        title="Order",
        json_schema_extra=ui(
            labels={"desc": "Highest / newest first", "asc": "Lowest / oldest first"}, hide_when={"sort_by": "empty"}
        ),
    )
    top_k: int = Field(5, ge=1, le=50, title="Results", description="How many results to return")
    sort_pool: int | None = Field(
        None,
        ge=1,
        le=200,
        title="Matches to sort",
        description=SORT_POOL_HELP,
        json_schema_extra=ui(hide_when={"mode": ["metadata"], "sort_by": "empty"}, placeholder="Same as Results"),
    )
    max_distance: float | None = Field(
        None,
        ge=0,
        le=2,
        title="Max vector distance",
        description="Drop vector matches farther than this cosine distance (0 = same meaning, 2 = opposite). "
        "Keyword matches are kept. Empty: no limit.",
        json_schema_extra=ui(hide_when={"mode": ["metadata"]}),
    )
    empty_response: str = Field(
        "No relevant information was found in the knowledge base.",
        title="Text when nothing is found",
        description="formalized_content when the search finds nothing",
        json_schema_extra=ui("prompt", group="advanced"),
    )
    chunk_template: str = Field(
        CHUNK_TEMPLATE,
        title="Result format",
        description=CHUNK_HELP,
        json_schema_extra=ui("prompt", group="advanced", placeholders=CHUNK_PLACEHOLDERS, variables=False),
    )

    _legacy = model_validator(mode="before")(legacy_metadata_filter)

    @field_validator("sort_by")
    @classmethod
    def blank_is_none(cls, value):
        return (value or "").strip() or None


class LLMParams(ModelOptions):
    system_prompt: str = Field(
        "",
        title="System prompt",
        description="Instructions for the model. Put retrieved knowledge here, e.g. {search@formalized_content}.",
        json_schema_extra=ui("prompt"),
    )
    prompt: str = Field(
        "{sys.query}",
        title="User message",
        description="The message the model answers (after the conversation history)",
        json_schema_extra=ui("prompt"),
    )
    history: int = Field(
        0,
        ge=0,
        le=50,
        title="History messages",
        description="How many previous messages of the conversation to include (0 = none)",
    )


class SearchTool(Params):
    name: str = Field(
        pattern=r"^[a-zA-Z0-9_]{1,64}$", title="Tool name", description="Letters, digits and _. The model sees it."
    )
    description: str = Field(
        title="Description",
        description="Tells the model what the tool searches and when to use it",
        json_schema_extra=ui("textarea"),
    )
    kb_ids: list[int] = Field(min_length=1, title="Knowledge bases", json_schema_extra=ui("kbs"))
    top_k: int = Field(5, ge=1, le=50, title="Results", description="How many results each call returns")
    metadata_fields: list[MetadataField] = Field(
        default_factory=list,
        title="Fields the model can filter and sort by",
        description="With fields, the model decides per call: search by a query, filter and sort by these fields, "
        "or both (e.g. the newest products, or the newest about a topic). Empty: it searches by query only.",
        json_schema_extra=ui(fill_from_kbs=True),
    )
    filters: list[MetadataFilter] = Field(
        default_factory=list,
        title="Fixed filters",
        description="Always applied, whatever the model asks, e.g. active == 1",
    )
    max_distance: float | None = Field(None, ge=0, le=2, title="Max vector distance", description="Empty: no limit")
    sort_pool: int | None = Field(
        None,
        ge=1,
        le=200,
        title="Matches to sort",
        description="When the model searches with a query and sorts. " + SORT_POOL_HELP,
        json_schema_extra=ui(group="metadata", hide_when={"metadata_fields": "empty"}, placeholder="Same as Results"),
    )
    metadata_prompt: str = Field(
        METADATA_PROMPT,
        title="Metadata instructions",
        description="Added to the tool description when there are fields. {fields} is replaced with the fields, "
        'as "- name (type): description" lines.',
        json_schema_extra=ui(
            "prompt", group="metadata", placeholders=["{fields}"], hide_when={"metadata_fields": "empty"}
        ),
    )
    filters_description: str = Field(
        "Conditions on the metadata fields; all must match. value: text, a number, true/false, or a date as "
        'YYYY-MM-DD (YYYY-MM is the whole month). Use a list of values with "in".',
        title="Filters argument",
        description="How the filters argument is described to the model",
        json_schema_extra=ui("textarea", group="metadata", hide_when={"metadata_fields": "empty"}),
    )
    sort_description: str = Field(
        "A metadata field to sort the results by, e.g. a date field for the newest or oldest",
        title="Sort argument",
        description="How the sort_by argument is described to the model",
        json_schema_extra=ui("textarea", group="metadata", hide_when={"metadata_fields": "empty"}),
    )
    order_description: str = Field(
        "desc: newest or largest first (default), asc: oldest or smallest first",
        title="Order argument",
        description="How the order argument is described to the model",
        json_schema_extra=ui("textarea", group="metadata", hide_when={"metadata_fields": "empty"}),
    )
    query_description: str = Field(
        "What to search for",
        title="Query description",
        description="How the tool's query argument is described to the model",
        json_schema_extra=ui(group="advanced"),
    )
    empty_response: str = Field(
        "No results.",
        title="Text when nothing is found",
        description="The tool result the model gets when a search finds nothing",
        json_schema_extra=ui(group="advanced"),
    )
    chunk_template: str = Field(
        CHUNK_TEMPLATE,
        title="Result format",
        description=CHUNK_HELP,
        json_schema_extra=ui("prompt", group="advanced", placeholders=CHUNK_PLACEHOLDERS, variables=False),
    )

    _legacy = model_validator(mode="before")(legacy_metadata_filter)

    @field_validator("metadata_fields")
    @classmethod
    def unique_fields(cls, fields):
        names = [f.name for f in fields]
        if len(names) != len(set(names)):
            raise ValueError("field names must be unique")
        return fields


class AgentParams(LLMParams):
    tools: list[SearchTool] = Field(
        min_length=1, title="Search tools", description="Searches the model can call, as often as it needs"
    )
    max_rounds: int = Field(
        4,
        ge=1,
        le=10,
        title="Max tool rounds",
        description="After this many rounds of tool calls the model must answer",
    )

    @field_validator("tools")
    @classmethod
    def unique_names(cls, tools):
        names = [t.name for t in tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        return tools


class Category(Params):
    name: str = Field(min_length=1, title="Name", description="The model answers with this name")
    description: str = Field("", title="Description", description="When to choose it", json_schema_extra=ui("textarea"))
    examples: list[str] = Field(default_factory=list, title="Examples", description="Example messages")
    next: str | None = Field(None, title="Go to", json_schema_extra=ui("node"))


class CategorizeParams(ModelOptions):
    input: str = Field("{sys.query}", title="Text to classify", json_schema_extra=ui("template"))
    history: int = Field(
        0,
        ge=0,
        le=50,
        title="History messages",
        description='Previous messages the model sees, to classify follow-ups like "and the cheaper one?"',
    )
    categories: list[Category] = Field(min_length=2, title="Categories")
    system_prompt: str = Field(
        CATEGORIZE_PROMPT,
        title="System prompt",
        description="{categories} is replaced with the categories, their descriptions and examples",
        json_schema_extra=ui("prompt", group="advanced", placeholders=["{categories}"]),
    )

    @field_validator("categories")
    @classmethod
    def unique_names(cls, categories):
        names = [c.name for c in categories]
        if len(names) != len(set(names)):
            raise ValueError("category names must be unique")
        return categories


Operator = Literal["==", "!=", ">", ">=", "<", "<=", "contains", "not_contains", "empty", "not_empty"]


class Condition(Params):
    var: str = Field(title="Variable", description="e.g. {search@count}", json_schema_extra=ui("template"))
    op: Operator = Field(title="Operator")
    value: Any = Field(
        None,
        title="Value",
        description="A number, text or a variable",
        json_schema_extra=ui("value", hide_when={"op": ["empty", "not_empty"]}),
    )


class Case(Params):
    conditions: list[Condition] = Field(min_length=1, title="Conditions")
    logic: Literal["and", "or"] = Field("and", title="Match", description="and: every condition, or: any condition")
    next: str | None = Field(None, title="Go to", json_schema_extra=ui("node"))


class SwitchParams(Params):
    cases: list[Case] = Field(
        min_length=1, title="Cases", description="Checked in order: the first case that matches picks the next node"
    )
    default: str | None = Field(
        None, title="Else go to", description="When no case matches. Empty: end the run.", json_schema_extra=ui("node")
    )


class MessageParams(Params):
    text: str = Field(
        title="Text",
        description="Shown to the user. Right after an LLM or agent node, {node@content} streams its answer.",
        json_schema_extra=ui("prompt"),
    )


PARAMS: dict[str, type[Params]] = {
    "begin": BeginParams,
    "retrieval": RetrievalParams,
    "llm": LLMParams,
    "agent": AgentParams,
    "categorize": CategorizeParams,
    "switch": SwitchParams,
    "message": MessageParams,
}
ROUTERS = ("categorize", "switch")

# What each node type is, its outputs ({node_id@output}) and the params a new node starts with.
NODE_INFO: dict[str, dict] = {
    "begin": {
        "title": "Begin",
        "description": "Where every run starts. Optionally defines inputs your app sends with each run.",
        "outputs": {},  # one per input
        "starter": {},
    },
    "retrieval": {
        "title": "Retrieval",
        "description": "Searches knowledge bases by meaning and keywords, or by metadata alone (e.g. the newest "
        "products), and writes the results as numbered text.",
        "outputs": {
            "formalized_content": "The results as numbered text for a prompt, or the text when nothing is found",
            "chunks": "The results as a list",
            "count": "How many results were found",
            "query": "The query that was searched (empty with Metadata only)",
        },
        "starter": {"kb_ids": []},
    },
    "llm": {
        "title": "LLM",
        "description": "One call to a chat model.",
        "outputs": {"content": "The model's answer"},
        "starter": {"system_prompt": "You are a helpful assistant.", "history": 6},
    },
    "agent": {
        "title": "Agent",
        "description": "A chat model with search tools: it decides what to search (by query, by metadata, or both), "
        "can search several times, then answers.",
        "outputs": {"content": "The model's answer"},
        "starter": {
            "system_prompt": "Use the search tool to find information before you answer, and cite the numbers of "
            "the results you use, like [1].",
            "history": 6,
            "tools": [
                {
                    "name": "search",
                    "description": "Search the knowledge base. Returns numbered results.",
                    "kb_ids": [],
                }
            ],
        },
    },
    "categorize": {
        "title": "Categorize",
        "description": "A chat model sorts the message into a category; each category goes to its own next node.",
        "outputs": {"category": "The name of the chosen category"},
        "starter": {
            "categories": [
                {"name": "question", "description": "A question about our products or services"},
                {"name": "other", "description": "Anything else"},
            ]
        },
    },
    "switch": {
        "title": "Switch",
        "description": "If / else on variables, without a model. The first case that matches picks the next node.",
        "outputs": {"case": 'The number of the case that matched (0, 1, ...) or "default"'},
        "starter": {"cases": [{"conditions": [{"var": "{sys.query}", "op": "not_empty"}]}]},
    },
    "message": {
        "title": "Message",
        "description": "Text shown to the user. Show an LLM's answer with {node@content}.",
        "outputs": {"content": "The text that was shown"},
        "starter": {"text": "Hello! How can I help you?"},
    },
}


def node_types() -> list[dict]:
    """Every node type with its params' JSON schema, for the studio editor."""
    return [
        {"type": t, **NODE_INFO[t], "router": t in ROUTERS, "schema": PARAMS[t].model_json_schema()} for t in PARAMS
    ]


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["begin", "retrieval", "llm", "agent", "categorize", "switch", "message"]
    params: dict = Field(default_factory=dict)
    next: str | None = None
    _parsed: Any = PrivateAttr(None)  # typed params, set by validate_dsl

    @property
    def parsed(self) -> Any:
        return self._parsed

    def targets(self) -> list[str | None]:
        """Every node this one can lead to. None = end of the run."""
        if self.type == "categorize":
            return [c.next for c in self.parsed.categories]
        if self.type == "switch":
            return [c.next for c in self.parsed.cases] + [self.parsed.default]
        return [self.next]


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, Node] = Field(min_length=2)
    max_steps: int = Field(50, ge=5, le=500, description="Stops loops that never end")
    layout: dict[str, Position] = Field(default_factory=dict, description="Node positions in the studio editor")

    @property
    def begin_id(self) -> str:
        return next(i for i, n in self.nodes.items() if n.type == "begin")


class DSLError(ValueError):
    """An invalid workflow. `errors` lists (node id or None, message)."""

    def __init__(self, errors: list[tuple[str | None, str]]):
        self.errors = errors
        super().__init__("; ".join(f"{node}: {message}" if node else message for node, message in errors))


def validate_dsl(dsl: dict, check_kbs: bool = True) -> WorkflowSpec:
    """Parse and check a workflow. Raises DSLError (a ValueError) with readable messages."""
    try:
        spec = WorkflowSpec.model_validate(dsl)
    except ValidationError as exc:
        raise DSLError(_errors(exc)) from None

    errors: list[tuple[str | None, str]] = []
    for node_id, node in spec.nodes.items():
        if not re.match(NAME, node_id):
            errors.append((node_id, "node ids may contain letters, digits, _ and - only"))
        try:
            node._parsed = PARAMS[node.type].model_validate(node.params)
        except ValidationError as exc:
            errors.extend(_errors(exc, node_id))
            continue
        if node.type in ROUTERS and node.next:
            errors.append((node_id, f"a {node.type} node sets the next node per category/case, not with 'next'"))
        for target in node.targets():
            if target is not None and target not in spec.nodes:
                errors.append((node_id, f"next node {target!r} does not exist"))
        for ref in _references(node.params):
            if ref.startswith("sys."):
                if ref[4:] not in SYS_VARS:
                    known = ", ".join("sys." + v for v in SYS_VARS)
                    errors.append((node_id, f"unknown variable {{{ref}}}; use one of {known}"))
            elif ref.split("@")[0] not in spec.nodes:
                errors.append((node_id, f"{{{ref}}} refers to a node that does not exist"))

    begins = [i for i, n in spec.nodes.items() if n.type == "begin"]
    if len(begins) != 1:
        errors.append((None, f"a workflow needs exactly one begin node (found {len(begins)})"))

    if check_kbs and not errors:
        from knowledge.models import KnowledgeBase

        used: dict[int, list[str]] = {}
        for node_id, node in spec.nodes.items():
            if node.type == "retrieval":
                ids = node.parsed.kb_ids
            elif node.type == "agent":
                ids = [i for t in node.parsed.tools for i in t.kb_ids]
            else:
                continue
            for kb_id in ids:
                used.setdefault(kb_id, []).append(node_id)
        existing = set(KnowledgeBase.objects.filter(id__in=used).values_list("id", flat=True))
        for kb_id in sorted(set(used) - existing):
            for node_id in dict.fromkeys(used[kb_id]):
                errors.append((node_id, f"knowledge base(s) [{kb_id}] do not exist"))

    if errors:
        raise DSLError(errors)
    return spec


def _references(value) -> list[str]:
    if isinstance(value, str):
        return VAR_PATTERN.findall(value)
    if isinstance(value, dict):
        return [r for v in value.values() for r in _references(v)]
    if isinstance(value, list):
        return [r for v in value for r in _references(v)]
    return []


def _errors(exc: ValidationError, node_id: str | None = None) -> list[tuple[str | None, str]]:
    """Pydantic errors as (node id, "field: message")."""
    found = []
    for e in exc.errors():
        loc = [str(p) for p in e["loc"]]
        node = node_id
        if node is None and len(loc) >= 2 and loc[0] == "nodes":
            node, loc = loc[1], loc[2:]
        message = e["msg"].removeprefix("Value error, ")
        found.append((node, f"{'.'.join(loc)}: {message}" if loc else message))
    return found
