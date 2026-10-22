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

Node types and their outputs:
  begin       inputs from the request ("inputs") over the defaults in params.inputs  -> {begin@<name>}
  retrieval   hybrid search in knowledge bases       -> formalized_content, chunks, count
  llm         one model call                         -> content
  agent       model + search tools, in a loop        -> content
  categorize  model picks a category                 -> category
  switch      if/else on variables, no model         -> case
  message     text shown to the user                 -> content
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
)

SYS_VARS = ("query", "user_id", "conversation_id", "date", "history", "turns")
VAR_PATTERN = re.compile(r"\{(sys\.[a-z_]+|[A-Za-z0-9_\-]+@[A-Za-z0-9_]+)\}")
NAME = r"^[A-Za-z][A-Za-z0-9_\-]*$"


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelOptions(Params):
    provider: Literal["openai", "anthropic"] | None = None  # default: LLM_PROVIDER
    model: str | None = None  # default: OPENAI_MODEL / ANTHROPIC_MODEL
    temperature: float | None = None
    effort: Literal["low", "medium", "high"] | None = None


class BeginParams(Params):
    inputs: dict[str, Any] = Field(default_factory=dict, description="Input names and their default values")


class RetrievalParams(Params):
    kb_ids: list[int] = Field(min_length=1)
    query: str = "{sys.query}"
    top_k: int = Field(5, ge=1, le=50)
    metadata_filter: dict | None = None
    max_distance: float | None = Field(None, ge=0, le=2)
    empty_response: str = "No relevant information was found in the knowledge base."


class LLMParams(ModelOptions):
    system_prompt: str = ""
    prompt: str = "{sys.query}"
    history: int = Field(0, ge=0, le=50, description="How many previous messages of the conversation to include")


class SearchTool(Params):
    name: str = Field(pattern=r"^[a-zA-Z0-9_]{1,64}$")
    description: str
    kb_ids: list[int] = Field(min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    metadata_filter: dict | None = None
    max_distance: float | None = Field(None, ge=0, le=2)


class AgentParams(LLMParams):
    tools: list[SearchTool] = Field(min_length=1)
    max_rounds: int = Field(4, ge=1, le=10, description="Maximum rounds of tool calls")

    @field_validator("tools")
    @classmethod
    def unique_names(cls, tools):
        names = [t.name for t in tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        return tools


class Category(Params):
    name: str = Field(min_length=1)
    description: str = ""
    examples: list[str] = Field(default_factory=list)
    next: str | None = None


class CategorizeParams(ModelOptions):
    input: str = "{sys.query}"
    history: int = Field(0, ge=0, le=50)
    categories: list[Category] = Field(min_length=2)

    @field_validator("categories")
    @classmethod
    def unique_names(cls, categories):
        names = [c.name for c in categories]
        if len(names) != len(set(names)):
            raise ValueError("category names must be unique")
        return categories


Operator = Literal["==", "!=", ">", ">=", "<", "<=", "contains", "not_contains", "empty", "not_empty"]


class Condition(Params):
    var: str = Field(description="A template, e.g. {search@count}")
    op: Operator
    value: Any = None


class Case(Params):
    conditions: list[Condition] = Field(min_length=1)
    logic: Literal["and", "or"] = "and"
    next: str | None = None


class SwitchParams(Params):
    cases: list[Case] = Field(min_length=1)
    default: str | None = None


class MessageParams(Params):
    text: str


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


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, Node] = Field(min_length=2)
    max_steps: int = Field(50, ge=5, le=500, description="Stops loops that never end")

    @property
    def begin_id(self) -> str:
        return next(i for i, n in self.nodes.items() if n.type == "begin")


def validate_dsl(dsl: dict, check_kbs: bool = True) -> WorkflowSpec:
    """Parse and check a workflow. Raises ValueError with a readable message."""
    try:
        spec = WorkflowSpec.model_validate(dsl)
    except ValidationError as exc:
        raise ValueError(_errors(exc)) from None

    errors = []
    for node_id, node in spec.nodes.items():
        if not re.match(NAME, node_id):
            errors.append(f"{node_id}: node ids may contain letters, digits, _ and - only")
        try:
            node._parsed = PARAMS[node.type].model_validate(node.params)
        except ValidationError as exc:
            errors.append(f"{node_id}: {_errors(exc)}")
            continue
        if node.type in ROUTERS and node.next:
            errors.append(f"{node_id}: a {node.type} node sets the next node per category/case, not with 'next'")
        for target in node.targets():
            if target is not None and target not in spec.nodes:
                errors.append(f"{node_id}: next node {target!r} does not exist")
        for ref in _references(node.params):
            if ref.startswith("sys."):
                if ref[4:] not in SYS_VARS:
                    errors.append(
                        f"{node_id}: unknown variable {{{ref}}}; use one of {', '.join('sys.' + v for v in SYS_VARS)}"
                    )
            elif ref.split("@")[0] not in spec.nodes:
                errors.append(f"{node_id}: {{{ref}}} refers to a node that does not exist")

    begins = [i for i, n in spec.nodes.items() if n.type == "begin"]
    if len(begins) != 1:
        errors.append(f"a workflow needs exactly one begin node (found {len(begins)})")

    if check_kbs and not errors:
        from knowledge.models import KnowledgeBase

        kb_ids = set()
        for node in spec.nodes.values():
            if node.type == "retrieval":
                kb_ids.update(node.parsed.kb_ids)
            elif node.type == "agent":
                kb_ids.update(i for t in node.parsed.tools for i in t.kb_ids)
        missing = kb_ids - set(KnowledgeBase.objects.filter(id__in=kb_ids).values_list("id", flat=True))
        if missing:
            errors.append(f"knowledge base(s) {sorted(missing)} do not exist")

    if errors:
        raise ValueError("; ".join(errors))
    return spec


def _references(value) -> list[str]:
    if isinstance(value, str):
        return VAR_PATTERN.findall(value)
    if isinstance(value, dict):
        return [r for v in value.values() for r in _references(v)]
    if isinstance(value, list):
        return [r for v in value for r in _references(v)]
    return []


def _errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'workflow'}: {e['msg']}" for e in exc.errors())
