"""Fill {sys.query} and {node_id@output} in templates from the run state."""

import json

from workflows.dsl import VAR_PATTERN


def lookup(ref: str, state: dict):
    """The raw value of one variable (e.g. "search@count" -> 3). Unknown -> None."""
    if ref == "sys.history":
        return history_text(state["sys"]["history"])
    if ref.startswith("sys."):
        return state["sys"].get(ref[4:])
    node_id, _, key = ref.partition("@")
    return state["outputs"].get(node_id, {}).get(key)


def to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def render(template: str, state: dict) -> str:
    return VAR_PATTERN.sub(lambda m: to_text(lookup(m.group(1), state)), template)


def resolve(template: str, state: dict):
    """Like render, but a template that is a single variable keeps its type (for switch conditions)."""
    match = VAR_PATTERN.fullmatch(template.strip())
    return lookup(match.group(1), state) if match else render(template, state)


def history_text(history: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)
