import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import UUID

from django.conf import settings
from django.db import transaction
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from ninja import Field, Router, Schema
from ninja.errors import HttpError

from workflows.dsl import SYS_VAR_INFO, DSLError, node_types, validate_dsl
from workflows.engine import run_draft, run_workflow
from workflows.models import Conversation, Message, Workflow

log = logging.getLogger(__name__)
router = Router()

HISTORY_LIMIT = 50  # previous messages loaded for a turn; nodes pick how many they use
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# --- Workflows ------------------------------------------------------------------


class WorkflowIn(Schema):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    dsl: dict


class WorkflowUpdate(WorkflowIn):
    base_updated_at: datetime | None = Field(
        None, description="The updated_at you loaded. If the workflow changed since, the update fails with 409."
    )


class WorkflowOut(Schema):
    id: int
    name: str
    description: str
    dsl: dict
    created_at: datetime
    updated_at: datetime


# --- For editors (the studio) -------------------------------------------------------


@router.get("/node-types")
def get_node_types(request):
    """Node types with the JSON schema of their params, the {sys.*} variables, and the model defaults."""
    return {
        "types": node_types(),
        "sys_vars": SYS_VAR_INFO,
        "model_defaults": {
            "provider": settings.LLM_PROVIDER,
            "openai_model": settings.OPENAI_MODEL,
            "anthropic_model": settings.ANTHROPIC_MODEL,
            "effort": settings.LLM_EFFORT,
            "max_tokens": settings.LLM_MAX_TOKENS,
        },
    }


@router.get("/workflow-templates")
def list_templates(request):
    """Example workflows to start from (workflows/templates/*.json)."""
    templates = []
    for path in sorted(TEMPLATES_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        templates.append({"key": path.stem, **data})
    return templates


class ValidateIn(Schema):
    dsl: dict


@router.post("/workflows/validate")
def validate_workflow(request, data: ValidateIn):
    """Check a workflow without saving it. errors[].node is the node id (null: the whole workflow)."""
    try:
        validate_dsl(data.dsl)
    except DSLError as exc:
        return {"valid": False, "errors": [{"node": node, "message": message} for node, message in exc.errors]}
    return {"valid": True, "errors": []}


class HistoryMessage(Schema):
    role: Literal["user", "assistant"]
    content: str


class TestRunIn(Schema):
    dsl: dict
    query: str = Field(..., min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=HISTORY_LIMIT)
    inputs: dict = Field(default_factory=dict)


@router.post("/workflows/test-run")
def draft_run(request, data: TestRunIn):
    """Run a workflow that isn't saved, e.g. the editor's current version. Nothing is stored.

    Always streamed (Server-Sent Events, like POST /workflows/{id}/run). The client keeps the
    conversation and sends it back as `history`.
    """
    user = getattr(request, "user", None)
    user_id = f"studio:{user.get_username()}" if user and user.is_authenticated else "studio"
    history = [m.dict() for m in data.history]
    events = run_draft(data.dsl, data.query, user_id, history, data.inputs)  # raises DSLError (400) first
    return event_stream(guarded(events, "draft"))


@router.post("/workflows", response={201: WorkflowOut})
def create_workflow(request, data: WorkflowIn):
    validate_dsl(data.dsl)
    if Workflow.objects.filter(name=data.name).exists():
        raise HttpError(409, f"A workflow named {data.name!r} exists")
    return 201, Workflow.objects.create(**data.dict())


@router.get("/workflows", response=list[WorkflowOut])
def list_workflows(request):
    return Workflow.objects.order_by("id")


@router.get("/workflows/{workflow_id}", response=WorkflowOut)
def get_workflow(request, workflow_id: int):
    return get_object_or_404(Workflow, pk=workflow_id)


@router.put("/workflows/{workflow_id}", response=WorkflowOut)
def update_workflow(request, workflow_id: int, data: WorkflowUpdate):
    validate_dsl(data.dsl)
    with transaction.atomic():  # the row lock makes "changed since you loaded it" reliable
        workflow = get_object_or_404(Workflow.objects.select_for_update(), pk=workflow_id)
        if data.base_updated_at:
            base = data.base_updated_at
            if timezone.is_naive(base):
                base = timezone.make_aware(base, UTC)
            # JSON times have milliseconds only, the database has microseconds.
            if abs(base - workflow.updated_at) >= timedelta(milliseconds=1):
                raise HttpError(409, "The workflow was changed by someone else since you loaded it")
        if Workflow.objects.filter(name=data.name).exclude(pk=workflow_id).exists():
            raise HttpError(409, f"A workflow named {data.name!r} exists")
        workflow.name, workflow.description, workflow.dsl = data.name, data.description, data.dsl
        workflow.save()
    return workflow


@router.delete("/workflows/{workflow_id}", response={204: None})
def delete_workflow(request, workflow_id: int):
    get_object_or_404(Workflow, pk=workflow_id).delete()
    return 204, None


# --- Running --------------------------------------------------------------------


class RunIn(Schema):
    query: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1, description="Your app's user id")
    conversation_id: UUID | None = Field(None, description="Omit to start a new conversation")
    inputs: dict = Field(default_factory=dict, description="Values for the begin node's inputs")
    stream: bool = True


@router.post("/workflows/{workflow_id}/run")
def run(request, workflow_id: int, data: RunIn):
    """Send a user message to a workflow and get the answer.

    stream=true (default): Server-Sent Events, one `data: {json}` line per event, ending with `data: [DONE]`.
    Events: conversation, node_started, node_finished, node_failed, tool_call, message {delta},
    done {answer, references}, error.
    stream=false: one JSON object {conversation_id, message_id, answer, references}.
    """
    workflow = get_object_or_404(Workflow, pk=workflow_id)
    if data.conversation_id:
        conversation = get_object_or_404(Conversation, pk=data.conversation_id, workflow=workflow, user_id=data.user_id)
    else:
        conversation = Conversation.objects.create(workflow=workflow, user_id=data.user_id, title=data.query[:200])

    recent = list(conversation.messages.order_by("-id")[:HISTORY_LIMIT])[::-1]
    history = [{"role": m.role, "content": m.content} for m in recent]
    Message.objects.create(conversation=conversation, role=Message.USER, content=data.query)
    events = turn(workflow, conversation, data.query, history, data.inputs)

    if data.stream:
        return event_stream(events)

    result = {}
    for event in events:
        if event["event"] == "error":
            raise HttpError(500, event["message"])
        if event["event"] == "done":
            result = event
    return {k: v for k, v in result.items() if k != "event"}


def turn(workflow, conversation, query, history, inputs) -> Iterator[dict]:
    """Run the workflow and save the answer to the conversation."""
    yield {"event": "conversation", "conversation_id": str(conversation.id)}
    events = run_workflow(workflow, query, conversation.user_id, str(conversation.id), history, inputs)
    for event in guarded(events, workflow.pk):
        if event["event"] == "done":
            message = Message.objects.create(
                conversation=conversation,
                role=Message.ASSISTANT,
                content=event["answer"],
                references=event["references"],
            )
            conversation.save(update_fields=["updated_at"])
            event = {**event, "conversation_id": str(conversation.id), "message_id": message.id}
        yield event


def guarded(events: Iterator[dict], label) -> Iterator[dict]:
    """A failure in the middle of a run becomes an "error" event (the response has already started)."""
    try:
        yield from events
    except Exception as exc:
        log.exception("workflow %s failed", label)
        yield {"event": "error", "message": f"{type(exc).__name__}: {exc}"}


def sse(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    yield "data: [DONE]\n\n"


def event_stream(events: Iterator[dict]) -> StreamingHttpResponse:
    response = StreamingHttpResponse(sse(events), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # tell nginx not to buffer the stream
    return response


# --- Conversations ----------------------------------------------------------------


class MessageOut(Schema):
    id: int
    role: str
    content: str
    references: list
    created_at: datetime


class ConversationOut(Schema):
    id: UUID
    workflow_id: int
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]

    @staticmethod
    def resolve_messages(obj):
        return list(obj.messages.all())


@router.get("/workflows/{workflow_id}/conversations", response=list[ConversationOut])
def list_conversations(request, workflow_id: int, user_id: str, limit: int = 50):
    """A user's conversations with a workflow, newest first."""
    return Conversation.objects.filter(workflow_id=workflow_id, user_id=user_id)[: min(max(limit, 1), 200)]


@router.get("/conversations/{conversation_id}", response=ConversationDetail)
def get_conversation(request, conversation_id: UUID, user_id: str):
    return get_object_or_404(Conversation, pk=conversation_id, user_id=user_id)


@router.delete("/conversations/{conversation_id}", response={204: None})
def delete_conversation(request, conversation_id: UUID, user_id: str):
    get_object_or_404(Conversation, pk=conversation_id, user_id=user_id).delete()
    return 204, None
