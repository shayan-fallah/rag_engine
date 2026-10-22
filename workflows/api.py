import json
import logging
from collections.abc import Iterator
from datetime import datetime
from uuid import UUID

from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from ninja import Field, Router, Schema
from ninja.errors import HttpError

from workflows.dsl import validate_dsl
from workflows.engine import run_workflow
from workflows.models import Conversation, Message, Workflow

log = logging.getLogger(__name__)
router = Router()

HISTORY_LIMIT = 50  # previous messages loaded for a turn; nodes pick how many they use


# --- Workflows ------------------------------------------------------------------


class WorkflowIn(Schema):
    name: str
    description: str = ""
    dsl: dict


class WorkflowOut(Schema):
    id: int
    name: str
    description: str
    dsl: dict
    created_at: datetime
    updated_at: datetime


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
def update_workflow(request, workflow_id: int, data: WorkflowIn):
    workflow = get_object_or_404(Workflow, pk=workflow_id)
    validate_dsl(data.dsl)
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
    Events: conversation, node_started, node_finished, tool_call, message {delta}, done {answer, references}, error.
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
        response = StreamingHttpResponse(sse(events), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"  # tell nginx not to buffer the stream
        return response

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
    try:
        for event in run_workflow(workflow, query, conversation.user_id, str(conversation.id), history, inputs):
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
    except Exception as exc:
        log.exception("workflow %s failed", workflow.pk)
        yield {"event": "error", "message": f"{type(exc).__name__}: {exc}"}


def sse(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    yield "data: [DONE]\n\n"


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
