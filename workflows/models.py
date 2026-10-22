import uuid

from django.core.exceptions import ValidationError
from django.db import models


class Workflow(models.Model):
    """A workflow (RAGFlow calls it an agent): a graph of nodes stored as JSON. See workflows/dsl.py."""

    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    dsl = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def clean(self):
        from workflows.dsl import validate_dsl

        try:
            validate_dsl(self.dsl)
        except ValueError as exc:
            raise ValidationError({"dsl": str(exc)}) from exc


class Conversation(models.Model):
    """One chat between a user of your app and a workflow. Its messages are the short-term memory."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow = models.ForeignKey(Workflow, on_delete=models.CASCADE, related_name="conversations")
    user_id = models.CharField(max_length=255, db_index=True, help_text="Your app's user id")
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title or str(self.id)


class Message(models.Model):
    USER, ASSISTANT = "user", "assistant"

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=[(USER, USER), (ASSISTANT, ASSISTANT)])
    content = models.TextField()
    references = models.JSONField(default=list, blank=True, help_text="Chunks the answer was based on")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]
