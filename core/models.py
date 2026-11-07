"""LLM profiles: model servers that workflow nodes can use, set up in the admin."""

from django.core.exceptions import ValidationError
from django.db import models, transaction
from django.db.models import ProtectedError

# Where a profile without a base URL goes. Never OPENAI_BASE_URL / ANTHROPIC_BASE_URL from .env:
# the profile's key must not be sent to the .env server.
DEFAULT_BASE_URLS = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com"}


def http_url(value: str) -> None:
    if not value.startswith(("http://", "https://")):
        raise ValidationError("Enter a URL starting with http:// or https://")


class LLMProfileQuerySet(models.QuerySet):
    def delete(self):
        for profile in self:
            profile.check_unused()
        return super().delete()


class LLMProfile(models.Model):
    """A model server: OpenAI or any OpenAI-compatible server (vLLM, Ollama, LiteLLM...), or Anthropic.

    A node uses it with "profile": "<name>"; nodes without a profile use the .env settings.
    A profile a saved workflow uses can't be deleted (the admin lists those workflows instead).
    """

    KINDS = [("openai", "OpenAI-compatible"), ("anthropic", "Anthropic")]

    name = models.CharField(max_length=100, unique=True, help_text="Workflows refer to the profile by this name")
    kind = models.CharField(max_length=20, choices=KINDS, default="openai")
    base_url = models.CharField(
        "Base URL",
        max_length=500,
        blank=True,
        validators=[http_url],
        help_text="e.g. http://localhost:8000/v1. Empty: api.openai.com or api.anthropic.com",
    )
    api_key = models.TextField("API key", blank=True)
    default_model = models.CharField(max_length=200, blank=True, help_text="For nodes that don't name a model")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = LLMProfileQuerySet.as_manager()

    class Meta:
        ordering = ["name"]
        verbose_name = "LLM profile"

    def __str__(self):
        return self.name

    @property
    def masked_key(self) -> str:
        """The key as the admin shows it: only the end of a long key."""
        if not self.api_key:
            return ""
        return f"…{self.api_key[-4:]}" if len(self.api_key) >= 16 else "set"

    def save(self, *args, **kwargs):
        old_name = LLMProfile.objects.filter(pk=self.pk).values_list("name", flat=True).first() if self.pk else None
        with transaction.atomic():
            super().save(*args, **kwargs)
            if old_name and old_name != self.name:
                # A rename carries over to the workflows, so they don't break.
                for workflow, node_ids in workflows_using(old_name):
                    for node_id in node_ids:
                        workflow.dsl["nodes"][node_id]["params"]["profile"] = self.name
                    workflow.save(update_fields=["dsl", "updated_at"])

    def delete(self, *args, **kwargs):
        self.check_unused()
        return super().delete(*args, **kwargs)

    def check_unused(self) -> None:
        used = workflows_using(self.name)
        if used:
            names = ", ".join(repr(w.name) for w, _ in used)
            raise ProtectedError(f"LLM profile {self.name!r} is used by the workflow(s) {names}", {w for w, _ in used})


def workflows_using(name: str) -> list[tuple]:
    """Saved workflows with nodes that use the profile `name`: [(workflow, [node ids])]."""
    from workflows.models import Workflow  # imported here: the workflows app imports core

    used = []
    for workflow in Workflow.objects.order_by("name"):
        nodes = workflow.dsl.get("nodes") or {}
        node_ids = [i for i, n in nodes.items() if (n.get("params") or {}).get("profile") == name]
        if node_ids:
            used.append((workflow, node_ids))
    return used
