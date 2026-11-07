from django import forms
from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html_join
from django.utils.safestring import mark_safe

from core.models import LLMProfile, workflows_using


class LLMProfileForm(forms.ModelForm):
    """The API key is write-only: never shown again after saving; left empty, the saved key stays."""

    api_key = forms.CharField(
        label="API key",
        required=False,
        # render_value=False (the default): the key is never put back into the page.
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password", "size": 60}),
        help_text="Empty for servers without a key.",
    )
    remove_api_key = forms.BooleanField(label="Remove the saved key", required=False)

    class Meta:
        model = LLMProfile
        fields = ["name", "kind", "base_url", "api_key", "remove_api_key", "default_model"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.api_key:
            self.fields["api_key"].help_text = f"Saved ({self.instance.masked_key}). Leave empty to keep it."

    def clean(self):
        data = super().clean()
        if data.get("remove_api_key"):
            data["api_key"] = ""
        elif not data.get("api_key"):
            data["api_key"] = self.instance.api_key
        return data


@admin.register(LLMProfile)
class LLMProfileAdmin(admin.ModelAdmin):
    form = LLMProfileForm
    list_display = ("name", "kind", "base_url", "default_model", "key", "updated_at")
    readonly_fields = ("used_by", "created_at", "updated_at")
    search_fields = ("name", "base_url", "default_model")

    def get_fields(self, request, obj=None):
        fields = super().get_fields(request, obj)
        return fields if obj and obj.api_key else [f for f in fields if f != "remove_api_key"]

    @admin.display(description="API key")
    def key(self, profile):
        return profile.masked_key or "-"

    @admin.display(description="Used by")
    def used_by(self, profile):
        used = workflows_using(profile.name) if profile.pk else []
        if not used:
            return "No saved workflow"
        return format_html_join(
            mark_safe("<br>"),
            '<a href="{}">{}</a> (nodes {})',
            ((f"{reverse('studio:index')}#/w/{w.id}", w.name, ", ".join(nodes)) for w, nodes in used),
        )

    def get_deleted_objects(self, objs, request):
        """Refuse deleting a profile a saved workflow uses: the admin lists them instead of deleting."""
        deleted, counts, perms_needed, protected = super().get_deleted_objects(objs, request)
        for profile in objs:
            for workflow, nodes in workflows_using(profile.name):
                protected.append(
                    f"Workflow “{workflow.name}” uses “{profile.name}” in nodes {', '.join(nodes)}. "
                    "Switch them to another profile in the studio first."
                )
        return deleted, counts, perms_needed, protected
