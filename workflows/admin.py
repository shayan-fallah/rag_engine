from django.contrib import admin

from workflows.models import Conversation, Message, Workflow


@admin.register(Workflow)
class WorkflowAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "updated_at")


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ("role", "content", "references", "created_at")
    can_delete = False


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "workflow", "user_id", "updated_at")
    list_filter = ("workflow",)
    search_fields = ("user_id", "title")
    inlines = [MessageInline]
