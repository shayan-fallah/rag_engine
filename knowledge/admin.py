from django.contrib import admin

from knowledge.jobs import enqueue_sync
from knowledge.models import Chunk, DataSource, Document, KnowledgeBase, SyncJob


@admin.register(KnowledgeBase)
class KnowledgeBaseAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "chunk_size", "chunk_overlap", "created_at")


@admin.register(DataSource)
class DataSourceAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "kb", "sync_interval_minutes", "last_synced_at")
    readonly_fields = ("last_synced_at", "last_sync_stats")
    actions = ["queue_sync"]

    @admin.action(description="Sync now (queued for the worker)")
    def queue_sync(self, request, queryset):
        for source in queryset:
            enqueue_sync(source)
        self.message_user(request, f"Queued {queryset.count()} sync(s)")


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "external_id", "title", "kb", "source", "chunk_count", "updated_at")
    list_filter = ("kb", "source")
    search_fields = ("external_id", "title")
    readonly_fields = ("content_hash", "chunk_count", "created_at", "updated_at")


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "position", "kb")
    list_filter = ("kb",)
    exclude = ("embedding",)
    readonly_fields = ("kb", "document", "position", "content", "metadata", "labels")


@admin.register(SyncJob)
class SyncJobAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "status", "created_at", "started_at", "finished_at")
    list_filter = ("status",)
    readonly_fields = ("source", "status", "stats", "error", "created_at", "started_at", "finished_at")
