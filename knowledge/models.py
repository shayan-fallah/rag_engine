from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.db import models
from pgvector.django import VectorField


class KnowledgeBase(models.Model):
    """A searchable collection of documents (RAGFlow calls this a dataset).

    pgvectorscale filters by "labels" (smallint[]) inside the vector index, and
    the knowledge base id is used as the label, so ids must stay below 32768.
    """

    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    chunk_size = models.PositiveIntegerField(default=settings.CHUNK_SIZE, help_text="characters")
    chunk_overlap = models.PositiveIntegerField(default=settings.CHUNK_OVERLAP, help_text="characters")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class DataSource(models.Model):
    """A SQL table (or query) whose rows become documents, e.g. your MySQL products table.

    Each row becomes one document:
      - external_id = row[id_column]          (must be unique and sortable)
      - title       = row[title_column]       (optional)
      - content     = content_template filled with the row, e.g. "{name}\\n{description}".
                      Empty template: every column as "column: value" lines.
      - metadata    = {column: row[column] for column in metadata_columns}
    """

    kb = models.ForeignKey(KnowledgeBase, on_delete=models.CASCADE, related_name="sources")
    name = models.CharField(max_length=200)
    url = models.CharField(
        max_length=500,
        help_text="mysql://user:pass@host:3306/db or postgresql://..., or env:VAR_NAME to read it from an environment variable",
    )
    query = models.TextField(help_text="SELECT ... FROM products WHERE ...")
    id_column = models.CharField(max_length=100, default="id")
    title_column = models.CharField(max_length=100, blank=True)
    content_template = models.TextField(blank=True)
    metadata_columns = models.JSONField(default=list, blank=True)
    sync_interval_minutes = models.PositiveIntegerField(
        default=0, help_text="The worker re-syncs this often. 0 = only when asked."
    )
    last_synced_at = models.DateTimeField(null=True, blank=True)
    last_sync_stats = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Document(models.Model):
    kb = models.ForeignKey(KnowledgeBase, on_delete=models.CASCADE, related_name="documents")
    source = models.ForeignKey(DataSource, on_delete=models.CASCADE, null=True, blank=True, related_name="documents")
    external_id = models.CharField(max_length=255, help_text="Your id, e.g. the product id")
    title = models.CharField(max_length=500, blank=True)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    content_hash = models.CharField(max_length=64, help_text="Unchanged hash = no re-embedding")
    chunk_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["kb", "source", "external_id"], nulls_distinct=False, name="document_unique_external_id"
            )
        ]

    def __str__(self):
        return self.title or self.external_id


class Chunk(models.Model):
    """A piece of a document with its embedding. This table is the vector store."""

    kb = models.ForeignKey(KnowledgeBase, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    position = models.PositiveIntegerField()
    content = models.TextField(help_text="Title + chunk text: what is embedded, keyword-indexed and shown to the LLM")
    metadata = models.JSONField(default=dict, blank=True, help_text="Copy of the document metadata, for filtering")
    labels = ArrayField(models.SmallIntegerField(), default=list, help_text="[kb id], for filtered DiskANN search")
    embedding = VectorField(dimensions=settings.EMBEDDING_DIMENSIONS)

    class Meta:
        ordering = ["document_id", "position"]


class SyncJob(models.Model):
    """A queued data source sync. The worker (manage.py worker) picks these up."""

    QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
    STATUSES = [(s, s) for s in (QUEUED, RUNNING, DONE, FAILED)]

    source = models.ForeignKey(DataSource, on_delete=models.CASCADE, related_name="jobs")
    status = models.CharField(max_length=10, choices=STATUSES, default=QUEUED, db_index=True)
    stats = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-id"]
