"""Vector + keyword indexes on the chunk table: DiskANN + BM25 when the
extensions are installed, HNSW + built-in full-text search otherwise.
See knowledge/search_indexes.py."""

from django.db import migrations

from knowledge.search_indexes import create_search_indexes, drop_search_indexes


def forwards(apps, schema_editor):
    create_search_indexes(schema_editor)


def backwards(apps, schema_editor):
    drop_search_indexes(schema_editor)


class Migration(migrations.Migration):
    dependencies = [("knowledge", "0001_initial")]
    operations = [migrations.RunPython(forwards, backwards)]
