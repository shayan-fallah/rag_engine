from django.core.management.base import BaseCommand
from django.db import transaction

from knowledge.search_indexes import create_search_indexes, drop_search_indexes


class Command(BaseCommand):
    help = (
        "Drop and re-create the vector and keyword indexes. Run it after installing "
        "pgvectorscale / pg_textsearch, or after changing TEXT_SEARCH_CONFIG."
    )

    def handle(self, *args, **options):
        with transaction.atomic():
            drop_search_indexes()
            result = create_search_indexes()
        self.stdout.write(f"Created: {result}")
