import json

from django.core.management.base import BaseCommand, CommandError

from knowledge.models import DataSource
from knowledge.sources import sync_source


class Command(BaseCommand):
    help = "Sync a data source now, in this process (handy for cron or a first import)."

    def add_arguments(self, parser):
        parser.add_argument("source_id", type=int)

    def handle(self, *args, source_id, **options):
        try:
            source = DataSource.objects.select_related("kb").get(pk=source_id)
        except DataSource.DoesNotExist as exc:
            raise CommandError(f"No data source {source_id}") from exc
        stats = sync_source(source)
        self.stdout.write(json.dumps(stats, indent=2))
