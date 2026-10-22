from django.core.management.base import BaseCommand

from knowledge.jobs import work


class Command(BaseCommand):
    help = "Run queued data source syncs, and re-sync sources on their schedule."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="Run the queued jobs, then exit")

    def handle(self, *args, once=False, **options):
        self.stdout.write("Worker started")
        work(once=once)
