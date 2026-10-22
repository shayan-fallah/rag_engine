"""A tiny job queue in Postgres (no Redis/Celery): the sync_job table + SELECT ... FOR UPDATE SKIP LOCKED.

`python manage.py worker` loops: queue sources that are due, claim one job, run it.
Run as many workers as you like; each job runs once.
"""

import logging
import time
from datetime import timedelta

from django.conf import settings
from django.db import close_old_connections, connection, transaction
from django.utils import timezone

from knowledge.models import DataSource, SyncJob
from knowledge.sources import SYNC_LOCK, sync_source

log = logging.getLogger(__name__)

ACTIVE = (SyncJob.QUEUED, SyncJob.RUNNING)


def enqueue_sync(source: DataSource) -> SyncJob:
    """Queue a sync, unless one is already queued or running."""
    with transaction.atomic():
        DataSource.objects.select_for_update().get(pk=source.pk)  # serialize concurrent calls
        job = SyncJob.objects.filter(source=source, status__in=ACTIVE).first()
        return job or SyncJob.objects.create(source=source)


def enqueue_due_sources() -> None:
    """Queue sources whose last sync *attempt* is older than their interval.

    Counting failed attempts too means a broken source is retried once per
    interval, not in a tight loop.
    """
    now = timezone.now()
    for source in DataSource.objects.filter(sync_interval_minutes__gt=0):
        latest_job = source.jobs.order_by("-id").first()
        attempts = [t for t in (source.last_synced_at, latest_job and latest_job.created_at) if t]
        due = not attempts or max(attempts) + timedelta(minutes=source.sync_interval_minutes) <= now
        if due and not source.jobs.filter(status__in=ACTIVE).exists():
            enqueue_sync(source)


def fail_abandoned_jobs() -> None:
    """A "running" job whose source lock is free belongs to a worker that died."""
    for job in SyncJob.objects.filter(status=SyncJob.RUNNING):
        with connection.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s, %s)", [SYNC_LOCK, job.source_id])
            if not cur.fetchone()[0]:
                continue  # a live worker holds it
            cur.execute("SELECT pg_advisory_unlock(%s, %s)", [SYNC_LOCK, job.source_id])
        # Re-check: the worker may have finished between the query and the lock.
        SyncJob.objects.filter(
            pk=job.pk, status=SyncJob.RUNNING, started_at__lt=timezone.now() - timedelta(minutes=1)
        ).update(status=SyncJob.FAILED, error="The worker running this job stopped", finished_at=timezone.now())


def claim_job() -> SyncJob | None:
    with transaction.atomic():
        job = SyncJob.objects.select_for_update(skip_locked=True).filter(status=SyncJob.QUEUED).order_by("id").first()
        if job:
            job.status, job.started_at = SyncJob.RUNNING, timezone.now()
            job.save(update_fields=["status", "started_at"])
        return job


def run_job(job: SyncJob) -> None:
    log.info("sync job %s: source %s started", job.id, job.source_id)
    try:
        job.stats = sync_source(job.source, job=job)
        job.status = SyncJob.DONE
    except Exception as exc:
        log.exception("sync job %s failed", job.id)
        job.status, job.error = SyncJob.FAILED, f"{type(exc).__name__}: {exc}"
    job.finished_at = timezone.now()
    # update(), not save(): the job is gone if its source was deleted during the sync.
    SyncJob.objects.filter(pk=job.pk).update(
        status=job.status, stats=job.stats, error=job.error, finished_at=job.finished_at
    )
    log.info("sync job %s: %s %s", job.id, job.status, job.stats)


def work(once: bool = False) -> None:
    while True:
        close_old_connections()  # survive database restarts
        try:
            fail_abandoned_jobs()
            enqueue_due_sources()
            job = claim_job()
        except Exception:
            log.exception("worker loop error")
            job = None
        if job:
            try:
                run_job(job)
            except Exception:  # e.g. the database went away mid-job: keep the worker alive
                log.exception("sync job %s: could not record the result", job.id)
            continue
        if once:
            return
        time.sleep(settings.WORKER_POLL_SECONDS)
