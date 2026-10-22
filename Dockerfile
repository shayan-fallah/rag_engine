FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN DJANGO_SECRET_KEY=build python manage.py collectstatic --noinput -v 0

EXPOSE 8000
# Threads, not processes: a streamed answer holds a thread while it waits on the model.
CMD ["gunicorn", "config.wsgi:application", "-k", "gthread", "--workers", "2", "--threads", "16", "--timeout", "300", "-b", "0.0.0.0:8000"]
