"""All settings come from environment variables (or a .env file next to manage.py)."""

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    return env(name, str(default)).lower() in ("1", "true", "yes", "on")


def database_from_url(url: str) -> dict:
    u = urlparse(url)
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": u.path.lstrip("/") or "postgres",
        "USER": unquote(u.username or ""),
        "PASSWORD": unquote(u.password or ""),
        "HOST": u.hostname or "localhost",
        "PORT": str(u.port or 5432),
        "CONN_MAX_AGE": 60,
        "CONN_HEALTH_CHECKS": True,
    }


# --- Django -------------------------------------------------------------------

SECRET_KEY = env("DJANGO_SECRET_KEY", "dev-only-change-me")
DEBUG = env_bool("DJANGO_DEBUG", False)
ALLOWED_HOSTS = [h.strip() for h in env("ALLOWED_HOSTS", "*").split(",") if h.strip()]
# For the admin behind an HTTPS proxy: CSRF_TRUSTED_ORIGINS=https://rag.example.com, BEHIND_HTTPS_PROXY=true
CSRF_TRUSTED_ORIGINS = [o.strip() for o in env("CSRF_TRUSTED_ORIGINS").split(",") if o.strip()]
if env_bool("BEHIND_HTTPS_PROXY"):
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SESSION_COOKIE_SECURE = CSRF_COOKIE_SECURE = True

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.postgres",
    "knowledge",
    "workflows",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

# Postgres is the only database: app tables, vectors and keyword search all live here.
DATABASES = {
    "default": database_from_url(env("DATABASE_URL", "postgresql://postgres:password@localhost:5432/postgres"))
}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
USE_TZ = True
TIME_ZONE = "UTC"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
    "root": {"handlers": ["console"], "level": env("LOG_LEVEL", "INFO")},
    "loggers": {"httpx": {"level": "WARNING"}, "httpx2": {"level": "WARNING"}},
}

# --- API ----------------------------------------------------------------------

# Clients send "Authorization: Bearer <API_KEY>". Empty = no auth (local development only).
API_KEY = env("API_KEY")

# --- Models -------------------------------------------------------------------

# Which provider answers questions: "openai" (any OpenAI-compatible server) or "anthropic".
LLM_PROVIDER = env("LLM_PROVIDER", "openai")
# OPENAI_API_KEY / OPENAI_BASE_URL are read by the openai SDK itself.
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-5.5")
# ANTHROPIC_API_KEY is read by the anthropic SDK itself.
ANTHROPIC_MODEL = env("ANTHROPIC_MODEL", "claude-opus-5-5")
# Retry declined requests on a fallback model (Claude API only; turn off for Bedrock/Vertex/proxies).
ANTHROPIC_FALLBACKS = env_bool("ANTHROPIC_FALLBACKS", True)
# Optional default reasoning effort: low | medium | high. Empty = the model's default.
LLM_EFFORT = env("LLM_EFFORT")
LLM_MAX_TOKENS = int(env("LLM_MAX_TOKENS", "16000"))

# Embeddings always go through an OpenAI-compatible endpoint.
EMBEDDING_MODEL = env("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(env("EMBEDDING_DIMENSIONS", "1536"))  # fixed by the first migration
EMBEDDING_BASE_URL = env("EMBEDDING_BASE_URL") or None  # default: OPENAI_BASE_URL / api.openai.com
EMBEDDING_API_KEY = env("EMBEDDING_API_KEY") or None  # default: OPENAI_API_KEY
EMBEDDING_BATCH_SIZE = int(env("EMBEDDING_BATCH_SIZE", "64"))

# --- Knowledge ----------------------------------------------------------------

# Postgres text search configuration for keyword search. Use "simple" for
# languages Postgres has no stemmer for (e.g. Persian, Arabic, Chinese).
TEXT_SEARCH_CONFIG = env("TEXT_SEARCH_CONFIG", "english")
CHUNK_SIZE = int(env("CHUNK_SIZE", "1500"))  # characters
CHUNK_OVERLAP = int(env("CHUNK_OVERLAP", "150"))  # characters

# --- Worker -------------------------------------------------------------------

WORKER_POLL_SECONDS = float(env("WORKER_POLL_SECONDS", "5"))
