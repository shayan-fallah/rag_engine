"""Text -> vector, through any OpenAI-compatible embeddings endpoint."""

import logging
import time
from functools import lru_cache

from django.conf import settings
from openai import OpenAI

log = logging.getLogger(__name__)


@lru_cache
def _client() -> OpenAI:
    # Falls back to OPENAI_API_KEY / OPENAI_BASE_URL when the EMBEDDING_* ones are empty.
    return OpenAI(api_key=settings.EMBEDDING_API_KEY, base_url=settings.EMBEDDING_BASE_URL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts, a batch per API call."""
    start = time.monotonic()
    vectors: list[list[float]] = []
    size = settings.EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), size):
        response = _client().embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts[i : i + size],
            encoding_format="float",  # some OpenAI-compatible servers don't support base64
        )
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))

    if vectors and len(vectors[0]) != settings.EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"{settings.EMBEDDING_MODEL} returned {len(vectors[0])} dimensions, but "
            f"EMBEDDING_DIMENSIONS is {settings.EMBEDDING_DIMENSIONS}"
        )
    log.info("embedded %d text(s) in %.2fs", len(texts), time.monotonic() - start)
    return vectors


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
