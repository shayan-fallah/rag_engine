import hmac

from django.conf import settings
from ninja import NinjaAPI
from ninja.security import HttpBearer

from knowledge.api import router as knowledge_router
from workflows.api import router as workflows_router


class ApiKey(HttpBearer):
    def authenticate(self, request, token):
        if hmac.compare_digest(token, settings.API_KEY):
            return token


def auth():
    # No API_KEY configured: open API (local development only).
    return [ApiKey()] if settings.API_KEY else None


api = NinjaAPI(title="RAG Engine", version="1.0", auth=auth())
api.add_router("", knowledge_router, tags=["knowledge"])
api.add_router("", workflows_router, tags=["workflows"])


@api.exception_handler(ValueError)
def value_error(request, exc):
    return api.create_response(request, {"detail": str(exc)}, status=400)
