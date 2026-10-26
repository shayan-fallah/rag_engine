import hmac

from django.conf import settings
from ninja import NinjaAPI
from ninja.errors import HttpError
from ninja.security import APIKeyCookie, HttpBearer
from ninja.utils import check_csrf

from knowledge.api import router as knowledge_router
from workflows.api import router as workflows_router
from workflows.dsl import DSLError


class ApiKey(HttpBearer):
    """Apps: "Authorization: Bearer <API_KEY>"."""

    def authenticate(self, request, token):
        if hmac.compare_digest(token, settings.API_KEY):
            return token


class AdminSession(APIKeyCookie):
    """The studio UI: a logged-in superuser (Django admin login), with the same rights as the API key.

    Writes need the CSRF token, like any Django form.
    """

    param_name = settings.SESSION_COOKIE_NAME

    def __init__(self):
        super().__init__(csrf=False)  # checked below, only for these sessions: API clients get a plain 401

    def authenticate(self, request, key):
        user = getattr(request, "user", None)
        if not (user and user.is_active and user.is_superuser):
            return None
        if check_csrf(request):
            raise HttpError(403, "CSRF check failed; reload the page")
        return user


class OpenAccess:
    """No API_KEY: anyone may call the API (local development only).

    JSON bodies must say they are JSON. Another website open in your browser can send
    text/plain (or untyped) bodies here without asking, but not application/json: that
    needs CORS permission, which this API never gives. Form encodings can't hold JSON.
    """

    FORMS = ("multipart/form-data", "application/x-www-form-urlencoded")

    def __call__(self, request):
        if request.method in ("POST", "PUT", "PATCH") and request.body:
            if request.content_type != "application/json" and request.content_type not in self.FORMS:
                raise HttpError(415, "Send the body as JSON (Content-Type: application/json)")
        return True


def auth():
    return [ApiKey(), AdminSession()] if settings.API_KEY else [OpenAccess()]


api = NinjaAPI(title="RAG Engine", version="1.0", auth=auth())
api.add_router("", knowledge_router, tags=["knowledge"])
api.add_router("", workflows_router, tags=["workflows"])


@api.exception_handler(ValueError)
def value_error(request, exc):
    body = {"detail": str(exc)}
    if isinstance(exc, DSLError):
        body["errors"] = [{"node": node, "message": message} for node, message in exc.errors]
    return api.create_response(request, body, status=400)
