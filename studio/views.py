"""The studio: a visual workflow editor in the browser (static JS, no build step).

It talks to the normal API (/api/...) with the user's session. That gives full API
rights, so only superusers can use it: python manage.py createsuperuser.
"""

from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import ensure_csrf_cookie

from config.api import api


@user_passes_test(lambda user: user.is_active and user.is_superuser, login_url="admin:login")
@ensure_csrf_cookie
def index(request):
    config = {
        "apiRoot": reverse(f"{api.urls_namespace}:api-root"),
        "user": request.user.get_username(),
        "adminUrl": reverse("admin:index"),
        "logoutUrl": reverse("admin:logout"),
        "docsUrl": reverse(f"{api.urls_namespace}:openapi-view"),
    }
    return render(request, "studio/index.html", {"config": config})
