from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

from config.api import api

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="studio:index")),
    path("admin/", admin.site.urls),
    path("api/", api.urls),
    path("studio/", include("studio.urls")),
]
