from django.urls import path

from studio import views

app_name = "studio"
urlpatterns = [path("", views.index, name="index")]
