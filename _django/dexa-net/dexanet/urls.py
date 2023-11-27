from django.contrib import admin
from django.urls import include, path
from django.conf import settings

admin.site.site_header = settings.ADMIN_SITE_HEADER

urlpatterns = [
    path("polls/", include("polls.urls")),
    path("admin/", admin.site.urls),
]
