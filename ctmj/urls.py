"""URL configuration for the CJPS project.

The root URLconf is named ``ctmj.urls`` (the historical project name) so
existing deployments and WSGI entry points keep working.
"""

from django.contrib import admin
from django.urls import path

from ctmj import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home_view, name="home"),
    path("predict/", views.predict_view, name="predict"),
    path("health/", views.health_view, name="health"),
]

# Only consulted when DEBUG is False; under DEBUG Django renders its own
# technical 404 page.
handler404 = "ctmj.views.page_not_found"
