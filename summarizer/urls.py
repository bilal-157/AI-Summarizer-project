# summarizer/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),  # ← ADD THIS
    path('', include('core.urls')),  # Your app URLs
]