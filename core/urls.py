# core/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),  # Landing page with login/signup
    path('dashboard/', views.dashboard, name='dashboard'),  # Dashboard after login
    path('signup/', views.signup, name='signup'),  # Sign up page
]