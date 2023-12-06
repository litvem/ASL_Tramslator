from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name = "home"),
    path("register/", views.register_user, name = "register"),
    path("login/", views.login_user, name = "login"),
    path("logout/", views.logout_user, name="logout"),
    path("training/", views.training, name = "training"),
    path("live/", views.live, name="live"),
    path("translations/", views.translations, name="translations"),
    path("explanation/", views.project_explanation, name = "explanation"),
]