from django.urls import path
from . import views

app_name = 'replica'

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('prediction/', views.prediction, name='prediction'),
]