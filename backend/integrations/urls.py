from django.urls import path
from . import views

urlpatterns = [
    path('', views.IntegrationListView.as_view(), name='integrations_list'),
    path('status/', views.integration_status, name='integration_status'),
    path('connect/<str:platform>/', views.connect_platform, name='connect_platform'),
    path('disconnect/<str:platform>/', views.disconnect_platform, name='disconnect_platform'),
]
