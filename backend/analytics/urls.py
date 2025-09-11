from django.urls import path
from . import views

urlpatterns = [
    path('platforms/', views.PlatformListView.as_view(), name='platforms'),
    path('dashboard/', views.dashboard_overview, name='dashboard_overview'),
    path('platform/<str:platform_name>/', views.platform_analytics, name='platform_analytics'),
    path('sync/<str:platform_name>/', views.sync_platform_data, name='sync_platform_data'),
]
