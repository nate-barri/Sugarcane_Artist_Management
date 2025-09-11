from django.urls import path
from . import views

urlpatterns = [
    path('', views.ReportListCreateView.as_view(), name='reports_list'),
    path('<int:pk>/', views.ReportDetailView.as_view(), name='report_detail'),
    path('generate/', views.generate_report, name='generate_report'),
    path('download/<int:report_id>/', views.download_report, name='download_report'),
    path('templates/', views.ReportTemplateListView.as_view(), name='report_templates'),
]
