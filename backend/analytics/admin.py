from django.contrib import admin
from .models import Platform, AnalyticsData, DashboardMetrics

@admin.register(Platform)
class PlatformAdmin(admin.ModelAdmin):
    list_display = ('name', 'display_name', 'is_active')
    list_filter = ('is_active',)

@admin.register(AnalyticsData)
class AnalyticsDataAdmin(admin.ModelAdmin):
    list_display = ('user', 'platform', 'date', 'followers', 'views', 'engagement_rate')
    list_filter = ('platform', 'date')
    search_fields = ('user__email', 'platform__name')

@admin.register(DashboardMetrics)
class DashboardMetricsAdmin(admin.ModelAdmin):
    list_display = ('user', 'date', 'total_followers', 'total_views', 'top_performing_platform')
    list_filter = ('date', 'top_performing_platform')
    search_fields = ('user__email',)
