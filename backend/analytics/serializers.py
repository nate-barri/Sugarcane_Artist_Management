from rest_framework import serializers
from .models import Platform, AnalyticsData, DashboardMetrics

class PlatformSerializer(serializers.ModelSerializer):
    class Meta:
        model = Platform
        fields = '__all__'

class AnalyticsDataSerializer(serializers.ModelSerializer):
    platform_name = serializers.CharField(source='platform.name', read_only=True)
    platform_display_name = serializers.CharField(source='platform.display_name', read_only=True)
    
    class Meta:
        model = AnalyticsData
        fields = '__all__'
        read_only_fields = ('user', 'created_at', 'updated_at')

class DashboardMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = DashboardMetrics
        fields = '__all__'
        read_only_fields = ('user', 'created_at', 'updated_at')

class DashboardOverviewSerializer(serializers.Serializer):
    """Serializer for dashboard overview data"""
    total_subscribers = serializers.IntegerField()
    total_views = serializers.IntegerField()
    total_watch_time = serializers.FloatField()
    total_spotify_streams = serializers.IntegerField()
    audience_growth = serializers.FloatField()
    top_performing_platform = serializers.CharField()
    
    # Chart data
    engagement_chart_data = serializers.JSONField()
    spotify_streams_chart = serializers.JSONField()
    impressions_reach_chart = serializers.JSONField()
    audience_retention_chart = serializers.JSONField()
    views_overtime_chart = serializers.JSONField()
