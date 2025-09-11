from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.db.models import Sum, Avg, Q
from django.utils import timezone
from datetime import datetime, timedelta
from .models import Platform, AnalyticsData, DashboardMetrics
from .serializers import (
    PlatformSerializer, AnalyticsDataSerializer, 
    DashboardMetricsSerializer, DashboardOverviewSerializer
)

class PlatformListView(generics.ListAPIView):
    queryset = Platform.objects.filter(is_active=True)
    serializer_class = PlatformSerializer
    permission_classes = [IsAuthenticated]

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_overview(request):
    """Get dashboard overview data"""
    user = request.user
    today = timezone.now().date()
    thirty_days_ago = today - timedelta(days=30)
    
    # Get latest metrics
    latest_metrics = DashboardMetrics.objects.filter(user=user).first()
    
    if not latest_metrics:
        # Create sample data if none exists
        latest_metrics = DashboardMetrics.objects.create(
            user=user,
            date=today,
            total_followers=125000,
            total_views=2500000,
            total_engagement=85000,
            audience_growth_rate=12.5,
            top_performing_platform='Spotify',
            watch_time_hours=15000,
            spotify_streams=1800000
        )
    
    # Generate chart data (sample data for now)
    chart_data = {
        'engagement_chart_data': [
            {'name': 'YouTube', 'value': 35},
            {'name': 'Spotify', 'value': 40},
            {'name': 'Instagram', 'value': 15},
            {'name': 'TikTok', 'value': 10}
        ],
        'spotify_streams_chart': [
            {'date': '2024-01-01', 'streams': 45000},
            {'date': '2024-01-02', 'streams': 52000},
            {'date': '2024-01-03', 'streams': 48000},
            {'date': '2024-01-04', 'streams': 61000},
            {'date': '2024-01-05', 'streams': 55000},
        ],
        'impressions_reach_chart': [
            {'date': '2024-01-01', 'impressions': 125000, 'reach': 98000},
            {'date': '2024-01-02', 'impressions': 142000, 'reach': 112000},
            {'date': '2024-01-03', 'impressions': 138000, 'reach': 105000},
        ],
        'audience_retention_chart': [
            {'time': '0s', 'retention': 100},
            {'time': '30s', 'retention': 85},
            {'time': '60s', 'retention': 72},
            {'time': '90s', 'retention': 65},
        ],
        'views_overtime_chart': [
            {'date': '2024-01-01', 'views': 25000},
            {'date': '2024-01-02', 'views': 32000},
            {'date': '2024-01-03', 'views': 28000},
            {'date': '2024-01-04', 'views': 41000},
        ]
    }
    
    overview_data = {
        'total_subscribers': latest_metrics.total_followers,
        'total_views': latest_metrics.total_views,
        'total_watch_time': latest_metrics.watch_time_hours,
        'total_spotify_streams': latest_metrics.spotify_streams,
        'audience_growth': latest_metrics.audience_growth_rate,
        'top_performing_platform': latest_metrics.top_performing_platform,
        **chart_data
    }
    
    serializer = DashboardOverviewSerializer(overview_data)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def platform_analytics(request, platform_name):
    """Get analytics for a specific platform"""
    try:
        platform = Platform.objects.get(name=platform_name)
        analytics = AnalyticsData.objects.filter(
            user=request.user, 
            platform=platform
        ).order_by('-date')[:30]
        
        serializer = AnalyticsDataSerializer(analytics, many=True)
        return Response(serializer.data)
    except Platform.DoesNotExist:
        return Response({'error': 'Platform not found'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def sync_platform_data(request, platform_name):
    """Sync data from external platform APIs"""
    try:
        platform = Platform.objects.get(name=platform_name)
        
        # Here you would integrate with actual platform APIs
        # For now, we'll create sample data
        today = timezone.now().date()
        
        analytics_data, created = AnalyticsData.objects.get_or_create(
            user=request.user,
            platform=platform,
            date=today,
            defaults={
                'followers': 50000,
                'views': 125000,
                'likes': 8500,
                'comments': 1200,
                'shares': 450,
                'engagement_rate': 8.2,
                'platform_specific_data': {
                    'subscriber_growth': 2.5,
                    'watch_time_minutes': 45000,
                    'top_video_views': 25000
                }
            }
        )
        
        serializer = AnalyticsDataSerializer(analytics_data)
        return Response({
            'message': f'Successfully synced {platform.display_name} data',
            'data': serializer.data
        })
        
    except Platform.DoesNotExist:
        return Response({'error': 'Platform not found'}, status=status.HTTP_404_NOT_FOUND)
