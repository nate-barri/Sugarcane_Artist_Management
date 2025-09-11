"""
Celery tasks for background data synchronization
"""
from celery import shared_task
from django.utils import timezone
from datetime import timedelta
from .models import SocialMediaIntegration
from .services.factory import PlatformServiceFactory
from analytics.models import AnalyticsData, Platform

@shared_task
def sync_platform_data(integration_id):
    """Sync data for a specific integration"""
    try:
        integration = SocialMediaIntegration.objects.get(id=integration_id)
        service = PlatformServiceFactory.create_service(integration)
        
        # Get analytics data for the last 30 days
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=30)
        
        analytics_data = service.get_analytics_data(
            start_date.isoformat(),
            end_date.isoformat()
        )
        
        # Save to database
        platform = Platform.objects.get(name=integration.platform)
        
        analytics_record, created = AnalyticsData.objects.get_or_create(
            user=integration.user,
            platform=platform,
            date=end_date,
            defaults={
                'followers': analytics_data.get('followers', 0),
                'views': analytics_data.get('views', 0),
                'likes': analytics_data.get('likes', 0),
                'comments': analytics_data.get('comments', 0),
                'shares': analytics_data.get('shares', 0),
                'engagement_rate': analytics_data.get('engagement_rate', 0.0),
                'platform_specific_data': analytics_data
            }
        )
        
        # Update integration last sync time
        integration.last_sync = timezone.now()
        integration.save()
        
        return f"Successfully synced data for {integration.platform}"
        
    except Exception as e:
        return f"Error syncing data: {str(e)}"

@shared_task
def sync_all_integrations():
    """Sync data for all active integrations"""
    integrations = SocialMediaIntegration.objects.filter(
        status='connected',
        auto_sync=True
    )
    
    results = []
    for integration in integrations:
        result = sync_platform_data.delay(integration.id)
        results.append(f"Queued sync for {integration.platform}")
    
    return results

@shared_task
def refresh_expired_tokens():
    """Refresh expired access tokens"""
    # Find integrations with tokens expiring in the next hour
    expiry_threshold = timezone.now() + timedelta(hours=1)
    
    integrations = SocialMediaIntegration.objects.filter(
        status='connected',
        token_expires_at__lt=expiry_threshold
    )
    
    results = []
    for integration in integrations:
        try:
            service = PlatformServiceFactory.create_service(integration)
            service.refresh_access_token()
            results.append(f"Refreshed token for {integration.platform}")
        except Exception as e:
            results.append(f"Failed to refresh token for {integration.platform}: {str(e)}")
    
    return results
