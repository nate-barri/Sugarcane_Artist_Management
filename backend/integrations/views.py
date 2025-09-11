from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.utils import timezone
from .models import SocialMediaIntegration, APIKey
from .serializers import SocialMediaIntegrationSerializer, APIKeySerializer

class IntegrationListView(generics.ListAPIView):
    serializer_class = SocialMediaIntegrationSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return SocialMediaIntegration.objects.filter(user=self.request.user)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def connect_platform(request, platform):
    """Connect a social media platform"""
    valid_platforms = ['youtube', 'spotify', 'instagram', 'tiktok', 'facebook', 'twitter']
    
    if platform not in valid_platforms:
        return Response({'error': 'Invalid platform'}, status=status.HTTP_400_BAD_REQUEST)
    
    # In a real implementation, this would handle OAuth flow
    # For now, we'll create a mock connection
    integration, created = SocialMediaIntegration.objects.get_or_create(
        user=request.user,
        platform=platform,
        defaults={
            'status': 'connected',
            'platform_username': f'user_{platform}',
            'platform_user_id': f'{platform}_123456',
            'last_sync': timezone.now(),
        }
    )
    
    if not created:
        integration.status = 'connected'
        integration.last_sync = timezone.now()
        integration.save()
    
    serializer = SocialMediaIntegrationSerializer(integration)
    return Response({
        'message': f'Successfully connected to {platform.title()}',
        'integration': serializer.data
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def disconnect_platform(request, platform):
    """Disconnect a social media platform"""
    try:
        integration = SocialMediaIntegration.objects.get(
            user=request.user,
            platform=platform
        )
        integration.status = 'disconnected'
        integration.access_token = ''
        integration.refresh_token = ''
        integration.save()
        
        return Response({'message': f'Successfully disconnected from {platform.title()}'})
    except SocialMediaIntegration.DoesNotExist:
        return Response({'error': 'Integration not found'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def integration_status(request):
    """Get status of all integrations"""
    integrations = SocialMediaIntegration.objects.filter(user=request.user)
    serializer = SocialMediaIntegrationSerializer(integrations, many=True)
    
    # Create a status summary
    status_summary = {
        'connected_platforms': [],
        'available_platforms': ['youtube', 'spotify', 'instagram', 'tiktok', 'facebook', 'twitter'],
        'total_connected': 0
    }
    
    for integration in integrations:
        if integration.status == 'connected':
            status_summary['connected_platforms'].append(integration.platform)
            status_summary['total_connected'] += 1
    
    return Response({
        'integrations': serializer.data,
        'summary': status_summary
    })
