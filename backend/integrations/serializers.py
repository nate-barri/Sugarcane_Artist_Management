from rest_framework import serializers
from .models import SocialMediaIntegration, APIKey

class SocialMediaIntegrationSerializer(serializers.ModelSerializer):
    platform_display = serializers.CharField(source='get_platform_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = SocialMediaIntegration
        fields = ['id', 'platform', 'platform_display', 'status', 'status_display', 
                 'platform_username', 'auto_sync', 'last_sync', 'created_at']
        read_only_fields = ['user', 'last_sync', 'created_at']

class APIKeySerializer(serializers.ModelSerializer):
    class Meta:
        model = APIKey
        fields = ['id', 'service_name', 'key_name', 'is_active', 'created_at']
        read_only_fields = ['user', 'created_at']
