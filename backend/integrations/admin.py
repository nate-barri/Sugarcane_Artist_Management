from django.contrib import admin
from .models import SocialMediaIntegration, APIKey

@admin.register(SocialMediaIntegration)
class SocialMediaIntegrationAdmin(admin.ModelAdmin):
    list_display = ('user', 'platform', 'status', 'platform_username', 'last_sync')
    list_filter = ('platform', 'status', 'auto_sync')
    search_fields = ('user__email', 'platform_username')

@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ('user', 'service_name', 'key_name', 'is_active')
    list_filter = ('service_name', 'is_active')
    search_fields = ('user__email', 'service_name')
