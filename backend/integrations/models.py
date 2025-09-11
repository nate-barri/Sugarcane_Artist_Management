from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class SocialMediaIntegration(models.Model):
    """Store social media platform integration details"""
    PLATFORM_CHOICES = [
        ('youtube', 'YouTube'),
        ('spotify', 'Spotify'),
        ('instagram', 'Instagram'),
        ('tiktok', 'TikTok'),
        ('facebook', 'Facebook'),
        ('twitter', 'Twitter'),
    ]
    
    STATUS_CHOICES = [
        ('connected', 'Connected'),
        ('disconnected', 'Disconnected'),
        ('error', 'Error'),
        ('pending', 'Pending'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    platform = models.CharField(max_length=20, choices=PLATFORM_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='disconnected')
    
    # OAuth tokens and credentials
    access_token = models.TextField(blank=True)
    refresh_token = models.TextField(blank=True)
    token_expires_at = models.DateTimeField(null=True, blank=True)
    
    # Platform-specific data
    platform_user_id = models.CharField(max_length=100, blank=True)
    platform_username = models.CharField(max_length=100, blank=True)
    platform_data = models.JSONField(default=dict)
    
    # Sync settings
    auto_sync = models.BooleanField(default=True)
    last_sync = models.DateTimeField(null=True, blank=True)
    sync_frequency = models.IntegerField(default=24)  # hours
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'platform']
    
    def __str__(self):
        return f"{self.user.email} - {self.get_platform_display()}"

class APIKey(models.Model):
    """Store API keys for different services"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    service_name = models.CharField(max_length=50)
    key_name = models.CharField(max_length=100)
    key_value = models.TextField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user', 'service_name', 'key_name']
    
    def __str__(self):
        return f"{self.user.email} - {self.service_name} - {self.key_name}"
