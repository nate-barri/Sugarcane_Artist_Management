from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Platform(models.Model):
    """Social media platforms"""
    name = models.CharField(max_length=50, unique=True)
    display_name = models.CharField(max_length=100)
    icon = models.CharField(max_length=100, blank=True)
    color = models.CharField(max_length=7, default='#000000')  # Hex color
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return self.display_name

class AnalyticsData(models.Model):
    """Store analytics data for different platforms"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    platform = models.ForeignKey(Platform, on_delete=models.CASCADE)
    date = models.DateField()
    
    # Common metrics
    followers = models.IntegerField(default=0)
    views = models.BigIntegerField(default=0)
    likes = models.BigIntegerField(default=0)
    comments = models.BigIntegerField(default=0)
    shares = models.BigIntegerField(default=0)
    engagement_rate = models.FloatField(default=0.0)
    
    # Platform-specific metrics (stored as JSON)
    platform_specific_data = models.JSONField(default=dict)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'platform', 'date']
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.user.email} - {self.platform.name} - {self.date}"

class DashboardMetrics(models.Model):
    """Aggregated dashboard metrics"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    
    # Aggregated metrics
    total_followers = models.IntegerField(default=0)
    total_views = models.BigIntegerField(default=0)
    total_engagement = models.BigIntegerField(default=0)
    audience_growth_rate = models.FloatField(default=0.0)
    top_performing_platform = models.CharField(max_length=50, blank=True)
    
    # Additional metrics
    watch_time_hours = models.FloatField(default=0.0)
    spotify_streams = models.BigIntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'date']
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.user.email} Dashboard - {self.date}"
