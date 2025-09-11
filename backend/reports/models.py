from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Report(models.Model):
    """Generated reports for users"""
    REPORT_TYPES = [
        ('dashboard', 'Dashboard Overview'),
        ('platform', 'Platform Specific'),
        ('engagement', 'Engagement Analysis'),
        ('growth', 'Growth Report'),
        ('custom', 'Custom Report'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('generating', 'Generating'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Report configuration
    date_range_start = models.DateField()
    date_range_end = models.DateField()
    platforms = models.JSONField(default=list)  # List of platforms to include
    metrics = models.JSONField(default=list)    # List of metrics to include
    
    # Generated report data
    report_data = models.JSONField(default=dict)
    file_path = models.CharField(max_length=500, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.user.email}"

class ReportTemplate(models.Model):
    """Predefined report templates"""
    name = models.CharField(max_length=100)
    description = models.TextField()
    report_type = models.CharField(max_length=20, choices=Report.REPORT_TYPES)
    default_metrics = models.JSONField(default=list)
    default_platforms = models.JSONField(default=list)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return self.name
