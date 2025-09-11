"""
Database setup script for Sugarcane Artist Management Backend
Run this script to initialize the database with sample data
"""

import os
import sys
import django
from datetime import datetime, timedelta

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sugarcane_backend.settings')
django.setup()

from django.contrib.auth import get_user_model
from analytics.models import Platform, AnalyticsData, DashboardMetrics
from integrations.models import SocialMediaIntegration
from reports.models import ReportTemplate

User = get_user_model()

def create_platforms():
    """Create social media platforms"""
    platforms = [
        {'name': 'youtube', 'display_name': 'YouTube', 'color': '#FF0000'},
        {'name': 'spotify', 'display_name': 'Spotify', 'color': '#1DB954'},
        {'name': 'instagram', 'display_name': 'Instagram', 'color': '#E4405F'},
        {'name': 'tiktok', 'display_name': 'TikTok', 'color': '#000000'},
        {'name': 'facebook', 'display_name': 'Facebook', 'color': '#1877F2'},
        {'name': 'twitter', 'display_name': 'Twitter', 'color': '#1DA1F2'},
    ]
    
    for platform_data in platforms:
        platform, created = Platform.objects.get_or_create(
            name=platform_data['name'],
            defaults=platform_data
        )
        if created:
            print(f"Created platform: {platform.display_name}")

def create_sample_user():
    """Create a sample user for testing"""
    email = 'demo@sugarcane.com'
    user, created = User.objects.get_or_create(
        email=email,
        defaults={
            'username': 'demo_user',
            'full_name': 'Demo Artist',
            'artist_name': 'Sugarcane Artist',
            'is_active': True,
        }
    )
    
    if created:
        user.set_password('demo123')
        user.save()
        print(f"Created sample user: {email}")
        
        # Create sample integrations
        platforms = Platform.objects.all()
        for platform in platforms[:3]:  # Connect first 3 platforms
            SocialMediaIntegration.objects.get_or_create(
                user=user,
                platform=platform.name,
                defaults={
                    'status': 'connected',
                    'platform_username': f'demo_{platform.name}',
                    'platform_user_id': f'{platform.name}_demo_123',
                }
            )
        
        # Create sample analytics data
        today = datetime.now().date()
        for i in range(30):  # Last 30 days
            date = today - timedelta(days=i)
            
            DashboardMetrics.objects.get_or_create(
                user=user,
                date=date,
                defaults={
                    'total_followers': 125000 + (i * 100),
                    'total_views': 2500000 + (i * 5000),
                    'total_engagement': 85000 + (i * 200),
                    'audience_growth_rate': 12.5 + (i * 0.1),
                    'top_performing_platform': 'Spotify',
                    'watch_time_hours': 15000 + (i * 50),
                    'spotify_streams': 1800000 + (i * 3000),
                }
            )
    
    return user

def create_report_templates():
    """Create default report templates"""
    templates = [
        {
            'name': 'Monthly Overview',
            'description': 'Comprehensive monthly performance report',
            'report_type': 'dashboard',
            'default_metrics': ['followers', 'views', 'engagement', 'growth'],
            'default_platforms': ['youtube', 'spotify', 'instagram'],
        },
        {
            'name': 'Engagement Analysis',
            'description': 'Detailed engagement metrics across platforms',
            'report_type': 'engagement',
            'default_metrics': ['likes', 'comments', 'shares', 'engagement_rate'],
            'default_platforms': ['youtube', 'instagram', 'tiktok'],
        },
        {
            'name': 'Growth Report',
            'description': 'Audience growth and retention analysis',
            'report_type': 'growth',
            'default_metrics': ['followers', 'subscriber_growth', 'retention'],
            'default_platforms': ['youtube', 'spotify'],
        },
    ]
    
    for template_data in templates:
        template, created = ReportTemplate.objects.get_or_create(
            name=template_data['name'],
            defaults=template_data
        )
        if created:
            print(f"Created report template: {template.name}")

def main():
    """Main setup function"""
    print("Setting up Sugarcane Artist Management Database...")
    
    print("\n1. Creating platforms...")
    create_platforms()
    
    print("\n2. Creating sample user...")
    user = create_sample_user()
    
    print("\n3. Creating report templates...")
    create_report_templates()
    
    print("\n✅ Database setup complete!")
    print(f"\nSample login credentials:")
    print(f"Email: demo@sugarcane.com")
    print(f"Password: demo123")
    print(f"\nYou can also use the original hardcoded credentials:")
    print(f"Username: capstone")
    print(f"Password: sugarcane")

if __name__ == '__main__':
    main()
