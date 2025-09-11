"""
YouTube Data API integration service
"""
import requests
from datetime import datetime, timedelta
from django.conf import settings
from .base import BasePlatformService

class YouTubeService(BasePlatformService):
    """YouTube Data API integration"""
    
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    OAUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
    TOKEN_URL = 'https://oauth2.googleapis.com/token'
    
    def get_auth_url(self) -> str:
        """Get YouTube OAuth authorization URL"""
        params = {
            'client_id': settings.YOUTUBE_CLIENT_ID,
            'redirect_uri': settings.YOUTUBE_REDIRECT_URI,
            'scope': 'https://www.googleapis.com/auth/youtube.readonly https://www.googleapis.com/auth/yt-analytics.readonly',
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
        return f'{self.OAUTH_URL}?{query_string}'
    
    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange authorization code for access token"""
        data = {
            'client_id': settings.YOUTUBE_CLIENT_ID,
            'client_secret': settings.YOUTUBE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': settings.YOUTUBE_REDIRECT_URI,
        }
        
        response = requests.post(self.TOKEN_URL, data=data)
        return response.json()
    
    def refresh_access_token(self) -> dict:
        """Refresh expired access token"""
        data = {
            'client_id': settings.YOUTUBE_CLIENT_ID,
            'client_secret': settings.YOUTUBE_CLIENT_SECRET,
            'refresh_token': self.integration.refresh_token,
            'grant_type': 'refresh_token',
        }
        
        response = requests.post(self.TOKEN_URL, data=data)
        token_data = response.json()
        
        if 'access_token' in token_data:
            self.integration.access_token = token_data['access_token']
            if 'refresh_token' in token_data:
                self.integration.refresh_token = token_data['refresh_token']
            self.integration.save()
        
        return token_data
    
    def get_user_info(self) -> dict:
        """Get YouTube channel information"""
        url = f'{self.BASE_URL}/channels'
        params = {
            'part': 'snippet,statistics',
            'mine': 'true',
            'key': settings.YOUTUBE_API_KEY
        }
        
        response = self.make_authenticated_request(url, params=params)
        return response.json()
    
    def get_analytics_data(self, start_date: str, end_date: str) -> dict:
        """Get YouTube Analytics data"""
        # This would use YouTube Analytics API
        # For demo purposes, returning sample data
        return {
            'subscribers': 75000,
            'views': 1500000,
            'watch_time_minutes': 45000,
            'likes': 125000,
            'comments': 8500,
            'shares': 3200,
            'engagement_rate': 8.7,
            'subscriber_growth': 2.3,
            'top_videos': [
                {'title': 'Latest Music Video', 'views': 250000},
                {'title': 'Behind the Scenes', 'views': 180000},
                {'title': 'Live Performance', 'views': 150000}
            ]
        }
    
    def get_channel_statistics(self) -> dict:
        """Get current channel statistics"""
        url = f'{self.BASE_URL}/channels'
        params = {
            'part': 'statistics',
            'mine': 'true',
            'key': settings.YOUTUBE_API_KEY
        }
        
        response = self.make_authenticated_request(url, params=params)
        data = response.json()
        
        if 'items' in data and data['items']:
            stats = data['items'][0]['statistics']
            return {
                'subscribers': int(stats.get('subscriberCount', 0)),
                'total_views': int(stats.get('viewCount', 0)),
                'video_count': int(stats.get('videoCount', 0))
            }
        
        return {}
