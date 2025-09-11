"""
Instagram Basic Display API integration service
"""
import requests
from datetime import datetime, timedelta
from django.conf import settings
from .base import BasePlatformService

class InstagramService(BasePlatformService):
    """Instagram Basic Display API integration"""
    
    BASE_URL = 'https://graph.instagram.com'
    AUTH_URL = 'https://api.instagram.com/oauth/authorize'
    TOKEN_URL = 'https://api.instagram.com/oauth/access_token'
    
    def get_auth_url(self) -> str:
        """Get Instagram OAuth authorization URL"""
        params = {
            'client_id': settings.INSTAGRAM_CLIENT_ID,
            'redirect_uri': settings.INSTAGRAM_REDIRECT_URI,
            'scope': 'user_profile,user_media',
            'response_type': 'code'
        }
        
        query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
        return f'{self.AUTH_URL}?{query_string}'
    
    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange authorization code for access token"""
        data = {
            'client_id': settings.INSTAGRAM_CLIENT_ID,
            'client_secret': settings.INSTAGRAM_CLIENT_SECRET,
            'grant_type': 'authorization_code',
            'redirect_uri': settings.INSTAGRAM_REDIRECT_URI,
            'code': code,
        }
        
        response = requests.post(self.TOKEN_URL, data=data)
        return response.json()
    
    def refresh_access_token(self) -> dict:
        """Refresh long-lived access token"""
        url = f'{self.BASE_URL}/refresh_access_token'
        params = {
            'grant_type': 'ig_refresh_token',
            'access_token': self.integration.access_token
        }
        
        response = requests.get(url, params=params)
        token_data = response.json()
        
        if 'access_token' in token_data:
            self.integration.access_token = token_data['access_token']
            self.integration.save()
        
        return token_data
    
    def get_user_info(self) -> dict:
        """Get Instagram user profile"""
        url = f'{self.BASE_URL}/me'
        params = {
            'fields': 'id,username,account_type,media_count',
            'access_token': self.integration.access_token
        }
        
        response = requests.get(url, params=params)
        return response.json()
    
    def get_analytics_data(self, start_date: str, end_date: str) -> dict:
        """Get Instagram analytics data (simulated)"""
        # Note: Instagram Insights require Instagram Business API
        return {
            'followers': 20000,
            'posts': 150,
            'likes': 45000,
            'comments': 3200,
            'shares': 1800,
            'saves': 2500,
            'reach': 85000,
            'impressions': 125000,
            'engagement_rate': 6.8,
            'top_posts': [
                {'id': '1', 'likes': 2500, 'comments': 180},
                {'id': '2', 'likes': 2200, 'comments': 150},
                {'id': '3', 'likes': 1900, 'comments': 120}
            ]
        }
    
    def get_media(self, limit: int = 25) -> dict:
        """Get user's media"""
        url = f'{self.BASE_URL}/me/media'
        params = {
            'fields': 'id,caption,media_type,media_url,permalink,timestamp',
            'limit': limit,
            'access_token': self.integration.access_token
        }
        
        response = requests.get(url, params=params)
        return response.json()
