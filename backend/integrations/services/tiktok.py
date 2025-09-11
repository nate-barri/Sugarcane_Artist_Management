"""
TikTok for Developers API integration service
"""
import requests
from datetime import datetime, timedelta
from django.conf import settings
from .base import BasePlatformService

class TikTokService(BasePlatformService):
    """TikTok for Developers API integration"""
    
    BASE_URL = 'https://open-api.tiktok.com'
    AUTH_URL = 'https://www.tiktok.com/auth/authorize'
    TOKEN_URL = 'https://open-api.tiktok.com/oauth/access_token'
    
    def get_auth_url(self) -> str:
        """Get TikTok OAuth authorization URL"""
        params = {
            'client_key': settings.TIKTOK_CLIENT_KEY,
            'scope': 'user.info.basic,video.list',
            'response_type': 'code',
            'redirect_uri': settings.TIKTOK_REDIRECT_URI,
            'state': 'tiktok_auth'
        }
        
        query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
        return f'{self.AUTH_URL}?{query_string}'
    
    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange authorization code for access token"""
        data = {
            'client_key': settings.TIKTOK_CLIENT_KEY,
            'client_secret': settings.TIKTOK_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': settings.TIKTOK_REDIRECT_URI,
        }
        
        response = requests.post(self.TOKEN_URL, json=data)
        return response.json()
    
    def refresh_access_token(self) -> dict:
        """Refresh expired access token"""
        data = {
            'client_key': settings.TIKTOK_CLIENT_KEY,
            'client_secret': settings.TIKTOK_CLIENT_SECRET,
            'grant_type': 'refresh_token',
            'refresh_token': self.integration.refresh_token,
        }
        
        response = requests.post(self.TOKEN_URL, json=data)
        token_data = response.json()
        
        if 'access_token' in token_data.get('data', {}):
            self.integration.access_token = token_data['data']['access_token']
            if 'refresh_token' in token_data['data']:
                self.integration.refresh_token = token_data['data']['refresh_token']
            self.integration.save()
        
        return token_data
    
    def get_user_info(self) -> dict:
        """Get TikTok user profile"""
        url = f'{self.BASE_URL}/user/info'
        params = {
            'access_token': self.integration.access_token,
            'fields': 'open_id,union_id,avatar_url,display_name,follower_count,following_count,likes_count,video_count'
        }
        
        response = requests.get(url, params=params)
        return response.json()
    
    def get_analytics_data(self, start_date: str, end_date: str) -> dict:
        """Get TikTok analytics data (simulated)"""
        return {
            'followers': 45000,
            'videos': 85,
            'likes': 125000,
            'comments': 8500,
            'shares': 12000,
            'views': 2800000,
            'engagement_rate': 9.2,
            'average_watch_time': 15.5,
            'top_videos': [
                {'id': '1', 'views': 450000, 'likes': 25000},
                {'id': '2', 'views': 380000, 'likes': 22000},
                {'id': '3', 'views': 320000, 'likes': 18000}
            ]
        }
    
    def get_videos(self, cursor: int = 0, max_count: int = 20) -> dict:
        """Get user's videos"""
        url = f'{self.BASE_URL}/video/list'
        data = {
            'access_token': self.integration.access_token,
            'cursor': cursor,
            'max_count': max_count,
            'fields': 'id,title,video_description,duration,cover_image_url,create_time,view_count,like_count,comment_count,share_count'
        }
        
        response = requests.post(url, json=data)
        return response.json()
