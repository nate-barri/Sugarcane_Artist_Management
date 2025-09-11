"""
Spotify Web API integration service
"""
import requests
import base64
from datetime import datetime, timedelta
from django.conf import settings
from .base import BasePlatformService

class SpotifyService(BasePlatformService):
    """Spotify Web API integration"""
    
    BASE_URL = 'https://api.spotify.com/v1'
    AUTH_URL = 'https://accounts.spotify.com/authorize'
    TOKEN_URL = 'https://accounts.spotify.com/api/token'
    
    def get_auth_url(self) -> str:
        """Get Spotify OAuth authorization URL"""
        params = {
            'client_id': settings.SPOTIFY_CLIENT_ID,
            'response_type': 'code',
            'redirect_uri': settings.SPOTIFY_REDIRECT_URI,
            'scope': 'user-read-private user-read-email user-top-read user-read-recently-played',
            'show_dialog': 'true'
        }
        
        query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
        return f'{self.AUTH_URL}?{query_string}'
    
    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange authorization code for access token"""
        auth_header = base64.b64encode(
            f"{settings.SPOTIFY_CLIENT_ID}:{settings.SPOTIFY_CLIENT_SECRET}".encode()
        ).decode()
        
        headers = {
            'Authorization': f'Basic {auth_header}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': settings.SPOTIFY_REDIRECT_URI,
        }
        
        response = requests.post(self.TOKEN_URL, headers=headers, data=data)
        return response.json()
    
    def refresh_access_token(self) -> dict:
        """Refresh expired access token"""
        auth_header = base64.b64encode(
            f"{settings.SPOTIFY_CLIENT_ID}:{settings.SPOTIFY_CLIENT_SECRET}".encode()
        ).decode()
        
        headers = {
            'Authorization': f'Basic {auth_header}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.integration.refresh_token,
        }
        
        response = requests.post(self.TOKEN_URL, headers=headers, data=data)
        token_data = response.json()
        
        if 'access_token' in token_data:
            self.integration.access_token = token_data['access_token']
            if 'refresh_token' in token_data:
                self.integration.refresh_token = token_data['refresh_token']
            self.integration.save()
        
        return token_data
    
    def get_user_info(self) -> dict:
        """Get Spotify user profile"""
        url = f'{self.BASE_URL}/me'
        response = self.make_authenticated_request(url)
        return response.json()
    
    def get_analytics_data(self, start_date: str, end_date: str) -> dict:
        """Get Spotify analytics data (simulated)"""
        # Note: Spotify doesn't provide analytics through their public API
        # This would typically require Spotify for Artists API access
        return {
            'followers': 30000,
            'monthly_listeners': 85000,
            'streams': 800000,
            'saves': 12000,
            'playlist_adds': 5500,
            'top_tracks': [
                {'name': 'Hit Single', 'streams': 150000},
                {'name': 'Popular Track', 'streams': 120000},
                {'name': 'Fan Favorite', 'streams': 95000}
            ],
            'countries': [
                {'country': 'US', 'listeners': 25000},
                {'country': 'UK', 'listeners': 15000},
                {'country': 'CA', 'listeners': 12000}
            ]
        }
    
    def get_top_tracks(self, time_range: str = 'medium_term') -> dict:
        """Get user's top tracks"""
        url = f'{self.BASE_URL}/me/top/tracks'
        params = {
            'time_range': time_range,  # short_term, medium_term, long_term
            'limit': 20
        }
        
        response = self.make_authenticated_request(url, params=params)
        return response.json()
