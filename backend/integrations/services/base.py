"""
Base class for social media platform integrations
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
from django.conf import settings

class BasePlatformService(ABC):
    """Base class for all social media platform services"""
    
    def __init__(self, integration):
        self.integration = integration
        self.user = integration.user
        
    @abstractmethod
    def get_auth_url(self) -> str:
        """Get OAuth authorization URL"""
        pass
    
    @abstractmethod
    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        pass
    
    @abstractmethod
    def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh expired access token"""
        pass
    
    @abstractmethod
    def get_user_info(self) -> Dict[str, Any]:
        """Get user profile information"""
        pass
    
    @abstractmethod
    def get_analytics_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get analytics data for date range"""
        pass
    
    def make_authenticated_request(self, url: str, method: str = 'GET', **kwargs) -> requests.Response:
        """Make authenticated API request"""
        headers = kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {self.integration.access_token}'
        kwargs['headers'] = headers
        
        response = requests.request(method, url, **kwargs)
        
        # Handle token refresh if needed
        if response.status_code == 401:
            self.refresh_access_token()
            headers['Authorization'] = f'Bearer {self.integration.access_token}'
            response = requests.request(method, url, **kwargs)
        
        return response
