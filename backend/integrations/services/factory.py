"""
Factory for creating platform service instances
"""
from .youtube import YouTubeService
from .spotify import SpotifyService
from .instagram import InstagramService
from .tiktok import TikTokService

class PlatformServiceFactory:
    """Factory for creating platform service instances"""
    
    SERVICES = {
        'youtube': YouTubeService,
        'spotify': SpotifyService,
        'instagram': InstagramService,
        'tiktok': TikTokService,
    }
    
    @classmethod
    def create_service(cls, integration):
        """Create a service instance for the given integration"""
        service_class = cls.SERVICES.get(integration.platform)
        if not service_class:
            raise ValueError(f"No service available for platform: {integration.platform}")
        
        return service_class(integration)
    
    @classmethod
    def get_available_platforms(cls):
        """Get list of available platforms"""
        return list(cls.SERVICES.keys())
