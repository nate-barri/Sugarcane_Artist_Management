"""
Management command to sync all platform data
"""
from django.core.management.base import BaseCommand
from integrations.tasks import sync_all_integrations

class Command(BaseCommand):
    help = 'Sync data from all connected social media platforms'
    
    def handle(self, *args, **options):
        self.stdout.write('Starting data synchronization...')
        
        results = sync_all_integrations()
        
        for result in results:
            self.stdout.write(self.style.SUCCESS(result))
        
        self.stdout.write(
            self.style.SUCCESS('Data synchronization completed!')
        )
