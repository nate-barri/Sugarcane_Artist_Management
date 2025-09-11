"""
Celery configuration for background tasks
"""
import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sugarcane_backend.settings')

app = Celery('sugarcane_backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Celery beat schedule for periodic tasks
app.conf.beat_schedule = {
    'sync-all-integrations': {
        'task': 'integrations.tasks.sync_all_integrations',
        'schedule': 3600.0,  # Run every hour
    },
    'refresh-expired-tokens': {
        'task': 'integrations.tasks.refresh_expired_tokens',
        'schedule': 1800.0,  # Run every 30 minutes
    },
}

app.conf.timezone = 'UTC'
