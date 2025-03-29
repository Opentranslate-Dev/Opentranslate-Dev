"""
Celery configuration settings
"""

from celery.schedules import crontab

# Broker settings
broker_url = 'redis://localhost:6379/1'
result_backend = 'redis://localhost:6379/2'

# Task settings
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True

# Worker settings
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 100
worker_max_memory_per_child = 200000  # 200MB

# Task execution settings
task_track_started = True
task_time_limit = 3600  # 1 hour
task_soft_time_limit = 3300  # 55 minutes

# Task routing
task_routes = {
    'opentranslate.worker.tasks.process_translation': {'queue': 'translations'},
    'opentranslate.worker.tasks.process_validation': {'queue': 'validations'},
    'opentranslate.worker.tasks.cleanup_old_tasks': {'queue': 'maintenance'}
}

# Task queues
task_queues = {
    'translations': {
        'exchange': 'translations',
        'routing_key': 'translations',
        'queue_arguments': {'x-max-priority': 10}
    },
    'validations': {
        'exchange': 'validations',
        'routing_key': 'validations',
        'queue_arguments': {'x-max-priority': 5}
    },
    'maintenance': {
        'exchange': 'maintenance',
        'routing_key': 'maintenance',
        'queue_arguments': {'x-max-priority': 1}
    }
}

# Beat schedule
beat_schedule = {
    'cleanup-old-tasks': {
        'task': 'opentranslate.worker.tasks.cleanup_old_tasks',
        'schedule': crontab(hour=0, minute=0)  # Run daily at midnight
    }
}

# Logging
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s' 