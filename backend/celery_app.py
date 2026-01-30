from celery import Celery

# Configure Celery to use Redis as the message broker and result backend
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    # --- NEW AND CRITICAL ---
    # Add the module where your tasks are defined to the 'include' list.
    include=['backend.tasks']
)

celery_app.conf.update(
    task_track_started=True,
)