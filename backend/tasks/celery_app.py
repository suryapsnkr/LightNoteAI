from celery import Celery
from app.config import settings

celery_app = Celery(
    "video_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["tasks.video_task"]
)

celery_app.conf.update(
    broker_connection_retry_on_startup=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)