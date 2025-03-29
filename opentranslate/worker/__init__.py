"""
OpenTranslate worker initialization
"""

from opentranslate.worker.tasks import celery_app
from opentranslate.worker.config import *

__all__ = ['celery_app'] 