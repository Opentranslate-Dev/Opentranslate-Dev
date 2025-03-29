"""
Celery tasks for translation processing
"""

from celery import Celery
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta

from opentranslate.core.translation import TranslationEngine
from opentranslate.models.translation import Translation, Translator, Validation
from opentranslate.utils.exceptions import TranslationError, ValidationError

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery('opentranslate')

# Load configuration from environment
celery_app.config_from_object('opentranslate.config.settings', namespace='CELERY')

@celery_app.task(bind=True, max_retries=3)
def process_translation(self, translation_id: str) -> Dict:
    """
    Process a translation task
    
    Args:
        translation_id: UUID of the translation record
        
    Returns:
        Dict containing translation results
    """
    try:
        # Get translation record
        translation = Translation.get_by_id(translation_id)
        if not translation:
            raise TranslationError(f"Translation {translation_id} not found")
            
        # Get translator
        translator = Translator.get_by_id(translation.translator_id)
        if not translator:
            raise TranslationError(f"Translator {translation.translator_id} not found")
            
        # Initialize translation engine
        engine = TranslationEngine()
        
        # Classify domain if not specified
        if not translation.domain:
            domain, _ = engine.domain_classifier.classify(translation.source_text)
            translation.domain = domain
            translation.save()
            
        # Perform translation
        target_text = engine.translate(
            text=translation.source_text,
            source_lang=translation.source_lang,
            target_lang=translation.target_lang,
            domain=translation.domain
        )
        
        # Validate translation
        score, metrics = engine.validation_model.validate(
            source_text=translation.source_text,
            translation=target_text,
            source_lang=translation.source_lang,
            target_lang=translation.target_lang
        )
        
        # Update translation record
        translation.target_text = target_text
        translation.score = score
        translation.status = "completed"
        translation.save()
        
        # Update translator stats
        translator.translations_count += 1
        translator.save()
        
        return {
            "translation_id": translation_id,
            "status": "success",
            "score": score,
            "metrics": metrics
        }
        
    except Exception as exc:
        logger.error(f"Translation failed: {str(exc)}")
        self.retry(exc=exc, countdown=60)

@celery_app.task(bind=True, max_retries=3)
def process_validation(self, validation_id: str) -> Dict:
    """
    Process a validation task
    
    Args:
        validation_id: UUID of the validation record
        
    Returns:
        Dict containing validation results
    """
    try:
        # Get validation record
        validation = Validation.get_by_id(validation_id)
        if not validation:
            raise ValidationError(f"Validation {validation_id} not found")
            
        # Get translation
        translation = Translation.get_by_id(validation.translation_id)
        if not translation:
            raise ValidationError(f"Translation {validation.translation_id} not found")
            
        # Get validator
        validator = Translator.get_by_id(validation.validator_id)
        if not validator:
            raise ValidationError(f"Validator {validation.validator_id} not found")
            
        # Initialize translation engine
        engine = TranslationEngine()
        
        # Validate translation
        score, metrics = engine.validation_model.validate(
            source_text=translation.source_text,
            translation=translation.target_text,
            source_lang=translation.source_lang,
            target_lang=translation.target_lang
        )
        
        # Update validation record
        validation.score = score
        validation.status = "completed"
        validation.save()
        
        # Update validator stats
        validator.validations_count += 1
        validator.save()
        
        # Update translation score if needed
        if score > translation.score or not translation.score:
            translation.score = score
            translation.save()
        
        return {
            "validation_id": validation_id,
            "status": "success",
            "score": score,
            "metrics": metrics
        }
        
    except Exception as exc:
        logger.error(f"Validation failed: {str(exc)}")
        self.retry(exc=exc, countdown=60)

@celery_app.task
def cleanup_old_tasks() -> None:
    """Clean up old completed tasks"""
    try:
        # Clean up translations older than 30 days
        old_translations = Translation.query.filter(
            Translation.created_at < datetime.utcnow() - timedelta(days=30),
            Translation.status == "completed"
        ).all()
        
        for translation in old_translations:
            translation.delete()
            
        # Clean up validations older than 30 days
        old_validations = Validation.query.filter(
            Validation.created_at < datetime.utcnow() - timedelta(days=30),
            Validation.status == "completed"
        ).all()
        
        for validation in old_validations:
            validation.delete()
            
    except Exception as exc:
        logger.error(f"Cleanup failed: {str(exc)}") 