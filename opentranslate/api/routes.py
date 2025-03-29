"""
API routes for translation service
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from uuid import UUID
import logging

from opentranslate.api.schemas import (
    TranslationRequest, TranslationResponse,
    ValidationRequest, ValidationResponse,
    TranslatorProfile, TranslationStats,
    ErrorResponse
)
from opentranslate.models.translation import Translation, Translator, Validation
from opentranslate.worker.tasks import process_translation, process_validation
from opentranslate.utils.exceptions import (
    TranslationError, ValidationError,
    AuthenticationError, AuthorizationError
)
from opentranslate.api.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/translations", response_model=TranslationResponse)
async def create_translation(
    request: TranslationRequest,
    current_user: Translator = Depends(get_current_user)
):
    """Create a new translation request"""
    try:
        # Create translation record
        translation = Translation(
            source_text=request.source_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            domain=request.domain,
            priority=request.priority,
            translator_id=current_user.id
        )
        translation.save()
        
        # Submit translation task
        process_translation.delay(str(translation.id))
        
        return translation.to_dict()
        
    except Exception as exc:
        logger.error(f"Translation creation failed: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )

@router.get("/translations/{translation_id}", response_model=TranslationResponse)
async def get_translation(
    translation_id: UUID,
    current_user: Translator = Depends(get_current_user)
):
    """Get translation details"""
    try:
        translation = Translation.get_by_id(translation_id)
        if not translation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Translation not found"
            )
            
        return translation.to_dict()
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Translation retrieval failed: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )

@router.post("/validations", response_model=ValidationResponse)
async def create_validation(
    request: ValidationRequest,
    current_user: Translator = Depends(get_current_user)
):
    """Create a new validation"""
    try:
        # Check if translation exists
        translation = Translation.get_by_id(request.translation_id)
        if not translation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Translation not found"
            )
            
        # Create validation record
        validation = Validation(
            translation_id=request.translation_id,
            validator_id=current_user.id,
            score=request.score,
            feedback=request.feedback
        )
        validation.save()
        
        # Submit validation task
        process_validation.delay(str(validation.id))
        
        return validation.to_dict()
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Validation creation failed: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )

@router.get("/translators/{address}", response_model=TranslatorProfile)
async def get_translator_profile(address: str):
    """Get translator profile"""
    try:
        translator = Translator.get_by_address(address)
        if not translator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Translator not found"
            )
            
        return translator.to_dict()
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Translator profile retrieval failed: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )

@router.get("/stats", response_model=TranslationStats)
async def get_translation_stats():
    """Get translation statistics"""
    try:
        stats = {
            "total_translations": Translation.query.count(),
            "total_validations": Validation.query.count(),
            "average_score": Translation.query.with_entities(
                func.avg(Translation.score)
            ).scalar() or 0.0,
            "languages": {},
            "domains": {},
            "active_translators": Translator.query.filter_by(
                is_active=True
            ).count(),
            "active_validators": Translator.query.filter_by(
                is_active=True
            ).count(),
            "total_stake": Translator.query.with_entities(
                func.sum(Translator.stake)
            ).scalar() or 0.0,
            "total_rewards": Translator.query.with_entities(
                func.sum(Translator.rewards)
            ).scalar() or 0.0
        }
        
        # Get language and domain statistics
        for translation in Translation.query.all():
            stats["languages"][translation.source_lang] = stats["languages"].get(
                translation.source_lang, 0
            ) + 1
            stats["languages"][translation.target_lang] = stats["languages"].get(
                translation.target_lang, 0
            ) + 1
            if translation.domain:
                stats["domains"][translation.domain] = stats["domains"].get(
                    translation.domain, 0
                ) + 1
                
        return stats
        
    except Exception as exc:
        logger.error(f"Statistics retrieval failed: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        ) 