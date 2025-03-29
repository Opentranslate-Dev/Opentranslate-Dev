"""
Core translation engine for OpenTranslate
"""

from typing import Dict, List, Optional, Union
import asyncio
from datetime import datetime
import logging
from uuid import UUID, uuid4

from ..config.default import settings
from ..ai.models import TranslationModel, ValidationModel, DomainClassifier
from ..blockchain.contracts import TranslationContract
from ..utils.exceptions import (
    InvalidLanguageError,
    InvalidDomainError,
    TranslationError,
    ValidationError
)

logger = logging.getLogger(__name__)

class TranslationEngine:
    """Core translation engine that coordinates AI models, validation, and blockchain"""
    
    def __init__(
        self,
        translation_model: Optional[str] = None,
        validation_model: Optional[str] = None,
        classifier_model: Optional[str] = None,
        device: Optional[str] = None,
        contract_address: Optional[str] = None,
    ):
        """
        Initialize the translation engine
        
        Args:
            translation_model: Path or name of the translation model
            validation_model: Path or name of the validation model
            classifier_model: Path or name of the domain classifier model
            device: Device to run models on (cuda/cpu)
            contract_address: Address of the translation smart contract
        """
        # Initialize AI models
        self.translator = TranslationModel(
            model_name=translation_model or settings.TRANSLATION_MODEL,
            device=device or settings.DEVICE
        )
        
        self.validator = ValidationModel(
            model_name=validation_model or settings.VALIDATION_MODEL,
            device=device or settings.DEVICE
        )
        
        self.classifier = DomainClassifier(
            model_name=classifier_model or settings.CLASSIFIER_MODEL,
            device=device or settings.DEVICE
        )
        
        # Initialize blockchain contract
        self.contract = TranslationContract(
            address=contract_address or settings.CONTRACT_ADDRESS
        )
        
        # Task management
        self._tasks: Dict[UUID, Dict] = {}
        self._lock = asyncio.Lock()
        
    def translate(
        self,
        text: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        domain: Optional[str] = None,
        priority: str = "normal",
        batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Translate text from source language to target language
        
        Args:
            text: Text or list of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain/field of the text
            priority: Translation priority (low/normal/high/urgent)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing translation results and metadata
        """
        # Validate inputs
        if source_lang not in settings.SUPPORTED_LANGUAGES:
            raise InvalidLanguageError(f"Unsupported source language: {source_lang}")
            
        if target_lang not in settings.SUPPORTED_LANGUAGES:
            raise InvalidLanguageError(f"Unsupported target language: {target_lang}")
            
        if domain and domain not in settings.SUPPORTED_DOMAINS:
            raise InvalidDomainError(f"Unsupported domain: {domain}")
            
        if priority not in settings.PRIORITY_LEVELS:
            priority = "normal"
            
        # Create translation task
        task_id = uuid4()
        self._tasks[task_id] = {
            "status": "pending",
            "created_at": datetime.utcnow(),
            "priority": priority,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "domain": domain
        }
        
        try:
            # Auto-detect domain if not provided
            if not domain:
                domain_result = self.classifier.classify(text)
                domain = domain_result["main_domain"]
            
            # Perform translation
            translation_result = self.translator.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain
            )
            
            # Validate translation
            validation_result = self.validator.validate(
                source=text,
                target=translation_result["translations"],
                domain=domain
            )
            
            # Update blockchain
            tx_hash = self.contract.record_translation(
                task_id=task_id,
                source_text=text,
                translation=translation_result["translations"],
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain,
                score=validation_result["score"]
            )
            
            # Update task status
            self._tasks[task_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "tx_hash": tx_hash,
                **translation_result,
                **validation_result
            })
            
            return self._tasks[task_id]
            
        except Exception as e:
            self._tasks[task_id]["status"] = "failed"
            self._tasks[task_id]["error"] = str(e)
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Translation failed: {e}")
            
    def get_task(self, task_id: UUID) -> Optional[Dict]:
        """Get translation task status and results"""
        return self._tasks.get(task_id)
        
    def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List translation tasks with optional filtering"""
        tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t["status"] == status]
            
        return tasks[offset:offset + limit]
        
    def get_model_info(self) -> Dict:
        """Get information about the translation models"""
        return {
            "translator": self.translator.get_model_info(),
            "validator": self.validator.get_model_info(),
            "classifier": self.classifier.get_model_info()
        }
        
    async def start_background_tasks(self):
        """Start background tasks like health checks"""
        while True:
            try:
                # Clean up old tasks
                current_time = datetime.utcnow()
                async with self._lock:
                    for task_id, task in list(self._tasks.items()):
                        if (
                            task["status"] in ["completed", "failed"]
                            and (current_time - task["created_at"]).days > 7
                        ):
                            del self._tasks[task_id]
                            
                # Health check
                model_info = self.get_model_info()
                logger.info(f"Models health check: {model_info}")
                
            except Exception as e:
                logger.error(f"Background task error: {e}")
                
            await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL) 