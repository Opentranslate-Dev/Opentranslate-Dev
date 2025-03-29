"""
OpenTranslate Translation Protocol Definitions
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

class TranslationPriority(str, Enum):
    """Translation Priority"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class TranslationDomain(str, Enum):
    """Translation Domain"""
    PHYSICS = "physics"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    MATHEMATICS = "mathematics"
    COMPUTER_SCIENCE = "computer_science"
    MEDICINE = "medicine"
    OTHER = "other"

class TranslationStatus(str, Enum):
    """Translation Status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"

class TranslationRequest(BaseModel):
    """Translation Request"""
    text: Union[str, List[str]]
    source_lang: str
    target_lang: str
    domain: Optional[TranslationDomain] = None
    priority: TranslationPriority = TranslationPriority.NORMAL
    metadata: Optional[Dict] = Field(default_factory=dict)

class TranslationResponse(BaseModel):
    """Translation Response"""
    request_id: str
    translations: List[str]
    source_lang: str
    target_lang: str
    model: str
    domain: Optional[TranslationDomain]
    priority: TranslationPriority
    status: TranslationStatus
    timestamp: float
    metadata: Dict = Field(default_factory=dict)
    validation: Optional[Dict] = None

@dataclass
class TranslationTask:
    """Translation Task"""
    request: TranslationRequest
    status: TranslationStatus = TranslationStatus.PENDING
    result: Optional[TranslationResponse] = None
    error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    validator_id: Optional[str] = None
    blockchain_hash: Optional[str] = None

class TranslationProtocol:
    """OpenTranslate Translation Protocol"""
    
    @staticmethod
    def create_request(
        text: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        domain: Optional[TranslationDomain] = None,
        priority: TranslationPriority = TranslationPriority.NORMAL,
        metadata: Optional[Dict] = None,
    ) -> TranslationRequest:
        """
        Create Translation Request
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Translation domain
            priority: Translation priority
            metadata: Metadata
            
        Returns:
            Translation request object
        """
        return TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            priority=priority,
            metadata=metadata or {}
        )
    
    @staticmethod
    def create_response(
        request: TranslationRequest,
        translations: List[str],
        model: str,
        status: TranslationStatus = TranslationStatus.COMPLETED,
        validation: Optional[Dict] = None,
    ) -> TranslationResponse:
        """
        Create Translation Response
        
        Args:
            request: Original request
            translations: List of translation results
            model: Model used
            status: Translation status
            validation: Validation result
            
        Returns:
            Translation response object
        """
        return TranslationResponse(
            request_id=str(id(request)),
            translations=translations,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            model=model,
            domain=request.domain,
            priority=request.priority,
            status=status,
            timestamp=0.0,  # Will be set when recorded
            metadata=request.metadata,
            validation=validation
        ) 