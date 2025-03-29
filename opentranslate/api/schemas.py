"""
API request and response schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from uuid import UUID

class TranslationBase(BaseModel):
    """Base translation schema"""
    source_text: str
    source_lang: str
    target_lang: str
    domain: Optional[str] = None
    priority: Optional[int] = 1

class TranslationCreate(TranslationBase):
    """Translation creation schema"""
    pass

class Translation(TranslationBase):
    """Translation response schema"""
    id: UUID
    target_text: Optional[str] = None
    status: str
    score: Optional[float] = None
    transaction_hash: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class TranslatorBase(BaseModel):
    """Base translator schema"""
    address: str
    name: str
    bio: Optional[str] = None
    languages: List[str]
    domains: Optional[List[str]] = None

class TranslatorCreate(TranslatorBase):
    """Translator creation schema"""
    pass

class Translator(TranslatorBase):
    """Translator response schema"""
    id: UUID
    reputation: float = 0.0
    translations_count: int = 0
    validations_count: int = 0
    stake: float = 0.0
    rewards: float = 0.0
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ValidationBase(BaseModel):
    """Base validation schema"""
    translation_id: UUID
    score: float = Field(ge=0.0, le=1.0)
    feedback: Optional[str] = None

class ValidationCreate(ValidationBase):
    """Validation creation schema"""
    pass

class Validation(ValidationBase):
    """Validation response schema"""
    id: UUID
    validator_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class Stats(BaseModel):
    """Statistics response schema"""
    total_translations: int
    total_translators: int
    total_validations: int

class TranslationRequest(BaseModel):
    """Translation request schema"""
    source_text: str = Field(..., min_length=1)
    source_lang: str = Field(..., min_length=2, max_length=5)
    target_lang: str = Field(..., min_length=2, max_length=5)
    domain: Optional[str] = None
    priority: Optional[int] = Field(default=1, ge=1, le=10)
    
    @validator('source_lang', 'target_lang')
    def validate_language_code(cls, v):
        """Validate language code format"""
        if not v.isalpha():
            raise ValueError('Language code must contain only letters')
        return v.lower()

class TranslationResponse(BaseModel):
    """Translation response schema"""
    id: UUID
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    domain: Optional[str]
    status: str
    score: Optional[float]
    created_at: datetime
    updated_at: datetime
    translator_id: Optional[UUID]
    metrics: Optional[Dict]

class ValidationRequest(BaseModel):
    """Translation validation request schema"""
    translation_id: UUID
    score: Optional[float] = Field(None, ge=0, le=1)
    feedback: Optional[str] = None

class ValidationResponse(BaseModel):
    """Translation validation response schema"""
    id: UUID
    translation_id: UUID
    validator_id: UUID
    score: float
    feedback: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime
    metrics: Optional[Dict]

class TranslatorProfile(BaseModel):
    """Translator profile schema"""
    id: UUID
    address: str
    name: Optional[str]
    bio: Optional[str]
    languages: List[str]
    domains: List[str]
    reputation: float
    translations_count: int
    validations_count: int
    stake: float
    rewards: float
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime

class TranslationStats(BaseModel):
    """Translation statistics schema"""
    total_translations: int
    total_validations: int
    average_score: float
    languages: Dict[str, int]
    domains: Dict[str, int]
    active_translators: int
    active_validators: int
    total_stake: float
    total_rewards: float

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    code: str
    details: Optional[Dict] = None 