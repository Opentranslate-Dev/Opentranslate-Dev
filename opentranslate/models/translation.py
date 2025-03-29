"""
Database models for translation data
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    JSON,
    ForeignKey,
    Enum,
    Boolean
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from ..database import Base
from ..config.default import settings

class Translation(Base):
    """Translation record model"""
    
    __tablename__ = "translations"
    
    id = Column(PGUUID, primary_key=True, default=uuid4)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Translation content
    source_text = Column(String, nullable=False)
    translation = Column(String, nullable=False)
    source_lang = Column(String, nullable=False)
    target_lang = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    
    # Metadata
    priority = Column(String, nullable=False, default="normal")
    status = Column(String, nullable=False, default="pending")
    score = Column(Float, nullable=True)
    tx_hash = Column(String, nullable=True)
    
    # Relationships
    translator_id = Column(PGUUID, ForeignKey("translators.id"), nullable=True)
    translator = relationship("Translator", back_populates="translations")
    
    validations = relationship("Validation", back_populates="translation")
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_text": self.source_text,
            "translation": self.translation,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "domain": self.domain,
            "priority": self.priority,
            "status": self.status,
            "score": self.score,
            "tx_hash": self.tx_hash,
            "translator": self.translator.to_dict() if self.translator else None,
            "validations": [v.to_dict() for v in self.validations]
        }

class Translator(Base):
    """Translator profile model"""
    
    __tablename__ = "translators"
    
    id = Column(PGUUID, primary_key=True, default=uuid4)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Profile
    address = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    languages = Column(JSON, nullable=False, default=list)
    domains = Column(JSON, nullable=False, default=list)
    
    # Stats
    reputation = Column(Float, nullable=False, default=0.0)
    total_translations = Column(Integer, nullable=False, default=0)
    total_validations = Column(Integer, nullable=False, default=0)
    stake = Column(Integer, nullable=False, default=0)
    rewards = Column(Integer, nullable=False, default=0)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    is_verified = Column(Boolean, nullable=False, default=False)
    
    # Relationships
    translations = relationship("Translation", back_populates="translator")
    validations = relationship("Validation", back_populates="validator")
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "address": self.address,
            "name": self.name,
            "bio": self.bio,
            "languages": self.languages,
            "domains": self.domains,
            "reputation": self.reputation,
            "total_translations": self.total_translations,
            "total_validations": self.total_validations,
            "stake": self.stake,
            "rewards": self.rewards,
            "is_active": self.is_active,
            "is_verified": self.is_verified
        }

class Validation(Base):
    """Translation validation record model"""
    
    __tablename__ = "validations"
    
    id = Column(PGUUID, primary_key=True, default=uuid4)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Validation details
    translation_id = Column(PGUUID, ForeignKey("translations.id"), nullable=False)
    validator_id = Column(PGUUID, ForeignKey("translators.id"), nullable=False)
    score = Column(Float, nullable=False)
    feedback = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending")
    
    # Relationships
    translation = relationship("Translation", back_populates="validations")
    validator = relationship("Translator", back_populates="validations")
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "translation_id": str(self.translation_id),
            "validator_id": str(self.validator_id),
            "score": self.score,
            "feedback": self.feedback,
            "status": self.status,
            "validator": self.validator.to_dict()
        } 