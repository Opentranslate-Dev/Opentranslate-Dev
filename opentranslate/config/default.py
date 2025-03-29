"""
OpenTranslate Default Configuration
"""

from typing import Dict, List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """OpenTranslate Configuration"""
    
    # Project Information
    PROJECT_NAME: str = "OpenTranslate"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Decentralized Multilingual Translation Network for Scientific Knowledge"
    API_V1_STR: str = "/api/v1"
    
    # Blockchain Configuration
    BLOCKCHAIN_PROVIDER: str = "http://localhost:8545"
    CONTRACT_ADDRESS: str = Field("", description="Translation Smart Contract Address")
    TOKEN_ADDRESS: str = Field("", description="PUMPFUN Token Contract Address")
    CHAIN_ID: int = 1
    GAS_LIMIT: int = 3000000
    
    # AI Model Configuration
    TRANSLATION_MODEL: str = "opentranslate/translator-v1"
    VALIDATION_MODEL: str = "opentranslate/validator-v1"
    CLASSIFIER_MODEL: str = "opentranslate/classifier-v1"
    DEVICE: Optional[str] = None
    MODEL_CACHE_DIR: str = "models"
    
    # Validation Configuration
    VALIDATION_THRESHOLD: float = 0.8
    MIN_VALIDATORS: int = 3
    VALIDATION_TIMEOUT: int = 24 * 3600  # 24 hours
    BATCH_SIZE: int = 32
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # Web Configuration
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 8001
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./opentranslate.db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    
    # Cache Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600  # 1 hour
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DIR: str = "logs"
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # File Upload Configuration
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".md", ".tex"]
    
    # Supported Languages
    SUPPORTED_LANGUAGES: Dict[str, str] = {
        "en": "English",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "ar": "Arabic",
        "pt": "Portuguese",
        "vi": "Vietnamese",
        "bn": "Bengali"
    }
    
    # Supported Subject Areas
    SUPPORTED_DOMAINS: Dict[str, str] = {
        "physics": "Physics",
        "biology": "Biology",
        "chemistry": "Chemistry",
        "mathematics": "Mathematics",
        "computer_science": "Computer Science",
        "medicine": "Medicine",
        "engineering": "Engineering",
        "earth_science": "Earth Science",
        "astronomy": "Astronomy",
        "psychology": "Psychology",
        "sociology": "Sociology",
        "economics": "Economics",
        "other": "Other"
    }
    
    # Translation Priority Levels
    PRIORITY_LEVELS: Dict[str, int] = {
        "low": 0,
        "normal": 1,
        "high": 2,
        "urgent": 3
    }
    
    # Token Economy Parameters
    TOKEN_DECIMALS: int = 18
    TOTAL_SUPPLY: int = 1_000_000_000  # 10 billion tokens
    BURN_RATE: float = 0.001  # 0.1% burn rate
    MIN_STAKE: int = 1000  # Minimum stake amount
    REWARD_PER_TRANSLATION: int = 10  # Base reward for each translation
    REWARD_PER_VALIDATION: int = 2  # Base reward for each validation
    
    # System Parameters
    MAX_CONCURRENT_TASKS: int = 100
    TASK_TIMEOUT: int = 3600  # 1 hour
    RETRY_LIMIT: int = 3
    HEALTH_CHECK_INTERVAL: int = 60  # 1 minute
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 