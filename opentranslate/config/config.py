"""
Configuration settings for OpenTranslate.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project metadata
PROJECT_NAME = "OpenTranslate"
PROJECT_VERSION = "0.1.0"
PROJECT_AUTHOR = "OpenTranslateDev"
PROJECT_WEBSITE = "https://opentranslate.world"
PROJECT_TWITTER = "@_OpenTranslate"

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///opentranslate.db")
DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "5"))
DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Quantum computing settings
QUANTUM_BACKEND = os.getenv("QUANTUM_BACKEND", "aer_simulator")
QUANTUM_SHOTS = int(os.getenv("QUANTUM_SHOTS", "1024"))
QUANTUM_OPTIMIZATION_LEVEL = int(os.getenv("QUANTUM_OPTIMIZATION_LEVEL", "2"))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "opentranslate.log"

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600  # 1 hour

# Rate limiting
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "100"))  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Cache settings
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# Monitoring settings
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "True").lower() == "true"
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))

# Blockchain settings
BLOCKCHAIN_NETWORK = os.getenv("BLOCKCHAIN_NETWORK", "mainnet")
BLOCKCHAIN_RPC_URL = os.getenv("BLOCKCHAIN_RPC_URL", "")
BLOCKCHAIN_CONTRACT_ADDRESS = os.getenv("BLOCKCHAIN_CONTRACT_ADDRESS", "")

# Translation settings
DEFAULT_SOURCE_LANGUAGE = os.getenv("DEFAULT_SOURCE_LANGUAGE", "en")
DEFAULT_TARGET_LANGUAGE = os.getenv("DEFAULT_TARGET_LANGUAGE", "zh")
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "5000"))

# Worker settings
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "4"))
WORKER_TIMEOUT = int(os.getenv("WORKER_TIMEOUT", "300"))  # 5 minutes

def get_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary."""
    return {
        key: value
        for key, value in globals().items()
        if not key.startswith("_") and isinstance(value, (str, int, bool, Path))
    } 