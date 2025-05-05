"""
OpenTranslate - Decentralized Multilingual Translation Network for Scientific Knowledge
"""

__version__ = "0.1.0"
__author__ = "OpenTranslate Team"
__email__ = "contact@opentranslate.world"
__license__ = "MIT"
__copyright__ = "Copyright 2024 OpenTranslate Team"

from .core.translator import TranslationEngine
from .core.blockchain import BlockchainManager
from .core.validator import ValidationEngine

__all__ = ["TranslationEngine", "BlockchainManager", "ValidationEngine"] 