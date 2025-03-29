"""
OpenTranslate - Revolutionary Scientific Translation Platform
"""

__version__ = "0.1.0"
__author__ = "OpenTranslate Team"
__email__ = "contact@opentranslate.org"

from .core.translator import Translator
from .core.validator import Validator
from .core.blockchain import Blockchain
from .ai.models import TranslationModel
from .api import APIServer
from .web import WebServer

__all__ = [
    "Translator",
    "Validator",
    "Blockchain",
    "TranslationModel",
    "APIServer",
    "WebServer",
] 