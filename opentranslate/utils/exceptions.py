"""
Custom exceptions for OpenTranslate
"""

class OpenTranslateError(Exception):
    """Base exception for OpenTranslate"""
    pass

class InvalidLanguageError(OpenTranslateError):
    """Raised when an unsupported language is specified"""
    pass

class InvalidDomainError(OpenTranslateError):
    """Raised when an unsupported domain is specified"""
    pass

class TranslationError(OpenTranslateError):
    """Raised when translation fails"""
    pass

class ValidationError(OpenTranslateError):
    """Raised when validation fails"""
    pass

class BlockchainError(OpenTranslateError):
    """Raised when blockchain operations fail"""
    pass

class ModelError(OpenTranslateError):
    """Raised when AI model operations fail"""
    pass

class AuthenticationError(OpenTranslateError):
    """Raised when authentication fails"""
    pass

class AuthorizationError(OpenTranslateError):
    """Raised when authorization fails"""
    pass

class RateLimitError(OpenTranslateError):
    """Raised when rate limit is exceeded"""
    pass

class ConfigurationError(OpenTranslateError):
    """Raised when configuration is invalid"""
    pass

class DatabaseError(OpenTranslateError):
    """Raised when database operations fail"""
    pass

class CacheError(OpenTranslateError):
    """Raised when cache operations fail"""
    pass

class APIError(OpenTranslateError):
    """Raised when API operations fail"""
    pass

class WebError(OpenTranslateError):
    """Raised when web operations fail"""
    pass 