"""
Quantum Translation API endpoints for OpenTranslate.

This module provides API endpoints for quantum-enhanced translation features.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Optional, List
from pydantic import BaseModel
import numpy as np
from datetime import datetime

from opentranslate.ai.quantum import (
    QuantumTranslator,
    QuantumContextAwareness,
    QuantumSecurityLayer,
    QuantumComputingManager,
    QuantumErrorCorrection
)

router = APIRouter(
    prefix="/quantum",
    tags=["quantum"],
    responses={404: {"description": "Not found"}},
)

class TranslationRequest(BaseModel):
    """Request model for quantum translation."""
    source_text: str
    target_language: str
    context: Optional[Dict] = None
    use_quantum: bool = True
    backend: Optional[str] = None
    error_correction: bool = False
    visualize: bool = False
    visualization_style: Optional[str] = "default"
    monitor: bool = True
    use_ml: bool = True
    optimize: bool = False
    optimization_level: int = 3
    secure: bool = True

class TranslationResponse(BaseModel):
    """Response model for quantum translation."""
    translated_text: str
    quantum_enhanced: bool
    backend_used: str
    error_correction: Dict
    confidence_score: float
    quality_score: Optional[float] = None
    security_score: Optional[float] = None
    authenticated: Optional[bool] = None
    visualization_urls: Optional[Dict[str, str]] = None
    performance_metrics: Optional[Dict] = None
    ml_metrics: Optional[Dict] = None
    optimization_metrics: Optional[Dict] = None
    security_metrics: Optional[Dict] = None

class QuantumOptimizationRequest(BaseModel):
    """Request model for quantum optimization."""
    source_embedding: List[float]
    target_embedding: List[float]
    backend: Optional[str] = None
    error_correction: bool = False
    visualize: bool = False
    visualization_style: Optional[str] = "default"
    monitor: bool = True
    use_ml: bool = True
    optimize: bool = False
    optimization_level: int = 3
    secure: bool = True

class QuantumOptimizationResponse(BaseModel):
    """Response model for quantum optimization."""
    optimized_embedding: List[float]
    backend_used: str
    error_correction: Dict
    optimization_score: float
    quality_score: Optional[float] = None
    security_score: Optional[float] = None
    authenticated: Optional[bool] = None
    visualization_urls: Optional[Dict[str, str]] = None
    performance_metrics: Optional[Dict] = None
    ml_metrics: Optional[Dict] = None
    optimization_metrics: Optional[Dict] = None
    security_metrics: Optional[Dict] = None

class MLTrainingRequest(BaseModel):
    """Request model for quantum ML training."""
    features: List[List[float]]
    labels: List[float]
    task: str = "both"  # "classifier", "optimizer", or "both"
    error_correction: bool = False
    optimize: bool = False
    distributed: bool = False
    visualize: bool = False
    monitor: bool = True
    secure: bool = True

class MLTrainingResponse(BaseModel):
    """Response model for quantum ML training."""
    status: str
    task: str
    training_metrics: Dict
    error_correction: Dict
    optimization: Dict
    visualization: Optional[str] = None
    performance_metrics: Optional[Dict] = None
    security: Dict

class MLEvaluationRequest(BaseModel):
    """Request model for quantum ML evaluation."""
    features: List[List[float]]
    labels: List[float]
    error_correction: bool = False
    optimize: bool = False
    distributed: bool = False
    visualize: bool = False
    monitor: bool = True
    secure: bool = True

class MLEvaluationResponse(BaseModel):
    """Response model for quantum ML evaluation."""
    classifier_metrics: Dict
    optimizer_metrics: Dict
    error_correction: Dict
    optimization: Dict
    visualization: Optional[str] = None
    performance_metrics: Optional[Dict] = None
    security: Dict

class SecurityRequest(BaseModel):
    """Request model for quantum security operations."""
    text: str
    key: Optional[List[int]] = None
    visualize: bool = False
    monitor: bool = True

class SecurityResponse(BaseModel):
    """Response model for quantum security operations."""
    result: str
    key: Optional[List[int]] = None
    error_correction: Dict
    optimization: Dict
    visualization_urls: Optional[Dict[str, str]] = None
    performance_metrics: Optional[Dict] = None
    security_score: float

class PerformanceReportRequest(BaseModel):
    """Request model for performance report."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    include_ml: bool = True
    include_optimization: bool = True
    include_security: bool = True
    visualize: bool = False
    monitor: bool = True

class PerformanceReportResponse(BaseModel):
    """Response model for performance report."""
    total_translations: int
    total_error_corrections: int
    average_execution_time: float
    average_accuracy: float
    quantum_enhancement_rate: float
    error_correction_success_rate: float
    average_quantum_advantage: float
    ml_metrics: Optional[Dict] = None
    optimization_metrics: Optional[Dict] = None
    security_metrics: Optional[Dict] = None
    visualization_urls: Dict[str, str]

# Initialize quantum components
quantum_translator = QuantumTranslator()
quantum_context = QuantumContextAwareness()
quantum_security = QuantumSecurityLayer()
quantum_computing = QuantumComputingManager()

@router.post("/translate", response_model=TranslationResponse)
async def quantum_translate(request: TranslationRequest):
    """
    Translate text using quantum-enhanced algorithms.
    
    Args:
        request: Translation request containing source text and target language
        
    Returns:
        Translated text with quantum enhancement information
    """
    try:
        # Set up quantum components
        quantum_translator.context_awareness = quantum_context
        quantum_translator.security_layer = quantum_security
        
        # Set visualization style
        if request.visualize:
            quantum_translator.set_visualization_style(request.visualization_style)
        
        # Initialize quantum backend if specified
        if request.use_quantum and request.backend:
            quantum_computing.initialize_backend(request.backend)
            
        # Set up error correction
        quantum_translator.error_correction = QuantumErrorCorrection(
            code_type=request.error_correction
        )
        
        # Perform translation
        translation = quantum_translator.translate(
            source_text=request.source_text,
            target_language=request.target_language,
            context=request.context,
            visualize=request.visualize,
            monitor=request.monitor,
            optimize=request.optimize,
            optimization_level=request.optimization_level,
            secure=request.secure
        )
        
        # Calculate confidence score
        confidence = 0.9 if request.use_quantum else 0.7
        
        # Generate visualization URLs if requested
        visualization_urls = None
        if request.visualize:
            visualization_urls = {
                "circuit": "/visualizations/circuit.png",
                "state": "/visualizations/state.png",
                "histogram": "/visualizations/histogram.png",
                "process": "/visualizations/process.html",
                "optimization": "/visualizations/optimization.png",
                "security": "/visualizations/security.png"
            }
            
        # Get performance metrics if monitoring
        performance_metrics = None
        if request.monitor:
            performance_metrics = quantum_translator.get_performance_report()
            
        # Get ML metrics if using ML
        ml_metrics = None
        if request.use_ml:
            ml_metrics = quantum_translator.get_ml_performance_report()
            
        # Get optimization metrics if optimizing
        optimization_metrics = None
        if request.optimize:
            optimization_metrics = quantum_translator.get_optimization_report()
            
        # Get security metrics if using security
        security_metrics = None
        if request.secure:
            security_metrics = quantum_translator.get_security_report()
        
        return TranslationResponse(
            translated_text=translation,
            quantum_enhanced=request.use_quantum,
            backend_used=request.backend or "classical",
            error_correction=quantum_translator.error_correction.to_dict(),
            confidence_score=confidence,
            quality_score=quantum_translator.quantum_ml.predict_quality(
                np.array([request.source_text])
            )[0] if request.use_ml else None,
            security_score=quantum_translator.security_layer.authenticate_translation(
                request.source_text,
                translation
            )[1] if request.secure else None,
            authenticated=quantum_translator.security_layer.authenticate_translation(
                request.source_text,
                translation
            )[0] if request.secure else None,
            visualization_urls=visualization_urls,
            performance_metrics=performance_metrics,
            ml_metrics=ml_metrics,
            optimization_metrics=optimization_metrics,
            security_metrics=security_metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize", response_model=QuantumOptimizationResponse)
async def quantum_optimize(request: QuantumOptimizationRequest):
    """
    Optimize translation embeddings using quantum computing.
    
    Args:
        request: Optimization request containing source and target embeddings
        
    Returns:
        Optimized embedding with quantum computing information
    """
    try:
        # Convert embeddings to numpy arrays
        source_embedding = np.array(request.source_embedding)
        target_embedding = np.array(request.target_embedding)
        
        # Set visualization style
        if request.visualize:
            quantum_computing.set_visualization_style(request.visualization_style)
        
        # Initialize quantum backend if specified
        if request.backend:
            quantum_computing.initialize_backend(request.backend)
            
        # Set up error correction
        quantum_computing.backends[request.backend or "ibm_quantum"].error_correction = (
            QuantumErrorCorrection(code_type=request.error_correction)
        )
        
        # Perform quantum optimization
        optimized_embedding = quantum_computing.optimize_translation(
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            backend_name=request.backend,
            visualize=request.visualize,
            monitor=request.monitor,
            use_ml=request.use_ml,
            optimize=request.optimize,
            optimization_level=request.optimization_level
        )
        
        # Calculate optimization score
        optimization_score = np.mean(
            np.abs(optimized_embedding - source_embedding)
        )
        
        # Generate visualization URLs if requested
        visualization_urls = None
        if request.visualize:
            visualization_urls = {
                "circuit": "/visualizations/optimization_circuit.png",
                "state": "/visualizations/optimization_state.png",
                "process": "/visualizations/optimization_process.html",
                "optimization": "/visualizations/optimization.png",
                "security": "/visualizations/security.png"
            }
            
        # Get performance metrics if monitoring
        performance_metrics = None
        if request.monitor:
            performance_metrics = quantum_computing.get_performance_report()
            
        # Get ML metrics if using ML
        ml_metrics = None
        if request.use_ml:
            ml_metrics = quantum_translator.get_ml_performance_report()
            
        # Get optimization metrics if optimizing
        optimization_metrics = None
        if request.optimize:
            optimization_metrics = quantum_computing.get_optimization_report()
            
        # Get security metrics if using security
        security_metrics = None
        if request.secure:
            security_metrics = quantum_translator.get_security_report()
        
        return QuantumOptimizationResponse(
            optimized_embedding=optimized_embedding.tolist(),
            backend_used=request.backend or "classical",
            error_correction=quantum_computing.backends[request.backend or "ibm_quantum"].error_correction.to_dict(),
            optimization_score=float(optimization_score),
            quality_score=quantum_translator.quantum_ml.predict_quality(
                np.array([optimized_embedding])
            )[0] if request.use_ml else None,
            security_score=quantum_translator.security_layer.authenticate_translation(
                str(source_embedding),
                str(optimized_embedding)
            )[1] if request.secure else None,
            authenticated=quantum_translator.security_layer.authenticate_translation(
                str(source_embedding),
                str(optimized_embedding)
            )[0] if request.secure else None,
            visualization_urls=visualization_urls,
            performance_metrics=performance_metrics,
            ml_metrics=ml_metrics,
            optimization_metrics=optimization_metrics,
            security_metrics=security_metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ml/train", response_model=MLTrainingResponse)
async def train_quantum_ml(request: MLTrainingRequest):
    """
    Train quantum machine learning models.
    
    Args:
        request: Training request containing features and labels
        
    Returns:
        Training status and metrics
    """
    try:
        # Convert features and labels to numpy arrays
        X = np.array(request.features)
        y = np.array(request.labels)
        
        # Train models
        quantum_translator.train_quantum_ml(X, y, request.task)
        
        # Get training metrics
        training_metrics = quantum_translator.evaluate_quantum_ml(X, y)
        
        # Get performance metrics
        performance_metrics = quantum_translator.get_ml_performance_report()
        
        return MLTrainingResponse(
            status="success",
            task=request.task,
            training_metrics=training_metrics,
            error_correction=quantum_translator.error_correction.to_dict(),
            optimization=quantum_translator.optimization_report,
            visualization=None,
            performance_metrics=performance_metrics,
            security=quantum_translator.security_layer.get_metrics()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ml/evaluate", response_model=MLEvaluationResponse)
async def evaluate_quantum_ml(request: MLEvaluationRequest):
    """
    Evaluate quantum machine learning models.
    
    Args:
        request: Evaluation request containing features and labels
        
    Returns:
        Evaluation metrics
    """
    try:
        # Convert features and labels to numpy arrays
        X = np.array(request.features)
        y = np.array(request.labels)
        
        # Evaluate models
        metrics = quantum_translator.evaluate_quantum_ml(X, y)
        
        # Get performance metrics
        performance_metrics = quantum_translator.get_ml_performance_report()
        
        return MLEvaluationResponse(
            classifier_metrics=metrics['classifier'],
            optimizer_metrics=metrics['optimizer'],
            error_correction=quantum_translator.error_correction.to_dict(),
            optimization=quantum_translator.optimization_report,
            visualization=None,
            performance_metrics=performance_metrics,
            security=quantum_translator.security_layer.get_metrics()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/security/encrypt", response_model=SecurityResponse)
async def encrypt_text(request: SecurityRequest):
    """
    Encrypt text using quantum key.
    
    Args:
        request: Security request containing text to encrypt
        
    Returns:
        Encrypted text and key
    """
    try:
        # Convert key to numpy array if provided
        key = np.array(request.key) if request.key else None
        
        # Encrypt text
        encrypted_text, key = quantum_security.encrypt_text(
            request.text,
            key=key,
            visualize=request.visualize,
            monitor=request.monitor
        )
        
        # Generate visualization URLs if requested
        visualization_urls = None
        if request.visualize:
            visualization_urls = {
                "circuit": "/visualizations/security_circuit.png",
                "process": "/visualizations/security_process.html"
            }
            
        # Get performance metrics if monitoring
        performance_metrics = None
        if request.monitor:
            performance_metrics = quantum_security.get_security_report()
            
        return SecurityResponse(
            result=encrypted_text,
            key=key.tolist() if key is not None else None,
            error_correction=quantum_security.error_correction.to_dict(),
            optimization=quantum_security.optimization_report,
            visualization_urls=visualization_urls,
            performance_metrics=performance_metrics,
            security_score=1.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/security/decrypt", response_model=SecurityResponse)
async def decrypt_text(request: SecurityRequest):
    """
    Decrypt text using quantum key.
    
    Args:
        request: Security request containing text to decrypt and key
        
    Returns:
        Decrypted text
    """
    try:
        if request.key is None:
            raise HTTPException(
                status_code=400,
                detail="Key is required for decryption"
            )
            
        # Convert key to numpy array
        key = np.array(request.key)
        
        # Decrypt text
        decrypted_text = quantum_security.decrypt_text(
            request.text,
            key=key,
            visualize=request.visualize,
            monitor=request.monitor
        )
        
        # Generate visualization URLs if requested
        visualization_urls = None
        if request.visualize:
            visualization_urls = {
                "circuit": "/visualizations/security_circuit.png",
                "process": "/visualizations/security_process.html"
            }
            
        # Get performance metrics if monitoring
        performance_metrics = None
        if request.monitor:
            performance_metrics = quantum_security.get_security_report()
            
        return SecurityResponse(
            result=decrypted_text,
            key=key.tolist(),
            error_correction=quantum_security.error_correction.to_dict(),
            optimization=quantum_security.optimization_report,
            visualization_urls=visualization_urls,
            performance_metrics=performance_metrics,
            security_score=1.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/security/authenticate", response_model=SecurityResponse)
async def authenticate_text(request: SecurityRequest):
    """
    Authenticate text using quantum authentication.
    
    Args:
        request: Security request containing text to authenticate
        
    Returns:
        Authentication result
    """
    try:
        # Convert key to numpy array if provided
        key = np.array(request.key) if request.key else None
        
        # Authenticate text
        authenticated, security_score = quantum_security.authenticate_translation(
            request.text,
            request.text,  # Compare with itself for authentication
            key=key,
            visualize=request.visualize,
            monitor=request.monitor
        )
        
        # Generate visualization URLs if requested
        visualization_urls = None
        if request.visualize:
            visualization_urls = {
                "circuit": "/visualizations/security_circuit.png",
                "process": "/visualizations/security_process.html"
            }
            
        # Get performance metrics if monitoring
        performance_metrics = None
        if request.monitor:
            performance_metrics = quantum_security.get_security_report()
            
        return SecurityResponse(
            result="authenticated" if authenticated else "not_authenticated",
            key=key.tolist() if key is not None else None,
            error_correction=quantum_security.error_correction.to_dict(),
            optimization=quantum_security.optimization_report,
            visualization_urls=visualization_urls,
            performance_metrics=performance_metrics,
            security_score=security_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backends")
async def list_quantum_backends():
    """
    List available quantum computing backends.
    
    Returns:
        List of available quantum backends
    """
    return {
        "available_backends": [
            "ibm_quantum",
            "google_quantum",
            "dwave"
        ],
        "default_backend": "ibm_quantum",
        "error_correction_codes": [
            "surface",
            "stabilizer",
            "repetition"
        ],
        "visualization_styles": [
            "default",
            "dark"
        ],
        "ml_models": [
            "classifier",
            "optimizer"
        ],
        "optimization_levels": [
            0, 1, 2, 3
        ],
        "security_features": [
            "encryption",
            "decryption",
            "authentication"
        ]
    }

@router.post("/initialize/{backend}")
async def initialize_quantum_backend(
    backend: str,
    api_token: Optional[str] = None,
    error_correction: Optional[str] = "surface",
    visualization_style: Optional[str] = "default"
):
    """
    Initialize a quantum computing backend.
    
    Args:
        backend: Name of the backend to initialize
        api_token: Optional API token for the backend
        error_correction: Type of error correction to use
        visualization_style: Style for quantum visualizations
        
    Returns:
        Status of backend initialization
    """
    try:
        quantum_computing.initialize_backend(backend, api_token)
        quantum_computing.backends[backend].error_correction = (
            QuantumErrorCorrection(code_type=error_correction)
        )
        quantum_computing.set_visualization_style(visualization_style)
        return {
            "status": "success",
            "backend": backend,
            "error_correction": error_correction,
            "visualization_style": visualization_style
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/report", response_model=PerformanceReportResponse)
async def get_performance_report(request: PerformanceReportRequest):
    """
    Get a performance report for quantum translation operations.
    
    Args:
        request: Performance report request with optional time range
        
    Returns:
        Performance report with metrics and visualizations
    """
    try:
        # Get performance report
        report = quantum_translator.get_performance_report(
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        # Get ML metrics if requested
        ml_metrics = None
        if request.include_ml:
            ml_metrics = quantum_translator.get_ml_performance_report(
                start_time=request.start_time,
                end_time=request.end_time
            )
            
        # Get optimization metrics if requested
        optimization_metrics = None
        if request.include_optimization:
            optimization_metrics = quantum_translator.get_optimization_report(
                start_time=request.start_time,
                end_time=request.end_time
            )
            
        # Get security metrics if requested
        security_metrics = None
        if request.include_security:
            security_metrics = quantum_translator.get_security_report(
                start_time=request.start_time,
                end_time=request.end_time
            )
        
        # Generate visualization URLs
        visualization_urls = {
            "translation_accuracy": "/figures/translation_accuracy.png",
            "error_correction": "/figures/error_correction.png",
            "quantum_advantage": "/figures/quantum_advantage.png",
            "optimization": "/figures/optimization.png",
            "security": "/figures/security.png"
        }
        
        return PerformanceReportResponse(
            total_translations=report['total_translations'],
            total_error_corrections=report['total_error_corrections'],
            average_execution_time=report['average_execution_time'],
            average_accuracy=report['average_accuracy'],
            quantum_enhancement_rate=report['quantum_enhancement_rate'],
            error_correction_success_rate=report['error_correction_success_rate'],
            average_quantum_advantage=report['average_quantum_advantage'],
            ml_metrics=ml_metrics,
            optimization_metrics=optimization_metrics,
            security_metrics=security_metrics,
            visualization_urls=visualization_urls
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 