"""
OpenTranslate API Server
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from ..database import get_db
from ..core.translation import TranslationEngine
from ..models.translation import Translation, Translator, Validation
from . import schemas

app = FastAPI(
    title="OpenTranslate API",
    description="API for the OpenTranslate decentralized translation platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize translation engine
engine = TranslationEngine()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to OpenTranslate API"}

@app.post("/translations/", response_model=schemas.Translation)
def create_translation(
    translation: schemas.TranslationCreate,
    db: Session = Depends(get_db)
):
    """Create a new translation task"""
    result = engine.translate(
        text=translation.source_text,
        source_lang=translation.source_lang,
        target_lang=translation.target_lang,
        domain=translation.domain
    )
    return result

@app.get("/translations/", response_model=List[schemas.Translation])
def list_translations(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all translations"""
    translations = db.query(Translation).offset(skip).limit(limit).all()
    return translations

@app.get("/translations/{translation_id}", response_model=schemas.Translation)
def get_translation(translation_id: str, db: Session = Depends(get_db)):
    """Get a specific translation"""
    translation = db.query(Translation).filter(Translation.id == translation_id).first()
    if not translation:
        raise HTTPException(status_code=404, detail="Translation not found")
    return translation

@app.post("/translators/", response_model=schemas.Translator)
def create_translator(
    translator: schemas.TranslatorCreate,
    db: Session = Depends(get_db)
):
    """Register a new translator"""
    db_translator = Translator(**translator.dict())
    db.add(db_translator)
    db.commit()
    db.refresh(db_translator)
    return db_translator

@app.get("/translators/", response_model=List[schemas.Translator])
def list_translators(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all translators"""
    translators = db.query(Translator).offset(skip).limit(limit).all()
    return translators

@app.post("/validations/", response_model=schemas.Validation)
def create_validation(
    validation: schemas.ValidationCreate,
    db: Session = Depends(get_db)
):
    """Create a new validation"""
    db_validation = Validation(**validation.dict())
    db.add(db_validation)
    db.commit()
    db.refresh(db_validation)
    return db_validation

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get platform statistics"""
    total_translations = db.query(Translation).count()
    total_translators = db.query(Translator).count()
    total_validations = db.query(Validation).count()
    
    return {
        "total_translations": total_translations,
        "total_translators": total_translators,
        "total_validations": total_validations
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"} 