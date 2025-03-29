"""
Translation model implementation
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import Dict, List, Optional

class TranslationModel:
    """Neural machine translation model using MarianMT"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.models: Dict[str, MarianMTModel] = {}
        self.tokenizers: Dict[str, MarianTokenizer] = {}
        
    def load_model(self, source_lang: str, target_lang: str) -> None:
        """Load translation model for a language pair"""
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        if f"{source_lang}-{target_lang}" not in self.models:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            
            self.models[f"{source_lang}-{target_lang}"] = model
            self.tokenizers[f"{source_lang}-{target_lang}"] = tokenizer
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
        domain: Optional[str] = None
    ) -> str:
        """Translate text from source language to target language"""
        model_key = f"{source_lang}-{target_lang}"
        
        if model_key not in self.models:
            self.load_model(source_lang, target_lang)
            
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        
        # Add domain tag if specified
        if domain:
            text = f">>{domain}<< {text}"
        
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translation
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        # Decode and return translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language pairs"""
        return [
            "en", "zh", "ja", "ko", "fr", "de", "es", "ru",
            "it", "pt", "nl", "pl", "ar", "hi", "vi", "th"
        ]
    
    def get_supported_domains(self) -> List[str]:
        """Get list of supported domains"""
        return [
            "general",
            "academic",
            "technical",
            "legal",
            "medical",
            "business",
            "news"
        ]
        
    def clear_cache(self) -> None:
        """Clear loaded models from memory"""
        self.models.clear()
        self.tokenizers.clear()
        torch.cuda.empty_cache() 