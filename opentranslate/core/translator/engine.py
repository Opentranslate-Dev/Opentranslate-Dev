"""
OpenTranslate Translation Engine Implementation
"""

from typing import Dict, List, Optional, Union
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..blockchain import Blockchain
from ..validator import Validator
from ...ai.models import TranslationModel

class TranslationEngine:
    """OpenTranslate Core Translation Engine"""
    
    def __init__(
        self,
        model_name: str = "opentranslate/translator-v1",
        blockchain: Optional[Blockchain] = None,
        validator: Optional[Validator] = None,
    ):
        """
        Initialize translation engine
        
        Args:
            model_name: Model name to use
            blockchain: Blockchain instance
            validator: Validator instance
        """
        self.model_name = model_name
        self.blockchain = blockchain
        self.validator = validator
        
        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def translate(
        self,
        text: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        domain: Optional[str] = None,
        priority: str = "normal",
    ) -> Dict:
        """
        Translate text
        
        Args:
            text: Text or list of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Text domain (e.g. physics, biology)
            priority: Translation priority
            
        Returns:
            Dictionary containing translation results
        """
        # Preprocess input
        if isinstance(text, str):
            text = [text]
            
        # Add domain-specific marker
        if domain:
            text = [f"[{domain}] {t}" for t in text]
            
        # Batch translate
        results = []
        for t in text:
            # Encode input
            inputs = self.tokenizer(
                t,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode output
            translation = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            results.append(translation)
            
        # Build result
        response = {
            "translations": results,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": self.model_name,
            "domain": domain,
            "priority": priority
        }
        
        # If blockchain is enabled, record translation
        if self.blockchain:
            self.blockchain.record_translation(response)
            
        # If validator is enabled, perform validation
        if self.validator:
            validation_result = self.validator.validate_translation(response)
            response["validation"] = validation_result
            
        return response
        
    def batch_translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 32,
        **kwargs
    ) -> List[Dict]:
        """
        Batch translate text
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Batch processing size
            **kwargs: Other parameters
            
        Returns:
            List of translation results
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self.translate(
                batch,
                source_lang,
                target_lang,
                **kwargs
            )
            results.extend(result["translations"])
        return results 