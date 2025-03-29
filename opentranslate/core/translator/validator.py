"""
OpenTranslate Translation Validator
"""

from typing import Dict, List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .protocol import TranslationResponse, TranslationStatus

class TranslationValidator:
    """OpenTranslate Translation Validator"""
    
    def __init__(
        self,
        model_name: str = "opentranslate/validator-v1",
        threshold: float = 0.8,
    ):
        """
        Initialize validator
        
        Args:
            model_name: Model name to use
            threshold: Validation threshold
        """
        self.model_name = model_name
        self.threshold = threshold
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def validate_translation(
        self,
        response: TranslationResponse,
    ) -> Dict:
        """
        Validate translation results
        
        Args:
            response: Translation response
            
        Returns:
            Validation result dictionary
        """
        # Prepare input
        source_texts = response.metadata.get("source_texts", [])
        translations = response.translations
        
        if not source_texts:
            return {
                "status": TranslationStatus.FAILED,
                "score": 0.0,
                "error": "Missing source texts"
            }
            
        # Validate each translation
        validation_results = []
        for source, target in zip(source_texts, translations):
            # Encode input
            inputs = self.tokenizer(
                source,
                target,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get validation scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                score = scores[0][1].item()
                
            validation_results.append({
                "source": source,
                "target": target,
                "score": score,
                "passed": score >= self.threshold
            })
            
        # Calculate overall score
        total_score = sum(r["score"] for r in validation_results) / len(validation_results)
        all_passed = all(r["passed"] for r in validation_results)
        
        # Determine status
        if all_passed:
            status = TranslationStatus.VALIDATED
        elif total_score > 0.5:
            status = TranslationStatus.VALIDATING
        else:
            status = TranslationStatus.REJECTED
            
        return {
            "status": status,
            "score": total_score,
            "details": validation_results,
            "threshold": self.threshold
        }
        
    def batch_validate(
        self,
        responses: List[TranslationResponse],
    ) -> List[Dict]:
        """
        Batch validate translation results
        
        Args:
            responses: List of translation responses
            
        Returns:
            Validation result list
        """
        return [
            self.validate_translation(response)
            for response in responses
        ] 