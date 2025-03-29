"""
Translation validation model implementation
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Optional, Tuple

class ValidationModel:
    """Model for validating translation quality"""
    
    def __init__(
        self,
        model_name: str = "microsoft/infoxlm-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1  # Regression task
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def validate(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate translation quality
        
        Args:
            source_text: Original text
            translation: Translated text
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (quality score, detailed metrics)
        """
        # Prepare input
        inputs = self.tokenizer(
            source_text,
            translation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(source_text, translation)
        
        return score, metrics
    
    def batch_validate(
        self,
        source_texts: List[str],
        translations: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 32
    ) -> List[Tuple[float, Dict[str, float]]]:
        """
        Validate multiple translations
        
        Args:
            source_texts: List of original texts
            translations: List of translated texts
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Batch size for processing
            
        Returns:
            List of (score, metrics) tuples
        """
        results = []
        for i in range(0, len(source_texts), batch_size):
            batch_sources = source_texts[i:i + batch_size]
            batch_translations = translations[i:i + batch_size]
            
            # Process batch
            inputs = self.tokenizer(
                batch_sources,
                batch_translations,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # Calculate metrics for each pair
            for j, (source, translation) in enumerate(zip(batch_sources, batch_translations)):
                metrics = self._calculate_metrics(source, translation)
                results.append((float(scores[j]), metrics))
        
        return results
    
    def _calculate_metrics(self, source_text: str, translation: str) -> Dict[str, float]:
        """Calculate detailed quality metrics"""
        metrics = {
            "length_ratio": len(translation) / len(source_text),
            "source_length": len(source_text),
            "translation_length": len(translation)
        }
        return metrics
    
    def get_threshold_recommendations(self) -> Dict[str, float]:
        """Get recommended quality thresholds for different use cases"""
        return {
            "high_quality": 0.8,
            "acceptable": 0.6,
            "needs_review": 0.4
        }
    
    @staticmethod
    def get_supported_metrics() -> List[str]:
        """Get list of supported quality metrics"""
        return [
            "overall_score",
            "length_ratio",
            "source_length",
            "translation_length"
        ] 