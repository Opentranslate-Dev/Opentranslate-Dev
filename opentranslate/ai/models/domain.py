"""
Domain classification model implementation
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Optional, Tuple

class DomainClassifier:
    """Model for classifying text domains"""
    
    DOMAINS = [
        "general",
        "academic",
        "technical",
        "legal",
        "medical",
        "business",
        "news"
    ]
    
    def __init__(
        self,
        model_name: str = "microsoft/mdeberta-v3-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.DOMAINS)
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def classify(
        self,
        text: str,
        max_length: int = 512
    ) -> Tuple[str, Dict[str, float]]:
        """
        Classify text domain
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (predicted domain, confidence scores)
        """
        # Prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]
            
        # Get predicted domain and confidence scores
        predicted_idx = scores.argmax().item()
        predicted_domain = self.DOMAINS[predicted_idx]
        
        confidence_scores = {
            domain: float(score)
            for domain, score in zip(self.DOMAINS, scores.cpu().numpy())
        }
        
        return predicted_domain, confidence_scores
    
    def batch_classify(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Classify multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of (domain, confidence scores) tuples
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                
            # Process each prediction
            for score in scores:
                predicted_idx = score.argmax().item()
                predicted_domain = self.DOMAINS[predicted_idx]
                
                confidence_scores = {
                    domain: float(s)
                    for domain, s in zip(self.DOMAINS, score.cpu().numpy())
                }
                
                results.append((predicted_domain, confidence_scores))
                
        return results
    
    def get_domain_keywords(self) -> Dict[str, List[str]]:
        """Get keywords associated with each domain"""
        return {
            "academic": [
                "research", "study", "analysis", "theory", "hypothesis",
                "methodology", "findings", "literature", "experiment"
            ],
            "technical": [
                "software", "hardware", "system", "protocol", "algorithm",
                "configuration", "implementation", "architecture"
            ],
            "legal": [
                "law", "regulation", "contract", "agreement", "clause",
                "jurisdiction", "compliance", "liability", "statute"
            ],
            "medical": [
                "patient", "treatment", "diagnosis", "symptoms", "clinical",
                "therapy", "medication", "healthcare", "disease"
            ],
            "business": [
                "market", "company", "finance", "investment", "strategy",
                "management", "revenue", "corporate", "business"
            ],
            "news": [
                "report", "announcement", "update", "coverage", "press",
                "media", "breaking", "latest", "current"
            ],
            "general": [
                "daily", "common", "regular", "standard", "typical",
                "normal", "ordinary", "usual", "general"
            ]
        }
    
    @staticmethod
    def get_supported_domains() -> List[str]:
        """Get list of supported domains"""
        return DomainClassifier.DOMAINS 