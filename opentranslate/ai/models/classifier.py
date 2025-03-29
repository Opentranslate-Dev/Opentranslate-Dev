"""
OpenTranslate Domain Classifier
"""

from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.generation import GenerationConfig

class DomainClassifier:
    """OpenTranslate Domain Classifier"""
    
    def __init__(
        self,
        model_name: str = "opentranslate/classifier-v1",
        device: Optional[str] = None,
    ):
        """
        Initialize domain classifier
        
        Args:
            model_name: Model name
            device: Running device
        """
        self.model_name = model_name
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move model to specified device
        self.model.to(self.device)
        
    def classify(
        self,
        text: Union[str, List[str]],
        top_k: int = 1,
    ) -> Dict:
        """
        Classify text domain
        
        Args:
            text: Input text or list of texts
            top_k: Return top k most likely domains
            
        Returns:
            Classification result dictionary
        """
        # Preprocess input
        if isinstance(text, str):
            text = [text]
            
        # Batch classify
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
            
            # Get classification scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                
            # Get top k most likely domains
            top_scores, top_indices = torch.topk(scores[0], k=min(top_k, scores.shape[1]))
            
            # Get domain labels
            labels = [self.model.config.id2label[idx.item()] for idx in top_indices]
            
            results.append({
                "text": t,
                "domains": [
                    {
                        "name": label,
                        "score": score.item()
                    }
                    for label, score in zip(labels, top_scores)
                ]
            })
            
        # Calculate overall scores
        if len(results) > 1:
            # Merge scores from all texts
            domain_scores = {}
            for result in results:
                for domain in result["domains"]:
                    name = domain["name"]
                    score = domain["score"]
                    if name not in domain_scores:
                        domain_scores[name] = []
                    domain_scores[name].append(score)
                    
            # Calculate average scores
            for name in domain_scores:
                domain_scores[name] = sum(domain_scores[name]) / len(domain_scores[name])
                
            # Get main domain
            main_domain = max(domain_scores.items(), key=lambda x: x[1])
            
            return {
                "main_domain": main_domain[0],
                "confidence": main_domain[1],
                "details": results,
                "domain_scores": domain_scores
            }
        else:
            result = results[0]
            return {
                "main_domain": result["domains"][0]["name"],
                "confidence": result["domains"][0]["score"],
                "details": results,
                "domain_scores": {
                    d["name"]: d["score"]
                    for d in result["domains"]
                }
            }
            
    def batch_classify(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> List[Dict]:
        """
        Batch classify text domains
        
        Args:
            texts: List of texts
            batch_size: Batch processing size
            **kwargs: Other parameters
            
        Returns:
            List of classification results
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self.classify(batch, **kwargs)
            results.extend(result["details"])
        return results
        
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Model information dictionary
        """
        return {
            "name": self.model_name,
            "device": str(self.device),
            "num_labels": self.model.config.num_labels,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
    def save(self, path: str):
        """
        Save model
        
        Args:
            path: Save path
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    @classmethod
    def load(cls, path: str, **kwargs):
        """
        Load model
        
        Args:
            path: Model path
            **kwargs: Other parameters
            
        Returns:
            Model instance
        """
        model = cls(model_name=path, **kwargs)
        return model 