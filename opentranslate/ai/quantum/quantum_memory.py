"""
Quantum Translation Memory implementation for OpenTranslate.

This module implements quantum-inspired translation memory that stores
and retrieves translations in a quantum state space for faster access
and better quality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors

@dataclass
class QuantumMemoryEntry:
    """Represents an entry in the quantum translation memory."""
    source_text: str
    target_text: str
    source_embedding: np.ndarray
    target_embedding: np.ndarray
    quality_score: float
    context_vectors: List[np.ndarray]
    metadata: Dict

class QuantumTranslationMemory:
    """
    Quantum-inspired translation memory system that stores translations
    in a quantum state space for efficient retrieval and quality enhancement.
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 max_entries: int = 10000,
                 similarity_threshold: float = 0.8):
        """
        Initialize the quantum translation memory.
        
        Args:
            embedding_dim: Dimension of text embeddings
            max_entries: Maximum number of entries to store
            similarity_threshold: Threshold for considering entries similar
        """
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.entries: List[QuantumMemoryEntry] = []
        self.nearest_neighbors = None
        self._update_nearest_neighbors()
        
    def add_entry(self, 
                 source_text: str,
                 target_text: str,
                 source_embedding: np.ndarray,
                 target_embedding: np.ndarray,
                 quality_score: float,
                 context_vectors: List[np.ndarray],
                 metadata: Optional[Dict] = None):
        """
        Add a new entry to the quantum translation memory.
        
        Args:
            source_text: Source text
            target_text: Translated text
            source_embedding: Source text embedding
            target_embedding: Target text embedding
            quality_score: Quality score of the translation
            context_vectors: Context vectors for the translation
            metadata: Optional metadata
        """
        # Create new entry
        entry = QuantumMemoryEntry(
            source_text=source_text,
            target_text=target_text,
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            quality_score=quality_score,
            context_vectors=context_vectors,
            metadata=metadata or {}
        )
        
        # Add to entries
        self.entries.append(entry)
        
        # Maintain maximum number of entries
        if len(self.entries) > self.max_entries:
            # Remove lowest quality entry
            self.entries.sort(key=lambda x: x.quality_score)
            self.entries.pop(0)
            
        # Update nearest neighbors
        self._update_nearest_neighbors()
    
    def find_similar(self, 
                    source_embedding: np.ndarray,
                    k: int = 5) -> List[Tuple[QuantumMemoryEntry, float]]:
        """
        Find similar entries in the quantum translation memory.
        
        Args:
            source_embedding: Source text embedding
            k: Number of similar entries to return
            
        Returns:
            List of (entry, similarity) tuples
        """
        if not self.entries:
            return []
            
        # Get nearest neighbors
        distances, indices = self.nearest_neighbors.kneighbors(
            source_embedding.reshape(1, -1),
            n_neighbors=min(k, len(self.entries))
        )
        
        # Convert to similarity scores
        similarities = 1 - distances.flatten()
        
        # Filter by similarity threshold
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if similarity >= self.similarity_threshold:
                results.append((self.entries[idx], similarity))
                
        return results
    
    def get_best_translation(self,
                           source_embedding: np.ndarray,
                           context_vectors: Optional[List[np.ndarray]] = None) -> Optional[str]:
        """
        Get the best translation for a given source embedding.
        
        Args:
            source_embedding: Source text embedding
            context_vectors: Optional context vectors
            
        Returns:
            Best translation if found, None otherwise
        """
        # Find similar entries
        similar = self.find_similar(source_embedding)
        if not similar:
            return None
            
        # Sort by quality and similarity
        similar.sort(key=lambda x: (x[0].quality_score, x[1]), reverse=True)
        
        # Get best entry
        best_entry = similar[0][0]
        
        # Apply context-aware adjustment if context vectors provided
        if context_vectors:
            return self._apply_context_adjustment(best_entry, context_vectors)
            
        return best_entry.target_text
    
    def _update_nearest_neighbors(self):
        """
        Update the nearest neighbors model with current entries.
        """
        if not self.entries:
            return
            
        # Get source embeddings
        embeddings = np.array([entry.source_embedding for entry in self.entries])
        
        # Create nearest neighbors model
        self.nearest_neighbors = NearestNeighbors(
            n_neighbors=min(5, len(self.entries)),
            metric='cosine'
        )
        self.nearest_neighbors.fit(embeddings)
    
    def _apply_context_adjustment(self,
                                entry: QuantumMemoryEntry,
                                context_vectors: List[np.ndarray]) -> str:
        """
        Apply context-aware adjustment to translation.
        
        Args:
            entry: Memory entry
            context_vectors: Context vectors
            
        Returns:
            Context-adjusted translation
        """
        # This is a simplified implementation
        # In practice, this would use more sophisticated NLP techniques
        
        # Calculate context similarity
        context_similarities = []
        for entry_context in entry.context_vectors:
            for current_context in context_vectors:
                similarity = np.dot(entry_context, current_context) / (
                    np.linalg.norm(entry_context) * np.linalg.norm(current_context)
                )
                context_similarities.append(similarity)
                
        # If context is very different, return original translation
        if not context_similarities or max(context_similarities) < 0.5:
            return entry.target_text
            
        # Otherwise, apply minor adjustments based on context
        # This is a placeholder for more sophisticated context adaptation
        return entry.target_text 