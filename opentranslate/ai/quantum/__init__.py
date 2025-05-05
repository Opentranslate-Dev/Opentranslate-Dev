"""
Quantum-enhanced translation module for OpenTranslate.

This module implements quantum-inspired algorithms for translation tasks,
including quantum entanglement-inspired translation pairs, superposition-based
context awareness, and quantum-state translation memory.
"""

from .quantum_translator import QuantumTranslator
from .quantum_memory import QuantumTranslationMemory
from .context_awareness import QuantumContextAwareness
from .security import QuantumSecurityLayer

__all__ = [
    'QuantumTranslator',
    'QuantumTranslationMemory',
    'QuantumContextAwareness',
    'QuantumSecurityLayer'
] 