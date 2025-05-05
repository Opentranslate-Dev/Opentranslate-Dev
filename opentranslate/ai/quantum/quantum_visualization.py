"""
Quantum Visualization for OpenTranslate.

This module implements visualization capabilities
to display quantum computing operations and results.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram, plot_state_city, plot_bloch_multivector
from qiskit.quantum_info import Statevector, DensityMatrix
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime

class QuantumVisualizer:
    """Implements quantum visualization capabilities."""
    
    def __init__(self, style: str = "default"):
        """
        Initialize the quantum visualizer.
        
        Args:
            style: Visualization style to use
        """
        self.style = style
        self._set_style()
        
    def _set_style(self) -> None:
        """Set the visualization style."""
        if self.style == "dark":
            plt.style.use('dark_background')
            sns.set_style("darkgrid")
        else:
            plt.style.use('default')
            sns.set_style("whitegrid")
            
    def visualize_circuit(self,
                         circuit: QuantumCircuit,
                         filename: Optional[str] = None) -> Figure:
        """
        Visualize a quantum circuit.
        
        Args:
            circuit: Circuit to visualize
            filename: Optional filename to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig = circuit.draw(output='mpl', style=self.style)
        if filename:
            fig.savefig(filename)
        return fig
        
    def visualize_state(self,
                       state: Statevector,
                       filename: Optional[str] = None) -> Figure:
        """
        Visualize a quantum state.
        
        Args:
            state: State to visualize
            filename: Optional filename to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig = plot_state_city(state, style=self.style)
        if filename:
            fig.savefig(filename)
        return fig
        
    def visualize_bloch(self,
                       state: Statevector,
                       filename: Optional[str] = None) -> Figure:
        """
        Visualize a quantum state on the Bloch sphere.
        
        Args:
            state: State to visualize
            filename: Optional filename to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig = plot_bloch_multivector(state, style=self.style)
        if filename:
            fig.savefig(filename)
        return fig
        
    def visualize_histogram(self,
                          counts: Dict,
                          filename: Optional[str] = None) -> Figure:
        """
        Visualize measurement results as a histogram.
        
        Args:
            counts: Measurement counts
            filename: Optional filename to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig = plot_histogram(counts, style=self.style)
        if filename:
            fig.savefig(filename)
        return fig
        
    def visualize_performance(self,
                            performance_data: Dict,
                            filename: Optional[str] = None) -> Figure:
        """
        Visualize performance metrics.
        
        Args:
            performance_data: Performance data dictionary
            filename: Optional filename to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot success rates
        success_rates = {
            "Translation": performance_data["translation_success_rate"],
            "Error Correction": performance_data["error_correction_success_rate"],
            "Optimization": performance_data["optimization_success_rate"],
            "ML Training": performance_data["ml_training_success_rate"],
            "ML Evaluation": performance_data["ml_evaluation_success_rate"],
            "Encryption": performance_data["encryption_success_rate"],
            "Decryption": performance_data["decryption_success_rate"],
            "Authentication": performance_data["authentication_success_rate"]
        }
        
        sns.barplot(x=list(success_rates.keys()), y=list(success_rates.values()), ax=axes[0, 0])
        axes[0, 0].set_title("Success Rates")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot execution times
        execution_times = {
            "Translation": performance_data["average_execution_time"],
            "Error Correction": performance_data["average_correction_time"],
            "Optimization": performance_data["average_optimization_time"],
            "ML Training": performance_data["average_ml_time"],
            "Security": performance_data["average_security_time"]
        }
        
        sns.barplot(x=list(execution_times.keys()), y=list(execution_times.values()), ax=axes[0, 1])
        axes[0, 1].set_title("Average Execution Times")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot operation counts
        operation_counts = {
            "Translations": performance_data["total_translations"],
            "Error Corrections": performance_data["total_error_corrections"],
            "Optimizations": performance_data["total_optimizations"],
            "ML Operations": performance_data["total_ml_operations"],
            "Security Operations": performance_data["total_security_operations"]
        }
        
        sns.barplot(x=list(operation_counts.keys()), y=list(operation_counts.values()), ax=axes[1, 0])
        axes[1, 0].set_title("Operation Counts")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot circuit metrics
        circuit_metrics = {
            "Average Depth": performance_data["average_circuit_depth"],
            "Average Gate Count": performance_data["average_gate_count"]
        }
        
        sns.barplot(x=list(circuit_metrics.keys()), y=list(circuit_metrics.values()), ax=axes[1, 1])
        axes[1, 1].set_title("Circuit Metrics")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if filename:
            fig.savefig(filename)
        return fig
        
    def visualize_error_correction(self,
                                 original_circuit: QuantumCircuit,
                                 corrected_circuit: QuantumCircuit,
                                 filename: Optional[str] = None) -> Figure:
        """
        Visualize error correction results.
        
        Args:
            original_circuit: Original circuit
            corrected_circuit: Corrected circuit
            filename: Optional filename to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot original circuit
        original_circuit.draw(output='mpl', ax=axes[0])
        axes[0].set_title("Original Circuit")
        
        # Plot corrected circuit
        corrected_circuit.draw(output='mpl', ax=axes[1])
        axes[1].set_title("Corrected Circuit")
        
        plt.tight_layout()
        if filename:
            fig.savefig(filename)
        return fig
        
    def visualize_optimization(self,
                             original_circuit: QuantumCircuit,
                             optimized_circuit: QuantumCircuit,
                             filename: Optional[str] = None) -> Figure:
        """
        Visualize circuit optimization results.
        
        Args:
            original_circuit: Original circuit
            optimized_circuit: Optimized circuit
            filename: Optional filename to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot original circuit
        original_circuit.draw(output='mpl', ax=axes[0])
        axes[0].set_title("Original Circuit")
        
        # Plot optimized circuit
        optimized_circuit.draw(output='mpl', ax=axes[1])
        axes[1].set_title("Optimized Circuit")
        
        plt.tight_layout()
        if filename:
            fig.savefig(filename)
        return fig
        
    def visualize_ml_training(self,
                            training_history: Dict,
                            filename: Optional[str] = None) -> Figure:
        """
        Visualize machine learning training history.
        
        Args:
            training_history: Training history dictionary
            filename: Optional filename to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training and validation scores
        epochs = range(1, len(training_history["training_scores"]) + 1)
        axes[0].plot(epochs, training_history["training_scores"], label="Training")
        axes[0].plot(epochs, training_history["validation_scores"], label="Validation")
        axes[0].set_title("Training and Validation Scores")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Score")
        axes[0].legend()
        
        # Plot training time
        axes[1].plot(epochs, training_history["training_times"])
        axes[1].set_title("Training Time per Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Time (s)")
        
        plt.tight_layout()
        if filename:
            fig.savefig(filename)
        return fig
        
    def visualize_security(self,
                         original_circuit: QuantumCircuit,
                         secured_circuit: QuantumCircuit,
                         filename: Optional[str] = None) -> Figure:
        """
        Visualize security operation results.
        
        Args:
            original_circuit: Original circuit
            secured_circuit: Secured circuit
            filename: Optional filename to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot original circuit
        original_circuit.draw(output='mpl', ax=axes[0])
        axes[0].set_title("Original Circuit")
        
        # Plot secured circuit
        secured_circuit.draw(output='mpl', ax=axes[1])
        axes[1].set_title("Secured Circuit")
        
        plt.tight_layout()
        if filename:
            fig.savefig(filename)
        return fig
        
    def set_style(self, style: str) -> None:
        """
        Set the visualization style.
        
        Args:
            style: Visualization style to use
        """
        self.style = style
        self._set_style() 