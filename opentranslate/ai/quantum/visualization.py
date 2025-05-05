"""
Quantum Circuit Visualization for OpenTranslate.

This module provides visualization capabilities for quantum circuits
used in the translation process.
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display, HTML

class QuantumVisualizer:
    """
    Provides visualization tools for quantum circuits and states.
    """
    
    def __init__(self, style: str = "default"):
        """
        Initialize the quantum visualizer.
        
        Args:
            style: Visualization style to use
        """
        self.style = style
        self.colors = {
            "default": {
                "circuit": "#1f77b4",
                "state": "#ff7f0e",
                "error": "#d62728",
                "background": "#f8f9fa"
            },
            "dark": {
                "circuit": "#00ff00",
                "state": "#ff00ff",
                "error": "#ff0000",
                "background": "#000000"
            }
        }
        
    def visualize_circuit(self, 
                         circuit: QuantumCircuit,
                         title: Optional[str] = None,
                         filename: Optional[str] = None) -> None:
        """
        Visualize a quantum circuit.
        
        Args:
            circuit: Quantum circuit to visualize
            title: Optional title for the visualization
            filename: Optional filename to save the visualization
        """
        # Create figure
        fig = go.Figure()
        
        # Add circuit elements
        for i, gate in enumerate(circuit.data):
            # Get gate information
            gate_name = gate[0].name
            qubits = [q.index for q in gate[1]]
            params = gate[0].params
            
            # Add gate to visualization
            self._add_gate_to_visualization(fig, gate_name, qubits, params, i)
            
        # Update layout
        fig.update_layout(
            title=title or "Quantum Circuit",
            xaxis_title="Time",
            yaxis_title="Qubit",
            showlegend=True,
            plot_bgcolor=self.colors[self.style]["background"],
            paper_bgcolor=self.colors[self.style]["background"],
            font=dict(color="white" if self.style == "dark" else "black")
        )
        
        # Save or display
        if filename:
            fig.write_html(filename)
        else:
            fig.show()
            
    def _add_gate_to_visualization(self, 
                                 fig: go.Figure,
                                 gate_name: str,
                                 qubits: List[int],
                                 params: List[float],
                                 time_step: int) -> None:
        """
        Add a quantum gate to the visualization.
        
        Args:
            fig: Plotly figure to add gate to
            gate_name: Name of the gate
            qubits: Qubits the gate acts on
            params: Gate parameters
            time_step: Time step of the gate
        """
        # Add gate symbol
        fig.add_trace(go.Scatter(
            x=[time_step] * len(qubits),
            y=qubits,
            mode="markers+text",
            marker=dict(
                size=20,
                color=self.colors[self.style]["circuit"]
            ),
            text=gate_name,
            textposition="middle center",
            name=f"{gate_name} Gate"
        ))
        
        # Add control lines if needed
        if gate_name in ["cx", "cz", "ccx"]:
            self._add_control_lines(fig, qubits, time_step)
            
        # Add parameters if any
        if params:
            self._add_parameters(fig, params, qubits, time_step)
            
    def _add_control_lines(self,
                          fig: go.Figure,
                          qubits: List[int],
                          time_step: int) -> None:
        """
        Add control lines to the visualization.
        
        Args:
            fig: Plotly figure to add lines to
            qubits: Qubits involved in the control
            time_step: Time step of the control
        """
        # Add vertical lines
        fig.add_trace(go.Scatter(
            x=[time_step] * 2,
            y=[min(qubits), max(qubits)],
            mode="lines",
            line=dict(
                color=self.colors[self.style]["circuit"],
                width=2
            ),
            showlegend=False
        ))
        
        # Add control dots
        for qubit in qubits[:-1]:
            fig.add_trace(go.Scatter(
                x=[time_step],
                y=[qubit],
                mode="markers",
                marker=dict(
                    size=10,
                    color=self.colors[self.style]["circuit"]
                ),
                showlegend=False
            ))
            
    def _add_parameters(self,
                       fig: go.Figure,
                       params: List[float],
                       qubits: List[int],
                       time_step: int) -> None:
        """
        Add gate parameters to the visualization.
        
        Args:
            fig: Plotly figure to add parameters to
            params: Gate parameters
            qubits: Qubits the gate acts on
            time_step: Time step of the gate
        """
        param_text = ", ".join([f"{p:.2f}" for p in params])
        fig.add_trace(go.Scatter(
            x=[time_step],
            y=[max(qubits) + 0.5],
            mode="text",
            text=[param_text],
            textposition="top center",
            showlegend=False
        ))
        
    def visualize_state(self,
                       state: np.ndarray,
                       title: Optional[str] = None,
                       filename: Optional[str] = None) -> None:
        """
        Visualize a quantum state.
        
        Args:
            state: Quantum state to visualize
            title: Optional title for the visualization
            filename: Optional filename to save the visualization
        """
        # Create Bloch sphere visualization
        fig = plot_bloch_multivector(Statevector(state))
        
        # Update style
        if self.style == "dark":
            fig.set_facecolor("black")
            for ax in fig.axes:
                ax.set_facecolor("black")
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                
        # Save or display
        if filename:
            fig.savefig(filename)
        else:
            plt.show()
            
    def visualize_histogram(self,
                          counts: Dict[str, int],
                          title: Optional[str] = None,
                          filename: Optional[str] = None) -> None:
        """
        Visualize measurement counts as a histogram.
        
        Args:
            counts: Measurement counts to visualize
            title: Optional title for the visualization
            filename: Optional filename to save the visualization
        """
        # Create histogram
        fig = plot_histogram(counts)
        
        # Update style
        if self.style == "dark":
            fig.set_facecolor("black")
            for ax in fig.axes:
                ax.set_facecolor("black")
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                
        # Save or display
        if filename:
            fig.savefig(filename)
        else:
            plt.show()
            
    def visualize_error_syndromes(self,
                                syndromes: List[Tuple[int, str]],
                                title: Optional[str] = None,
                                filename: Optional[str] = None) -> None:
        """
        Visualize error syndromes.
        
        Args:
            syndromes: List of (qubit, error_type) tuples
            title: Optional title for the visualization
            filename: Optional filename to save the visualization
        """
        # Create figure
        fig = go.Figure()
        
        # Add error markers
        for qubit, error_type in syndromes:
            fig.add_trace(go.Scatter(
                x=[qubit],
                y=[0],
                mode="markers+text",
                marker=dict(
                    size=20,
                    color=self.colors[self.style]["error"]
                ),
                text=error_type,
                textposition="middle center",
                name=f"Error on qubit {qubit}"
            ))
            
        # Update layout
        fig.update_layout(
            title=title or "Error Syndromes",
            xaxis_title="Qubit",
            yaxis_title="",
            showlegend=True,
            plot_bgcolor=self.colors[self.style]["background"],
            paper_bgcolor=self.colors[self.style]["background"],
            font=dict(color="white" if self.style == "dark" else "black")
        )
        
        # Save or display
        if filename:
            fig.write_html(filename)
        else:
            fig.show()
            
    def visualize_translation_process(self,
                                    source_text: str,
                                    target_text: str,
                                    quantum_states: List[np.ndarray],
                                    title: Optional[str] = None,
                                    filename: Optional[str] = None) -> None:
        """
        Visualize the quantum translation process.
        
        Args:
            source_text: Source text being translated
            target_text: Target text after translation
            quantum_states: List of quantum states during translation
            title: Optional title for the visualization
            filename: Optional filename to save the visualization
        """
        # Create figure
        fig = go.Figure()
        
        # Add quantum states
        for i, state in enumerate(quantum_states):
            # Calculate state probabilities
            probs = np.abs(state)**2
            
            # Add state trace
            fig.add_trace(go.Scatter(
                x=list(range(len(probs))),
                y=probs,
                mode="lines+markers",
                name=f"State {i}",
                line=dict(color=self.colors[self.style]["state"])
            ))
            
        # Update layout
        fig.update_layout(
            title=title or f"Translation: {source_text} → {target_text}",
            xaxis_title="Basis State",
            yaxis_title="Probability",
            showlegend=True,
            plot_bgcolor=self.colors[self.style]["background"],
            paper_bgcolor=self.colors[self.style]["background"],
            font=dict(color="white" if self.style == "dark" else "black")
        )
        
        # Save or display
        if filename:
            fig.write_html(filename)
        else:
            fig.show() 