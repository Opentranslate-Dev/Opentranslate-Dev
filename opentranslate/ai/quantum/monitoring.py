"""
Quantum Performance Monitoring for OpenTranslate.

This module provides performance monitoring and reporting capabilities
for quantum translation operations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import logging

class QuantumPerformanceMonitor:
    """
    Monitors and reports quantum translation performance metrics.
    """
    
    def __init__(self, prometheus_port: int = 8000):
        """
        Initialize the quantum performance monitor.
        
        Args:
            prometheus_port: Port for Prometheus metrics server
        """
        # Initialize Prometheus metrics
        self.translation_counter = Counter(
            'quantum_translation_total',
            'Total number of quantum translations'
        )
        self.error_correction_counter = Counter(
            'quantum_error_correction_total',
            'Total number of error corrections'
        )
        self.circuit_execution_time = Histogram(
            'quantum_circuit_execution_seconds',
            'Time taken to execute quantum circuits'
        )
        self.translation_accuracy = Gauge(
            'quantum_translation_accuracy',
            'Accuracy of quantum translations'
        )
        self.quantum_advantage = Gauge(
            'quantum_advantage_ratio',
            'Ratio of quantum to classical performance'
        )
        
        # Start Prometheus server
        start_http_server(prometheus_port)
        
        # Initialize logging
        self.logger = logging.getLogger('quantum_monitor')
        self.logger.setLevel(logging.INFO)
        
        # Initialize performance history
        self.performance_history = {
            'translations': [],
            'error_corrections': [],
            'execution_times': [],
            'accuracies': [],
            'quantum_advantages': []
        }
        
    def record_translation(self,
                         source_text: str,
                         target_text: str,
                         quantum_enhanced: bool,
                         execution_time: float,
                         accuracy: float) -> None:
        """
        Record a quantum translation performance.
        
        Args:
            source_text: Source text being translated
            target_text: Translated text
            quantum_enhanced: Whether quantum enhancement was used
            execution_time: Time taken for translation
            accuracy: Translation accuracy score
        """
        # Update Prometheus metrics
        self.translation_counter.inc()
        self.circuit_execution_time.observe(execution_time)
        self.translation_accuracy.set(accuracy)
        
        # Record in history
        self.performance_history['translations'].append({
            'timestamp': datetime.now(),
            'source_text': source_text,
            'target_text': target_text,
            'quantum_enhanced': quantum_enhanced,
            'execution_time': execution_time,
            'accuracy': accuracy
        })
        
        # Log the translation
        self.logger.info(
            f"Translation recorded: {source_text[:50]}... -> {target_text[:50]}... "
            f"(Quantum: {quantum_enhanced}, Time: {execution_time:.2f}s, "
            f"Accuracy: {accuracy:.2f})"
        )
        
    def record_error_correction(self,
                              error_type: str,
                              correction_time: float,
                              success: bool) -> None:
        """
        Record an error correction performance.
        
        Args:
            error_type: Type of error corrected
            correction_time: Time taken for correction
            success: Whether correction was successful
        """
        # Update Prometheus metrics
        self.error_correction_counter.inc()
        
        # Record in history
        self.performance_history['error_corrections'].append({
            'timestamp': datetime.now(),
            'error_type': error_type,
            'correction_time': correction_time,
            'success': success
        })
        
        # Log the correction
        self.logger.info(
            f"Error correction recorded: {error_type} "
            f"(Time: {correction_time:.2f}s, Success: {success})"
        )
        
    def record_quantum_advantage(self,
                               quantum_time: float,
                               classical_time: float,
                               quantum_accuracy: float,
                               classical_accuracy: float) -> None:
        """
        Record quantum advantage metrics.
        
        Args:
            quantum_time: Time taken for quantum translation
            classical_time: Time taken for classical translation
            quantum_accuracy: Accuracy of quantum translation
            classical_accuracy: Accuracy of classical translation
        """
        # Calculate quantum advantage ratio
        time_advantage = classical_time / quantum_time
        accuracy_advantage = quantum_accuracy / classical_accuracy
        overall_advantage = (time_advantage + accuracy_advantage) / 2
        
        # Update Prometheus metrics
        self.quantum_advantage.set(overall_advantage)
        
        # Record in history
        self.performance_history['quantum_advantages'].append({
            'timestamp': datetime.now(),
            'time_advantage': time_advantage,
            'accuracy_advantage': accuracy_advantage,
            'overall_advantage': overall_advantage
        })
        
        # Log the advantage
        self.logger.info(
            f"Quantum advantage recorded: Time {time_advantage:.2f}x, "
            f"Accuracy {accuracy_advantage:.2f}x, Overall {overall_advantage:.2f}x"
        )
        
    def generate_performance_report(self,
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None) -> Dict:
        """
        Generate a performance report for the specified time period.
        
        Args:
            start_time: Start time for the report
            end_time: End time for the report
            
        Returns:
            Performance report dictionary
        """
        # Filter history by time period
        filtered_history = self._filter_history(start_time, end_time)
        
        # Calculate statistics
        stats = {
            'total_translations': len(filtered_history['translations']),
            'total_error_corrections': len(filtered_history['error_corrections']),
            'average_execution_time': np.mean([
                t['execution_time'] for t in filtered_history['translations']
            ]),
            'average_accuracy': np.mean([
                t['accuracy'] for t in filtered_history['translations']
            ]),
            'quantum_enhancement_rate': np.mean([
                t['quantum_enhanced'] for t in filtered_history['translations']
            ]),
            'error_correction_success_rate': np.mean([
                e['success'] for e in filtered_history['error_corrections']
            ]),
            'average_quantum_advantage': np.mean([
                a['overall_advantage'] for a in filtered_history['quantum_advantages']
            ])
        }
        
        # Generate visualizations
        self._generate_visualizations(filtered_history)
        
        return stats
        
    def _filter_history(self,
                       start_time: Optional[datetime],
                       end_time: Optional[datetime]) -> Dict:
        """
        Filter performance history by time period.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            
        Returns:
            Filtered history dictionary
        """
        filtered_history = {}
        
        for key, records in self.performance_history.items():
            filtered_records = records
            if start_time:
                filtered_records = [
                    r for r in filtered_records if r['timestamp'] >= start_time
                ]
            if end_time:
                filtered_records = [
                    r for r in filtered_records if r['timestamp'] <= end_time
                ]
            filtered_history[key] = filtered_records
            
        return filtered_history
        
    def _generate_visualizations(self, history: Dict) -> None:
        """
        Generate performance visualizations.
        
        Args:
            history: Performance history to visualize
        """
        # Set style
        sns.set_style("darkgrid")
        
        # Create figures directory if it doesn't exist
        import os
        if not os.path.exists('figures'):
            os.makedirs('figures')
            
        # Plot translation performance over time
        if history['translations']:
            df = pd.DataFrame(history['translations'])
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x='timestamp', y='accuracy')
            plt.title('Translation Accuracy Over Time')
            plt.savefig('figures/translation_accuracy.png')
            plt.close()
            
        # Plot error correction performance
        if history['error_corrections']:
            df = pd.DataFrame(history['error_corrections'])
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x='error_type', y='correction_time')
            plt.title('Error Correction Performance by Type')
            plt.savefig('figures/error_correction.png')
            plt.close()
            
        # Plot quantum advantage
        if history['quantum_advantages']:
            df = pd.DataFrame(history['quantum_advantages'])
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x='timestamp', y='overall_advantage')
            plt.title('Quantum Advantage Over Time')
            plt.savefig('figures/quantum_advantage.png')
            plt.close() 