"""Predictive Maintenance Edge AI System.

A comprehensive Edge AI/IoT system for predictive maintenance using
LSTM-based time-series analysis optimized for edge deployment.
"""

__version__ = "0.1.0"
__author__ = "Edge AI Team"
__email__ = "team@example.com"

from .models.lstm_model import PredictiveMaintenanceModel, set_deterministic_seed
from .pipelines.data_pipeline import DataPipeline, SensorDataGenerator, DataPreprocessor
from .export.edge_exporter import EdgeModelExporter
from .utils.evaluation import ModelEvaluator, PerformanceBenchmark, EdgePerformanceAnalyzer

__all__ = [
    "PredictiveMaintenanceModel",
    "set_deterministic_seed",
    "DataPipeline",
    "SensorDataGenerator", 
    "DataPreprocessor",
    "EdgeModelExporter",
    "ModelEvaluator",
    "PerformanceBenchmark",
    "EdgePerformanceAnalyzer",
]
