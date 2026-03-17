"""Evaluation and benchmarking utilities for predictive maintenance system.

This module provides comprehensive evaluation metrics and benchmarking tools
for model quality, efficiency, and edge performance.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for predictive maintenance.
    
    Evaluates model quality, efficiency, and edge performance with detailed
    metrics and benchmarking capabilities.
    """
    
    def __init__(self) -> None:
        """Initialize the model evaluator."""
        logger.info("Initialized ModelEvaluator")
    
    def evaluate_model_quality(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate model quality metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Prediction probabilities (optional).
            
        Returns:
            Dictionary containing quality metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary"),
            "recall": recall_score(y_true, y_pred, average="binary"),
            "f1_score": f1_score(y_true, y_pred, average="binary"),
        }
        
        # Add AUC if probabilities are provided
        if y_pred_proba is not None:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
        
        # Log results
        logger.info("Model quality evaluation:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_edge_efficiency(
        self,
        model,
        X_test: np.ndarray,
        batch_size: int = 1,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Evaluate model efficiency for edge deployment.
        
        Args:
            model: Model to evaluate.
            X_test: Test data for benchmarking.
            batch_size: Batch size for inference.
            num_runs: Number of benchmark runs.
            warmup_runs: Number of warmup runs.
            
        Returns:
            Dictionary containing efficiency metrics.
        """
        logger.info("Evaluating edge efficiency metrics")
        
        # Warmup runs
        for _ in range(warmup_runs):
            _ = model.predict(X_test[:batch_size], verbose=0)
        
        # Benchmark runs
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.predict(X_test[:batch_size], verbose=0)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Estimate memory usage (rough approximation)
            memory_mb = self._estimate_memory_usage(model, X_test[:batch_size])
            memory_usage.append(memory_mb)
        
        # Calculate latency statistics
        times_ms = [t * 1000 for t in times]  # Convert to milliseconds
        
        efficiency_metrics = {
            "mean_latency_ms": np.mean(times_ms),
            "std_latency_ms": np.std(times_ms),
            "p50_latency_ms": np.percentile(times_ms, 50),
            "p95_latency_ms": np.percentile(times_ms, 95),
            "p99_latency_ms": np.percentile(times_ms, 99),
            "min_latency_ms": np.min(times_ms),
            "max_latency_ms": np.max(times_ms),
            "throughput_fps": 1000 / np.mean(times_ms),  # Frames per second
            "mean_memory_mb": np.mean(memory_usage),
            "max_memory_mb": np.max(memory_usage),
        }
        
        # Add model size
        model_size_info = self._get_model_size_info(model)
        efficiency_metrics.update(model_size_info)
        
        logger.info("Edge efficiency evaluation:")
        for metric, value in efficiency_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return efficiency_metrics
    
    def evaluate_robustness(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        noise_levels: List[float] = [0.01, 0.05, 0.1],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model robustness to noise and perturbations.
        
        Args:
            model: Model to evaluate.
            X_test: Test data.
            y_test: Test labels.
            noise_levels: List of noise levels to test.
            
        Returns:
            Dictionary containing robustness metrics for each noise level.
        """
        logger.info("Evaluating model robustness")
        
        robustness_results = {}
        
        for noise_level in noise_levels:
            # Add noise to test data
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_noisy = X_test + noise
            
            # Get predictions
            y_pred_proba = model.predict(X_noisy, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Calculate metrics
            metrics = self.evaluate_model_quality(y_test, y_pred, y_pred_proba.flatten())
            
            robustness_results[f"noise_{noise_level}"] = metrics
            
            logger.info(f"Robustness at noise level {noise_level}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return robustness_results
    
    def _estimate_memory_usage(self, model, X: np.ndarray) -> float:
        """Estimate memory usage for inference.
        
        Args:
            model: Model to analyze.
            X: Input data.
            
        Returns:
            Estimated memory usage in MB.
        """
        # Rough estimation based on model parameters and input size
        param_count = model.count_params()
        input_size = X.nbytes
        
        # Estimate total memory (parameters + activations + input/output)
        total_bytes = (param_count * 4) + (input_size * 2)  # float32 = 4 bytes
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _get_model_size_info(self, model) -> Dict[str, float]:
        """Get model size information.
        
        Args:
            model: Model to analyze.
            
        Returns:
            Dictionary containing size metrics.
        """
        param_count = model.count_params()
        memory_mb = (param_count * 4) / (1024 * 1024)  # Assuming float32
        
        return {
            "model_size_mb": memory_mb,
            "total_parameters": param_count,
            "layers": len(model.layers),
        }


class PerformanceBenchmark:
    """Performance benchmarking for edge deployment comparison.
    
    Compares different model variants and deployment formats for edge performance.
    """
    
    def __init__(self) -> None:
        """Initialize the performance benchmark."""
        self.results = []
        logger.info("Initialized PerformanceBenchmark")
    
    def benchmark_model_variants(
        self,
        models: Dict[str, any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Benchmark multiple model variants.
        
        Args:
            models: Dictionary of model variants.
            X_test: Test data.
            y_test: Test labels.
            model_names: Names for the models (optional).
            
        Returns:
            DataFrame containing benchmark results.
        """
        if model_names is None:
            model_names = list(models.keys())
        
        logger.info(f"Benchmarking {len(models)} model variants")
        
        evaluator = ModelEvaluator()
        benchmark_results = []
        
        for name, model in zip(model_names, models.values()):
            logger.info(f"Benchmarking model: {name}")
            
            # Get predictions
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Evaluate quality
            quality_metrics = evaluator.evaluate_model_quality(
                y_test, y_pred, y_pred_proba.flatten()
            )
            
            # Evaluate efficiency
            efficiency_metrics = evaluator.evaluate_edge_efficiency(model, X_test)
            
            # Combine results
            result = {
                "model_name": name,
                **quality_metrics,
                **efficiency_metrics,
            }
            
            benchmark_results.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(benchmark_results)
        
        # Sort by F1 score (quality) and latency (efficiency)
        results_df = results_df.sort_values(
            ["f1_score", "mean_latency_ms"], 
            ascending=[False, True]
        )
        
        logger.info("Benchmark results:")
        logger.info(f"\n{results_df.to_string(index=False)}")
        
        return results_df
    
    def create_leaderboard(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Create a performance leaderboard.
        
        Args:
            results_df: Benchmark results DataFrame.
            save_path: Path to save leaderboard (optional).
            
        Returns:
            Formatted leaderboard DataFrame.
        """
        # Select key metrics for leaderboard
        leaderboard_cols = [
            "model_name",
            "accuracy",
            "f1_score",
            "mean_latency_ms",
            "p95_latency_ms",
            "model_size_mb",
            "throughput_fps",
        ]
        
        leaderboard = results_df[leaderboard_cols].copy()
        
        # Format numeric columns
        numeric_cols = ["accuracy", "f1_score", "throughput_fps"]
        for col in numeric_cols:
            leaderboard[col] = leaderboard[col].round(4)
        
        latency_cols = ["mean_latency_ms", "p95_latency_ms"]
        for col in latency_cols:
            leaderboard[col] = leaderboard[col].round(2)
        
        leaderboard["model_size_mb"] = leaderboard["model_size_mb"].round(2)
        
        # Add ranking
        leaderboard["rank"] = range(1, len(leaderboard) + 1)
        
        # Reorder columns
        leaderboard = leaderboard[["rank"] + leaderboard_cols]
        
        logger.info("Performance Leaderboard:")
        logger.info(f"\n{leaderboard.to_string(index=False)}")
        
        # Save if requested
        if save_path:
            leaderboard.to_csv(save_path, index=False)
            logger.info(f"Leaderboard saved to {save_path}")
        
        return leaderboard
    
    def generate_ablation_study(
        self,
        base_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ablation_configs: List[Dict[str, any]],
    ) -> pd.DataFrame:
        """Generate ablation study results.
        
        Args:
            base_model: Base model for ablation study.
            X_test: Test data.
            y_test: Test labels.
            ablation_configs: List of ablation configurations.
            
        Returns:
            DataFrame containing ablation study results.
        """
        logger.info("Generating ablation study")
        
        evaluator = ModelEvaluator()
        ablation_results = []
        
        for i, config in enumerate(ablation_configs):
            config_name = config.get("name", f"config_{i}")
            logger.info(f"Testing ablation: {config_name}")
            
            # Apply configuration to model (this would need to be implemented
            # based on specific ablation types)
            modified_model = self._apply_ablation_config(base_model, config)
            
            # Evaluate modified model
            y_pred_proba = modified_model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            quality_metrics = evaluator.evaluate_model_quality(
                y_test, y_pred, y_pred_proba.flatten()
            )
            efficiency_metrics = evaluator.evaluate_edge_efficiency(modified_model, X_test)
            
            result = {
                "config_name": config_name,
                **config,
                **quality_metrics,
                **efficiency_metrics,
            }
            
            ablation_results.append(result)
        
        ablation_df = pd.DataFrame(ablation_results)
        
        logger.info("Ablation study results:")
        logger.info(f"\n{ablation_df.to_string(index=False)}")
        
        return ablation_df
    
    def _apply_ablation_config(self, model, config: Dict[str, any]):
        """Apply ablation configuration to model.
        
        Args:
            model: Base model.
            config: Ablation configuration.
            
        Returns:
            Modified model.
        """
        # This is a placeholder - actual implementation would depend on
        # specific ablation types (e.g., removing layers, changing activation functions)
        return model


class EdgePerformanceAnalyzer:
    """Analyzer for edge-specific performance characteristics.
    
    Analyzes performance under edge constraints and deployment scenarios.
    """
    
    def __init__(self) -> None:
        """Initialize the edge performance analyzer."""
        logger.info("Initialized EdgePerformanceAnalyzer")
    
    def analyze_edge_constraints(
        self,
        model,
        X_test: np.ndarray,
        constraints: Dict[str, Union[int, float]],
    ) -> Dict[str, bool]:
        """Analyze model compliance with edge constraints.
        
        Args:
            model: Model to analyze.
            X_test: Test data.
            constraints: Edge deployment constraints.
            
        Returns:
            Dictionary of constraint compliance results.
        """
        logger.info("Analyzing edge constraint compliance")
        
        # Get model efficiency metrics
        evaluator = ModelEvaluator()
        efficiency_metrics = evaluator.evaluate_edge_efficiency(model, X_test)
        
        # Check constraint compliance
        compliance = {
            "latency_compliant": efficiency_metrics["p95_latency_ms"] <= constraints.get("max_latency_ms", 100),
            "memory_compliant": efficiency_metrics["max_memory_mb"] <= constraints.get("max_memory_mb", 50),
            "size_compliant": efficiency_metrics["model_size_mb"] <= constraints.get("max_model_size_mb", 10),
            "throughput_compliant": efficiency_metrics["throughput_fps"] >= constraints.get("min_throughput_fps", 1),
        }
        
        logger.info("Edge constraint compliance:")
        for constraint, compliant in compliance.items():
            status = "✓" if compliant else "✗"
            logger.info(f"  {constraint}: {status}")
        
        return compliance
    
    def simulate_edge_deployment(
        self,
        model,
        X_test: np.ndarray,
        deployment_scenarios: List[Dict[str, any]],
    ) -> pd.DataFrame:
        """Simulate edge deployment scenarios.
        
        Args:
            model: Model to deploy.
            X_test: Test data.
            deployment_scenarios: List of deployment scenarios.
            
        Returns:
            DataFrame containing deployment simulation results.
        """
        logger.info("Simulating edge deployment scenarios")
        
        evaluator = ModelEvaluator()
        deployment_results = []
        
        for scenario in deployment_scenarios:
            scenario_name = scenario.get("name", "unknown")
            logger.info(f"Simulating scenario: {scenario_name}")
            
            # Simulate deployment conditions (e.g., reduced precision, limited resources)
            simulated_model = self._simulate_deployment_conditions(model, scenario)
            
            # Evaluate under simulated conditions
            y_pred_proba = simulated_model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            quality_metrics = evaluator.evaluate_model_quality(
                X_test, y_pred, y_pred_proba.flatten()
            )
            efficiency_metrics = evaluator.evaluate_edge_efficiency(simulated_model, X_test)
            
            result = {
                "scenario_name": scenario_name,
                **scenario,
                **quality_metrics,
                **efficiency_metrics,
            }
            
            deployment_results.append(result)
        
        deployment_df = pd.DataFrame(deployment_results)
        
        logger.info("Deployment simulation results:")
        logger.info(f"\n{deployment_df.to_string(index=False)}")
        
        return deployment_df
    
    def _simulate_deployment_conditions(self, model, scenario: Dict[str, any]):
        """Simulate deployment conditions for testing.
        
        Args:
            model: Base model.
            scenario: Deployment scenario configuration.
            
        Returns:
            Model with simulated deployment conditions.
        """
        # This is a placeholder - actual implementation would simulate
        # various deployment conditions (e.g., quantization, pruning, etc.)
        return model
