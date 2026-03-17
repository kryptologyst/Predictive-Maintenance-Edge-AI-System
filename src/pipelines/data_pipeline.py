"""Data pipeline for predictive maintenance system.

This module handles data generation, preprocessing, and streaming for the
predictive maintenance system, including sensor simulation and edge constraints.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SensorDataGenerator:
    """Generator for synthetic sensor data for predictive maintenance.
    
    Simulates realistic sensor patterns for normal and faulty equipment states
    with configurable noise levels and failure patterns.
    """
    
    def __init__(
        self,
        time_steps: int = 50,
        features: int = 3,
        normal_mean: Optional[List[float]] = None,
        normal_std: Optional[List[float]] = None,
        faulty_mean: Optional[List[float]] = None,
        faulty_std: Optional[List[float]] = None,
        seed: int = 42,
    ) -> None:
        """Initialize the sensor data generator.
        
        Args:
            time_steps: Number of time steps per sequence.
            features: Number of sensor features (vibration, temperature, pressure).
            normal_mean: Mean values for normal operation.
            normal_std: Standard deviation for normal operation.
            faulty_mean: Mean values for faulty operation.
            faulty_std: Standard deviation for faulty operation.
            seed: Random seed for reproducibility.
        """
        self.time_steps = time_steps
        self.features = features
        self.seed = seed
        
        # Set default sensor characteristics
        self.normal_mean = normal_mean or [0.3, 60.0, 30.0]  # vibration, temp, pressure
        self.normal_std = normal_std or [0.05, 2.0, 1.0]
        self.faulty_mean = faulty_mean or [0.5, 70.0, 27.0]
        self.faulty_std = faulty_std or [0.1, 3.0, 2.0]
        
        # Validate input dimensions
        if len(self.normal_mean) != features:
            raise ValueError(f"normal_mean length ({len(self.normal_mean)}) must equal features ({features})")
        if len(self.normal_std) != features:
            raise ValueError(f"normal_std length ({len(self.normal_std)}) must equal features ({features})")
        if len(self.faulty_mean) != features:
            raise ValueError(f"faulty_mean length ({len(self.faulty_mean)}) must equal features ({features})")
        if len(self.faulty_std) != features:
            raise ValueError(f"faulty_std length ({len(self.faulty_std)}) must equal features ({features})")
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(
            f"Initialized SensorDataGenerator with "
            f"time_steps={time_steps}, features={features}"
        )
    
    def generate_normal_data(
        self,
        n_samples: int,
        add_trend: bool = True,
        add_seasonality: bool = True,
    ) -> np.ndarray:
        """Generate normal operational sensor data.
        
        Args:
            n_samples: Number of samples to generate.
            add_trend: Whether to add gradual trends to the data.
            add_seasonality: Whether to add seasonal patterns.
            
        Returns:
            Generated sensor data of shape (n_samples, time_steps, features).
        """
        data = np.zeros((n_samples, self.time_steps, self.features))
        
        for i in range(n_samples):
            for j in range(self.features):
                # Base normal distribution
                base_data = np.random.normal(
                    self.normal_mean[j],
                    self.normal_std[j],
                    self.time_steps
                )
                
                # Add trend if enabled
                if add_trend:
                    trend = np.linspace(0, 0.1 * self.normal_std[j], self.time_steps)
                    base_data += trend
                
                # Add seasonality if enabled
                if add_seasonality:
                    seasonal = 0.05 * self.normal_std[j] * np.sin(
                        2 * np.pi * np.arange(self.time_steps) / (self.time_steps / 3)
                    )
                    base_data += seasonal
                
                data[i, :, j] = base_data
        
        logger.info(f"Generated {n_samples} normal sensor data samples")
        
        return data
    
    def generate_faulty_data(
        self,
        n_samples: int,
        fault_type: str = "gradual",
        fault_start: Optional[int] = None,
    ) -> np.ndarray:
        """Generate faulty operational sensor data.
        
        Args:
            n_samples: Number of samples to generate.
            fault_type: Type of fault ("gradual", "sudden", "intermittent").
            fault_start: Time step when fault begins (defaults to random).
            
        Returns:
            Generated faulty sensor data of shape (n_samples, time_steps, features).
        """
        data = np.zeros((n_samples, self.time_steps, self.features))
        
        for i in range(n_samples):
            # Determine fault start time
            if fault_start is None:
                start_time = random.randint(self.time_steps // 4, self.time_steps // 2)
            else:
                start_time = fault_start
            
            for j in range(self.features):
                # Start with normal data
                normal_data = np.random.normal(
                    self.normal_mean[j],
                    self.normal_std[j],
                    self.time_steps
                )
                
                # Apply fault pattern
                if fault_type == "gradual":
                    # Gradual degradation
                    fault_progression = np.linspace(0, 1, self.time_steps - start_time)
                    fault_effect = (
                        (self.faulty_mean[j] - self.normal_mean[j]) * fault_progression
                    )
                    normal_data[start_time:] += fault_effect
                    
                elif fault_type == "sudden":
                    # Sudden fault
                    fault_effect = self.faulty_mean[j] - self.normal_mean[j]
                    normal_data[start_time:] += fault_effect
                    
                elif fault_type == "intermittent":
                    # Intermittent fault
                    fault_effect = self.faulty_mean[j] - self.normal_mean[j]
                    fault_pattern = np.random.choice(
                        [0, 1], 
                        size=self.time_steps - start_time,
                        p=[0.7, 0.3]  # 30% chance of fault at each time step
                    )
                    normal_data[start_time:] += fault_effect * fault_pattern
                
                # Add fault noise
                fault_noise = np.random.normal(0, self.faulty_std[j], self.time_steps)
                normal_data[start_time:] += fault_noise[start_time:]
                
                data[i, :, j] = normal_data
        
        logger.info(f"Generated {n_samples} faulty sensor data samples ({fault_type} fault)")
        
        return data
    
    def generate_dataset(
        self,
        n_samples: int = 1000,
        normal_ratio: float = 0.7,
        fault_types: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete dataset with normal and faulty samples.
        
        Args:
            n_samples: Total number of samples to generate.
            normal_ratio: Ratio of normal samples to total samples.
            fault_types: List of fault types to include.
            
        Returns:
            Tuple of (sensor_data, labels).
        """
        if fault_types is None:
            fault_types = ["gradual", "sudden", "intermittent"]
        
        # Calculate sample counts
        n_normal = int(n_samples * normal_ratio)
        n_faulty = n_samples - n_normal
        
        # Generate normal data
        normal_data = self.generate_normal_data(n_normal)
        normal_labels = np.zeros(n_normal)
        
        # Generate faulty data
        faulty_data_list = []
        faulty_labels_list = []
        
        samples_per_fault = n_faulty // len(fault_types)
        for fault_type in fault_types:
            fault_data = self.generate_faulty_data(samples_per_fault, fault_type)
            faulty_data_list.append(fault_data)
            faulty_labels_list.append(np.ones(samples_per_fault))
        
        # Handle remaining samples
        remaining_samples = n_faulty - (samples_per_fault * len(fault_types))
        if remaining_samples > 0:
            fault_data = self.generate_faulty_data(remaining_samples, fault_types[0])
            faulty_data_list.append(fault_data)
            faulty_labels_list.append(np.ones(remaining_samples))
        
        # Combine all data
        faulty_data = np.vstack(faulty_data_list)
        faulty_labels = np.concatenate(faulty_labels_list)
        
        # Combine normal and faulty data
        X = np.vstack([normal_data, faulty_data])
        y = np.concatenate([normal_labels, faulty_labels])
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        logger.info(
            f"Generated complete dataset: {len(X)} samples "
            f"({n_normal} normal, {n_faulty} faulty)"
        )
        
        return X, y


class DataPreprocessor:
    """Data preprocessing utilities for predictive maintenance.
    
    Handles normalization, scaling, and feature engineering for sensor data.
    """
    
    def __init__(self, scaler: Optional[StandardScaler] = None) -> None:
        """Initialize the data preprocessor.
        
        Args:
            scaler: Pre-fitted scaler (optional).
        """
        self.scaler = scaler or StandardScaler()
        self.is_fitted = scaler is not None
        
        logger.info("Initialized DataPreprocessor")
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data.
        
        Args:
            X: Input data of shape (n_samples, time_steps, features).
            
        Returns:
            Transformed data.
        """
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X_reshaped)
        self.is_fitted = True
        
        # Reshape back
        X_scaled = X_scaled.reshape(original_shape)
        
        logger.info("Fitted scaler and transformed data")
        
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler.
        
        Args:
            X: Input data of shape (n_samples, time_steps, features).
            
        Returns:
            Transformed data.
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Transform
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data.
        
        Args:
            X: Scaled data of shape (n_samples, time_steps, features).
            
        Returns:
            Original scale data.
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        # Reshape for inverse scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Inverse transform
        X_original = self.scaler.inverse_transform(X_reshaped)
        
        # Reshape back
        X_original = X_original.reshape(original_shape)
        
        return X_original


class DataPipeline:
    """Complete data pipeline for predictive maintenance system.
    
    Handles data generation, preprocessing, splitting, and edge constraints.
    """
    
    def __init__(
        self,
        time_steps: int = 50,
        features: int = 3,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Initialize the data pipeline.
        
        Args:
            time_steps: Number of time steps per sequence.
            features: Number of sensor features.
            test_size: Fraction of data for testing.
            val_size: Fraction of training data for validation.
            seed: Random seed for reproducibility.
        """
        self.time_steps = time_steps
        self.features = features
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        
        # Initialize components
        self.generator = SensorDataGenerator(
            time_steps=time_steps,
            features=features,
            seed=seed
        )
        self.preprocessor = DataPreprocessor()
        
        logger.info(
            f"Initialized DataPipeline with "
            f"time_steps={time_steps}, features={features}"
        )
    
    def create_dataset(
        self,
        n_samples: int = 1000,
        normal_ratio: float = 0.7,
        fault_types: Optional[List[str]] = None,
        preprocess: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a complete dataset.
        
        Args:
            n_samples: Total number of samples.
            normal_ratio: Ratio of normal samples.
            fault_types: Types of faults to include.
            preprocess: Whether to apply preprocessing.
            
        Returns:
            Tuple of (sensor_data, labels).
        """
        # Generate data
        X, y = self.generator.generate_dataset(
            n_samples=n_samples,
            normal_ratio=normal_ratio,
            fault_types=fault_types
        )
        
        # Preprocess if requested
        if preprocess:
            X = self.preprocessor.fit_transform(X)
        
        logger.info(f"Created dataset with {len(X)} samples")
        
        return X, y
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets.
        
        Args:
            X: Input data.
            y: Labels.
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_size, random_state=self.seed, stratify=y_temp
        )
        
        logger.info(
            f"Split data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_edge_constraints(self) -> Dict[str, Union[int, float]]:
        """Get edge deployment constraints.
        
        Returns:
            Dictionary containing edge constraints.
        """
        constraints = {
            "max_batch_size": 1,  # Edge devices typically process one sample at a time
            "max_sequence_length": self.time_steps,
            "max_features": self.features,
            "target_latency_ms": 100,  # Target inference latency
            "max_memory_mb": 50,  # Maximum memory usage
            "max_model_size_mb": 10,  # Maximum model size
        }
        
        return constraints
    
    def validate_edge_compatibility(
        self,
        X: np.ndarray,
        constraints: Optional[Dict[str, Union[int, float]]] = None,
    ) -> Dict[str, bool]:
        """Validate data compatibility with edge constraints.
        
        Args:
            X: Input data to validate.
            constraints: Edge constraints to check against.
            
        Returns:
            Dictionary of validation results.
        """
        if constraints is None:
            constraints = self.get_edge_constraints()
        
        validation_results = {
            "batch_size_compatible": X.shape[0] <= constraints["max_batch_size"],
            "sequence_length_compatible": X.shape[1] <= constraints["max_sequence_length"],
            "features_compatible": X.shape[2] <= constraints["max_features"],
        }
        
        logger.info(f"Edge compatibility validation: {validation_results}")
        
        return validation_results
