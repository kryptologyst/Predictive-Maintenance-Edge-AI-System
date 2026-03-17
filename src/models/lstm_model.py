"""Predictive Maintenance LSTM Model for Edge AI/IoT.

This module implements a lightweight LSTM-based predictive maintenance system
optimized for edge deployment with quantization and pruning support.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow_model_optimization.quantization.keras import vitis_quantize

logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducible results.
    
    Args:
        seed: Random seed value for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Additional TensorFlow determinism settings
    tf.config.experimental.enable_op_determinism()


class PredictiveMaintenanceModel:
    """LSTM-based predictive maintenance model for edge deployment.
    
    This class implements a lightweight LSTM model for predicting equipment
    failures based on time-series sensor data (vibration, temperature, pressure).
    Supports quantization and pruning for edge optimization.
    """
    
    def __init__(
        self,
        time_steps: int = 50,
        features: int = 3,
        lstm_units: int = 32,
        dense_units: int = 16,
        dropout_rate: float = 0.2,
        seed: int = 42,
    ) -> None:
        """Initialize the predictive maintenance model.
        
        Args:
            time_steps: Number of time steps in input sequences.
            features: Number of sensor features (vibration, temperature, pressure).
            lstm_units: Number of LSTM units.
            dense_units: Number of dense layer units.
            dropout_rate: Dropout rate for regularization.
            seed: Random seed for reproducibility.
        """
        self.time_steps = time_steps
        self.features = features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.seed = seed
        
        # Set deterministic seed
        set_deterministic_seed(seed)
        
        self.model: Optional[keras.Model] = None
        self.quantized_model: Optional[keras.Model] = None
        self.pruned_model: Optional[keras.Model] = None
        
        logger.info(
            f"Initialized PredictiveMaintenanceModel with "
            f"time_steps={time_steps}, features={features}, "
            f"lstm_units={lstm_units}, dense_units={dense_units}"
        )
    
    def build_model(self) -> keras.Model:
        """Build the LSTM-based predictive maintenance model.
        
        Returns:
            Compiled Keras model ready for training.
        """
        if self.model is not None:
            return self.model
            
        # Build model architecture
        model = models.Sequential([
            layers.LSTM(
                self.lstm_units,
                input_shape=(self.time_steps, self.features),
                return_sequences=False,
                name="lstm_layer"
            ),
            layers.Dropout(self.dropout_rate, name="dropout_1"),
            layers.Dense(
                self.dense_units,
                activation="relu",
                name="dense_1"
            ),
            layers.Dropout(self.dropout_rate, name="dropout_2"),
            layers.Dense(1, activation="sigmoid", name="output")
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )
        
        self.model = model
        
        logger.info("Built LSTM predictive maintenance model")
        logger.info(f"Model summary:\n{model.summary()}")
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> keras.callbacks.History:
        """Train the predictive maintenance model.
        
        Args:
            X_train: Training input sequences of shape (n_samples, time_steps, features).
            y_train: Training labels of shape (n_samples,).
            X_val: Validation input sequences (optional).
            y_val: Validation labels (optional).
            epochs: Number of training epochs.
            batch_size: Training batch size.
            verbose: Verbosity level.
            
        Returns:
            Training history object.
        """
        if self.model is None:
            self.build_model()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if validation_data else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("Model training completed")
        
        return history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model: Optional[keras.Model] = None,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test input sequences.
            y_test: Test labels.
            model: Specific model to evaluate (defaults to self.model).
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model available for evaluation")
        
        # Evaluate model
        results = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for additional metrics
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate additional metrics
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        
        metrics = {
            "loss": results[0],
            "accuracy": results[1],
            "precision": results[2] if len(results) > 2 else precision_score(y_test, y_pred),
            "recall": results[3] if len(results) > 3 else recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        
        # Log detailed results
        logger.info("Model evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        return metrics
    
    def predict_maintenance(
        self,
        sensor_data: np.ndarray,
        model: Optional[keras.Model] = None,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict maintenance needs for given sensor data.
        
        Args:
            sensor_data: Input sensor sequences of shape (n_samples, time_steps, features).
            model: Specific model to use for prediction (defaults to self.model).
            threshold: Classification threshold for binary predictions.
            
        Returns:
            Tuple of (predictions, probabilities).
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model available for prediction")
        
        # Get prediction probabilities
        probabilities = model.predict(sensor_data, verbose=0)
        
        # Convert to binary predictions
        predictions = (probabilities > threshold).astype(int).flatten()
        
        return predictions, probabilities.flatten()
    
    def quantize_model(
        self,
        X_calibration: np.ndarray,
        quantization_type: str = "int8",
    ) -> keras.Model:
        """Create a quantized version of the model for edge deployment.
        
        Args:
            X_calibration: Calibration data for quantization.
            quantization_type: Type of quantization ("int8", "float16").
            
        Returns:
            Quantized Keras model.
        """
        if self.model is None:
            raise ValueError("No trained model available for quantization")
        
        logger.info(f"Quantizing model to {quantization_type}")
        
        # Create quantized model
        quantized_model = vitis_quantize.quantize_model(
            self.model,
            X_calibration,
            quantize_config=vitis_quantize.VitisQuantizeConfig(
                quantize_policy=vitis_quantize.VitisQuantizeConfig.Policy(
                    quantize_policy=vitis_quantize.VitisQuantizeConfig.Policy.QuantizePolicy(
                        weight_bits=8,
                        activation_bits=8,
                    )
                )
            )
        )
        
        self.quantized_model = quantized_model
        
        logger.info("Model quantization completed")
        
        return quantized_model
    
    def prune_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sparsity: float = 0.5,
    ) -> keras.Model:
        """Create a pruned version of the model for edge deployment.
        
        Args:
            X_train: Training data for pruning.
            y_train: Training labels.
            sparsity: Target sparsity level (0.0 to 1.0).
            
        Returns:
            Pruned Keras model.
        """
        if self.model is None:
            raise ValueError("No trained model available for pruning")
        
        logger.info(f"Pruning model to {sparsity:.1%} sparsity")
        
        # Import pruning utilities
        from tensorflow_model_optimization.sparsity import keras as sparsity
        
        # Define pruning schedule
        pruning_params = {
            "pruning_schedule": sparsity.ConstantSparsity(
                target_sparsity=sparsity,
                begin_step=0,
                end_step=-1,
                frequency=100
            )
        }
        
        # Create pruned model
        pruned_model = sparsity.prune_low_magnitude(self.model, **pruning_params)
        
        # Compile pruned model
        pruned_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # Fine-tune pruned model
        pruned_model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        # Strip pruning wrappers
        pruned_model = sparsity.strip_pruning(pruned_model)
        
        self.pruned_model = pruned_model
        
        logger.info("Model pruning completed")
        
        return pruned_model
    
    def get_model_size(self, model: Optional[keras.Model] = None) -> Dict[str, float]:
        """Get model size information for edge deployment analysis.
        
        Args:
            model: Specific model to analyze (defaults to self.model).
            
        Returns:
            Dictionary containing size metrics in MB.
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model available for size analysis")
        
        # Calculate model size
        total_params = model.count_params()
        
        # Estimate memory usage (rough approximation)
        # Assuming float32 (4 bytes per parameter)
        memory_mb = (total_params * 4) / (1024 * 1024)
        
        size_info = {
            "total_parameters": total_params,
            "memory_mb": memory_mb,
            "layers": len(model.layers),
        }
        
        logger.info(f"Model size analysis:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Estimated memory: {memory_mb:.2f} MB")
        logger.info(f"  Number of layers: {len(model.layers)}")
        
        return size_info
    
    def save_model(self, filepath: str, model: Optional[keras.Model] = None) -> None:
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model.
            model: Specific model to save (defaults to self.model).
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model available to save")
        
        model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> keras.Model:
        """Load a model from disk.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            Loaded Keras model.
        """
        model = keras.models.load_model(filepath)
        self.model = model
        logger.info(f"Model loaded from {filepath}")
        return model
