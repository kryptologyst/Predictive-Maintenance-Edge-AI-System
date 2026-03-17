"""Unit tests for predictive maintenance system.

This module contains unit tests for the core components of the
predictive maintenance system.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.lstm_model import PredictiveMaintenanceModel, set_deterministic_seed
from pipelines.data_pipeline import DataPipeline, SensorDataGenerator, DataPreprocessor
from utils.evaluation import ModelEvaluator


class TestPredictiveMaintenanceModel:
    """Test cases for PredictiveMaintenanceModel."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = PredictiveMaintenanceModel(
            time_steps=50,
            features=3,
            lstm_units=32,
            dense_units=16
        )
        
        assert model.time_steps == 50
        assert model.features == 3
        assert model.lstm_units == 32
        assert model.dense_units == 16
        assert model.model is None
    
    def test_model_building(self):
        """Test model building."""
        model = PredictiveMaintenanceModel(time_steps=50, features=3)
        built_model = model.build_model()
        
        assert built_model is not None
        assert model.model is not None
        assert built_model.input_shape == (None, 50, 3)
        assert built_model.output_shape == (None, 1)
    
    def test_model_training(self):
        """Test model training."""
        set_deterministic_seed(42)
        
        model = PredictiveMaintenanceModel(time_steps=10, features=3)
        model.build_model()
        
        # Generate small training data
        X_train = np.random.randn(20, 10, 3)
        y_train = np.random.randint(0, 2, 20)
        
        # Train model
        history = model.train(X_train, y_train, epochs=2, verbose=0)
        
        assert history is not None
        assert len(history.history['loss']) == 2
        assert model.model is not None
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        set_deterministic_seed(42)
        
        model = PredictiveMaintenanceModel(time_steps=10, features=3)
        model.build_model()
        
        # Generate test data
        X_test = np.random.randn(10, 10, 3)
        y_test = np.random.randint(0, 2, 10)
        
        # Train briefly
        X_train = np.random.randn(20, 10, 3)
        y_train = np.random.randint(0, 2, 20)
        model.train(X_train, y_train, epochs=1, verbose=0)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_model_prediction(self):
        """Test model prediction."""
        set_deterministic_seed(42)
        
        model = PredictiveMaintenanceModel(time_steps=10, features=3)
        model.build_model()
        
        # Generate test data
        X_test = np.random.randn(5, 10, 3)
        
        # Train briefly
        X_train = np.random.randn(20, 10, 3)
        y_train = np.random.randint(0, 2, 20)
        model.train(X_train, y_train, epochs=1, verbose=0)
        
        # Predict
        predictions, probabilities = model.predict_maintenance(X_test)
        
        assert len(predictions) == 5
        assert len(probabilities) == 5
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)


class TestSensorDataGenerator:
    """Test cases for SensorDataGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SensorDataGenerator(time_steps=50, features=3)
        
        assert generator.time_steps == 50
        assert generator.features == 3
        assert len(generator.normal_mean) == 3
        assert len(generator.faulty_mean) == 3
    
    def test_normal_data_generation(self):
        """Test normal data generation."""
        generator = SensorDataGenerator(time_steps=10, features=3)
        data = generator.generate_normal_data(n_samples=5)
        
        assert data.shape == (5, 10, 3)
        assert isinstance(data, np.ndarray)
    
    def test_faulty_data_generation(self):
        """Test faulty data generation."""
        generator = SensorDataGenerator(time_steps=10, features=3)
        data = generator.generate_faulty_data(n_samples=5)
        
        assert data.shape == (5, 10, 3)
        assert isinstance(data, np.ndarray)
    
    def test_dataset_generation(self):
        """Test complete dataset generation."""
        generator = SensorDataGenerator(time_steps=10, features=3)
        X, y = generator.generate_dataset(n_samples=20)
        
        assert X.shape == (20, 10, 3)
        assert y.shape == (20,)
        assert len(np.unique(y)) == 2  # Binary classification
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)


class TestDataPipeline:
    """Test cases for DataPipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = DataPipeline(time_steps=50, features=3)
        
        assert pipeline.time_steps == 50
        assert pipeline.features == 3
        assert pipeline.generator is not None
        assert pipeline.preprocessor is not None
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        pipeline = DataPipeline(time_steps=10, features=3)
        X, y = pipeline.create_dataset(n_samples=20)
        
        assert X.shape == (20, 10, 3)
        assert y.shape == (20,)
        assert len(np.unique(y)) == 2
    
    def test_data_splitting(self):
        """Test data splitting."""
        pipeline = DataPipeline(time_steps=10, features=3)
        X, y = pipeline.create_dataset(n_samples=100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(X, y)
        
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
    
    def test_edge_constraints(self):
        """Test edge constraints."""
        pipeline = DataPipeline(time_steps=50, features=3)
        constraints = pipeline.get_edge_constraints()
        
        assert isinstance(constraints, dict)
        assert 'max_latency_ms' in constraints
        assert 'max_memory_mb' in constraints
        assert 'max_model_size_mb' in constraints


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor.scaler is not None
        assert not preprocessor.is_fitted
    
    def test_fit_transform(self):
        """Test fit and transform."""
        preprocessor = DataPreprocessor()
        X = np.random.randn(10, 5, 3)
        
        X_scaled = preprocessor.fit_transform(X)
        
        assert X_scaled.shape == X.shape
        assert preprocessor.is_fitted
        assert isinstance(X_scaled, np.ndarray)
    
    def test_transform(self):
        """Test transform with fitted scaler."""
        preprocessor = DataPreprocessor()
        X = np.random.randn(10, 5, 3)
        
        # Fit first
        preprocessor.fit_transform(X)
        
        # Transform new data
        X_new = np.random.randn(5, 5, 3)
        X_scaled = preprocessor.transform(X_new)
        
        assert X_scaled.shape == X_new.shape
        assert isinstance(X_scaled, np.ndarray)
    
    def test_inverse_transform(self):
        """Test inverse transform."""
        preprocessor = DataPreprocessor()
        X = np.random.randn(10, 5, 3)
        
        # Fit and transform
        X_scaled = preprocessor.fit_transform(X)
        
        # Inverse transform
        X_original = preprocessor.inverse_transform(X_scaled)
        
        assert X_original.shape == X.shape
        assert isinstance(X_original, np.ndarray)


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator is not None
    
    def test_model_quality_evaluation(self):
        """Test model quality evaluation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
        
        metrics = evaluator.evaluate_model_quality(y_true, y_pred, y_pred_proba)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics
        
        # Check metric ranges
        for metric, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_edge_efficiency_evaluation(self):
        """Test edge efficiency evaluation."""
        set_deterministic_seed(42)
        
        evaluator = ModelEvaluator()
        
        # Create a simple model for testing
        model = PredictiveMaintenanceModel(time_steps=10, features=3)
        model.build_model()
        
        # Generate test data
        X_test = np.random.randn(10, 10, 3)
        
        # Train briefly
        X_train = np.random.randn(20, 10, 3)
        y_train = np.random.randint(0, 2, 20)
        model.train(X_train, y_train, epochs=1, verbose=0)
        
        # Evaluate efficiency
        efficiency = evaluator.evaluate_edge_efficiency(model, X_test, num_runs=5)
        
        assert isinstance(efficiency, dict)
        assert 'mean_latency_ms' in efficiency
        assert 'throughput_fps' in efficiency
        assert 'model_size_mb' in efficiency
        
        # Check that metrics are positive
        for metric, value in efficiency.items():
            assert value > 0


def test_deterministic_seed():
    """Test deterministic seed setting."""
    set_deterministic_seed(42)
    
    # Generate some random numbers
    np_rand1 = np.random.randn(5)
    tf_rand1 = tf.random.normal((5,))
    
    # Reset seed and generate again
    set_deterministic_seed(42)
    np_rand2 = np.random.randn(5)
    tf_rand2 = tf.random.normal((5,))
    
    # Should be the same
    np.testing.assert_array_equal(np_rand1, np_rand2)
    np.testing.assert_array_equal(tf_rand1.numpy(), tf_rand2.numpy())


if __name__ == "__main__":
    pytest.main([__file__])
