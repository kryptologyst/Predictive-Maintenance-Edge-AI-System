#!/usr/bin/env python3
"""Simple test script to verify the predictive maintenance system.

This script tests the basic functionality without heavy dependencies.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports."""
    try:
        from src.pipelines.data_pipeline import SensorDataGenerator, DataPreprocessor
        print("✅ Data pipeline imports successful")
        return True
    except Exception as e:
        print(f"❌ Data pipeline import failed: {e}")
        return False

def test_data_generation():
    """Test data generation."""
    try:
        from src.pipelines.data_pipeline import SensorDataGenerator
        
        generator = SensorDataGenerator(time_steps=10, features=3)
        X, y = generator.generate_dataset(n_samples=20)
        
        assert X.shape == (20, 10, 3)
        assert y.shape == (20,)
        assert len(np.unique(y)) == 2
        
        print("✅ Data generation successful")
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing."""
    try:
        from src.pipelines.data_pipeline import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        X = np.random.randn(10, 5, 3)
        
        X_scaled = preprocessor.fit_transform(X)
        assert X_scaled.shape == X.shape
        
        X_original = preprocessor.inverse_transform(X_scaled)
        assert X_original.shape == X.shape
        
        print("✅ Data preprocessing successful")
        return True
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    try:
        import yaml
        
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'device_configs' in config
            assert 'model_configs' in config
            assert 'training_configs' in config
            
            print("✅ Configuration loading successful")
            return True
        else:
            print("❌ Configuration file not found")
            return False
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Predictive Maintenance System")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_data_generation,
        test_data_preprocessing,
        test_config_loading,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
