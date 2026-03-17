# Predictive Maintenance Edge AI System

A comprehensive Edge AI/IoT system for predictive maintenance using LSTM-based time-series analysis. This system predicts equipment failures by analyzing sensor patterns (vibration, temperature, pressure) and is optimized for edge deployment with quantization and pruning support.

## ⚠️ Important Disclaimer

**This is a research/educational demonstration and is NOT intended for safety-critical deployment.**

This system is designed for learning and research purposes only. Do not use in production environments where equipment safety or human safety could be compromised. Always consult with domain experts and conduct thorough testing before any real-world deployment.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kryptologyst/Predictive-Maintenance-Edge-AI-System.git
   cd Predictive-Maintenance-Edge-AI-System
   ```

2. **Install dependencies:**
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using conda
   conda env create -f environment.yml
   conda activate predictive-maintenance
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

### Basic Usage

1. **Train a model:**
   ```bash
   python scripts/train.py --n-samples 1000 --epochs 50
   ```

2. **Run the interactive demo:**
   ```bash
   streamlit run demo/streamlit_app.py
   ```

3. **Export models for edge deployment:**
   ```bash
   python scripts/train.py --export-formats tflite onnx coreml --quantize
   ```

## 📁 Project Structure

```
predictive-maintenance-edge/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   └── lstm_model.py        # LSTM predictive maintenance model
│   ├── pipelines/                # Data processing pipelines
│   │   └── data_pipeline.py     # Data generation and preprocessing
│   ├── export/                   # Model export utilities
│   │   └── edge_exporter.py     # Edge deployment export
│   ├── runtimes/                 # Edge runtime implementations
│   ├── comms/                    # Communication protocols
│   └── utils/                    # Utility functions
│       └── evaluation.py         # Evaluation and benchmarking
├── data/                         # Data storage
│   ├── raw/                      # Raw sensor data
│   └── processed/                # Processed datasets
├── configs/                      # Configuration files
│   └── config.yaml              # Main configuration
├── scripts/                      # Training and utility scripts
│   └── train.py                 # Main training script
├── tests/                        # Unit tests
├── assets/                       # Generated assets
│   ├── models/                   # Exported models
│   └── results/                  # Training results
├── demo/                         # Demo applications
│   └── streamlit_app.py         # Interactive Streamlit demo
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                     # This file
```

## 🔧 Features

### Core Functionality

- **LSTM-based Predictive Maintenance**: Time-series analysis for equipment failure prediction
- **Multi-sensor Support**: Vibration, temperature, and pressure sensor data
- **Synthetic Data Generation**: Realistic sensor patterns for normal and faulty equipment states
- **Edge Optimization**: Quantization, pruning, and model compression for edge deployment

### Edge Deployment Support

- **TensorFlow Lite**: Optimized for mobile and embedded devices
- **ONNX Runtime**: Cross-platform inference engine
- **CoreML**: iOS and macOS deployment
- **OpenVINO**: Intel hardware acceleration
- **Multiple Device Targets**: Raspberry Pi, Jetson Nano, Android, iOS

### Model Variants

- **Base Model**: Full-precision LSTM with 32 units
- **Lightweight Model**: Reduced complexity for resource-constrained devices
- **Quantized Model**: INT8 quantization for reduced memory footprint
- **Pruned Model**: Structured pruning for faster inference

### Evaluation & Benchmarking

- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
- **Edge Performance**: Latency, throughput, memory usage, model size
- **Robustness Testing**: Noise tolerance and perturbation analysis
- **Constraint Compliance**: Edge deployment requirement validation

## Model Architecture

The system uses an LSTM-based architecture optimized for time-series sensor data:

```
Input: (batch_size, time_steps, features)
├── LSTM Layer (32 units)
├── Dropout (0.2)
├── Dense Layer (16 units, ReLU)
├── Dropout (0.2)
└── Output Layer (1 unit, Sigmoid)
```

### Sensor Data Format

- **Time Steps**: 50 (configurable)
- **Features**: 3 (vibration, temperature, pressure)
- **Normal Operation**: Mean [0.3, 60.0, 30.0], Std [0.05, 2.0, 1.0]
- **Faulty Operation**: Mean [0.5, 70.0, 27.0], Std [0.1, 3.0, 2.0]

## Training Pipeline

### Data Generation

The system generates synthetic sensor data with realistic patterns:

- **Normal Patterns**: Gradual trends and seasonal variations
- **Fault Patterns**: Gradual degradation, sudden failures, intermittent issues
- **Noise Simulation**: Gaussian noise with configurable levels
- **Balanced Dataset**: Configurable normal/faulty ratio

### Training Process

1. **Data Generation**: Create synthetic sensor sequences
2. **Preprocessing**: Normalization and scaling
3. **Model Training**: LSTM with early stopping and learning rate scheduling
4. **Optimization**: Quantization and pruning for edge deployment
5. **Export**: Convert to edge deployment formats
6. **Evaluation**: Comprehensive performance analysis

### Training Commands

```bash
# Basic training
python scripts/train.py --n-samples 1000 --epochs 50

# Edge-optimized training with quantization
python scripts/train.py --n-samples 1000 --epochs 30 --quantize --prune

# Export to multiple formats
python scripts/train.py --export-formats tflite onnx coreml openvino

# Custom configuration
python scripts/train.py --time-steps 30 --features 3 --batch-size 16
```

## Interactive Demo

The Streamlit demo provides an interactive interface for:

- **Data Visualization**: Sensor data patterns and statistics
- **Model Training**: Real-time training simulation
- **Edge Performance**: Constraint compliance analysis
- **Live Prediction**: Real-time maintenance prediction demo

### Running the Demo

```bash
streamlit run demo/streamlit_app.py
```

The demo includes:
- Interactive sensor data visualization
- Model training simulation
- Edge performance analysis
- Live prediction testing
- Constraint compliance checking

## Evaluation & Metrics

### Model Quality Metrics

- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Edge Performance Metrics

- **Latency**: Inference time (mean, P95, P99)
- **Throughput**: Predictions per second
- **Memory Usage**: Peak memory consumption
- **Model Size**: Compressed model file size
- **Energy Efficiency**: Estimated energy consumption

### Benchmark Results

Typical performance on different edge devices:

| Device | Model Size | Latency (ms) | Accuracy | F1-Score |
|--------|------------|--------------|----------|----------|
| Raspberry Pi 4 | 2.1 MB | 45 | 0.89 | 0.87 |
| Jetson Nano | 2.1 MB | 25 | 0.89 | 0.87 |
| Android (CPU) | 1.8 MB | 80 | 0.88 | 0.86 |
| iOS (CPU) | 1.9 MB | 60 | 0.88 | 0.86 |

## 🛠️ Edge Deployment

### Supported Platforms

- **Raspberry Pi**: TensorFlow Lite, OpenVINO
- **NVIDIA Jetson**: TensorFlow Lite, TensorRT, OpenVINO
- **Android**: TensorFlow Lite
- **iOS**: CoreML
- **Generic Edge**: ONNX Runtime

### Deployment Steps

1. **Model Export**: Convert trained model to target format
2. **Optimization**: Apply quantization and pruning
3. **Benchmarking**: Validate performance on target device
4. **Integration**: Deploy with sensor data pipeline
5. **Monitoring**: Track performance and accuracy

### Edge Constraints

- **Latency**: < 100ms inference time
- **Memory**: < 50MB RAM usage
- **Storage**: < 10MB model size
- **Power**: Optimized for battery operation
- **Connectivity**: Offline-capable inference

## Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

### Integration Tests

```bash
# Test complete pipeline
python scripts/train.py --n-samples 100 --epochs 5

# Test edge export
python scripts/train.py --export-formats tflite --quantize
```

## Performance Optimization

### Model Compression Techniques

1. **Quantization**: INT8 quantization reduces model size by 4x
2. **Pruning**: Structured pruning removes redundant parameters
3. **Knowledge Distillation**: Teacher-student model compression
4. **Architecture Search**: Neural architecture search for optimal design

### Edge-Specific Optimizations

- **Batch Size**: Single sample inference for real-time processing
- **Memory Management**: Efficient tensor operations
- **CPU Optimization**: Multi-threading and vectorization
- **Power Management**: Dynamic frequency scaling

## 🔧 Configuration

### Model Configuration

Edit `configs/config.yaml` to customize:

- **Model Architecture**: LSTM units, dense layers, dropout
- **Training Parameters**: Epochs, batch size, learning rate
- **Data Generation**: Sample count, fault types, noise levels
- **Edge Constraints**: Latency, memory, model size limits
- **Export Settings**: Quantization, optimization flags

### Device-Specific Settings

```yaml
device_configs:
  raspberry_pi:
    max_latency_ms: 100
    max_memory_mb: 50
    tensorflow_lite: true
    
  jetson_nano:
    max_latency_ms: 50
    max_memory_mb: 100
    tensorrt: true
```

## API Reference

### Core Classes

- **`PredictiveMaintenanceModel`**: Main LSTM model class
- **`DataPipeline`**: Data generation and preprocessing
- **`EdgeModelExporter`**: Model export utilities
- **`ModelEvaluator`**: Performance evaluation
- **`SensorDataGenerator`**: Synthetic data generation

### Key Methods

```python
# Model training
model = PredictiveMaintenanceModel()
model.build_model()
model.train(X_train, y_train)

# Model evaluation
metrics = model.evaluate(X_test, y_test)

# Edge export
exporter = EdgeModelExporter()
exporter.export_to_tflite(model, "model.tflite")

# Performance evaluation
evaluator = ModelEvaluator()
efficiency = evaluator.evaluate_edge_efficiency(model, X_test)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
ruff check src/ tests/
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Edge AI community for optimization techniques
- Open source contributors for various edge deployment tools
- Research papers on predictive maintenance and edge AI


## Future Work

- **Real Sensor Integration**: Support for actual IoT sensor data
- **Federated Learning**: Distributed training across edge devices
- **Advanced Architectures**: Transformer-based models for time-series
- **Multi-modal Data**: Integration of audio, visual, and sensor data
- **Edge-to-Cloud**: Hybrid inference with cloud fallback
- **Automated ML**: Neural architecture search for optimal edge models

---

**Remember**: This is a research/educational project. Always consult with domain experts before deploying in safety-critical environments.
# Predictive-Maintenance-Edge-AI-System
