"""Model export utilities for edge deployment.

This module handles exporting trained models to various edge deployment formats
including TensorFlow Lite, ONNX, CoreML, and OpenVINO.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class EdgeModelExporter:
    """Exporter for converting models to edge deployment formats.
    
    Supports TensorFlow Lite, ONNX, CoreML, and OpenVINO formats with
    optimization for edge devices.
    """
    
    def __init__(self, output_dir: str = "assets/models") -> None:
        """Initialize the edge model exporter.
        
        Args:
            output_dir: Directory to save exported models.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized EdgeModelExporter with output_dir={output_dir}")
    
    def export_to_tflite(
        self,
        model: keras.Model,
        filename: str = "model.tflite",
        quantization: str = "int8",
        representative_data: Optional[np.ndarray] = None,
        optimize_for_size: bool = True,
    ) -> str:
        """Export model to TensorFlow Lite format.
        
        Args:
            model: Keras model to export.
            filename: Output filename.
            quantization: Quantization type ("int8", "float16", "dynamic").
            representative_data: Representative data for quantization.
            optimize_for_size: Whether to optimize for size.
            
        Returns:
            Path to exported TFLite model.
        """
        logger.info(f"Exporting model to TensorFlow Lite format with {quantization} quantization")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        if optimize_for_size:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Apply quantization
        if quantization == "int8":
            if representative_data is None:
                raise ValueError("Representative data required for int8 quantization")
            
            converter.representative_dataset = lambda: self._representative_dataset_gen(representative_data)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
        elif quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        output_path = self.output_dir / filename
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        
        # Get model size
        model_size_mb = len(tflite_model) / (1024 * 1024)
        
        logger.info(f"TensorFlow Lite model exported to {output_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        return str(output_path)
    
    def export_to_onnx(
        self,
        model: keras.Model,
        filename: str = "model.onnx",
        opset_version: int = 11,
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            model: Keras model to export.
            filename: Output filename.
            opset_version: ONNX opset version.
            
        Returns:
            Path to exported ONNX model.
        """
        try:
            import tf2onnx
        except ImportError:
            logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
            raise ImportError("tf2onnx is required for ONNX export")
        
        logger.info(f"Exporting model to ONNX format (opset {opset_version})")
        
        # Convert to ONNX
        spec = (tf.TensorSpec((None, model.input_shape[1], model.input_shape[2]), tf.float32, name="input"),)
        output_path = self.output_dir / filename
        
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=opset_version,
            output_path=str(output_path)
        )
        
        # Get model size
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        logger.info(f"ONNX model exported to {output_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        return str(output_path)
    
    def export_to_coreml(
        self,
        model: keras.Model,
        filename: str = "model.mlmodel",
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> str:
        """Export model to CoreML format for iOS deployment.
        
        Args:
            model: Keras model to export.
            filename: Output filename.
            input_names: Input tensor names.
            output_names: Output tensor names.
            
        Returns:
            Path to exported CoreML model.
        """
        try:
            import coremltools as ct
        except ImportError:
            logger.error("coremltools not installed. Install with: pip install coremltools")
            raise ImportError("coremltools is required for CoreML export")
        
        logger.info("Exporting model to CoreML format")
        
        # Convert to CoreML
        coreml_model = ct.convert(
            model,
            inputs=[ct.TensorType(shape=(1, model.input_shape[1], model.input_shape[2]))],
            outputs=[ct.TensorType(name="output")],
            minimum_deployment_target=ct.target.iOS13,
        )
        
        # Save model
        output_path = self.output_dir / filename
        coreml_model.save(str(output_path))
        
        # Get model size
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        logger.info(f"CoreML model exported to {output_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        return str(output_path)
    
    def export_to_openvino(
        self,
        model: keras.Model,
        filename: str = "model.xml",
        precision: str = "FP32",
    ) -> str:
        """Export model to OpenVINO format.
        
        Args:
            model: Keras model to export.
            filename: Output filename.
            precision: Model precision ("FP32", "FP16", "INT8").
            
        Returns:
            Path to exported OpenVINO model.
        """
        try:
            import openvino as ov
        except ImportError:
            logger.error("openvino not installed. Install with: pip install openvino")
            raise ImportError("openvino is required for OpenVINO export")
        
        logger.info(f"Exporting model to OpenVINO format ({precision})")
        
        # Convert to OpenVINO
        ov_model = ov.convert_model(model)
        
        # Compile model
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, "CPU")
        
        # Save model
        output_path = self.output_dir / filename
        ov.save_model(ov_model, str(output_path))
        
        # Get model size
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        logger.info(f"OpenVINO model exported to {output_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        return str(output_path)
    
    def _representative_dataset_gen(self, data: np.ndarray):
        """Generate representative dataset for quantization.
        
        Args:
            data: Representative data samples.
            
        Yields:
            Representative data batches.
        """
        for i in range(min(100, len(data))):  # Use up to 100 samples
            yield [data[i:i+1].astype(np.float32)]
    
    def benchmark_model(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark exported model performance.
        
        Args:
            model_path: Path to exported model.
            input_shape: Input tensor shape.
            num_runs: Number of benchmark runs.
            warmup_runs: Number of warmup runs.
            
        Returns:
            Dictionary containing benchmark results.
        """
        logger.info(f"Benchmarking model: {model_path}")
        
        # Determine model format and load accordingly
        if model_path.endswith('.tflite'):
            return self._benchmark_tflite(model_path, input_shape, num_runs, warmup_runs)
        elif model_path.endswith('.onnx'):
            return self._benchmark_onnx(model_path, input_shape, num_runs, warmup_runs)
        elif model_path.endswith('.mlmodel'):
            return self._benchmark_coreml(model_path, input_shape, num_runs, warmup_runs)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
    
    def _benchmark_tflite(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int,
        warmup_runs: int,
    ) -> Dict[str, float]:
        """Benchmark TensorFlow Lite model."""
        import time
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input data
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup runs
        for _ in range(warmup_runs):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times_ms = [t * 1000 for t in times]  # Convert to milliseconds
        
        results = {
            "mean_latency_ms": np.mean(times_ms),
            "std_latency_ms": np.std(times_ms),
            "p50_latency_ms": np.percentile(times_ms, 50),
            "p95_latency_ms": np.percentile(times_ms, 95),
            "p99_latency_ms": np.percentile(times_ms, 99),
            "min_latency_ms": np.min(times_ms),
            "max_latency_ms": np.max(times_ms),
        }
        
        logger.info(f"TFLite benchmark results: {results}")
        
        return results
    
    def _benchmark_onnx(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int,
        warmup_runs: int,
    ) -> Dict[str, float]:
        """Benchmark ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX benchmarking")
        
        import time
        
        # Load ONNX model
        session = ort.InferenceSession(model_path)
        
        # Prepare input data
        input_name = session.get_inputs()[0].name
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup runs
        for _ in range(warmup_runs):
            session.run(None, {input_name: input_data})
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            session.run(None, {input_name: input_data})
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times_ms = [t * 1000 for t in times]
        
        results = {
            "mean_latency_ms": np.mean(times_ms),
            "std_latency_ms": np.std(times_ms),
            "p50_latency_ms": np.percentile(times_ms, 50),
            "p95_latency_ms": np.percentile(times_ms, 95),
            "p99_latency_ms": np.percentile(times_ms, 99),
            "min_latency_ms": np.min(times_ms),
            "max_latency_ms": np.max(times_ms),
        }
        
        logger.info(f"ONNX benchmark results: {results}")
        
        return results
    
    def _benchmark_coreml(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int,
        warmup_runs: int,
    ) -> Dict[str, float]:
        """Benchmark CoreML model."""
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("coremltools is required for CoreML benchmarking")
        
        import time
        
        # Load CoreML model
        model = ct.models.MLModel(model_path)
        
        # Prepare input data
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup runs
        for _ in range(warmup_runs):
            model.predict({"input": input_data})
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            model.predict({"input": input_data})
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times_ms = [t * 1000 for t in times]
        
        results = {
            "mean_latency_ms": np.mean(times_ms),
            "std_latency_ms": np.std(times_ms),
            "p50_latency_ms": np.percentile(times_ms, 50),
            "p95_latency_ms": np.percentile(times_ms, 95),
            "p99_latency_ms": np.percentile(times_ms, 99),
            "min_latency_ms": np.min(times_ms),
            "max_latency_ms": np.max(times_ms),
        }
        
        logger.info(f"CoreML benchmark results: {results}")
        
        return results
    
    def export_all_formats(
        self,
        model: keras.Model,
        model_name: str = "predictive_maintenance",
        representative_data: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """Export model to all supported formats.
        
        Args:
            model: Keras model to export.
            model_name: Base name for exported models.
            representative_data: Representative data for quantization.
            
        Returns:
            Dictionary mapping format names to file paths.
        """
        exported_models = {}
        
        try:
            # Export to TensorFlow Lite
            tflite_path = self.export_to_tflite(
                model,
                f"{model_name}.tflite",
                quantization="int8",
                representative_data=representative_data
            )
            exported_models["tflite"] = tflite_path
        except Exception as e:
            logger.warning(f"Failed to export to TensorFlow Lite: {e}")
        
        try:
            # Export to ONNX
            onnx_path = self.export_to_onnx(model, f"{model_name}.onnx")
            exported_models["onnx"] = onnx_path
        except Exception as e:
            logger.warning(f"Failed to export to ONNX: {e}")
        
        try:
            # Export to CoreML
            coreml_path = self.export_to_coreml(model, f"{model_name}.mlmodel")
            exported_models["coreml"] = coreml_path
        except Exception as e:
            logger.warning(f"Failed to export to CoreML: {e}")
        
        try:
            # Export to OpenVINO
            openvino_path = self.export_to_openvino(model, f"{model_name}.xml")
            exported_models["openvino"] = openvino_path
        except Exception as e:
            logger.warning(f"Failed to export to OpenVINO: {e}")
        
        logger.info(f"Exported models to formats: {list(exported_models.keys())}")
        
        return exported_models
