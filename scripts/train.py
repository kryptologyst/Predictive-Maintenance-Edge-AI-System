#!/usr/bin/env python3
"""Main training script for predictive maintenance system.

This script demonstrates the complete pipeline from data generation to model
training, optimization, and edge deployment.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from models.lstm_model import PredictiveMaintenanceModel, set_deterministic_seed
from pipelines.data_pipeline import DataPipeline
from export.edge_exporter import EdgeModelExporter
from utils.evaluation import ModelEvaluator, PerformanceBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train predictive maintenance model")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--time-steps", type=int, default=50, help="Number of time steps")
    parser.add_argument("--features", type=int, default=3, help="Number of sensor features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--export-formats", nargs="+", default=["tflite"], 
                       help="Export formats (tflite, onnx, coreml, openvino)")
    parser.add_argument("--output-dir", type=str, default="assets", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")
    parser.add_argument("--prune", action="store_true", help="Enable pruning")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    
    args = parser.parse_args()
    
    # Set deterministic seed
    set_deterministic_seed(args.seed)
    
    logger.info("Starting predictive maintenance training pipeline")
    logger.info(f"Configuration: {vars(args)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data pipeline
    data_pipeline = DataPipeline(
        time_steps=args.time_steps,
        features=args.features,
        seed=args.seed
    )
    
    # Generate dataset
    logger.info("Generating synthetic sensor data")
    X, y = data_pipeline.create_dataset(
        n_samples=args.n_samples,
        normal_ratio=0.7,
        fault_types=["gradual", "sudden", "intermittent"]
    )
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline.split_data(X, y)
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Initialize model
    model = PredictiveMaintenanceModel(
        time_steps=args.time_steps,
        features=args.features,
        seed=args.seed
    )
    
    # Build and train model
    logger.info("Building and training model")
    model.build_model()
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate base model
    logger.info("Evaluating base model")
    base_metrics = model.evaluate(X_test, y_test)
    
    # Create model variants for comparison
    model_variants = {"base": model.model}
    
    # Quantization
    if args.quantize:
        logger.info("Creating quantized model")
        try:
            quantized_model = model.quantize_model(X_train[:100])  # Use subset for calibration
            model_variants["quantized"] = quantized_model
            
            # Evaluate quantized model
            quantized_metrics = model.evaluate(X_test, y_test, quantized_model)
            logger.info(f"Quantized model metrics: {quantized_metrics}")
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    
    # Pruning
    if args.prune:
        logger.info("Creating pruned model")
        try:
            pruned_model = model.prune_model(X_train, y_train, sparsity=0.5)
            model_variants["pruned"] = pruned_model
            
            # Evaluate pruned model
            pruned_metrics = model.evaluate(X_test, y_test, pruned_model)
            logger.info(f"Pruned model metrics: {pruned_metrics}")
            
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
    
    # Export models
    logger.info("Exporting models to edge formats")
    exporter = EdgeModelExporter(output_dir / "models")
    
    exported_models = {}
    for variant_name, variant_model in model_variants.items():
        logger.info(f"Exporting {variant_name} model")
        
        try:
            exported = exporter.export_all_formats(
                variant_model,
                model_name=f"predictive_maintenance_{variant_name}",
                representative_data=X_train[:100]
            )
            exported_models[variant_name] = exported
            
        except Exception as e:
            logger.warning(f"Export failed for {variant_name}: {e}")
    
    # Performance benchmarking
    if args.benchmark:
        logger.info("Running performance benchmarks")
        
        benchmark = PerformanceBenchmark()
        evaluator = ModelEvaluator()
        
        # Benchmark model variants
        results_df = benchmark.benchmark_model_variants(
            model_variants,
            X_test,
            y_test,
            list(model_variants.keys())
        )
        
        # Create leaderboard
        leaderboard = benchmark.create_leaderboard(
            results_df,
            save_path=output_dir / "leaderboard.csv"
        )
        
        # Save detailed results
        results_df.to_csv(output_dir / "benchmark_results.csv", index=False)
        
        # Edge constraint analysis
        edge_constraints = data_pipeline.get_edge_constraints()
        compliance_results = {}
        
        for variant_name, variant_model in model_variants.items():
            compliance = evaluator.analyze_edge_constraints(
                variant_model, X_test, edge_constraints
            )
            compliance_results[variant_name] = compliance
        
        # Save compliance results
        compliance_df = pd.DataFrame(compliance_results).T
        compliance_df.to_csv(output_dir / "edge_compliance.csv")
        
        logger.info("Benchmark results saved to assets/")
    
    # Generate predictions for demo
    logger.info("Generating sample predictions")
    predictions, probabilities = model.predict_maintenance(X_test[:10])
    
    # Create prediction summary
    prediction_summary = []
    for i in range(10):
        prediction_summary.append({
            "sample_id": i + 1,
            "maintenance_required": bool(predictions[i]),
            "confidence": float(probabilities[i]),
            "actual_label": bool(y_test[i]),
            "correct": predictions[i] == y_test[i]
        })
    
    prediction_df = pd.DataFrame(prediction_summary)
    prediction_df.to_csv(output_dir / "sample_predictions.csv", index=False)
    
    logger.info("Sample predictions:")
    logger.info(f"\n{prediction_df.to_string(index=False)}")
    
    # Save model
    model.save_model(output_dir / "models" / "base_model.h5")
    
    # Create training summary
    summary = {
        "dataset_info": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "time_steps": args.time_steps,
            "features": args.features,
        },
        "model_info": {
            "architecture": "LSTM",
            "lstm_units": model.lstm_units,
            "dense_units": model.dense_units,
            "dropout_rate": model.dropout_rate,
        },
        "training_info": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "performance": base_metrics,
        "exported_formats": list(exported_models.get("base", {}).keys()),
    }
    
    # Save summary
    import json
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    
    # Print final summary
    print("\n" + "="*60)
    print("PREDICTIVE MAINTENANCE TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset: {len(X)} samples ({args.time_steps} time steps, {args.features} features)")
    print(f"Model: LSTM with {model.lstm_units} units")
    print(f"Training: {args.epochs} epochs, batch size {args.batch_size}")
    print(f"Test Accuracy: {base_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {base_metrics['f1_score']:.4f}")
    print(f"Exported Formats: {', '.join(summary['exported_formats'])}")
    print("="*60)


if __name__ == "__main__":
    main()
