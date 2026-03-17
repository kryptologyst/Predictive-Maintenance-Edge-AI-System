"""Streamlit demo for predictive maintenance system.

This demo simulates edge deployment constraints and provides an interactive
interface for testing the predictive maintenance model.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.lstm_model import PredictiveMaintenanceModel, set_deterministic_seed
from pipelines.data_pipeline import DataPipeline, SensorDataGenerator
from utils.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Edge AI Demo",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model() -> Optional[PredictiveMaintenanceModel]:
    """Load the trained model."""
    try:
        model_path = Path("assets/models/base_model.h5")
        if model_path.exists():
            model = PredictiveMaintenanceModel()
            model.load_model(str(model_path))
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def generate_sample_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample sensor data for demo."""
    generator = SensorDataGenerator(seed=42)
    X, y = generator.generate_dataset(n_samples=n_samples)
    return X, y


def create_sensor_plot(sensor_data: np.ndarray, sample_idx: int = 0) -> go.Figure:
    """Create interactive plot of sensor data."""
    time_steps = np.arange(sensor_data.shape[1])
    
    fig = go.Figure()
    
    # Add sensor traces
    sensor_names = ["Vibration", "Temperature", "Pressure"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, (name, color) in enumerate(zip(sensor_names, colors)):
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=sensor_data[sample_idx, :, i],
            mode='lines',
            name=name,
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title=f"Sensor Data - Sample {sample_idx + 1}",
        xaxis_title="Time Steps",
        yaxis_title="Sensor Values",
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_performance_plot(metrics: Dict[str, float]) -> go.Figure:
    """Create performance metrics visualization."""
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        )
    ])
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metrics",
        yaxis_title="Values",
        height=400
    )
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔧 Predictive Maintenance Edge AI Demo</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This is a research/educational demonstration and is NOT intended for safety-critical deployment.</strong></p>
        <p>This system is designed for learning and research purposes only. Do not use in production environments 
        where equipment safety or human safety could be compromised.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    time_steps = st.sidebar.slider("Time Steps", 10, 100, 50)
    features = st.sidebar.slider("Sensor Features", 1, 5, 3)
    
    # Data generation settings
    st.sidebar.subheader("Data Generation")
    n_samples = st.sidebar.slider("Number of Samples", 50, 500, 100)
    normal_ratio = st.sidebar.slider("Normal Operation Ratio", 0.5, 0.9, 0.7)
    
    # Edge constraints
    st.sidebar.subheader("Edge Constraints")
    max_latency_ms = st.sidebar.slider("Max Latency (ms)", 10, 1000, 100)
    max_memory_mb = st.sidebar.slider("Max Memory (MB)", 10, 200, 50)
    max_model_size_mb = st.sidebar.slider("Max Model Size (MB)", 1, 50, 10)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Visualization", "🤖 Model Training", "⚡ Edge Performance", "🔍 Live Demo"])
    
    with tab1:
        st.header("Sensor Data Visualization")
        
        # Generate sample data
        with st.spinner("Generating sample sensor data..."):
            X, y = generate_sample_data(n_samples)
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(X))
        with col2:
            st.metric("Normal Samples", int(np.sum(y == 0)))
        with col3:
            st.metric("Faulty Samples", int(np.sum(y == 1)))
        with col4:
            st.metric("Fault Rate", f"{np.mean(y):.1%}")
        
        # Sample selection
        sample_idx = st.selectbox("Select Sample to Visualize", range(len(X)))
        
        # Sensor data plot
        fig = create_sensor_plot(X, sample_idx)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data statistics
        st.subheader("Data Statistics")
        
        stats_data = []
        sensor_names = ["Vibration", "Temperature", "Pressure"]
        
        for i, name in enumerate(sensor_names):
            if i < X.shape[2]:
                stats_data.append({
                    "Sensor": name,
                    "Mean": f"{np.mean(X[:, :, i]):.3f}",
                    "Std": f"{np.std(X[:, :, i]):.3f}",
                    "Min": f"{np.min(X[:, :, i]):.3f}",
                    "Max": f"{np.max(X[:, :, i]):.3f}"
                })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    
    with tab2:
        st.header("Model Training Simulation")
        
        # Training configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            lstm_units = st.slider("LSTM Units", 16, 128, 32)
            dense_units = st.slider("Dense Units", 8, 64, 16)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        
        with col2:
            st.subheader("Training Parameters")
            epochs = st.slider("Epochs", 10, 100, 50)
            batch_size = st.slider("Batch Size", 8, 64, 32)
            learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1])
        
        # Train model button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Set deterministic seed
                set_deterministic_seed(42)
                
                # Initialize model
                model = PredictiveMaintenanceModel(
                    time_steps=time_steps,
                    features=features,
                    lstm_units=lstm_units,
                    dense_units=dense_units,
                    dropout_rate=dropout_rate
                )
                
                # Generate training data
                data_pipeline = DataPipeline(time_steps=time_steps, features=features)
                X_train, y_train = data_pipeline.create_dataset(n_samples=n_samples)
                X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline.split_data(X_train, y_train)
                
                # Build and train model
                model.build_model()
                history = model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                
                # Evaluate model
                evaluator = ModelEvaluator()
                metrics = model.evaluate(X_test, y_test)
                
                # Store model in session state
                st.session_state.model = model
                st.session_state.test_data = (X_test, y_test)
                st.session_state.metrics = metrics
                st.session_state.training_history = history.history
            
            st.success("Model training completed!")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            
            # Training history plot
            if 'training_history' in st.session_state:
                history = st.session_state.training_history
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['loss'], mode='lines', name='Training Loss'
                ))
                fig.add_trace(go.Scatter(
                    y=history['val_loss'], mode='lines', name='Validation Loss'
                ))
                
                fig.update_layout(
                    title="Training History",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Edge Performance Analysis")
        
        if 'model' in st.session_state and 'test_data' in st.session_state:
            model = st.session_state.model
            X_test, y_test = st.session_state.test_data
            
            # Edge constraints
            constraints = {
                "max_latency_ms": max_latency_ms,
                "max_memory_mb": max_memory_mb,
                "max_model_size_mb": max_model_size_mb,
                "min_throughput_fps": 1.0
            }
            
            # Performance analysis
            evaluator = ModelEvaluator()
            
            # Efficiency metrics
            efficiency_metrics = evaluator.evaluate_edge_efficiency(model, X_test)
            
            # Constraint compliance
            compliance = evaluator.analyze_edge_constraints(model, X_test, constraints)
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                
                perf_data = [
                    ("Mean Latency", f"{efficiency_metrics['mean_latency_ms']:.2f} ms"),
                    ("P95 Latency", f"{efficiency_metrics['p95_latency_ms']:.2f} ms"),
                    ("Throughput", f"{efficiency_metrics['throughput_fps']:.2f} FPS"),
                    ("Model Size", f"{efficiency_metrics['model_size_mb']:.2f} MB"),
                    ("Memory Usage", f"{efficiency_metrics['mean_memory_mb']:.2f} MB"),
                ]
                
                for metric, value in perf_data:
                    st.metric(metric, value)
            
            with col2:
                st.subheader("Constraint Compliance")
                
                compliance_data = [
                    ("Latency", compliance['latency_compliant']),
                    ("Memory", compliance['memory_compliant']),
                    ("Model Size", compliance['size_compliant']),
                    ("Throughput", compliance['throughput_compliant']),
                ]
                
                for constraint, compliant in compliance_data:
                    status = "✅" if compliant else "❌"
                    st.write(f"{status} {constraint}")
                
                # Overall compliance
                all_compliant = all(compliance.values())
                if all_compliant:
                    st.markdown("""
                    <div class="success-box">
                        <h4>✅ Edge Compatible</h4>
                        <p>Model meets all edge deployment constraints!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Edge Constraints Not Met</h4>
                        <p>Model may not be suitable for edge deployment.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Performance visualization
            fig = create_performance_plot(efficiency_metrics)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please train a model first in the 'Model Training' tab.")
    
    with tab4:
        st.header("Live Maintenance Prediction Demo")
        
        if 'model' in st.session_state:
            model = st.session_state.model
            
            # Generate new sample
            if st.button("🎲 Generate New Sample"):
                generator = SensorDataGenerator(time_steps=time_steps, features=features)
                
                # Randomly choose normal or faulty
                is_faulty = np.random.choice([0, 1], p=[0.7, 0.3])
                
                if is_faulty:
                    sample_data = generator.generate_faulty_data(1, fault_type="gradual")
                    actual_label = "Maintenance Required"
                else:
                    sample_data = generator.generate_normal_data(1)
                    actual_label = "Normal Operation"
                
                st.session_state.demo_sample = sample_data[0]
                st.session_state.demo_label = actual_label
            
            if 'demo_sample' in st.session_state:
                sample_data = st.session_state.demo_sample
                actual_label = st.session_state.demo_label
                
                # Display sensor data
                fig = create_sensor_plot(sample_data.reshape(1, -1, features), 0)
                st.plotly_chart(fig, use_container_width=True)
                
                # Make prediction
                prediction, probability = model.predict_maintenance(sample_data.reshape(1, -1, features))
                
                pred_label = "Maintenance Required" if prediction[0] else "Normal Operation"
                confidence = probability[0]
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", pred_label)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                with col3:
                    st.metric("Actual", actual_label)
                
                # Prediction accuracy
                correct = (prediction[0] == (actual_label == "Maintenance Required"))
                if correct:
                    st.success("✅ Prediction is correct!")
                else:
                    st.error("❌ Prediction is incorrect!")
                
                # Confidence visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Prediction Confidence (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please train a model first in the 'Model Training' tab.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>Predictive Maintenance Edge AI Demo | Research/Educational Use Only</p>
        <p>⚠️ Not for safety-critical deployment</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
