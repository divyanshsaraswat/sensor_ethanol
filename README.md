# VOC Sensor Fusion & Concentration Predictor

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Deployment-black?logo=vercel)](https://vercel.com)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)](https://gradio.app)

A high-accuracy multi-modal intelligent sensing system designed to quantify ethanol concentration (v/v)% using sensor fusion of **Refractive Index (RI)**, **Conductivity**, and **pH level**.

## 🚀 Overview

Single-feature sensing often fails in non-ideal liquid systems due to non-linear physical variations. This project leverages **Physics-Informed Machine Learning** to combine complementary sensor data, providing robust quantification across the full concentration range (0-60%+).

### Key Features
- **Multi-Modal Fusion**: Integrates RI, Conductivity, and pH for superior accuracy.
- **Automated Model Selection**: Evaluates multiple architectures (Polynomial, SVR, Neural Networks) and selects the best performer.
- **Intelligent Expert Fallback**: Dynamically routes to specialized "Expert" models if only a subset of sensors (e.g., just RI or just Conductivity) is available.
- **Real-time Reliability Validation**: UI provides instant feedback if sensor readings fall outside the calibrated training range.
- **Neural Latent Spaces**: Includes a Wide MLP (128 units) for learned embeddings of complex chemical interactions.
- **Premium Gradio UI**: Clean, light-mode design with training benchmarks and technical architecture details.

## 🛠️ Technology Stack
- **Environment**: [uv](https://github.com/astral-sh/uv) (Extremely fast Python package manager)
- **UI Framework**: [Gradio](https://gradio.app) (FastAPI-based UI)
- **Machine Learning**: Scikit-Learn (MLP, SVR, Random Forest, Polynomial Regression)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 🧠 Methodology & Intelligent Routing

The system addresses the limitations of individual sensors by combining their strengths:

- **Refractive Index (RI)**: Highly sensitive in the low concentration range (0–40%) but flattens out at higher concentrations.
- **Conductivity**: Highly sensitive in the mid-to-high range (40–60%+), providing an inverse trend that complements RI.
- **pH Level**: Adds a third dimension to compensate for chemical noise and temperature-induced drift.

### Prediction Modes
The interface dynamically adjusts its logic based on the sensors you have active:
1. **Fusion Mode (3 Sensors)**: Uses a Polynomial-D2 architecture to merge all features. This provides the lowest RMSE (0.179) and highest robustness.
2. **Expert Fallback (1-2 Sensors)**: If sensors are missing, the system engages specialized "Expert" models optimized for partial data.
    - **Conductivity Expert**: Highly precise (RMSE: 0.24) standalone predictor.
    - **RI Expert**: Optimized for low-concentration optical sensing.
    - **pH Expert**: Provides chemical baseline verification.

## 💻 Getting Started

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your machine.

### Installation
```bash
# Clone the repository
git clone https://github.com/divyanshsaraswat/sensor_ethanol.git
cd sensor_ethanol

# Install dependencies
uv sync
```

### Running the Application
To train the models and start the Gradio UI:
```bash
# Train the best model
uv run train_best_model.py

# Start the UI
uv run app.py
```
Access the UI at `http://127.0.0.1:7860`.

## 🌐 Deployment

This project is configured for seamless deployment on **Vercel**.

1. Connect your GitHub repository to Vercel.
2. Vercel will automatically detect the `vercel.json` and `requirements.txt`.
3. The app will be served as a serverless FastAPI application.

## 📁 Project Structure
- `app.py`: Main Gradio application entry point.
- `train_best_model.py`: Script to evaluate and save the highest-performing model.
- `fit.csv`: Dataset used for training and validation.
- `regressor_pipeline.py`: Original research pipeline and EDA.
- `latent_space_model.py`: Advanced neural network and latent space research.
- `vercel.json`: Configuration for Vercel deployment.
- `pyproject.toml`: Modern Python project definition.

---
Built with ❤️ for advanced chemical sensing research.
