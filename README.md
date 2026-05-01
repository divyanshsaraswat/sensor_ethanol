# VOC Sensor Fusion & Concentration Predictor

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Deployment-black?logo=vercel)](https://vercel.com)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)](https://gradio.app)

A high-accuracy multi-modal intelligent sensing system designed to quantify ethanol concentration (v/v)% using sensor fusion of **Refractive Index (RI)**, **Conductivity**, and **pH level**.

## 🚀 Overview

Single-feature sensing often fails in non-ideal liquid systems due to non-linear physical variations. This project leverages **Physics-Informed Machine Learning** to combine complementary sensor data, providing robust quantification across the full concentration range (0-60%+).

### Key Features
- **Multi-Modal Fusion**: Integrates RI, Conductivity, and pH for superior accuracy.
- **Automated Model Selection**: Evaluates multiple architectures (Polynomial, SVR, Random Forest, and Neural Networks) and selects the best performer.
- **Neural Latent Spaces**: Includes a Wide MLP (128 units) for learned embeddings of complex chemical interactions.
- **Premium Gradio UI**: Real-time inference with training benchmarks and technical visualizations.

## 🛠️ Technology Stack
- **Environment**: [uv](https://github.com/astral-sh/uv) (Extremely fast Python package manager)
- **UI Framework**: [Gradio](https://gradio.app) (FastAPI-based UI)
- **Machine Learning**: Scikit-Learn (MLP, SVR, Random Forest, Polynomial Regression)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📋 Methodology

The system addresses the limitations of individual sensors:
- **Refractive Index (RI)**: Highly sensitive in the low concentration range (0–40%) but flattens out at higher concentrations.
- **Conductivity**: Highly sensitive in the mid-to-high range (40–60%+), providing an inverse trend that complements RI.
- **pH Level**: Adds a third dimension to compensate for chemical noise and temperature-induced drift.

**Best Model**: Polynomial Regression (Degree 2) or Wide MLP, depending on the dataset profile.

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
