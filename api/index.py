import os
import sys
import joblib
import pandas as pd
import numpy as np
import gradio as gr

# Environment overrides for Serverless compatibility
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = "/tmp"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

# Robust path handling for Vercel
# On Vercel, the app runs from the root, but the file is in api/
BASE_DIR = os.getcwd()

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

print(f"DEBUG: BASE_DIR is {BASE_DIR}")
print(f"DEBUG: Looking for model in {get_path('best_model.joblib')}")

try:
    # Load the model and features
    model = joblib.load(get_path('best_model.joblib'))
    feature_names = joblib.load(get_path('features.joblib'))
    metadata = joblib.load(get_path('model_metadata.joblib'))
    summary_data = joblib.load(get_path('model_summary.joblib'))
    print("DEBUG: All models loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load models: {str(e)}")
    # Fallback/Dummy metadata to prevent crash during import
    metadata = {"name": "Error Loading Model", "rmse": 0.0}
    summary_data = []
    feature_names = []
    model = None

# Calibration Ranges (from fit.csv)
RANGES = {
    'R': (1.331, 1.345),
    'Cond': (16.3, 36.1),
    'pH': (6.9, 7.7)
}

def predict_concentration(r, cond, ph):
    """
    Predicts ethanol concentration and checks input reliability.
    """
    if model is None:
        return "Error: Model not loaded.", "🔴 System Offline"
    
    # Check reliability
    out_of_bounds = []
    if not (RANGES['R'][0] <= r <= RANGES['R'][1]):
        out_of_bounds.append("RI")
    if not (RANGES['Cond'][0] <= cond <= RANGES['Cond'][1]):
        out_of_bounds.append("Conductivity")
    if not (RANGES['pH'][0] <= ph <= RANGES['pH'][1]):
        out_of_bounds.append("pH")
    
    reliability = "🟢 High Reliability (Within Calibration)"
    if out_of_bounds:
        reliability = f"⚠️ Low Reliability: {', '.join(out_of_bounds)} outside training range"

    # Create input dataframe
    input_data = pd.DataFrame([[r, cond, ph]], columns=feature_names)
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        # Ensure prediction is within realistic bounds (e.g., 0-100%)
        prediction = max(0, min(100, prediction))
        result = f"{prediction:.2f}%"
    except Exception as e:
        result = "Error during prediction"
        reliability = f"🔴 Prediction Failed: {str(e)}"
    
    return result, reliability

# Convert summary data to DataFrame for Gradio table
summary_df = pd.DataFrame(summary_data).sort_values(by="RMSE")

# Custom CSS for a clean, professional, and high-end aesthetic
custom_css = """
footer {visibility: hidden}
.gradio-container {
    font-family: 'Outfit', 'Inter', sans-serif;
    background-color: #f8fafc !important;
    color: #1e293b;
}
.main-header {
    text-align: left;
    padding: 3rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid #e2e8f0;
}
.main-header h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.025em;
}
.main-header p {
    font-size: 1.125rem;
    color: #64748b;
    max-width: 800px;
}
.metadata-badge {
    background-color: #e0e7ff;
    color: #4338ca;
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 1rem;
}
.section-card {
    background: white !important;
    padding: 2rem !important;
    border-radius: 12px !important;
    border: 1px solid #f1f5f9 !important;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1) !important;
}
.section-title {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin-bottom: 1rem !important;
}
.instruction-text {
    color: #94a3b8;
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
}
.gradio-button.primary {
    background: #4f46e5 !important;
    border: none !important;
    font-weight: 600 !important;
}
.gradio-button.primary:hover {
    background: #4338ca !important;
}
"""

head_html = """
<title>VOC Predictor | Precision Ethanol Sensing</title>
<meta name="description" content="Intelligent multi-modal sensor fusion for ethanol concentration prediction using RI, Conductivity, and pH.">
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="shortcut icon" type="image/x-icon" href="/favicon.ico">
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate", font=["Outfit", "sans-serif"]), css=custom_css, head=head_html) as demo:
    with gr.Column(elem_classes="main-header"):
        gr.Markdown("# VOC Predictor")
        gr.Markdown("High-precision ethanol quantification using physics-informed machine learning. Monitor sensor health and concentration levels in real-time.")
        gr.Markdown(f'<div class="metadata-badge">Active Engine: {metadata["name"]} (RMSE: {metadata["rmse"]:.4f})</div>', sanitize_html=False)
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes="section-card"):
                gr.Markdown('<div class="section-title">Inference Engine</div>')
                gr.Markdown('<p class="instruction-text">Enter real-time sensor data below. The system will automatically validate the reliability of your input ranges.</p>')
                
                with gr.Row():
                    r_input = gr.Number(label="Refractive Index (R)", value=1.335, info="Range: 1.331 - 1.345")
                    cond_input = gr.Number(label="Conductivity (μS/cm)", value=25.0, info="Range: 16.3 - 36.1")
                    ph_input = gr.Number(label="pH Level", value=7.3, info="Range: 6.9 - 7.7")
                
                predict_btn = gr.Button("Calculate Concentration", variant="primary")
                
                gr.Markdown('<div style="margin-top: 2rem; border-top: 1px solid #f1f5f9; padding-top: 2rem;"></div>')
                
                with gr.Row():
                    output = gr.Textbox(label="Ethanol Concentration (v/v)%", placeholder="Waiting for input...", interactive=False)
                    reliability_out = gr.Textbox(label="Confidence Status", value="Ready", interactive=False)
        
        with gr.Column(scale=2):
            with gr.Group(elem_classes="section-card"):
                gr.Markdown('<div class="section-title">Model Benchmarks</div>')
                gr.DataFrame(summary_df, label=None, interactive=False)
                gr.Markdown("""
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 1rem">
                Models evaluated on a 20% validation split. RMSE values represent the root mean squared error.
                </div>
                """, sanitize_html=False)

    with gr.Accordion("Methodology Details", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Sensor Fusion Logic
                - **Refractive Index (RI):** Dominant feature for low concentrations (0-40%).
                - **Conductivity:** Critical for high concentrations (40-60%) where RI sensitivity decreases.
                - **pH Level:** Adds dimensionality for improved chemical robustness.
                """)
            with gr.Column():
                r2_val = summary_df[summary_df['Model'] == metadata['name']]['R2 Score'].values[0] if not summary_df.empty else 0.0
                gr.Markdown(f"""
                ### Active Intelligence
                - **Best Model:** {metadata["name"]}
                - **Training RMSE:** {metadata["rmse"]:.4f}
                - **R² Score:** {r2_val:.6f}
                """)

    predict_btn.click(
        fn=predict_concentration,
        inputs=[r_input, cond_input, ph_input],
        outputs=[output, reliability_out]
    )
    
    gr.Markdown("---")
    gr.Markdown("Built with ❤️ using Gradio & Scikit-Learn | Research Project: VOC_Model")

from fastapi import FastAPI

# Create a FastAPI app and mount Gradio to it
# This is more robust for serverless environments (like Vercel) as it ensures
# that the Gradio configuration is properly initialized before the first request.
main_app = FastAPI()
app = gr.mount_gradio_app(main_app, demo, path="/")

if __name__ == "__main__":
    demo.launch()
