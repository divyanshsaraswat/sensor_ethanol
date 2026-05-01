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

# Custom CSS for a premium feel and improved eye-space
custom_css = """
footer {visibility: hidden}
.gradio-container {
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
    background-color: #fcfcfd;
    max-width: 1200px !important;
}
.main-header {
    text-align: center;
    padding: 3.5rem 1rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    border-radius: 1rem;
    margin-bottom: 3rem;
    box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
}
.main-header h1 {
    font-size: 3.25rem !important;
    font-weight: 900;
    margin-bottom: 1rem;
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p {
    font-size: 1.25rem;
    opacity: 0.85;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}
.metadata-badge {
    background-color: rgba(96, 165, 250, 0.15);
    border: 1px solid rgba(96, 165, 250, 0.3);
    padding: 0.5rem 1.25rem;
    border-radius: 9999px;
    font-size: 0.95rem;
    font-weight: 600;
    color: #60a5fa;
    display: inline-block;
    margin-top: 1.75rem;
}
.section-card {
    background: white;
    padding: 2rem;
    border-radius: 1rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
    margin-bottom: 2rem;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.instruction-text {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2rem;
    line-height: 1.5;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", spacing_size="lg", radius_size="lg"), css=custom_css, head='<link rel="icon" href="/favicon.ico">') as demo:
    with gr.Column(elem_classes="main-header"):
        gr.Markdown("# VOC Predictor")
        gr.Markdown("High-precision ethanol quantification using physics-informed machine learning. Enter your sensor readings below to get an instant calculation.")
        gr.Markdown(f'<div class="metadata-badge">🚀 Active Engine: {metadata["name"]} (RMSE: {metadata["rmse"]:.4f})</div>', sanitize_html=False)
    
    with gr.Row(variant="compact"):
        with gr.Column(scale=3):
            with gr.Group(elem_classes="section-card"):
                gr.Markdown('<div class="section-title">🔍 Step 1: Input Sensor Data</div>')
                gr.Markdown('<p class="instruction-text">Ensure your sensors are calibrated. Values outside the training range will trigger a low-reliability warning.</p>')
                
                with gr.Row():
                    r_input = gr.Number(label="Refractive Index (R)", value=1.335, info="Target: 1.331 - 1.345")
                    cond_input = gr.Number(label="Conductivity (μS/cm)", value=25.0, info="Target: 16.3 - 36.1")
                    ph_input = gr.Number(label="pH Level", value=7.3, info="Target: 6.9 - 7.7")
                
                gr.Markdown('<div style="margin-top: 1.5rem"></div>')
                predict_btn = gr.Button("🚀 Calculate Ethanol Concentration", variant="primary", size="lg")
                
                gr.Markdown('<div style="margin-top: 2.5rem"></div>')
                gr.Markdown('<div class="section-title">📊 Step 2: Prediction Analysis</div>')
                
                with gr.Row():
                    output = gr.Textbox(label="Estimated Concentration (v/v)%", placeholder="Result...", interactive=False, scale=2)
                    reliability_out = gr.Textbox(label="Confidence Status", value="🟢 Ready", interactive=False, scale=3)
        
        with gr.Column(scale=2):
            with gr.Group(elem_classes="section-card"):
                gr.Markdown('<div class="section-title">🏆 Model Benchmarks</div>')
                gr.Markdown('<p class="instruction-text">We evaluate multiple architectures to ensure the highest accuracy for your specific chemical mixture.</p>')
                gr.DataFrame(summary_df, label=None, interactive=False)
                gr.Markdown("""
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 1rem">
                *Accuracy measured via Root Mean Squared Error (RMSE) on a 20% validation split.
                </div>
                """, sanitize_html=False)

    with gr.Accordion("🧬 Technical Methodology & Sensor Fusion", open=False):
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
