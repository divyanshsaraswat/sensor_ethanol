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
    # Load the registry and baseline metadata
    model_registry = joblib.load(get_path('model_registry.joblib'))
    metadata = joblib.load(get_path('model_metadata.joblib'))
    summary_data = joblib.load(get_path('model_summary.joblib'))
    
    # Pre-load all expert models from the registry
    experts = {}
    for key, info in model_registry.items():
        experts[key] = {
            'model': joblib.load(get_path(info['filename'])),
            'rmse': info['rmse'],
            'features': info['features']
        }
    print("DEBUG: Model Registry and all 7 Experts loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load models: {str(e)}")
    metadata = {"name": "Error Loading Model", "rmse": 0.0}
    summary_data, model_registry, experts = [], {}, {}

# Calibration Ranges (from fit.csv)
RANGES = {
    'R': (1.331, 1.345),
    'Cond': (16.3, 36.1),
    'pH': (6.9, 7.7)
}

def predict_concentration(r, cond, ph):
    """
    Intelligently routes prediction to the exact expert model for any sensor combination.
    """
    if not experts:
        return "Error: System Offline", "🔴 Models not loaded"
    
    # Identify provided features
    provided_vals = {}
    if r is not None: provided_vals['R'] = r
    if cond is not None: provided_vals['Cond (?S/cm)'] = cond
    if ph is not None: provided_vals['pH'] = ph

    if not provided_vals:
        return "Waiting...", "⚪ Please enter sensor data"

    # Generate registry key (sorted features)
    registry_key = ",".join(sorted(provided_vals.keys()))
    expert_info = experts.get(registry_key)

    if not expert_info:
        return "Error", f"🔴 No expert for: {registry_key}"

    # Prepare input data in the correct order for this specific expert
    ordered_vals = [provided_vals[f] for f in expert_info['features']]
    input_data = pd.DataFrame([ordered_vals], columns=expert_info['features'])
    
    # Reliability check
    out_of_bounds = []
    # Map back to display names for reliability check
    display_map = {'R': 'RI', 'Cond (?S/cm)': 'Cond', 'pH': 'pH'}
    for feat_name, val in provided_vals.items():
        range_key = 'R' if feat_name == 'R' else ('Cond' if 'Cond' in feat_name else 'pH')
        if not (RANGES[range_key][0] <= val <= RANGES[range_key][1]):
            out_of_bounds.append(display_map[feat_name])

    mode_name = f"{len(provided_vals)}-Sensor Expert"
    reliability = f"🟢 {mode_name} (RMSE: {expert_info['rmse']:.4f})"
    if out_of_bounds:
        reliability = f"⚠️ Low Reliability ({', '.join(out_of_bounds)} Out-of-Range)"

    # Make prediction
    try:
        prediction = expert_info['model'].predict(input_data)[0]
        prediction = max(0, min(100, prediction))
        result = f"{prediction:.2f}%"
    except Exception as e:
        result = "Error"
        reliability = f"🔴 prediction Failed: {str(e)}"
    
    return result, reliability

# Convert summary data to DataFrame for Gradio table
if summary_data:
    summary_df = pd.DataFrame(summary_data).sort_values(by="RMSE")
else:
    summary_df = pd.DataFrame(columns=["Model", "RMSE", "R2 Score"])

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
<title>Ethanol Concentration Prediction | Precision Sensing</title>
<meta name="description" content="Intelligent multi-modal sensor fusion for ethanol concentration prediction using RI, Conductivity, and pH.">
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="shortcut icon" type="image/x-icon" href="/favicon.ico">
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate", font=["Outfit", "sans-serif"]), css=custom_css, head=head_html) as demo:
    with gr.Column(elem_classes="main-header"):
        gr.Markdown("# Ethanol Concentration Prediction")
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
                - **Refractive Index (RI):** Dominant for low range (0-40%).
                - **Conductivity:** Critical for high range (40-60%+).
                - **pH Level:** Robustness against chemical interference.
                """)
            with gr.Column():
                # Extract some key expert RMSEs for the UI
                get_rmse = lambda k: f"{experts[k]['rmse']:.4f}" if k in experts else "N/A"
                gr.Markdown(f"""
                ### Expert Registry (7 Total)
                System selects the optimal expert for any input subset:
                - **Full Fusion (R,C,pH):** {get_rmse('Cond (?S/cm),R,pH')}
                - **Strong Pair (R,Cond):** {get_rmse('Cond (?S/cm),R')}
                - **Cond Expert:** {get_rmse('Cond (?S/cm)')}
                """)
            with gr.Column():
                r2_val = summary_df[summary_df['Model'] == metadata['name']]['R2 Score'].values[0] if not summary_df.empty else 0.0
                gr.Markdown(f"""
                ### Primary Engine
                - **Global Best:** {metadata["name"]}
                - **Validation R²:** {r2_val:.6f}
                - **Training RMSE:** {metadata["rmse"]:.4f}
                """)

    predict_btn.click(
        fn=predict_concentration,
        inputs=[r_input, cond_input, ph_input],
        outputs=[output, reliability_out]
    )
    
    gr.Markdown("---")
    gr.Markdown("Built with ❤️ using Gradio & Scikit-Learn | Research Project: Ethanol_Model")

from fastapi import FastAPI

# Create a FastAPI app and mount Gradio to it
# This is more robust for serverless environments (like Vercel) as it ensures
# that the Gradio configuration is properly initialized before the first request.
main_app = FastAPI()
app = gr.mount_gradio_app(main_app, demo, path="/")

if __name__ == "__main__":
    demo.launch()
