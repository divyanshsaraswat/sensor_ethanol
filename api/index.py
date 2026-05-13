import os
import sys
import time
import joblib
from fastapi import FastAPI
import gradio as gr

# Global placeholders
pd = None
np = None
experts_cache = {}
model_registry = None
metadata = None
summary_data = []

# Environment overrides
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = "/tmp"

BASE_DIR = os.getcwd()
def get_path(filename): return os.path.join(BASE_DIR, filename)

def load_essentials():
    global model_registry, metadata, summary_data, pd, np
    # Lazy imports to speed up initial script boot
    if pd is None:
        import pandas as pd_internal
        pd = pd_internal
    if np is None:
        import numpy as np_internal
        np = np_internal
    
    if model_registry is not None: return
    model_registry = joblib.load(get_path('model_registry.joblib'))
    metadata = joblib.load(get_path('model_metadata.joblib'))
    summary_data = joblib.load(get_path('model_summary.joblib'))

def get_expert(registry_key):
    load_essentials()
    if registry_key in experts_cache: return experts_cache[registry_key]
    info = model_registry[registry_key]
    model = joblib.load(get_path(info['filename']))
    experts_cache[registry_key] = {'model': model, 'rmse': info['rmse'], 'features': info['features']}
    return experts_cache[registry_key]

RANGES = {'R': (1.331, 1.345), 'Cond': (16.3, 36.1), 'pH': (6.9, 7.7)}

def predict_concentration(r, cond, ph):
    load_essentials()
    provided_vals = {}
    if r is not None: provided_vals['R'] = r
    if cond is not None: provided_vals['Cond (?S/cm)'] = cond
    if ph is not None: provided_vals['pH'] = ph
    if not provided_vals: return "Waiting...", "⚪ Enter sensor data"
    
    registry_key = ",".join(sorted(provided_vals.keys()))
    expert_info = get_expert(registry_key)
    if not expert_info: return "Error", "🔴 Registry Error"

    ordered_vals = [provided_vals[f] for f in expert_info['features']]
    input_data = pd.DataFrame([ordered_vals], columns=expert_info['features'])
    
    out_of_bounds = []
    display_map = {'R': 'RI', 'Cond (?S/cm)': 'Cond', 'pH': 'pH'}
    for feat_name, val in provided_vals.items():
        range_key = 'R' if feat_name == 'R' else ('Cond' if 'Cond' in feat_name else 'pH')
        if not (RANGES[range_key][0] <= val <= RANGES[range_key][1]):
            out_of_bounds.append(display_map[feat_name])

    reliability = f"🟢 Expert Mode (RMSE: {expert_info['rmse']:.4f})"
    if out_of_bounds: reliability = f"⚠️ Low Reliability ({', '.join(out_of_bounds)})"

    try:
        prediction = expert_info['model'].predict(input_data)[0]
        prediction = max(0, min(100, prediction))
        return f"{prediction:.2f}%", reliability
    except: return "Error", "🔴 Prediction Failed"

custom_css = """
@keyframes pulse { 0% { opacity: 0.5; transform: scale(0.98); } 50% { opacity: 1; transform: scale(1); } 100% { opacity: 0.5; transform: scale(0.98); } }
.loader-pulse { animation: pulse 2s infinite ease-in-out; text-align: center; padding: 100px 0; max-width: 600px; margin: 0 auto; }
.gradio-container { font-family: 'Outfit', sans-serif; background-color: #f8fafc !important; }
.main-header { padding: 2rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 2rem; }
.section-card { background: white !important; padding: 2rem !important; border-radius: 12px !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
"""

def init_experience():
    # Phase 1: Libraries
    yield gr.update(visible=True), gr.update(visible=False), "📦 Loading Libraries...", "Importing Pandas & NumPy..."
    load_essentials()
    time.sleep(0.5)
    
    # Phase 2: Registry
    yield gr.update(visible=True), gr.update(visible=False), "🧠 Initializing Experts...", "Warming up the 7-sensor registry..."
    time.sleep(0.5)
    
    # Phase 3: Finalize
    yield gr.update(visible=True), gr.update(visible=False), "✨ Finalizing UI...", "Revealing the Sensing Dashboard..."
    time.sleep(0.5)
    
    # Reveal
    yield gr.update(visible=False), gr.update(visible=True), "", ""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo"), css=custom_css) as demo:
    # 1. LOADING SPLASH SCREEN
    with gr.Column(visible=True, elem_classes="loader-pulse") as loading_screen:
        gr.Markdown("# 🚀 System Boot")
        status_msg = gr.Markdown("## Initializing...")
        status_sub = gr.Markdown("Please wait while we prepare the sensing engine.")
    
    # 2. MAIN APPLICATION (Hidden initially)
    with gr.Column(visible=False) as main_app:
        with gr.Column(elem_classes="main-header"):
            gr.Markdown("# Ethanol Concentration Prediction")
            gr.Markdown("High-precision ethanol quantification using physics-informed machine learning.")
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group(elem_classes="section-card"):
                    with gr.Row():
                        r_input = gr.Number(label="Refractive Index (R)", value=1.335)
                        cond_input = gr.Number(label="Conductivity (μS/cm)", value=25.0)
                        ph_input = gr.Number(label="pH Level", value=7.3)
                    predict_btn = gr.Button("Calculate Concentration", variant="primary")
                    with gr.Row():
                        output = gr.Textbox(label="Concentration (v/v)%", interactive=False)
                        reliability_out = gr.Textbox(label="Status", value="Ready", interactive=False)
            
            with gr.Column(scale=2):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Model Benchmarks")
                    benchmark_table = gr.DataFrame(interactive=False)

    # REVEAL LOGIC
    demo.load(
        fn=init_experience,
        outputs=[loading_screen, main_app, status_msg, status_sub]
    ).then(
        fn=lambda: pd.DataFrame(summary_data).sort_values(by="RMSE") if summary_data else None,
        outputs=[benchmark_table]
    )

    predict_btn.click(
        fn=predict_concentration,
        inputs=[r_input, cond_input, ph_input],
        outputs=[output, reliability_out]
    )

app = gr.mount_gradio_app(FastAPI(), demo, path="/")

if __name__ == "__main__":
    demo.launch()
