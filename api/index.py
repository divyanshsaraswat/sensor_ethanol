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
    global model_registry, metadata, summary_data
    if model_registry is not None: return
    # Load ONLY the joblib data (very fast)
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
    global pd
    # HEAVY LIFTING HAPPENS HERE (Only once per session)
    if pd is None:
        import pandas as pd_internal
        pd = pd_internal
    
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
.gradio-container { font-family: 'Outfit', sans-serif; background-color: #f8fafc !important; border: none !important; }
.main-header { padding: 2rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 2rem; }
.section-card { background: white !important; padding: 2rem !important; border-radius: 12px !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: none !important; }
.metadata-badge { background-color: #e0e7ff; color: #4338ca; padding: 0.4rem 1rem; border-radius: 6px; font-size: 0.875rem; font-weight: 600; display: inline-block; margin-top: 1rem; }
/* Remove orange focus rings and borders */
* { outline: none !important; box-shadow: none !important; border-color: transparent !important; }
.gradio-container, .gr-group, .gr-box, .gr-form { border: none !important; }
.gradio-button.primary { background: #4f46e5 !important; color: white !important; border: none !important; }
.gradio-button.primary:hover { background: #4338ca !important; }
"""

def load_libs_step():
    """Step 1: Load heavy libraries."""
    global pd, np
    import pandas as pd_internal
    import numpy as np_internal
    pd = pd_internal
    np = np_internal
    return "🧠 Loading Model Experts...", "Warming up the 7-sensor registry..."

def load_models_step():
    """Step 2: Load expert registry."""
    load_essentials()
    return "✨ Finalizing UI...", "Preparing the sensing dashboard..."

def reveal_app_step():
    """Step 3: Reveal the main app."""
    # Build benchmark table data
    table_data = pd.DataFrame(summary_data).sort_values(by="RMSE") if summary_data else pd.DataFrame()
    return gr.update(visible=False), gr.update(visible=True), table_data

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
            metadata_badge = gr.Markdown('<div class="metadata-badge">Active Engine: Initializing...</div>', sanitize_html=False)
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group(elem_classes="section-card"):
                    with gr.Row():
                        r_input = gr.Number(label="Refractive Index (R)", value=1.335, info="Range: 1.331 - 1.345")
                        cond_input = gr.Number(label="Conductivity (μS/cm)", value=25.0, info="Range: 16.3 - 36.1")
                        ph_input = gr.Number(label="pH Level", value=7.3, info="Range: 6.9 - 7.7")
                    
                    predict_btn = gr.Button("Calculate Concentration", variant="primary")
                    
                    with gr.Row():
                        output = gr.Textbox(label="Concentration (v/v)%", interactive=False)
                        reliability_out = gr.Textbox(label="Status", value="Ready", interactive=False)
            
            with gr.Column(scale=2):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Model Benchmarks")
                    benchmark_table = gr.Markdown("Loading benchmarks...")

        # Methodology INSIDE the main app container
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
                    registry_info = gr.Markdown("### Expert Registry\nRegistry loading...")
                with gr.Column():
                    primary_engine_info = gr.Markdown("### Primary Engine\nEngine loading...")

    # CONSOLIDATED REVEAL LOGIC (Single request to prevent Vercel connection drops)
    def boot_sequence():
        # Step 1: Libraries & Essentials
        load_essentials()
        
        # Step 2: Prepare UI Data
        reg_text = f"### Expert Registry (7 Total)\n"
        reg_text += f"- **Full Fusion:** {model_registry['Cond (?S/cm),R,pH']['rmse']:.4f}\n"
        reg_text += f"- **Strong Pair:** {model_registry['Cond (?S/cm),R']['rmse']:.4f}\n"
        reg_text += f"- **Cond Expert:** {model_registry['Cond (?S/cm)']['rmse']:.4f}"
        
        engine_text = f"### Primary Engine\n"
        engine_text += f"- **Best Model:** {metadata['name']}\n"
        engine_text += f"- **Training RMSE:** {metadata['rmse']:.4f}"
        
        badge_html = f'<div class="metadata-badge">Active Engine: {metadata["name"]} (RMSE: {metadata["rmse"]:.4f})</div>'
        
        # Generate HIGH PERFORMANCE Markdown Table
        if summary_data:
            sorted_data = sorted(summary_data, key=lambda x: x['RMSE'])
            table_md = "| Model | RMSE | R² Score |\n| :--- | :--- | :--- |\n"
            for row in sorted_data:
                table_md += f"| {row['Model']} | {row['RMSE']:.4f} | {row['R2 Score']:.6f} |\n"
        else:
            table_md = "*No benchmark data available.*"
        
        return (
            gr.update(visible=False), # loading_screen
            gr.update(visible=True),  # main_app
            table_md, 
            badge_html, 
            reg_text, 
            engine_text
        )

    demo.load(
        fn=boot_sequence,
        outputs=[loading_screen, main_app, benchmark_table, metadata_badge, registry_info, primary_engine_info]
    )

    predict_btn.click(
        fn=predict_concentration,
        inputs=[r_input, cond_input, ph_input],
        outputs=[output, reliability_out]
    )

app = gr.mount_gradio_app(FastAPI(), demo, path="/")

if __name__ == "__main__":
    demo.launch()
