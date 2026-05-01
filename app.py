import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load the model and features
model = joblib.load('best_model.joblib')
feature_names = joblib.load('features.joblib')
metadata = joblib.load('model_metadata.joblib')
summary_data = joblib.load('model_summary.joblib')

def predict_concentration(r, cond, ph):
    """
    Predicts ethanol concentration based on RI, Conductivity, and pH.
    """
    # Create input dataframe
    input_data = pd.DataFrame([[r, cond, ph]], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Ensure prediction is within realistic bounds (e.g., 0-100%)
    prediction = max(0, min(100, prediction))
    
    return f"{prediction:.2f}%"

# Convert summary data to DataFrame for Gradio table
summary_df = pd.DataFrame(summary_data).sort_values(by="RMSE")

# Custom CSS for a premium feel
custom_css = """
footer {visibility: hidden}
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: #f8fafc;
}
.main-header {
    text-align: center;
    padding: 2.5rem 0;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    border-radius: 0.75rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
}
.main-header h1 {
    font-size: 2.75rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
    background: linear-gradient(to right, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p {
    font-size: 1.15rem;
    opacity: 0.9;
    max-width: 800px;
    margin: 0 auto;
}
.metadata-badge {
    background-color: rgba(96, 165, 250, 0.1);
    border: 1px solid rgba(96, 165, 250, 0.2);
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #60a5fa;
    display: inline-block;
    margin-top: 1.25rem;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"), css=custom_css) as demo:
    with gr.Div(elem_classes="main-header"):
        gr.Markdown("# VOC Concentration Predictor")
        gr.Markdown("An intelligent multi-modal sensing system using physics-informed machine learning and latent space embeddings.")
        gr.Markdown(f'<div class="metadata-badge">✨ Optimized via {metadata["name"]} (RMSE: {metadata["rmse"]:.4f})</div>', sanitize_html=False)
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown('<div class="section-title">🔍 Real-time Inference</div>')
                with gr.Row():
                    r_input = gr.Number(label="Refractive Index (R)", value=1.33, info="Range: 1.33 - 1.37")
                    cond_input = gr.Number(label="Conductivity (μS/cm)", value=100.0, info="Range: 0 - 500")
                    ph_input = gr.Number(label="pH Level", value=7.0, info="Range: 4 - 10")
                
                predict_btn = gr.Button("🚀 Calculate Ethanol Concentration", variant="primary")
                
                gr.Markdown('<div style="margin-top: 1.5rem"></div>')
                output = gr.Textbox(label="Resulting Concentration (v/v)%", placeholder="Awaiting sensor inputs...", interactive=False)
        
        with gr.Column(scale=2):
            gr.Markdown('<div class="section-title">📊 Training Benchmarks</div>')
            gr.DataFrame(summary_df, label="Model Performance Summary", interactive=False)
            gr.Markdown("""
            *All models were evaluated using a 20% hold-out test set from the `fit.csv` dataset.*
            """)

    with gr.Accordion("🛠️ Technical Architecture & Methodology", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Sensor Fusion Logic
                - **Refractive Index (RI):** Dominant feature for low concentrations (0-40%).
                - **Conductivity:** Critical for high concentrations (40-60%) where RI sensitivity decreases.
                - **pH Level:** Adds dimensionality for improved chemical robustness.
                """)
            with gr.Column():
                gr.Markdown(f"""
                ### Active Intelligence
                - **Best Model:** {metadata["name"]}
                - **Training RMSE:** {metadata["rmse"]:.4f}
                - **R² Score:** {summary_df[summary_df['Model'] == metadata['name']]['R2 Score'].values[0]:.6f}
                """)

    predict_btn.click(
        fn=predict_concentration,
        inputs=[r_input, cond_input, ph_input],
        outputs=output
    )
    
    gr.Markdown("---")
    gr.Markdown("Built with ❤️ using Gradio & Scikit-Learn | Research Project: VOC_Model")

# Expose the FastAPI app for Vercel
app = demo.app

if __name__ == "__main__":
    demo.launch()
