import json
import os
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from pathlib import Path

# --- Load Data ---
def load_data():
    root_dir = Path(__file__).parent.absolute()
    summary_path = root_dir / "summary_report.json"
    scored_path = root_dir / "scored_responses.json"

    # Mock data fallback
    summary_data = {
        "behavioral_fingerprint": 0.0,
        "dimensions": {
            "sycophancy": {"model_a": 0.0, "model_b": 0.0, "distance": 0.0},
            "refusal_rate": {"model_a": 0.0, "model_b": 0.0, "distance": 0.0},
            "hallucination": {"model_a": 0.0, "model_b": 0.0, "distance": 0.0},
            "confidence_calibration": {"model_a": 0.0, "model_b": 0.0, "distance": 0.0},
            "reasoning_style": {"model_a": 0.0, "model_b": 0.0, "distance": 0.0},
            "verbosity_caveat_bloat": {"model_a": 0.0, "model_b": 0.0, "distance": 0.0}
        }
    }
    
    scored_data = []

    try:
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        else:
            print(f"Warning: {summary_path} not found. Using mock summary data.")
            
        if scored_path.exists():
            with open(scored_path, "r", encoding="utf-8") as f:
                scored_data = json.load(f)
        else:
            print(f"Warning: {scored_path} not found. Using empty scored data.")
            scored_data = [{
                "dimension": "N/A", 
                "prompt": "No data available. Run runner.py and scorer.py first.",
                "model_a_response": "N/A",
                "model_b_response": "N/A",
                "distance": 0.0
            }]
    except Exception as e:
        print(f"Error loading data: {e}")

    return summary_data, scored_data

# --- Create Charts & Dataframes ---
def create_radar_chart(summary_data):
    dimensions = list(summary_data["dimensions"].keys())
    
    if not dimensions:
        return go.Figure()

    model_a_scores = [summary_data["dimensions"][dim]["model_a"] for dim in dimensions]
    model_b_scores = [summary_data["dimensions"][dim]["model_b"] for dim in dimensions]

    # Close the polygon for Plotly
    dimensions.append(dimensions[0])
    model_a_scores.append(model_a_scores[0])
    model_b_scores.append(model_b_scores[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=model_a_scores,
        theta=dimensions,
        fill='toself',
        name='Model A',
        line_color='rgba(99, 110, 250, 0.8)',
        fillcolor='rgba(99, 110, 250, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=model_b_scores,
        theta=dimensions,
        fill='toself',
        name='Model B',
        line_color='rgba(239, 85, 59, 0.8)',
        fillcolor='rgba(239, 85, 59, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Behavioral Dimension Comparison (0 to 1 scale)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def create_divergence_df(scored_data):
    if not scored_data:
        return pd.DataFrame(columns=["Dimension", "Prompt", "Model A Response", "Model B Response", "Distance"])
        
    df = pd.DataFrame(scored_data)
    
    # Ensure columns exist even if mock data is sparse
    for col in ["dimension", "prompt", "model_a_response", "model_b_response", "distance"]:
        if col not in df.columns:
            df[col] = "N/A" if col != "distance" else 0.0

    df = df[["dimension", "prompt", "model_a_response", "model_b_response", "distance"]]
    df = df.rename(columns={
        "dimension": "Dimension",
        "prompt": "Prompt",
        "model_a_response": "Model A Response",
        "model_b_response": "Model B Response",
        "distance": "Distance"
    })
    
    df = df.sort_values(by="Distance", ascending=False).reset_index(drop=True)
    return df

# --- Gradio UI ---
summary, raw_scores = load_data()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LLM Diff: Behavioral Divergence Report")
    gr.Markdown("### *A 'git diff' for model behavior.* Analyze how different versions of LLMs react to identical prompts.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Global Behavioral Fingerprint")
            gr.Markdown(f"# {summary.get('behavioral_fingerprint', 0.0):.4f}")
            gr.Markdown("*(Scale: 0.0 to 1.0. Higher = More Divergent)*")
        
        with gr.Column(scale=2):
            radar_plot = gr.Plot(value=create_radar_chart(summary))
            
    gr.Markdown("---")
    gr.Markdown("## Top Divergent Prompts")
    gr.Markdown("Explore the specific prompts where the models disagreed the most.")
    
    df_divergence = create_divergence_df(raw_scores)
    gr.Dataframe(
        value=df_divergence,
        wrap=True,
        interactive=False,
        row_count=(10, "dynamic")
    )

if __name__ == "__main__":
    demo.launch(share=True)
