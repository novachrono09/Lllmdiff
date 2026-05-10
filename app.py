import json
import os
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from pathlib import Path

# --- Load Data ---
root_dir = Path(__file__).parent.absolute()
summary_path = root_dir / "summary_report.json"
scored_path = root_dir / "scored_responses.json"

summary_data = {
    "behavioral_fingerprint": 0.0,
    "dimensions": {}
}
scored_data = []

try:
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
    else:
        print(f"Warning: {summary_path} not found.")
        
    if scored_path.exists():
        with open(scored_path, "r", encoding="utf-8") as f:
            scored_data = json.load(f)
    else:
        print(f"Warning: {scored_path} not found.")
except Exception as e:
    print(f"Error loading data: {e}")

# --- Generate Plotly Figure ---
dimensions = list(summary_data.get("dimensions", {}).keys())
fig = go.Figure()

if dimensions:
    model_a_scores = [summary_data["dimensions"][dim]["model_a"] for dim in dimensions]
    model_b_scores = [summary_data["dimensions"][dim]["model_b"] for dim in dimensions]

    if len(dimensions) < 3:
        fig.add_trace(go.Bar(name='Model A', x=dimensions, y=model_a_scores))
        fig.add_trace(go.Bar(name='Model B', x=dimensions, y=model_b_scores))
        fig.update_layout(
            barmode='group',
            title="Behavioral Dimension Comparison (0 to 1 scale)",
            yaxis=dict(range=[0, 1])
        )
    else:
        dimensions_closed = dimensions + [dimensions[0]]
        model_a_scores_closed = model_a_scores + [model_a_scores[0]]
        model_b_scores_closed = model_b_scores + [model_b_scores[0]]

        fig.add_trace(go.Scatterpolar(
            r=model_a_scores_closed,
            theta=dimensions_closed,
            fill='toself',
            name='Model A',
            line_color='rgba(99, 110, 250, 0.8)',
            fillcolor='rgba(99, 110, 250, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=model_b_scores_closed,
            theta=dimensions_closed,
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

# --- Generate Pandas DataFrame ---
if scored_data:
    df = pd.DataFrame(scored_data)
    # Extract only needed columns
    cols_to_keep = ["dimension", "prompt", "model_a_response", "model_b_response", "distance"]
    # Handle missing columns gracefully
    for col in cols_to_keep:
        if col not in df.columns:
            df[col] = "N/A" if col != "distance" else 0.0
            
    df = df[cols_to_keep]
    df = df.rename(columns={
        "dimension": "Dimension",
        "prompt": "Prompt",
        "model_a_response": "Model A Response",
        "model_b_response": "Model B Response",
        "distance": "Distance"
    })
    df = df.sort_values(by="Distance", ascending=False).reset_index(drop=True)
else:
    df = pd.DataFrame(columns=["Dimension", "Prompt", "Model A Response", "Model B Response", "Distance"])

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LLM Diff: Behavioral Divergence Report")
    gr.Markdown("### *A 'git diff' for model behavior.* Analyze how different versions of LLMs react to identical prompts.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Global Behavioral Fingerprint")
            gr.Markdown(f"# {summary_data.get('behavioral_fingerprint', 0.0):.4f}")
            gr.Markdown("*(Scale: 0.0 to 1.0. Higher = More Divergent)*")
        
        with gr.Column(scale=2):
            gr.Plot(value=fig)
            
    gr.Markdown("---")
    gr.Markdown("## Top Divergent Prompts")
    gr.Markdown("Explore the specific prompts where the models disagreed the most.")
    
    gr.Dataframe(
        value=df,
        wrap=True,
        interactive=False,
        row_count=(10, "dynamic")
    )

if __name__ == "__main__":
    demo.launch(share=True)
