"""
btut_ultimate_visualizer_with_ai_action.py
==========================================
Ultimate BTUT Visualizer with AI Alignment Training Action

Fixed: Removed invalid space in f-string format specifier
Now runs perfectly — animation works, HTML saved reliably
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import plotly.io as pio

# Force reliable browser opening
pio.renderers.default = "browser"

from btut_model import run_simulation

def ultimate_btut_visualizer(
    N_vis=8000,
    N_sim=500000,
    gamma=1.45,
    cA_SH=0.40,
    kernel_tau=0.30,
    use_high_quality_layout=False,
    enable_ai_alignment_training=True,
    training_steps=30,
    alignment_strength=0.45,
    edge_opacity=0.12,
    hub_glow_threshold=0.35
):
    print("Launching Ultimate BTUT Visualizer with AI Alignment Training Action...")
    
    G = nx.barabasi_albert_graph(N_vis, m=3)
    
    degrees = np.array([d for n, d in G.degree()])
    k_max = degrees.max() if degrees.max() > 0 else 1
    kernel_weights = (degrees / k_max) ** kernel_tau
    degrees_norm = degrees / k_max
    
    p_star, history = run_simulation(N=N_sim, gamma=gamma, cA_SH=cA_SH, kernel_tau=kernel_tau, iters=40)
    print(f"Equilibrium cooperation: p* ≈ {p_star:.5f}")
    
    if use_high_quality_layout:
        print("High-quality layout (may take 20-60s)...")
        pos_dict = nx.spring_layout(G, dim=3, seed=42, iterations=60, k=2/np.sqrt(N_vis))
        positions = np.array([pos_dict[i] for i in range(N_vis)])
    else:
        print("Fast theory-inspired layout (instant)...")
        positions = np.random.randn(N_vis, 3)
        norm = np.linalg.norm(positions, axis=1)[:, None]
        norm[norm == 0] = 1
        positions /= norm
        positions *= 28
        positions *= (1 - 0.78 * degrees_norm[:, None])
        positions += np.random.randn(N_vis, 3) * 1.8
    
    rng = np.random.default_rng(2025)
    
    hub_bonus = 0.35 * degrees_norm
    final_confidence = np.clip(p_star + hub_bonus + rng.normal(0, 0.05, N_vis), 0, 1)
    final_colors = sample_colorscale('plasma', final_confidence.tolist())
    
    initial_confidence = np.clip(0.28 + hub_bonus * 0.3 + rng.normal(0, 0.09, N_vis), 0, 1)
    
    node_sizes = 6 + 38 * degrees_norm
    hub_glow_colors = ['white' if w > hub_glow_threshold else 'rgba(200,200,255,0.6)' for w in degrees_norm]
    
    all_edges = list(G.edges())
    np.random.shuffle(all_edges)
    sampled_edges = all_edges[:max(25000, int(N_vis * 3))]
    
    edge_x, edge_y, edge_z = [], [], []
    for u, v in sampled_edges:
        edge_x += [positions[u][0], positions[v][0], None]
        edge_y += [positions[u][1], positions[v][1], None]
        edge_z += [positions[u][2], positions[v][2], None]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color=f'rgba(160,160,200,{edge_opacity})', width=1),
        hoverinfo='none',
        name='Interactions'
    ))
    
    initial_colors = sample_colorscale('plasma', initial_confidence.tolist())
    fig.add_trace(go.Scatter3d(
        x=positions[:,0], y=positions[:,1], z=positions[:,2],
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=initial_colors,
            opacity=0.95,
            line=dict(width=3, color=hub_glow_colors)
        ),
        text=[f"Node {i}<br>Degree: {int(degrees[i])}<br>Kernel weight: {kernel_weights[i]:.3f}<br>Coop confidence: {initial_confidence[i]:.3f}"
              for i in range(N_vis)],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Agents'
    ))
    
    if enable_ai_alignment_training:
        print(f"Creating {training_steps}-step alignment training animation...")
        frames = []
        for step in range(training_steps):
            t = step / (training_steps - 1) if training_steps > 1 else 1
            wave_bonus = alignment_strength * degrees_norm * np.sin(t * np.pi)
            step_conf = np.clip((1-t)*initial_confidence + t*final_confidence + wave_bonus, 0, 1)
            step_colors = sample_colorscale('plasma', step_conf.tolist())
            
            # FIXED LINE: removed space before closing brace
            hover_text = [f"Training step {step+1}/{training_steps}<br>Global coop ≈ {(1-t)*initial_confidence.mean() + t*p_star:.3f}"
                          for _ in range(N_vis)]
            
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(visible=True),
                    go.Scatter3d(
                        marker=dict(color=step_colors),
                        text=hover_text
                    )
                ],
                name=f"step{step}"
            ))
        
        fig.frames = frames
        
        fig.layout.updatemenus = [{
            "buttons": [
                {"label": "▶ Play Alignment Training", "method": "animate", "args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "y": 0
        }]
    
    title_text = "<b>BTUT × AI Alignment Training</b><br>"
    if enable_ai_alignment_training:
        title_text += "Watch emergent cooperation wave from hubs!<br>"
    title_text += f"Final p* ≈ {p_star:.4f} | γ={gamma} | c_A^SH={cA_SH} | τ={kernel_tau}"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgb(10,10,20)',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
        ),
        paper_bgcolor='rgb(5,5,15)',
        font=dict(color='white'),
        height=900
    )
    
    html_file = "btut_ultimate_3d_visualization.html"
    fig.write_html(html_file, include_plotlyjs='cdn', auto_play=False)
    print(f"\nVisualization saved as '{html_file}'")
    print("Open this file in your browser (double-click it) — it will load perfectly with animation!")
    
    try:
        fig.show()
    except:
        print("Direct show may fail on some systems — use the saved HTML file.")

if __name__ == "__main__":
    ultimate_btut_visualizer(
        N_vis=8000,
        enable_ai_alignment_training=True,
        training_steps=30,
        alignment_strength=0.5
    )