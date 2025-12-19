"""
btut_plots.py
=============
All plotting and visualization functions for BTUT results.
Generates the exact plots you love: convergence, phase maps, PD curve, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from btut_model import run_simulation

# Create plots folder
os.makedirs("plots", exist_ok=True)

def save_plot(name: str):
    """Save current plot with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"plots/{name}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved: {path}")

def plot_convergence(gamma=1.45, cA_SH=0.40, kernel_tau=0.30, N=200_000, iters=25):
    p_star, hist = run_simulation(N=N, gamma=gamma, cA_SH=cA_SH, kernel_tau=kernel_tau, iters=iters)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(hist)+1), hist, 'o-', linewidth=2, markersize=6)
    plt.title(f"Hybrid Mean-Field Convergence\nFinal Fraction A ≈ {p_star:.4f}")
    plt.xlabel("Iteration")
    plt.ylabel("Fraction Playing A")
    plt.grid(True, alpha=0.3)
    save_plot("convergence")

def plot_pd_curve():
    lam = np.linspace(0, 300, 300)
    m = 3
    frac = np.where(lam < m, 1.0, (m / (2*lam))**2)
    frac = np.clip(frac, 0, 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(lam, frac, '.-', linewidth=2)
    plt.title("BTUT PD Cooperation Curve (m=3)")
    plt.xlabel("λ (per-edge cost for A)")
    plt.ylabel("Fraction Cooperating (A)")
    plt.grid(True, alpha=0.3)
    save_plot("pd_curve")

def plot_phase_map_A(steps=12, N=300_000, kernel_tau=0.30):
    gammas = np.linspace(1.1, 1.6, steps)
    costs = np.linspace(0.3, 0.9, steps)
    heat = np.zeros((steps, steps))
    
    print("Computing Phase Map A...")
    for i, g in enumerate(gammas):
        for j, c in enumerate(costs):
            p, _ = run_simulation(N=N, gamma=g, cA_SH=c, kernel_tau=kernel_tau, iters=20)
            heat[i, j] = p
    
    plt.figure(figsize=(8, 6))
    plt.imshow(heat, origin='lower', extent=[0.3, 0.9, 1.1, 1.6], aspect='auto', cmap='viridis')
    plt.colorbar(label='Equilibrium Fraction A')
    plt.xlabel("SH A-cost (c_A^SH)")
    plt.ylabel("SH bonus γ")
    plt.title(f"Phase Map A  (τ = {kernel_tau})")
    save_plot("phase_map_A")

def plot_phase_map_B(steps=10, cA_SH=0.55):
    taus = np.linspace(0.0, 0.8, steps)
    gammas = np.linspace(1.1, 1.6, 12)
    heat = np.zeros((steps, len(gammas)))
    
    print("Computing Phase Map B...")
    for i, tau in enumerate(taus):
        for j, g in enumerate(gammas):
            p, _ = run_simulation(gamma=g, cA_SH=cA_SH, kernel_tau=tau, iters=20)
            heat[i, j] = p
    
    plt.figure(figsize=(8, 6))
    plt.imshow(heat, origin='lower', extent=[1.1, 1.6, 0.0, 0.8], aspect='auto', cmap='plasma')
    plt.colorbar(label='Equilibrium Fraction A')
    plt.xlabel("SH bonus γ")
    plt.ylabel("Kernel exponent τ")
    plt.title(f"Phase Map B  (c_A^SH ≈ {cA_SH})")
    save_plot("phase_map_B")