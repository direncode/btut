#!/usr/bin/env python3
"""
BTUT Animation Generator
------------------------
Generates animated GIFs showing convergence or phase-map sweeps.

Example:
  py btut_animate.py --mode converge
  py btut_animate.py --mode phase
"""

import os, argparse, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from btut_grok_test import hybrid_meanfield_equilibrium, PLOTS_DIR

def animate_convergence(gamma=1.45, cA_SH=0.40, cA_PD=0.20, alpha=0.60, tau=0.30, iters=25):
    _, hist = hybrid_meanfield_equilibrium(
        gamma=gamma, cSH=(cA_SH,0.10), cPD=(cA_PD,0.08),
        alpha=alpha, kernel_tau=tau, iters=iters
    )
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlim(0, len(hist))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction A")
    line, = ax.plot([], [], "o-", lw=2)
    text = ax.text(0.7, 0.9, "", transform=ax.transAxes)

    def init(): line.set_data([], []); text.set_text(""); return line, text
    def update(frame):
        x = np.arange(frame+1)
        y = hist[:frame+1]
        line.set_data(x, y)
        text.set_text(f"A≈{y[-1]:.3f}")
        return line, text

    ani = FuncAnimation(fig, update, frames=len(hist), init_func=init, blit=True)
    out = os.path.join(PLOTS_DIR, "btut_convergence.gif")
    ani.save(out, writer=PillowWriter(fps=2))
    plt.close()
    print(f"[✓] Saved animation: {out}")

def animate_phase(gammas=(1.1,1.6), cA_SH_vals=(0.3,0.9), steps=8):
    from btut_grok_test import hybrid_meanfield_equilibrium
    gammas = np.linspace(*gammas, steps)
    cA_SH_vals = np.linspace(*cA_SH_vals, steps)
    fig, ax = plt.subplots(figsize=(6,5))
    img = ax.imshow(np.zeros((steps, steps)), origin="lower", aspect="auto", extent=[0,1,0,1])
    plt.colorbar(img, ax=ax, label="Equilibrium A")

    def frame(i):
        heat = np.zeros((steps, steps))
        for gi, g in enumerate(gammas):
            for cj, cA in enumerate(cA_SH_vals):
                p_star,_ = hybrid_meanfield_equilibrium(
                    N=150_000, seed=1000+gi*steps+cj, gamma=g, cSH=(cA,0.10),
                    cPD=(0.20,0.08), alpha=0.60, kernel_tau=0.3, iters=i+1
                )
                heat[gi,cj] = p_star
        img.set_data(heat)
        ax.set_title(f"Phase Sweep Frame {i+1}")
        return [img]

    ani = FuncAnimation(fig, frame, frames=steps, blit=True)
    out = os.path.join(PLOTS_DIR, "btut_phase_sweep.gif")
    ani.save(out, writer=PillowWriter(fps=1))
    plt.close()
    print(f"[✓] Saved animation: {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["converge","phase"], required=True)
    args = ap.parse_args()
    if args.mode == "converge": animate_convergence()
    else: animate_phase()
