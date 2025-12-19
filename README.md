# Bivariate Trajectory–Undercurrent Theory (BTUT)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Last Updated](https://img.shields.io/github/last-commit/direncode/btut)

# BTUT: Bivariate Trajectory–Undercurrent Theory

*A PDE-Free, Scalable Framework for Large-Scale Game Dynamics — Addressing DARPA Mathematical Challenge 13*

## Overview

**BTUT** is a novel, kernel-based game-theoretic framework that directly tackles **DARPA Mathematical Challenge 13 (2008)**:  

> *"Creating a Game Theory that Scales — What new scalable mathematics is needed to replace the traditional Partial Differential Equations (PDE) approach to differential games?"*

BTUT replaces PDEs and static Nash equilibria with **dynamic flow equilibria** that balance observable **trajectories** (agent actions) and latent **undercurrents** (network pressures). Built on scale-free networks with hub-weighted influence \( w_i = (k_i / k_{\max})^\tau \), it models mixed interactions (Prisoner's Dilemma, Hawk-Dove, Stag Hunt) via local rules and momentum updates.

## Key Innovations & Results

- **PDE-free scalability**: Strict O(N) complexity, verified up to 1e6+ agents.
- **Ultra-robust convergence**: Equilibrium \( p^* \approx 1.000 \) (std ≈ 4.9×10⁻¹⁰) in <20 iterations.
- **Rich phase behavior**: Sharp transitions (see Phase Map A staircase at γ=1.45, c_A^SH=0.40).
- **Scale-invariant continuous-action extension**: Full cooperation in 2–3 iterations (no dense matrices).
- Runtime: ~0.12s for 30 iterations at N=1,000 (~8.3M agents/sec equivalent).

## Repository Contents

- **Core implementation**: `btut_model.py` (main hybrid mean-field engine), `btut_continuous.py` (continuous-action extension)
- **Visualization & demos**: `btut_fleet_convergence_visual.py` (fleet coordination animation), `btut_thought_3d_visualizer.py` (emergent thought wave), `btut_organic_ai_thought_interactive.py` (interactive thought generator)
- **Diagnostics & plots**: `btut_plots.py` (convergence, variance, phase maps, PD curve, scaling)
- **Launcher**: `run_btut.py`
- **Theory**: `btut_math.pdf` (full mathematical derivation)
- **Interactive outputs**: `btut_unified_thought_3d.html` (3D thought visualization), `btut_fleet_convergence.gif` (fleet coordination animation)
- **Requirements**: `requirements.txt`

## Applications

Ideal for modeling emergent cooperation in:
- AI agent fleets
- Cyber-physical systems
- Resource allocation
- Alignment scenarios

## Developed by

Diren Kumaratilleke  
Independent Researcher
University of North Carolina at Chapel Hill 

AI-assisted development via Grok-4 (xAI) and GPT models.

**License**: MIT  










