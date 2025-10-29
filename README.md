# Bivariate Trajectory–Undercurrent Theory (BTUT)
[![BTUT Automated Validation](https://github.com/direnk/btut-darpa-submission/actions/workflows/btut_autotest.yml/badge.svg?branch=main)](https://github.com/direnk/btut-darpa-submission/actions/workflows/btut_autotest.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Last Updated](https://img.shields.io/github/last-commit/direnk/btut)

# Bivariate Trajectory–Undercurrent Theory (BTUT)
DARPA Mathematical Challenge 13 Submission  
**Author:** Diren Kumaratilleke  
**Affiliation:** Independent Researcher, University of North Carolina  
**Disclaimer:** This repository is made public for academic transparency and DARPA Challenge 13 evaluation purposes.  
All content © 2025 Diren Kumaratilleke. Use permitted under the terms of the LICENSE file.

**Acknowledgment:**  
This work was developed with assistance from OpenAI’s GPT-5 model, which supported the formal articulation, documentation, and structure of the BTUT framework based on my original theoretical intuition and design.

---

## Table of Contents
- [Overview](#overview)
- [Project Files](#project-files)
- [Running BTUT](#running-the-system)
- [Main Run Command](#main-run-command)
- [Intuition Behind BTUT](./docs/BTUT%20Intuition%20and%20Process.docx)
- [Mathematical Foundations](https://github.com/direnk/btut-darpa-submission/blob/main/docs/btut_math.pdf)
- [Ethics and Compliance](https://github.com/direnk/btut-darpa-submission/blob/main/ETHICS.md)
- [Citation](https://github.com/direnk/btut-darpa-submission/blob/main/CITATION.cff)

---

## Overview

BTUT (Bivariate Trajectory–Undercurrent Theory) is a scalable, kernel-based game theory framework designed to model dynamic convergence across large-scale agent systems. It replaces static Nash equilibria and mean-field approximations with a dynamic flow equilibrium that balances trajectories (visible actions) and undercurrents (latent pressures).

This repository contains all code, data, and visualizations associated with the BTUT simulations. The model achieves linear scaling across millions to trillions of agents and provides convergence without PDE bottlenecks.

---

## Project Files

- `btut_random_sweep.py` – random parameter exploration.
- `btut_scaling_test.py` – validates linear scaling of BTUT.
- `btut_diagnostics.py` – performs diagnostics on convergence and stability.
- `btut_animate.py` – generates animated visualizations of dynamic equilibria.
- `btut_math.pdf` – theoretical formulation and equations.
- `ETHICS.md` – ethical usage guidelines.
- `CITATION.cff` – citation metadata for academic and DARPA-aligned submissions.
- `run_all.ps1` – executes all core simulations and visualization routines automatically.

---

## Running the System

To execute all tests and visualize results:

## Main Run Command 
```powershell
./run_all.ps1

## 🔧 Manual Run Commands

```bash
### Randomized Parameter Sweep
python btut_random_sweep.py --nodes 50000 --iterations 2000 --kernel_tau 0.45 --latent_scale 1.0

### Scaling Validation
python btut_scaling_test.py --min_nodes 1000 --max_nodes 1000000 --scaling_factor 10

### Diagnostics and Sensitivity
python btut_diagnostics.py --alpha 1.45 --tau 0.40 --reps 20

### Hybrid Convergence
python btut_diagnostics.py --hybrid True --steps 2500

### Animation Generation
python btut_animate.py --source diagnostics_var_1.45_0.40.json --fps 15 --duration 20

### Custom Kernel Test
python btut_random_sweep.py --nodes 20000 --kernel_tau 0.3 --latent_scale 1.5 --log_intensity 2.2 --save_custom True

### Real-Time Visualization
python btut_random_sweep.py --nodes 5000 --iterations 1000 --visualize True

### Reproducibility Check
python btut_random_sweep.py --nodes 10000 --seed 42 --kernel_tau 0.5 --latent_scale 0.8

### Export All Results
zip -r BTUT_results_$(date +%Y%m%d_%H%M%S).zip results diagnostics






