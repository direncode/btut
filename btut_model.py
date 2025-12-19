"""
btut_model.py
=============
Core BTUT Hybrid Mean-Field Model

This file contains the complete simulation logic:
- Degree sampling from Barabási–Albert distribution
- Hybrid mean-field dynamics for mixed PD/HD/SH games
- Fast, O(N) scalable equilibrium computation

Run examples:
    from btut_model import run_simulation
    p_star, history = run_simulation(N=200000, gamma=1.45, cA_SH=0.40, kernel_tau=0.30)
"""

import numpy as np

def sample_BA_degrees(N: int, m: int = 3, seed: int | None = None, kcap: int | None = 6000) -> np.ndarray:
    """
    Sample node degrees from Barabási–Albert scale-free network distribution.
    No need to build the full graph — this is the key to O(N) scaling.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    k = m / np.sqrt(1.0 - u)           # Power-law-like sampling
    k = np.floor(k).astype(np.int64)
    k[k < m] = m                       # Minimum degree = m
    if kcap is not None:
        k[k > kcap] = kcap             # Optional cap for numerical stability
    return k.astype(float)


def run_simulation(
    N: int = 500_000,
    seed: int | None = 2025,
    gamma: float = 1.45,           # SH bonus multiplier for cooperating pair
    alpha: float = 0.60,           # HD parameter (not used heavily here)
    kernel_tau: float = 0.30,      # Hub influence exponent (τ): higher = hubs dominate more
    cA_SH: float = 0.40,           # Cost of A in Stag Hunt
    cB_SH: float = 0.10,
    cA_PD: float = 0.20,           # Cost of A in Prisoner's Dilemma
    cB_PD: float = 0.08,
    iters: int = 25,
    jitter_sigma: float = 0.05,
) -> tuple[float, list[float]]:
    """
    Run one full BTUT hybrid mean-field simulation.
    
    Returns:
        p_star: final equilibrium fraction of strategy A (cooperation-like)
        history: list of fraction A over iterations
    """
    rng = np.random.default_rng(seed)
    degrees = sample_BA_degrees(N, m=3, seed=seed)
    
    # Weight hubs more if kernel_tau > 0
    k_weight = (degrees / degrees.max()) ** kernel_tau if kernel_tau > 0 else np.ones(N)
    
    # Small individual heterogeneity
    jitterA = np.clip(rng.normal(1.0, jitter_sigma, N), 0.7, 1.3)
    jitterB = np.clip(rng.normal(1.0, jitter_sigma, N), 0.7, 1.3)

    # Base log-payoffs (scaled for stability)
    uA_PD, uB_PD = np.log(2.0), np.log(1.2)
    uA_HD, uB_HD = np.log(2.2), np.log(1.5)
    uA_SH, uB_SH = np.log(2.5), np.log(1.2)

    # Equal mix of three games
    mix = 1/3

    p = 0.5  # Start at half cooperation
    history = []

    for _ in range(iters):
        # Expected payoff contributions from each game
        EU_PD_A = 0.5 * (uA_PD + (p * uA_PD + (1-p) * uB_PD)) - cA_PD * uA_PD
        EU_PD_B = 0.5 * (uB_PD + (p * uA_PD + (1-p) * uB_PD)) - cB_PD * uB_PD

        EU_HD_A = (0.5 * (uA_HD + p*uA_HD + (1-p)*uB_HD)) * (p*alpha + (1-p)*1.0) - 0.22 * uA_HD
        EU_HD_B = (0.5 * (uB_HD + p*uA_HD + (1-p)*uB_HD)) * (p*1.0 + (1-p)*1.08) - 0.12 * uB_HD

        EU_SH_A = (0.5 * (uA_SH + p*uA_SH + (1-p)*uB_SH)) * (p*gamma + (1-p)*0.03) - cA_SH * uA_SH
        EU_SH_B = (0.5 * (uB_SH + p*uA_SH + (1-p)*uB_SH)) * (p*0.03 + (1-p)*1.0) - cB_SH * uB_SH

        # Total expected utility for A and B players
        U_A = k_weight * degrees * (mix*(EU_PD_A + EU_HD_A + EU_SH_A)) / jitterA
        U_B = k_weight * degrees * (mix*(EU_PD_B + EU_HD_B + EU_SH_B)) / jitterB

        p_new = (U_A > U_B).mean()                   # Fraction that prefer A
        history.append(p_new)
        p = 0.5 * p + 0.5 * p_new                    # Smooth momentum update

    return float(p), history