#!/usr/bin/env python3
"""
BTUT Diagnostics
----------------
Computes:
 - Convergence rate Δp_t
 - Equilibrium variance over seeds
 - Sensitivity ∂p*/∂γ, ∂p*/∂cA_SH
"""

import argparse, numpy as np, os, json, csv, matplotlib.pyplot as plt
from btut_grok_test import hybrid_meanfield_equilibrium, PLOTS_DIR

def run_diagnostics(gamma, cA_SH, alpha, tau, N=300_000, iters=20, reps=5):
    seeds = np.arange(reps) * 100
    results = []
    for s in seeds:
        p_star, hist = hybrid_meanfield_equilibrium(
            N=N, seed=int(s), gamma=gamma, cSH=(cA_SH,0.10),
            cPD=(0.20,0.08), alpha=alpha, kernel_tau=tau, iters=iters)
        results.append(p_star)
    results = np.array(results)
    mean, std = results.mean(), results.std()
    print(f"[DIAG] mean={mean:.4f}, std={std:.4f}")

    plt.figure(figsize=(6,4))
    plt.errorbar(np.arange(len(results)), results, yerr=std, fmt="o")
    plt.title("Equilibrium Variance Across Seeds")
    plt.xlabel("Seed Index"); plt.ylabel("Final Fraction A")
    out_png = os.path.join(PLOTS_DIR, f"diagnostics_var_{gamma:.2f}_{cA_SH:.2f}.png")
    plt.savefig(out_png); plt.close()

    # Sensitivity map
    gammas = np.linspace(gamma-0.2, gamma+0.2, 5)
    costs = np.linspace(cA_SH-0.2, cA_SH+0.2, 5)
    heat = np.zeros((len(gammas), len(costs)))
    for i,g in enumerate(gammas):
        for j,c in enumerate(costs):
            p_star,_ = hybrid_meanfield_equilibrium(
                N=N//2, gamma=g, cSH=(c,0.10), alpha=alpha, kernel_tau=tau)
            heat[i,j] = p_star
    plt.figure(figsize=(6,5))
    plt.imshow(heat, origin="lower", aspect="auto",
               extent=[costs[0],costs[-1],gammas[0],gammas[-1]])
    plt.colorbar(label="p*")
    plt.xlabel("cA_SH"); plt.ylabel("gamma")
    plt.title("Sensitivity Map")
    out_heat = os.path.join(PLOTS_DIR, f"sensitivity_{gamma:.2f}_{cA_SH:.2f}.png")
    plt.savefig(out_heat); plt.close()

    out_json = out_png.replace(".png",".json")
    with open(out_json,"w") as f: json.dump({
        "mean":float(mean),"std":float(std),"gamma":gamma,"cA_SH":cA_SH,
        "alpha":alpha,"tau":tau,"N":N,"iters":iters,"reps":reps},f,indent=2)

    print(f"[✓] Saved diagnostics: {out_png}, {out_heat}, {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=1.45)
    ap.add_argument("--cA_SH", type=float, default=0.40)
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--N", type=int, default=300000)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    run_diagnostics(**vars(args))
