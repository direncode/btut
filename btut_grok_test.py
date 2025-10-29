#!/usr/bin/env python3
"""
BTUT Demo Harness — streamlined edition
---------------------------------------
Run examples:
  py btut_grok_test.py --demo pd_curve
  py btut_grok_test.py --demo hybrid_converge
  py btut_grok_test.py --demo phase_maps_a
  py btut_grok_test.py --demo phase_maps_b
  py btut_grok_test.py --demo all
"""

import argparse, os, datetime
import numpy as np, matplotlib.pyplot as plt

# ---------------------------------------------------------------------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def savefig_unique(basename, ext="png"):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(PLOTS_DIR, f"{basename}_{stamp}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[✓] Saved {path}")
    return path

# ---------------------------------------------------------------------
def pd_curve(m=3.0, lam_max=300.0, npts=240):
    lam = np.linspace(0.0, lam_max, npts)
    deg_thresh = 2.0 * lam
    frac = np.where(deg_thresh < m, 1.0, (m / deg_thresh)**2)
    frac = np.clip(frac, 0.0, 1.0)

    plt.figure(figsize=(7,4.5))
    plt.plot(lam, frac, marker='.', linewidth=1.2)
    plt.title(f"BTUT PD Cooperation Curve (m={m:g})")
    plt.xlabel("λ  (per-edge cost for A)")
    plt.ylabel("Fraction Cooperating (A)")
    plt.grid(True)
    return savefig_unique("pd_curve")

# ---------------------------------------------------------------------
def sample_BA_degrees(N, m=3, seed=1234, kcap=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    k = m / np.sqrt(1.0 - u)
    k = np.floor(k).astype(np.int64)
    k[k < m] = m
    if kcap is not None:
        k[k > kcap] = kcap
    return k.astype(float)

def hybrid_meanfield_equilibrium(N=500_000, m=3, seed=2025,
        mix_PD=1/3, mix_HD=1/3, mix_SH=1/3,
        uPD=(np.log(2.0), np.log(1.2)),
        uHD=(np.log(2.2), np.log(1.5)),
        uSH=(np.log(2.5), np.log(1.2)),
        alpha=0.7, beta=1.08, delta_hd=1.0,
        gamma=1.30, delta_sh=0.03,
        cPD=(0.30, 0.08), cHD=(0.22, 0.12), cSH=(0.70, 0.10),
        jitter_sigma=0.05, kernel_tau=0.0, iters=20, kcap=6000):
    rng = np.random.default_rng(seed)
    k = sample_BA_degrees(N, m=m, seed=seed, kcap=kcap)
    k_weight = (k / k.max()) ** kernel_tau if kernel_tau > 0 else np.ones_like(k)
    jitterA = np.clip(rng.normal(1.0, jitter_sigma, size=N), 0.7, 1.3)
    jitterB = np.clip(rng.normal(1.0, jitter_sigma, size=N), 0.7, 1.3)

    uA_PD,uB_PD = uPD; uA_HD,uB_HD = uHD; uA_SH,uB_SH = uSH
    cA_PD,cB_PD = cPD; cA_HD,cB_HD = cHD; cA_SH,cB_SH = cSH

    p = 0.5
    hist = []
    for _ in range(iters):
        EU_PD_A = 0.5*(uA_PD + (p*uA_PD + (1-p)*uB_PD)) - cA_PD*uA_PD
        EU_PD_B = 0.5*(uB_PD + (p*uA_PD + (1-p)*uB_PD)) - cB_PD*uB_PD

        EU_HD_A = (0.5*(uA_HD + (p*uA_HD + (1-p)*uB_HD))) * (p*alpha + (1-p)*delta_hd) - cA_HD*uA_HD
        EU_HD_B = (0.5*(uB_HD + (p*uA_HD + (1-p)*uB_HD))) * (p*delta_hd + (1-p)*beta) - cB_HD*uB_HD

        EU_SH_A = (0.5*(uA_SH + (p*uA_SH + (1-p)*uB_SH))) * (p*gamma + (1-p)*delta_sh) - cA_SH*uA_SH
        EU_SH_B = (0.5*(uB_SH + (p*uA_SH + (1-p)*uB_SH))) * (p*delta_sh + (1-p)*1.0) - cB_SH*uB_SH

        U_A = k_weight * k * (mix_PD*EU_PD_A + mix_HD*EU_HD_A + mix_SH*EU_SH_A) / jitterA
        U_B = k_weight * k * (mix_PD*EU_PD_B + mix_HD*EU_HD_B + mix_SH*EU_SH_B) / jitterB

        p_new = (U_A > U_B).mean()
        hist.append(p_new)
        p = 0.5*p + 0.5*p_new
    return float(p), hist

# ---------------------------------------------------------------------
def demo_hybrid_converge(gamma=1.45, cA_SH=0.40, cA_PD=0.20, alpha=0.60,
                         tau=0.30, N=500_000, seed=271828, ext="png"):
    p_star, hist = hybrid_meanfield_equilibrium(
        N=N, seed=seed, gamma=gamma, cSH=(cA_SH,0.10),
        cPD=(cA_PD,0.08), alpha=alpha, kernel_tau=tau
    )
    plt.figure(figsize=(6.5,4))
    plt.plot(np.arange(1, len(hist)+1), hist, marker='o')
    plt.title(f"Hybrid Mean-Field Convergence\nFinal A≈{p_star:.3f}")
    plt.xlabel("Iteration"); plt.ylabel("Fraction A")
    path = savefig_unique("hybrid_convergence", ext)
    print(f"[DONE] Final A≈{p_star:.4f}")
    return path, p_star

def demo_phase_maps_A(N=600_000, tau=0.30, ext="png"):
    gammas = np.linspace(1.10, 1.60, 11)
    cA_SH_vals = np.linspace(0.30, 0.90, 11)
    heat = np.zeros((len(gammas), len(cA_SH_vals)))
    for i,g in enumerate(gammas):
        for j,cA in enumerate(cA_SH_vals):
            p_star,_ = hybrid_meanfield_equilibrium(
                N=N, seed=9000+i*20+j, gamma=g, cSH=(cA,0.10),
                cPD=(0.20,0.08), alpha=0.60, kernel_tau=tau, iters=18)
            heat[i,j] = p_star
    plt.figure(figsize=(7,5.2))
    plt.imshow(heat, origin='lower', aspect='auto',
               extent=[cA_SH_vals[0], cA_SH_vals[-1], gammas[0], gammas[-1]])
    plt.colorbar(label='Equilibrium Fraction A')
    plt.xlabel('SH A-cost  c_A^SH'); plt.ylabel('SH bonus  γ')
    plt.title(f'Phase Map A  (τ={tau})')
    return savefig_unique("phase_map_A", ext)

def demo_phase_maps_B(N=600_000, cA_SH=0.55, ext="png"):
    taus = np.linspace(0.0, 0.8, 9)
    gammas = np.linspace(1.10, 1.60, 11)
    heat = np.zeros((len(taus), len(gammas)))
    for i,tau in enumerate(taus):
        for j,g in enumerate(gammas):
            p_star,_ = hybrid_meanfield_equilibrium(
                N=N, seed=12000+i*20+j, kernel_tau=tau, gamma=g,
                cSH=(cA_SH,0.10), cPD=(0.22,0.08), alpha=0.65, iters=18)
            heat[i,j] = p_star
    plt.figure(figsize=(7,5.2))
    plt.imshow(heat, origin='lower', aspect='auto',
               extent=[gammas[0], gammas[-1], taus[0], taus[-1]])
    plt.colorbar(label='Equilibrium Fraction A')
    plt.xlabel('SH bonus  γ'); plt.ylabel('Kernel exponent  τ')
    plt.title(f'Phase Map B  (c_A^SH≈{cA_SH})')
    return savefig_unique("phase_map_B", ext)

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="BTUT Demo Harness (streamlined)")
    ap.add_argument("--demo", required=True,
        choices=["pd_curve","hybrid_converge","phase_maps_a","phase_maps_b","all"])
    ap.add_argument("--gamma", type=float, default=1.45)
    ap.add_argument("--cA_SH", type=float, default=0.40)
    ap.add_argument("--cA_PD", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--ext", type=str, default="png")
    args = ap.parse_args()

    if args.demo == "pd_curve":
        pd_curve()
    elif args.demo == "hybrid_converge":
        demo_hybrid_converge(args.gamma, args.cA_SH, args.cA_PD, args.alpha, args.tau, ext=args.ext)
    elif args.demo == "phase_maps_a":
        demo_phase_maps_A(tau=args.tau, ext=args.ext)
    elif args.demo == "phase_maps_b":
        demo_phase_maps_B(ext=args.ext)
    elif args.demo == "all":
        pd_curve()
        demo_hybrid_converge(args.gamma, args.cA_SH, args.cA_PD, args.alpha, args.tau, ext=args.ext)
        demo_phase_maps_A(tau=args.tau, ext=args.ext)
        demo_phase_maps_B(ext=args.ext)
        print("\nAll demos completed successfully.")

if __name__ == "__main__":
    main()

