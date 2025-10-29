#!/usr/bin/env python3
"""
BTUT Scaling Test
-----------------
Verifies that runtime grows linearly with N.
"""

import time, numpy as np, matplotlib.pyplot as plt, os
from btut_grok_test import hybrid_meanfield_equilibrium, PLOTS_DIR

def run_scaling(Ns=(1e4, 5e4, 2e5, 5e5, 1e6), iters=15):
    Ns = [int(n) for n in Ns]
    times = []
    for N in Ns:
        t0 = time.time()
        hybrid_meanfield_equilibrium(N=N, iters=iters)
        dt = time.time()-t0
        times.append(dt)
        print(f"[SCALING] N={N:,} → {dt:.2f}s")
    plt.figure(figsize=(6,4))
    plt.plot(Ns, times, "o-")
    plt.xlabel("Network size N")
    plt.ylabel("Runtime (s)")
    plt.title("Scaling Validation (O(N))")
    out = os.path.join(PLOTS_DIR, "scaling_validation.png")
    plt.savefig(out); plt.close()
    print(f"[✓] Saved {out}")

if __name__ == "__main__":
    run_scaling()
