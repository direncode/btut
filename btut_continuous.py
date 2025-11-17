#!/usr/bin/env python3
"""
BTUT – Continuous Action (Memory-Efficient, O(N))
================================================
* Fixes _ArrayMemoryError by removing dense distance matrix
* Kernel = 1-hop or 2-hop adjacency diffusion
* Full BTUT logic (θ trajectory, ψ undercurrent, w_i hubs, continuous payoff)
* Scale-invariant: Tested with small N; increase as hardware allows
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy.sparse import csr_matrix

# ============================= PARAMETERS =============================
N               = 10_000       # Start small (scale-invariant); increase to 200k+ on high-RAM machines
M               = 6            # BA-graph edges per new node
TAU             = 0.30         # Hub exponent τ
LAMBDA          = 2.0          # Decay for 2-hop (optional)
KAPPA_THETA     = 0.12         # Trajectory momentum
KAPPA_PSI       = 0.08         # Undercurrent strength
ALPHA           = 0.60         # Payoff vs internal signal
ITERS           = 30           # Max iterations
SEED            = 42
HOP             = 1            # 1 = 1-hop (minimal memory), 2 = 2-hop diffusion
PD_PARAMS       = {'R': 3.0, 'T': 5.0, 'S': 0.0, 'P': 1.0}  # PD matrix
# =====================================================================

np.random.seed(SEED)
print(f"[BTUT-CONT] Initializing N={N:,} agents...")

# ---------- 1. Build Scale-Free Network ----------
print("   → Generating Barabási–Albert graph...")
G = nx.barabasi_albert_graph(N, M)
A = nx.to_scipy_sparse_array(G, format='csr')   # adjacency matrix

# ---------- 2. Diffusion Kernel (1-hop or 2-hop) ----------
if HOP == 1:
    K = A.tocsr()
elif HOP == 2:
    K = (A @ A).astype(float)
    K.data = np.exp(-LAMBDA * np.sqrt(K.data))   # optional decay
    K = K.tocsr()
else:
    raise ValueError("HOP must be 1 or 2")
print(f"   → Using {HOP}-hop kernel (nnz = {K.nnz:,})")

# ---------- 3. Hub Weights w_i ----------
degrees = np.array(G.degree())[:, 1].astype(float)
k_max = degrees.max()
w = (degrees / k_max) ** TAU
w = w / w.mean()                                 # normalize to mean 1

# ---------- 4. Continuous PD Payoff ----------
def pd_payoff(ai, aj, p=PD_PARAMS):
    R, T, S, P = p['R'], p['T'], p['S'], p['P']
    return ai*aj*R + ai*(1-aj)*S + (1-ai)*aj*T + (1-ai)*(1-aj)*P
payoff_vec = np.vectorize(pd_payoff)

# ---------- 5. Initialize State ----------
a = np.random.uniform(0, 1, N)      # actions ∈ [0,1]
a_prev = a.copy()
theta = np.zeros(N)                 # trajectory momentum
psi   = np.zeros(N)                 # undercurrent energy
history = [a.mean()]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# ---------- 6. Main Simulation Loop ----------
print(f"   → Starting {ITERS} iterations...")
t_start = time.time()

for it in range(1, ITERS + 1):
    # ---- Local Payoff U_i = Σ_j K_ij * payoff(a_i, a_j) ----
    neigh_a   = K @ a
    neigh_one = K @ np.ones(N)
    neigh_not_a = neigh_one - neigh_a

    U = (a * (PD_PARAMS['R']*neigh_a + PD_PARAMS['S']*neigh_not_a) +
         (1-a) * (PD_PARAMS['T']*neigh_a + PD_PARAMS['P']*neigh_not_a))

    # ---- Undercurrent: log-ratio of coop vs defect utilities ----
    U_coop   = PD_PARAMS['R']*neigh_a + PD_PARAMS['S']*neigh_not_a
    U_defect = PD_PARAMS['T']*neigh_a + PD_PARAMS['P']*neigh_not_a
    eps = 1e-12
    psi += KAPPA_PSI * np.log((U_coop + eps) / (U_defect + eps))

    # ---- Trajectory momentum ----
    if it > 1:
        theta += KAPPA_THETA * (a - a_prev)

    # ---- Combined signal & update ----
    signal = ALPHA * U + (1 - ALPHA) * (theta + psi)
    a_new  = sigmoid(w * signal)

    a_prev = a.copy()
    a = a_new

    p_t = a.mean()
    history.append(p_t)

    if it <= 5 or it % 5 == 0 or it == ITERS:
        delta = p_t - history[-2]
        print(f"   [Iter {it:2d}] ⟨a⟩ = {p_t:.6f}  Δ = {delta:+.6f}")

# ---------- 7. Final Diagnostics ----------
dt = time.time() - t_start
final_mean = a.mean()
final_std  = a.std()

print("\n" + "="*60)
print(f"BTUT CONTINUOUS ACTION SIMULATION COMPLETE")
print(f"   Agents (N)        : {N:,}")
print(f"   Final ⟨a⟩         : {final_mean:.6f}")
print(f"   Final σ(a)        : {final_std:.6f}")
print(f"   Runtime           : {dt:.2f}s")
print(f"   Speed             : {N/dt/1e6:.3f} M agents/sec")
print("="*60)

# ---------- 8. Plot Convergence ----------
plt.figure(figsize=(8, 5))
plt.plot(history, 'o-', color='#1f77b4', lw=2, markersize=4)
plt.title('BTUT Continuous Action: Global Cooperation ⟨a⟩ vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Global Average Action ⟨a⟩')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("btut_continuous_convergence.png", dpi=300, bbox_inches='tight')
print(f"   → Plot saved: btut_continuous_convergence.png")
plt.show()