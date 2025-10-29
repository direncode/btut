import os, numpy as np

for i in range(10):
    gamma = np.round(np.random.uniform(1.1, 1.9), 2)
    cA_SH = np.round(np.random.uniform(0.3, 0.8), 2)
    cA_PD = np.round(np.random.uniform(0.1, 0.3), 2)
    alpha  = np.round(np.random.uniform(0.4, 0.8), 2)
    tau    = np.round(np.random.uniform(0.0, 0.8), 2)

    cmd = f"py btut_grok_test.py --demo hybrid_converge --gamma {gamma} --cA_SH {cA_SH} --cA_PD {cA_PD} --alpha {alpha} --tau {tau}"
    print(f"\n[RUN {i+1}] {cmd}")
    os.system(cmd)
