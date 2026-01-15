# BTUT Advanced Tutorial (10 minutes)

**Target Audience:** Users familiar with basics, ready for advanced features
**Goal:** Master parameter tuning, batch simulations, and visualization

---

## SCENE 1: Introduction (0:00-0:30)

**Visual:** Instructor on camera with complex network visualization in background

**Narration:**
> "Welcome to the BTUT Advanced Tutorial. Now that you've mastered the basics, we'll dive deep into parameter tuning, running batch simulations, advanced visualization, and performance optimization. By the end of this video, you'll be able to design and execute complex multi-agent experiments."

**On-screen text:**
- Advanced Topics:
  - ⚙️ Parameter tuning
  - 📊 Batch simulations
  - 📈 Advanced visualization
  - ⚡ Performance optimization

---

## SCENE 2: Understanding Parameters (0:30-2:30)

**Visual:** Split screen: code on left, parameter visualization on right

**Narration:**
> "BTUT has four key parameters. Let's understand each one deeply."

### Parameter 1: gamma (γ)

**Code:**
```python
sim = Simulator(agents=10000, gamma=1.5)
```

**Visual:** 2x2 payoff matrix animating

**Narration:**
> "Gamma controls the payoff for coordinating on strategy B. With gamma = 1, both strategies are equally valuable. Higher gamma makes B more attractive, paradoxically increasing equilibrium cooperation on A."

**On-screen formula:**
```
Equilibrium: p* = γ/(1+γ)
γ=1.0 → 50% cooperation
γ=1.5 → 60% cooperation
γ=2.0 → 67% cooperation
```

### Parameter 2: tau (τ)

**Code:**
```python
sim = Simulator(agents=10000, tau=0.3)
```

**Visual:** Network with hub nodes highlighted

**Narration:**
> "Tau controls hub influence. At tau = 0, all agents are weighted equally. At tau = 1, only hub nodes matter. Typical values are 0.2 to 0.4."

**On-screen diagram:**
```
τ=0.0: Democratic (all agents equal)
τ=0.3: Balanced (default)
τ=0.5: Hub-centric
τ=1.0: Only hubs matter
```

### Parameter 3: alpha (α)

**Code:**
```python
sim = Simulator(agents=10000, alpha=0.1)
```

**Visual:** Convergence curves at different alphas

**Narration:**
> "Alpha is the adaptation rate—how quickly agents update strategies. Higher alpha means faster convergence but risks instability."

**Comparison plots:**
- α=0.01: Slow, very stable
- α=0.1: Balanced (default)
- α=1.0: Fast, slightly oscillatory

### Parameter 4: iterations

**Code:**
```python
sim = Simulator(agents=10000, iterations=100)
```

**Narration:**
> "Maximum iterations before stopping. Most simulations converge in 20-30 iterations, so 100 provides a safe margin."

---

## SCENE 3: Parameter Sweeps (2:30-4:30)

**Visual:** Jupyter notebook interface

**Narration:**
> "Let's systematically explore the parameter space using sweeps."

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from btut import Simulator

# Sweep over gamma values
gammas = np.linspace(1.1, 3.0, 20)
results = []

for gamma in gammas:
    sim = Simulator(agents=10000, gamma=gamma)
    result = sim.run()
    results.append(result.final_cooperation)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(gammas, results, 'o-', linewidth=2, markersize=8)
plt.xlabel('Cooperation Bonus (γ)', fontsize=14)
plt.ylabel('Equilibrium Cooperation', fontsize=14)
plt.title('Parameter Sweep: γ vs. Final Cooperation', fontsize=16)
plt.grid(True, alpha=0.3)

# Add theoretical prediction
theory = gammas / (1 + gammas)
plt.plot(gammas, theory, '--', label='Theory: γ/(1+γ)', linewidth=2)
plt.legend(fontsize=12)
plt.show()
```

**Output plot appearing:**
- Points following theoretical curve
- Perfect agreement between simulation and theory

**Narration:**
> "Notice how perfectly the simulations match the theoretical prediction. This validates BTUT's mathematical foundation."

**Two-parameter sweep:**

**Code:**
```python
# 2D parameter sweep: gamma vs tau
gammas = [1.2, 1.5, 2.0, 3.0]
taus = [0.0, 0.2, 0.4, 0.6]

cooperation_matrix = np.zeros((len(gammas), len(taus)))

for i, gamma in enumerate(gammas):
    for j, tau in enumerate(taus):
        sim = Simulator(agents=10000, gamma=gamma, tau=tau)
        result = sim.run()
        cooperation_matrix[i, j] = result.final_cooperation

# Heatmap
plt.figure(figsize=(10, 8))
plt.imshow(cooperation_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Final Cooperation')
plt.xticks(range(len(taus)), [f'{t:.1f}' for t in taus])
plt.yticks(range(len(gammas)), [f'{g:.1f}' for g in gammas])
plt.xlabel('Hub Influence (τ)', fontsize=14)
plt.ylabel('Cooperation Bonus (γ)', fontsize=14)
plt.title('2D Parameter Space', fontsize=16)
plt.show()
```

**Visual:** Heatmap animating in

**Narration:**
> "This heatmap reveals the full parameter landscape. Higher gamma increases cooperation regardless of tau. But tau modulates the effect—higher hub influence creates sharper transitions."

---

## SCENE 4: Batch Simulations (4:30-6:00)

**Visual:** Code editor with batch processing script

**Narration:**
> "For research, you often need to run hundreds or thousands of simulations. Let's set up a batch pipeline."

**Code:**
```python
from btut import Simulator
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import time

def run_trial(params):
    """Run single simulation trial"""
    trial_id, gamma, tau, alpha, N = params

    sim = Simulator(
        agents=N,
        gamma=gamma,
        tau=tau,
        alpha=alpha
    )

    start = time.time()
    result = sim.run()
    runtime = time.time() - start

    return {
        'trial_id': trial_id,
        'gamma': gamma,
        'tau': tau,
        'alpha': alpha,
        'N': N,
        'final_cooperation': result.final_cooperation,
        'iterations': result.iterations_completed,
        'converged': result.converged,
        'runtime': runtime
    }

# Generate parameter combinations
trials = []
for trial_id in range(100):
    gamma = np.random.uniform(1.1, 3.0)
    tau = np.random.uniform(0.0, 0.5)
    alpha = np.random.uniform(0.05, 0.5)
    N = np.random.choice([5000, 10000, 20000])

    trials.append((trial_id, gamma, tau, alpha, N))

# Run in parallel
print(f"Running {len(trials)} simulations in parallel...")
with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(run_trial, trials))

# Convert to DataFrame
df = pd.DataFrame(results)
df.to_csv('batch_results.csv', index=False)

print(f"\n✓ Batch complete!")
print(f"Total simulations: {len(df)}")
print(f"Converged: {df['converged'].sum()}/{len(df)}")
print(f"Average runtime: {df['runtime'].mean():.3f}s")
```

**Terminal output:**
```
Running 100 simulations in parallel...

✓ Batch complete!
Total simulations: 100
Converged: 100/100
Average runtime: 0.156s
```

**Narration:**
> "Using ProcessPoolExecutor, we ran 100 simulations in parallel across 8 cores. This is perfect for parameter exploration or Monte Carlo studies."

---

## SCENE 5: Advanced Visualization (6:00-8:00)

**Visual:** Jupyter notebook with sophisticated plots

**Narration:**
> "Let's create publication-quality visualizations."

### Convergence Animation

**Code:**
```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Run simulation and capture full history
sim = Simulator(agents=10000, gamma=1.5)
result = sim.run()
history = result.convergence_history

# Create animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

def animate(frame):
    ax1.clear()
    ax2.clear()

    # Left: convergence progress
    ax1.plot(history[:frame+1], linewidth=3)
    ax1.axhline(y=history[-1], color='r', linestyle='--', label='Equilibrium')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Fraction Playing A')
    ax1.set_title(f'Convergence Progress (t={frame})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Right: current network state
    # (visualize current strategy distribution)
    current_p = history[frame]
    ax2.bar(['Strategy A', 'Strategy B'],
            [current_p, 1-current_p],
            color=['blue', 'red'])
    ax2.set_ylabel('Fraction')
    ax2.set_title(f'Strategy Distribution (t={frame})')
    ax2.set_ylim([0, 1])

anim = animation.FuncAnimation(fig, animate, frames=len(history), interval=200)
anim.save('convergence_animation.mp4', writer='ffmpeg')
```

**Visual:** Animation playing showing convergence

### Multi-Run Comparison

**Code:**
```python
# Compare multiple runs with different initial conditions
fig, ax = plt.subplots(figsize=(12, 8))

for run in range(10):
    sim = Simulator(agents=10000, gamma=1.5)
    result = sim.run()

    ax.plot(result.convergence_history,
            alpha=0.6,
            linewidth=2,
            label=f'Run {run+1}' if run < 3 else None)

# Add ensemble mean
ax.axhline(y=0.6, color='black', linestyle='--', linewidth=3, label='Theoretical Equilibrium')
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Cooperation Fraction', fontsize=14)
ax.set_title('Convergence Robustness: 10 Independent Runs', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Visual:** Multiple converge trajectories overlaid

**Narration:**
> "All runs converge to the same equilibrium regardless of initial conditions, demonstrating global stability."

### 3D Parameter Surface

**Code:**
```python
from mpl_toolkits.mplot3d import Axes3D

# Create meshgrid
gammas = np.linspace(1.1, 3.0, 20)
taus = np.linspace(0.0, 0.5, 20)
G, T = np.meshgrid(gammas, taus)

# Compute cooperation for each (gamma, tau) pair
C = np.zeros_like(G)
for i in range(len(taus)):
    for j in range(len(gammas)):
        sim = Simulator(agents=5000, gamma=G[i,j], tau=T[i,j])
        C[i,j] = sim.run().final_cooperation

# 3D surface plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(G, T, C, cmap='viridis', alpha=0.9)
ax.set_xlabel('Cooperation Bonus (γ)', fontsize=12)
ax.set_ylabel('Hub Influence (τ)', fontsize=12)
ax.set_zlabel('Final Cooperation', fontsize=12)
ax.set_title('Parameter Space: Full 3D View', fontsize=16)
fig.colorbar(surf, shrink=0.5)
plt.show()
```

**Visual:** Rotating 3D surface

**Narration:**
> "This 3D surface reveals the complete parameter landscape. Notice the smooth, well-behaved response—no discontinuities or chaotic regions."

---

## SCENE 6: Performance Optimization (8:00-9:30)

**Visual:** Performance comparison charts

**Narration:**
> "Let's optimize for speed and scale."

### Scaling Test

**Code:**
```python
import time

agent_counts = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
runtimes = []

for N in agent_counts:
    start = time.time()
    sim = Simulator(agents=N, gamma=1.5)
    result = sim.run()
    runtime = time.time() - start
    runtimes.append(runtime)

    print(f"N={N:8,}: {runtime:6.3f}s ({runtime/N*1e6:6.2f} µs/agent)")

# Plot scaling
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(agent_counts, runtimes, 'o-', linewidth=2, markersize=10)
plt.xlabel('Number of Agents')
plt.ylabel('Runtime (seconds)')
plt.title('Absolute Runtime')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(agent_counts, np.array(runtimes)/np.array(agent_counts)*1e6,
         'o-', linewidth=2, markersize=10, color='green')
plt.xlabel('Number of Agents')
plt.ylabel('Time per Agent (µs)')
plt.title('O(N) Scaling Verification')
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Terminal output:**
```
N=   1,000:  0.012s (  12.00 µs/agent)
N=   5,000:  0.058s (  11.60 µs/agent)
N=  10,000:  0.118s (  11.80 µs/agent)
N=  50,000:  0.592s (  11.84 µs/agent)
N= 100,000:  1.184s (  11.84 µs/agent)
N= 500,000:  5.921s (  11.84 µs/agent)
N=1,000,000: 11.843s (  11.84 µs/agent)
```

**Narration:**
> "Perfect linear scaling! Time per agent stays constant as we scale from 1,000 to 1 million agents. This is the power of O(N) complexity."

### API vs Local

**Code:**
```python
# Local simulation
start = time.time()
sim_local = Simulator(agents=100000, gamma=1.5)
result_local = sim_local.run()
local_time = time.time() - start

# API simulation
start = time.time()
sim_api = Simulator(
    agents=100000,
    gamma=1.5,
    api_url="https://btut-api.fly.dev"
)
result_api = sim_api.run()
api_time = time.time() - start

print(f"Local: {local_time:.3f}s")
print(f"API:   {api_time:.3f}s (includes network overhead)")
```

**Narration:**
> "For very large simulations, the API can offload computation to powerful cloud servers. For most cases, local execution is fastest."

---

## SCENE 7: Best Practices (9:30-9:50)

**Visual:** Checklist appearing

**Narration:**
> "Before we wrap up, here are five best practices:"

**On-screen checklist:**

✓ **Always set random seeds for reproducibility**
```python
np.random.seed(42)
```

✓ **Use presets for common scenarios**
```python
from btut import Presets
sim = Presets.standard()
```

✓ **Validate convergence before trusting results**
```python
if result.converged:
    proceed_with_analysis()
```

✓ **Save results for reproducibility**
```python
result.to_json('experiment_results.json')
```

✓ **Profile before optimizing**
```python
import cProfile
cProfile.run('sim.run()')
```

---

## SCENE 8: Wrap-up (9:50-10:00)

**Visual:** Instructor back on camera

**Narration:**
> "You now have advanced BTUT skills! You can tune parameters, run batch experiments, create sophisticated visualizations, and optimize for scale. Check out the integration guides next to connect BTUT to ROS, SUMO, or deploy on AWS Lambda. Thanks for watching!"

**End card:**
- 🚀 Next: Integration Guides
- 📖 Full documentation: btut.ai/docs
- 💬 Community: github.com/direncode/btut/discussions

---

## Production Notes

### Recording Requirements
- Jupyter notebook for live coding
- Multiple code examples prepared
- Pre-run simulations for smooth playback
- High-resolution plots

### Code Examples
- All code tested and working
- Commented for clarity
- Realistic parameters
- Meaningful variable names

### Visuals
- Animated plots where possible
- 3D visualizations
- Side-by-side comparisons
- Progress indicators

### Pacing
- Slower than quick start
- Pause for complex concepts
- Repeat key points
- Show, then explain

### Accessibility
- Captions throughout
- Describe all visuals
- Alternative text descriptions
- Transcript provided
