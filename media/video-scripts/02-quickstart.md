# BTUT Quick Start Tutorial (5 minutes)

**Target Audience:** New users wanting hands-on experience
**Goal:** Get users running their first simulation

---

## SCENE 1: Introduction (0:00-0:20)

**Visual:** Instructor on camera with BTUT logo in corner

**Narration:**
> "Welcome to the BTUT Quick Start Tutorial. In the next 5 minutes, you'll install BTUT, run your first simulation, and understand the results. Let's dive in!"

**On-screen text:**
- What you'll learn:
  - ✓ Installation
  - ✓ First simulation
  - ✓ Understanding results
  - ✓ Parameter tuning

---

## SCENE 2: Installation (0:20-1:00)

**Visual:** Screen recording of terminal

**Narration:**
> "First, let's install BTUT. Make sure you have Python 3.8 or newer. Open your terminal and type:"

**Terminal commands:**
```bash
# Check Python version
python --version

# Install BTUT from PyPI
pip install btut-sdk

# Verify installation
python -c "import btut; print(btut.__version__)"
```

**On-screen text:**
- Requirements: Python ≥ 3.8
- Installation time: ~30 seconds
- Dependencies: numpy, requests

**Narration:**
> "BTUT installs in about 30 seconds. Once installed, let's write our first simulation."

---

## SCENE 3: Your First Simulation (1:00-2:30)

**Visual:** Code editor (VS Code or Jupyter) side-by-side with terminal

**Narration:**
> "Create a new Python file called 'first_simulation.py'. Here's the complete code:"

**Code appearing line by line:**

```python
# first_simulation.py
from btut import Simulator

# Create simulator with 10,000 agents
print("Creating simulator...")
sim = Simulator(agents=10000, gamma=1.5)

# Run simulation
print("Running simulation...")
results = sim.run()

# Display results
print(f"\nSimulation Complete!")
print(f"Agents: {results.agent_count:,}")
print(f"Final cooperation: {results.final_cooperation:.2%}")
print(f"Converged: {results.converged}")
print(f"Iterations: {results.iterations_completed}")
print(f"Runtime: {results.runtime_seconds:.3f}s")
```

**Narration explaining each section:**
> "Line 1: Import the Simulator class.
>
> Line 4: Create a simulator with 10,000 agents and a cooperation bonus gamma of 1.5.
>
> Line 8: Run the simulation. This executes the mean-field dynamics until convergence.
>
> Lines 11-16: Print the results—how many agents, the final cooperation rate, whether it converged, how many iterations it took, and the total runtime."

**Terminal output:**
```
Creating simulator...
Running simulation...

Simulation Complete!
Agents: 10,000
Final cooperation: 60.00%
Converged: True
Iterations: 19
Runtime: 0.124s
```

**On-screen callout:**
- 10,000 agents simulated in 0.12 seconds!

---

## SCENE 4: Understanding Results (2:30-3:30)

**Visual:** Result explanation with annotations

**Narration:**
> "Let's understand what these results mean."

**On-screen annotations:**

**Final cooperation: 60.00%**
> "This is the fraction of agents playing strategy A at equilibrium. In a coordination game with gamma = 1.5, theory predicts 60% cooperation, which is exactly what we see."

**Converged: True**
> "The simulation reached a stable Nash equilibrium. All agents have settled on their optimal strategies given what others are doing."

**Iterations: 19**
> "The dynamics converged in just 19 iterations. BTUT's mean-field approach is extremely efficient—typically converging in 15-30 iterations regardless of agent count."

**Runtime: 0.124s**
> "Total time for the complete simulation. Note the O(N) scaling: doubling the agents roughly doubles the time."

---

## SCENE 5: Visualizing Convergence (3:30-4:15)

**Visual:** Code editor, then plot appearing

**Narration:**
> "Let's visualize how the system converged. Add this code:"

**Code:**
```python
# Visualize convergence
results.plot()
```

**Plot appearing:**
- Line graph showing cooperation fraction over iterations
- Starts around 0.5 (random)
- Smoothly approaches 0.6
- Plateaus around iteration 15-20

**Narration:**
> "The plot method shows cooperation over time. Notice how it starts near 50%—agents begin with random strategies. Then it smoothly approaches the equilibrium at 60% and stabilizes."

**On-screen callout:**
- Smooth convergence
- No oscillations
- Reaches equilibrium quickly

---

## SCENE 6: Parameter Exploration (4:15-4:45)

**Visual:** Code with different parameters

**Narration:**
> "Now let's explore how different parameters affect outcomes. Try varying gamma:"

**Code:**
```python
# Try different cooperation bonuses
for gamma in [1.2, 1.5, 2.0, 3.0]:
    sim = Simulator(agents=10000, gamma=gamma)
    result = sim.run()
    print(f"γ={gamma:.1f}: {result.final_cooperation:.2%} cooperation")
```

**Terminal output:**
```
γ=1.2: 54.55% cooperation
γ=1.5: 60.00% cooperation
γ=2.0: 66.67% cooperation
γ=3.0: 75.00% cooperation
```

**Narration:**
> "Higher gamma means a bigger bonus for coordinating on strategy B, which actually increases cooperation on A at equilibrium. This is a key insight from game theory!"

**On-screen formula:**
```
p* = γ / (1 + γ)
```

---

## SCENE 7: Next Steps (4:45-5:00)

**Visual:** Documentation page thumbnails

**Narration:**
> "Congratulations! You've run your first BTUT simulation. To learn more, check out:
>
> The Parameter Tuning Guide for advanced configuration.
>
> The API Reference for all available options.
>
> And the Examples folder for real-world use cases.
>
> Happy simulating!"

**On-screen text:**
- 📖 Parameter Tuning Guide
- 📚 API Reference
- 💡 Example Notebooks
- 💬 Community Forum

**End card:**
> "BTUT Quick Start Complete!
>  Next: Advanced Tutorial"

---

## Production Notes

### Screen Recording Setup
- Clean desktop
- Large font in terminal (16pt+)
- Large font in code editor (14pt+)
- Syntax highlighting on
- Line numbers visible

### Code Editor Configuration
- Theme: Dark (easier to watch)
- Font: Fira Code or similar
- No distractions (hide sidebar)
- Full screen mode

### Terminal Configuration
- Theme: Dark with high contrast
- Font: 16pt minimum
- Clear prompt (simple $)
- Wide window (120 columns)

### Narration Style
- Clear and paced
- Pause after commands
- Explain what's happening
- Emphasize key points

### Visual Effects
- Highlight lines as they're discussed
- Zoom in on important output
- Animated arrows pointing to key values
- Smooth transitions between scenes

### Tools
- Recording: OBS Studio
- Editing: DaVinci Resolve
- Annotations: After Effects
- Code highlighting: Carbon.sh for stills

### Accessibility
- Closed captions
- High contrast visuals
- Large text
- Clear audio
- Transcript available

### Additional Materials
- Downloadable code files
- Jupyter notebook version
- Sample datasets
- FAQ document
