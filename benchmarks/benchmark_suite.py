"""
BTUT Benchmark Suite
====================
Comprehensive comparison against established multi-agent simulation frameworks.

Competitors:
- NetLogo: Agent-based modeling (educational/research standard)
- MASON: Java-based ABM (George Mason University)
- RePast: Agent-based social simulation
- Mesa: Python ABM framework

Metrics:
1. Runtime scaling (agents vs time)
2. Memory usage
3. Convergence speed
4. Accuracy of equilibrium prediction
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import subprocess
import json

# Import BTUT
import sys
sys.path.append('../lib/simulation')
from btut_engine import BTUTSimulator, PRESETS


class BenchmarkSuite:
    """Comprehensive benchmarking framework"""
    
    def __init__(self):
        self.results = {
            'btut': [],
            'netlogo': [],
            'mason': [],
            'repast': [],
            'mesa': []
        }
    
    def benchmark_btut(self, N: int, iterations: int = 20) -> Dict:
        """Benchmark BTUT performance"""
        config = {
            **PRESETS['standard'],
            'N': N,
            'iterations': iterations
        }
        
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start = time.time()
        sim = BTUTSimulator(config)
        states = sim.run()
        elapsed = time.time() - start
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        
        # Results
        final_state = states[-1]
        
        return {
            'framework': 'BTUT',
            'agents': N,
            'iterations': iterations,
            'runtime_seconds': elapsed,
            'memory_mb': mem_used,
            'final_cooperation': final_state['fractionA'],
            'convergence_speed': self._compute_convergence_speed(
                [s['fractionA'] for s in states]
            ),
            'throughput_agent_steps_per_sec': (N * iterations) / elapsed
        }
    
    def benchmark_netlogo(self, N: int, iterations: int = 20) -> Dict:
        """
        Benchmark NetLogo via headless execution
        Requires NetLogo installed and cooperation.nlogo model
        """
        # Create NetLogo script
        netlogo_script = f"""
        setup
        set num-agents {N}
        repeat {iterations} [
          go
        ]
        export-world "netlogo_results.csv"
        """
        
        try:
            start = time.time()
            
            # Run NetLogo headless (adjust path to your installation)
            subprocess.run([
                'netlogo-headless.sh',
                '--model', 'netlogo/cooperation.nlogo',
                '--experiment', 'benchmark',
                '--table', 'netlogo_results.csv'
            ], timeout=300, check=True)
            
            elapsed = time.time() - start
            
            # Parse results
            with open('netlogo_results.csv', 'r') as f:
                # Extract final cooperation level
                final_coop = float(f.readlines()[-1].split(',')[2])
            
            return {
                'framework': 'NetLogo',
                'agents': N,
                'iterations': iterations,
                'runtime_seconds': elapsed,
                'memory_mb': None,  # Hard to measure external process
                'final_cooperation': final_coop,
                'convergence_speed': None,
                'throughput_agent_steps_per_sec': (N * iterations) / elapsed
            }
        
        except Exception as e:
            print(f"NetLogo benchmark failed: {e}")
            return None
    
    def benchmark_mason(self, N: int, iterations: int = 20) -> Dict:
        """
        Benchmark MASON via Java execution
        Requires compiled MASON cooperation model
        """
        mason_config = {
            'numAgents': N,
            'steps': iterations,
            'gamma': 1.45,
            'alpha': 0.60
        }
        
        # Write config
        with open('mason_config.json', 'w') as f:
            json.dump(mason_config, f)
        
        try:
            start = time.time()
            
            # Run MASON simulation
            result = subprocess.run([
                'java', '-jar', 'mason/CooperationSim.jar',
                '--config', 'mason_config.json'
            ], capture_output=True, timeout=300)
            
            elapsed = time.time() - start
            
            # Parse output
            output = json.loads(result.stdout)
            
            return {
                'framework': 'MASON',
                'agents': N,
                'iterations': iterations,
                'runtime_seconds': elapsed,
                'memory_mb': None,
                'final_cooperation': output['finalCooperation'],
                'convergence_speed': None,
                'throughput_agent_steps_per_sec': (N * iterations) / elapsed
            }
        
        except Exception as e:
            print(f"MASON benchmark failed: {e}")
            return None
    
    def benchmark_mesa(self, N: int, iterations: int = 20) -> Dict:
        """
        Benchmark Mesa (Python ABM)
        Can run in-process
        """
        try:
            from mesa import Model, Agent
            from mesa.time import RandomActivation
            from mesa.datacollection import DataCollector
            
            # Define simple cooperation model
            class CooperationAgent(Agent):
                def __init__(self, unique_id, model, strategy='A'):
                    super().__init__(unique_id, model)
                    self.strategy = strategy
                    self.utility = 0
                
                def step(self):
                    # Simple cooperation logic
                    neighbors = self.model.grid.get_neighbors(
                        self.pos, moore=True, include_center=False
                    )
                    
                    coop_neighbors = sum(1 for n in neighbors 
                                        if n.strategy == 'A')
                    total_neighbors = len(neighbors)
                    
                    if total_neighbors > 0:
                        coop_ratio = coop_neighbors / total_neighbors
                        # Update strategy based on neighbors
                        if coop_ratio > 0.5:
                            self.strategy = 'A'
                        else:
                            self.strategy = 'B'
            
            class CooperationModel(Model):
                def __init__(self, N):
                    self.num_agents = N
                    self.schedule = RandomActivation(self)
                    
                    # Create agents
                    for i in range(N):
                        agent = CooperationAgent(i, self)
                        self.schedule.add(agent)
                    
                    self.datacollector = DataCollector(
                        model_reporters={
                            "Cooperation": lambda m: sum(
                                1 for a in m.schedule.agents 
                                if a.strategy == 'A'
                            ) / m.num_agents
                        }
                    )
                
                def step(self):
                    self.datacollector.collect(self)
                    self.schedule.step()
            
            # Benchmark
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            start = time.time()
            model = CooperationModel(N)
            for _ in range(iterations):
                model.step()
            elapsed = time.time() - start
            
            mem_after = process.memory_info().rss / 1024 / 1024
            
            # Get final cooperation
            data = model.datacollector.get_model_vars_dataframe()
            final_coop = data['Cooperation'].iloc[-1]
            
            return {
                'framework': 'Mesa',
                'agents': N,
                'iterations': iterations,
                'runtime_seconds': elapsed,
                'memory_mb': mem_after - mem_before,
                'final_cooperation': final_coop,
                'convergence_speed': self._compute_convergence_speed(
                    data['Cooperation'].values
                ),
                'throughput_agent_steps_per_sec': (N * iterations) / elapsed
            }
        
        except Exception as e:
            print(f"Mesa benchmark failed: {e}")
            return None
    
    def _compute_convergence_speed(self, history: List[float]) -> float:
        """
        Compute convergence speed as iterations to reach 95% of final value
        """
        if len(history) < 2:
            return None
        
        final_val = history[-1]
        target = 0.95 * final_val
        
        for i, val in enumerate(history):
            if val >= target:
                return i
        
        return len(history)
    
    def run_scaling_benchmark(self, 
                            agent_counts: List[int],
                            frameworks: List[str] = ['btut', 'mesa']):
        """Run scaling benchmarks across multiple frameworks"""
        
        results = []
        
        for N in agent_counts:
            print(f"\n{'='*60}")
            print(f"Benchmarking N={N:,} agents")
            print(f"{'='*60}")
            
            for framework in frameworks:
                print(f"\nTesting {framework.upper()}...")
                
                if framework == 'btut':
                    result = self.benchmark_btut(N)
                elif framework == 'netlogo':
                    result = self.benchmark_netlogo(N)
                elif framework == 'mason':
                    result = self.benchmark_mason(N)
                elif framework == 'mesa':
                    result = self.benchmark_mesa(N)
                else:
                    continue
                
                if result:
                    results.append(result)
                    print(f"  Runtime: {result['runtime_seconds']:.2f}s")
                    print(f"  Memory: {result['memory_mb']:.1f} MB")
                    print(f"  Throughput: {result['throughput_agent_steps_per_sec']:,.0f} agent-steps/s")
        
        return pd.DataFrame(results)
    
    def visualize_results(self, df: pd.DataFrame, output_path: str = 'benchmark_results.png'):
        """Create visualization of benchmark results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Runtime comparison
        ax = axes[0, 0]
        for framework in df['framework'].unique():
            data = df[df['framework'] == framework]
            ax.plot(data['agents'], data['runtime_seconds'], 
                   marker='o', label=framework, linewidth=2)
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Scaling: Runtime vs Agents')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Throughput comparison
        ax = axes[0, 1]
        frameworks = df['framework'].unique()
        throughputs = [df[df['framework'] == f]['throughput_agent_steps_per_sec'].mean() 
                      for f in frameworks]
        ax.bar(frameworks, throughputs)
        ax.set_ylabel('Agent-Steps per Second')
        ax.set_title('Average Throughput')
        ax.set_yscale('log')
        
        # Memory usage
        ax = axes[1, 0]
        for framework in df['framework'].unique():
            data = df[df['framework'] == framework]
            if data['memory_mb'].notna().any():
                ax.plot(data['agents'], data['memory_mb'], 
                       marker='o', label=framework, linewidth=2)
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Scaling')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Speedup over baseline
        ax = axes[1, 1]
        baseline = df[df['framework'] == 'mesa']['runtime_seconds'].values
        if len(baseline) > 0:
            for framework in df['framework'].unique():
                if framework == 'mesa':
                    continue
                data = df[df['framework'] == framework]
                speedup = baseline / data['runtime_seconds'].values
                ax.plot(data['agents'], speedup, 
                       marker='o', label=f'{framework} vs Mesa', linewidth=2)
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Speedup vs Mesa (Baseline)')
        ax.set_xscale('log')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to {output_path}")
        
        return fig


def main():
    """Run full benchmark suite"""
    
    suite = BenchmarkSuite()
    
    # Test scaling from 1K to 100K agents
    agent_counts = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    
    # Only test BTUT and Mesa by default (others require installation)
    frameworks = ['btut', 'mesa']
    
    print("="*60)
    print("BTUT BENCHMARK SUITE")
    print("="*60)
    print(f"Testing frameworks: {', '.join(frameworks)}")
    print(f"Agent counts: {agent_counts}")
    print("="*60)
    
    # Run benchmarks
    results_df = suite.run_scaling_benchmark(agent_counts, frameworks)
    
    # Save results
    results_df.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to benchmark_results.csv")
    
    # Visualize
    suite.visualize_results(results_df)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for framework in results_df['framework'].unique():
        data = results_df[results_df['framework'] == framework]
        avg_throughput = data['throughput_agent_steps_per_sec'].mean()
        max_agents = data['agents'].max()
        
        print(f"\n{framework.upper()}:")
        print(f"  Max agents tested: {max_agents:,}")
        print(f"  Avg throughput: {avg_throughput:,.0f} agent-steps/s")
        print(f"  Avg runtime (100K agents): {data[data['agents']==100000]['runtime_seconds'].values[0]:.2f}s")
    
    # Compute speedup
    if 'mesa' in results_df['framework'].values and 'btut' in results_df['framework'].values:
        mesa_time = results_df[
            (results_df['framework'] == 'mesa') & 
            (results_df['agents'] == 100_000)
        ]['runtime_seconds'].values[0]
        
        btut_time = results_df[
            (results_df['framework'] == 'btut') & 
            (results_df['agents'] == 100_000)
        ]['runtime_seconds'].values[0]
        
        speedup = mesa_time / btut_time
        print(f"\nBTUT is {speedup:.1f}x faster than Mesa at 100K agents")


if __name__ == '__main__':
    main()
