"""
BTUT Python SDK
===============
Pythonic interface for BTUT simulations.

Installation:
    pip install btut-sdk

Usage:
    from btut import Simulator, Presets
    
    sim = Simulator(agents=10000, gamma=1.45)
    results = sim.run()
    print(f"Final cooperation: {results.final_cooperation}")
"""

from typing import Optional, List, Dict, Any, Tuple
import requests
import time
from dataclasses import dataclass, field
import json


@dataclass
class SimulationConfig:
    """Configuration for BTUT simulation"""
    N: int = 10000
    gamma: float = 1.45
    tau: float = 0.30
    cA_SH: float = 0.40
    cB_SH: float = 0.10
    cA_PD: float = 0.20
    cB_PD: float = 0.08
    alpha: float = 0.60
    iterations: int = 20
    m: int = 3
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class SimulationResults:
    """Results from a simulation"""
    final_cooperation: float
    convergence_history: List[float]
    iterations_completed: int
    converged: bool
    agent_count: int
    simulation_id: Optional[str] = None
    runtime_seconds: Optional[float] = None
    
    def plot(self, save_path: Optional[str] = None):
        """Plot convergence history"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.convergence_history) + 1), 
                    self.convergence_history, 
                    marker='o', linewidth=2)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, 
                       label='Full Cooperation')
            plt.xlabel('Iteration')
            plt.ylabel('Fraction Cooperating')
            plt.title(f'BTUT Convergence ({self.agent_count:,} agents)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
        
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class Simulator:
    """
    BTUT Simulator - Main interface for running simulations
    
    Examples:
        # Basic usage
        sim = Simulator(agents=10000, gamma=1.45)
        results = sim.run()
        
        # Using presets
        sim = Simulator.from_preset('quick')
        results = sim.run()
        
        # Parameter sweep
        results_list = sim.sweep('gamma', [1.2, 1.4, 1.6, 1.8])
    """
    
    def __init__(
        self,
        agents: int = 10000,
        gamma: float = 1.45,
        tau: float = 0.30,
        cA_SH: float = 0.40,
        cB_SH: float = 0.10,
        cA_PD: float = 0.20,
        cB_PD: float = 0.08,
        alpha: float = 0.60,
        iterations: int = 20,
        m: int = 3,
        seed: Optional[int] = None,
        api_url: Optional[str] = None
    ):
        """
        Initialize simulator
        
        Args:
            agents: Number of agents
            gamma: Cooperation bonus (Stag Hunt)
            tau: Hub influence exponent
            cA_SH: Cost for strategy A in Stag Hunt
            cB_SH: Cost for strategy B in Stag Hunt
            cA_PD: Cost for strategy A in Prisoner's Dilemma
            cB_PD: Cost for strategy B in Prisoner's Dilemma
            alpha: Aggression parameter (Hawk-Dove)
            iterations: Number of update iterations
            m: Minimum degree (Barabási-Albert)
            seed: Random seed for reproducibility
            api_url: API server URL (if using remote execution)
        """
        
        self.config = SimulationConfig(
            N=agents,
            gamma=gamma,
            tau=tau,
            cA_SH=cA_SH,
            cB_SH=cB_SH,
            cA_PD=cA_PD,
            cB_PD=cB_PD,
            alpha=alpha,
            iterations=iterations,
            m=m,
            seed=seed
        )
        
        self.api_url = api_url or "http://localhost:8000"
        self.use_api = api_url is not None
    
    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> 'Simulator':
        """
        Create simulator from preset configuration
        
        Args:
            preset_name: 'quick', 'standard', or 'massive'
            **kwargs: Override preset parameters
        
        Example:
            sim = Simulator.from_preset('standard', gamma=1.6)
        """
        
        presets = {
            'quick': {'agents': 10000, 'iterations': 20},
            'standard': {'agents': 100000, 'iterations': 25},
            'massive': {'agents': 500000, 'iterations': 30}
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        config = {**presets[preset_name], **kwargs}
        return cls(**config)
    
    def run(self, async_mode: bool = False) -> SimulationResults:
        """
        Run simulation
        
        Args:
            async_mode: If True, returns immediately (requires API)
        
        Returns:
            SimulationResults object
        """
        
        if self.use_api:
            return self._run_api(async_mode)
        else:
            return self._run_local()
    
    def _run_local(self) -> SimulationResults:
        """Run simulation locally"""
        try:
            from btut_engine import BTUTSimulator
            
            start = time.time()
            sim = BTUTSimulator(self.config.to_dict())
            states = sim.run()
            elapsed = time.time() - start
            
            final_state = states[-1]
            
            return SimulationResults(
                final_cooperation=final_state['fractionA'],
                convergence_history=[s['fractionA'] for s in states],
                iterations_completed=len(states),
                converged=final_state['fractionA'] > 0.99,
                agent_count=self.config.N,
                runtime_seconds=elapsed
            )
        
        except ImportError:
            raise ImportError(
                "Local execution requires btut_engine. "
                "Use api_url parameter for remote execution."
            )
    
    def _run_api(self, async_mode: bool = False) -> SimulationResults:
        """Run simulation via API"""
        
        response = requests.post(
            f"{self.api_url}/api/simulate",
            json={
                "config": self.config.to_dict(),
                "async_mode": async_mode
            }
        )
        
        response.raise_for_status()
        data = response.json()
        
        if async_mode:
            return self._wait_for_completion(data['simulation_id'])
        
        return SimulationResults(
            final_cooperation=data['results']['final_fraction_a'],
            convergence_history=data['results']['convergence_history'],
            iterations_completed=data['results']['iterations_completed'],
            converged=data['results']['converged'],
            agent_count=data['results']['agent_count'],
            simulation_id=data['simulation_id']
        )
    
    def _wait_for_completion(
        self, 
        simulation_id: str, 
        poll_interval: float = 1.0
    ) -> SimulationResults:
        """Poll API until simulation completes"""
        
        while True:
            response = requests.get(
                f"{self.api_url}/api/simulate/{simulation_id}"
            )
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'completed':
                return SimulationResults(
                    final_cooperation=data['results']['final_fraction_a'],
                    convergence_history=data['results']['convergence_history'],
                    iterations_completed=data['results']['iterations_completed'],
                    converged=data['results']['converged'],
                    agent_count=data['results']['agent_count'],
                    simulation_id=simulation_id
                )
            
            elif data['status'] == 'failed':
                raise RuntimeError(f"Simulation failed: {data['error']}")
            
            time.sleep(poll_interval)
    
    def sweep(
        self, 
        parameter: str, 
        values: List[float]
    ) -> List[SimulationResults]:
        """
        Parameter sweep - run multiple simulations varying one parameter
        
        Args:
            parameter: Name of parameter to vary
            values: List of values to test
        
        Returns:
            List of SimulationResults
        
        Example:
            results = sim.sweep('gamma', [1.2, 1.4, 1.6, 1.8])
        """
        
        results = []
        
        for value in values:
            # Create new config with modified parameter
            config_dict = self.config.to_dict()
            config_dict[parameter] = value
            
            # Create temporary simulator
            temp_sim = Simulator(**config_dict, api_url=self.api_url)
            result = temp_sim.run()
            results.append(result)
        
        return results
    
    def benchmark(
        self, 
        agent_counts: List[int] = [1000, 10000, 100000]
    ) -> Dict[str, Any]:
        """
        Run performance benchmark
        
        Args:
            agent_counts: List of agent counts to test
        
        Returns:
            Dictionary with benchmark results
        """
        
        results = []
        
        for N in agent_counts:
            config_dict = self.config.to_dict()
            config_dict['N'] = N
            
            temp_sim = Simulator(**config_dict, api_url=self.api_url)
            
            start = time.time()
            temp_sim.run()
            elapsed = time.time() - start
            
            results.append({
                'agents': N,
                'runtime_seconds': elapsed,
                'throughput': (N * self.config.iterations) / elapsed
            })
        
        return {'benchmark_results': results}


class Presets:
    """Convenient access to preset configurations"""
    
    @staticmethod
    def quick(**kwargs) -> Simulator:
        """Quick simulation (10K agents, 20 iterations)"""
        return Simulator.from_preset('quick', **kwargs)
    
    @staticmethod
    def standard(**kwargs) -> Simulator:
        """Standard simulation (100K agents, 25 iterations)"""
        return Simulator.from_preset('standard', **kwargs)
    
    @staticmethod
    def massive(**kwargs) -> Simulator:
        """Massive simulation (500K agents, 30 iterations)"""
        return Simulator.from_preset('massive', **kwargs)


# Convenience functions
def quick_run(agents: int = 10000, **kwargs) -> SimulationResults:
    """Quick one-line simulation"""
    sim = Simulator(agents=agents, **kwargs)
    return sim.run()


def parameter_sweep(
    parameter: str, 
    values: List[float], 
    agents: int = 10000
) -> List[SimulationResults]:
    """Quick parameter sweep"""
    sim = Simulator(agents=agents)
    return sim.sweep(parameter, values)


# Example usage
if __name__ == '__main__':
    print("BTUT SDK Example Usage")
    print("="*60)
    
    # Basic simulation
    print("\n1. Basic Simulation:")
    sim = Simulator(agents=10000, gamma=1.45)
    results = sim.run()
    print(f"   Final cooperation: {results.final_cooperation:.4f}")
    print(f"   Converged: {results.converged}")
    print(f"   Runtime: {results.runtime_seconds:.2f}s")
    
    # Using presets
    print("\n2. Using Presets:")
    sim = Presets.quick(gamma=1.6)
    results = sim.run()
    print(f"   Final cooperation: {results.final_cooperation:.4f}")
    
    # Parameter sweep
    print("\n3. Parameter Sweep:")
    results_list = sim.sweep('gamma', [1.2, 1.4, 1.6, 1.8])
    print("   Gamma | Cooperation")
    for gamma, res in zip([1.2, 1.4, 1.6, 1.8], results_list):
        print(f"   {gamma:.1f}   | {res.final_cooperation:.4f}")
    
    print("\n" + "="*60)
