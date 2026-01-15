"""
Unit tests for BTUT Simulator class
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python-sdk'))

import pytest
import numpy as np
from btut import Simulator


class TestSimulatorInitialization:
    """Test Simulator initialization"""

    def test_default_initialization(self):
        """Test simulator with default parameters"""
        sim = Simulator(agents=1000)
        assert sim is not None

    def test_custom_parameters(self):
        """Test simulator with custom parameters"""
        sim = Simulator(
            agents=5000,
            gamma=2.0,
            tau=0.4,
            alpha=0.2
        )
        assert sim is not None

    def test_invalid_agents(self):
        """Test invalid agent count"""
        with pytest.raises(ValueError):
            Simulator(agents=0)

        with pytest.raises(ValueError):
            Simulator(agents=-100)

    def test_invalid_gamma(self):
        """Test invalid gamma parameter"""
        with pytest.raises(ValueError):
            Simulator(agents=1000, gamma=0.5)  # gamma must be > 1

    def test_invalid_tau(self):
        """Test invalid tau parameter"""
        with pytest.raises(ValueError):
            Simulator(agents=1000, tau=-0.1)  # tau must be >= 0

        with pytest.raises(ValueError):
            Simulator(agents=1000, tau=1.5)  # tau must be < 1

    def test_invalid_alpha(self):
        """Test invalid alpha parameter"""
        with pytest.raises(ValueError):
            Simulator(agents=1000, alpha=0)  # alpha must be > 0

        with pytest.raises(ValueError):
            Simulator(agents=1000, alpha=-0.1)


class TestSimulationExecution:
    """Test simulation execution"""

    def test_basic_simulation(self):
        """Test basic simulation run"""
        sim = Simulator(agents=1000, gamma=1.5)
        result = sim.run()

        assert result is not None
        assert result.converged is True
        assert 0 <= result.final_cooperation <= 1
        assert result.iterations_completed > 0

    def test_convergence(self):
        """Test that simulation converges"""
        sim = Simulator(agents=5000, gamma=1.5, tau=0.3)
        result = sim.run()

        assert result.converged is True
        assert result.iterations_completed < 100  # Should converge quickly

    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        np.random.seed(42)
        sim1 = Simulator(agents=1000, gamma=1.5)
        result1 = sim1.run()

        np.random.seed(42)
        sim2 = Simulator(agents=1000, gamma=1.5)
        result2 = sim2.run()

        assert abs(result1.final_cooperation - result2.final_cooperation) < 1e-10

    def test_convergence_history(self):
        """Test convergence history tracking"""
        sim = Simulator(agents=1000, gamma=1.5)
        result = sim.run()

        assert len(result.convergence_history) > 0
        assert len(result.convergence_history) == result.iterations_completed

    def test_runtime_tracking(self):
        """Test that runtime is tracked"""
        sim = Simulator(agents=10000, gamma=1.5)
        result = sim.run()

        assert result.runtime_seconds > 0


class TestParameterEffects:
    """Test effects of different parameters"""

    def test_gamma_effect(self):
        """Test that higher gamma increases cooperation"""
        sim_low = Simulator(agents=5000, gamma=1.2)
        result_low = sim_low.run()

        sim_high = Simulator(agents=5000, gamma=3.0)
        result_high = sim_high.run()

        # Higher gamma should lead to higher cooperation
        assert result_high.final_cooperation > result_low.final_cooperation

    def test_theoretical_equilibrium(self):
        """Test that results match theoretical prediction"""
        gamma = 1.5
        sim = Simulator(agents=10000, gamma=gamma)
        result = sim.run()

        # Theoretical equilibrium: p* = gamma / (1 + gamma)
        theoretical = gamma / (1 + gamma)

        # Should be within 5% of theory for large N
        assert abs(result.final_cooperation - theoretical) < 0.05

    def test_scaling_with_agents(self):
        """Test that convergence time doesn't scale with N"""
        results = []

        for N in [1000, 5000, 10000]:
            sim = Simulator(agents=N, gamma=1.5)
            result = sim.run()
            results.append(result.iterations_completed)

        # All should converge in similar number of iterations
        assert max(results) - min(results) < 10


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_small_population(self):
        """Test with very small population"""
        sim = Simulator(agents=10, gamma=1.5)
        result = sim.run()

        assert result is not None
        assert result.converged is True

    def test_large_population(self):
        """Test with large population"""
        sim = Simulator(agents=100000, gamma=1.5)
        result = sim.run()

        assert result is not None
        assert result.converged is True

    def test_extreme_gamma(self):
        """Test with extreme gamma values"""
        # Very low gamma (barely cooperative)
        sim_low = Simulator(agents=1000, gamma=1.01)
        result_low = sim_low.run()
        assert result_low.converged is True

        # Very high gamma
        sim_high = Simulator(agents=1000, gamma=10.0)
        result_high = sim_high.run()
        assert result_high.converged is True

    def test_zero_tau(self):
        """Test with tau = 0 (no hub influence)"""
        sim = Simulator(agents=1000, gamma=1.5, tau=0.0)
        result = sim.run()

        assert result.converged is True

    def test_high_tau(self):
        """Test with high tau (strong hub influence)"""
        sim = Simulator(agents=1000, gamma=1.5, tau=0.9)
        result = sim.run()

        assert result.converged is True


class TestResultFormat:
    """Test result object format"""

    def test_result_attributes(self):
        """Test that result has all expected attributes"""
        sim = Simulator(agents=1000, gamma=1.5)
        result = sim.run()

        assert hasattr(result, 'agent_count')
        assert hasattr(result, 'final_cooperation')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'iterations_completed')
        assert hasattr(result, 'runtime_seconds')
        assert hasattr(result, 'convergence_history')

    def test_result_to_dict(self):
        """Test result conversion to dictionary"""
        sim = Simulator(agents=1000, gamma=1.5)
        result = sim.run()

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'agent_count' in result_dict
        assert 'final_cooperation' in result_dict
        assert 'converged' in result_dict

    def test_result_to_json(self):
        """Test result JSON serialization"""
        sim = Simulator(agents=1000, gamma=1.5)
        result = sim.run()

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert 'final_cooperation' in json_str


class TestPresets:
    """Test preset configurations"""

    def test_quick_preset(self):
        """Test quick preset"""
        from btut import Presets

        sim = Presets.quick()
        result = sim.run()

        assert result.converged is True

    def test_standard_preset(self):
        """Test standard preset"""
        from btut import Presets

        sim = Presets.standard()
        result = sim.run()

        assert result.converged is True

    def test_massive_preset(self):
        """Test massive preset"""
        from btut import Presets

        sim = Presets.massive()
        result = sim.run()

        assert result.converged is True


class TestQuickRun:
    """Test quick_run utility function"""

    def test_quick_run_basic(self):
        """Test quick_run function"""
        from btut import quick_run

        result = quick_run(agents=1000, gamma=1.5)

        assert result is not None
        assert hasattr(result, 'final_cooperation')

    def test_quick_run_with_params(self):
        """Test quick_run with custom parameters"""
        from btut import quick_run

        result = quick_run(
            agents=5000,
            gamma=2.0,
            tau=0.4
        )

        assert result.converged is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
