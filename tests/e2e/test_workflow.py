"""
End-to-end workflow tests for BTUT platform
"""

import pytest
import requests
import time
import json
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python-sdk'))

from btut import Simulator, Presets, quick_run


class TestCompleteWorkflow:
    """Test complete user workflows"""

    def test_python_sdk_workflow(self):
        """Test complete Python SDK workflow"""
        # 1. Create simulator
        sim = Simulator(agents=1000, gamma=1.5)

        # 2. Run simulation
        result = sim.run()

        # 3. Verify results
        assert result.converged is True
        assert 0 < result.final_cooperation < 1

        # 4. Export results
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)

        # 5. Convert to JSON
        json_str = result.to_json()
        assert isinstance(json_str, str)

        # 6. Verify can be parsed back
        parsed = json.loads(json_str)
        assert parsed["final_cooperation"] == result.final_cooperation

    def test_preset_workflow(self):
        """Test preset-based workflow"""
        # 1. Use preset
        sim = Presets.quick()

        # 2. Run simulation
        result = sim.run()

        # 3. Verify
        assert result.converged is True

    def test_parameter_sweep_workflow(self):
        """Test parameter sweep workflow"""
        # 1. Define parameter range
        gammas = [1.2, 1.5, 2.0]

        # 2. Run sweep
        results = []
        for gamma in gammas:
            sim = Simulator(agents=1000, gamma=gamma)
            result = sim.run()
            results.append(result.final_cooperation)

        # 3. Verify results increase with gamma
        assert results[0] < results[1] < results[2]

    def test_batch_simulation_workflow(self):
        """Test batch simulation workflow"""
        import numpy as np

        # 1. Run multiple trials
        trials = 10
        cooperation_rates = []

        for _ in range(trials):
            sim = Simulator(agents=1000, gamma=1.5)
            result = sim.run()
            cooperation_rates.append(result.final_cooperation)

        # 2. Compute statistics
        mean_coop = np.mean(cooperation_rates)
        std_coop = np.std(cooperation_rates)

        # 3. Verify consistency
        assert std_coop < 0.1  # Low variance expected


class TestAPIWorkflow:
    """Test API-based workflows"""

    BASE_URL = "http://localhost:8000"

    def test_synchronous_api_workflow(self):
        """Test synchronous API workflow"""
        # 1. Submit simulation
        payload = {
            "config": {
                "N": 1000,
                "gamma": 1.5
            },
            "async_mode": False
        }

        response = requests.post(f"{self.BASE_URL}/api/simulate", json=payload)

        # 2. Verify response
        assert response.status_code == 200
        data = response.json()

        # 3. Extract results
        assert "results" in data
        cooperation = data["results"]["final_cooperation"]

        # 4. Verify result is reasonable
        assert 0.5 < cooperation < 0.7

    @pytest.mark.asynctest
    def test_asynchronous_api_workflow(self):
        """Test asynchronous API workflow"""
        # 1. Submit async simulation
        payload = {
            "config": {
                "N": 10000,
                "gamma": 1.5
            },
            "async_mode": True
        }

        submit_response = requests.post(f"{self.BASE_URL}/api/simulate", json=payload)
        assert submit_response.status_code in [200, 202]

        # 2. Get simulation ID
        sim_id = submit_response.json()["simulation_id"]

        # 3. Poll for completion
        max_attempts = 30
        for _ in range(max_attempts):
            status_response = requests.get(f"{self.BASE_URL}/api/simulate/{sim_id}")
            status_data = status_response.json()

            if status_data["status"] == "completed":
                break

            time.sleep(1)

        # 4. Verify completion
        assert status_data["status"] == "completed"
        assert "results" in status_data


class TestResearchWorkflow:
    """Test research-oriented workflows"""

    def test_parameter_exploration_workflow(self):
        """Test systematic parameter exploration"""
        import numpy as np

        # 1. Define parameter grid
        gammas = np.linspace(1.1, 3.0, 10)
        taus = [0.0, 0.2, 0.4]

        # 2. Run all combinations
        results = {}

        for gamma in gammas:
            for tau in taus:
                sim = Simulator(agents=5000, gamma=gamma, tau=tau)
                result = sim.run()

                results[(gamma, tau)] = {
                    'cooperation': result.final_cooperation,
                    'iterations': result.iterations_completed,
                    'converged': result.converged
                }

        # 3. Verify all converged
        assert all(r['converged'] for r in results.values())

        # 4. Verify trends
        # Higher gamma should increase cooperation
        for tau in taus:
            cooperations = [results[(g, tau)]['cooperation'] for g in gammas]
            # Check mostly increasing
            increases = sum(1 for i in range(len(cooperations)-1)
                          if cooperations[i+1] > cooperations[i])
            assert increases >= 7  # At least 7 out of 9 transitions should increase

    def test_convergence_analysis_workflow(self):
        """Test convergence analysis workflow"""
        # 1. Run simulation with history tracking
        sim = Simulator(agents=5000, gamma=1.5)
        result = sim.run()

        # 2. Extract convergence history
        history = result.convergence_history

        # 3. Analyze convergence behavior
        assert len(history) > 0

        # Check monotonic approach (mostly)
        equilibrium = result.final_cooperation
        distances = [abs(p - equilibrium) for p in history]

        # Distance should generally decrease
        assert distances[-1] < distances[0]

    def test_scaling_analysis_workflow(self):
        """Test scaling analysis workflow"""
        import time

        # 1. Test different population sizes
        sizes = [1000, 5000, 10000, 50000]
        runtimes = []

        for N in sizes:
            start = time.time()
            sim = Simulator(agents=N, gamma=1.5)
            result = sim.run()
            runtime = time.time() - start

            runtimes.append(runtime)

            # Verify convergence
            assert result.converged is True

        # 2. Verify O(N) scaling
        # Runtime should roughly double when N doubles
        # (allowing for some variance)
        ratio = runtimes[-1] / runtimes[0]
        size_ratio = sizes[-1] / sizes[0]

        # Should be linear: ratio ≈ size_ratio
        assert 0.5 * size_ratio < ratio < 2.0 * size_ratio


class TestProductionWorkflow:
    """Test production deployment workflows"""

    def test_error_handling_workflow(self):
        """Test error handling in production"""
        # 1. Try invalid parameters
        try:
            sim = Simulator(agents=-1000, gamma=1.5)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected

        # 2. Try boundary cases
        sim = Simulator(agents=10, gamma=1.01)
        result = sim.run()
        assert result is not None

    def test_result_persistence_workflow(self):
        """Test saving and loading results"""
        import tempfile
        import os

        # 1. Run simulation
        sim = Simulator(agents=1000, gamma=1.5)
        result = sim.run()

        # 2. Save to file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write(result.to_json())
            temp_path = f.name

        # 3. Load from file
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)

        # 4. Verify data matches
        assert loaded_data['final_cooperation'] == result.final_cooperation
        assert loaded_data['converged'] == result.converged

        # 5. Cleanup
        os.unlink(temp_path)

    def test_monitoring_workflow(self):
        """Test monitoring and logging workflow"""
        import logging

        # 1. Set up logging
        logger = logging.getLogger('btut_test')
        logger.setLevel(logging.INFO)

        # 2. Run simulation with logging
        logger.info("Starting simulation")
        sim = Simulator(agents=5000, gamma=1.5)
        result = sim.run()
        logger.info(f"Simulation completed: cooperation={result.final_cooperation:.2%}")

        # 3. Verify result
        assert result.converged is True


class TestIntegrationWorkflow:
    """Test integration with other systems"""

    def test_web_app_workflow(self):
        """Test workflow for web application integration"""
        # Simulate web app calling API

        # 1. User submits form
        user_config = {
            "agents": 10000,
            "gamma": 1.8,
            "tau": 0.35
        }

        # 2. Frontend sends to backend API
        payload = {
            "config": {
                "N": user_config["agents"],
                "gamma": user_config["gamma"],
                "tau": user_config["tau"]
            },
            "async_mode": False
        }

        # 3. API processes request (local simulation)
        sim = Simulator(
            agents=payload["config"]["N"],
            gamma=payload["config"]["gamma"],
            tau=payload["config"]["tau"]
        )
        result = sim.run()

        # 4. Format response for frontend
        response = {
            "success": True,
            "results": {
                "cooperation": result.final_cooperation,
                "converged": result.converged,
                "iterations": result.iterations_completed
            }
        }

        # 5. Verify response structure
        assert response["success"] is True
        assert "cooperation" in response["results"]

    def test_batch_processing_workflow(self):
        """Test batch processing workflow"""
        # 1. Define batch of simulations
        batch = [
            {"agents": 1000, "gamma": 1.2},
            {"agents": 1000, "gamma": 1.5},
            {"agents": 1000, "gamma": 2.0},
            {"agents": 1000, "gamma": 3.0}
        ]

        # 2. Process batch
        results = []
        for config in batch:
            sim = Simulator(agents=config["agents"], gamma=config["gamma"])
            result = sim.run()
            results.append({
                "config": config,
                "cooperation": result.final_cooperation,
                "converged": result.converged
            })

        # 3. Verify all succeeded
        assert len(results) == len(batch)
        assert all(r["converged"] for r in results)

        # 4. Verify trend
        cooperations = [r["cooperation"] for r in results]
        assert all(cooperations[i] < cooperations[i+1] for i in range(len(cooperations)-1))


class TestDocumentationExamples:
    """Test all examples from documentation"""

    def test_quickstart_example(self):
        """Test the quickstart guide example"""
        # From docs/getting-started/quickstart.md

        # Create simulator with 10,000 agents
        sim = Simulator(agents=10000, gamma=1.45)

        # Run simulation
        results = sim.run()

        # View results
        assert results.final_cooperation > 0
        assert results.converged is True
        assert results.iterations_completed > 0

    def test_preset_example(self):
        """Test preset example from docs"""
        # Quick test (10K agents)
        sim = Presets.quick()
        results = sim.run()
        assert results.converged is True

        # Standard simulation (100K agents)
        sim = Presets.standard()
        results = sim.run()
        assert results.converged is True

    def test_parameter_sweep_example(self):
        """Test parameter sweep example"""
        sim = Simulator(agents=10000)

        # Test different cooperation bonuses
        gammas = [1.2, 1.4, 1.6, 1.8, 2.0]
        cooperations = []

        for gamma in gammas:
            sim = Simulator(agents=10000, gamma=gamma)
            result = sim.run()
            cooperations.append(result.final_cooperation)

        # Verify all succeeded
        assert len(cooperations) == len(gammas)


def run_e2e_tests():
    """Run all end-to-end tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_e2e_tests()
