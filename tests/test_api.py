"""
BTUT API Tests
==============
Comprehensive test suite for REST API
"""

import pytest
import requests
from typing import Dict, Any


class TestAPI:
    """Test suite for BTUT REST API"""

    BASE_URL = "http://localhost:8000"

    def test_health_check(self):
        """Test health endpoint"""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'total_simulations' in data

    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = requests.get(f"{self.BASE_URL}/")
        assert response.status_code == 200

        data = response.json()
        assert data['service'] == 'BTUT API'
        assert 'endpoints' in data
        assert 'version' in data

    def test_get_presets(self):
        """Test presets endpoint"""
        response = requests.get(f"{self.BASE_URL}/api/presets")
        assert response.status_code == 200

        data = response.json()
        assert 'presets' in data
        assert 'quick' in data['presets']
        assert 'standard' in data['presets']

    def test_run_simulation(self):
        """Test running a simulation"""
        config = {
            "config": {
                "N": 1000,
                "gamma": 1.45,
                "tau": 0.3,
                "iterations": 10
            },
            "async_mode": False
        }

        response = requests.post(f"{self.BASE_URL}/api/simulate", json=config)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'completed'
        assert 'results' in data
        assert 'simulation_id' in data

        # Validate results structure
        results = data['results']
        assert 'final_fraction_a' in results
        assert 'convergence_history' in results
        assert isinstance(results['convergence_history'], list)

    def test_validation_errors(self):
        """Test parameter validation"""
        # Invalid agent count (too low)
        config = {"config": {"N": 50}, "async_mode": False}
        response = requests.post(f"{self.BASE_URL}/api/simulate", json=config)
        assert response.status_code == 422

        # Invalid gamma (negative)
        config = {"config": {"N": 1000, "gamma": -1.0}, "async_mode": False}
        response = requests.post(f"{self.BASE_URL}/api/simulate", json=config)
        assert response.status_code == 422

    def test_get_simulation_by_id(self):
        """Test retrieving simulation by ID"""
        # First run a simulation
        config = {"config": {"N": 1000, "iterations": 5}, "async_mode": False}
        response = requests.post(f"{self.BASE_URL}/api/simulate", json=config)
        sim_id = response.json()['simulation_id']

        # Then retrieve it
        response = requests.get(f"{self.BASE_URL}/api/simulate/{sim_id}")
        assert response.status_code == 200

        data = response.json()
        assert data['simulation_id'] == sim_id

    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = requests.get(f"{self.BASE_URL}/api/stats")
        assert response.status_code == 200

        data = response.json()
        assert 'total_simulations' in data
        assert 'by_status' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
