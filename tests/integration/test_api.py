"""
Integration tests for BTUT API
"""

import pytest
import requests
import time
import json


# Test configuration
BASE_URL = "http://localhost:8000"  # Local testing
# BASE_URL = "https://btut-api.fly.dev"  # Production testing


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self):
        """Test health endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_has_version(self):
        """Test health check includes version"""
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()

        assert "version" in data


class TestSimulationEndpoint:
    """Test simulation endpoint"""

    def test_basic_simulation(self):
        """Test basic simulation request"""
        payload = {
            "config": {
                "N": 1000,
                "gamma": 1.5,
                "tau": 0.3,
                "alpha": 0.1
            },
            "async_mode": False
        }

        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "final_cooperation" in data["results"]
        assert data["results"]["converged"] is True

    def test_simulation_with_defaults(self):
        """Test simulation with minimal config"""
        payload = {
            "config": {
                "N": 5000
            },
            "async_mode": False
        }

        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_large_simulation(self):
        """Test large simulation"""
        payload = {
            "config": {
                "N": 100000,
                "gamma": 1.5
            },
            "async_mode": False
        }

        start = time.time()
        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)
        duration = time.time() - start

        assert response.status_code == 200
        data = response.json()

        # Should complete in reasonable time
        assert duration < 10  # seconds

    def test_parameter_validation(self):
        """Test that invalid parameters are rejected"""
        payload = {
            "config": {
                "N": -1000,  # Invalid: negative agents
                "gamma": 1.5
            },
            "async_mode": False
        }

        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)

        # Should return error
        assert response.status_code in [400, 422]

    def test_gamma_validation(self):
        """Test gamma parameter validation"""
        payload = {
            "config": {
                "N": 1000,
                "gamma": 0.5  # Invalid: gamma must be > 1
            },
            "async_mode": False
        }

        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)
        assert response.status_code in [400, 422]


class TestAsyncSimulation:
    """Test async simulation mode"""

    def test_async_submission(self):
        """Test async simulation submission"""
        payload = {
            "config": {
                "N": 10000,
                "gamma": 1.5
            },
            "async_mode": True
        }

        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)

        assert response.status_code in [200, 202]
        data = response.json()

        assert "simulation_id" in data

    def test_async_status_check(self):
        """Test checking status of async simulation"""
        # Submit async simulation
        payload = {
            "config": {
                "N": 5000,
                "gamma": 1.5
            },
            "async_mode": True
        }

        submit_response = requests.post(f"{BASE_URL}/api/simulate", json=payload)
        sim_id = submit_response.json()["simulation_id"]

        # Check status
        time.sleep(1)  # Wait a bit
        status_response = requests.get(f"{BASE_URL}/api/simulate/{sim_id}")

        assert status_response.status_code == 200
        status_data = status_response.json()

        assert "status" in status_data


class TestCORS:
    """Test CORS headers"""

    def test_cors_headers(self):
        """Test that CORS headers are present"""
        response = requests.options(f"{BASE_URL}/api/simulate")

        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "*"

    def test_cors_methods(self):
        """Test allowed CORS methods"""
        response = requests.options(f"{BASE_URL}/api/simulate")

        assert "Access-Control-Allow-Methods" in response.headers


class TestRateLimiting:
    """Test rate limiting behavior"""

    @pytest.mark.slow
    def test_many_requests(self):
        """Test handling multiple requests"""
        payload = {
            "config": {
                "N": 100,
                "gamma": 1.5
            },
            "async_mode": False
        }

        # Send 10 requests
        responses = []
        for _ in range(10):
            response = requests.post(f"{BASE_URL}/api/simulate", json=payload)
            responses.append(response)

        # All should succeed (or some may be rate limited)
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count >= 5  # At least half should succeed


class TestErrorHandling:
    """Test error handling"""

    def test_malformed_json(self):
        """Test malformed JSON request"""
        response = requests.post(
            f"{BASE_URL}/api/simulate",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code in [400, 422]

    def test_missing_config(self):
        """Test request without config"""
        response = requests.post(
            f"{BASE_URL}/api/simulate",
            json={}
        )

        # Should use defaults or return error
        assert response.status_code in [200, 400, 422]

    def test_invalid_endpoint(self):
        """Test invalid endpoint"""
        response = requests.get(f"{BASE_URL}/api/nonexistent")

        assert response.status_code == 404


class TestResponseFormat:
    """Test response format"""

    def test_response_structure(self):
        """Test response has correct structure"""
        payload = {
            "config": {
                "N": 1000,
                "gamma": 1.5
            },
            "async_mode": False
        }

        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)
        data = response.json()

        # Check required fields
        assert "results" in data
        assert "final_cooperation" in data["results"]
        assert "converged" in data["results"]
        assert "iterations_completed" in data["results"]
        assert "runtime_seconds" in data["results"]

    def test_convergence_history(self):
        """Test that convergence history is returned"""
        payload = {
            "config": {
                "N": 1000,
                "gamma": 1.5
            },
            "async_mode": False
        }

        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)
        data = response.json()

        assert "convergence_history" in data["results"]
        assert len(data["results"]["convergence_history"]) > 0


class TestPerformance:
    """Test performance characteristics"""

    def test_small_simulation_speed(self):
        """Test that small simulations are fast"""
        payload = {
            "config": {
                "N": 1000,
                "gamma": 1.5
            },
            "async_mode": False
        }

        start = time.time()
        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)
        duration = time.time() - start

        assert response.status_code == 200
        assert duration < 2  # Should complete in under 2 seconds

    def test_medium_simulation_speed(self):
        """Test medium simulation performance"""
        payload = {
            "config": {
                "N": 10000,
                "gamma": 1.5
            },
            "async_mode": False
        }

        start = time.time()
        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)
        duration = time.time() - start

        assert response.status_code == 200
        assert duration < 3  # Should complete in under 3 seconds

    @pytest.mark.slow
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import concurrent.futures

        payload = {
            "config": {
                "N": 1000,
                "gamma": 1.5
            },
            "async_mode": False
        }

        def make_request():
            return requests.post(f"{BASE_URL}/api/simulate", json=payload)

        # Send 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in futures]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)


class TestEdgeCases:
    """Test edge cases"""

    def test_maximum_agents(self):
        """Test with maximum allowed agents"""
        payload = {
            "config": {
                "N": 1000000,  # 1 million
                "gamma": 1.5
            },
            "async_mode": True  # Use async for large simulation
        }

        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)

        # Should accept request
        assert response.status_code in [200, 202]

    def test_minimum_agents(self):
        """Test with minimum agents"""
        payload = {
            "config": {
                "N": 10,
                "gamma": 1.5
            },
            "async_mode": False
        }

        response = requests.post(f"{BASE_URL}/api/simulate", json=payload)

        assert response.status_code == 200


def run_integration_tests():
    """Run all integration tests"""
    pytest.main([__file__, '-v', '-m', 'not slow'])


if __name__ == '__main__':
    run_integration_tests()
