"""
BTUT Analytics - Prometheus Metrics
====================================
Production metrics for monitoring and observability
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable

# ============================================================================
# Simulation Metrics
# ============================================================================

simulation_counter = Counter(
    'btut_simulations_total',
    'Total number of simulations run',
    ['status', 'preset']
)

simulation_duration = Histogram(
    'btut_simulation_duration_seconds',
    'Simulation execution time in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

simulation_agent_count = Histogram(
    'btut_simulation_agent_count',
    'Number of agents in simulation',
    buckets=[100, 1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
)

active_simulations = Gauge(
    'btut_simulations_active',
    'Currently running simulations'
)

simulation_iterations = Histogram(
    'btut_simulation_iterations',
    'Number of iterations to convergence',
    buckets=[5, 10, 15, 20, 30, 50, 100]
)

# ============================================================================
# API Metrics
# ============================================================================

http_requests_total = Counter(
    'btut_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'btut_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

websocket_connections = Gauge(
    'btut_websocket_connections_active',
    'Active WebSocket connections'
)

# ============================================================================
# System Metrics
# ============================================================================

system_info = Info(
    'btut_system',
    'System information'
)

memory_usage = Gauge(
    'btut_memory_usage_bytes',
    'Memory usage in bytes'
)

cpu_usage = Gauge(
    'btut_cpu_usage_percent',
    'CPU usage percentage'
)

# ============================================================================
# Decorators for Easy Instrumentation
# ============================================================================

def track_simulation(func: Callable) -> Callable:
    """Decorator to track simulation metrics"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        active_simulations.inc()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Record metrics
            simulation_counter.labels(status='success', preset='custom').inc()
            simulation_duration.observe(duration)

            if hasattr(result, 'agent_count'):
                simulation_agent_count.observe(result.agent_count)
            if hasattr(result, 'iterations_completed'):
                simulation_iterations.observe(result.iterations_completed)

            return result

        except Exception as e:
            simulation_counter.labels(status='failed', preset='custom').inc()
            raise
        finally:
            active_simulations.dec()

    return wrapper


def track_http_request(endpoint: str):
    """Decorator to track HTTP request metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = getattr(e, 'status_code', 500)
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(
                    method='POST',
                    endpoint=endpoint,
                    status=str(status)
                ).inc()
                http_request_duration.labels(
                    method='POST',
                    endpoint=endpoint
                ).observe(duration)

        return wrapper
    return decorator


# ============================================================================
# System Monitoring
# ============================================================================

def update_system_metrics():
    """Update system-level metrics"""
    try:
        import psutil

        # Memory usage
        process = psutil.Process()
        memory_usage.set(process.memory_info().rss)

        # CPU usage
        cpu_usage.set(process.cpu_percent(interval=1))

    except ImportError:
        pass  # psutil not installed


# Set system info
system_info.info({
    'version': '1.0.0',
    'python_version': '3.11',
    'platform': 'production'
})
