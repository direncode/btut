"""
BTUT API Middleware
===================
Custom middleware for logging, metrics, and error handling
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("btut.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing information"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration:.3f}s"
            )

            # Add timing header
            response.headers["X-Process-Time"] = str(duration)

            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """Track API metrics (request counts, response times, etc.)"""

    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.total_duration = 0.0
        self.endpoint_stats = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Increment request counter
        self.request_count += 1

        # Process request
        response = await call_next(request)

        # Update metrics
        duration = time.time() - start_time
        self.total_duration += duration

        # Track per-endpoint stats
        path = request.url.path
        if path not in self.endpoint_stats:
            self.endpoint_stats[path] = {"count": 0, "total_time": 0.0}

        self.endpoint_stats[path]["count"] += 1
        self.endpoint_stats[path]["total_time"] += duration

        return response

    def get_stats(self) -> dict:
        """Get current metrics"""
        return {
            "total_requests": self.request_count,
            "total_duration": self.total_duration,
            "average_duration": self.total_duration / self.request_count if self.request_count > 0 else 0,
            "endpoints": {
                path: {
                    "count": stats["count"],
                    "avg_time": stats["total_time"] / stats["count"]
                }
                for path, stats in self.endpoint_stats.items()
            }
        }


# Global metrics instance
metrics = None

def get_metrics() -> MetricsMiddleware:
    """Get the global metrics instance"""
    return metrics
