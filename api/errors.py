"""
BTUT API Error Handlers
=======================
Custom exception handlers for better error responses
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import logging

logger = logging.getLogger("btut.api")


class BTUTException(Exception):
    """Base exception for BTUT API"""

    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class SimulationError(BTUTException):
    """Raised when simulation execution fails"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)


class ValidationException(BTUTException):
    """Raised when input validation fails"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=422, details=details)


class ResourceNotFound(BTUTException):
    """Raised when requested resource doesn't exist"""

    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} not found: {resource_id}"
        super().__init__(message, status_code=404, details={
            "resource_type": resource_type,
            "resource_id": resource_id
        })


class RateLimitExceeded(BTUTException):
    """Raised when rate limit is exceeded"""

    def __init__(self, limit: int, window: int):
        message = f"Rate limit exceeded: {limit} requests per {window} seconds"
        super().__init__(message, status_code=429, details={
            "limit": limit,
            "window": window
        })


async def btut_exception_handler(request: Request, exc: BTUTException):
    """Handler for custom BTUT exceptions"""
    logger.error(f"BTUT Exception: {exc.message}", extra=exc.details)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
                "path": str(request.url),
                "method": request.method
            }
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler for Pydantic validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "ValidationError",
                "message": "Invalid request data",
                "details": exc.errors(),
                "path": str(request.url),
                "method": request.method
            }
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handler for unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": {"error": str(exc)},
                "path": str(request.url),
                "method": request.method
            }
        }
    )


def register_error_handlers(app):
    """Register all error handlers with the FastAPI app"""
    app.add_exception_handler(BTUTException, btut_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
