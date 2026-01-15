"""
BTUT Analytics - Structured Logging
====================================
Production-grade structured logging
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra fields
        if hasattr(record, 'simulation_id'):
            log_data['simulation_id'] = record.simulation_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'duration'):
            log_data['duration'] = record.duration

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(level: str = 'INFO', structured: bool = True):
    """
    Configure application logging

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        structured: Use structured JSON logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if structured:
        console_handler.setFormatter(StructuredFormatter())
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    return root_logger


class BTUTLogger:
    """
    Application logger with context management
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs):
        """Set logging context"""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear logging context"""
        self.context.clear()

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method"""
        extra = {**self.context, **kwargs}
        getattr(self.logger, level)(message, extra=extra)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log('debug', message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log('info', message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log('warning', message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log('error', message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log('critical', message, **kwargs)


# Global logger instance
logger = BTUTLogger('btut')


# Usage example
if __name__ == '__main__':
    setup_logging(level='INFO', structured=True)

    logger.info("Application started")
    logger.set_context(simulation_id="sim-123", user_id="user-456")
    logger.info("Running simulation", agent_count=10000, gamma=1.45)
    logger.clear_context()
    logger.info("Simulation complete")
