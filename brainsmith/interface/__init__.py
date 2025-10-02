"""Brainsmith command-line interface.

Provides both brainsmith and smith CLI tools for neural network hardware acceleration.
"""

from .cli import brainsmith_main, smith_main
from .exceptions import (
    BrainsmithError,
    ConfigurationError,
    ValidationError
)

__all__ = [
    "brainsmith_main",
    "smith_main",
    "BrainsmithError",
    "ConfigurationError",
    "ValidationError"
]