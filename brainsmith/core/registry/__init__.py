"""
Core Registry Infrastructure

Provides the unified BaseRegistry abstract base class and supporting infrastructure
for consistent registry implementation across all Brainsmith components.
"""

from .base import BaseRegistry, ComponentInfo
from .exceptions import RegistryError, ComponentNotFoundError, ValidationError

__all__ = [
    "BaseRegistry",
    "ComponentInfo", 
    "RegistryError",
    "ComponentNotFoundError",
    "ValidationError"
]