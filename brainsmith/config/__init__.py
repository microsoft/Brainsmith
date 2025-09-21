"""Brainsmith configuration module.

Provides type-safe configuration management with Pydantic.
"""

from .schema import BrainsmithConfig, ConfigPriority
from .loader import load_config, get_config, validate_and_report, reset_config
from .migrate import export_to_environment, import_from_environment

__all__ = [
    "BrainsmithConfig",
    "ConfigPriority", 
    "load_config",
    "get_config",
    "validate_and_report",
    "reset_config",
    "export_to_environment",
    "import_from_environment",
]