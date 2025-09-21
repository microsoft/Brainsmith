"""Brainsmith configuration module.

Provides type-safe configuration management with Pydantic Settings.
"""

from .schema import BrainsmithConfig, ConfigPriority
from .loader import load_config, get_config, validate_and_report, reset_config
from .helpers import get_build_dir, get_deps_dir, get_bsmith_dir, is_plugins_strict
from .export import export_to_environment

__all__ = [
    # Core configuration classes
    "BrainsmithConfig",
    "ConfigPriority",
    
    # Configuration loading and management
    "load_config",
    "get_config",
    "validate_and_report",
    "reset_config",
    
    # Convenience helpers
    "get_build_dir",
    "get_deps_dir",
    "get_bsmith_dir",
    "is_plugins_strict",
    
    # Legacy compatibility
    "export_to_environment",
]