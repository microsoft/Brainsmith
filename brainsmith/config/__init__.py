"""Brainsmith configuration module.

Provides type-safe configuration management with Pydantic Settings.
"""

from .schema import BrainsmithConfig
from .loader import load_config, get_config, reset_config, get_default_config
from .helpers import get_build_dir, get_deps_dir, get_bsmith_dir, is_plugins_strict

__all__ = [
    # Core configuration classes
    "BrainsmithConfig",
    
    # Configuration loading and management
    "load_config",
    "get_config",
    "reset_config",
    "get_default_config",
    
    # Convenience helpers
    "get_build_dir",
    "get_deps_dir",
    "get_bsmith_dir",
    "is_plugins_strict"
]