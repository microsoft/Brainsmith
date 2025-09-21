"""Convenience helper functions for accessing configuration values.

These functions provide quick access to commonly used configuration values
without needing to import and call get_config() directly.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import BrainsmithConfig


def get_build_dir() -> Path:
    """Get the configured build directory.
    
    This is a convenience function to quickly access the build directory
    without needing to import and call get_config().
    
    Returns:
        Path to the build directory
    """
    from .loader import get_config
    return get_config().bsmith_build_dir


def get_deps_dir() -> Path:
    """Get the configured dependencies directory.
    
    This is a convenience function to quickly access the deps directory
    without needing to import and call get_config().
    
    Returns:
        Path to the dependencies directory
    """
    from .loader import get_config
    return get_config().bsmith_deps_dir


def get_bsmith_dir() -> Path:
    """Get the Brainsmith root directory.
    
    This is a convenience function to quickly access the root directory
    without needing to import and call get_config().
    
    Returns:
        Path to the Brainsmith root directory
    """
    from .loader import get_config
    return get_config().bsmith_dir


def is_plugins_strict() -> bool:
    """Check if plugins should be loaded in strict mode.
    
    This is a convenience function to quickly check the plugins_strict setting
    without needing to import and call get_config().
    
    Returns:
        True if plugins should be loaded in strict mode, False otherwise
    """
    from .loader import get_config
    return get_config().plugins_strict