"""
Phase 1: Design Space Constructor

This module handles parsing blueprints and constructing design spaces.
"""

from .data_structures import (
    HWCompilerSpace,
    GlobalConfig,
    DesignSpace,
    OutputStage,
    BuildMetrics,
)
from .exceptions import (
    BrainsmithError,
    BlueprintParseError,
    ValidationError,
    ConfigurationError,
)
from .forge import ForgeAPI, forge

__all__ = [
    # Data structures
    "HWCompilerSpace",
    "GlobalConfig",
    "DesignSpace",
    "OutputStage",
    "BuildMetrics",
    # Exceptions
    "BrainsmithError",
    "BlueprintParseError",
    "ValidationError",
    "ConfigurationError",
    # API
    "ForgeAPI",
    "forge",
]