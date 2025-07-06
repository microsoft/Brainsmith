"""
Phase 1: Design Space Constructor

This module handles parsing blueprints and constructing design spaces.
"""

from .data_structures import (
    HWCompilerSpace,
    SearchConstraint,
    SearchConfig,
    GlobalConfig,
    DesignSpace,
    SearchStrategy,
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
    "SearchConstraint",
    "SearchConfig",
    "GlobalConfig",
    "DesignSpace",
    "SearchStrategy",
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