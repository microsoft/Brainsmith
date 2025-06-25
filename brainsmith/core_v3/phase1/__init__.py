"""
Phase 1: Design Space Constructor

This module handles parsing blueprints and constructing design spaces.
"""

from .data_structures import (
    ProcessingStep,
    HWCompilerSpace,
    ProcessingSpace,
    SearchConstraint,
    SearchConfig,
    GlobalConfig,
    DesignSpace,
    SearchStrategy,
    OutputStage,
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
    "ProcessingStep",
    "HWCompilerSpace",
    "ProcessingSpace",
    "SearchConstraint",
    "SearchConfig",
    "GlobalConfig",
    "DesignSpace",
    "SearchStrategy",
    "OutputStage",
    # Exceptions
    "BrainsmithError",
    "BlueprintParseError",
    "ValidationError",
    "ConfigurationError",
    # API
    "ForgeAPI",
    "forge",
]