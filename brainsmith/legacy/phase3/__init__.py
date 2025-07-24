"""
Phase 3: Build Runner - Public API.

This module provides the build execution functionality for DSE V3, including
support for multiple backends and standardized metrics collection.
"""

# Data structures
from .data_structures import (
    BuildStatus,
    BuildMetrics,
    BuildResult,
)

# Core
from .build_runner import BuildRunner

# Interfaces
from .interfaces import BuildRunnerInterface

# Backends
from .legacy_finn_backend import LegacyFINNBackend
from .future_brainsmith_backend import FutureBrainsmithBackend

# Supporting components
from .preprocessing import PreprocessingPipeline
from .postprocessing import PostprocessingPipeline
from .metrics_collector import MetricsCollector
from .error_handler import BuildErrorHandler

# Factory
from .factory import create_build_runner_factory

__all__ = [
    # Data structures
    "BuildStatus",
    "BuildMetrics",
    "BuildResult",
    
    # Core
    "BuildRunner",
    
    # Interfaces
    "BuildRunnerInterface",
    
    # Backends  
    "LegacyFINNBackend",
    "FutureBrainsmithBackend",
    
    # Supporting components
    "PreprocessingPipeline",
    "PostprocessingPipeline",
    "MetricsCollector",
    "BuildErrorHandler",
    
    # Factory
    "create_build_runner_factory",
]