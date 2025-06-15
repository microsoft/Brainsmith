"""
FINN V2 Integration Module

This module provides the bridge between Blueprint V2 design space exploration
and real FINN execution. It converts 6-entrypoint configurations to current
FINN DataflowBuildConfig format and executes actual FINN builds.

Key Components:
- LegacyConversionLayer: 6-entrypoint → DataflowBuildConfig translation
- FINNEvaluationBridge: Main DSE → FINN interface
- MetricsExtractor: FINN results → DSE metrics parsing
- ConfigBuilder: FINN configuration utilities
"""

from .evaluation_bridge import FINNEvaluationBridge
from .legacy_conversion import LegacyConversionLayer
from .metrics_extractor import MetricsExtractor
from .config_builder import ConfigBuilder

__all__ = [
    'FINNEvaluationBridge',
    'LegacyConversionLayer', 
    'MetricsExtractor',
    'ConfigBuilder'
]

__version__ = "2.0.0"