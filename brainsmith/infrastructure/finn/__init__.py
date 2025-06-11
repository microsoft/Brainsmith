"""
FINN Infrastructure Module

Simplified FINN interface for BrainSmith FPGA accelerator builds.
Provides clean abstractions over FINN functionality with preparation
for future 4-hooks interface evolution.

Main exports:
- FINNInterface: Main interface class
- build_accelerator: Simple function interface
- validate_finn_config: Configuration validation
- prepare_4hooks_config: 4-hooks preparation
"""

from .interface import (
    FINNInterface,
    build_accelerator,
    validate_finn_config,
    prepare_4hooks_config
)

from .types import (
    FINNConfig,
    FINNResult,
    FINNBuildMetrics,
    FINNHooksConfig,
    FINNDevice,
    FINNOptimization,
    FINNConfigDict,
    FINNMetricsDict,
    FINNResultDict
)

__all__ = [
    # Main interface
    "FINNInterface",
    
    # Convenience functions
    "build_accelerator",
    "validate_finn_config", 
    "prepare_4hooks_config",
    
    # Types
    "FINNConfig",
    "FINNResult",
    "FINNBuildMetrics", 
    "FINNHooksConfig",
    "FINNDevice",
    "FINNOptimization",
    
    # Type aliases
    "FINNConfigDict",
    "FINNMetricsDict",
    "FINNResultDict"
]