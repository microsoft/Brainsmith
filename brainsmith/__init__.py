# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith: Neural network hardware acceleration toolkit."""

# Export configuration to environment for FINN and other tools
# This needs to happen early before any FINN imports
try:
    from .settings import get_config
    config = get_config()
    config.export_to_environment()
except Exception:
    # Config might not be available during initial setup
    pass

# Public API exports
from .dse import explore_design_space, DesignSpace, DSEConfig, DSETree, DSESegment
from .settings import SystemConfig
from .dataflow import KernelDefinition

__all__ = [
    # Main API
    'explore_design_space',

    # DSE
    'DesignSpace',
    'DSEConfig',
    'DSETree',
    'DSESegment',

    # Configuration
    'SystemConfig',

    # Dataflow
    'KernelDefinition',
]

__version__ = "0.1.0"