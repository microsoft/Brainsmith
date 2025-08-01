# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Core - DSE Architecture

This package implements the DSE architecture for FPGA accelerator design.
"""

# Main API
from .dse_api import explore_design_space

# Key components exported for external use
from .dse import DSESegment, DSETree, SegmentRunner
from .design import DesignSpace, BlueprintParser, DSETreeBuilder
from .config import ForgeConfig

__all__ = [
    # Main API
    "explore_design_space",
    # DSE components
    "DSESegment",
    "DSETree", 
    "SegmentRunner",
    # Design components
    "DesignSpace",
    "BlueprintParser",
    "DSETreeBuilder",
    # Config
    "ForgeConfig",
]