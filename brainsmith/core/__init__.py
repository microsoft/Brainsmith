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
from .design import DesignSpace, parse_blueprint, DSETreeBuilder
from .config import BlueprintConfig

__all__ = [
    # Main API
    "explore_design_space",
    # DSE components
    "DSESegment",
    "DSETree", 
    "SegmentRunner",
    # Design components
    "DesignSpace",
    "parse_blueprint",
    "DSETreeBuilder",
    # Config
    "BlueprintConfig",
]