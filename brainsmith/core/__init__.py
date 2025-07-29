# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Core - DSE Architecture

This package implements the DSE architecture for FPGA accelerator design.
"""

# Main API
from .forge import forge

# Internal APIs - needed by other parts of brainsmith
from .execution_tree import ExecutionSegment, print_tree, get_tree_stats
from .design_space import DesignSpace
from .config import ForgeConfig
from .blueprint_parser import BlueprintParser

__all__ = [
    # Main API
    "forge",
]