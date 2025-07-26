# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Core - DSE Architecture

This package implements the DSE architecture for FPGA accelerator design.
"""

# Execution Tree API
from .forge import forge, print_tree_summary
from .execution_tree import ExecutionNode, print_tree, get_tree_stats
from .design_space import DesignSpace, GlobalConfig
from .blueprint_parser import BlueprintParser
from .utils import apply_transforms, apply_transforms_with_params

__all__ = [
    # Execution Tree API
    "forge",
    "print_tree_summary",
    "ExecutionNode",
    "DesignSpace",
    "GlobalConfig",
    "BlueprintParser",
    "print_tree",
    "get_tree_stats",
    # Utilities
    "apply_transforms",
    "apply_transforms_with_params",
]