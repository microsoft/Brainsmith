"""
Brainsmith Core - DSE Architecture

This package implements the DSE architecture for FPGA accelerator design.
"""

# Execution Tree API
from .forge import forge, print_tree_summary
from .execution_tree import ExecutionNode, TransformStage, print_tree, get_tree_stats
from .design_space import DesignSpace, GlobalConfig
from .blueprint_parser import BlueprintParser

__all__ = [
    # Execution Tree API
    "forge",
    "print_tree_summary",
    "ExecutionNode",
    "TransformStage",
    "DesignSpace",
    "GlobalConfig",
    "BlueprintParser",
    "print_tree",
    "get_tree_stats",
]