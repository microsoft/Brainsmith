"""
Brainsmith Core - DSE Architecture

This package implements the DSE architecture for FPGA accelerator design.
"""

# Execution Tree API
from .forge import forge, print_tree_summary
from .execution_tree import ExecutionNode, TransformStage, print_tree, get_tree_stats
from .design_space import DesignSpace, GlobalConfig
from .tree_builder import build_execution_tree

__all__ = [
    # Execution Tree API
    "forge",
    "print_tree_summary",
    "ExecutionNode",
    "TransformStage",
    "DesignSpace",
    "GlobalConfig",
    "build_execution_tree",
    "print_tree",
    "get_tree_stats",
]