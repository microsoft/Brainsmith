"""
Brainsmith Core V3 - Clean DSE Architecture

This package implements the new three-phase DSE architecture:
1. Design Space Constructor - Parse blueprints into design spaces
2. Design Space Explorer - Systematically explore configurations  
3. Build Runner - Execute builds and collect metrics
"""

__version__ = "3.0.0"

# New Execution Tree API
from .forge_v2 import forge_tree, print_tree_summary
from .execution_tree import ExecutionNode, TransformStage, print_tree, get_tree_stats
from .design_space import DesignSpace, GlobalConfig
from .tree_builder import build_execution_tree

__all__ = [
    # Execution Tree API
    "forge_tree",
    "print_tree_summary",
    "ExecutionNode",
    "TransformStage",
    "DesignSpace",
    "GlobalConfig",
    "build_execution_tree",
    "print_tree",
    "get_tree_stats",
]