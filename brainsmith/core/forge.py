# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Forge - Blueprint to Execution Tree Pipeline

This module provides the forge API that creates execution trees
directly from blueprints with automatic prefix sharing.
"""

import os
import logging
from typing import Tuple

from .blueprint_parser import BlueprintParser
from .design_space import DesignSpace

from .execution_tree import ExecutionNode, print_tree, get_tree_stats

logger = logging.getLogger(__name__)


def forge(model_path: str, blueprint_path: str) -> Tuple[DesignSpace, ExecutionNode]:
    """
    Create execution tree from model and blueprint.
    
    Args:
        model_path: Path to ONNX model file
        blueprint_path: Path to Blueprint YAML file
        
    Returns:
        Tuple of (DesignSpace, ExecutionTree root)
        
    Raises:
        FileNotFoundError: If model or blueprint file doesn't exist
        ValueError: If blueprint is invalid or tree exceeds size limits
    """
    # Verify files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(blueprint_path):
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    logger.info(f"Forging execution tree from {model_path} and {blueprint_path}")
    
    # Parse blueprint and build tree in one step
    parser = BlueprintParser()
    design_space, tree = parser.parse(blueprint_path, os.path.abspath(model_path))
    
    logger.info(f"Parsed design space: {len(design_space.steps)} steps, "
                f"{len(design_space.kernel_backends)} kernels")
    
    # Log statistics
    stats = get_tree_stats(tree)
    logger.info(f"✅ Created execution tree:")
    logger.info(f"   - Total paths: {stats['total_paths']:,}")
    
    # Segment-based statistics
    logger.info(f"   - Total segments: {stats['total_segments']:,}")
    logger.info(f"   - Segment efficiency: {stats['segment_efficiency']}%")
    logger.info(f"   - Avg steps/segment: {stats['avg_steps_per_segment']}")
    
    return design_space, tree


def print_tree_summary(tree: ExecutionNode, max_depth: int = 3) -> None:
    """
    Print a summary of the execution tree structure.
    
    Args:
        tree: Root node of execution tree
        max_depth: Maximum depth to display
    """
    print("\nExecution Tree Structure:")
    print("=" * 60)
    
    # Print tree with limited depth
    _print_tree_limited(tree, max_depth=max_depth)
    
    # Print statistics
    stats = get_tree_stats(tree)
    print("\nTree Statistics:")
    print("-" * 30)
    print(f"Total execution paths: {stats['total_paths']:,}")
    
    # Segment-based statistics
    print(f"Total segments: {stats['total_segments']:,}")
    print(f"Maximum depth: {stats['max_depth']}")
    print(f"Segment efficiency: {stats['segment_efficiency']}%")
    print(f"Average steps per segment: {stats['avg_steps_per_segment']}")
    print(f"Total steps saved: {stats['steps_without_segments'] - stats['total_steps']:,}")


def _print_tree_limited(node: ExecutionNode, max_depth: int, current_depth: int = 0, 
                       indent: str = "", last: bool = True):
    """Print tree with depth limit."""
    if current_depth > max_depth:
        return
    
    # Segment-based node structure
    node_id = node.segment_id
    is_root = node_id == "root"
    
    if not is_root:
        prefix = "└── " if last else "├── "
        
        # Show segment info
        if node.branch_decision:
            info = f" ({len(node.segment_steps)} steps)"
        else:
            info = ""
        print(f"{indent}{prefix}{node.branch_decision or 'segment'}{info}")
        
        if current_depth == max_depth and node.children:
            extension = "    " if last else "│   "
            print(f"{indent}{extension}└── ... ({len(node.children)} children)")
            return
    
    extension = "    " if last else "│   "
    
    # Children is a dict
    child_items = list(node.children.items())
    for i, (_, child) in enumerate(child_items):
        _print_tree_limited(
            child, 
            max_depth, 
            current_depth + 1,
            indent + extension if not is_root else "",
            i == len(child_items) - 1
        )