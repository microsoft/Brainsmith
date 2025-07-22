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

# Feature flag for segment-based tree
use_segment_tree = os.environ.get("BRAINSMITH_SEGMENT_TREE", "false").lower() == "true"

if use_segment_tree:
    from .execution_tree_v2 import ExecutionNode, print_tree, get_tree_stats
    from .tree_builder_v2 import build_execution_tree, validate_tree_size
    logger = logging.getLogger(__name__)
    logger.info("Using segment-based execution tree (v2)")
else:
    from .execution_tree import ExecutionNode, print_tree, get_tree_stats
    from .tree_builder import build_execution_tree, validate_tree_size
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
    
    # Parse blueprint to design space
    parser = BlueprintParser()
    design_space = parser.parse(blueprint_path, os.path.abspath(model_path))
    
    logger.info(f"Parsed design space: {len(design_space.transform_stages)} stages, "
                f"{len(design_space.kernel_backends)} kernels")
    
    # Build execution tree
    tree = build_execution_tree(design_space)
    
    # Validate tree size
    validate_tree_size(tree, design_space.global_config.max_combinations)
    
    # Log statistics
    stats = get_tree_stats(tree)
    logger.info(f"✅ Created execution tree:")
    logger.info(f"   - Total paths: {stats['total_paths']:,}")
    
    if use_segment_tree:
        # v2 statistics
        logger.info(f"   - Total segments: {stats['total_segments']:,}")
        logger.info(f"   - Segment efficiency: {stats['segment_efficiency']}%")
        logger.info(f"   - Avg steps/segment: {stats['avg_steps_per_segment']}")
    else:
        # v1 statistics
        logger.info(f"   - Total nodes: {stats['total_nodes']:,}")
        logger.info(f"   - Sharing factor: {stats['sharing_factor']}x")
        logger.info(f"   - Saved nodes: {stats['saved_nodes']:,}")
    
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
    
    if use_segment_tree:
        # v2 statistics
        print(f"Total segments: {stats['total_segments']:,}")
        print(f"Maximum depth: {stats['max_depth']}")
        print(f"Segment efficiency: {stats['segment_efficiency']}%")
        print(f"Average steps per segment: {stats['avg_steps_per_segment']}")
        print(f"Total steps saved: {stats['steps_without_segments'] - stats['total_steps']:,}")
    else:
        # v1 statistics
        print(f"Total tree nodes: {stats['total_nodes']:,}")
        print(f"Maximum depth: {stats['max_depth']}")
        print(f"Sharing factor: {stats['sharing_factor']}x")
        print(f"Computation saved: {stats['saved_nodes']:,} nodes")


def _print_tree_limited(node: ExecutionNode, max_depth: int, current_depth: int = 0, 
                       indent: str = "", last: bool = True):
    """Print tree with depth limit."""
    if current_depth > max_depth:
        return
    
    # Handle both v1 and v2 node structures
    if use_segment_tree:
        # v2: segment-based
        node_id = node.segment_id
        is_root = node_id == "root"
    else:
        # v1: step-based
        node_id = node.step_name
        is_root = node_id == "root"
    
    if not is_root:
        prefix = "└── " if last else "├── "
        
        # Format node info based on version
        if use_segment_tree:
            # v2: Show segment info
            if node.branch_decision:
                info = f" ({len(node.segment_steps)} steps)"
            else:
                info = ""
            print(f"{indent}{prefix}{node.branch_decision or 'segment'}{info}")
        else:
            # v1: Show step info
            if "transforms" in node.config:
                transforms = node.config["transforms"]
                if transforms:
                    names = [t.__name__ for t in transforms]
                    info = f" ({', '.join(names)})"
                else:
                    info = " (empty)"
            elif "kernel_backends" in node.config:
                info = f" ({len(node.config['kernel_backends'])} kernels)"
            else:
                info = ""
            print(f"{indent}{prefix}{node_id}{info}")
        
        if current_depth == max_depth and node.children:
            extension = "    " if last else "│   "
            print(f"{indent}{extension}└── ... ({len(node.children)} children)")
            return
    
    extension = "    " if last else "│   "
    
    # Handle both dict (v2) and list (v1) children
    if hasattr(node.children, 'items'):
        # v2: children is a dict
        child_items = list(node.children.items())
        for i, (_, child) in enumerate(child_items):
            _print_tree_limited(
                child, 
                max_depth, 
                current_depth + 1,
                indent + extension if not is_root else "",
                i == len(child_items) - 1
            )
    else:
        # v1: children is a list
        for i, child in enumerate(node.children):
            _print_tree_limited(
                child, 
                max_depth, 
                current_depth + 1,
                indent + extension if not is_root else "",
                i == len(node.children) - 1
            )