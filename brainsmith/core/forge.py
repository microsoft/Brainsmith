# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Forge - End-to-End FPGA Accelerator Synthesis

This module provides the forge API that transforms neural network models
into FPGA accelerators through blueprint-driven design space exploration.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

from .blueprint_parser import BlueprintParser
from .execution_tree import ExecutionNode, get_tree_stats

logger = logging.getLogger(__name__)


def forge(model_path: str, blueprint_path: str, output_dir: str = None):
    """
    Forge an FPGA accelerator from model and blueprint.
    
    Transforms a neural network model into an FPGA accelerator through
    blueprint-driven design space exploration and synthesis.
    
    Args:
        model_path: Path to ONNX model file
        blueprint_path: Path to Blueprint YAML file
        output_dir: Output directory (defaults to $BSMITH_BUILD_DIR/forge_YYYYMMDD_HHMMSS)
        
    Returns:
        TreeExecutionResult containing build artifacts and statistics
        
    Raises:
        FileNotFoundError: If model or blueprint file doesn't exist
        ValueError: If blueprint is invalid or tree exceeds size limits
        RuntimeError: If no successful builds were produced
    """
    # Verify files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(blueprint_path):
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    # Determine output directory
    if output_dir is None:
        build_dir = os.environ.get("BSMITH_BUILD_DIR", "./build")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(build_dir, f"forge_{timestamp}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Forging FPGA accelerator:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Blueprint: {blueprint_path}")
    logger.info(f"  Output: {output_dir}")
    
    # Parse blueprint and build tree
    parser = BlueprintParser()
    design_space, tree = parser.parse(blueprint_path, os.path.abspath(model_path))
    
    logger.info(f"Design space: {len(design_space.steps)} steps, "
                f"{len(design_space.kernel_backends)} kernels")
    
    # Log tree statistics
    stats = get_tree_stats(tree)
    logger.info(f"Execution tree:")
    logger.info(f"  - Total paths: {stats['total_paths']:,}")
    logger.info(f"  - Total segments: {stats['total_segments']:,}")
    logger.info(f"  - Segment efficiency: {stats['segment_efficiency']}%")
    
    # Explore the execution tree
    logger.info("Starting design space exploration...")
    
    # Import here to avoid circular dependency
    from .explorer import explore_execution_tree
    
    results = explore_execution_tree(
        tree=tree,
        model_path=model_path,
        output_dir=output_dir,
        forge_config=design_space.config,
        design_space=design_space
    )
    
    # Check results
    result_stats = results.stats
    if result_stats['successful'] == 0:
        raise RuntimeError(f"Forge failed: No successful builds "
                         f"({result_stats['failed']} failed, {result_stats['skipped']} skipped)")
    
    logger.info(f"✅ Forge completed successfully!")
    logger.info(f"   Successful builds: {result_stats['successful']}/{result_stats['total']}")
    logger.info(f"   Total time: {results.total_time:.2f}s")
    logger.info(f"   Output directory: {output_dir}")
    
    # Attach design space and tree to results for inspection
    results.design_space = design_space
    results.execution_tree = tree
    
    return results


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