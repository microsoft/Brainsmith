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
from .interfaces import run_exploration

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
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(blueprint_path).exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    # Determine output directory
    if output_dir is None:
        build_dir = Path(os.environ.get("BSMITH_BUILD_DIR", "./build"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(build_dir / f"forge_{timestamp}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Forging FPGA accelerator:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Blueprint: {blueprint_path}")
    logger.info(f"  Output: {output_dir}")
    
    # Parse blueprint and build tree
    parser = BlueprintParser()
    design_space, tree = parser.parse(blueprint_path, str(Path(model_path).absolute()))
    
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
    
    results = run_exploration(
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

