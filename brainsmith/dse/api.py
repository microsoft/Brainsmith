# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
DSE API - Design Space Exploration for FPGA Accelerator Synthesis

This module provides the DSE API that transforms neural network models
into FPGA accelerators through blueprint-driven design space exploration.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from brainsmith.dse._parser import parse_blueprint
from brainsmith.dse._builder import DSETreeBuilder
from brainsmith.dse.design_space import _slice_steps
from brainsmith.dse.tree import DSETree
from brainsmith.dse.runner import SegmentRunner
from brainsmith._internal.finn.adapter import FINNAdapter
from brainsmith.dse.types import TreeExecutionResult

logger = logging.getLogger(__name__)


def explore_design_space(
    model_path: str,
    blueprint_path: str,
    output_dir: Optional[str] = None,
    start_step_override: Optional[str] = None,
    stop_step_override: Optional[str] = None
) -> TreeExecutionResult:
    """
    Explore the design space for an FPGA accelerator.

    Transforms a neural network model into an FPGA accelerator through
    blueprint-driven design space exploration and synthesis.

    Args:
        model_path: Path to ONNX model file
        blueprint_path: Path to Blueprint YAML file
        output_dir: Output directory (defaults to $BSMITH_BUILD_DIR/dfc_YYYYMMDD_HHMMSS)
        start_step_override: Override blueprint start_step (CLI takes precedence)
        stop_step_override: Override blueprint stop_step (CLI takes precedence)

    Returns:
        TreeExecutionResult containing build artifacts and statistics

    Raises:
        FileNotFoundError: If model or blueprint file doesn't exist
        ValueError: If blueprint is invalid or tree exceeds size limits
        RuntimeError: If no successful builds were produced
    """
    # Verify files exist
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path_obj}")
    blueprint_path_obj = Path(blueprint_path)
    if not blueprint_path_obj.exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path_obj}")

    # Load config early to ensure BSMITH_* vars are exported for YAML expansion
    from brainsmith.settings import get_config
    get_config()  # Exports BSMITH_DIR and other env vars

    # Determine output directory
    if not output_dir:
        build_dir = get_config().build_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(build_dir / f"dfc_{timestamp}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exploring design space for FPGA accelerator:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Blueprint: {blueprint_path}")
    logger.info(f"  Output: {output_dir}")

    # Parse blueprint
    design_space, blueprint_config = parse_blueprint(blueprint_path, str(Path(model_path).absolute()))

    # Apply CLI overrides (CLI > blueprint)
    start_step = start_step_override or blueprint_config.start_step
    stop_step = stop_step_override or blueprint_config.stop_step

    # Slice steps if specified
    if start_step or stop_step:
        logger.info(f"Applying step range: start={start_step or 'beginning'}, stop={stop_step or 'end'}")
        design_space.steps = _slice_steps(design_space.steps, start_step, stop_step)
    
    # Build DSE tree
    tree_builder = DSETreeBuilder()
    tree = tree_builder.build_tree(design_space, blueprint_config)
    
    logger.info(f"Design space: {len(design_space.steps)} steps, "
                f"{len(design_space.kernel_backends)} kernels")
    
    # Log tree statistics
    stats = tree.get_statistics()
    logger.info(f"DSE tree:")
    logger.info(f"  - Total paths: {stats['total_paths']:,}")
    logger.info(f"  - Total segments: {stats['total_segments']:,}")
    logger.info(f"  - Segment efficiency: {stats['segment_efficiency']}%")
    
    # Explore the DSE tree
    logger.info("Starting design space exploration...")
    
    # Create runner and execute
    finn_adapter = FINNAdapter()
    runner = SegmentRunner(finn_adapter, tree.root.finn_config)
    results = runner.run_tree(
        tree=tree,
        initial_model=Path(model_path),
        output_dir=Path(output_dir)
    )
    
    # Validate results
    results.validate_success(Path(output_dir))

    # Log success
    result_stats = results.compute_stats()
    logger.info("Design space exploration completed successfully")
    logger.info(f"  Successful builds: {result_stats['successful']}/{result_stats['total']}")
    logger.info(f"  Total time: {results.total_time:.2f}s")
    logger.info(f"  Output directory: {output_dir}")

    return TreeExecutionResult(
        segment_results=results.segment_results,
        total_time=results.total_time,
        design_space=design_space,
        dse_tree=tree
    )


def build_tree(
    design_space: GlobalDesignSpace,
    config: DSEConfig
) -> DSETree:
    """Build execution tree from design space.

    Separates tree construction from execution for inspection and validation.

    Raises:
        ValueError: If tree exceeds max_combinations limit
    """
    builder = DSETreeBuilder()
    return builder.build_tree(design_space, config)


def execute_tree(
    tree: DSETree,
    model_path: str,
    config: DSEConfig,
    output_dir: str,
    runner: Optional[SegmentRunner] = None
) -> TreeExecutionResult:
    """Execute a pre-built DSE tree.

    Separates execution from tree construction for custom execution strategies.

    Args:
        runner: Custom segment runner (optional, uses default if None)

    Raises:
        RuntimeError: If no successful builds were produced
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create default runner if not provided
    if runner is None:
        finn_adapter = FINNAdapter()
        runner = SegmentRunner(finn_adapter, tree.root.finn_config)

    # Execute tree
    results = runner.run_tree(
        tree=tree,
        initial_model=Path(model_path),
        output_dir=output_path
    )

    # Validate results
    results.validate_success(output_path)

    # Return result with all fields
    return TreeExecutionResult(
        segment_results=results.segment_results,
        total_time=results.total_time,
        design_space=None,  # Not available in this path
        dse_tree=tree
    )


