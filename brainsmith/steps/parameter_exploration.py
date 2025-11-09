# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parameter exploration step for DSE (Phase 7)."""
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any

from qonnx.custom_op.registry import getCustomOp

from brainsmith.registry import step
from brainsmith.dataflow.utils import iter_valid_configurations
from brainsmith.dataflow.kernel_op import KernelOp

logger = logging.getLogger(__name__)


@step(name="explore_kernel_params")
def explore_kernel_params_step(model, cfg):
    """Explore parallelization parameters for all KernelOp nodes.

    This step systematically explores all valid parallelization parameter
    configurations (SIMD, PE, etc.) for each KernelOp in the model. It uses
    the two-phase kernel construction system to efficiently validate configurations.

    The step:
    1. Finds all KernelOp nodes (domain="finn.custom_op.fpgadataflow")
    2. For each KernelOp, gets valid parameter ranges via get_valid_ranges()
    3. Explores all configurations using iter_valid_configurations()
    4. Validates each configuration with get_design_point() (returns KernelDesignPoint)
    5. Logs results and optionally saves to JSON

    This is useful for:
    - Verifying all configs work before full DSE
    - Understanding the design space size
    - Debugging configuration issues
    - Collecting baseline metrics

    Args:
        model: ModelWrapper containing the ONNX model
        cfg: FINN config object with output_dir

    Returns:
        ModelWrapper (unchanged - exploration only)

    Blueprint usage:
        steps:
          - infer_kernels
          - explore_kernel_params  # Add after kernel inference
          - create_dataflow_partition
    """
    logger.debug("=" * 80)
    logger.debug("Exploring kernel parallelization parameters...")
    logger.debug("=" * 80)

    # Find all KernelOp nodes
    kernel_nodes = []
    for node in model.graph.node:
        # Check if this is a custom op (any domain)
        # Skip standard ONNX ops (they have empty domain)
        if not node.domain or node.domain == "":
            continue

        try:
            custom_op = getCustomOp(node)
            # Check if it's a KernelOp (has get_valid_ranges method)
            if isinstance(custom_op, KernelOp):
                kernel_nodes.append((node, custom_op))
        except Exception as e:
            # Not a registered custom op or not a KernelOp, skip silently
            logger.debug(f"Skipping {node.name} ({node.op_type}): {e}")
            continue

    if not kernel_nodes:
        logger.warning("No KernelOp nodes found in model")
        logger.info("Skipping parameter exploration")
        return model

    logger.debug(f"Found {len(kernel_nodes)} KernelOp nodes to explore:")
    for node, _ in kernel_nodes:
        logger.debug(f"  - {node.name} ({node.op_type})")

    # Explore each kernel
    all_results = []
    total_start = time.time()

    for node, kernel_op in kernel_nodes:
        logger.debug("-" * 80)
        logger.debug(f"Exploring {node.name} ({node.op_type})...")

        # Get valid ranges
        try:
            valid_ranges = kernel_op.get_valid_ranges(model)
        except Exception as e:
            logger.error(f"Failed to get valid ranges for {node.name}: {e}")
            continue

        if not valid_ranges:
            logger.warning(f"  No parallelization parameters for {node.name}")
            continue

        # Log parameter space
        logger.debug(f"  Parameters: {list(valid_ranges.keys())}")
        for param_name, param_values in valid_ranges.items():
            logger.debug(f"    {param_name}: {len(param_values)} values "
                       f"(range: {min(param_values)}-{max(param_values)})")

        # Calculate total configs
        total_configs = 1
        for param_values in valid_ranges.values():
            total_configs *= len(param_values)
        logger.debug(f"  Total configurations: {total_configs:,}")

        # Explore configurations
        results = _explore_kernel_configs(node.name, kernel_op, model, total_configs)
        all_results.append(results)

    total_elapsed = time.time() - total_start

    # Log summary
    logger.debug("=" * 80)
    logger.debug("Parameter Exploration Summary")
    logger.debug("=" * 80)

    total_kernels = len(all_results)
    total_configs_explored = sum(r["configs_explored"] for r in all_results)
    total_successful = sum(r["configs_successful"] for r in all_results)
    total_failed = sum(r["configs_failed"] for r in all_results)

    logger.info(f"Kernels explored: {total_kernels}")
    logger.info(f"Total configurations: {total_configs_explored:,}")
    logger.info(f"Successful: {total_successful:,}")
    logger.info(f"Failed: {total_failed:,}")
    logger.info(f"Total time: {total_elapsed:.2f}s")

    if total_configs_explored > 0:
        avg_time_per_config = (total_elapsed / total_configs_explored) * 1000
        logger.info(f"Average time per config: {avg_time_per_config:.2f}ms")

    # Save results to JSON
    if hasattr(cfg, 'output_dir'):
        output_path = Path(cfg.output_dir) / "parameter_exploration_results.json"
        _save_results(output_path, all_results, total_elapsed)
        logger.info(f"Results saved to: {output_path}")

    logger.debug("=" * 80)

    return model


def _explore_kernel_configs(
    node_name: str,
    kernel_op: KernelOp,
    model,
    expected_count: int
) -> Dict[str, Any]:
    """Explore all configurations for a single kernel.

    Args:
        node_name: Name of the ONNX node
        kernel_op: KernelOp instance
        model: ModelWrapper
        expected_count: Expected number of configurations

    Returns:
        Dict with exploration results
    """
    start_time = time.time()
    successful = 0
    failed = 0
    config_details = []

    logger.debug(f"  Exploring {expected_count:,} configurations...")

    for i, config in enumerate(iter_valid_configurations(kernel_op, model)):
        config_start = time.time()

        try:
            # Set parameters
            for param_name, param_value in config.items():
                kernel_op.set_nodeattr(param_name, param_value)

            # Validate configuration
            design_point = kernel_op.get_design_point(model)

            # Verify parameters match
            for param_name, param_value in config.items():
                actual_value = design_point.params.get(param_name)
                if actual_value != param_value:
                    raise ValueError(
                        f"Parameter mismatch: {param_name}={actual_value}, "
                        f"expected {param_value}"
                    )

            config_time = time.time() - config_start
            successful += 1

            config_details.append({
                "config": config,
                "status": "success",
                "time_ms": config_time * 1000
            })

        except Exception as e:
            config_time = time.time() - config_start
            failed += 1
            logger.warning(f"    Config {config} failed: {e}")

            config_details.append({
                "config": config,
                "status": "failed",
                "error": str(e),
                "time_ms": config_time * 1000
            })

        # Log progress every 10 configs
        if (i + 1) % 10 == 0 or (i + 1) == expected_count:
            logger.debug(f"    Progress: {i+1}/{expected_count} configs "
                       f"({successful} successful, {failed} failed)")

    elapsed = time.time() - start_time

    logger.debug(f"  Completed in {elapsed:.2f}s")
    logger.debug(f"  Success rate: {successful}/{expected_count} "
               f"({100*successful/max(expected_count,1):.1f}%)")

    return {
        "node_name": node_name,
        "configs_explored": expected_count,
        "configs_successful": successful,
        "configs_failed": failed,
        "time_seconds": elapsed,
        "config_details": config_details
    }


def _save_results(output_path: Path, results: List[Dict[str, Any]], total_time: float):
    """Save exploration results to JSON file.

    Args:
        output_path: Path to save results
        results: List of per-kernel results
        total_time: Total exploration time
    """
    output_data = {
        "summary": {
            "total_kernels": len(results),
            "total_configs": sum(r["configs_explored"] for r in results),
            "total_successful": sum(r["configs_successful"] for r in results),
            "total_failed": sum(r["configs_failed"] for r in results),
            "total_time_seconds": total_time,
        },
        "kernels": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
