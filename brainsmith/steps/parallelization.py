# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parallelization transformation steps.

These steps provide drop-in replacements for FINN's parallelization pipeline,
working with both legacy FINN HWCustomOp nodes and modern Brainsmith KernelOp nodes.

The step layer focuses on extracting configuration from the build system (cfg),
while ApplyParallelizationConfig and SetParallelization handle the actual logic.
"""
import logging
from typing import Any

from brainsmith.primitives.transforms.parallelization import (
    ApplyParallelizationConfig,
    SetParallelization,
)
from brainsmith.registry import step

logger = logging.getLogger(__name__)


@step(name="apply_parallelization_config")
def apply_parallelization_config_step(model: Any, cfg: Any) -> Any:
    """Apply parallelization config from JSON file.

    Drop-in replacement for FINN's ApplyConfig for parallelization.
    Works with both FINN HWCustomOp and Brainsmith KernelOp nodes.

    Config file path is read from cfg.folding_config_file (FINN convention).

    Args:
        model: ModelWrapper to transform
        cfg: Build configuration with folding_config_file attribute

    Returns:
        ModelWrapper with parallelization applied

    Example config format:
        {
            "Defaults": {
                "PE": [1, ["all"]]
            },
            "MVAU_0": {"PE": 8, "SIMD": 4},
            "LayerNorm_0": {"PE": 16}
        }
    """
    config_file = getattr(cfg, "folding_config_file", None)

    if config_file is None:
        logger.warning("No folding_config_file specified in config, skipping parallelization")
        return model

    logger.debug(f"Applying parallelization config from: {config_file}")
    model = model.transform(ApplyParallelizationConfig(config_file))

    return model


@step(name="target_fps_parallelization")
def target_fps_parallelization_step(model: Any, cfg: Any) -> Any:
    """Auto-generate parallelization from target FPS.

    Drop-in replacement for FINN's SetFolding/target_fps_parallelization.
    Works with both FINN HWCustomOp and Brainsmith KernelOp nodes.

    Target cycles are calculated from cfg.target_fps and cfg.synth_clk_period_ns:
        target_cycles = (1 / target_fps) / (clock_period_ns * 1e-9)

    Args:
        model: ModelWrapper to transform
        cfg: Build configuration with target_fps and synth_clk_period_ns attributes

    Returns:
        ModelWrapper with parallelization optimized for target FPS

    Example:
        target_fps = 100 (frames per second)
        synth_clk_period_ns = 5.0 (5ns clock = 200MHz)
        target_cycles = 1e9 / (100 * 5.0) = 2,000,000 cycles per frame
    """
    target_fps = getattr(cfg, "target_fps", None)

    if target_fps is None:
        logger.warning("No target_fps specified in config, skipping auto-parallelization")
        return model

    # Get clock period (default to 5ns if not specified)
    clock_period_ns = getattr(cfg, "synth_clk_period_ns", 5.0)

    # Calculate target cycles from FPS
    # Cycles = (1 second / target_fps) / clock_period
    # Convert to integer cycles
    target_cycles = int(1e9 / (target_fps * clock_period_ns))

    logger.debug(
        f"Auto-generating parallelization for target_fps={target_fps}, "
        f"clock={clock_period_ns}ns, target_cycles={target_cycles}"
    )

    # Get optional MVAU weight stream width constraint (default 36 bits)
    mvau_wwidth_max = getattr(cfg, "mvau_wwidth_max", 36)

    # Get optional two-pass relaxation flag (default True)
    two_pass_relaxation = getattr(cfg, "two_pass_relaxation", True)

    model = model.transform(
        SetParallelization(
            target_cycles_per_frame=target_cycles,
            mvau_wwidth_max=mvau_wwidth_max,
            two_pass_relaxation=two_pass_relaxation,
        )
    )

    return model
