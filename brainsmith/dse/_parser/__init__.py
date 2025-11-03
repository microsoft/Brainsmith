# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint Parser - YAML to GlobalDesignSpace

This module parses blueprint YAML files and creates GlobalDesignSpace objects
with all components resolved from the registry.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

from brainsmith.dse.design_space import GlobalDesignSpace, _slice_steps
from brainsmith.dse.config import DSEConfig, extract_config

from .loader import load_blueprint_with_inheritance
from .steps import parse_steps
from .kernels import parse_kernels

logger = logging.getLogger(__name__)


def parse_blueprint(
    blueprint_path: str,
    model_path: str,
    start_step: Optional[str] = None,
    stop_step: Optional[str] = None
) -> Tuple[GlobalDesignSpace, DSEConfig]:
    """
    Parse blueprint YAML and return GlobalDesignSpace and DSEConfig.

    Inheritance is resolved bottom-up:
    1. Start from the root parent (no extends)
    2. Fully resolve its steps (including operations)
    3. Pass resolved steps to child for its operations
    4. Repeat until we reach the target blueprint

    Args:
        blueprint_path: Path to blueprint YAML file
        model_path: Path to model file
        start_step: Optional start step for slicing (overrides YAML default)
        stop_step: Optional stop step for slicing (overrides YAML default)

    Returns:
        Tuple of (GlobalDesignSpace, DSEConfig)
    """
    # Load blueprint data and check for inheritance
    raw_data, merged_data, parent_path = load_blueprint_with_inheritance(blueprint_path)

    parent_steps = None

    # If this blueprint extends another, first parse the parent
    if parent_path:
        # Recursively parse parent to get its fully resolved steps
        parent_design_space, _ = parse_blueprint(parent_path, model_path)
        parent_steps = parent_design_space.steps

    # Extract config from merged data
    blueprint_config = extract_config(merged_data)

    # Parse steps from THIS blueprint only (not inherited steps)
    # Use raw_data to get only the steps defined in this file
    steps_data = raw_data.get('design_space', {}).get('steps', [])
    steps = parse_steps(steps_data, parent_steps=parent_steps)

    # Apply step slicing if requested (parameters override YAML defaults)
    yaml_start = merged_data.get('start_step')
    yaml_stop = merged_data.get('stop_step')
    effective_start = start_step if start_step is not None else yaml_start
    effective_stop = stop_step if stop_step is not None else yaml_stop

    if effective_start or effective_stop:
        logger.info(f"Slicing steps: start={effective_start or 'beginning'}, stop={effective_stop or 'end'}")
        steps = _slice_steps(steps, effective_start, effective_stop)

    # Parse kernels (use merged data to inherit kernels)
    kernel_backends = parse_kernels(merged_data.get('design_space', {}).get('kernels', []))

    # Get max_combinations from environment or use default
    max_combinations = int(os.environ.get("BRAINSMITH_MAX_COMBINATIONS", "100000"))

    design_space = GlobalDesignSpace(
        model_path=model_path,
        steps=steps,
        kernel_backends=kernel_backends,
        max_combinations=max_combinations
    )
    return design_space, blueprint_config


__all__ = ['parse_blueprint']

