# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint Parser - YAML to GlobalDesignSpace

This module parses blueprint YAML files and creates GlobalDesignSpace objects
with all plugins resolved from the registry.
"""

import os
import logging
from typing import Any, Dict, List, Tuple

from brainsmith.dse.design_space import GlobalDesignSpace
from brainsmith.dse.config import DSEConfig, extract_config

from .loader import load_blueprint_with_inheritance
from .steps import parse_steps
from .kernels import parse_kernels

logger = logging.getLogger(__name__)


def _preload_blueprint_components(raw_data: Dict[str, Any], merged_data: Dict[str, Any]) -> None:
    """Pre-load all kernels declared in blueprint.

    Triggers lazy loading to ensure @kernel and @backend decorators fire before
    we try to validate or list components. This allows us to keep lazy loading
    for kernels (which have heavy dependencies) while ensuring blueprint
    components are available when needed.

    Note: Steps are eagerly loaded, so no preloading needed for them.

    Args:
        raw_data: Raw blueprint data (current file only)
        merged_data: Merged blueprint data (includes parent inheritance)
    """
    from brainsmith.registry import get_kernel

    # Pre-load kernels (use merged data to include inherited kernels)
    kernels_data = merged_data.get('design_space', {}).get('kernels', [])
    for kernel_name in kernels_data:
        # Kernels are now always strings (backend selection removed)
        if isinstance(kernel_name, str):
            try:
                logger.debug(f"Pre-loading kernel: {kernel_name}")
                get_kernel(kernel_name)
            except KeyError as e:
                # Will fail again during parse_kernels with better error message
                logger.debug(f"Failed to pre-load kernel '{kernel_name}': {e}")


def parse_blueprint(blueprint_path: str, model_path: str) -> Tuple[GlobalDesignSpace, DSEConfig]:
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

    Returns:
        Tuple of (GlobalDesignSpace, DSEConfig)
    """
    # Load blueprint data and check for inheritance
    raw_data, merged_data, parent_path = load_blueprint_with_inheritance(blueprint_path)

    # Pre-load all components mentioned in blueprint
    # This triggers lazy loading so @kernel, @backend, and @step decorators
    # fire before we try to validate or list components
    _preload_blueprint_components(raw_data, merged_data)

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

