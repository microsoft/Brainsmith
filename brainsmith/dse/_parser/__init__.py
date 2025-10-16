# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint Parser - YAML to DesignSpace

This module parses blueprint YAML files and creates DesignSpace objects
with all plugins resolved from the registry.
"""

import os
from typing import Tuple

from brainsmith.dse.design_space import DesignSpace
from brainsmith.dse.config import DSEConfig, extract_config

from .loader import load_blueprint_with_inheritance
from .steps import parse_steps
from .kernels import parse_kernels


def parse_blueprint(blueprint_path: str, model_path: str) -> Tuple[DesignSpace, DSEConfig]:
    """
    Parse blueprint YAML and return DesignSpace and DSEConfig.

    Inheritance is resolved bottom-up:
    1. Start from the root parent (no extends)
    2. Fully resolve its steps (including operations)
    3. Pass resolved steps to child for its operations
    4. Repeat until we reach the target blueprint

    Args:
        blueprint_path: Path to blueprint YAML file
        model_path: Path to model file

    Returns:
        Tuple of (DesignSpace, DSEConfig)
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

    # Parse kernels (use merged data to inherit kernels)
    kernel_backends = parse_kernels(merged_data.get('design_space', {}).get('kernels', []))

    # Get max_combinations from environment or use default
    max_combinations = int(os.environ.get("BRAINSMITH_MAX_COMBINATIONS", "100000"))

    design_space = DesignSpace(
        model_path=model_path,
        steps=steps,
        kernel_backends=kernel_backends,
        max_combinations=max_combinations
    )
    design_space.validate_size()
    return design_space, blueprint_config


__all__ = ['parse_blueprint']
