# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint Parser - YAML to DesignSpace

This module parses blueprint YAML files and creates DesignSpace objects
with all plugins resolved from the registry.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass

from .space import DesignSpace
from brainsmith.core.config import ForgeConfig
from brainsmith.core.plugins.registry import get_registry, has_step, list_backends_by_kernel, get_backend
from brainsmith.utils.yaml_parser import load_yaml, expand_env_vars_with_context

# Type definitions
StepSpec = Union[str, List[Optional[str]]]

# Skip indicators
SKIP_VALUES = frozenset([None, "~", ""])
SKIP_NORMALIZED = "~"


@dataclass
class StepOperation:
    """Represents a step manipulation operation"""
    op_type: Literal["after", "before", "replace", "remove", "at_start", "at_end"]
    target: Optional[StepSpec] = None
    insert: Optional[StepSpec] = None
    with_step: Optional[StepSpec] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['StepOperation']:
        """Parse operation from YAML dict"""
        op_mappings = {
            "after": lambda d: cls(op_type="after", target=d["after"], insert=d.get("insert")),
            "before": lambda d: cls(op_type="before", target=d["before"], insert=d.get("insert")),
            "replace": lambda d: cls(op_type="replace", target=d["replace"], with_step=d.get("with")),
            "remove": lambda d: cls(op_type="remove", target=d["remove"]),
            "at_start": lambda d: cls(op_type="at_start", insert=d["at_start"]["insert"]),
            "at_end": lambda d: cls(op_type="at_end", insert=d["at_end"]["insert"]),
        }

        for key, factory in op_mappings.items():
            if key in data:
                return factory(data)
        return None


def parse_blueprint(blueprint_path: str, model_path: str) -> Tuple[DesignSpace, ForgeConfig]:
    """
    Parse blueprint YAML and return DesignSpace and ForgeConfig.

    Inheritance is resolved bottom-up:
    1. Start from the root parent (no extends)
    2. Fully resolve its steps (including operations)
    3. Pass resolved steps to child for its operations
    4. Repeat until we reach the target blueprint
    """
    # Load raw data to check inheritance chain
    raw_data = load_yaml(
        blueprint_path,
        expand_env_vars=True,
        support_inheritance=False,
        context_vars={'BLUEPRINT_DIR': str(Path(blueprint_path).parent.absolute())}
    )

    parent_steps = None

    # If this blueprint extends another, first parse the parent
    if 'extends' in raw_data:
        parent_path = raw_data['extends']
        # Expand env vars in parent path
        parent_path = expand_env_vars_with_context(
            parent_path,
            {'BSMITH_DIR': os.environ.get('BSMITH_DIR', str(Path(__file__).parent.parent.parent.parent.absolute()))}
        )

        # Resolve parent path relative to current file
        if not Path(parent_path).is_absolute():
            parent_path = str(Path(blueprint_path).parent / parent_path)

        # Recursively parse parent to get its fully resolved steps
        parent_design_space, _ = parse_blueprint(parent_path, model_path)
        parent_steps = parent_design_space.steps

    # Now load the full merged data for config extraction
    blueprint_data = load_yaml(
        blueprint_path,
        expand_env_vars=True,
        support_inheritance=True,
        context_vars={'BLUEPRINT_DIR': str(Path(blueprint_path).parent.absolute())}
    )

    forge_config = _extract_config_and_mappings(blueprint_data)

    # Parse steps from THIS blueprint only (not inherited steps)
    # Use raw_data to get only the steps defined in this file
    steps_data = raw_data.get('design_space', {}).get('steps', [])
    steps = _parse_steps(steps_data, parent_steps=parent_steps)

    # Parse kernels (use merged data to inherit kernels)
    kernel_backends = _parse_kernels(blueprint_data.get('design_space', {}).get('kernels', []))

    # Get max_combinations from environment or use default
    max_combinations = int(os.environ.get("BRAINSMITH_MAX_COMBINATIONS", "100000"))

    design_space = DesignSpace(
        model_path=model_path,
        steps=steps,
        kernel_backends=kernel_backends,
        max_combinations=max_combinations
    )
    design_space.validate_size()
    return design_space, forge_config


def _extract_config_and_mappings(data: Dict[str, Any]) -> ForgeConfig:
    """Extract ForgeConfig from blueprint data."""
    # Extract config - check both flat and global_config
    config_data = {**data.get('global_config', {}), **data}

    # Validate required field
    if 'clock_ns' not in config_data:
        raise ValueError("Missing required field 'clock_ns' in blueprint")

    return ForgeConfig(
        clock_ns=float(config_data['clock_ns']),
        output=config_data.get('output', 'estimates'),
        board=config_data.get('board'),
        verify=config_data.get('verify', False),
        verify_data=Path(config_data['verify_data']) if 'verify_data' in config_data else None,
        parallel_builds=config_data.get('parallel_builds', 4),
        debug=config_data.get('debug', False),
        save_intermediate_models=config_data.get('save_intermediate_models', False),
        start_step=config_data.get('start_step'),
        stop_step=config_data.get('stop_step'),
        finn_overrides=data.get('finn_config', {})
    )






def _parse_steps_raw(steps_data: List[Any]) -> List[Union[str, List[Optional[str]]]]:
    """Parse steps without operations (for parent blueprints)."""
    registry = get_registry()
    return [_validate_spec(spec, registry) for spec in steps_data if not isinstance(spec, dict)]


def _parse_steps(
    steps_data: List[Any],
    parent_steps: Optional[List[Union[str, List[Optional[str]]]]] = None
) -> List[Union[str, List[Optional[str]]]]:
    """Parse steps from design_space, preserving variations and supporting operations."""
    registry = get_registry()

    # Separate operations from direct steps
    operations = []
    direct_steps = []
    for item in steps_data:
        if isinstance(item, dict):
            op = StepOperation.from_dict(item)
            if op:
                operations.append(op)
        else:
            direct_steps.append(item)

    # Determine base steps
    if direct_steps:
        # Direct steps specified: use them (child replaces parent)
        result = [_validate_spec(spec, registry) for spec in direct_steps]
    elif parent_steps:
        # No direct steps but have parent: start with parent
        result = parent_steps.copy()
    else:
        # No steps at all: empty
        result = []

    # Apply operations
    for op in operations:
        result = _apply_step_operation(result, op)

    # Validate result (operations might have added unvalidated steps)
    return [_validate_spec(spec, registry) for spec in result]


def _apply_step_operation(steps: List[StepSpec], op: StepOperation) -> List[StepSpec]:
    """Apply a single operation to the step list"""

    # Get registry for normalization
    registry = get_registry()

    # Normalize the operation target to match already-normalized steps
    normalized_target = None
    if op.target is not None:
        normalized_target = _validate_spec(op.target, registry)

    # Validate nested lists in operation specs
    _validate_nested_lists(op.insert, registry)
    _validate_nested_lists(op.with_step, registry)

    # Dispatch to specific handler
    handlers = {
        "remove": _apply_remove,
        "replace": _apply_replace,
        "after": _apply_after,
        "before": _apply_before,
        "at_start": _apply_at_start,
        "at_end": _apply_at_end,
    }

    handler = handlers.get(op.op_type)
    if handler:
        return handler(steps, op, normalized_target)
    return steps


def _validate_nested_lists(spec: Optional[StepSpec], registry) -> None:
    """Validate nested lists in step specifications"""
    if spec is not None and isinstance(spec, list):
        for item in spec:
            if isinstance(item, list):
                _validate_spec(item, registry)


def _apply_remove(steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
    """Apply remove operation"""
    return [s for s in steps if not _step_matches(s, target)]


def _apply_replace(steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
    """Apply replace operation"""
    new_steps = []
    for step in steps:
        if _step_matches(step, target):
            _insert_steps(new_steps, op.with_step)
        else:
            new_steps.append(step)
    return new_steps


def _apply_after(steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
    """Apply after operation"""
    new_steps = []
    for step in steps:
        new_steps.append(step)
        if _step_matches(step, target):
            _insert_steps(new_steps, op.insert)
    return new_steps


def _apply_before(steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
    """Apply before operation"""
    new_steps = []
    for step in steps:
        if _step_matches(step, target):
            _insert_steps(new_steps, op.insert)
        new_steps.append(step)
    return new_steps


def _apply_at_start(steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
    """Apply at_start operation"""
    new_steps = []
    _insert_steps(new_steps, op.insert)
    new_steps.extend(steps)
    return new_steps


def _apply_at_end(steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
    """Apply at_end operation"""
    new_steps = steps.copy()
    _insert_steps(new_steps, op.insert)
    return new_steps


def _insert_steps(target_list: List[StepSpec], steps: StepSpec) -> None:
    """Insert steps as sequential or branch based on content.

    Args:
        target_list: List to insert steps into
        steps: Steps to insert (string or list)
    """
    if isinstance(steps, list):
        # Handle lists that may contain mixed types (strings and sublists)
        for step in steps:
            if isinstance(step, list):
                # This is a branch point - append as-is
                target_list.append(step)
            else:
                # This is a regular step - append directly
                target_list.append(step)
    else:
        # Single step (string) or list to be treated as branching point
        target_list.append(steps)


def _step_matches(step: StepSpec, target: Optional[StepSpec]) -> bool:
    """Check if a step matches the target pattern"""
    if target is None:
        return False
    elif isinstance(step, str) and isinstance(target, str):
        return step == target
    elif isinstance(step, list) and isinstance(target, list):
        return set(step) == set(target)
    return False


def _validate_spec(spec: Union[str, List[Optional[str]], None], registry=None) -> Union[str, List[str]]:
    """Validate a step specification (string or list).

    Rules:
    - Strings are regular steps
    - Lists are branch points (can only contain strings or None/~)
    - No nested lists allowed within branch points
    - Branch points must have at least one non-skip option
    - Branch points can have at most one skip option
    """
    if isinstance(spec, str):
        return _validate_step(spec)
    elif isinstance(spec, list):
        # This is a branch point - validate each option
        validated = []
        skip_count = 0
        non_skip_count = 0

        for opt in spec:
            if isinstance(opt, str) or opt is None:
                validated_opt = _validate_step(opt)
                validated.append(validated_opt)
                if validated_opt == SKIP_NORMALIZED:
                    skip_count += 1
                else:
                    non_skip_count += 1
            elif isinstance(opt, list):
                raise ValueError(
                    f"Invalid branch point: contains nested list {opt}. "
                    "Branch points can only contain strings or skip (~). "
                    "To insert a branch point via operations, use double brackets: [[option1, option2]]"
                )
            else:
                raise ValueError(f"Invalid option in branch point: {opt}. Expected string or None, got {type(opt)}")

        # Validate branch point constraints
        if skip_count > 1:
            raise ValueError(
                f"Invalid branch point {spec}: contains {skip_count} skip options. "
                "Branch points can have at most one skip option."
            )
        if non_skip_count == 0:
            raise ValueError(
                f"Invalid branch point {spec}: contains only skip options. "
                "Branch points must have at least one non-skip step."
            )

        return validated
    elif spec is None:
        # Handle bare None values
        return _validate_step(None)
    else:
        raise ValueError(f"Invalid step specification: {spec}")


def _validate_step(step: Optional[str]) -> str:
    """Validate a step name against the registry, handle skip."""
    if step in SKIP_VALUES:
        return SKIP_NORMALIZED
    if not has_step(step):
        raise ValueError(f"Step '{step}' not found in registry")
    return step


def _extract_kernel_spec(spec) -> Tuple[str, Optional[List[str]]]:
    """Extract kernel name and optional backend names from spec."""
    if isinstance(spec, str):
        return spec, None
    elif isinstance(spec, dict) and len(spec) == 1:
        kernel_name, backend_specs = next(iter(spec.items()))
        backend_names = backend_specs if isinstance(backend_specs, list) else [backend_specs]
        return kernel_name, backend_names
    else:
        raise ValueError(f"Invalid kernel spec: {spec}")


def _parse_kernels(kernels_data: list) -> list:
    """Parse kernels section."""
    kernel_backends = []

    for spec in kernels_data:
        kernel_name, backend_names = _extract_kernel_spec(spec)

        # If no backends specified, get all available
        if not backend_names:
            backend_names = list_backends_by_kernel(kernel_name)

        # Skip if no backends available
        if not backend_names:
            continue

        # Resolve backend classes
        backend_classes = []
        for name in backend_names:
            backend_class = get_backend(name)
            if not backend_class:
                raise ValueError(f"Backend '{name}' not found in registry")
            backend_classes.append(backend_class)

        kernel_backends.append((kernel_name, backend_classes))

    return kernel_backends
