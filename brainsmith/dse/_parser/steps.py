# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint step parsing and operations.

Handles parsing of step specifications, including variations (branch points)
and step manipulation operations (before, after, replace, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Literal

from brainsmith.dse._constants import SKIP_INDICATOR
from brainsmith.registry import has_step

# Skip values during parsing (normalized to SKIP_INDICATOR after parsing)
SKIP_VALUES = frozenset([None, "~", ""])

# Type definitions
StepSpec = Union[str, List[Optional[str]]]


@dataclass
class StepOperation:
    """Represents a step manipulation operation"""
    op_type: Literal["after", "before", "replace", "remove", "at_start", "at_end"]
    target: Optional[StepSpec] = None
    insert: Optional[StepSpec] = None
    with_step: Optional[StepSpec] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional[StepOperation]:
        """Parse operation from YAML dict"""
        if "after" in data:
            return cls(op_type="after", target=data["after"], insert=data.get("insert"))
        elif "before" in data:
            return cls(op_type="before", target=data["before"], insert=data.get("insert"))
        elif "replace" in data:
            return cls(op_type="replace", target=data["replace"], with_step=data.get("with"))
        elif "remove" in data:
            return cls(op_type="remove", target=data["remove"])
        elif "at_start" in data:
            return cls(op_type="at_start", insert=data["at_start"]["insert"])
        elif "at_end" in data:
            return cls(op_type="at_end", insert=data["at_end"]["insert"])
        return None


def parse_steps(
    steps_data: List[Any],
    parent_steps: Optional[List[Union[str, List[Optional[str]]]]] = None
) -> List[Union[str, List[Optional[str]]]]:
    """Parse steps from design_space, preserving variations and supporting operations.

    Args:
        steps_data: Raw steps data from blueprint YAML
        parent_steps: Resolved steps from parent blueprint (if any)

    Returns:
        List of validated step specifications (strings and/or lists)
    """
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
        result = [_validate_spec(spec) for spec in direct_steps]
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
    return [_validate_spec(spec) for spec in result]


def _apply_step_operation(steps: List[StepSpec], op: StepOperation) -> List[StepSpec]:
    """Apply a single operation to the step list"""

    # Normalize the operation target to match already-normalized steps
    normalized_target = None
    if op.target is not None:
        normalized_target = _validate_spec(op.target)

    # Validate nested lists in operation specs
    _validate_nested_lists(op.insert)
    _validate_nested_lists(op.with_step)

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


def _validate_nested_lists(spec: Optional[StepSpec]) -> None:
    """Validate nested lists in step specifications"""
    if spec is not None and isinstance(spec, list):
        for item in spec:
            if isinstance(item, list):
                _validate_spec(item)


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


def _validate_spec(spec: Union[str, List[Optional[str]], None]) -> Union[str, List[str]]:
    """Validate a step specification (string or list).

    Dispatcher that routes to specialized validation based on type.

    Args:
        spec: Step specification to validate

    Returns:
        Validated and normalized step specification

    Raises:
        ValueError: If specification is invalid
    """
    if isinstance(spec, str):
        return _validate_step(spec)
    elif isinstance(spec, list):
        return _validate_branch_point(spec)
    elif spec is None:
        return _validate_step(None)
    else:
        raise ValueError(f"Invalid step specification: {spec}")


def _validate_branch_point(options: List[Optional[str]]) -> List[str]:
    """Validate branch point specification.

    Branch points are lists of step options. Constraints:
    - No nested lists allowed
    - Options must be strings or None/~/""
    - Maximum 1 skip option (~)
    - Minimum 1 non-skip option

    Args:
        options: List of step options (strings or skip indicators)

    Returns:
        Normalized list with skip indicators converted to SKIP_INDICATOR

    Raises:
        ValueError: If branch point violates constraints
    """
    # Validate types first (fail fast)
    _validate_branch_types(options)

    # Normalize step names and skip indicators
    normalized = [_validate_step(opt) for opt in options]

    # Validate business constraints
    _validate_branch_constraints(normalized)

    return normalized


def _validate_branch_types(options: List[Optional[str]]) -> None:
    """Validate branch point contains only valid types.

    Args:
        options: List of options to type-check

    Raises:
        ValueError: If nested list found or non-string type found
    """
    for opt in options:
        if isinstance(opt, list):
            raise ValueError(
                f"Branch point contains nested list {opt}. "
                "Use double brackets [[opt1, opt2]] for branch insertion."
            )
        if not isinstance(opt, str):
            raise ValueError(
                f"Branch option must be string, got {type(opt).__name__}"
            )


def _validate_branch_constraints(normalized: List[str]) -> None:
    """Validate branch point skip/non-skip ratio.

    Args:
        normalized: List of normalized step names

    Raises:
        ValueError: If too many skips or no non-skip options
    """
    skip_count = sum(1 for opt in normalized if opt == SKIP_INDICATOR)

    if skip_count > 1:
        raise ValueError(f"Branch point has {skip_count} skip options (max 1)")
    if len(normalized) == skip_count:
        raise ValueError("Branch point must have at least one non-skip step")


def _validate_step(step: Optional[str]) -> str:
    """Validate a step name.

    Normalizes skip indicators (None, "", ~) to canonical form (~).

    Args:
        step: Step name or skip indicator

    Returns:
        Validated step name or SKIP_INDICATOR

    Raises:
        ValueError: If step not found
    """
    if step in SKIP_VALUES:
        return SKIP_INDICATOR
    if not has_step(step):
        raise ValueError(f"Step '{step}' not found. Use 'brainsmith list steps' to see available steps.")
    return step

