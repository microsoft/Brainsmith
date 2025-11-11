# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint YAML loading and inheritance resolution.

Handles loading YAML blueprints and resolving inheritance chains.
"""

import os
from pathlib import Path
from string import Template
from typing import Any

import yaml

from brainsmith._internal.io.yaml import deep_merge, load_yaml


def _expand_env_vars_with_context(data: Any, context_vars: dict[str, str]) -> Any:
    """Expand environment variables with additional context variables.

    This implementation is thread-safe and does not mutate os.environ.
    Uses string.Template for variable expansion with ${VAR} syntax.

    Args:
        data: Data structure to process
        context_vars: Additional variables to make available during expansion

    Returns:
        Data with environment variables expanded

    Note:
        Only ${VAR} syntax is supported (not $VAR without braces).
        Context variables take precedence over environment variables.
    """
    if isinstance(data, str):
        combined = {**os.environ, **context_vars}
        template = Template(data)
        return template.safe_substitute(combined)

    elif isinstance(data, dict):
        return {k: _expand_env_vars_with_context(v, context_vars) for k, v in data.items()}

    elif isinstance(data, list):
        return [_expand_env_vars_with_context(item, context_vars) for item in data]

    else:
        return data


def _load_with_inheritance(
    file_path: Path, context_vars: dict[str, str] | None = None
) -> dict[str, Any]:
    """Load a YAML file with inheritance support via 'extends' field.

    Args:
        file_path: Path to the YAML file
        context_vars: Context variables for environment expansion

    Returns:
        Merged YAML data
    """
    with open(file_path) as f:
        data = yaml.safe_load(f) or {}

    if "extends" in data:
        parent_path = data.pop("extends")

        if context_vars:
            parent_path = _expand_env_vars_with_context(parent_path, context_vars)
        else:
            parent_path = os.path.expandvars(parent_path)

        if not Path(parent_path).is_absolute():
            parent_path = file_path.parent / parent_path
        else:
            parent_path = Path(parent_path)

        parent_context = context_vars.copy() if context_vars else {}
        parent_context["YAML_DIR"] = str(parent_path.parent.absolute())

        parent_data = _load_with_inheritance(parent_path, parent_context)

        return deep_merge(parent_data, data)

    return data


def load_blueprint_with_inheritance(
    blueprint_path: str,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Load blueprint YAML and resolve inheritance.

    Args:
        blueprint_path: Path to blueprint YAML file

    Returns:
        Tuple of (raw_data, merged_data, parent_path)
        - raw_data: Blueprint data without inheritance merging
        - merged_data: Blueprint data with inheritance merged
        - parent_path: Path to parent blueprint if extends is used, None otherwise
    """
    # Context vars for environment expansion
    context_vars = {
        "BLUEPRINT_DIR": str(Path(blueprint_path).parent.absolute()),
        "BSMITH_DIR": os.environ.get(
            "BSMITH_DIR", str(Path(__file__).parent.parent.parent.parent.absolute())
        ),
    }

    # Load raw data without inheritance to check extends field
    raw_data = load_yaml(blueprint_path)

    parent_path = None

    # Resolve parent path if this blueprint extends another
    if "extends" in raw_data:
        parent_path = raw_data["extends"]
        # Expand env vars in parent path
        parent_path = _expand_env_vars_with_context(parent_path, context_vars)

        # Resolve parent path relative to current file
        if not Path(parent_path).is_absolute():
            parent_path = str(Path(blueprint_path).parent / parent_path)

    # Load the full merged data with inheritance support (uses local _load_with_inheritance)
    merged_data = _load_with_inheritance(Path(blueprint_path), context_vars)

    # Expand environment variables in the merged data (e.g., ${BSMITH_DIR})
    merged_data = _expand_env_vars_with_context(merged_data, context_vars)

    return raw_data, merged_data, parent_path
