# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""YAML utilities for Brainsmith.

Provides basic YAML operations:
- load_yaml(): Load YAML with no processing
- expand_env_vars(): Recursively expand ${VAR} syntax
- deep_merge(): Deep merge two dictionaries
- dump_yaml(): Write YAML to file

All implementations are thread-safe and do not mutate os.environ.
"""

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load YAML with no processing (for env var expansion, see expand_env_vars()).

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge with overlay taking precedence. Recursively merges nested dicts.

    Returns new dict without mutating inputs.
    """
    result = base.copy()

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables (supports ${VAR} and $VAR).

    Leaves undefined variables unchanged (e.g., "${UNDEFINED_VAR}" stays as-is).
    """
    if isinstance(data, str):
        return os.path.expandvars(data)
    elif isinstance(data, dict):
        return {k: expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [expand_env_vars(item) for item in data]
    else:
        return data


def dump_yaml(
    data: dict[str, Any],
    file_path: str | Path,
    add_copyright: bool = False,
    **kwargs
) -> None:
    """Write YAML file with blueprint-compatible formatting.

    Produces clean YAML suitable for blueprints and config files:
    - 2-space indentation
    - Block style (not inline)
    - Preserved key ordering (Python 3.7+ dict order)
    - None → 'null', empty lists → '[]'
    - No document markers (---, ...)
    - 80-character line width

    Args:
        data: Dictionary to write
        file_path: Output file path
        add_copyright: If True, prepend Microsoft copyright header
        **kwargs: Additional arguments passed to yaml.safe_dump()

    Note: Comments are not supported (PyYAML limitation).
          For blueprints with inline comments, edit manually after generation.

    Example:
        >>> dump_yaml({
        ...     'name': 'Test Blueprint',
        ...     'clock_ns': 5.0,
        ...     'design_space': {
        ...         'kernels': [],
        ...         'steps': ['qonnx_to_finn', 'tidy_up']
        ...     }
        ... }, 'blueprint.yaml', add_copyright=True)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Blueprint-compatible defaults
    default_kwargs = {
        'default_flow_style': False,  # Lists as '- item' not '[item1, item2]'
        'sort_keys': False,            # Preserve dict insertion order
        'allow_unicode': True,         # Support unicode characters
        'width': 80,                   # Line wrap at 80 characters
        'indent': 2,                   # 2-space indentation (YAML standard)
        'explicit_start': False,       # No '---' document marker
        'explicit_end': False,         # No '...' document marker
    }
    default_kwargs.update(kwargs)

    with open(file_path, 'w') as f:
        # Add copyright header if requested
        if add_copyright:
            f.write("# Copyright (c) Microsoft Corporation.\n")
            f.write("# Licensed under the MIT License.\n\n")

        # Use safe_dump (safer than dump, prevents arbitrary code execution)
        yaml.safe_dump(data, f, **default_kwargs)
