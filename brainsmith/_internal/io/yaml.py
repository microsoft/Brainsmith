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
from typing import Any, Dict, List, Optional, Union

import yaml


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load a YAML file with no additional processing.

    Basic YAML loading - returns parsed data as-is. Use this when you just
    need to read YAML without env expansion, inheritance, or path resolution.

    Args:
        file_path: Path to the YAML file

    Returns:
        Parsed YAML data (empty dict if file is empty)

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with overlay values taking precedence.

    Recursively merges nested dicts. Overlay values replace base values,
    except when both are dicts - then they are merged recursively.

    Args:
        base: Base dictionary
        overlay: Dictionary to merge on top

    Returns:
        Merged dictionary (new dict, does not mutate inputs)
    """
    result = base.copy()

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def expand_env_vars(data: Any) -> Any:
    """
    Recursively expand environment variables in a data structure.

    Supports both ${VAR} and $VAR syntax. Handles nested dicts and lists.

    Args:
        data: Data structure to process

    Returns:
        Data with environment variables expanded
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
    data: Dict[str, Any],
    file_path: Union[str, Path],
    **kwargs
) -> None:
    """
    Write data to a YAML file.

    Args:
        data: Data to write
        file_path: Path to write to
        **kwargs: Additional arguments passed to yaml.dump()
    """
    file_path = Path(file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    default_kwargs = {
        'default_flow_style': False,
        'sort_keys': False,
        'width': 80,
        'indent': 2
    }
    default_kwargs.update(kwargs)

    with open(file_path, 'w') as f:
        yaml.dump(data, f, **default_kwargs)
