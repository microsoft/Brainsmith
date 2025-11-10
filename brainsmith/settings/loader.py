# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration loading and management for Brainsmith."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any
from unittest.mock import patch
from rich.console import Console
from pydantic import ValidationError

from .schema import SystemConfig

console = Console()


def _is_path_field(key: str) -> bool:
    """Check if a field name suggests it's a path field.

    Uses naming convention: fields ending with _dir, _path, _file, _root, or _cache
    are treated as paths.
    """
    return key.endswith(('_dir', '_path', '_file', '_root', '_cache'))


def _resolve_cli_paths(cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve relative paths in CLI overrides to CWD.

    Simple rule: CLI paths resolve relative to where you ran the command.
    This follows standard shell semantics (like mkdir, tar, etc.).

    Uses naming convention to detect path fields (ending with _dir, _path, _file, _root).

    Args:
        cli_overrides: Dictionary of CLI argument overrides

    Returns:
        Dictionary with path overrides resolved to absolute paths
    """
    result = {}
    cwd = Path.cwd()

    for key, value in cli_overrides.items():
        if _is_path_field(key) and value is not None and isinstance(value, (str, Path)):
            path = Path(value)
            result[key] = str((cwd / path).resolve()) if not path.is_absolute() else str(value)
        else:
            result[key] = value

    return result


def load_config(
    project_file: Optional[Path] = None,
    **cli_overrides
) -> SystemConfig:
    """Load configuration with hierarchical priority.

    Priority order (highest to lowest):
    1. CLI arguments (passed as kwargs)
    2. Environment variables (BSMITH_* prefix)
    3. Project config file (brainsmith.yaml)
    4. Built-in defaults

    Special handling:
    - BSMITH_LOG_LEVEL env var overrides logging.level (shorthand for BSMITH_LOGGING__LEVEL)

    Path Resolution:
    - Absolute paths: Used as-is
    - Relative CLI paths: Resolve to current working directory
    - Relative paths (YAML/env/defaults): Resolve to project directory

    Args:
        project_file: Path to project config file (for non-standard locations)
        **cli_overrides: CLI argument overrides

    Returns:
        SystemConfig object
    """
    try:
        cli_overrides = _resolve_cli_paths(cli_overrides)

        # Handle BSMITH_LOG_LEVEL shorthand (if not overridden by CLI)
        if 'logging' not in cli_overrides and 'BSMITH_LOG_LEVEL' in os.environ:
            log_level = os.environ['BSMITH_LOG_LEVEL']
            cli_overrides['logging'] = {'level': log_level}

        if project_file:
            cli_overrides['_project_file'] = project_file

        return SystemConfig(**cli_overrides)

    except ValidationError as e:
        console.print("[bold red]Configuration validation failed:[/bold red]")
        for error in e.errors():
            field = " → ".join(str(x) for x in error["loc"])
            console.print(f"  [red]{field}: {error['msg']}[/red]")
        raise


@lru_cache(maxsize=1)
def get_config() -> SystemConfig:
    """Get cached configuration instance.

    Environment must be sourced first:
        source .brainsmith/env.sh  # or: direnv allow
    """
    config = load_config()
    return config


def reset_config() -> None:
    """Reset the cached configuration (mainly for testing)."""
    get_config.cache_clear()


def get_default_config() -> SystemConfig:
    """Get a configuration instance with only default values (no files or env vars)."""
    filtered_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith('BSMITH_')
    }

    with patch.dict(os.environ, filtered_env, clear=True):
        # Prevent loading config files
        from pathlib import Path
        return load_config(
            project_file=Path('/dev/null')
        )
