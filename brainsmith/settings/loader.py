# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration loading and management for Brainsmith."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, List
from unittest.mock import patch
from rich.console import Console
from pydantic import ValidationError

from .schema import SystemConfig

console = Console()


def load_config(
    project_file: Optional[Path] = None,
    user_file: Optional[Path] = None,
    **cli_overrides
) -> SystemConfig:
    """Load configuration with pydantic-settings priority resolution.

    Priority order (highest to lowest):
    1. CLI arguments (passed as kwargs)
    2. Environment variables (BSMITH_* prefix)
    3. Project config file (brainsmith_config.yaml)
    4. User config file (~/.brainsmith/config.yaml)
    5. Built-in defaults (from schema Field defaults)

    Args:
        project_file: Path to project config file (for non-standard locations)
        user_file: Path to user config file (defaults to ~/.brainsmith/config.yaml)
        **cli_overrides: CLI argument overrides

    Returns:
        Validated SystemConfig object
    """
    try:
        # Temporarily set config file paths via environment variables
        env_overrides = {}
        if project_file:
            env_overrides['_BRAINSMITH_PROJECT_FILE'] = str(project_file)
        if user_file:
            env_overrides['_BRAINSMITH_USER_FILE'] = str(user_file)

        with patch.dict(os.environ, env_overrides):
            return SystemConfig(**cli_overrides)

    except ValidationError as e:
        console.print("[bold red]Configuration validation failed:[/bold red]")
        for error in e.errors():
            field = " → ".join(str(x) for x in error["loc"])
            console.print(f"  [red]{field}: {error['msg']}[/red]")
        raise


@lru_cache(maxsize=1)
def get_config() -> SystemConfig:
    """Get singleton configuration instance (cached).

    This loads the configuration once and caches it for the session.
    Call reset_config() to clear the cache.
    """
    return load_config()


def reset_config() -> None:
    """Reset the cached configuration (mainly for testing)."""
    get_config.cache_clear()


def get_default_config() -> SystemConfig:
    """Get a configuration instance with only default values (no files or env vars)."""
    # Back up all Brainsmith env vars
    env_backup = {
        key: os.environ.pop(key)
        for key in list(os.environ.keys())
        if key.startswith('BSMITH_') or key.startswith('_BRAINSMITH_')
    }

    try:
        return SystemConfig()
    finally:
        # Restore backed up env vars
        os.environ.update(env_backup)
