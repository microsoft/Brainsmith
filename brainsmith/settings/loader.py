# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration loading and management for Brainsmith."""

import os
from pathlib import Path
from typing import Optional, List
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
    3. Project settings file (brainsmith_settings.yaml)
    4. User settings file (~/.brainsmith/config.yaml)
    5. Built-in defaults (from schema Field defaults)

    Args:
        project_file: Path to project settings file (for non-standard locations)
        user_file: Path to user settings file (defaults to ~/.brainsmith/config.yaml)
        **cli_overrides: CLI argument overrides

    Returns:
        Validated SystemConfig object
    """
    try:
        # Set environment variables for config file paths
        env_vars_to_clean = []
        
        if project_file:
            os.environ['_BRAINSMITH_PROJECT_FILE'] = str(project_file)
            env_vars_to_clean.append('_BRAINSMITH_PROJECT_FILE')
            
        if user_file:
            os.environ['_BRAINSMITH_USER_FILE'] = str(user_file)
            env_vars_to_clean.append('_BRAINSMITH_USER_FILE')
        
        # Create config with CLI overrides
        # Pydantic-settings handles the rest automatically
        config = SystemConfig(**cli_overrides)
        
        # Clean up temp env vars
        for var in env_vars_to_clean:
            os.environ.pop(var, None)
        
        return config
        
    except ValidationError as e:
        console.print("[bold red]Configuration validation failed:[/bold red]")
        for error in e.errors():
            field = " → ".join(str(x) for x in error["loc"])
            console.print(f"  [red]{field}: {error['msg']}[/red]")
        raise


# Singleton instance
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get singleton configuration instance.
    
    This loads the configuration once and caches it for the session.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config():
    """Reset the singleton configuration (mainly for testing)."""
    global _config
    _config = None


def get_default_config() -> SystemConfig:
    """Get a configuration instance with only default values (no files or env vars)."""
    # Temporarily clear env vars that would affect config
    env_backup = {}
    for key in list(os.environ.keys()):
        if key.startswith('BSMITH_') or key.startswith('_BRAINSMITH_'):
            env_backup[key] = os.environ.pop(key)

    try:
        # Create config with defaults only
        config = SystemConfig()
        return config
    finally:
        # Restore env vars
        os.environ.update(env_backup)