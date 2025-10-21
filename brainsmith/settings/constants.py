# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration file discovery constants.

This module defines the standard locations for Brainsmith configuration files.
These constants are used by both the Pydantic config loader and CLI tools
to ensure consistent configuration discovery.
"""

from pathlib import Path

# ============================================================================
# User Configuration
# ============================================================================

USER_CONFIG_DIR = Path.home() / ".brainsmith"
USER_CONFIG_FILE = "config.yaml"

def get_user_config_path() -> Path:
    """Get the default user configuration file path.

    Returns:
        Path to user config: ~/.brainsmith/config.yaml
    """
    return USER_CONFIG_DIR / USER_CONFIG_FILE


# ============================================================================
# Project Configuration
# ============================================================================

PROJECT_CONFIG_FILE = "brainsmith_config.yaml"
PROJECT_CONFIG_ALT_DIR = ".brainsmith"
PROJECT_CONFIG_ALT_FILE = "config.yaml"

def get_project_config_search_paths() -> list[Path]:
    """Get the list of paths to search for project configuration.

    Search order (first found wins):
    1. brainsmith_config.yaml (in current directory)
    2. .brainsmith/config.yaml (in current directory)

    Returns:
        List of paths to check for project config
    """
    return [
        Path.cwd() / PROJECT_CONFIG_FILE,
        Path.cwd() / PROJECT_CONFIG_ALT_DIR / PROJECT_CONFIG_ALT_FILE,
    ]


# ============================================================================
# Legacy Compatibility
# ============================================================================

# For backward compatibility with cli/constants.py usage
PROJECT_CONFIG_FILE_ALT = f"{PROJECT_CONFIG_ALT_DIR}/{PROJECT_CONFIG_ALT_FILE}"
