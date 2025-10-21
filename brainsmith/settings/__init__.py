# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith configuration module.

Provides type-safe configuration management with Pydantic Settings.
"""

from .schema import SystemConfig
from .loader import load_config, get_config, reset_config, get_default_config
from .env_export import EnvironmentExporter, EXTERNAL_ENV_MAPPINGS
from .constants import (
    USER_CONFIG_DIR,
    USER_CONFIG_FILE,
    PROJECT_CONFIG_FILE,
    PROJECT_CONFIG_FILE_ALT,
    PROJECT_CONFIG_ALT_DIR,
    PROJECT_CONFIG_ALT_FILE,
    get_user_config_path,
    get_project_config_search_paths,
)

__all__ = [
    # Core configuration classes
    "SystemConfig",

    # Configuration loading and management
    "load_config",
    "get_config",
    "reset_config",
    "get_default_config",

    # Helper classes (extracted from SystemConfig)
    "EnvironmentExporter",
    "EXTERNAL_ENV_MAPPINGS",

    # Configuration file discovery constants
    "USER_CONFIG_DIR",
    "USER_CONFIG_FILE",
    "PROJECT_CONFIG_FILE",
    "PROJECT_CONFIG_FILE_ALT",
    "PROJECT_CONFIG_ALT_DIR",
    "PROJECT_CONFIG_ALT_FILE",
    "get_user_config_path",
    "get_project_config_search_paths",
]
