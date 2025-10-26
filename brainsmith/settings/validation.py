# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration validation logic.

Centralized validation for settings and configuration files.
Separates validation logic from UI/CLI concerns.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import SystemConfig


def validate_config_file_creation(path: Path, force: bool) -> None:
    """Validate that config file creation is allowed.

    Args:
        path: Path where config file will be created
        force: Whether to allow overwriting existing file

    Raises:
        ValidationError: If file exists and force=False
    """
    from brainsmith.cli.exceptions import ValidationError
    from brainsmith.cli.messages import CONFIG_OVERWRITE_HINT

    if path.exists() and not force:
        raise ValidationError(
            f"{path} already exists. Use --force to overwrite.",
            details=[CONFIG_OVERWRITE_HINT]
        )


def get_config_warnings(config: SystemConfig) -> list[str]:
    """Collect configuration validation warnings.

    Args:
        config: System configuration to validate

    Returns:
        List of warning messages for display to user
    """
    warnings = []

    if config.deps_dir and not config.deps_dir.is_absolute():
        expected = config.bsmith_dir / config.deps_dir
        if config.deps_dir.absolute() != expected.absolute():
            warnings.append("Relative deps_dir may not resolve correctly")

    return warnings
