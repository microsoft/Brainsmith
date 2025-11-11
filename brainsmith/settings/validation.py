# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration validation logic.

Centralized validation for settings and configuration files.
Includes environment validation to ensure brainsmith environment is sourced.
Separates validation logic from UI/CLI concerns.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import SystemConfig

logger = logging.getLogger(__name__)


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
            f"{path} already exists. Use --force to overwrite.", details=[CONFIG_OVERWRITE_HINT]
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


# Environment Validation
# ----------------------
# Functions to ensure brainsmith environment is properly sourced before
# running commands or tests. Environment must be activated via:
#     source .brainsmith/env.sh
# or:
#     direnv allow


def ensure_environment_sourced(marker_var: str = "BSMITH_DIR") -> None:
    """Verify brainsmith environment is sourced, exit with helpful error if not.

    Checks for BSMITH_DIR environment variable as marker that env.sh or direnv
    has been sourced. This ensures consistent environment across Python runtime
    and all subprocesses (including FINN's shell invocations).

    Called by CLI entry points and pytest conftest for early validation.

    Args:
        marker_var: Environment variable to check (default: BSMITH_DIR)

    Raises:
        SystemExit: If environment not detected (exit code 1)

    Example:
        >>> from brainsmith.settings.validation import ensure_environment_sourced
        >>> ensure_environment_sourced()  # Exits if env not sourced
    """
    if marker_var not in os.environ:
        print("❌ Brainsmith environment not detected", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Before running brainsmith commands or tests, activate the environment:",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("  Option 1 (Recommended): direnv", file=sys.stderr)
        print("    brainsmith project allow-direnv", file=sys.stderr)
        print("    cd .  # Reload directory to activate", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Option 2: Manual activation (per-shell)", file=sys.stderr)
        print("    source .brainsmith/env.sh", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"Validation failed: {marker_var} not found in environment", file=sys.stderr)
        sys.exit(1)


def warn_if_environment_not_sourced(marker_var: str = "BSMITH_DIR") -> bool:
    """Warn if environment not sourced, but continue execution.

    Provides soft validation for non-critical paths where we want to alert
    the user but allow natural error handling to proceed.

    Args:
        marker_var: Environment variable to check (default: BSMITH_DIR)

    Returns:
        True if environment detected, False otherwise

    Example:
        >>> from brainsmith.settings.validation import warn_if_environment_not_sourced
        >>> if not warn_if_environment_not_sourced():
        ...     logger.info("Some features may not work correctly")
    """
    if marker_var not in os.environ:
        logger.warning(
            "Brainsmith environment not detected (%s not set). "
            "Some features may not work correctly. "
            "Run: source .brainsmith/env.sh",
            marker_var,
        )
        return False
    return True


def get_environment_info() -> dict[str, str | None]:
    """Get diagnostic information about current environment state.

    Useful for debugging environment-related issues.

    Returns:
        Dict with environment status information

    Example:
        >>> info = get_environment_info()
        >>> print(f"Environment sourced: {info['sourced']}")
        >>> print(f"Project directory: {info['project_dir']}")
    """
    return {
        "sourced": os.environ.get("BSMITH_DIR") is not None,
        "project_dir": os.environ.get("BSMITH_PROJECT_DIR"),
        "finn_root": os.environ.get("FINN_ROOT"),
        "vivado_path": os.environ.get("VIVADO_PATH"),
        "vitis_path": os.environ.get("VITIS_PATH"),
        "hls_path": os.environ.get("HLS_PATH"),
    }
