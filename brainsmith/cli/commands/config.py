# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations  # PEP 563: Postponed evaluation of annotations

import logging
import os
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import click
import yaml
from rich.table import Table
from rich.panel import Panel
from rich.console import Console as RichConsole

logger = logging.getLogger(__name__)

from ..context import ApplicationContext
from ..utils import console, success
from ..formatters import ConfigFormatter
from ..exceptions import ConfigurationError
from ..messages import (
    CONFIG_EDIT_HINT,
    CONFIG_CREATED_PROJECT,
    CONFIG_CREATED_USER,
)

# Lazy import settings - deferred until command actually runs
if TYPE_CHECKING:
    from brainsmith.settings import SystemConfig


def _generate_config_template(defaults) -> str:
    return dedent("""\
        # Brainsmith Configuration
        # Relative paths resolve to the directory containing the .brainsmith folder

        # Build output directory
        build_dir: build

        # Xilinx tools (adjust paths for your installation)
        xilinx_path: /tools/Xilinx
        xilinx_version: "2024.2"

        # Advanced options
        default_workers: 4
        netron_port: 8080
        components_strict: true
        vendor_platform_paths: /opt/xilinx/platforms
    """)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def config():
    """\b
    Configuration can be managed by editing YAML files directly:
      Project: ./.brainsmith/config.yaml
      User:    ~/.brainsmith/config.yaml
    """
    pass


@config.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--finn", is_flag=True, help="Include FINN-specific configuration settings")
@click.pass_obj
def show(ctx: ApplicationContext, finn: bool) -> None:
    """Display current configuration with source information."""
    try:
        logger.debug(f"Showing config with finn={finn}")
        config = ctx.get_effective_config()
        formatter = ConfigFormatter(console)

        # Always show detailed view with sources
        table = formatter.format_table(config, include_finn=finn)
        console.print(table)
        formatter.show_validation_warnings(config)

    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e


@config.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file")
@click.option("--user", is_flag=True, help="Create user-level config (~/.brainsmith/config.yaml)")
@click.pass_obj
def init(ctx: ApplicationContext, force: bool, user: bool) -> None:
    """By default creates project-level config (./.brainsmith/config.yaml).
    Use --user to create user-level config (~/.brainsmith/config.yaml).
    """
    # Determine output path
    output = ctx.user_config_path if user else Path(".brainsmith/config.yaml")

    # Validate file creation
    from brainsmith.settings.validation import validate_config_file_creation
    validate_config_file_creation(output, force)

    try:
        # Create parent directory if needed (especially for user config)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Get actual defaults WITHOUT loading from existing config files
        from brainsmith.settings import get_default_config  # Lazy import
        defaults = get_default_config()

        yaml_content = _generate_config_template(defaults)

        with open(output, "w") as f:
            f.write(yaml_content)

        success(f"Created configuration file: {output}")
        console.print(f"[dim]{CONFIG_EDIT_HINT}[/dim]")

        if user:
            console.print(CONFIG_CREATED_USER)
        else:
            console.print(CONFIG_CREATED_PROJECT)

    except Exception as e:
        raise ConfigurationError(f"Failed to create configuration: {e}") from e
