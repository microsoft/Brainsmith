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
from ..exceptions import ConfigurationError, ValidationError

# Lazy import settings - deferred until command actually runs
if TYPE_CHECKING:
    from brainsmith.settings import SystemConfig


def _generate_config_template(defaults) -> str:
    return dedent(f"""\
        # Brainsmith Configuration

        # ============================================================================
        # Core Paths
        # ============================================================================

        # Project working directory (defaults to brainsmith root if not set)
        project_dir: {defaults.project_dir}
        build_dir: {defaults.build_dir}
        deps_dir: {defaults.deps_dir}

        # ============================================================================
        # Plugin System
        # ============================================================================

        # Default source when component has no prefix (e.g., 'LayerNorm' -> 'brainsmith:LayerNorm')
        default_source: {defaults.default_source}

        # Uncomment to add custom plugin sources
        # plugin_sources:
        #   custom: ~/my-custom-plugins     # Example: additional personal plugins

        # Plugin loading mode (strict=error on load failure, false=warn)
        plugins_strict: {str(defaults.plugins_strict).lower()}

        # ============================================================================
        # Xilinx Tools
        # ============================================================================

        # Xilinx root installation path
        xilinx_path: {defaults.xilinx_path}
        xilinx_version: "{defaults.xilinx_version}"

        # Vendor platform repository paths (colon-separated paths to Xilinx/Intel FPGA platform files)
        vendor_platform_paths: {defaults.vendor_platform_paths}

        # Auto-detected tool paths (uncomment to override):
        # vivado_path: /tools/Xilinx/Vivado/{defaults.xilinx_version}
        # vitis_path: /tools/Xilinx/Vitis/{defaults.xilinx_version}
        # vitis_hls_path: /tools/Xilinx/Vitis_HLS/{defaults.xilinx_version}

        # Vivado IP cache (auto-computed from build_dir if not set)
        # vivado_ip_cache: {{build_dir}}/vivado_ip_cache

        # ============================================================================
        # Build Settings
        # ============================================================================

        # Default number of workers for parallel operations
        default_workers: {defaults.default_workers}

        # ============================================================================
        # Development Tools
        # ============================================================================

        # Port for Netron neural network visualization
        netron_port: {defaults.netron_port}

        # ============================================================================
        # FINN Configuration (Advanced)
        # ============================================================================
        # FINN paths auto-detect from deps_dir, build_dir. Uncomment to override:
        # finn:
        #   finn_root: {{deps_dir}}/finn
        #   finn_build_dir: {{build_dir}}
        #   finn_deps_dir: {{deps_dir}}
    """)


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def config():
    """\b
    Configuration can be managed by editing YAML files directly:
      Project: ./brainsmith_config.yaml
      User:    ~/.brainsmith/config.yaml
    """
    pass


@config.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--finn', is_flag=True, help='Include FINN-specific configuration settings')
@click.pass_obj
def show(app_ctx: ApplicationContext, finn: bool) -> None:
    """Display current configuration with source information."""
    try:
        logger.debug(f"Showing config with finn={finn}")
        config = app_ctx.get_effective_config()
        formatter = ConfigFormatter(console)

        # Always show detailed view with sources
        table = formatter.format_table(config, include_finn=finn)
        console.print(table)
        formatter.show_validation_warnings(config)

    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e


@config.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
@click.option('--user', is_flag=True, help='Create user-level config (~/.brainsmith/config.yaml)')
@click.pass_obj
def init(app_ctx: ApplicationContext, force: bool, user: bool) -> None:
    """By default creates project-level config (./brainsmith_config.yaml).
    Use --user to create user-level config (~/.brainsmith/config.yaml).
    """
    # Determine output path
    output = app_ctx.user_config_path if user else Path("brainsmith_config.yaml")

    if output.exists() and not force:
        raise ValidationError(
            f"{output} already exists. Use --force to overwrite.",
            details=["Run with --force flag to overwrite existing configuration"]
        )

    try:
        # Create parent directory if needed (especially for user config)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Get actual defaults WITHOUT loading from existing config files
        # Temporarily override env vars to prevent loading existing configs
        original_user_file = os.environ.pop('_BRAINSMITH_USER_FILE', None)
        original_project_file = os.environ.pop('_BRAINSMITH_PROJECT_FILE', None)

        try:
            # Set dummy values to prevent YamlSettingsSource from loading real files
            os.environ['_BRAINSMITH_USER_FILE'] = '/dev/null'
            os.environ['_BRAINSMITH_PROJECT_FILE'] = '/dev/null'
            from brainsmith.settings import SystemConfig  # Lazy import
            defaults = SystemConfig()
        finally:
            # Restore original env vars
            os.environ.pop('_BRAINSMITH_USER_FILE', None)
            os.environ.pop('_BRAINSMITH_PROJECT_FILE', None)
            if original_user_file:
                os.environ['_BRAINSMITH_USER_FILE'] = original_user_file
            if original_project_file:
                os.environ['_BRAINSMITH_PROJECT_FILE'] = original_project_file

        yaml_content = _generate_config_template(defaults)

        with open(output, 'w') as f:
            f.write(yaml_content)

        success(f"Created configuration file: {output}")
        console.print("\n[dim]Edit the file to customize settings for your environment.[/dim]")

        if user:
            console.print("These settings apply to all Brainsmith projects.")
        else:
            console.print("These settings apply to this project only.")

    except Exception as e:
        raise ConfigurationError(f"Failed to create configuration: {e}") from e
