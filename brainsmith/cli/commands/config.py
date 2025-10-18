# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from pathlib import Path
from textwrap import dedent

import click
import yaml
from rich.table import Table
from rich.panel import Panel
from rich.console import Console as RichConsole

logger = logging.getLogger(__name__)

from brainsmith.settings import SystemConfig
from ..context import ApplicationContext
from ..utils import console, error_exit, success
from ..formatters import ConfigFormatter


def _generate_config_template(defaults: SystemConfig) -> str:
    return dedent(f"""\
        # Brainsmith Configuration

        # Build directory (relative paths resolve from project root)
        build_dir: {defaults.build_dir}

        # Parallel workers for builds
        default_workers: {defaults.default_workers}

        # Xilinx tools
        xilinx_path: {defaults.xilinx_path}
        xilinx_version: "{defaults.xilinx_version}"
    """)


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def config():
    """Manage Brainsmith configuration.

    \b
    Configuration can be managed by editing YAML files directly:
      Project: ./brainsmith_settings.yaml
      User:    ~/.brainsmith/settings.yaml
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
        table = formatter.format_table(config, finn=finn)
        console.print(table)
        formatter.show_validation_warnings(config)

    except Exception as e:
        error_exit(f"Failed to load configuration: {e}")


@config.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--shell', type=click.Choice(['bash', 'zsh', 'fish', 'powershell']),
              default='bash', help='Shell format for output')
@click.pass_obj
def export(app_ctx: ApplicationContext, shell: str) -> None:
    """Export configuration as shell environment script.

    \b
    Usage:
      eval $(brainsmith config export)
      eval $(brainsmith config export --shell fish)
    """
    try:
        activation_script = app_ctx.export_environment(shell)
        
        # Print to stdout for eval
        click.echo(activation_script)
        
        # Add helpful comment based on shell
        if shell in ['bash', 'zsh', 'sh']:
            click.echo("# Run: eval $(brainsmith config export)")
        elif shell == 'fish':
            click.echo("# Run: eval (brainsmith config export --shell fish)")
        elif shell == 'powershell':
            click.echo("# Run: brainsmith config export --shell powershell | Invoke-Expression")
            
    except ValueError as e:
        error_exit(f"Unsupported shell: {e}")
    except Exception as e:
        error_exit(f"Failed to export configuration: {e}")


@config.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
@click.option('--user', is_flag=True, help='Create user-level config (~/.brainsmith/settings.yaml)')
@click.pass_obj
def init(app_ctx: ApplicationContext, force: bool, user: bool) -> None:
    """Initialize a new configuration file with sensible defaults.

    By default creates project-level config (./brainsmith_settings.yaml).
    Use --user to create user-level config (~/.brainsmith/settings.yaml).
    """
    # Determine output path
    output = app_ctx.user_config_path if user else Path("brainsmith_settings.yaml")

    if output.exists() and not force:
        error_exit(f"{output} already exists. Use --force to overwrite.")

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
        error_exit(f"Failed to create configuration: {e}")
