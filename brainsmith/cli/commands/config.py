# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration management commands for brainsmith CLI."""

# Standard library imports
import logging
import os
from pathlib import Path
from typing import Any, Optional, Dict

# Third-party imports
import click
import yaml
from rich.table import Table
from rich.panel import Panel
from rich.console import Console as RichConsole

logger = logging.getLogger(__name__)

# Local imports
from brainsmith.settings import SystemConfig
from brainsmith._internal.io.yaml import dump_yaml
from ..context import ApplicationContext
from ..utils import console, error_exit, success
from ..exceptions import ConfigurationError, ValidationError
from ..formatters import ConfigFormatter


@click.group()
def config():
    """Manage Brainsmith configuration.

    \b
    Configuration can be managed by editing YAML files directly:
      Project: ./brainsmith_settings.yaml
      User:    ~/.brainsmith/config.yaml
    """
    pass


@config.command()
@click.option('--verbose', '-v', is_flag=True, help='Include source information and path validation')
@click.pass_obj
def show(ctx: ApplicationContext, verbose: bool) -> None:
    """Display current configuration.
    
    Shows the effective configuration after merging all sources:
    user defaults, project settings, environment variables, and CLI overrides.
    """
    try:
        logger.debug(f"Showing config with verbose={verbose}")
        config = ctx.get_effective_config()
        formatter = ConfigFormatter(console)
        
        table = formatter.format_table(config, verbose=verbose)
        console.print(table)
        if verbose:
            formatter.show_validation_warnings(config)
                
    except Exception as e:
        error_exit(f"Failed to load configuration: {e}")


@config.command()
@click.option('--shell', type=click.Choice(['bash', 'zsh', 'fish', 'powershell']),
              default='bash', help='Shell format for output')
@click.pass_obj
def export(ctx: ApplicationContext, shell: str) -> None:
    """Export configuration as shell environment script.
    
    \b
    Usage:
      eval $(brainsmith config export)
      eval $(brainsmith config export --shell fish)
    """
    try:
        activation_script = ctx.export_environment(shell)
        
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


@config.command()
@click.option('--user', is_flag=True, help='Create user-level config (~/.brainsmith/config.yaml)')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
@click.pass_obj
def init(ctx: ApplicationContext, user: bool, force: bool) -> None:
    """Initialize a new configuration file.
    
    By default creates project-level config (./brainsmith_settings.yaml).
    """
    # Determine output path
    if user:
        output = ctx.user_config_path
    else:
        # Default to project config
        output = Path("brainsmith_settings.yaml")
    
    if output.exists() and not force:
        error_exit(f"{output} already exists. Use --force to overwrite.")
    
    try:
        # Create parent directory if needed (especially for user config)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Load current config to get sensible defaults
        config = ctx.get_effective_config()
        
        # Always create simple config matching brainsmith_settings.yaml format
        config_dict = {
            "__comment__": "Brainsmith Project Settings",
            "build_dir": "${HOME}/.brainsmith/builds",
            "default_workers": 4,
            "xilinx_path": "/tools/Xilinx",
            "xilinx_version": "2024.2",
            "plugins_strict": True,
            "debug": False,
        }
        
        # Write formatted YAML matching the target style
        lines = []
        
        # Add header comment
        lines.append("# Brainsmith Project Settings")
        lines.append("")
        
        # Build directory comment and value
        lines.append("# Build directory for compilation artifacts")
        lines.append(f"build_dir: {config_dict['build_dir']}")
        lines.append("")

        # Worker settings
        lines.append("# Default number of workers for parallel operations")
        lines.append(f"default_workers: {config_dict['default_workers']}")
        lines.append("")
        
        # Xilinx tool configuration
        lines.append("# Xilinx tool configuration")
        lines.append(f"xilinx_path: {config_dict['xilinx_path']}")
        lines.append(f'xilinx_version: "{config_dict["xilinx_version"]}"')
        lines.append("")
        
        # Debug settings
        lines.append("# Debug settings")
        lines.append(f"plugins_strict: {str(config_dict['plugins_strict']).lower()}")
        lines.append(f"debug: {str(config_dict['debug']).lower()}")
        lines.append("")

        with open(output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        
        success(f"Created configuration file: {output}")
        console.print("\nYou can now edit this file to customize your settings.")
        if user:
            console.print("These settings will be used as defaults for all Brainsmith projects.")
        else:
            console.print("These settings will be used for this project.")
        
    except Exception as e:
        error_exit(f"Failed to create configuration: {e}")
