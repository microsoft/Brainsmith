"""Environment management commands for brainsmith CLI."""

import sys
from typing import Dict, Any

import click
from rich.table import Table
from rich.syntax import Syntax

from ..context import ApplicationContext
from ..utils import console, info


@click.group()
def env():
    """Manage Brainsmith environment settings."""
    pass


@env.command()
@click.pass_obj
def show(ctx: ApplicationContext) -> None:
    """Show effective environment configuration.
    
    Displays all environment variables that would be set for tools.
    """
    config = ctx.get_effective_config()
    
    # Get all environment variables
    env_vars = config.to_external_env_dict()
    
    # Create table
    table = Table(title="Brainsmith Environment", show_header=True)
    table.add_column("Variable", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Category", style="yellow")
    
    # Categorize variables
    categories = {
        "XILINX": ["XILINX_", "VIVADO_", "VITIS_", "HLS_"],
        "FINN": ["FINN_", "NUM_DEFAULT_"],
        "Platform": ["PLATFORM_", "OHMYXILINX"],
        "Other": []
    }
    
    # Sort and categorize
    for var, value in sorted(env_vars.items()):
        category = "Other"
        for cat_name, prefixes in categories.items():
            if any(var.startswith(prefix) for prefix in prefixes):
                category = cat_name
                break
        
        table.add_row(var, str(value), category)
    
    console.print(table)
    
    # Also show internal config if verbose
    if ctx.verbose:
        console.print("\n[bold]Internal Configuration:[/bold]")
        internal_vars = {
            "build_dir": str(config.build_dir),
            "deps_dir": str(config.deps_dir),
            "bsmith_dir": str(config.bsmith_dir),
            "verbose": str(config.verbose),
            "debug": str(config.debug)
        }
        
        for key, value in internal_vars.items():
            console.print(f"  {key}: {value}")


@env.command()
@click.option('--shell', type=click.Choice(['bash', 'zsh', 'fish', 'powershell']),
              default='bash', help='Shell format for output')
@click.pass_obj
def activate(ctx: ApplicationContext, shell: str) -> None:
    """Output shell commands to activate Brainsmith environment.
    
    Usage:
        eval $(brainsmith env activate)
        eval $(brainsmith env activate --shell fish)
    """
    try:
        activation_script = ctx.export_environment(shell)
        
        # Print to stdout for eval
        click.echo(activation_script)
        
        # Add helpful comment based on shell
        if shell in ['bash', 'zsh', 'sh']:
            click.echo("# Run: eval $(brainsmith env activate)")
        elif shell == 'fish':
            click.echo("# Run: eval (brainsmith env activate --shell fish)")
        elif shell == 'powershell':
            click.echo("# Run: brainsmith env activate --shell powershell | Invoke-Expression")
            
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}", file=sys.stderr)
        sys.exit(1)