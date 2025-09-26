"""Brainsmith application-level CLI.

This is the main entry point for application configuration and setup tasks.
Operational commands are delegated to the smith CLI.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.logging import RichHandler

from brainsmith.config import load_config
from .context import ApplicationContext
from .utils import console, error_exit


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), 
              help='Configuration file (overrides default locations)')
@click.option('--build-dir', type=click.Path(path_type=Path),
              help='Override build directory')
@click.option('--xilinx-path', type=click.Path(path_type=Path),
              help='Override Xilinx installation path')
@click.option('--xilinx-version', help='Override Xilinx version (e.g., 2024.2)')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.version_option(package_name='brainsmith', prog_name='Brainsmith')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[Path], 
        build_dir: Optional[Path], xilinx_path: Optional[Path],
        xilinx_version: Optional[str], debug: bool):
    """Brainsmith - Application-level configuration and setup.
    
    Use 'smith' for operational commands like DSE and kernel generation.
    """
    # Initialize application context
    context = ApplicationContext(
        verbose=verbose,
        debug=debug,
        config_file=config
    )
    
    # Apply overrides
    if build_dir:
        context.overrides['build_dir'] = str(build_dir)
    if xilinx_path:
        context.overrides['xilinx_path'] = str(xilinx_path)
    if xilinx_version:
        context.overrides['xilinx_version'] = xilinx_version
    
    # Load configuration with overrides
    context.load_configuration()
    
    # Store in Click context
    ctx.obj = context
    
    # Set up logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


# Import and add subcommands
from .commands import config as config_commands
from .commands import setup as setup_commands
from .commands import env as env_commands

# Add command groups
cli.add_command(config_commands.config)
cli.add_command(setup_commands.setup) 
cli.add_command(env_commands.env)


# Add smith as a subcommand for "brainsmith smith" invocation
@cli.command()
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def smith(ctx: click.Context, args):
    """Run smith operational commands with inherited settings.
    
    This allows 'brainsmith --verbose smith model.onnx blueprint.yaml' syntax.
    """
    # Import here to avoid circular dependency
    from .smith_cli import run_smith
    
    # Get the application context
    app_ctx = ctx.obj
    
    # Run smith with the inherited context
    run_smith(args, app_ctx)


def main() -> None:
    """Main entry point for the brainsmith CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logging.exception("Unexpected error in brainsmith CLI")
        sys.exit(1)


if __name__ == "__main__":
    main()