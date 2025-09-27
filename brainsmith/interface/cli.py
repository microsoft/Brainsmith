"""Unified CLI implementation for Brainsmith.

This module provides a factory function to create CLI instances for both
'brainsmith' and 'smith' entry points, sharing common implementation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.logging import RichHandler

from brainsmith.config import BrainsmithConfig
from .commands import OPERATIONAL_COMMANDS, ADMIN_COMMANDS
from .context import ApplicationContext
from .utils import console, setup_logging

logger = logging.getLogger(__name__)


def create_cli(name: str, include_admin: bool = True) -> click.Group:
    """Factory to create CLI with appropriate commands.
    
    Args:
        name: CLI name ('brainsmith' or 'smith')
        include_admin: Whether to include administrative commands
        
    Returns:
        Configured Click group
    """
    
    @click.group(invoke_without_command=True)
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
    @click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), 
                  help='Configuration file (overrides default locations)')
    @click.option('--build-dir', type=click.Path(path_type=Path),
                  help='Override build directory')
    @click.option('--xilinx-path', type=click.Path(path_type=Path),
                  help='Override Xilinx installation path')
    @click.option('--xilinx-version', help='Override Xilinx version (e.g., 2024.2)')
    @click.option('--debug', is_flag=True, help='Enable debug mode')
    @click.version_option(package_name='brainsmith', prog_name=name)
    @click.pass_context
    def cli(ctx: click.Context, verbose: bool, config: Optional[Path], 
            build_dir: Optional[Path], xilinx_path: Optional[Path],
            xilinx_version: Optional[str], debug: bool):
        """Brainsmith - Neural network hardware acceleration toolkit."""
        
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
        
        # Load configuration
        context.load_configuration()
        
        # Store in Click context
        ctx.obj = context
        
        # Set up logging
        setup_logging(verbose)
        logger.debug(f"{name} CLI initialized with verbose={verbose}")
        
        # Export to environment for legacy compatibility
        effective_config = context.get_effective_config()
        effective_config.export_to_environment(verbose=verbose)
        
        # Handle default behavior when no subcommand is provided
        if ctx.invoked_subcommand is None:
            if name == 'smith':
                # Smith: run DSE with positional args if provided
                if len(ctx.args) >= 2:
                    from .commands.dse import dse
                    ctx.invoke(dse, model=ctx.args[0], blueprint=ctx.args[1], 
                              output_dir=ctx.args[2] if len(ctx.args) > 2 else None)
                else:
                    click.echo(ctx.get_help())
            else:
                # Brainsmith: always show help when no subcommand
                click.echo(ctx.get_help())
    
    # Update docstring based on CLI type
    if name == 'smith':
        cli.help = """Smith - Operational commands for Brainsmith.
        
When invoked without a subcommand, runs design space exploration
if model and blueprint arguments are provided."""
    else:
        cli.help = """Brainsmith - Neural network hardware acceleration toolkit.
        
Use 'smith' for streamlined access to operational commands (dse, kernel)."""
    
    # Add operational commands (both CLIs)
    for cmd_name, cmd in OPERATIONAL_COMMANDS.items():
        cli.add_command(cmd, name=cmd_name)
    
    # Add admin commands (brainsmith only)
    if include_admin:
        for cmd_name, cmd in ADMIN_COMMANDS.items():
            cli.add_command(cmd, name=cmd_name)
        
        # Add smith as a subcommand for "brainsmith smith" invocation
        @cli.command()
        @click.argument('args', nargs=-1, type=click.UNPROCESSED)
        @click.pass_context
        def smith(ctx: click.Context, args):
            """Run smith operational commands with inherited settings.
            
            This allows 'brainsmith --verbose smith dse model.onnx blueprint.yaml' syntax.
            """
            # Get the application context
            app_ctx = ctx.obj
            
            # Create smith CLI and run with inherited context
            smith_cli = create_cli('smith', include_admin=False)
            
            # Convert args to sys.argv format
            original_argv = sys.argv
            try:
                sys.argv = ['smith'] + list(args)
                
                # Pass context through Click's obj
                smith_cli(obj=app_ctx, standalone_mode=False)
                
            except SystemExit as e:
                # Re-raise to preserve exit code
                raise
            finally:
                # Restore original argv
                sys.argv = original_argv
    
    return cli


def brainsmith_main() -> None:
    """Main entry point for the brainsmith CLI."""
    try:
        cli = create_cli('brainsmith', include_admin=True)
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logging.exception("Unexpected error in brainsmith CLI")
        sys.exit(1)


def smith_main() -> None:
    """Main entry point for the smith CLI."""
    try:
        # Handle the case where positional args are passed for default DSE
        # Click doesn't handle default commands with positional args well,
        # so we need to preprocess sys.argv
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            # Check if first arg is a subcommand
            subcommands = ['dse', 'kernel']
            if sys.argv[1] not in subcommands:
                # Assume it's a model path for DSE, inject 'dse' command
                sys.argv.insert(1, 'dse')
        
        cli = create_cli('smith', include_admin=False)
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logging.exception("Unexpected error in smith CLI")
        sys.exit(1)


# For backwards compatibility
main = smith_main


if __name__ == "__main__":
    # Determine which CLI to run based on invocation
    import os
    prog_name = os.path.basename(sys.argv[0])
    
    if 'brainsmith' in prog_name:
        brainsmith_main()
    else:
        smith_main()