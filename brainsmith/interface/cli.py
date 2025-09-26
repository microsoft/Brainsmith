"""Smith operational command-line interface.

Entry point for the smith operational CLI tool.
"""

# Standard library imports
import logging
import sys
from pathlib import Path
from typing import Optional

# Third-party imports
import click

# Local imports
from brainsmith.config import BrainsmithConfig
from .commands import dse as dse_command
from .commands import kernel as kernel_commands
from .context import ApplicationContext, get_context_from_parent
from .utils import console, setup_logging


@click.group(invoke_without_command=True)
@click.version_option(package_name='brainsmith', prog_name='smith')
@click.pass_context
def cli(ctx: click.Context):
    """Smith - Operational commands for Brainsmith.
    
    When invoked without a subcommand, runs design space exploration
    if model and blueprint arguments are provided.
    """
    # Get context from parent (if run via brainsmith) or use passed object
    app_ctx = ctx.obj if ctx.obj else ApplicationContext()
    
    # Ensure it's an ApplicationContext
    if not isinstance(app_ctx, ApplicationContext):
        app_ctx = ApplicationContext()
    
    ctx.obj = app_ctx
    
    # Set up logging based on context
    setup_logging(app_ctx.verbose)
    
    # Ensure configuration is loaded
    if app_ctx.config is None:
        app_ctx.load_configuration()
    
    # Export to environment for legacy compatibility
    config = app_ctx.get_effective_config()
    config.export_to_environment(verbose=app_ctx.verbose)
    
    # Handle default command behavior (DSE when model/blueprint provided)
    if ctx.invoked_subcommand is None:
        # Check if we have positional args for DSE
        if len(ctx.args) >= 2:
            # Run DSE as default command
            ctx.invoke(dse_command.dse, model=ctx.args[0], blueprint=ctx.args[1], 
                      output_dir=ctx.args[2] if len(ctx.args) > 2 else None)
        else:
            # Show help if no subcommand and insufficient args
            click.echo(ctx.get_help())


# Add subcommand groups
cli.add_command(dse_command.dse)
cli.add_command(kernel_commands.kernel)


def main() -> None:
    """Main entry point for the smith CLI."""
    try:
        # Handle the case where positional args are passed for default DSE
        # Click doesn't handle default commands with positional args well,
        # so we need to preprocess sys.argv
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            # Check if first arg is a subcommand
            subcommands = ['config', 'kernel', 'dse', 'setup']
            if sys.argv[1] not in subcommands:
                # Assume it's a model path for DSE, inject 'dse' command
                sys.argv.insert(1, 'dse')
        
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logging.exception("Unexpected error in CLI")
        sys.exit(1)


if __name__ == "__main__":
    main()