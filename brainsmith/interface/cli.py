"""Brainsmith command-line interface.

Main entry point for the smith CLI tool.
"""

# Standard library imports
import logging
import sys
from pathlib import Path
from typing import Optional

# Third-party imports
import click
from rich.logging import RichHandler

# Local imports
from brainsmith.config import load_config, BrainsmithConfig, export_to_environment
from brainsmith.core.dse_api import explore_design_space
from .commands import config as config_commands
from .commands import kernel as kernel_commands
from .commands import setup as setup_commands
from .utils import console, error_exit, success


class SmithContext:
    """Context object passed between CLI commands."""
    
    def __init__(self):
        self.config: Optional[BrainsmithConfig] = None
        self.verbose: bool = False
        self.config_file: Optional[Path] = None


pass_context = click.make_pass_decorator(SmithContext, ensure=True)


def setup_logging(verbose: bool = False):
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@click.group(invoke_without_command=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.version_option(package_name='brainsmith', prog_name='Brainsmith')
@click.pass_context
def cli(ctx, verbose: bool, config: Optional[str]):
    """Brainsmith - FPGA Accelerator Compiler for AI.
    
    When invoked without a subcommand, runs design space exploration
    if model and blueprint arguments are provided.
    """
    # Initialize context
    smith_ctx = SmithContext()
    smith_ctx.verbose = verbose
    smith_ctx.config_file = Path(config) if config else None
    ctx.obj = smith_ctx
    
    # Set up logging
    setup_logging(verbose)
    
    # Load configuration
    try:
        smith_ctx.config = load_config(project_file=smith_ctx.config_file)
        # Export to environment for legacy compatibility
        export_to_environment(smith_ctx.config, verbose=verbose)
    except Exception as e:
        error_exit(f"Failed to load configuration: {e}")
    
    # Handle default command behavior (DSE when model/blueprint provided)
    if ctx.invoked_subcommand is None:
        # Check if we have positional args for DSE
        if len(ctx.args) >= 2:
            # Run DSE as default command
            ctx.invoke(dse, model=ctx.args[0], blueprint=ctx.args[1], 
                      output_dir=ctx.args[2] if len(ctx.args) > 2 else None)
        else:
            # Show help if no subcommand and insufficient args
            click.echo(ctx.get_help())


@cli.command()
@click.argument('model', type=click.Path(exists=True, path_type=Path))
@click.argument('blueprint', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory (defaults to build dir with timestamp)')
@pass_context
def dse(ctx: SmithContext, model: Path, blueprint: Path, output_dir: Optional[Path]) -> None:
    """Run design space exploration for neural network acceleration.
    
    MODEL: Path to ONNX model file
    BLUEPRINT: Path to Blueprint YAML file defining exploration strategy
    """
    console.print(f"[bold blue]Brainsmith DSE[/bold blue] - Design Space Exploration")
    console.print(f"Model: {model}")
    console.print(f"Blueprint: {blueprint}")
    
    if output_dir:
        console.print(f"Output: {output_dir}")
    
    try:
        # Run design space exploration
        result = explore_design_space(
            model_path=str(model),
            blueprint_path=str(blueprint),
            output_dir=str(output_dir) if output_dir else None
        )
        
        success("Design space exploration completed successfully!")
        
        # Display summary if available
        if hasattr(result, 'summary'):
            console.print(f"\nSummary: {result.summary}")
            
    except FileNotFoundError as e:
        error_exit(str(e))
    except Exception as e:
        if ctx.verbose:
            console.print_exception()
        error_exit(f"Failed during exploration: {e}")


# Add subcommand groups
cli.add_command(config_commands.config)
cli.add_command(kernel_commands.kernel)
cli.add_command(setup_commands.setup)


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