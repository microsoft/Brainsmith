# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Design Space Exploration command for smith CLI."""

import logging
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)

from brainsmith.core.dse_api import explore_design_space
from ..context import ApplicationContext, get_context_from_parent
from ..utils import console, error_exit, success


@click.command()
@click.argument('model', type=click.Path(exists=True, path_type=Path))
@click.argument('blueprint', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory (defaults to build dir with timestamp)')
@click.pass_context
def dse(ctx: click.Context, model: Path, blueprint: Path, output_dir: Optional[Path]) -> None:
    """Run design space exploration for neural network acceleration.

    \b
    MODEL: Path to ONNX model file
    BLUEPRINT: Path to Blueprint YAML file defining exploration strategy
    """
    # Get context from parent or create default
    app_ctx = get_context_from_parent(ctx) or ApplicationContext()
    config = app_ctx.get_effective_config()
    
    console.print(f"[bold blue]Brainsmith DSE[/bold blue] - Design Space Exploration")
    console.print(f"Model: {model}")
    console.print(f"Blueprint: {blueprint}")
    
    if output_dir:
        console.print(f"Output: {output_dir}")
    
    logger.info(f"Starting DSE with model={model}, blueprint={blueprint}, output_dir={output_dir}")
    
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
        error_exit(str(e),
                  details=[
                      "Check that the model file exists and is accessible",
                      "Verify the blueprint file path is correct",
                      "Ensure you have read permissions for both files"
                  ])
    except Exception as e:
        if config.verbose:
            console.print_exception()
        error_exit(f"Failed during exploration: {e}",
                  details=[
                      "Check the model is a valid ONNX file",
                      "Verify the blueprint YAML syntax is correct",
                      "Ensure all required dependencies are installed",
                      "Run with --verbose for more details"
                  ])