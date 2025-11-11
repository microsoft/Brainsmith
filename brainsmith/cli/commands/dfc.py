# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from pathlib import Path

import click

# Import explore_design_space lazily inside function to keep --help fast
from ..context import ApplicationContext
from ..exceptions import CLIError
from ..messages import DFC_ERROR_HINT
from ..utils import console, success

logger = logging.getLogger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("model", type=click.Path(exists=True, path_type=Path))
@click.argument("blueprint", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(path_type=Path),
              help="Output directory (defaults to build dir with timestamp)")
@click.option("--start-step", type=str,
              help="Override blueprint start_step - start execution from this step (inclusive)")
@click.option("--stop-step", type=str,
              help="Override blueprint stop_step - stop execution at this step (inclusive)")
@click.pass_obj
def dfc(
    ctx: ApplicationContext,
    model: Path,
    blueprint: Path,
    output_dir: Path | None,
    start_step: str | None,
    stop_step: str | None
) -> None:
    """Create a dataflow core accelerator for neural network acceleration.

    \b
    MODEL: Path to ONNX model file
    BLUEPRINT: Path to Blueprint YAML file defining the dataflow architecture
    """
    config = ctx.get_effective_config()

    console.print("[bold blue]Brainsmith DFC[/bold blue] - Dataflow Core Creation")
    console.print(f"Model: {model}")
    console.print(f"Blueprint: {blueprint}")

    if output_dir:
        console.print(f"Output: {output_dir}")
    if start_step:
        console.print(f"Start step: {start_step}")
    if stop_step:
        console.print(f"Stop step: {stop_step}")

    logger.info(f"Starting dataflow core creation with model={model}, blueprint={blueprint}, output_dir={output_dir}")

    # Import DSE module only when command executes (not for --help)
    from brainsmith.dse import explore_design_space

    try:
        # Run design space exploration to create dataflow core
        result = explore_design_space(
            model_path=str(model),
            blueprint_path=str(blueprint),
            output_dir=str(output_dir) if output_dir else None,
            start_step_override=start_step,
            stop_step_override=stop_step
        )

        success("Dataflow core created successfully!")

        # Display summary if available
        if hasattr(result, "summary"):
            console.print(f"\nSummary: {result.summary}")

    except FileNotFoundError as e:
        raise CLIError(f"File not found: {e}")
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            console.print_exception()
        raise CLIError(f"Dataflow core creation failed: {e}", details=[DFC_ERROR_HINT])

