# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import subprocess
import sys
from pathlib import Path

import click

from ..utils import console, error_exit, success, progress_spinner


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('rtl_file', type=click.Path(exists=True, path_type=Path))
@click.option('--artifacts', multiple=True,
              type=click.Choice(['autohwcustomop', 'rtlbackend', 'wrapper']),
              help='Generate specific files only (can specify multiple)')
@click.option('--include-rtl', multiple=True, type=click.Path(exists=True, path_type=Path),
              help='Additional RTL files to include (can specify multiple)')
@click.option('--info', is_flag=True,
              help='Display parsed kernel metadata and exit')
@click.option('--no-strict', is_flag=True,
              help='Disable strict validation')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Directory where generated files will be saved (default: same as RTL file)')
@click.option('--rtl-path', type=str,
              help='Colon-separated paths to search for RTL files')
@click.option('--validate', is_flag=True,
              help='Validate RTL only without generating files')
@click.option('--verbose', '-v', is_flag=True,
              help='Show detailed output from kernel integrator tool')
def kernel(
    rtl_file: Path,
    artifacts: tuple[str, ...],
    include_rtl: tuple[Path, ...],
    info: bool,
    no_strict: bool,
    output_dir: Path | None,
    rtl_path: str | None,
    validate: bool,
    verbose: bool
) -> None:
    """Generate hardware kernel from RTL for FINN integration.
    
    RTL_FILE: Path to SystemVerilog RTL source file (.sv) with embedded pragmas
    """
    console.print(f"[bold blue]Brainsmith Kernel Integrator[/bold blue]")
    console.print(f"RTL File: {rtl_file}")

    if output_dir is None:
        output_dir = rtl_file.parent
    console.print(f"Output Directory: {output_dir}")

    cmd = [sys.executable, "-m", "brainsmith.tools.kernel_integrator", str(rtl_file)]

    cmd.extend(["-o", str(output_dir)])

    if validate:
        cmd.append("--validate")
    if info:
        cmd.append("--info")
    if no_strict:
        cmd.append("--no-strict")
    if verbose:
        cmd.append("-v")

    for artifact in artifacts:
        cmd.extend(["--artifacts", artifact])

    for rtl in include_rtl:
        cmd.extend(["--include-rtl", str(rtl)])

    if rtl_path:
        cmd.extend(["--rtl-path", rtl_path])

    try:
        action = "Validating RTL..." if validate else "Generating hardware kernel..."

        if info:
            # For info mode, run directly without spinner
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                console.print(result.stdout)
            else:
                error_exit(f"Failed to parse RTL: {result.stderr}")
        else:
            with progress_spinner(action) as task:
                result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                if validate:
                    success("RTL validation completed successfully!")
                else:
                    success("Hardware kernel generation completed!")

                if result.stdout and verbose:
                    console.print("\n[dim]Tool output:[/dim]")
                    console.print(result.stdout)
            else:
                error_exit(
                    f"Kernel integrator failed: {result.stderr}",
                    details=[
                        "Verify the RTL file contains valid SystemVerilog with @brainsmith pragmas",
                        "Check that all pragma syntax is correct",
                        "Ensure any referenced RTL files in pragmas exist",
                        "Confirm you have write permissions to the output directory"
                    ]
                )
            
    except FileNotFoundError:
        error_exit(
            "Kernel integrator tool not found",
            details=[
                "The brainsmith.tools.kernel_integrator module is installed",
                "Python can find the module in the Python path"
            ]
        )
    except Exception as e:
        error_exit(f"Unexpected error: {e}")
