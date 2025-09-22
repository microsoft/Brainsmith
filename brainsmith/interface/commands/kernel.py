"""Hardware kernel generation command for the smith CLI."""

# Standard library imports
from pathlib import Path
from typing import Optional, Dict, Any

# Third-party imports
import click

# Local imports
from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator
from ..utils import console, error_exit, success, progress_spinner


@click.command()
@click.argument('rtl_file', type=click.Path(exists=True, path_type=Path))
@click.argument('compiler_data', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), required=True,
              help='Directory where generated files will be saved')
@click.option('--custom-doc', '-d', type=click.Path(exists=True, path_type=Path),
              help='Optional Markdown file with custom documentation sections')
@click.option('--stop-after', type=click.Choice([
    'parse_rtl', 'parse_compiler_data', 'load_custom_documentation',
    'generate_rtl_template', 'generate_hw_custom_op', 'generate_rtl_backend',
    'generate_documentation'
]), help='Stop execution after specified phase (for debugging)')
def kernel(rtl_file: Path, compiler_data: Path, output_dir: Path, 
          custom_doc: Optional[Path], stop_after: Optional[str]) -> None:
    """Generate hardware kernel from RTL for FINN integration.
    
    RTL_FILE: Path to SystemVerilog RTL source file (.sv)
    COMPILER_DATA: Path to Python file with compiler data (ONNX pattern, cost functions)
    """
    console.print(f"[bold blue]Hardware Kernel Generator[/bold blue]")
    console.print(f"RTL File: {rtl_file}")
    console.print(f"Compiler Data: {compiler_data}")
    console.print(f"Output Directory: {output_dir}")
    
    if custom_doc:
        console.print(f"Custom Documentation: {custom_doc}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize the generator
        generator = HardwareKernelGenerator(
            rtl_file=str(rtl_file),
            compiler_data_file=str(compiler_data),
            output_dir=str(output_dir),
            custom_doc_file=str(custom_doc) if custom_doc else None
        )
        
        # Run the generator with progress indication
        with progress_spinner("Generating hardware kernel...") as task:
            # Run the generator
            generated_files = generator.run(stop_after=stop_after)
        
        # Display results
        success("Hardware kernel generation completed!")
        
        if generated_files:
            console.print("\nGenerated files:")
            for file_type, file_path in generated_files.items():
                if file_path:
                    console.print(f"  • {file_type}: {file_path}")
        
        if stop_after:
            console.print(f"\n[yellow]Stopped after phase:[/yellow] {stop_after}")
            
    except FileNotFoundError as e:
        error_exit(str(e))
    except Exception as e:
        error_exit(
            f"Failed during generation: {e}",
            details=[
                "The RTL file is valid SystemVerilog",
                "The compiler data file is a valid Python module",
                "You have write permissions to the output directory"
            ]
        )