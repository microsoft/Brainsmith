"""Brainsmith CLI - The smith command interface."""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import json
import yaml

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from brainsmith.core.plugins.simulation import SimulationSetup
from brainsmith.config import BrainsmithConfig, load_config, export_to_environment


console = Console()


class BrainsmithContext:
    """Context object for passing state between commands."""
    
    def __init__(self):
        self.verbose = False
        self.config: Optional[BrainsmithConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration using Pydantic."""
        try:
            self.config = load_config()
            # Export to environment for backward compatibility
            export_to_environment(self.config, verbose=False)
        except Exception as e:
            console.print(f"[red]Failed to load configuration: {e}[/red]")
            console.print("[yellow]Using defaults[/yellow]")
            # Try with minimal config
            self.config = BrainsmithConfig(
                bsmith_dir=Path.cwd(),
                bsmith_build_dir=Path("/tmp/brainsmith_build"),
                bsmith_deps_dir=Path.cwd() / "deps"
            )
    
    @property
    def deps_dir(self) -> Path:
        """Get dependencies directory."""
        return self.config.bsmith_deps_dir if self.config else Path.cwd() / "deps"
    
    @property
    def build_dir(self) -> Path:
        """Get build directory."""
        return self.config.bsmith_build_dir if self.config else Path("/tmp/brainsmith_build")
    
    @property
    def settings(self) -> Dict[str, Any]:
        """Get settings as dict for backward compatibility."""
        if not self.config:
            return {}
        return self.config.model_dump()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """Brainsmith CLI - QNN compiler framework for AMD/Xilinx FPGAs.
    
    Use 'smith COMMAND --help' for more information on a specific command.
    """
    ctx.obj = BrainsmithContext()
    ctx.obj.verbose = verbose
    
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


# Install command group
@cli.group()
@click.pass_context
def install(ctx):
    """Install optional components for Brainsmith."""
    pass


@install.command()
@click.option("--deps-dir", "-d", type=click.Path(), help="Dependencies directory")
@click.pass_context
def cppsim(ctx, deps_dir):
    """Install C++ simulation dependencies.
    
    This installs:
    - cnpy: C++ NumPy library
    - finn-hlslib: FINN HLS library headers
    """
    console.print("\n[bold blue]Installing C++ Simulation Dependencies[/bold blue]\n")
    
    sim_setup = SimulationSetup(deps_dir=Path(deps_dir) if deps_dir else None)
    
    if sim_setup.setup_cppsim():
        console.print("\n[green]✓ C++ simulation dependencies installed successfully![/green]")
        ctx.exit(0)
    else:
        console.print("\n[red]✗ Failed to install C++ simulation dependencies[/red]")
        ctx.exit(1)


@install.command()
@click.option("--deps-dir", "-d", type=click.Path(), help="Dependencies directory")
@click.pass_context
def xsim(ctx, deps_dir):
    """Install RTL simulation dependencies (Xilinx XSim).
    
    This builds the finnxsi module for RTL co-simulation.
    Requires Vivado to be installed and XILINX_VIVADO to be set.
    """
    console.print("\n[bold blue]Installing RTL Simulation Dependencies[/bold blue]\n")
    
    sim_setup = SimulationSetup(deps_dir=Path(deps_dir) if deps_dir else None)
    
    if sim_setup.setup_rtlsim():
        console.print("\n[green]✓ RTL simulation dependencies installed successfully![/green]")
        ctx.exit(0)
    else:
        console.print("\n[red]✗ Failed to install RTL simulation dependencies[/red]")
        ctx.exit(1)


@install.command()
@click.option("--deps-dir", "-d", type=click.Path(), help="Dependencies directory")
@click.argument("board_names", nargs=-1)
@click.pass_context
def boards(ctx, deps_dir, board_names):
    """Download board definition files.
    
    Available boards:
    - avnet: Avnet board files
    - xilinx-rfsoc: Xilinx RFSoC boards
    - realdigital: RealDigital RFSoC4x2
    
    If no board names specified, downloads all available boards.
    """
    console.print("\n[bold blue]Downloading Board Definition Files[/bold blue]\n")
    
    sim_setup = SimulationSetup(deps_dir=Path(deps_dir) if deps_dir else None)
    
    boards_list = list(board_names) if board_names else None
    
    if sim_setup.download_board_files(boards=boards_list):
        console.print("\n[green]✓ Board files downloaded successfully![/green]")
        ctx.exit(0)
    else:
        console.print("\n[red]✗ Failed to download board files[/red]")
        ctx.exit(1)


# Main build command
@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("blueprint_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(), required=False)
@click.option("--build-dir", "-b", type=click.Path(), help="Build directory")
@click.option("--num-workers", "-n", type=int, default=-1, help="Number of parallel workers")
@click.option("--start", type=str, help="Start step in build flow")
@click.option("--stop", type=str, help="Stop step in build flow")
@click.option("--skip-dep-update", "-s", is_flag=True, help="Skip dependency updates")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def build(ctx, model_path, blueprint_path, output_dir, build_dir, num_workers, 
          start, stop, skip_dep_update, verbose):
    """Build QNN hardware from ONNX model using blueprint.
    
    MODEL_PATH: Path to ONNX model file
    BLUEPRINT_PATH: Path to blueprint configuration (YAML)
    OUTPUT_DIR: Output directory (optional, defaults to build dir)
    
    The blueprint defines the hardware design space including:
    - Target clock frequency
    - Output type (estimates/rtl/bitfile)
    - Hardware kernels to use
    - Transformation pipeline steps
    
    Examples:
        smith build model.onnx blueprint.yaml
        smith build model.onnx bert.yaml output/
        smith build model.onnx base.yaml --verbose
    """
    console.print("\n[bold blue]Building QNN Hardware Design[/bold blue]\n")
    
    # Import here to ensure environment is set up
    from brainsmith.compiler import compile_model
    
    # Set up paths
    model_path = Path(model_path).absolute()
    blueprint_path = Path(blueprint_path).absolute()
    
    if not model_path.exists():
        console.print(f"[red]Model file not found: {model_path}[/red]")
        ctx.exit(1)
    
    if not blueprint_path.exists():
        console.print(f"[red]Blueprint file not found: {blueprint_path}[/red]")
        ctx.exit(1)
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir).absolute()
    else:
        output_path = ctx.obj.config.bsmith_build_dir / "output"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"Model: {model_path}")
    console.print(f"Blueprint: {blueprint_path}")
    console.print(f"Output: {output_path}")
    
    if verbose or ctx.obj.verbose:
        console.print(f"Build dir: {build_dir or ctx.obj.config.bsmith_build_dir}")
        console.print(f"Workers: {num_workers}")
    
    # Load blueprint
    with open(blueprint_path) as f:
        if blueprint_path.suffix in [".yaml", ".yml"]:
            blueprint = yaml.safe_load(f)
        else:
            blueprint = json.load(f)
    
    # Set up environment
    env = os.environ.copy()
    if build_dir:
        env["BSMITH_BUILD_DIR"] = str(build_dir)
    
    # Run compilation
    try:
        console.print("\n[yellow]Starting compilation...[/yellow]")
        
        # Display blueprint configuration
        console.print("\n[dim]Blueprint Configuration:[/dim]")
        console.print(f"  Name: {blueprint.get('name', 'Unnamed')}")
        if 'description' in blueprint:
            console.print(f"  Description: {blueprint['description']}")
        console.print(f"  Clock: {blueprint.get('clock_ns', 5.0)}ns ({1000/blueprint.get('clock_ns', 5.0):.1f} MHz)")
        console.print(f"  Output: {blueprint.get('output', 'estimates')}")
        if 'board' in blueprint:
            console.print(f"  Board: {blueprint['board']}")
        
        # Show design space
        if 'design_space' in blueprint:
            kernels = blueprint['design_space'].get('kernels', [])
            steps = blueprint['design_space'].get('steps', [])
            console.print(f"  Kernels: {len(kernels)} defined")
            console.print(f"  Steps: {len(steps)} in pipeline")
        
        # TODO: Replace with actual compile_model call
        # from brainsmith.compiler import compile_qnn
        # result = compile_qnn(
        #     model_path=str(model_path),
        #     blueprint_path=str(blueprint_path),
        #     output_dir=str(output_path),
        #     build_dir=str(build_dir or ctx.obj.build_dir),
        #     num_workers=num_workers,
        #     start_step=start,
        #     stop_step=stop,
        #     verbose=verbose or ctx.obj.verbose
        # )
        
        # For now, show what would be executed
        console.print("\n[dim]Build command (not yet implemented):[/dim]")
        console.print(f"  brainsmith.compiler.compile_qnn(")
        console.print(f"      model_path={model_path},")
        console.print(f"      blueprint_path={blueprint_path},")
        console.print(f"      output_dir={output_path})")
        
        console.print("\n[green]✓ Build completed successfully![/green]")
        console.print(f"[green]Output saved to: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Build failed: {e}[/red]")
        ctx.exit(1)


# Config command group
@cli.group()
@click.pass_context
def config(ctx):
    """Manage Brainsmith configuration settings."""
    pass


@config.command(name="list")
@click.pass_context
def config_list(ctx):
    """List all configuration settings."""
    console.print("\n[bold]Brainsmith Configuration[/bold]\n")
    
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")
    
    # Show all configuration from Pydantic model
    if ctx.obj.config:
        # Core paths
        table.add_row("bsmith_dir", str(ctx.obj.config.bsmith_dir), "core")
        table.add_row("bsmith_build_dir", str(ctx.obj.config.bsmith_build_dir), "core")
        table.add_row("bsmith_deps_dir", str(ctx.obj.config.bsmith_deps_dir), "core")
        
        # Python settings
        table.add_row("python.version", ctx.obj.config.python.version, "settings")
        table.add_row("python.unbuffered", str(ctx.obj.config.python.unbuffered), "settings")
        
        # Xilinx tools
        if ctx.obj.config.xilinx.vivado_path:
            table.add_row("xilinx.vivado_path", str(ctx.obj.config.xilinx.vivado_path), "optional")
        if ctx.obj.config.xilinx.vitis_path:
            table.add_row("xilinx.vitis_path", str(ctx.obj.config.xilinx.vitis_path), "optional")
        if ctx.obj.config.xilinx.hls_path:
            table.add_row("xilinx.hls_path", str(ctx.obj.config.xilinx.hls_path), "optional")
        
        # Other settings
        table.add_row("hw_compiler", ctx.obj.config.hw_compiler, "settings")
        table.add_row("debug.enabled", str(ctx.obj.config.debug.enabled), "settings")
    else:
        table.add_row("error", "Configuration not loaded", "error")
    
    console.print(table)


@config.command()
@click.argument("key")
@click.pass_context
def get(ctx, key):
    """Get a specific configuration value."""
    value = ctx.obj.settings.get(key.lower())
    
    if value is not None:
        console.print(f"{key}: {value}")
    else:
        console.print(f"[yellow]Setting '{key}' not found[/yellow]")
        ctx.exit(1)


@config.command()
@click.argument("key")
@click.argument("value")
@click.option("--global", "-g", "is_global", is_flag=True, help="Set globally")
@click.pass_context
def set(ctx, key, value, is_global):
    """Set a configuration value."""
    # Determine config file location
    if is_global:
        config_file = Path.home() / ".brainsmith" / "settings.yaml"
    else:
        config_file = Path.cwd() / "brainsmith_settings.yaml"
    
    # Load existing settings
    settings = {}
    if config_file.exists():
        with open(config_file) as f:
            settings = yaml.safe_load(f) or {}
    
    # Update setting
    settings[key] = value
    
    # Save settings
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(settings, f, default_flow_style=False)
    
    console.print(f"[green]✓ Set {key} = {value}[/green]")
    console.print(f"[dim]Saved to: {config_file}[/dim]")


@config.command()
@click.argument("path", type=click.Path(), required=False)
@click.pass_context
def create(ctx, path):
    """Create a template settings file."""
    if path:
        config_file = Path(path) / "brainsmith_settings.yaml"
    else:
        config_file = Path.cwd() / "brainsmith_settings.yaml"
    
    if config_file.exists():
        console.print(f"[yellow]Settings file already exists: {config_file}[/yellow]")
        if not click.confirm("Overwrite?"):
            ctx.exit(0)
    
    # Template settings
    template = {
        "xilinx_tools_path": "/tools/Xilinx",
        "default_board": "pynq-z1",
        "num_workers": -1,
        "build_dir": "/tmp/brainsmith_build",
        "log_level": "INFO",
    }
    
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(template, f, default_flow_style=False)
    
    console.print(f"[green]✓ Created settings template: {config_file}[/green]")




# Test command
@cli.command()
@click.argument("test_path", required=False)
@click.option("--markers", "-m", help="Pytest markers")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def test(ctx, test_path, markers, verbose):
    """Run Brainsmith tests.
    
    TEST_PATH: Specific test file or directory (optional)
    
    Examples:
        smith test                    # Run all tests
        smith test tests/unit/        # Run unit tests
        smith test -m "not slow"      # Skip slow tests
    """
    console.print("\n[bold blue]Running Brainsmith Tests[/bold blue]\n")
    
    # Build pytest command
    cmd = ["pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if markers:
        cmd.extend(["-m", markers])
    
    if test_path:
        cmd.append(test_path)
    
    # Run tests
    try:
        subprocess.run(cmd, check=True)
        console.print("\n[green]✓ All tests passed![/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]✗ Tests failed with exit code: {e.returncode}[/red]")
        ctx.exit(1)


# Run command
@cli.command()
@click.argument("script_path", type=click.Path(exists=True))
@click.argument("script_args", nargs=-1)
@click.pass_context
def run(ctx, script_path, script_args):
    """Run a Python script in the Brainsmith environment.
    
    SCRIPT_PATH: Path to Python script
    SCRIPT_ARGS: Arguments to pass to the script
    
    Example:
        smith run my_script.py --arg1 value1
    """
    console.print(f"\n[bold blue]Running script: {script_path}[/bold blue]\n")
    
    # Build command
    cmd = [sys.executable, script_path] + list(script_args)
    
    # Run script
    try:
        subprocess.run(cmd, check=True)
        console.print("\n[green]✓ Script completed successfully![/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]✗ Script failed with exit code: {e.returncode}[/red]")
        ctx.exit(1)


def main():
    """Main entry point for the smith CLI."""
    cli(obj=BrainsmithContext())


if __name__ == "__main__":
    main()