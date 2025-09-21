"""Configuration management commands for the smith CLI."""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax
import yaml

from brainsmith.config import get_config, load_config, validate_and_report, ConfigPriority
from brainsmith.config.export import export_to_environment

console = Console()


@click.group()
def config():
    """Manage Brainsmith configuration."""
    pass


@config.command()
@click.option('--format', '-f', type=click.Choice(['table', 'yaml', 'json', 'env']), 
              default='table', help='Output format')
@click.option('--show-source', is_flag=True, help='Show configuration sources')
def show(format: str, show_source: bool):
    """Display current configuration."""
    try:
        config = get_config()
        
        if format == 'table':
            _show_table_format(config, show_source)
        elif format == 'yaml':
            config_dict = config.model_dump()
            console.print(Syntax(yaml.dump(config_dict, default_flow_style=False), "yaml"))
        elif format == 'json':
            config_dict = config.model_dump()
            console.print(Syntax(json.dumps(config_dict, indent=2, default=str), "json"))
        elif format == 'env':
            env_dict = config.to_env_dict()
            for key, value in sorted(env_dict.items()):
                console.print(f"export {key}={value}")
                
    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}")
        sys.exit(1)


def _show_table_format(config, show_source: bool):
    """Display configuration in a formatted table."""
    table = Table(title="Brainsmith Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    if show_source:
        table.add_column("Priority", style="yellow")
    
    # Core paths
    table.add_row("Core Paths", "", "")
    table.add_row("  BSMITH_DIR", str(config.bsmith_dir), 
                  config.get_priority("bsmith_dir").value if show_source else None)
    table.add_row("  BSMITH_BUILD_DIR", str(config.bsmith_build_dir),
                  config.get_priority("bsmith_build_dir").value if show_source else None)
    table.add_row("  BSMITH_DEPS_DIR", str(config.bsmith_deps_dir),
                  config.get_priority("bsmith_deps_dir").value if show_source else None)
    
    # Python config
    table.add_row("", "", "")
    table.add_row("Python", "", "")
    table.add_row("  Version", config.python.version, "")
    table.add_row("  Unbuffered", str(config.python.unbuffered), "")
    
    # Xilinx tools
    table.add_row("", "", "")
    table.add_row("Xilinx Tools", "", "")
    table.add_row("  Vivado", str(config.xilinx.vivado_path) if config.xilinx.vivado_path else "Not found",
                  config.get_priority("xilinx.vivado_path").value if show_source else None)
    table.add_row("  Vitis", str(config.xilinx.vitis_path) if config.xilinx.vitis_path else "Not found",
                  config.get_priority("xilinx.vitis_path").value if show_source else None)
    table.add_row("  HLS", str(config.xilinx.hls_path) if config.xilinx.hls_path else "Not found",
                  config.get_priority("xilinx.hls_path").value if show_source else None)
    table.add_row("  Version", config.xilinx.version, "")
    
    # Other settings
    table.add_row("", "", "")
    table.add_row("Other Settings", "", "")
    table.add_row("  HW Compiler", config.hw_compiler, "")
    table.add_row("  Plugins Strict", str(config.plugins_strict), "")
    table.add_row("  Debug Enabled", str(config.debug.enabled), "")
    
    console.print(table)


@config.command()
@click.option('--config-file', '-c', type=click.Path(exists=True, path_type=Path),
              help='Configuration file to validate')
def validate(config_file: Path):
    """Validate current configuration."""
    try:
        # Load config with optional override file
        if config_file:
            config = load_config(project_file=config_file)
            console.print(f"Validating configuration from: {config_file}")
        else:
            config = get_config()
            console.print("Validating current configuration")
        
        # Run validation
        results = config.validate_by_priority()
        
        # Display results
        error_count = len(results.get("errors", []))
        warning_count = len(results.get("warnings", []))
        info_count = len(results.get("info", []))
        
        if error_count > 0:
            console.print("\n[red]Errors:[/red]")
            for error in results["errors"]:
                console.print(f"  ✗ {error}")
        
        if warning_count > 0:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in results["warnings"]:
                console.print(f"  ⚠ {warning}")
        
        if info_count > 0:
            console.print("\n[blue]Info:[/blue]")
            for info in results["info"]:
                console.print(f"  ℹ {info}")
        
        if error_count == 0:
            console.print("\n[green]✓ Configuration is valid![/green]")
        else:
            console.print(f"\n[red]Configuration has {error_count} error(s)[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error validating configuration:[/red] {e}")
        sys.exit(1)


@config.command()
@click.option('--verbose', '-v', is_flag=True, help='Show exported variables')
def export(verbose: bool):
    """Export configuration as environment variables."""
    try:
        config = get_config()
        export_to_environment(config, verbose=verbose)
        
        if not verbose:
            console.print("[green]✓ Configuration exported to environment[/green]")
            console.print("Use --verbose to see exported variables")
            
    except Exception as e:
        console.print(f"[red]Error exporting configuration:[/red] {e}")
        sys.exit(1)


@config.command()
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default=Path("brainsmith_settings.yaml"),
              help='Output file path')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
def init(output: Path, force: bool):
    """Initialize a new configuration file with defaults."""
    if output.exists() and not force:
        console.print(f"[red]Error:[/red] {output} already exists. Use --force to overwrite.")
        sys.exit(1)
    
    try:
        # Load current config to get sensible defaults
        config = get_config()
        
        # Create a minimal config with important settings
        minimal_config = {
            "bsmith_build_dir": str(config.bsmith_build_dir),
            "xilinx": {
                "version": config.xilinx.version,
            },
            "dependencies": {
                "fetch_boards": config.dependencies.fetch_boards,
                "fetch_experimental": config.dependencies.fetch_experimental,
            },
            "debug": {
                "enabled": config.debug.enabled,
            }
        }
        
        # Add Xilinx paths if found
        if config.xilinx.vivado_path:
            minimal_config["xilinx"]["vivado_path"] = str(config.xilinx.vivado_path)
        if config.xilinx.vitis_path:
            minimal_config["xilinx"]["vitis_path"] = str(config.xilinx.vitis_path)
        if config.xilinx.hls_path:
            minimal_config["xilinx"]["hls_path"] = str(config.xilinx.hls_path)
        
        # Write YAML file
        with open(output, 'w') as f:
            yaml.dump(minimal_config, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[green]✓ Created configuration file:[/green] {output}")
        console.print("\nYou can now edit this file to customize your settings.")
        console.print("Run 'smith config validate' to check your configuration.")
        
    except Exception as e:
        console.print(f"[red]Error creating configuration:[/red] {e}")
        sys.exit(1)