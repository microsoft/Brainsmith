"""Configuration management commands for the smith CLI."""

# Standard library imports
import json
from pathlib import Path
from typing import Any

# Third-party imports
import click
import yaml
from rich.syntax import Syntax
from rich.table import Table

# Local imports
from brainsmith.config import get_config, load_config, BrainsmithConfig
from brainsmith.config.export import export_to_environment
from ..utils import console, error_exit, success, warning, tip


@click.group()
def config():
    """Manage Brainsmith configuration."""
    pass


@config.command()
@click.option('--format', '-f', type=click.Choice(['table', 'yaml', 'json', 'env']), 
              default='table', help='Output format')
def show(format: str) -> None:
    """Display current configuration.
    
    Args:
        format: Output format (table, yaml, json, or env)
    """
    try:
        config = get_config()
        
        if format == 'table':
            _show_table_format(config)
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
        error_exit(f"Failed to load configuration: {e}")


def _show_table_format(config: BrainsmithConfig) -> None:
    """Display configuration in a formatted table.
    
    Args:
        config: The Brainsmith configuration object
    """
    table = Table(title="Brainsmith Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Core paths
    table.add_row("Core Paths", "")
    table.add_row("  BSMITH_DIR", str(config.bsmith_dir))
    table.add_row("  BSMITH_BUILD_DIR", str(config.build_dir))
    table.add_row("  BSMITH_DEPS_DIR", str(config.deps_dir))
    
    # Xilinx tools
    table.add_row("", "")
    table.add_row("Xilinx Tools", "")
    table.add_row("  Vivado", str(config.effective_vivado_path) if config.effective_vivado_path else "Not found")
    table.add_row("  Vitis", str(config.effective_vitis_path) if config.effective_vitis_path else "Not found")
    table.add_row("  HLS", str(config.effective_vitis_hls_path) if config.effective_vitis_hls_path else "Not found")
    table.add_row("  Version", config.xilinx_version)
    
    # Other settings
    table.add_row("", "")
    table.add_row("Other Settings", "")
    table.add_row("  Plugins Strict", str(config.plugins_strict))
    table.add_row("  Debug Enabled", str(config.debug))
    
    console.print(table)


@config.command()
def validate() -> None:
    """Validate current configuration.
    
    Checks that the configuration can be loaded and that
    the BSMITH_DIR exists.
    """
    try:
        config = get_config()
        console.print("Validating current configuration")
        
        # Basic validation - just check that we can load the config
        # The validators in the schema will catch any real issues
        if config.bsmith_dir and config.bsmith_dir.exists():
            success("Configuration is valid!")
        else:
            error_exit("Configuration error: BSMITH_DIR not found")
            
    except Exception as e:
        error_exit(f"Failed to validate configuration: {e}")


@config.command()
@click.option('--verbose', '-v', is_flag=True, help='Show exported variables')
def export(verbose: bool) -> None:
    """Export configuration as environment variables.
    
    Args:
        verbose: Whether to show the exported variables
    """
    try:
        config = get_config()
        export_to_environment(config, verbose=verbose)
        
        if not verbose:
            success("Configuration exported to environment")
            console.print("Use --verbose to see exported variables")
            
    except Exception as e:
        error_exit(f"Failed to export configuration: {e}")


@config.command()
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default=Path("brainsmith_settings.yaml"),
              help='Output file path')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
def init(output: Path, force: bool) -> None:
    """Initialize a new configuration file with defaults.
    
    Args:
        output: Path to the output configuration file
        force: Whether to overwrite an existing file
    """
    if output.exists() and not force:
        error_exit(f"{output} already exists. Use --force to overwrite.")
    
    try:
        # Load current config to get sensible defaults
        config = get_config()
        
        # Create a minimal config with important settings
        minimal_config = {
            "build_dir": str(config.build_dir),
            "xilinx_path": str(config.xilinx_path) if config.xilinx_path else None,
            "xilinx_version": config.xilinx_version,
            "netron_port": config.netron_port,
            "debug": config.debug,
        }
        
        # Remove None values for cleaner YAML
        minimal_config = {k: v for k, v in minimal_config.items() if v is not None}
        
        # Write YAML file
        with open(output, 'w') as f:
            yaml.dump(minimal_config, f, default_flow_style=False, sort_keys=False)
        
        success(f"Created configuration file: {output}")
        console.print("\nYou can now edit this file to customize your settings.")
        console.print("Run 'smith config validate' to check your configuration.")
        
    except Exception as e:
        error_exit(f"Failed to create configuration: {e}")