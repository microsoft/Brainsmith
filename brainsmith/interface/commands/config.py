"""Configuration management commands for brainsmith CLI."""

# Standard library imports
import json
import os
from pathlib import Path
from typing import Any, Optional, Dict

# Third-party imports
import click
import yaml
from rich.syntax import Syntax
from rich.table import Table

# Local imports
from brainsmith.config import BrainsmithConfig
from ..context import ApplicationContext
from ..utils import console, error_exit, success, warning, tip


@click.group()
def config():
    """Manage Brainsmith configuration."""
    pass


@config.command()
@click.option('--format', '-f', type=click.Choice(['table', 'yaml', 'json', 'env']), 
              default='table', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Include source information and path validation')
@click.option('--external-only', is_flag=True, help='For env format, show only external tool variables')
@click.pass_obj
def show(ctx: ApplicationContext, format: str, verbose: bool, external_only: bool) -> None:
    """Display current configuration.
    
    Shows the effective configuration after merging all sources:
    user defaults, project settings, environment variables, and CLI overrides.
    """
    try:
        config = ctx.get_effective_config()
        
        if format == 'table':
            if verbose:
                _show_verbose_table_format(config)
            else:
                _show_table_format(config)
        elif format == 'yaml':
            config_dict = config.model_dump()
            console.print(Syntax(yaml.dump(config_dict, default_flow_style=False), "yaml"))
        elif format == 'json':
            config_dict = config.model_dump()
            console.print(Syntax(json.dumps(config_dict, indent=2, default=str), "json"))
        elif format == 'env':
            if external_only:
                env_dict = config.to_external_env_dict()
            else:
                env_dict = config.to_all_env_dict()
            for key, value in sorted(env_dict.items()):
                console.print(f"export {key}={value}")
                
    except Exception as e:
        error_exit(f"Failed to load configuration: {e}")


def _format_path_display(path: Optional[Path], base_path: Optional[Path] = None, original_value: Optional[Path] = None) -> str:
    """Format a path for display, showing relative portions in green and base in grey.
    
    Args:
        path: The resolved path to format
        base_path: The base path to check against
        original_value: The original configured value (if different from path)
        
    Returns:
        Formatted string with appropriate styling
    """
    if path is None:
        return "None"
    
    path_obj = Path(path)
    
    # Determine the actual path to check for existence
    if not path_obj.is_absolute() and base_path:
        check_path = base_path / path_obj
    else:
        check_path = path_obj
    
    # Determine color based on existence
    color = "green" if check_path.exists() else "yellow"
    
    # If path is not absolute, it's a relative path that should be shown relative to base_path
    if not path_obj.is_absolute() and base_path:
        return f"[dim]{base_path}/[/dim][{color}]{path}[/{color}]"
    
    # Use original_value if provided (for cases where path is computed)
    if original_value is not None and not Path(original_value).is_absolute():
        # Show as relative with full path
        base_str = str(base_path) if base_path else str(path_obj.parent)
        rel_str = str(original_value)
        return f"[dim]{base_str}/[/dim][{color}]{rel_str}[/{color}]"
    
    # If path is under base_path, show it as relative
    if base_path and path_obj.is_absolute() and base_path.is_absolute():
        try:
            rel_path = path_obj.relative_to(base_path)
            return f"[dim]{base_path}/[/dim][{color}]{rel_path}[/{color}]"
        except ValueError:
            # Path is not relative to base_path
            pass
    
    # Otherwise, show as-is with appropriate color
    return f"[{color}]{path}[/{color}]"


def _show_table_format(config: BrainsmithConfig) -> None:
    """Display configuration in a formatted table.
    
    Args:
        config: The Brainsmith configuration object
    """
    table = Table(title="Brainsmith Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")  # No style to allow markup in cells
    
    # Core paths
    table.add_row("Core Paths", "")
    table.add_row("  Build Directory", _format_path_display(config.build_dir, config.bsmith_dir))
    table.add_row("  Dependencies Directory", _format_path_display(config.deps_dir, config.bsmith_dir, config.deps_dir))
    
    # Toolchain Settings
    table.add_row("", "")
    table.add_row("Toolchain Settings", "")
    table.add_row("  Plugins Strict", str(config.plugins_strict))
    table.add_row("  Debug Mode", str(config.debug))
    table.add_row("  Netron Port", str(config.netron_port))
    if config.vivado_ip_cache:
        table.add_row("  Vivado IP Cache", _format_path_display(config.vivado_ip_cache, config.bsmith_dir))
    
    # Xilinx tools
    table.add_row("", "")
    table.add_row("Xilinx Tools", "")
    
    # Format Vivado path with base and version highlighted, color based on existence
    if config.effective_vivado_path:
        color = "green" if config.effective_vivado_path.exists() else "yellow"
        vivado_display = f"[{color}]{config.xilinx_path}[/{color}][dim]/Vivado/[/dim][{color}]{config.xilinx_version}[/{color}]"
    else:
        vivado_display = "[yellow]Not found[/yellow]"
    table.add_row("  Vivado", vivado_display)
    
    # Format Vitis path with base and version highlighted, color based on existence
    if config.effective_vitis_path:
        color = "green" if config.effective_vitis_path.exists() else "yellow"
        vitis_display = f"[{color}]{config.xilinx_path}[/{color}][dim]/Vitis/[/dim][{color}]{config.xilinx_version}[/{color}]"
    else:
        vitis_display = "[yellow]Not found[/yellow]"
    table.add_row("  Vitis", vitis_display)
    
    # Format Vitis HLS path with base and version highlighted, color based on existence
    if config.effective_vitis_hls_path:
        color = "green" if config.effective_vitis_hls_path.exists() else "yellow"
        vitis_hls_display = f"[{color}]{config.xilinx_path}[/{color}][dim]/Vitis_HLS/[/dim][{color}]{config.xilinx_version}[/{color}]"
    else:
        vitis_hls_display = "[yellow]Not found[/yellow]"
    table.add_row("  Vitis HLS", vitis_hls_display)
    
    console.print(table)


def _show_verbose_table_format(config: BrainsmithConfig) -> None:
    """Display configuration with source information.
    
    Args:
        config: The Brainsmith configuration object
    """
    # Create main table with source column
    table = Table(title="Brainsmith Configuration (Verbose)")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")  # No style to allow markup in cells
    table.add_column("Source", style="yellow")
    
    # Helper to determine source
    def get_source(setting_name: str, env_var: str, is_derived: bool = False) -> str:
        if os.environ.get(env_var):
            return f"env: {env_var}"
        # Try to find YAML file
        yaml_file = None
        for possible in ["brainsmith_settings.yaml", ".brainsmith.yaml"]:
            if Path(possible).exists():
                yaml_file = Path(possible)
                break
        if yaml_file:
            try:
                yaml_data = yaml.safe_load(yaml_file.read_text())
                if setting_name in yaml_data:
                    return f"yaml: {yaml_file.name}"
            except:
                pass
        return "derived" if is_derived else "default"
    
    # Core paths
    table.add_row("Core Paths", "", "")
    table.add_row("  Build Directory", _format_path_display(config.build_dir, config.bsmith_dir), 
                  get_source("build_dir", "BSMITH_BUILD_DIR"))
    table.add_row("  Dependencies Directory", _format_path_display(config.deps_dir, config.bsmith_dir, config.deps_dir),
                  get_source("deps_dir", "BSMITH_DEPS_DIR"))
    
    # Toolchain Settings
    table.add_row("", "", "")
    table.add_row("Toolchain Settings", "", "")
    table.add_row("  Debug Mode", str(config.debug),
                  get_source("debug", "BSMITH_DEBUG"))
    table.add_row("  Plugins Strict", str(config.plugins_strict),
                  get_source("plugins_strict", "BSMITH_PLUGINS_STRICT"))
    table.add_row("  Netron Port", str(config.netron_port),
                  get_source("netron_port", "BSMITH_NETRON_PORT"))
    if config.vivado_ip_cache:
        table.add_row("  Vivado IP Cache", _format_path_display(config.vivado_ip_cache, config.bsmith_dir),
                      get_source("vivado_ip_cache", "BSMITH_VIVADO_IP_CACHE", is_derived=not config._fields_set.get("vivado_ip_cache", False)))
    
    # Xilinx tools
    table.add_row("", "", "")
    table.add_row("Xilinx Tools", "", "")
    table.add_row("  Base Path", str(config.xilinx_path) if config.xilinx_path else "Not configured",
                  get_source("xilinx_path", "BSMITH_XILINX_PATH"))
    table.add_row("  Version", config.xilinx_version,
                  get_source("xilinx_version", "BSMITH_XILINX_VERSION"))
    
    # Format Vivado path with base and version highlighted, color based on existence
    if config.effective_vivado_path:
        color = "green" if config.effective_vivado_path.exists() else "yellow"
        vivado_display = f"[{color}]{config.xilinx_path}[/{color}][dim]/Vivado/[/dim][{color}]{config.xilinx_version}[/{color}]"
        table.add_row("  Vivado", vivado_display,
                      "derived" if not config.vivado_path else get_source("vivado_path", "BSMITH_VIVADO_PATH"))
    else:
        table.add_row("  Vivado", "[yellow]Not found[/yellow]", "—")
    
    # Format Vitis path with base and version highlighted, color based on existence
    if config.effective_vitis_path:
        color = "green" if config.effective_vitis_path.exists() else "yellow"
        vitis_display = f"[{color}]{config.xilinx_path}[/{color}][dim]/Vitis/[/dim][{color}]{config.xilinx_version}[/{color}]"
        table.add_row("  Vitis", vitis_display,
                      "derived" if not config.vitis_path else get_source("vitis_path", "BSMITH_VITIS_PATH"))
    else:
        table.add_row("  Vitis", "[yellow]Not found[/yellow]", "—")
    
    # Format Vitis HLS path with base and version highlighted, color based on existence
    if config.effective_vitis_hls_path:
        color = "green" if config.effective_vitis_hls_path.exists() else "yellow"
        vitis_hls_display = f"[{color}]{config.xilinx_path}[/{color}][dim]/Vitis_HLS/[/dim][{color}]{config.xilinx_version}[/{color}]"
        table.add_row("  Vitis HLS", vitis_hls_display,
                      "derived" if not config.vitis_hls_path else get_source("vitis_hls_path", "BSMITH_VITIS_HLS_PATH"))
    else:
        table.add_row("  Vitis HLS", "[yellow]Not found[/yellow]", "—")
    
    # FINN configuration (only in verbose mode)
    table.add_row("", "", "")
    table.add_row("FINN Configuration", "", "")
    
    # Use effective properties and check if they're using defaults
    finn_root = config.effective_finn_root
    finn_root_is_derived = config.finn.finn_root is None  # None means using default
    finn_root_original = None if finn_root_is_derived else config.finn.finn_root
    if finn_root_is_derived:
        finn_root_original = Path("deps/finn")  # Show the relative path for derived
    
    table.add_row("  FINN_ROOT", _format_path_display(finn_root, config.bsmith_dir, finn_root_original), 
                  "derived" if finn_root_is_derived else get_source("finn.finn_root", "BSMITH_FINN__FINN_ROOT"))
    
    # FINN_BUILD_DIR
    finn_build = config.effective_finn_build_dir
    finn_build_is_derived = config.finn.finn_build_dir is None  # None means using default
    table.add_row("  FINN_BUILD_DIR", _format_path_display(finn_build, config.bsmith_dir),
                  "derived" if finn_build_is_derived else get_source("finn.finn_build_dir", "BSMITH_FINN__FINN_BUILD_DIR"))
    
    # FINN_DEPS_DIR
    finn_deps = config.effective_finn_deps_dir
    finn_deps_is_derived = config.finn.finn_deps_dir is None  # None means using default
    finn_deps_original = None if finn_deps_is_derived else config.finn.finn_deps_dir
    if finn_deps_is_derived and finn_deps == config.deps_dir:
        finn_deps_original = config.deps_dir  # Show the actual deps_dir value
    table.add_row("  FINN_DEPS_DIR", _format_path_display(finn_deps, config.bsmith_dir, finn_deps_original),
                  "derived" if finn_deps_is_derived else get_source("finn.finn_deps_dir", "BSMITH_FINN__FINN_DEPS_DIR"))
    
    if config.finn.num_default_workers:
        table.add_row("  Default Workers", str(config.finn.num_default_workers),
                      get_source("finn.num_default_workers", "BSMITH_FINN__NUM_DEFAULT_WORKERS"))
    
    console.print(table)
    
    # Show validation warnings
    warnings = []
    if config.deps_dir and not config.deps_dir.is_absolute():
        expected = config.bsmith_dir / config.deps_dir
        if config.deps_dir.absolute() != expected.absolute():
            warnings.append(f"Relative deps_dir may not resolve correctly")
    
    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in warnings:
            console.print(f"  ⚠ {warning}")



@config.command()
@click.argument('key')
@click.argument('value')
@click.pass_obj
def set(ctx: ApplicationContext, key: str, value: str) -> None:
    """Set a configuration value in user defaults.
    
    Examples:
        brainsmith config set verbose true
        brainsmith config set build_dir /opt/builds
        brainsmith config set finn.num_default_workers 8
    """
    try:
        # Convert string value to appropriate type
        typed_value: Any = value
        
        # Handle boolean values
        if value.lower() in ['true', 'false']:
            typed_value = value.lower() == 'true'
        # Handle numeric values
        elif value.isdigit():
            typed_value = int(value)
        # Handle paths
        elif key.endswith('_dir') or key.endswith('_path'):
            typed_value = value  # Keep as string, will be converted to Path by config
        
        # Set the value
        ctx.set_user_config_value(key, typed_value)
        
        success(f"Set {key} = {value}")
        
        # Show where it was saved
        console.print(f"Saved to: {ctx.user_config_path}")
        
    except Exception as e:
        error_exit(f"Failed to set configuration: {e}")


@config.command()
@click.option('--shell', type=click.Choice(['bash', 'zsh', 'fish', 'powershell']),
              default='bash', help='Shell format for output')
@click.pass_obj
def export(ctx: ApplicationContext, shell: str) -> None:
    """Export configuration as shell environment script.
    
    Usage:
        eval $(brainsmith config export)
        eval $(brainsmith config export --shell fish)
    """
    try:
        # Use the env activate functionality
        from ..commands import env as env_commands
        ctx.invoke(env_commands.activate, shell=shell)
            
    except Exception as e:
        error_exit(f"Failed to export configuration: {e}")


@config.command()
@click.option('--user', is_flag=True, help='Create user-level config (~/.brainsmith/config.yaml)')
@click.option('--project', is_flag=True, help='Create project-level config (./brainsmith_settings.yaml)')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
@click.option('--full', is_flag=True, help='Include all possible configuration fields')
@click.pass_obj
def init(ctx: ApplicationContext, user: bool, project: bool, force: bool, full: bool) -> None:
    """Initialize a new configuration file.
    
    Creates either user-level config (~/.brainsmith/config.yaml) or
    project-level config (./brainsmith_settings.yaml).
    """
    # Determine output path
    if user and project:
        error_exit("Cannot specify both --user and --project")
    elif user:
        output = ctx.user_config_path
    elif project:
        output = Path("brainsmith_settings.yaml")
    else:
        # Default to project config
        output = Path("brainsmith_settings.yaml")
    
    if output.exists() and not force:
        error_exit(f"{output} already exists. Use --force to overwrite.")
    
    try:
        # Create parent directory if needed (especially for user config)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Load current config to get sensible defaults
        config = ctx.get_effective_config()
        
        if False:  # Remove minimal option
            # Old minimal behavior
            config_dict = {
                "build_dir": str(config.build_dir),
                "xilinx_path": str(config.xilinx_path) if config.xilinx_path else None,
                "xilinx_version": config.xilinx_version,
                "netron_port": config.netron_port,
                "debug": config.debug,
            }
            # Remove None values
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            
        elif full:
            # Full config with all fields
            config_dict = config.model_dump()
            # Convert Path objects to strings
            def convert_paths(obj):
                if isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, Path):
                    return str(obj)
                return obj
            config_dict = convert_paths(config_dict)
            
        else:
            # Default: complete config with commonly used fields
            config_dict = {}
            
            # Add header comment
            config_dict = {
                "__comment__": [
                    "Brainsmith Configuration File",
                    "Priority: CLI args > Environment vars > This file > Defaults",
                    "Environment variables use BSMITH_ prefix (e.g., BSMITH_BUILD_DIR)"
                ]
            }
            
            # Core paths
            config_dict["# Core paths"] = None
            config_dict["build_dir"] = str(config.build_dir)
            config_dict["deps_dir"] = "deps"  # Keep relative for portability
            
            # Xilinx tools
            config_dict[""] = None  # Empty line
            config_dict["# Xilinx tools"] = None
            if config.xilinx_path:
                config_dict["xilinx_path"] = str(config.xilinx_path)
            config_dict["xilinx_version"] = config.xilinx_version
            
            # FINN configuration (optional)
            config_dict[" "] = None  # Empty line
            config_dict["# FINN configuration (optional)"] = None
            config_dict["finn"] = {}
            if config.finn.finn_root:
                config_dict["finn"]["finn_root"] = str(config.finn.finn_root)
            if config.finn.finn_build_dir:
                config_dict["finn"]["finn_build_dir"] = str(config.finn.finn_build_dir)
            if config.finn.num_default_workers:
                config_dict["finn"]["num_default_workers"] = config.finn.num_default_workers
            
            # If finn section is empty, remove it
            if not config_dict["finn"]:
                config_dict["finn"] = {"__comment__": "Uses defaults: deps/finn and build_dir"}
            
            # Other settings
            config_dict["  "] = None  # Empty line
            config_dict["# Other settings"] = None
            config_dict["plugins_strict"] = config.plugins_strict
            config_dict["debug"] = config.debug
            config_dict["netron_port"] = config.netron_port
        
        # Special handling for YAML with comments
        if not minimal and not full:
            # Write custom YAML with comments
            lines = []
            for key, value in config_dict.items():
                if key.startswith("#"):
                    # Comment line
                    lines.append(key)
                elif key == "__comment__":
                    # Multi-line comment at top
                    for comment in value:
                        lines.append(f"# {comment}")
                elif key.strip() == "":
                    # Empty line
                    lines.append("")
                elif value is None:
                    # Skip None values (used for spacing)
                    continue
                elif isinstance(value, dict):
                    if "__comment__" in value:
                        lines.append(f"{key}:  # {value['__comment__']}")
                    else:
                        lines.append(f"{key}:")
                        for k, v in value.items():
                            if k != "__comment__":
                                lines.append(f"  {k}: {v}")
                else:
                    lines.append(f"{key}: {value}")
            
            with open(output, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        else:
            # Standard YAML dump for minimal/full
            with open(output, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        success(f"Created configuration file: {output}")
        console.print("\nYou can now edit this file to customize your settings.")
        if user:
            console.print("These settings will be used as defaults for all Brainsmith projects.")
        else:
            console.print("These settings will be used for this project.")
        
    except Exception as e:
        error_exit(f"Failed to create configuration: {e}")