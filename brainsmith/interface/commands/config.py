"""Configuration management commands for the smith CLI."""

# Standard library imports
import json
import os
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
@click.option('--external-only', is_flag=True, help='For env format, show only external tool variables')
def show(format: str, external_only: bool) -> None:
    """Display current configuration.
    
    Args:
        format: Output format (table, yaml, json, or env)
        external_only: For env format, whether to show only external tool variables
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
            if external_only:
                env_dict = config.to_external_env_dict()
            else:
                env_dict = config.to_all_env_dict()
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
    table.add_row("  Build Directory", str(config.build_dir))
    table.add_row("  Dependencies Directory", str(config.deps_dir))
    
    # FINN configuration
    table.add_row("", "")
    table.add_row("FINN Configuration", "")
    finn_root = config.finn.finn_root or config.deps_dir / "finn"
    table.add_row("  FINN_ROOT", str(finn_root))
    finn_build = config.finn.finn_build_dir or config.build_dir
    table.add_row("  FINN_BUILD_DIR", str(finn_build))
    if config.finn.num_default_workers:
        table.add_row("  Default Workers", str(config.finn.num_default_workers))
    
    # Xilinx tools
    table.add_row("", "")
    table.add_row("Xilinx Tools", "")
    table.add_row("  Base Path", str(config.xilinx_path) if config.xilinx_path else "Not configured")
    table.add_row("  Vivado", str(config.effective_vivado_path) if config.effective_vivado_path else "Not found")
    table.add_row("  Vitis", str(config.effective_vitis_path) if config.effective_vitis_path else "Not found")
    table.add_row("  Vitis HLS", str(config.effective_vitis_hls_path) if config.effective_vitis_hls_path else "Not found")
    table.add_row("  Version", config.xilinx_version)
    
    # Other settings
    table.add_row("", "")
    table.add_row("Other Settings", "")
    table.add_row("  Plugins Strict", str(config.plugins_strict))
    table.add_row("  Debug Mode", str(config.debug))
    table.add_row("  Netron Port", str(config.netron_port))
    if config.vivado_ip_cache:
        table.add_row("  Vivado IP Cache", str(config.vivado_ip_cache))
    
    console.print(table)


@config.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed validation results')
def validate(verbose: bool) -> None:
    """Validate current configuration.
    
    Checks paths exist, tools are available, and configuration is consistent.
    
    Args:
        verbose: Show detailed validation results
    """
    try:
        config = get_config()
        console.print("Validating current configuration...")
        
        issues = []
        warnings = []
        
        # Check core paths
        if verbose:
            console.print("\n[bold]Checking core paths:[/bold]")
        
        # BSMITH_DIR
        if config.bsmith_dir and config.bsmith_dir.exists():
            if verbose:
                console.print(f"  ✓ BSMITH_DIR exists: {config.bsmith_dir}")
        else:
            issues.append(f"BSMITH_DIR not found: {config.bsmith_dir}")
        
        # Build directory
        if config.build_dir:
            if not config.build_dir.exists():
                try:
                    config.build_dir.mkdir(parents=True, exist_ok=True)
                    if verbose:
                        console.print(f"  ✓ Build directory created: {config.build_dir}")
                except:
                    warnings.append(f"Cannot create build directory: {config.build_dir}")
            elif verbose:
                console.print(f"  ✓ Build directory exists: {config.build_dir}")
        
        # Deps directory
        if config.deps_dir:
            if config.deps_dir.exists():
                if verbose:
                    console.print(f"  ✓ Dependencies directory exists: {config.deps_dir}")
            else:
                warnings.append(f"Dependencies directory not found: {config.deps_dir}")
        
        # Check path resolution
        if config.deps_dir and not config.deps_dir.is_absolute():
            # Check if relative path resolves correctly
            expected = config.bsmith_dir / config.deps_dir if config.bsmith_dir else None
            if expected and config.deps_dir.absolute() != expected.absolute():
                warnings.append(
                    f"Relative deps_dir may not resolve correctly:\n"
                    f"    Expected: {expected}\n"
                    f"    Actual: {config.deps_dir.absolute()}"
                )
        
        # Check Xilinx tools
        if verbose:
            console.print("\n[bold]Checking Xilinx tools:[/bold]")
        
        if config.xilinx_path:
            if config.effective_vivado_path:
                if verbose:
                    console.print(f"  ✓ Vivado found: {config.effective_vivado_path}")
            else:
                warnings.append("Vivado not found at expected path")
                
            if config.effective_vitis_hls_path:
                if verbose:
                    console.print(f"  ✓ Vitis HLS found: {config.effective_vitis_hls_path}")
            else:
                warnings.append("Vitis HLS not found at expected path")
        elif verbose:
            console.print("  - Xilinx tools not configured")
        
        # Check FINN paths
        if verbose:
            console.print("\n[bold]Checking FINN configuration:[/bold]")
        
        finn_root = config.finn.finn_root or config.deps_dir / "finn"
        if finn_root.exists():
            if verbose:
                console.print(f"  ✓ FINN_ROOT exists: {finn_root}")
            # Check for finn_xsi if present
            finn_xsi = finn_root / "finn_xsi" / "xsi.so"
            if finn_xsi.exists():
                if verbose:
                    console.print(f"  ✓ FINN XSI module built")
            elif verbose:
                console.print(f"  - FINN XSI module not built (optional)")
        else:
            warnings.append(f"FINN not found at: {finn_root}")
        
        # Show results
        console.print()
        if issues:
            console.print("[bold red]Configuration has errors:[/bold red]")
            for issue in issues:
                console.print(f"  ✗ {issue}")
        
        if warnings:
            console.print("[bold yellow]Configuration warnings:[/bold yellow]")
            for warning in warnings:
                console.print(f"  ⚠ {warning}")
        
        if not issues:
            if warnings:
                warning("Configuration is valid with warnings")
            else:
                success("Configuration is valid!")
        else:
            error_exit("Configuration validation failed")
            
    except Exception as e:
        error_exit(f"Failed to validate configuration: {e}")


@config.command()
@click.option('--verbose', '-v', is_flag=True, help='Show exported variables')
@click.option('--all', '-a', is_flag=True, help='Export ALL variables including internal BSMITH_*')
def export(verbose: bool, all: bool) -> None:
    """Export configuration as environment variables.
    
    By default, only exports external tool variables (FINN_*, XILINX_*, etc).
    Use --all to include internal BSMITH_* variables.
    
    Args:
        verbose: Whether to show the exported variables
        all: Export all variables including internal ones
    """
    try:
        config = get_config()
        
        if all:
            # Export ALL variables including internal BSMITH_*
            if verbose:
                console.print("[yellow]Warning:[/yellow] Exporting ALL variables including internal BSMITH_*")
                console.print("This may cause configuration feedback loops if not used carefully.\n")
            
            env_dict = config.to_all_env_dict()
            for key, value in env_dict.items():
                if value is not None:
                    os.environ[key] = str(value)
                    if verbose:
                        console.print(f"[dim]Export {key}={value}[/dim]")
            
            count = len([v for v in env_dict.values() if v is not None])
            if not verbose:
                success(f"Exported {count} variables (including internal)")
                console.print("Use --verbose to see exported variables")
        else:
            # Default: use existing export_to_environment (external only)
            export_to_environment(config, verbose=verbose)
            
            if not verbose:
                # Count exported variables
                env_dict = config.to_external_env_dict()
                count = len([v for v in env_dict.values() if v is not None])
                success(f"Exported {count} external tool variables")
                console.print("Use --all to include internal BSMITH_* variables")
                console.print("Use --verbose to see exported variables")
            
    except Exception as e:
        error_exit(f"Failed to export configuration: {e}")


@config.command()
def where() -> None:
    """Show where each configuration value comes from.
    
    This helps debug configuration precedence issues by showing
    whether each value comes from CLI, environment, YAML, or defaults.
    """
    try:
        config = get_config()
        
        table = Table(title="Configuration Sources")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Source", style="yellow")
        
        # Get values and try to determine sources
        # Note: This is a simplified version - full source tracking would require
        # modifying the config loading to track sources
        
        import os
        from pathlib import Path
        
        # Check key paths
        settings = [
            ("bsmith_dir", config.bsmith_dir, "BSMITH_DIR"),
            ("build_dir", config.build_dir, "BSMITH_BUILD_DIR"),
            ("deps_dir", config.deps_dir, "BSMITH_DEPS_DIR"),
            ("xilinx_path", config.xilinx_path, "BSMITH_XILINX_PATH"),
            ("xilinx_version", config.xilinx_version, "BSMITH_XILINX_VERSION"),
            ("debug", config.debug, "BSMITH_DEBUG"),
            ("plugins_strict", config.plugins_strict, "BSMITH_PLUGINS_STRICT"),
        ]
        
        # Try to find YAML file
        yaml_file = None
        for possible in ["brainsmith_settings.yaml", ".brainsmith.yaml"]:
            if Path(possible).exists():
                yaml_file = Path(possible)
                break
        
        yaml_data = {}
        if yaml_file:
            try:
                yaml_data = yaml.safe_load(yaml_file.read_text())
            except:
                pass
        
        for setting_name, value, env_var in settings:
            source = "default"
            
            # Check environment
            if os.environ.get(env_var):
                source = f"env: {env_var}"
            # Check YAML
            elif yaml_data and setting_name in yaml_data:
                source = f"yaml: {yaml_file.name}"
            # Check if it's auto-detected (like bsmith_dir)
            elif setting_name == "bsmith_dir" and value:
                source = "auto-detected"
                
            table.add_row(setting_name, str(value) if value else "None", source)
        
        console.print(table)
        
        if yaml_file:
            console.print(f"\n[dim]Configuration file: {yaml_file.absolute()}[/dim]")
        else:
            console.print("\n[dim]No configuration file found in current directory[/dim]")
            
    except Exception as e:
        error_exit(f"Failed to analyze configuration: {e}")


@config.command()
def paths() -> None:
    """Show all resolved paths in the configuration.
    
    This is a convenience command to quickly see all paths
    and whether they exist.
    """
    try:
        config = get_config()
        
        table = Table(title="Configuration Paths")
        table.add_column("Path Type", style="cyan")
        table.add_column("Value", style="green") 
        table.add_column("Exists", style="yellow")
        table.add_column("Resolved", style="dim")
        
        # Helper to check and format paths
        def add_path(name: str, path: Any):
            if path is None:
                table.add_row(name, "Not set", "—", "—")
            else:
                path_obj = Path(path)
                exists = "✓" if path_obj.exists() else "✗"
                resolved = str(path_obj.absolute()) if not path_obj.is_absolute() else "—"
                table.add_row(name, str(path), exists, resolved)
        
        # Core paths
        table.add_row("[bold]Core Paths[/bold]", "", "", "")
        add_path("BSMITH_DIR", config.bsmith_dir)
        add_path("Build Directory", config.build_dir)
        add_path("Dependencies", config.deps_dir)
        
        # FINN paths
        table.add_row("", "", "", "")
        table.add_row("[bold]FINN Paths[/bold]", "", "", "")
        finn_root = config.finn.finn_root or config.deps_dir / "finn"
        add_path("FINN_ROOT", finn_root)
        finn_build = config.finn.finn_build_dir or config.build_dir
        add_path("FINN_BUILD_DIR", finn_build)
        
        # Xilinx paths
        table.add_row("", "", "", "")
        table.add_row("[bold]Xilinx Paths[/bold]", "", "", "")
        add_path("Xilinx Base", config.xilinx_path)
        add_path("Vivado", config.effective_vivado_path)
        add_path("Vitis", config.effective_vitis_path)
        add_path("Vitis HLS", config.effective_vitis_hls_path)
        
        # Other paths
        table.add_row("", "", "", "")
        table.add_row("[bold]Other Paths[/bold]", "", "", "")
        if config.vivado_ip_cache:
            add_path("Vivado IP Cache", config.vivado_ip_cache)
        add_path("Platform Repos", config.platform_repo_paths)
        
        console.print(table)
        
        # Show any path warnings
        if config.deps_dir and not config.deps_dir.is_absolute():
            console.print(
                "\n[yellow]Warning:[/yellow] deps_dir is relative. "
                "It will resolve differently when run from different directories."
            )
            
    except Exception as e:
        error_exit(f"Failed to show paths: {e}")


@config.command()
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default=Path("brainsmith_settings.yaml"),
              help='Output file path')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
@click.option('--minimal', is_flag=True, help='Create minimal config (backward compatibility)')
@click.option('--full', is_flag=True, help='Include all possible configuration fields')
def init(output: Path, force: bool, minimal: bool, full: bool) -> None:
    """Initialize a new configuration file with defaults.
    
    Args:
        output: Path to the output configuration file
        force: Whether to overwrite an existing file
        minimal: Create minimal config (old behavior)
        full: Include all possible fields
    """
    if output.exists() and not force:
        error_exit(f"{output} already exists. Use --force to overwrite.")
    
    try:
        # Load current config to get sensible defaults
        config = get_config()
        
        if minimal:
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
            config_dict["bsmith_dir"] = str(config.bsmith_dir)  # Auto-detected value
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
        console.print("Run 'smith config validate' to check your configuration.")
        
    except Exception as e:
        error_exit(f"Failed to create configuration: {e}")