"""Configuration management commands for brainsmith CLI."""

# Standard library imports
import logging
import os
from pathlib import Path
from typing import Any, Optional, Dict

# Third-party imports
import click
import yaml
from rich.table import Table
from rich.panel import Panel
from rich.console import Console as RichConsole

logger = logging.getLogger(__name__)

# Local imports
from brainsmith.config import BrainsmithConfig
from brainsmith.utils import dump_yaml
from ..context import ApplicationContext
from ..utils import console, error_exit, success
from ..exceptions import ConfigurationError, ValidationError
# ConfigFormatter inlined from formatters.py
# This class is only used by the config command


def validate_config_key(key: str) -> bool:
    """Validate that a configuration key is valid.
    
    Returns True if valid, raises ValidationError if not.
    """
    # Valid top-level keys from BrainsmithConfig
    valid_keys = {
        'build_dir', 'deps_dir', 'xilinx_path', 'xilinx_version',
        'vivado_path', 'vitis_path', 'vitis_hls_path',
        'vivado_ip_cache', 'platform_repo_paths', 'netron_port',
        'debug', 'verbose', 'plugins_strict'
    }
    
    # Valid nested keys
    valid_nested = {
        'finn.finn_root', 'finn.finn_build_dir', 'finn.finn_deps_dir',
        'finn.num_default_workers'
    }
    
    # Check if it's a valid key
    if key in valid_keys or key in valid_nested:
        return True
    
    # Check if it's a nested key we recognize
    if '.' in key:
        parts = key.split('.')
        if parts[0] == 'finn' and len(parts) == 2:
            # Could be a valid FINN key
            return True
    
    # Provide helpful error message
    all_keys = sorted(valid_keys | valid_nested)
    raise ValidationError(
        f"Invalid configuration key: '{key}'",
        details=[
            "Valid keys include:",
            *[f"  - {k}" for k in all_keys[:10]],  # Show first 10
            "...",
            "Run 'brainsmith config show --verbose' to see all available settings"
        ]
    )


class ConfigFormatter:
    """Formatter for displaying Brainsmith configuration."""
    
    def __init__(self, console=None):
        self.console = console or RichConsole()
    
    def format_table(self, config: BrainsmithConfig, verbose: bool = False) -> Table:
        """Format configuration as a Rich table.
        
        Args:
            config: The configuration to display
            verbose: Whether to include source information
            
        Returns:
            Formatted Rich table
        """
        if verbose:
            return self._create_verbose_table(config)
        else:
            return self._create_simple_table(config)
    
    def _create_simple_table(self, config: BrainsmithConfig) -> Table:
        """Create a simple configuration table."""
        table = Table(title="Brainsmith Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")  # No style to allow markup
        
        # Core paths section
        table.add_row("Core Paths", "")
        table.add_row("  Build Directory", self._format_path(config.build_dir, config.bsmith_dir))
        table.add_row("  Dependencies Directory", 
                      self._format_path(config.deps_dir, config.bsmith_dir, config.deps_dir))
        
        # Toolchain settings section
        table.add_row("", "")
        table.add_row("Toolchain Settings", "")
        table.add_row("  Plugins Strict", str(config.plugins_strict))
        table.add_row("  Debug Mode", str(config.debug))
        table.add_row("  Netron Port", str(config.netron_port))
        if config.vivado_ip_cache:
            table.add_row("  Vivado IP Cache", 
                          self._format_path(config.vivado_ip_cache, config.bsmith_dir))
        
        # Xilinx tools section
        self._add_xilinx_tools_section(table, config)
        
        return table
    
    def _create_verbose_table(self, config: BrainsmithConfig) -> Table:
        """Create a verbose configuration table with source information."""
        table = Table(title="Brainsmith Configuration (Verbose)")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_column("Source", style="yellow")
        
        # Core paths section
        table.add_row("Core Paths", "", "")
        table.add_row("  Build Directory", 
                      self._format_path(config.build_dir, config.bsmith_dir),
                      self._get_source("build_dir", "BSMITH_BUILD_DIR"))
        table.add_row("  Dependencies Directory",
                      self._format_path(config.deps_dir, config.bsmith_dir, config.deps_dir),
                      self._get_source("deps_dir", "BSMITH_DEPS_DIR"))
        
        # Toolchain settings section
        table.add_row("", "", "")
        table.add_row("Toolchain Settings", "", "")
        table.add_row("  Debug Mode", str(config.debug),
                      self._get_source("debug", "BSMITH_DEBUG"))
        table.add_row("  Plugins Strict", str(config.plugins_strict),
                      self._get_source("plugins_strict", "BSMITH_PLUGINS_STRICT"))
        table.add_row("  Netron Port", str(config.netron_port),
                      self._get_source("netron_port", "BSMITH_NETRON_PORT"))
        
        # Xilinx tools section with sources
        self._add_xilinx_tools_verbose_section(table, config)
        
        # FINN configuration section
        self._add_finn_section(table, config)
        
        return table
    
    def _add_xilinx_tools_section(self, table: Table, config: BrainsmithConfig) -> None:
        """Add Xilinx tools section to the table."""
        table.add_row("", "")
        table.add_row("Xilinx Tools", "")
        
        # Vivado
        vivado_display = self._format_xilinx_tool_path(
            config.effective_vivado_path, config.xilinx_path, 
            config.xilinx_version, "Vivado"
        )
        table.add_row("  Vivado", vivado_display)
        
        # Vitis
        vitis_display = self._format_xilinx_tool_path(
            config.effective_vitis_path, config.xilinx_path,
            config.xilinx_version, "Vitis"
        )
        table.add_row("  Vitis", vitis_display)
        
        # Vitis HLS
        vitis_hls_display = self._format_xilinx_tool_path(
            config.effective_vitis_hls_path, config.xilinx_path,
            config.xilinx_version, "Vitis_HLS"
        )
        table.add_row("  Vitis HLS", vitis_hls_display)
    
    def _add_xilinx_tools_verbose_section(self, table: Table, config: BrainsmithConfig) -> None:
        """Add Xilinx tools section with source information."""
        table.add_row("", "", "")
        table.add_row("Xilinx Tools", "", "")
        table.add_row("  Base Path", 
                      str(config.xilinx_path) if config.xilinx_path else "Not configured",
                      self._get_source("xilinx_path", "BSMITH_XILINX_PATH"))
        table.add_row("  Version", config.xilinx_version,
                      self._get_source("xilinx_version", "BSMITH_XILINX_VERSION"))
        
        # Add individual tools with sources
        for tool_name, path_attr, env_var in [
            ("Vivado", "effective_vivado_path", "BSMITH_VIVADO_PATH"),
            ("Vitis", "effective_vitis_path", "BSMITH_VITIS_PATH"),
            ("Vitis HLS", "effective_vitis_hls_path", "BSMITH_VITIS_HLS_PATH")
        ]:
            path = getattr(config, path_attr)
            if path:
                display = self._format_xilinx_tool_path(
                    path, config.xilinx_path, config.xilinx_version, 
                    tool_name.replace(" ", "_")
                )
                source = "derived" if not getattr(config, path_attr.replace("effective_", "")) else self._get_source(path_attr.replace("effective_", ""), env_var)
            else:
                display = "[yellow]Not found[/yellow]"
                source = "—"
            table.add_row(f"  {tool_name}", display, source)
    
    def _add_finn_section(self, table: Table, config: BrainsmithConfig) -> None:
        """Add FINN configuration section to verbose table."""
        table.add_row("", "", "")
        table.add_row("FINN Configuration", "", "")
        
        # FINN_ROOT
        finn_root = config.effective_finn_root
        finn_root_is_derived = config.finn.finn_root is None
        finn_root_original = Path("deps/finn") if finn_root_is_derived else config.finn.finn_root
        
        table.add_row("  FINN_ROOT", 
                      self._format_path(finn_root, config.bsmith_dir, finn_root_original),
                      "derived" if finn_root_is_derived else self._get_source("finn.finn_root", "BSMITH_FINN__FINN_ROOT"))
        
        # FINN_BUILD_DIR
        finn_build = config.effective_finn_build_dir
        finn_build_is_derived = config.finn.finn_build_dir is None
        table.add_row("  FINN_BUILD_DIR", 
                      self._format_path(finn_build, config.bsmith_dir),
                      "derived" if finn_build_is_derived else self._get_source("finn.finn_build_dir", "BSMITH_FINN__FINN_BUILD_DIR"))
        
        # FINN_DEPS_DIR
        finn_deps = config.effective_finn_deps_dir
        finn_deps_is_derived = config.finn.finn_deps_dir is None
        finn_deps_original = config.deps_dir if finn_deps_is_derived and finn_deps == config.deps_dir else config.finn.finn_deps_dir
        
        table.add_row("  FINN_DEPS_DIR", 
                      self._format_path(finn_deps, config.bsmith_dir, finn_deps_original),
                      "derived" if finn_deps_is_derived else self._get_source("finn.finn_deps_dir", "BSMITH_FINN__FINN_DEPS_DIR"))
        
        if config.finn.num_default_workers:
            table.add_row("  Default Workers", str(config.finn.num_default_workers),
                          self._get_source("finn.num_default_workers", "BSMITH_FINN__NUM_DEFAULT_WORKERS"))
    
    def _format_path(self, path: Optional[Path], base_path: Optional[Path] = None, 
                     original_value: Optional[Path] = None) -> str:
        """Format a path for display with color coding."""
        if path is None:
            return "None"
        
        path_obj = Path(path)
        
        # Determine the actual path to check for existence
        if not path_obj.is_absolute() and base_path:
            check_path = base_path / path_obj
        else:
            check_path = path_obj
        
        # Color based on existence
        color = "green" if check_path.exists() else "yellow"
        
        # Format the display
        if not path_obj.is_absolute() and base_path:
            return f"[dim]{base_path}/[/dim][{color}]{path}[/{color}]"
        
        # Use original_value if provided
        if original_value is not None and not Path(original_value).is_absolute():
            base_str = str(base_path) if base_path else str(path_obj.parent)
            rel_str = str(original_value)
            return f"[dim]{base_str}/[/dim][{color}]{rel_str}[/{color}]"
        
        # Try to show as relative to base_path
        if base_path and path_obj.is_absolute() and base_path.is_absolute():
            try:
                rel_path = path_obj.relative_to(base_path)
                return f"[dim]{base_path}/[/dim][{color}]{rel_path}[/{color}]"
            except ValueError:
                pass
        
        # Show as-is
        return f"[{color}]{path}[/{color}]"
    
    def _format_xilinx_tool_path(self, path: Optional[Path], base: Optional[Path],
                                 version: str, tool: str) -> str:
        """Format Xilinx tool path with base and version highlighted."""
        if path and path.exists():
            color = "green"
            return f"[{color}]{base}[/{color}][dim]/{tool}/[/dim][{color}]{version}[/{color}]"
        else:
            return "[yellow]Not found[/yellow]"
    
    def _get_source(self, setting_name: str, env_var: str) -> str:
        """Determine the source of a configuration value."""
        if os.environ.get(env_var):
            return f"env: {env_var}"
        
        # Check for YAML files
        yaml_file = self._check_yaml_files(setting_name)
        if yaml_file:
            return f"yaml: {yaml_file}"
        
        return "default"
    
    def _check_yaml_files(self, setting_name: str) -> Optional[str]:
        """Check if setting exists in YAML files."""
        for filename in ["brainsmith_settings.yaml", ".brainsmith.yaml"]:
            yaml_path = Path(filename)
            if yaml_path.exists():
                try:
                    with open(yaml_path) as f:
                        data = yaml.safe_load(f)
                        if data and setting_name in data:
                            return filename
                except:
                    pass
        return None
    
    def show_validation_warnings(self, config: BrainsmithConfig) -> None:
        """Display validation warnings for the configuration."""
        warnings = []
        
        if config.deps_dir and not config.deps_dir.is_absolute():
            expected = config.bsmith_dir / config.deps_dir
            if config.deps_dir.absolute() != expected.absolute():
                warnings.append("Relative deps_dir may not resolve correctly")
        
        if warnings:
            self.console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in warnings:
                self.console.print(f"  ⚠ {warning}")


@click.group()
def config():
    """Manage Brainsmith configuration."""
    pass


@config.command()
@click.option('--verbose', '-v', is_flag=True, help='Include source information and path validation')
@click.pass_obj
def show(ctx: ApplicationContext, verbose: bool) -> None:
    """Display current configuration.
    
    Shows the effective configuration after merging all sources:
    user defaults, project settings, environment variables, and CLI overrides.
    """
    try:
        logger.debug(f"Showing config with verbose={verbose}")
        config = ctx.get_effective_config()
        formatter = ConfigFormatter(console)
        
        table = formatter.format_table(config, verbose=verbose)
        console.print(table)
        if verbose:
            formatter.show_validation_warnings(config)
                
    except Exception as e:
        error_exit(f"Failed to load configuration: {e}")





@config.command()
@click.argument('key')
@click.argument('value')
@click.pass_obj
def set(ctx: ApplicationContext, key: str, value: str) -> None:
    """Set a configuration value in user defaults.
    
    \b
    Examples:
      brainsmith config set verbose true
      brainsmith config set build_dir /opt/builds
      brainsmith config set finn.num_default_workers 8
    """
    try:
        logger.debug(f"Setting config key={key}, value={value}")
        
        # Validate the key first
        validate_config_key(key)
        
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
            # Validate path format
            if value.startswith('~'):
                typed_value = value  # expanduser will handle it
            elif not value:
                raise ValidationError("Path cannot be empty")
            else:
                typed_value = value  # Keep as string, will be converted to Path by config
        
        # Set the value
        ctx.set_user_config_value(key, typed_value)
        
        success(f"Set {key} = {value}")
        
        # Show where it was saved
        console.print(f"Saved to: {ctx.user_config_path}")
        
    except ValidationError as e:
        # ValidationError already has proper details
        error_exit(e.message, details=e.details)
    except Exception as e:
        error_exit(f"Failed to set configuration: {e}",
                  details=[
                      "Check that the key is valid",
                      "Ensure the value is in the correct format",
                      "Run 'brainsmith config show --verbose' to see available keys"
                  ])


@config.command()
@click.option('--shell', type=click.Choice(['bash', 'zsh', 'fish', 'powershell']),
              default='bash', help='Shell format for output')
@click.pass_obj
def export(ctx: ApplicationContext, shell: str) -> None:
    """Export configuration as shell environment script.
    
    \b
    Usage:
      eval $(brainsmith config export)
      eval $(brainsmith config export --shell fish)
    """
    try:
        activation_script = ctx.export_environment(shell)
        
        # Print to stdout for eval
        click.echo(activation_script)
        
        # Add helpful comment based on shell
        if shell in ['bash', 'zsh', 'sh']:
            click.echo("# Run: eval $(brainsmith config export)")
        elif shell == 'fish':
            click.echo("# Run: eval (brainsmith config export --shell fish)")
        elif shell == 'powershell':
            click.echo("# Run: brainsmith config export --shell powershell | Invoke-Expression")
            
    except ValueError as e:
        error_exit(f"Unsupported shell: {e}")
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
            # Default: simple config matching brainsmith_settings.yaml format
            config_dict = {
                "__comment__": "Brainsmith Project Settings",
                "build_dir": "${HOME}/.brainsmith/builds",
                "xilinx_path": "/tools/Xilinx",
                "xilinx_version": "2024.2",
                "plugins_strict": True,
                "debug": False
            }
        
        # Write YAML with proper formatting
        if full:
            # Use standardized YAML dumper for full config
            dump_yaml(config_dict, output, sort_keys=False)
        else:
            # Write formatted YAML matching the target style
            lines = []
            
            # Add header comment
            lines.append("# Brainsmith Project Settings")
            lines.append("")
            
            # Build directory comment and value
            lines.append("# Build directory for compilation artifacts")
            lines.append(f"build_dir: {config_dict['build_dir']}")
            lines.append("")
            
            # Xilinx tool configuration
            lines.append("# Xilinx tool configuration")
            lines.append(f"xilinx_path: {config_dict['xilinx_path']}")
            lines.append(f'xilinx_version: "{config_dict["xilinx_version"]}"')
            lines.append("")
            
            # Debug settings
            lines.append("# Debug settings")
            lines.append(f"plugins_strict: {str(config_dict['plugins_strict']).lower()}")
            lines.append(f"debug: {str(config_dict['debug']).lower()}")
            
            with open(output, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        
        success(f"Created configuration file: {output}")
        console.print("\nYou can now edit this file to customize your settings.")
        if user:
            console.print("These settings will be used as defaults for all Brainsmith projects.")
        else:
            console.print("These settings will be used for this project.")
        
    except Exception as e:
        error_exit(f"Failed to create configuration: {e}")