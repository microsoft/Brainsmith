# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path

import yaml
from rich.table import Table
from rich.console import Console as RichConsole

from brainsmith.settings import SystemConfig
from .constants import PROJECT_CONFIG_FILE, PROJECT_CONFIG_FILE_ALT

# Source constants for configuration display
_SOURCE_DERIVED = "derived"
_SOURCE_DEFAULT = "default"


class ConfigFormatter:
    """Formatter for displaying Brainsmith configuration."""

    def __init__(self, console: RichConsole | None = None):
        self.console = console or RichConsole()
    
    def format_table(self, config: SystemConfig, finn: bool = False) -> Table:
        """Format configuration as Rich table with source information.

        Args:
            config: System configuration
            finn: Include FINN-specific settings

        Returns:
            Rich table with configuration details (always detailed view with sources)
        """
        return self._create_detailed_table(config, include_finn=finn)
    
    def _create_detailed_table(self, config: SystemConfig, include_finn: bool = False) -> Table:
        """Create detailed configuration table with sources."""
        table = Table(title="Brainsmith Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_column("Source", style="yellow")

        table.add_row("Core Paths", "", "")
        table.add_row("  Build Directory",
                      self._format_path(config.build_dir, config.bsmith_dir),
                      self._get_source("build_dir", "BSMITH_BUILD_DIR"))
        table.add_row("  Dependencies Directory",
                      self._format_path(config.deps_dir, config.bsmith_dir, config.deps_dir),
                      self._get_source("deps_dir", "BSMITH_DEPS_DIR"))

        table.add_row("", "", "")
        table.add_row("Toolchain Settings", "", "")
        table.add_row("  Plugins Strict", str(config.plugins_strict),
                      self._get_source("plugins_strict", "BSMITH_PLUGINS_STRICT"))
        table.add_row("  Netron Port", str(config.netron_port),
                      self._get_source("netron_port", "BSMITH_NETRON_PORT"))
        if config.default_workers:
            table.add_row("  Default Workers", str(config.default_workers),
                          self._get_source("default_workers", "BSMITH_DEFAULT_WORKERS"))

        self._add_xilinx_tools_section(table, config)

        if include_finn:
            self._add_finn_section(table, config)

        return table

    def _add_xilinx_tools_section(self, table: Table, config: SystemConfig) -> None:
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
                source = _SOURCE_DERIVED if not getattr(config, path_attr.replace("effective_", "")) else self._get_source(path_attr.replace("effective_", ""), env_var)
            else:
                display = "[yellow]Not found[/yellow]"
                source = "—"
            table.add_row(f"  {tool_name}", display, source)
    
    def _add_finn_section(self, table: Table, config: SystemConfig) -> None:
        table.add_row("", "", "")
        table.add_row("FINN Configuration", "", "")

        finn_root = config.effective_finn_root
        finn_root_is_derived = config.finn.finn_root is None
        finn_root_original = Path("deps/finn") if finn_root_is_derived else config.finn.finn_root

        table.add_row("  FINN_ROOT",
                      self._format_path(finn_root, config.bsmith_dir, finn_root_original),
                      _SOURCE_DERIVED if finn_root_is_derived else self._get_source("finn.finn_root", "BSMITH_FINN__FINN_ROOT"))

        finn_build = config.effective_finn_build_dir
        finn_build_is_derived = config.finn.finn_build_dir is None
        table.add_row("  FINN_BUILD_DIR",
                      self._format_path(finn_build, config.bsmith_dir),
                      _SOURCE_DERIVED if finn_build_is_derived else self._get_source("finn.finn_build_dir", "BSMITH_FINN__FINN_BUILD_DIR"))

        finn_deps = config.effective_finn_deps_dir
        finn_deps_is_derived = config.finn.finn_deps_dir is None
        finn_deps_original = config.deps_dir if finn_deps_is_derived and finn_deps == config.deps_dir else config.finn.finn_deps_dir

        table.add_row("  FINN_DEPS_DIR",
                      self._format_path(finn_deps, config.bsmith_dir, finn_deps_original),
                      _SOURCE_DERIVED if finn_deps_is_derived else self._get_source("finn.finn_deps_dir", "BSMITH_FINN__FINN_DEPS_DIR"))

    def _format_path(
        self,
        path: Path | None,
        base_path: Path | None = None,
        original_value: Path | None = None
    ) -> str:
        if path is None:
            return "None"

        path_obj = Path(path)

        if not path_obj.is_absolute() and base_path:
            check_path = base_path / path_obj
        else:
            check_path = path_obj

        color = "green" if check_path.exists() else "yellow"

        if not path_obj.is_absolute() and base_path:
            return f"[dim]{base_path}/[/dim][{color}]{path_obj}[/{color}]"

        if original_value is not None and not Path(original_value).is_absolute():
            base_str = str(base_path) if base_path else str(path_obj.parent)
            return f"[dim]{base_str}/[/dim][{color}]{original_value}[/{color}]"

        if base_path and path_obj.is_absolute():
            try:
                rel_path = path_obj.relative_to(base_path)
                return f"[dim]{base_path}/[/dim][{color}]{rel_path}[/{color}]"
            except ValueError:
                pass

        return f"[{color}]{path_obj}[/{color}]"

    def _format_xilinx_tool_path(
        self,
        path: Path | None,
        base: Path | None,
        version: str,
        tool: str
    ) -> str:
        if path and path.exists():
            color = "green"
            return f"[{color}]{base}[/{color}][dim]/{tool}/[/dim][{color}]{version}[/{color}]"
        else:
            return "[yellow]Not found[/yellow]"
    
    def _get_source(self, setting_name: str, env_var: str) -> str:
        if os.environ.get(env_var):
            return f"env: {env_var}"

        yaml_file = self._check_yaml_files(setting_name)
        if yaml_file:
            return f"yaml: {yaml_file}"

        return _SOURCE_DEFAULT
    
    def _check_yaml_files(self, setting_name: str) -> str | None:
        for filename in [PROJECT_CONFIG_FILE, PROJECT_CONFIG_FILE_ALT]:
            yaml_path = Path(filename)
            if yaml_path.exists():
                try:
                    with open(yaml_path) as f:
                        data = yaml.safe_load(f)
                        if data and setting_name in data:
                            return filename
                except (OSError, yaml.YAMLError):
                    pass
        return None
    
    def show_validation_warnings(self, config: SystemConfig) -> None:
        warnings = []
        
        if config.deps_dir and not config.deps_dir.is_absolute():
            expected = config.bsmith_dir / config.deps_dir
            if config.deps_dir.absolute() != expected.absolute():
                warnings.append("Relative deps_dir may not resolve correctly")
        
        if warnings:
            self.console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in warnings:
                self.console.print(f"  ⚠ {warning}")
