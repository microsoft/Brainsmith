# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations  # PEP 563: Postponed evaluation of annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from rich.table import Table
from rich.console import Console as RichConsole

from brainsmith.settings.constants import PROJECT_CONFIG_FILE, PROJECT_CONFIG_FILE_ALT

# Lazy import settings
if TYPE_CHECKING:
    from brainsmith.settings import SystemConfig

# Source constants for configuration display
_SOURCE_DERIVED = "derived"
_SOURCE_DEFAULT = "default"


class ConfigFormatter:
    """Formatter for displaying Brainsmith configuration."""

    def __init__(self, console: RichConsole | None = None):
        self.console = console or RichConsole()
        self._yaml_cache: dict[str, dict] = {}  # Cache parsed YAML files
    
    def format_table(self, config: SystemConfig, include_finn: bool = False) -> Table:
        """Format configuration as Rich table with source information.

        Args:
            config: System configuration
            include_finn: Include FINN-specific settings

        Returns:
            Rich table with configuration details
        """
        table = Table(title="Brainsmith Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_column("Source", style="yellow")

        table.add_row("Core Paths", "", "")
        table.add_row("  Build Directory",
                      self._format_path(config.build_dir, config.bsmith_dir),
                      self._get_source("build_dir", "BSMITH_BUILD_DIR"))
        table.add_row("  Dependencies Directory",
                      self._format_path(config.deps_dir, config.bsmith_dir),
                      self._get_source("deps_dir", "BSMITH_DEPS_DIR"))

        table.add_row("", "", "")
        table.add_row("Plugin Settings", "", "")

        table.add_row("  Default Source", config.default_source,
                      self._get_source("default_source", "BSMITH_DEFAULT_SOURCE"))

        # Plugin sources - show each source with its path (exclude brainsmith)
        plugin_sources = {k: v for k, v in config.plugin_sources.items() if k != 'brainsmith'}
        source = self._get_source("plugin_sources", "BSMITH_PLUGIN_SOURCES")
        for i, (source_name, source_path) in enumerate(sorted(plugin_sources.items())):
            label = "  Plugin Sources" if i == 0 else ""
            formatted_path = self._format_path(source_path, config.bsmith_dir)
            display_value = f"{source_name}: {formatted_path}"
            row_source = source if i == 0 else ""
            table.add_row(label, display_value, row_source)

        table.add_row("  Plugins Strict", str(config.plugins_strict),
                      self._get_source("plugins_strict", "BSMITH_PLUGINS_STRICT"))

        table.add_row("", "", "")
        table.add_row("Toolchain Settings", "", "")
        if config.default_workers:
            table.add_row("  Default Workers", str(config.default_workers),
                          self._get_source("default_workers", "BSMITH_DEFAULT_WORKERS"))
        table.add_row("  Netron Port", str(config.netron_port),
                      self._get_source("netron_port", "BSMITH_NETRON_PORT"))

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
            ("Vivado", "vivado_path", "BSMITH_VIVADO_PATH"),
            ("Vitis", "vitis_path", "BSMITH_VITIS_PATH"),
            ("Vitis HLS", "vitis_hls_path", "BSMITH_VITIS_HLS_PATH")
        ]:
            path = getattr(config, path_attr)
            if path:
                display = self._format_xilinx_tool_path(
                    path, config.xilinx_path, config.xilinx_version,
                    tool_name.replace(" ", "_")
                )
                source = self._get_source(path_attr, env_var)
            else:
                display = "[yellow]Not found[/yellow]"
                source = "—"
            table.add_row(f"  {tool_name}", display, source)
    
    def _add_finn_section(self, table: Table, config: SystemConfig) -> None:
        table.add_row("", "", "")
        table.add_row("FINN Configuration", "", "")

        finn_build = config.finn.finn_build_dir
        table.add_row("  FINN_BUILD_DIR",
                      self._format_path(finn_build, config.bsmith_dir),
                      self._get_source("finn.finn_build_dir", "BSMITH_FINN__FINN_BUILD_DIR"))

        finn_deps = config.finn.finn_deps_dir
        table.add_row("  FINN_DEPS_DIR",
                      self._format_path(finn_deps, config.bsmith_dir),
                      self._get_source("finn.finn_deps_dir", "BSMITH_FINN__FINN_DEPS_DIR"))

        finn_root = config.finn.finn_root
        table.add_row("  FINN_ROOT",
                      self._format_path(finn_root, config.bsmith_dir),
                      self._get_source("finn.finn_root", "BSMITH_FINN__FINN_ROOT"))

    def _format_path(self, path: Path | None, base_path: Path | None = None) -> str:
        if not path:
            return "[dim]not set[/dim]"

        color = "green" if path.exists() else "yellow"

        # Always show full absolute path
        display = path.absolute() if not path.is_absolute() else path

        return f"[{color}]{display}[/{color}]"

    def _format_xilinx_tool_path(
        self,
        path: Path | None,
        base: Path | None,
        version: str,
        tool: str
    ) -> str:
        """Format Xilinx tool path with color based on existence."""
        if not path:
            return "[yellow]Not configured[/yellow]"

        if not path.exists():
            return "[yellow]Not found[/yellow]"

        # Show as: base/tool/version
        return f"[green]{base}[/green][dim]/{tool}/[/dim][green]{version}[/green]"
    
    def _get_source(self, setting_name: str, env_var: str) -> str:
        if os.environ.get(env_var):
            return f"env: {env_var}"

        yaml_file = self._check_yaml_files(setting_name)
        if yaml_file:
            return f"yaml: {yaml_file}"

        return _SOURCE_DEFAULT
    
    def _check_yaml_files(self, setting_name: str) -> str | None:
        """Check if setting exists in project YAML files. Supports nested paths."""
        for filename in [PROJECT_CONFIG_FILE, PROJECT_CONFIG_FILE_ALT]:
            # Cache parsed YAML
            if filename not in self._yaml_cache:
                yaml_path = Path(filename)
                if not yaml_path.exists():
                    continue
                try:
                    with open(yaml_path) as f:
                        self._yaml_cache[filename] = yaml.safe_load(f) or {}
                except (OSError, yaml.YAMLError):
                    self._yaml_cache[filename] = {}

            data = self._yaml_cache[filename]
            if self._nested_key_exists(data, setting_name):
                return filename
        return None

    def _nested_key_exists(self, data: dict, key: str) -> bool:
        """Supports nested notation (e.g., 'finn.finn_root')."""
        parts = key.split('.')
        current = data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True
    
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
