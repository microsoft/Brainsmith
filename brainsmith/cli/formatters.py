# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations  # PEP 563: Postponed evaluation of annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from rich.table import Table
from rich.panel import Panel
from rich.console import Console as RichConsole

from .messages import XILINX_NOT_CONFIGURED, XILINX_NOT_FOUND

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

    def _format_metadata_section(self, config: SystemConfig) -> Panel:
        """Format configuration metadata as Rich panel.

        Shows:
        - Project directory (where .brainsmith/config.yaml is located)
        - Brainsmith directory (where brainsmith package is installed)
        - Environment status (direnv, venv)
        - Component manifest status (cache enabled, file exists)

        Args:
            config: System configuration

        Returns:
            Rich Panel with metadata
        """
        lines = []

        # Project directory
        project_dir = config.project_dir
        lines.append(f"[cyan]Project directory:[/cyan]    {project_dir}")

        # Brainsmith directory
        bsmith_dir = config.bsmith_dir
        lines.append(f"[cyan]Brainsmith directory:[/cyan] {bsmith_dir}")

        # Environment status
        env_parts = []

        # Check direnv
        if os.environ.get('DIRENV_DIR'):
            env_parts.append("[green]direnv active[/green]")
        else:
            env_parts.append("[dim]direnv inactive[/dim]")

        # Check venv
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            venv_name = Path(venv_path).name
            env_parts.append(f"venv: [green]{venv_name}[/green]")
        else:
            env_parts.append("[dim]no venv[/dim]")

        env_status = ", ".join(env_parts)
        lines.append(f"[cyan]Environment:[/cyan]          {env_status}")

        # Component manifest status
        cache_parts = []

        if config.cache_components:
            cache_parts.append("[green]enabled[/green]")

            # Check if manifest file exists
            manifest_path = config.project_dir / '.brainsmith' / 'component_manifest.json'
            if manifest_path.exists():
                cache_parts.append("[green]manifest found[/green]")
            else:
                cache_parts.append("[yellow]no manifest[/yellow]")
        else:
            cache_parts.append("[dim]disabled[/dim]")

        cache_status = ", ".join(cache_parts)
        lines.append(f"[cyan]Component cache:[/cyan]      {cache_status}")

        # Create panel with all metadata
        content = "\n".join(lines)
        return Panel(content, title="Configuration Metadata", border_style="cyan")

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
                      self._get_source("build_dir", "BSMITH_BUILD_DIR", config))
        table.add_row("  Dependencies Directory",
                      self._format_path(config.deps_dir, config.bsmith_dir),
                      self._get_source("deps_dir", "BSMITH_DEPS_DIR", config))

        table.add_row("", "", "")
        table.add_row("Component Registry", "", "")

        table.add_row("  Source Priority", ", ".join(config.source_priority),
                      self._get_source("source_priority", "BSMITH_SOURCE_PRIORITY", config))

        # Component sources - show configured filesystem sources (project, user, custom)
        # Core namespace (brainsmith) and entry points (finn) are not shown here as they're not configurable
        source = self._get_source("component_sources", "BSMITH_COMPONENT_SOURCES", config)
        for i, (source_name, source_path) in enumerate(sorted(config.component_sources.items())):
            label = "  Component Sources" if i == 0 else ""
            formatted_path = self._format_path(source_path, config.bsmith_dir)
            display_value = f"{source_name}: {formatted_path}"
            row_source = source if i == 0 else ""
            table.add_row(label, display_value, row_source)

        table.add_row("  Components Strict", str(config.components_strict),
                      self._get_source("components_strict", "BSMITH_COMPONENTS_STRICT", config))

        table.add_row("", "", "")
        table.add_row("Toolchain Settings", "", "")
        if config.default_workers:
            table.add_row("  Default Workers", str(config.default_workers),
                          self._get_source("default_workers", "BSMITH_DEFAULT_WORKERS", config))
        table.add_row("  Netron Port", str(config.netron_port),
                      self._get_source("netron_port", "BSMITH_NETRON_PORT", config))

        self._add_xilinx_tools_section(table, config)

        if include_finn:
            self._add_finn_section(table, config)

        return table

    def _add_xilinx_tools_section(self, table: Table, config: SystemConfig) -> None:
        table.add_row("", "", "")
        table.add_row("Xilinx Tools", "", "")
        table.add_row("  Base Path",
                      str(config.xilinx_path) if config.xilinx_path else XILINX_NOT_CONFIGURED,
                      self._get_source("xilinx_path", "BSMITH_XILINX_PATH", config))
        table.add_row("  Version", config.xilinx_version,
                      self._get_source("xilinx_version", "BSMITH_XILINX_VERSION", config))

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
                source = self._get_source(path_attr, env_var, config)
            else:
                display = f"[yellow]{XILINX_NOT_FOUND}[/yellow]"
                source = "—"
            table.add_row(f"  {tool_name}", display, source)

    def _add_finn_section(self, table: Table, config: SystemConfig) -> None:
        table.add_row("", "", "")
        table.add_row("FINN Configuration", "", "")

        finn_build = config.finn_build_dir
        table.add_row("  FINN_BUILD_DIR",
                      self._format_path(finn_build, config.bsmith_dir),
                      self._get_source("finn_build_dir", "BSMITH_FINN_BUILD_DIR", config))

        finn_deps = config.finn_deps_dir
        table.add_row("  FINN_DEPS_DIR",
                      self._format_path(finn_deps, config.bsmith_dir),
                      self._get_source("finn_deps_dir", "BSMITH_FINN_DEPS_DIR", config))

        finn_root = config.finn_root
        table.add_row("  FINN_ROOT",
                      self._format_path(finn_root, config.bsmith_dir),
                      self._get_source("finn_root", "BSMITH_FINN_ROOT", config))

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
            return f"[yellow]{XILINX_NOT_CONFIGURED}[/yellow]"

        if not path.exists():
            return f"[yellow]{XILINX_NOT_FOUND}[/yellow]"

        # Show as: base/tool/version
        return f"[green]{base}[/green][dim]/{tool}/[/dim][green]{version}[/green]"

    def _get_source(self, setting_name: str, env_var: str, config: SystemConfig | None = None) -> str:
        """Get simplified source string for a configuration setting.

        Uses Pydantic's model_fields_set to detect if a field was explicitly configured,
        combined with environment variable checking to distinguish env from yaml sources.

        Args:
            setting_name: Name of the setting (e.g., 'vivado_path')
            env_var: Environment variable name (e.g., 'BSMITH_VIVADO_PATH')
            config: System configuration (needed to detect derived values and field metadata)

        Returns:
            Source string: "env", "yaml", "derived", or "default"
        """
        # Check environment variable first (highest priority)
        if os.environ.get(env_var):
            return "env"

        # Use Pydantic's model_fields_set to detect explicit configuration
        # If field was set (and not from env), it came from yaml file
        if config and setting_name in config.model_fields_set:
            return "yaml"

        # Check if this is an auto-derived Xilinx tool path
        if config and setting_name in ('vivado_path', 'vitis_path', 'vitis_hls_path'):
            # If the tool path exists but wasn't set via env or yaml, it was auto-derived
            tool_path = getattr(config, setting_name, None)
            if tool_path is not None:
                return _SOURCE_DERIVED

        return _SOURCE_DEFAULT

    def show_validation_warnings(self, config: SystemConfig) -> None:
        """Display configuration validation warnings."""
        from brainsmith.settings.validation import get_config_warnings

        warnings = get_config_warnings(config)
        if warnings:
            self.console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in warnings:
                self.console.print(f"  ⚠️ {warning}")
