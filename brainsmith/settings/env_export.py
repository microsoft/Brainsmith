# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Environment variable export for external tool integration.

This module provides the EnvironmentExporter class which handles all environment
variable export logic that was previously in SystemConfig.export_to_environment().
This improves testability and separates infrastructure concerns from configuration.
"""

import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Callable, Optional

if TYPE_CHECKING:
    from .schema import SystemConfig


# Declarative environment variable export mappings
# Maps environment variable names to functions that extract values from config
#
# Xilinx Tool Path Variables:
# Both XILINX_* and *_PATH variants are exported for maximum FINN compatibility.
# - XILINX_* variants: Used by FINN's Python runtime and internal scripts
# - *_PATH variants: Used by FINN's TCL scripts during Vivado/Vitis integration
# Both naming conventions must be set for full compatibility with FINN's toolchain.
EXTERNAL_ENV_MAPPINGS: Dict[str, Callable[['SystemConfig'], Optional[str]]] = {
    # Xilinx tool paths (both naming conventions required for FINN compatibility)
    'XILINX_VIVADO': lambda c: str(c.vivado_path) if c.vivado_path else None,
    'VIVADO_PATH': lambda c: str(c.vivado_path) if c.vivado_path else None,  # FINN TCL scripts
    'XILINX_VITIS': lambda c: str(c.vitis_path) if c.vitis_path else None,
    'VITIS_PATH': lambda c: str(c.vitis_path) if c.vitis_path else None,      # FINN TCL scripts
    'XILINX_HLS': lambda c: str(c.vitis_hls_path) if c.vitis_hls_path else None,
    'HLS_PATH': lambda c: str(c.vitis_hls_path) if c.vitis_hls_path else None, # FINN TCL scripts

    # Platform and tool paths
    'PLATFORM_REPO_PATHS': lambda c: c.vendor_platform_paths,
    'OHMYXILINX': lambda c: str(c.deps_dir / "oh-my-xilinx"),

    # Vivado specific
    'VIVADO_IP_CACHE': lambda c: str(c.vivado_ip_cache) if c.vivado_path else None,

    # Visualization
    'NETRON_PORT': lambda c: str(c.netron_port),

    # FINN environment variables
    'FINN_ROOT': lambda c: str(c.finn.finn_root),
    'FINN_BUILD_DIR': lambda c: str(c.finn.finn_build_dir),
    'FINN_DEPS_DIR': lambda c: str(c.finn.finn_deps_dir),
    'NUM_DEFAULT_WORKERS': lambda c: str(c.default_workers) if c.default_workers else None,
}


class EnvironmentExporter:
    """Handles environment variable export for external tools.

    Exports configuration to environment variables consumed by:
    - FINN (FINN_ROOT, FINN_BUILD_DIR, etc.)
    - Xilinx tools (VIVADO_PATH, XILINX_VIVADO, etc.)
    - Visualization tools (NETRON_PORT)

    Does NOT export internal BSMITH_* variables by default to prevent
    configuration feedback loops.

    Example:
        >>> config = SystemConfig()
        >>> exporter = EnvironmentExporter(config)
        >>> env_dict = exporter.to_external_dict()
        >>> print(env_dict['FINN_ROOT'])
    """

    @staticmethod
    def _add_to_path_if_needed(
        path_components: list[str],
        path: Path | str | None,
        check_exists: bool = False
    ) -> None:
        """Add path to PATH components if not already present."""
        if not path:
            return

        path_str = str(path)

        if check_exists and isinstance(path, Path) and not path.exists():
            return

        if path_str not in path_components:
            path_components.append(path_str)

    def __init__(self, config: 'SystemConfig'):
        """Initialize environment exporter with configuration.

        Args:
            config: SystemConfig instance to export from
        """
        self.config = config

    def to_external_dict(self) -> Dict[str, str]:
        """Generate dict of external environment variables.

        Returns only variables consumed by external tools (FINN, Xilinx, etc).
        Internal BSMITH_* variables are excluded to prevent configuration loops.

        Returns:
            Dict of environment variable names to string values
        """
        env_dict = {}

        # Apply all mappings
        for env_var, getter in EXTERNAL_ENV_MAPPINGS.items():
            value = getter(self.config)
            if value is not None:
                env_dict[env_var] = value

        return env_dict

    def to_all_dict(self) -> Dict[str, str]:
        """Generate dict of ALL environment variables including internal ones.

        WARNING: Includes internal BSMITH_* variables. Use only when needed.

        Returns:
            Dict of all environment variables
        """
        env_dict = self.to_external_dict()

        # Add internal BSMITH variables
        env_dict['BSMITH_BUILD_DIR'] = str(self.config.build_dir)
        env_dict['BSMITH_DEPS_DIR'] = str(self.config.deps_dir)
        env_dict['BSMITH_DIR'] = str(self.config.bsmith_dir)
        env_dict['BSMITH_PROJECT_DIR'] = str(self.config.project_dir)

        return env_dict

    def export_to_environment(
        self,
        include_internal: bool = False,
        verbose: bool = False,
        export: bool = True
    ) -> Dict[str, str]:
        """Export configuration to environment variables.

        This is the unified method for exporting configuration to the environment.
        By default, exports only external tool configuration values (FINN_*, XILINX_*, etc)
        and sets up PATH, PYTHONPATH, and LD_LIBRARY_PATH for tool compatibility.

        Args:
            include_internal: If True, also export internal BSMITH_* variables
                            (WARNING: may cause configuration feedback loops)
            verbose: Whether to print export information
            export: If False, only return dict without modifying os.environ

        Returns:
            Dict of exported environment variables
        """
        # Get environment variables based on what's requested
        if include_internal:
            env_dict = self.to_all_dict()
        else:
            env_dict = self.to_external_dict()

        # Handle PATH updates
        path_components = os.environ.get("PATH", "").split(":")

        # Add oh-my-xilinx to PATH if it exists
        self._add_to_path_if_needed(path_components, self.config.deps_dir / "oh-my-xilinx", check_exists=True)

        # Add ~/.local/bin to PATH
        self._add_to_path_if_needed(path_components, Path.home() / ".local" / "bin")

        # Add Xilinx tool bin directories to PATH
        if self.config.vivado_path:
            self._add_to_path_if_needed(path_components, self.config.vivado_path / "bin")

        if self.config.vitis_path:
            self._add_to_path_if_needed(path_components, self.config.vitis_path / "bin")

        if self.config.vitis_hls_path:
            self._add_to_path_if_needed(path_components, self.config.vitis_hls_path / "bin")

        env_dict["PATH"] = ":".join(path_components)

        # FINN XSI no longer requires PYTHONPATH manipulation
        # The new finn.xsi module handles path management internally

        # Handle LD_LIBRARY_PATH updates
        ld_lib_components = os.environ.get("LD_LIBRARY_PATH", "").split(":")

        # Add libudev if needed and exists for Xilinx tool compatibility
        libudev_path = "/lib/x86_64-linux-gnu/libudev.so.1"
        if self.config.vivado_path and Path(libudev_path).exists():
            env_dict["LD_PRELOAD"] = libudev_path

        # Add Vivado libraries
        if self.config.vivado_path:
            # Add architecture-specific system library path
            arch = platform.machine()
            if arch == 'x86_64':
                ld_lib_components.append("/lib/x86_64-linux-gnu/")
            elif arch in ('aarch64', 'arm64'):
                ld_lib_components.append("/lib/aarch64-linux-gnu/")

            vivado_lib = str(self.config.vivado_path / "lib" / "lnx64.o")
            ld_lib_components.append(vivado_lib)

        # Add Vitis FPO libraries
        if self.config.vitis_path:
            vitis_fpo_lib = str(self.config.vitis_path / "lnx64" / "tools" / "fpo_v7_1")
            if vitis_fpo_lib not in ld_lib_components:
                ld_lib_components.append(vitis_fpo_lib)

        env_dict["LD_LIBRARY_PATH"] = ":".join(filter(None, ld_lib_components))

        # Set Xilinx environment variables for better caching behavior
        # The actual HOME override is handled at container level in entrypoint scripts
        if self.config.vivado_path:
            # Ensure XILINX_LOCAL_USER_DATA is set to prevent network operations
            env_dict["XILINX_LOCAL_USER_DATA"] = "no"

        # Apply all environment variables only if export=True
        if export:
            for key, value in env_dict.items():
                if value is not None:
                    os.environ[key] = str(value)
                    if verbose and key not in ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH"]:
                        from rich.console import Console
                        console = Console()
                        console.print(f"[dim]Export {key}={value}[/dim]")

            if verbose:
                from rich.console import Console
                console = Console()
                console.print("[green]✓ Configuration exported to environment[/green]")

        return env_dict
