# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Environment variable dictionary generation for shell script creation.

This module provides the EnvironmentExporter class which builds environment
variable dictionaries for shell script generation (env.sh, .env, .envrc).

The generated dictionaries are used to create activation scripts, but do NOT
directly mutate os.environ. Python runtime expects environment to be sourced
via shell scripts before Python starts.
"""

import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .schema import SystemConfig

# System library paths
_LIBUDEV_PATH = "/lib/x86_64-linux-gnu/libudev.so.1"


class EnvironmentExporter:
    """Handles environment variable export for external tools.

    Exports configuration to environment variables consumed by:
    - FINN (FINN_ROOT, FINN_BUILD_DIR, etc.)
    - Xilinx tools (VIVADO_PATH, XILINX_VIVADO, etc.)
    - Visualization tools (NETRON_PORT)
    - BSMITH_* variables (for YAML ${var} expansion in blueprints)

    Internal BSMITH_* variables are exported by default in normal operation
    to support blueprint YAML expansion (e.g., ${BSMITH_DIR}/examples/...).
    The @lru_cache decorator on get_config() prevents feedback loops.

    Example:
        >>> config = SystemConfig()
        >>> exporter = EnvironmentExporter(config)
        >>> env_dict = exporter.to_external_dict()
        >>> print(env_dict['FINN_ROOT'])
    """

    def __init__(self, config: 'SystemConfig'):
        """Initialize environment exporter with configuration.

        Args:
            config: SystemConfig instance to export from
        """
        self.config = config

    def to_external_dict(self) -> Dict[str, str]:
        """Generate dict of external environment variables.

        Returns only variables consumed by external tools (FINN, Xilinx, etc).
        Internal BSMITH_* variables are excluded (use to_all_dict() for those).

        Returns:
            Dict of environment variable names to string values
        """
        env = {}
        cfg = self.config

        # Xilinx tool paths (dual naming for FINN compatibility)
        # Both XILINX_* and *_PATH variants are exported for maximum FINN compatibility.
        # - XILINX_* variants: Used by FINN's Python runtime and internal scripts
        # - *_PATH variants: Used by FINN's TCL scripts during Vivado/Vitis integration
        if cfg.vivado_path:
            vivado_str = str(cfg.vivado_path)
            env["XILINX_VIVADO"] = vivado_str
            env["VIVADO_PATH"] = vivado_str

            if cfg.vivado_ip_cache:
                env["VIVADO_IP_CACHE"] = str(cfg.vivado_ip_cache)

        if cfg.vitis_path:
            vitis_str = str(cfg.vitis_path)
            env["XILINX_VITIS"] = vitis_str
            env["VITIS_PATH"] = vitis_str

        if cfg.vitis_hls_path:
            hls_str = str(cfg.vitis_hls_path)
            env["XILINX_HLS"] = hls_str
            env["HLS_PATH"] = hls_str

        env["PLATFORM_REPO_PATHS"] = cfg.vendor_platform_paths
        env["OHMYXILINX"] = str(cfg.deps_dir / "oh-my-xilinx")

        env["NETRON_PORT"] = str(cfg.netron_port)

        if cfg.finn_root:
            env["FINN_ROOT"] = str(cfg.finn_root)
        if cfg.finn_build_dir:
            env["FINN_BUILD_DIR"] = str(cfg.finn_build_dir)
        if cfg.finn_deps_dir:
            env["FINN_DEPS_DIR"] = str(cfg.finn_deps_dir)

        if cfg.default_workers:
            env["NUM_DEFAULT_WORKERS"] = str(cfg.default_workers)

        return env

    def to_all_dict(self) -> Dict[str, str]:
        """Generate dict of ALL environment variables including internal ones.

        Includes internal BSMITH_* variables needed for YAML expansion.

        Returns:
            Dict of all environment variables
        """
        env_dict = self.to_external_dict()

        env_dict['BSMITH_BUILD_DIR'] = str(self.config.build_dir)
        env_dict['BSMITH_DEPS_DIR'] = str(self.config.deps_dir)
        env_dict['BSMITH_DIR'] = str(self.config.bsmith_dir)
        env_dict['BSMITH_PROJECT_DIR'] = str(self.config.project_dir)

        return env_dict

    def to_env_dict(
        self,
        include_internal: bool = True
    ) -> Dict[str, str]:
        """Generate environment variable dictionary for shell script generation.

        This method builds a complete environment dictionary but does NOT mutate
        os.environ. It is used ONLY for generating activation scripts (env.sh,
        .env, .envrc).

        Python runtime expects environment to be set externally via sourced shell
        scripts before Python starts, ensuring consistent environment across Python
        processes and all subprocesses (including FINN's shell invocations).

        By default, includes all configuration including internal BSMITH_* variables
        (needed for blueprint YAML ${var} expansion and kernel compilation).

        Args:
            include_internal: If True (default), include internal BSMITH_* variables.
                            Set to False to include only external tool variables
                            (FINN_*, XILINX_*, etc).

        Returns:
            Dict of environment variable names to string values (read-only)
        """
        if include_internal:
            env_dict = self.to_all_dict()
        else:
            env_dict = self.to_external_dict()

        path_components = os.environ.get("PATH", "").split(":")

        new_paths = [
            str(p)
            for p in self._collect_path_additions()
            if str(p) not in path_components
        ]

        env_dict["PATH"] = ":".join(path_components + new_paths)

        # FINN XSI no longer requires PYTHONPATH manipulation
        # The new finn.xsi module handles path management internally

        ld_lib_components = os.environ.get("LD_LIBRARY_PATH", "").split(":")

        if self.config.vivado_path and Path(_LIBUDEV_PATH).exists():
            env_dict["LD_PRELOAD"] = _LIBUDEV_PATH

        if self.config.vivado_path:
            arch = platform.machine()
            if arch != "x86_64":
                raise RuntimeError(
                    f"Brainsmith currently only supports x86_64 architecture.\n"
                    f"Detected architecture: {arch}\n"
                    f"Vivado integration has not been tested on this platform.\n"
                    f"If you need ARM support, please open an issue."
                )

            ld_lib_components.append("/lib/x86_64-linux-gnu/")

            vivado_lib = str(self.config.vivado_path / "lib" / "lnx64.o")
            ld_lib_components.append(vivado_lib)

        if self.config.vitis_path:
            vitis_fpo_lib = str(self.config.vitis_path / "lnx64" / "tools" / "fpo_v7_1")
            if vitis_fpo_lib not in ld_lib_components:
                ld_lib_components.append(vitis_fpo_lib)

        env_dict["LD_LIBRARY_PATH"] = ":".join(filter(None, ld_lib_components))

        # The actual HOME override is handled at container level in entrypoint scripts
        if self.config.vivado_path:
            # Ensure XILINX_LOCAL_USER_DATA is set to prevent network operations
            env_dict["XILINX_LOCAL_USER_DATA"] = "no"

        return env_dict

    def _collect_path_additions(self) -> list[Path]:
        """Collect paths to prepend to PATH in priority order.

        Returns:
            List of paths to add (in order)
        """
        paths = []

        # oh-my-xilinx (if exists)
        oh_my_xilinx = self.config.deps_dir / "oh-my-xilinx"
        if oh_my_xilinx.exists():
            paths.append(oh_my_xilinx)

        # User local bin
        paths.append(Path.home() / ".local" / "bin")

        # Xilinx tool binaries
        if self.config.vivado_path:
            paths.append(self.config.vivado_path / "bin")
        if self.config.vitis_path:
            paths.append(self.config.vitis_path / "bin")
        if self.config.vitis_hls_path:
            paths.append(self.config.vitis_hls_path / "bin")

        return paths
