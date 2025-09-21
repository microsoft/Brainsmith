"""Migration utilities for backward compatibility with environment variables."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

from .schema import BrainsmithConfig


console = Console()


def export_to_environment(config: BrainsmithConfig, verbose: bool = False) -> None:
    """Export validated config to FINN/legacy environment variables.
    
    This exports only the environment variables needed by FINN and other
    legacy dependencies. We explicitly do NOT export BSMITH_* variables
    to avoid configuration feedback loops.
    
    Args:
        config: Validated configuration object
        verbose: Whether to print export information
    """
    env_dict = {}
    
    # FINN-specific environment variables
    env_dict["FINN_ROOT"] = str(config.finn.finn_root) if config.finn.finn_root else str(config.bsmith_dir)
    env_dict["FINN_BUILD_DIR"] = str(config.finn.finn_build_dir) if config.finn.finn_build_dir else str(config.bsmith_build_dir)
    env_dict["FINN_DEPS_DIR"] = str(config.finn.finn_deps_dir) if config.finn.finn_deps_dir else str(config.bsmith_deps_dir)
    
    if config.finn.num_default_workers:
        env_dict["NUM_DEFAULT_WORKERS"] = str(config.finn.num_default_workers)
    
    # Legacy Xilinx paths that FINN expects
    if config.xilinx.vivado_path:
        env_dict["XILINX_VIVADO"] = str(config.xilinx.vivado_path)
        env_dict["VIVADO_PATH"] = str(config.xilinx.vivado_path)
    if config.xilinx.vitis_path:
        env_dict["XILINX_VITIS"] = str(config.xilinx.vitis_path)
        env_dict["VITIS_PATH"] = str(config.xilinx.vitis_path)
    if config.xilinx.hls_path:
        env_dict["XILINX_HLS"] = str(config.xilinx.hls_path)
        env_dict["HLS_PATH"] = str(config.xilinx.hls_path)
    
    # Python settings that external tools might need
    env_dict["PYTHON"] = config.python.version
    env_dict["PYTHONUNBUFFERED"] = "1" if config.python.unbuffered else "0"
    
    # Platform repo paths for Xilinx tools
    env_dict["PLATFORM_REPO_PATHS"] = config.tools.platform_repo_paths
    
    # oh-my-xilinx path (required by FINN)
    ohmyxilinx = config.tools.ohmyxilinx_path or (config.bsmith_deps_dir / "oh-my-xilinx")
    env_dict["OHMYXILINX"] = str(ohmyxilinx)
    
    # Handle PATH updates
    path_components = os.environ.get("PATH", "").split(":")
    
    # Add oh-my-xilinx to PATH if it exists (hardcoded convention)
    ohmyxilinx_path = config.bsmith_deps_dir / "oh-my-xilinx"
    if ohmyxilinx_path.exists() and str(ohmyxilinx_path) not in path_components:
        path_components.append(str(ohmyxilinx_path))
    
    # Add ~/.local/bin to PATH
    home_local_bin = str(Path.home() / ".local" / "bin")
    if home_local_bin not in path_components:
        path_components.append(home_local_bin)
    
    env_dict["PATH"] = ":".join(path_components)
    
    # Handle PYTHONPATH updates for FINN XSI
    if config.xilinx.vivado_path and config.finn.finn_root:
        finn_xsi_path = config.finn.finn_root / "finn_xsi"
        if finn_xsi_path.exists():
            pythonpath_components = os.environ.get("PYTHONPATH", "").split(":")
            finn_xsi_str = str(finn_xsi_path)
            if finn_xsi_str not in pythonpath_components:
                pythonpath_components.append(finn_xsi_str)
                env_dict["PYTHONPATH"] = ":".join(filter(None, pythonpath_components))
    
    # Handle LD_LIBRARY_PATH updates
    ld_lib_components = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    
    # Add libudev if needed and exists (hardcoded for now)
    libudev_path = "/lib/x86_64-linux-gnu/libudev.so.1"
    if config.xilinx.vivado_path and Path(libudev_path).exists():
        env_dict["LD_PRELOAD"] = libudev_path
    
    # Add Vivado libraries
    if config.xilinx.vivado_path:
        ld_lib_components.append("/lib/x86_64-linux-gnu/")
        vivado_lib = str(config.xilinx.vivado_path / "lib" / "lnx64.o")
        ld_lib_components.append(vivado_lib)
    
    # Add Vitis FPO libraries
    if config.xilinx.vitis_path:
        vitis_fpo_lib = str(config.xilinx.vitis_path / "lnx64" / "tools" / "fpo_v7_1")
        if vitis_fpo_lib not in ld_lib_components:
            ld_lib_components.append(vitis_fpo_lib)
    
    env_dict["LD_LIBRARY_PATH"] = ":".join(filter(None, ld_lib_components))
    
    # Apply all environment variables
    for key, value in env_dict.items():
        if value is not None:
            os.environ[key] = str(value)
            if verbose and key not in ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH"]:
                console.print(f"[dim]Export {key}={value}[/dim]")
    
    if verbose:
        console.print("[green]✓ Configuration exported to environment[/green]")


