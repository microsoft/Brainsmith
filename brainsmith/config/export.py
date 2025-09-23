"""Export configuration to environment variables."""

import os
from pathlib import Path
from rich.console import Console

from .schema import BrainsmithConfig


console = Console()


def export_to_environment(config: BrainsmithConfig, verbose: bool = False) -> None:
    """Export validated config to environment variables.
    
    This exports only external tool configuration values (FINN_*, XILINX_*, etc)
    and sets up PATH, PYTHONPATH, and LD_LIBRARY_PATH for tool compatibility.
    Internal BSMITH_* variables are NOT exported to prevent feedback loops.
    
    Args:
        config: Validated configuration object
        verbose: Whether to print export information
    """
    # Get external environment variables only
    env_dict = config.to_external_env_dict()
    
    # Handle PATH updates
    path_components = os.environ.get("PATH", "").split(":")
    
    # Add oh-my-xilinx to PATH if it exists (hardcoded convention)
    ohmyxilinx_path = config.deps_dir / "oh-my-xilinx"
    if ohmyxilinx_path.exists() and str(ohmyxilinx_path) not in path_components:
        path_components.append(str(ohmyxilinx_path))
    
    # Add ~/.local/bin to PATH
    home_local_bin = str(Path.home() / ".local" / "bin")
    if home_local_bin not in path_components:
        path_components.append(home_local_bin)
    
    # Add Xilinx tool bin directories to PATH
    if config.effective_vivado_path:
        vivado_bin = str(config.effective_vivado_path / "bin")
        if vivado_bin not in path_components:
            path_components.append(vivado_bin)
    
    if config.effective_vitis_path:
        vitis_bin = str(config.effective_vitis_path / "bin")
        if vitis_bin not in path_components:
            path_components.append(vitis_bin)
    
    if config.effective_vitis_hls_path:
        hls_bin = str(config.effective_vitis_hls_path / "bin")
        if hls_bin not in path_components:
            path_components.append(hls_bin)
    
    env_dict["PATH"] = ":".join(path_components)
    
    # Handle PYTHONPATH updates for FINN XSI
    if config.effective_vivado_path and config.finn.finn_root:
        finn_xsi_path = config.finn.finn_root / "finn_xsi"
        if finn_xsi_path.exists():
            pythonpath_components = os.environ.get("PYTHONPATH", "").split(":")
            finn_xsi_str = str(finn_xsi_path)
            if finn_xsi_str not in pythonpath_components:
                pythonpath_components.append(finn_xsi_str)
                env_dict["PYTHONPATH"] = ":".join(filter(None, pythonpath_components))
    
    # Handle LD_LIBRARY_PATH updates
    ld_lib_components = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    
    # Add libudev if needed and exists for Xilinx tool compatibility
    libudev_path = "/lib/x86_64-linux-gnu/libudev.so.1"
    if config.effective_vivado_path and Path(libudev_path).exists():
        env_dict["LD_PRELOAD"] = libudev_path
    
    # Add Vivado libraries
    if config.effective_vivado_path:
        ld_lib_components.append("/lib/x86_64-linux-gnu/")
        vivado_lib = str(config.effective_vivado_path / "lib" / "lnx64.o")
        ld_lib_components.append(vivado_lib)
    
    # Add Vitis FPO libraries
    if config.effective_vitis_path:
        vitis_fpo_lib = str(config.effective_vitis_path / "lnx64" / "tools" / "fpo_v7_1")
        if vitis_fpo_lib not in ld_lib_components:
            ld_lib_components.append(vitis_fpo_lib)
    
    env_dict["LD_LIBRARY_PATH"] = ":".join(filter(None, ld_lib_components))
    
    # Set Xilinx environment variables for better caching behavior
    # The actual HOME override is handled at container level in entrypoint scripts
    if config.effective_vivado_path:
        # Ensure XILINX_LOCAL_USER_DATA is set to prevent network operations
        env_dict["XILINX_LOCAL_USER_DATA"] = "no"
    
    # Apply all environment variables
    for key, value in env_dict.items():
        if value is not None:
            os.environ[key] = str(value)
            if verbose and key not in ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH"]:
                console.print(f"[dim]Export {key}={value}[/dim]")
    
    if verbose:
        console.print("[green]✓ Configuration exported to environment[/green]")


