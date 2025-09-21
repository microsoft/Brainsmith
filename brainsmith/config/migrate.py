"""Migration utilities for backward compatibility with environment variables."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

from .schema import BrainsmithConfig


console = Console()


def export_to_environment(config: BrainsmithConfig, verbose: bool = False) -> None:
    """Export validated config back to environment variables for legacy code.
    
    This ensures backward compatibility with code that directly reads
    environment variables.
    
    Args:
        config: Validated configuration object
        verbose: Whether to print export information
    """
    env_dict = config.to_env_dict()
    
    # Additional legacy mappings not in to_env_dict
    legacy_mappings = {
        # Vivado IP cache with build dir substitution
        "VIVADO_IP_CACHE": config.xilinx.vivado_ip_cache.replace(
            "{build_dir}", str(config.bsmith_build_dir)
        ),
    }
    
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
    
    # Add libudev if needed and exists
    if config.xilinx.vivado_path and Path(config.library_paths.libudev).exists():
        env_dict["LD_PRELOAD"] = config.library_paths.libudev
    
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
    env_dict.update(legacy_mappings)
    
    for key, value in env_dict.items():
        if value is not None:
            os.environ[key] = str(value)
            if verbose and key not in ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH"]:
                console.print(f"[dim]Export {key}={value}[/dim]")
    
    if verbose:
        console.print("[green]✓ Configuration exported to environment[/green]")


def import_from_environment() -> Dict[str, Any]:
    """Import existing environment variables to config dict.
    
    This helps with migration from environment-based configuration
    to Pydantic-based configuration.
    
    Returns:
        Dictionary suitable for creating BrainsmithConfig
    """
    config: Dict[str, Any] = {}
    
    # Core paths
    if "BSMITH_DIR" in os.environ:
        config["bsmith_dir"] = os.environ["BSMITH_DIR"]
    if "BSMITH_BUILD_DIR" in os.environ:
        config["bsmith_build_dir"] = os.environ["BSMITH_BUILD_DIR"]
    if "BSMITH_DEPS_DIR" in os.environ:
        config["bsmith_deps_dir"] = os.environ["BSMITH_DEPS_DIR"]
    
    # Python configuration
    python_config = {}
    if "PYTHON" in os.environ:
        python_config["version"] = os.environ["PYTHON"]
    if "PYTHONUNBUFFERED" in os.environ:
        python_config["unbuffered"] = os.environ["PYTHONUNBUFFERED"] == "1"
    if python_config:
        config["python"] = python_config
    
    # Xilinx configuration
    xilinx_config = {}
    if "XILINX_VIVADO" in os.environ:
        xilinx_config["vivado_path"] = os.environ["XILINX_VIVADO"]
    if "XILINX_VITIS" in os.environ:
        xilinx_config["vitis_path"] = os.environ["XILINX_VITIS"]
    if "XILINX_HLS" in os.environ:
        xilinx_config["hls_path"] = os.environ["XILINX_HLS"]
    elif "XILINX_VITIS_HLS" in os.environ:
        xilinx_config["hls_path"] = os.environ["XILINX_VITIS_HLS"]
    if "XILINX_LOCAL_USER_DATA" in os.environ:
        xilinx_config["local_user_data"] = os.environ["XILINX_LOCAL_USER_DATA"]
    if "BSMITH_XILINX_PATH" in os.environ:
        xilinx_config["xilinx_path"] = os.environ["BSMITH_XILINX_PATH"]
    if "BSMITH_XILINX_VERSION" in os.environ:
        xilinx_config["version"] = os.environ["BSMITH_XILINX_VERSION"]
    if xilinx_config:
        config["xilinx"] = xilinx_config
    
    # Tools configuration
    tools_config = {}
    # OHMYXILINX is deprecated - oh-my-xilinx is now added directly to PATH
    if "PLATFORM_REPO_PATHS" in os.environ:
        tools_config["platform_repo_paths"] = os.environ["PLATFORM_REPO_PATHS"]
    if tools_config:
        config["tools"] = tools_config
    
    # Compiler configuration
    if "BSMITH_HW_COMPILER" in os.environ:
        config["hw_compiler"] = os.environ["BSMITH_HW_COMPILER"]
    
    # Network configuration
    if "NETRON_PORT" in os.environ:
        config.setdefault("network", {})["netron_port"] = os.environ["NETRON_PORT"]
    
    # Dependency configuration
    if "BSMITH_FETCH_BOARDS" in os.environ:
        config["fetch_boards"] = os.environ["BSMITH_FETCH_BOARDS"].lower() in ("true", "1", "yes")
    if "BSMITH_FETCH_EXPERIMENTAL" in os.environ:
        config["fetch_experimental"] = os.environ["BSMITH_FETCH_EXPERIMENTAL"].lower() in ("true", "1", "yes")
    
    # Handle legacy FINN_SKIP_BOARD_FILES
    if os.environ.get("FINN_SKIP_BOARD_FILES") == "1":
        config["fetch_boards"] = False
    
    # Debug configuration
    if "BSMITH_DEBUG" in os.environ:
        config.setdefault("debug", {})["enabled"] = os.environ["BSMITH_DEBUG"] == "1"
    
    # FINN configuration
    finn_config = {}
    if "FINN_ROOT" in os.environ:
        finn_config["finn_root"] = os.environ["FINN_ROOT"]
    if "FINN_BUILD_DIR" in os.environ:
        finn_config["finn_build_dir"] = os.environ["FINN_BUILD_DIR"]
    if "FINN_DEPS_DIR" in os.environ:
        finn_config["finn_deps_dir"] = os.environ["FINN_DEPS_DIR"]
    if "NUM_DEFAULT_WORKERS" in os.environ:
        finn_config["num_default_workers"] = int(os.environ["NUM_DEFAULT_WORKERS"])
    
    # FINN module paths
    for module_name in ["FINN_RTLLIB", "FINN_CUSTOM_HLS", "FINN_QNN_DATA", "FINN_NOTEBOOKS", "FINN_TESTS"]:
        if module_name in os.environ:
            finn_config[module_name.lower()] = os.environ[module_name]
    
    if finn_config:
        config["finn"] = finn_config
    
    # Other top-level configs
    if "BSMITH_PLUGINS_STRICT" in os.environ:
        config["plugins_strict"] = os.environ["BSMITH_PLUGINS_STRICT"].lower() in ("true", "1", "yes")
    
    return config
