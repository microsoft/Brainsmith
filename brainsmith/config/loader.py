"""Configuration loading and priority resolution for Brainsmith."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from rich.console import Console
from rich.text import Text

from .schema import BrainsmithConfig, ConfigPriority


console = Console()


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def _load_yaml_config(file_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not file_path.exists():
        return {}
    
    try:
        with open(file_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load {file_path}: {e}[/yellow]")
        return {}


def _auto_detect_bsmith_dir() -> Optional[Path]:
    """Auto-detect BSMITH_DIR from module location or current directory."""
    # Try from current working directory
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        # Check if it's a Brainsmith project
        with open(cwd / "pyproject.toml") as f:
            content = f.read()
            if "brainsmith" in content:
                return cwd
    
    # Try from module location
    try:
        import brainsmith
        module_path = Path(brainsmith.__file__).parent.parent
        if (module_path / "pyproject.toml").exists():
            return module_path
    except ImportError:
        pass
    
    return None


def _prepare_build_dir(build_dir_template: str, user: Optional[str] = None) -> str:
    """Prepare build directory path from template."""
    if "{user}" in build_dir_template:
        if user is None:
            user = os.environ.get("USER", "brainsmith")
        return build_dir_template.format(user=user)
    return build_dir_template


def load_config(
    defaults_file: Optional[Path] = None,
    project_file: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    env_prefix: str = "BSMITH_"
) -> BrainsmithConfig:
    """Load configuration with priority resolution.
    
    Priority order (later overrides earlier):
    1. Built-in defaults (from schema)
    2. Defaults file (env_defaults.yaml)
    3. Environment variables
    4. Project settings file (brainsmith_settings.yaml)
    5. CLI arguments
    
    Args:
        defaults_file: Path to env_defaults.yaml
        project_file: Path to project settings file
        cli_overrides: CLI argument overrides
        env_prefix: Environment variable prefix
        
    Returns:
        Validated BrainsmithConfig object
    """
    # Start with empty config dict
    config_dict = {}
    
    # 1. Load defaults file
    if defaults_file is None:
        # Try standard locations
        for location in [
            Path(__file__).parent / "env_defaults.yaml",
            Path.home() / ".brainsmith" / "env_defaults.yaml",
            Path("/etc/brainsmith/env_defaults.yaml"),
        ]:
            if location.exists():
                defaults_file = location
                break
    
    if defaults_file and defaults_file.exists():
        defaults = _load_yaml_config(defaults_file)
        config_dict = _merge_dicts(config_dict, defaults)
    
    # 2. Process environment variables
    # First check for BSMITH_DIR
    if "BSMITH_DIR" not in os.environ:
        auto_dir = _auto_detect_bsmith_dir()
        if auto_dir:
            os.environ["BSMITH_DIR"] = str(auto_dir)
    
    # Handle build dir template
    if "paths" in config_dict and "build_dir" in config_dict["paths"]:
        build_dir_template = config_dict["paths"]["build_dir"]
        if isinstance(build_dir_template, str) and "{user}" in build_dir_template:
            config_dict["paths"]["build_dir"] = _prepare_build_dir(build_dir_template)
    
    # 3. Load project settings file
    if project_file is None:
        # Try standard locations
        for location in [
            Path.cwd() / "brainsmith_settings.yaml",
            Path.cwd() / ".brainsmith" / "settings.yaml",
        ]:
            if location.exists():
                project_file = location
                break
    
    if project_file and project_file.exists():
        project_config = _load_yaml_config(project_file)
        config_dict = _merge_dicts(config_dict, project_config)
    
    # 4. Apply CLI overrides
    if cli_overrides:
        config_dict = _merge_dicts(config_dict, cli_overrides)
    
    # Create config object (will also read from environment)
    config = BrainsmithConfig(**config_dict)
    
    # Resolve paths that might be relative
    if not config.bsmith_deps_dir.is_absolute():
        config.bsmith_deps_dir = (config.bsmith_dir / config.bsmith_deps_dir).absolute()
    
    # Set FINN compatibility paths if not already set
    if not config.finn.finn_root:
        config.finn.finn_root = config.bsmith_dir
    if not config.finn.finn_build_dir:
        config.finn.finn_build_dir = config.bsmith_build_dir
    if not config.finn.finn_deps_dir:
        config.finn.finn_deps_dir = config.bsmith_deps_dir
    
    # Handle Xilinx legacy paths
    if config.xilinx.xilinx_path and not config.xilinx.vivado_path:
        xilinx_version = config.xilinx.version
        vivado_path = config.xilinx.xilinx_path / "Vivado" / xilinx_version
        if vivado_path.exists():
            config.xilinx.vivado_path = vivado_path
        
        vitis_path = config.xilinx.xilinx_path / "Vitis" / xilinx_version
        if vitis_path.exists():
            config.xilinx.vitis_path = vitis_path
            
        hls_path = config.xilinx.xilinx_path / "Vitis_HLS" / xilinx_version
        if hls_path.exists():
            config.xilinx.hls_path = hls_path
    
    return config


def validate_and_report(config: BrainsmithConfig, raise_on_error: bool = True) -> bool:
    """Validate configuration and report issues based on priority.
    
    Args:
        config: Configuration to validate
        raise_on_error: Whether to raise exception on core errors
        
    Returns:
        True if validation passed, False if there were errors
    """
    results = config.validate_by_priority()
    
    # Report errors
    if results["errors"]:
        console.print("\n[bold red]Configuration Errors:[/bold red]")
        for error in results["errors"]:
            console.print(f"  ❌ {error}")
        if raise_on_error:
            raise ValueError("Configuration validation failed")
    
    # Report warnings
    if results["warnings"]:
        console.print("\n[bold yellow]Configuration Warnings:[/bold yellow]")
        for warning in results["warnings"]:
            console.print(f"  ⚠️  {warning}")
    
    # Report info (only if verbose)
    if results["info"] and os.environ.get("BSMITH_DEBUG") == "1":
        console.print("\n[bold blue]Configuration Info:[/bold blue]")
        for info in results["info"]:
            console.print(f"  ℹ️  {info}")
    
    return len(results["errors"]) == 0


# Singleton instance
_config: Optional[BrainsmithConfig] = None


def get_config() -> BrainsmithConfig:
    """Get singleton configuration instance.
    
    This loads the configuration once and caches it for the session.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config():
    """Reset the singleton configuration (mainly for testing)."""
    global _config
    _config = None