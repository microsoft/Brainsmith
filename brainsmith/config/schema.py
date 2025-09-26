"""Brainsmith configuration schema using Pydantic.

This module defines the complete configuration schema for Brainsmith,
providing type safety and validation.
"""

import os
import yaml
from pathlib import Path
from functools import cached_property
from typing import Optional, Dict, Any, List, Tuple, Type, Callable
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
from brainsmith.utils.yaml_parser import load_yaml


# Declarative environment variable export mappings
EXTERNAL_ENV_MAPPINGS: Dict[str, Callable[['BrainsmithConfig'], Optional[str]]] = {
    # Xilinx tool paths
    'XILINX_VIVADO': lambda c: str(c.effective_vivado_path) if c.effective_vivado_path else None,
    'VIVADO_PATH': lambda c: str(c.effective_vivado_path) if c.effective_vivado_path else None,
    'XILINX_VITIS': lambda c: str(c.effective_vitis_path) if c.effective_vitis_path else None,
    'VITIS_PATH': lambda c: str(c.effective_vitis_path) if c.effective_vitis_path else None,
    'XILINX_HLS': lambda c: str(c.effective_vitis_hls_path) if c.effective_vitis_hls_path else None,
    'HLS_PATH': lambda c: str(c.effective_vitis_hls_path) if c.effective_vitis_hls_path else None,
    
    # Platform and tool paths
    'PLATFORM_REPO_PATHS': lambda c: c.platform_repo_paths,
    'OHMYXILINX': lambda c: str(c.deps_dir / "oh-my-xilinx"),
    
    # Vivado specific
    'VIVADO_IP_CACHE': lambda c: str(c.effective_vivado_ip_cache) if c.effective_vivado_path else None,
    
    # Visualization
    'NETRON_PORT': lambda c: str(c.netron_port),
    
    # FINN environment variables
    'FINN_ROOT': lambda c: str(c.effective_finn_root),
    'FINN_BUILD_DIR': lambda c: str(c.effective_finn_build_dir),
    'FINN_DEPS_DIR': lambda c: str(c.effective_finn_deps_dir),
    'NUM_DEFAULT_WORKERS': lambda c: str(c.finn.num_default_workers) if c.finn.num_default_workers else None,
}


# Keep FinnConfig nested as it makes sense
class FinnConfig(BaseModel):
    """FINN-specific path configuration.
    
    These paths default to sensible values derived from the parent config:
    - finn_root: defaults to deps_dir/finn
    - finn_build_dir: defaults to build_dir
    - finn_deps_dir: defaults to deps_dir
    """
    finn_root: Optional[Path] = Field(default=None, description="FINN root directory (defaults to deps_dir/finn)")
    finn_build_dir: Optional[Path] = Field(default=None, description="FINN build directory (defaults to build_dir)")
    finn_deps_dir: Optional[Path] = Field(default=None, description="FINN dependencies directory (defaults to deps_dir)")
    num_default_workers: Optional[int] = Field(default=None, description="Default number of workers")
    
    @field_validator('finn_root', 'finn_build_dir', 'finn_deps_dir', mode='before')
    @classmethod
    def ensure_path_type(cls, v: Optional[Any]) -> Optional[Path]:
        """Convert to Path object."""
        if v is None:
            return None
        if not isinstance(v, Path):
            return Path(v)
        return v
    
    model_config = ConfigDict(validate_assignment=True)


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for YAML files with support for user and project configs."""
    
    def __init__(self, settings_cls: Type[BaseSettings], yaml_file: Optional[Path] = None):
        super().__init__(settings_cls)
        self.yaml_files = []
        
        # Check for explicit project file from loader
        if '_BRAINSMITH_PROJECT_FILE' in os.environ:
            self.yaml_files.append(Path(os.environ['_BRAINSMITH_PROJECT_FILE']))
        else:
            # Find project file in standard locations
            project_file = self._find_project_yaml_file()
            if project_file:
                self.yaml_files.append(project_file)
        
        # Check for user file from loader
        if '_BRAINSMITH_USER_FILE' in os.environ:
            self.yaml_files.append(Path(os.environ['_BRAINSMITH_USER_FILE']))
        else:
            # Check default user config location
            user_file = Path.home() / ".brainsmith" / "config.yaml"
            if user_file.exists():
                self.yaml_files.append(user_file)
        
        # Load and merge all YAML files
        self._data = self._load_and_merge_yaml_files()
        
    def _find_project_yaml_file(self) -> Optional[Path]:
        """Find project YAML file in standard locations.
        
        Search order:
        1. Current working directory
        2. Brainsmith project root (auto-detected)
        """
        # First check current working directory
        for location in [
            Path.cwd() / "brainsmith_settings.yaml",
            Path.cwd() / ".brainsmith" / "settings.yaml",
        ]:
            if location.exists():
                return location
        
        # Then check brainsmith project root
        try:
            import brainsmith
            bsmith_root = Path(brainsmith.__file__).parent.parent
            root_settings = bsmith_root / "brainsmith_settings.yaml"
            if root_settings.exists():
                return root_settings
        except Exception:
            pass
            
        return None
    
    def _load_and_merge_yaml_files(self) -> Dict[str, Any]:
        """Load and merge multiple YAML files with proper priority.
        
        Files are loaded in order with later files having higher priority
        (project config overrides user config).
        """
        merged_data = {}
        
        for yaml_file in self.yaml_files:
            if yaml_file and yaml_file.exists():
                try:
                    # Use unified YAML parser with env var expansion and schema-based path resolution
                    data = load_yaml(
                        yaml_file,
                        expand_env_vars=True,
                        support_inheritance=False,  # Config files don't use inheritance
                        schema_class=self.settings_cls  # Pass the BrainsmithConfig class to extract path fields
                    )
                    # Deep merge the data
                    merged_data = self._deep_merge(merged_data, data)
                except yaml.YAMLError as e:
                    # Import here to avoid circular dependency
                    from brainsmith.interface.utils import warning
                    
                    # Extract useful error information
                    error_msg = str(e)
                    if hasattr(e, 'problem_mark'):
                        mark = e.problem_mark
                        error_msg = f"line {mark.line + 1}, column {mark.column + 1}: {e.problem or 'invalid syntax'}"
                    
                    warning(
                        f"Failed to parse settings file: {yaml_file}",
                        details=[
                            f"YAML syntax error at {error_msg}",
                            "Skipping this configuration file",
                            f"Fix the syntax in {yaml_file} to load these settings"
                        ]
                    )
                except Exception as e:
                    # For other unexpected errors, still warn but less specifically
                    from brainsmith.interface.utils import warning
                    warning(
                        f"Failed to read settings file: {yaml_file}",
                        details=[
                            f"Error: {e}",
                            "Skipping this configuration file"
                        ]
                    )
        
        return merged_data
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with update taking precedence."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_field_value(self, field_name: str, field_info: Any) -> Tuple[Any, str, bool]:
        """Get field value from YAML source."""
        if field_name in self._data:
            return self._data[field_name], field_name, True
        
        return None, field_name, False
    
    def __call__(self) -> Dict[str, Any]:
        """Return all settings from YAML file."""
        return self._data


class BrainsmithConfig(BaseSettings):
    """The complete truth about Brainsmith configuration.
    
    Configuration priority (following pydantic-settings convention):
    1. CLI arguments (passed to constructor) - HIGHEST
    2. Environment variables (BSMITH_* prefix)
    3. Project settings (brainsmith_settings.yaml)
    4. Built-in defaults (Field defaults) - LOWEST
    
    Note: We only read BSMITH_* env vars, and only export FINN_* vars
    to avoid configuration feedback loops.
    """
    
    # Core paths
    # NOTE: bsmith_dir is now a cached property, not a configurable field
    build_dir: Path = Field(
        default=Path("/tmp/brainsmith_build"),
        description="Build directory for artifacts"
    )
    deps_dir: Path = Field(
        default=Path("deps"),
        description="Dependencies directory"
    )
    
    # Xilinx configuration (flattened)
    xilinx_path: Optional[Path] = Field(
        default=None,
        description="Xilinx root installation path (e.g., /tools/Xilinx)"
    )
    xilinx_version: str = Field(
        default="2024.2",
        description="Xilinx tool version"
    )
    # These are auto-derived from xilinx_path but can be overridden
    vivado_path: Optional[Path] = Field(
        default=None,
        description="Path to Vivado (auto-detected from xilinx_path)"
    )
    vitis_path: Optional[Path] = Field(
        default=None,
        description="Path to Vitis (auto-detected from xilinx_path)"
    )
    vitis_hls_path: Optional[Path] = Field(
        default=None,
        description="Path to Vitis HLS (auto-detected from xilinx_path)"
    )
    
    # Tool paths (flattened)
    platform_repo_paths: str = Field(
        default="/opt/xilinx/platforms",
        description="Platform repository paths"
    )
    
    # Debug and output settings
    debug: bool = Field(
        default=False,
        description="Enable debug output"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output"
    )
    
    # Plugin settings
    plugins_strict: bool = Field(
        default=True,
        description="Strict plugin loading"
    )
    
    # Vivado-specific settings
    vivado_ip_cache: Optional[Path] = Field(
        default=None,
        description="Vivado IP cache directory (auto-computed from build_dir if not set)"
    )
    
    # Network/visualization settings
    netron_port: int = Field(
        default=8080,
        description="Port for Netron neural network visualization"
    )
    
    # FINN configuration (keep nested - it's a clear subsystem)
    finn: FinnConfig = Field(default_factory=FinnConfig)
    
    model_config = SettingsConfigDict(
        env_prefix='BSMITH_',
        env_nested_delimiter='__',
        validate_assignment=True,
        extra="allow",
        case_sensitive=False,
        env_file=None,  # We handle config files via custom source
    )
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to match our priority order."""
        # Priority order in pydantic-settings: first source wins!
        # 1. Init settings (CLI/constructor args) - highest priority
        # 2. Environment variables (standard pydantic-settings behavior)
        # 3. YAML files (custom source) - handles both user and project configs
        # 4. Field defaults (built into pydantic)
        return (
            init_settings,
            env_settings,
            YamlSettingsSource(settings_cls),
        )
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to handle essential setup only."""
        # Note: Relative paths are already resolved by the YAML parser
        # based on the location of the YAML file
        pass  # No longer need to set FINN paths here
    
    @property
    def bsmith_dir(self) -> Path:
        """Get the Brainsmith package root directory.
        
        This is always auto-detected from the package location and cannot be overridden.
        The directory is validated to ensure it contains pyproject.toml.
        """
        # Cache the value in a private attribute
        if not hasattr(self, '_bsmith_dir'):
            try:
                import brainsmith
                bsmith_dir = Path(brainsmith.__file__).parent.parent
                
                # Validate that this is indeed the Brainsmith root
                if not (bsmith_dir / "pyproject.toml").exists():
                    raise ValueError(f"Auto-detected directory {bsmith_dir} does not contain pyproject.toml")
                
                self._bsmith_dir = bsmith_dir
            except ImportError:
                raise ValueError("Cannot determine BSMITH_DIR: brainsmith package not importable")
        
        return self._bsmith_dir
    
    
    @field_validator('build_dir', 'deps_dir', mode='before')
    @classmethod
    def ensure_path_type(cls, v: Any) -> Path:
        """Convert to Path object."""
        # Ensure we have a Path object
        if not isinstance(v, Path):
            v = Path(v)
        return v
    
    @field_validator('xilinx_path', 'vivado_path', 'vitis_path', 'vitis_hls_path', 'vivado_ip_cache', mode='before')
    @classmethod
    def validate_tool_path(cls, v: Optional[Any]) -> Optional[Path]:
        """Convert tool paths to Path objects."""
        if v is None:
            return None
        if not isinstance(v, Path):
            return Path(v)
        return v
    
    
    def to_external_env_dict(self) -> Dict[str, str]:
        """Export external tool environment variables.
        
        This exports only environment variables consumed by external tools
        (FINN, Xilinx, etc). Internal BSMITH_* variables are NOT exported
        to prevent configuration feedback loops.
        """
        env_dict = {}
        
        # Apply all mappings
        for env_var, getter in EXTERNAL_ENV_MAPPINGS.items():
            value = getter(self)
            if value is not None:
                env_dict[env_var] = value
        
        return env_dict
    
    def to_all_env_dict(self) -> Dict[str, str]:
        """Export ALL environment variables including internal BSMITH_* ones.
        
        WARNING: This includes internal configuration variables that may cause
        feedback loops if used incorrectly. Use this only when you need access
        to BSMITH_BUILD_DIR and similar internal variables.
        """
        env_dict = self.to_external_env_dict()
        
        # Add internal BSMITH variables
        env_dict['BSMITH_BUILD_DIR'] = str(self.build_dir)
        env_dict['BSMITH_DEPS_DIR'] = str(self.deps_dir)
        env_dict['BSMITH_DIR'] = str(self.bsmith_dir)
        
        return env_dict
    
    # Keep old method for backwards compatibility, but have it use the new one
    def to_env_dict(self) -> Dict[str, str]:
        """Deprecated: Use to_external_env_dict() instead."""
        return self.to_external_env_dict()
    
    @cached_property
    def effective_vivado_path(self) -> Optional[Path]:
        """Get effective Vivado path with auto-detection."""
        if self.vivado_path:
            return self.vivado_path
        
        if self.xilinx_path and self.xilinx_path.exists():
            vivado_path = self.xilinx_path / "Vivado" / self.xilinx_version
            if vivado_path.exists():
                return vivado_path
        
        return None
    
    @cached_property
    def effective_vitis_path(self) -> Optional[Path]:
        """Get effective Vitis path with auto-detection."""
        if self.vitis_path:
            return self.vitis_path
        
        if self.xilinx_path and self.xilinx_path.exists():
            vitis_path = self.xilinx_path / "Vitis" / self.xilinx_version
            if vitis_path.exists():
                return vitis_path
        
        return None
    
    @cached_property
    def effective_vitis_hls_path(self) -> Optional[Path]:
        """Get effective Vitis HLS path with auto-detection."""
        if self.vitis_hls_path:
            return self.vitis_hls_path
        
        if self.xilinx_path and self.xilinx_path.exists():
            hls_path = self.xilinx_path / "Vitis_HLS" / self.xilinx_version
            if hls_path.exists():
                return hls_path
        
        return None
    
    @cached_property
    def effective_vivado_ip_cache(self) -> Path:
        """Get effective Vivado IP cache directory."""
        if self.vivado_ip_cache:
            return self.vivado_ip_cache
        return self.build_dir / "vivado_ip_cache"
    
    @property
    def effective_finn_root(self) -> Path:
        """Get effective FINN root directory."""
        if self.finn.finn_root:
            return self.finn.finn_root
        # Default: deps_dir / "finn"
        deps = self.deps_dir
        if not deps.is_absolute():
            deps = self.bsmith_dir / deps
        return deps / "finn"
    
    @property
    def effective_finn_build_dir(self) -> Path:
        """Get effective FINN build directory."""
        return self.finn.finn_build_dir or self.build_dir
    
    @property
    def effective_finn_deps_dir(self) -> Path:
        """Get effective FINN dependencies directory."""
        if self.finn.finn_deps_dir:
            return self.finn.finn_deps_dir
        # Default: deps_dir
        deps = self.deps_dir
        if not deps.is_absolute():
            deps = self.bsmith_dir / deps
        return deps
    
    def export_to_environment(self, include_internal: bool = False, verbose: bool = False, export: bool = True) -> Dict[str, str]:
        """Export configuration to environment variables.
        
        This is the unified method for exporting configuration to the environment.
        By default, exports only external tool configuration values (FINN_*, XILINX_*, etc)
        and sets up PATH, PYTHONPATH, and LD_LIBRARY_PATH for tool compatibility.
        
        Args:
            include_internal: If True, also export internal BSMITH_* variables
                            (WARNING: may cause configuration feedback loops)
            verbose: Whether to print export information
        """
        # Get environment variables based on what's requested
        if include_internal:
            env_dict = self.to_all_env_dict()
        else:
            env_dict = self.to_external_env_dict()
        
        # Handle PATH updates
        path_components = os.environ.get("PATH", "").split(":")
        
        # Add oh-my-xilinx to PATH if it exists (hardcoded convention)
        ohmyxilinx_path = self.deps_dir / "oh-my-xilinx"
        if ohmyxilinx_path.exists() and str(ohmyxilinx_path) not in path_components:
            path_components.append(str(ohmyxilinx_path))
        
        # Add ~/.local/bin to PATH
        home_local_bin = str(Path.home() / ".local" / "bin")
        if home_local_bin not in path_components:
            path_components.append(home_local_bin)
        
        # Add Xilinx tool bin directories to PATH
        if self.effective_vivado_path:
            vivado_bin = str(self.effective_vivado_path / "bin")
            if vivado_bin not in path_components:
                path_components.append(vivado_bin)
        
        if self.effective_vitis_path:
            vitis_bin = str(self.effective_vitis_path / "bin")
            if vitis_bin not in path_components:
                path_components.append(vitis_bin)
        
        if self.effective_vitis_hls_path:
            hls_bin = str(self.effective_vitis_hls_path / "bin")
            if hls_bin not in path_components:
                path_components.append(hls_bin)
        
        env_dict["PATH"] = ":".join(path_components)
        
        # FINN XSI no longer requires PYTHONPATH manipulation
        # The new finn.xsi module handles path management internally
        
        # Handle LD_LIBRARY_PATH updates
        ld_lib_components = os.environ.get("LD_LIBRARY_PATH", "").split(":")
        
        # Add libudev if needed and exists for Xilinx tool compatibility
        libudev_path = "/lib/x86_64-linux-gnu/libudev.so.1"
        if self.effective_vivado_path and Path(libudev_path).exists():
            env_dict["LD_PRELOAD"] = libudev_path
        
        # Add Vivado libraries
        if self.effective_vivado_path:
            ld_lib_components.append("/lib/x86_64-linux-gnu/")
            vivado_lib = str(self.effective_vivado_path / "lib" / "lnx64.o")
            ld_lib_components.append(vivado_lib)
        
        # Add Vitis FPO libraries
        if self.effective_vitis_path:
            vitis_fpo_lib = str(self.effective_vitis_path / "lnx64" / "tools" / "fpo_v7_1")
            if vitis_fpo_lib not in ld_lib_components:
                ld_lib_components.append(vitis_fpo_lib)
        
        env_dict["LD_LIBRARY_PATH"] = ":".join(filter(None, ld_lib_components))
        
        # Set Xilinx environment variables for better caching behavior
        # The actual HOME override is handled at container level in entrypoint scripts
        if self.effective_vivado_path:
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