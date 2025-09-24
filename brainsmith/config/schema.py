"""Brainsmith configuration schema using Pydantic.

This module defines the complete configuration schema for Brainsmith,
providing type safety and validation.
"""

import os
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
    'FINN_ROOT': lambda c: str(c.finn.finn_root or c.deps_dir / "finn"),
    'FINN_BUILD_DIR': lambda c: str(c.finn.finn_build_dir or c.build_dir),
    'FINN_DEPS_DIR': lambda c: str(c.finn.finn_deps_dir or c.deps_dir),
    'NUM_DEFAULT_WORKERS': lambda c: str(c.finn.num_default_workers) if c.finn.num_default_workers else None,
}


# Keep FinnConfig nested as it makes sense
class FinnConfig(BaseModel):
    """FINN-specific path configuration."""
    finn_root: Optional[Path] = Field(default=None, description="FINN root directory")
    finn_build_dir: Optional[Path] = Field(default=None, description="FINN build directory")
    finn_deps_dir: Optional[Path] = Field(default=None, description="FINN dependencies directory")
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
    """Custom settings source for YAML files."""
    
    def __init__(self, settings_cls: Type[BaseSettings], yaml_file: Optional[Path] = None):
        super().__init__(settings_cls)
        # Check for override from loader
        if '_BRAINSMITH_YAML_FILE' in os.environ:
            self.yaml_file = Path(os.environ['_BRAINSMITH_YAML_FILE'])
        else:
            self.yaml_file = yaml_file or self._find_yaml_file()
        
    def _find_yaml_file(self) -> Optional[Path]:
        """Find YAML file in standard locations.
        
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
    
    def _read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse YAML file with environment variable expansion."""
        if not file_path or not file_path.exists():
            return {}
        try:
            # Use unified YAML parser with env var expansion and schema-based path resolution
            return load_yaml(
                file_path,
                expand_env_vars=True,
                support_inheritance=False,  # Config files don't use inheritance
                schema_class=self.settings_cls  # Pass the BrainsmithConfig class to extract path fields
            )
        except Exception:
            return {}
    
    def get_field_value(self, field_name: str, field_info: Any) -> Tuple[Any, str, bool]:
        """Get field value from YAML source."""
        data = self._read_file(self.yaml_file)
        
        if field_name in data:
            return data[field_name], field_name, True
        
        return None, field_name, False
    
    def __call__(self) -> Dict[str, Any]:
        """Return all settings from YAML file."""
        return self._read_file(self.yaml_file)


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
    bsmith_dir: Optional[Path] = Field(
        default=None,
        description="Root directory of Brainsmith (auto-detected if not set)",
        alias="dir"  # Maps to BSMITH_DIR env var
    )
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
    
    # Debug (flattened)
    debug: bool = Field(
        default=False,
        description="Enable debug output"
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
        # 3. YAML file (custom source)
        # 4. Field defaults (built into pydantic)
        return (
            init_settings,
            env_settings,
            YamlSettingsSource(settings_cls),
        )
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to handle essential setup only."""
        # Auto-detect BSMITH_DIR if not set
        if self.bsmith_dir is None:
            try:
                import brainsmith
                self.bsmith_dir = Path(brainsmith.__file__).parent.parent
            except ImportError:
                raise ValueError("BSMITH_DIR not set and could not be auto-detected")
        
        # Note: Relative paths are already resolved by the YAML parser
        # based on the location of the YAML file
        
        # Set FINN compatibility paths if not already set
        if not self.finn.finn_root:
            self.finn.finn_root =  self.deps_dir / "finn"
        if not self.finn.finn_build_dir:
            self.finn.finn_build_dir = self.build_dir
        if not self.finn.finn_deps_dir:
            self.finn.finn_deps_dir = self.deps_dir
    
    
    @field_validator('bsmith_dir', mode='before')
    @classmethod
    def validate_bsmith_dir(cls, v: Optional[Any]) -> Optional[Path]:
        """Validate that BSMITH_DIR exists and contains pyproject.toml."""
        if v is None:
            # Will be handled in model_post_init
            return v
        
        # Convert to Path if needed
        if not isinstance(v, Path):
            v = Path(v)
            
        if not v.exists():
            raise ValueError(f"BSMITH_DIR {v} does not exist")
        if not (v / "pyproject.toml").exists():
            raise ValueError(f"BSMITH_DIR {v} does not contain pyproject.toml")
        return v
    
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