"""Brainsmith configuration schema using Pydantic.

This module defines the complete configuration schema for Brainsmith,
providing type safety and validation.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Type
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
import yaml


# Keep FinnConfig nested as it makes sense
class FinnConfig(BaseModel):
    """FINN-specific path configuration."""
    finn_root: Optional[Path] = Field(default=None, description="FINN root directory")
    finn_build_dir: Optional[Path] = Field(default=None, description="FINN build directory")
    finn_deps_dir: Optional[Path] = Field(default=None, description="FINN dependencies directory")
    num_default_workers: Optional[int] = Field(default=None, description="Default number of workers")
    
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
        """Find YAML file in standard locations."""
        for location in [
            Path.cwd() / "brainsmith_settings.yaml",
            Path.cwd() / ".brainsmith" / "settings.yaml",
        ]:
            if location.exists():
                return location
        return None
    
    def _read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse YAML file."""
        if not file_path or not file_path.exists():
            return {}
        try:
            with open(file_path) as f:
                return yaml.safe_load(f) or {}
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
        description="Root directory of Brainsmith (auto-detected if not set)"
    )
    bsmith_build_dir: Path = Field(
        default=Path("/tmp/brainsmith_build"),
        description="Build directory for artifacts"
    )
    bsmith_deps_dir: Path = Field(
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
        """Post-initialization to handle auto-detection and path resolution."""
        # Auto-detect BSMITH_DIR if not set
        if self.bsmith_dir is None:
            auto_dir = self._auto_detect_bsmith_dir()
            if auto_dir:
                self.bsmith_dir = auto_dir
            else:
                raise ValueError("BSMITH_DIR not set and could not be auto-detected")
        
        # Resolve relative paths
        if self.bsmith_dir and not self.bsmith_deps_dir.is_absolute():
            self.bsmith_deps_dir = (self.bsmith_dir / self.bsmith_deps_dir).absolute()
        
        # Set FINN compatibility paths if not already set
        if not self.finn.finn_root:
            self.finn.finn_root = self.bsmith_dir
        if not self.finn.finn_build_dir:
            self.finn.finn_build_dir = self.bsmith_build_dir
        if not self.finn.finn_deps_dir:
            self.finn.finn_deps_dir = self.bsmith_deps_dir
        
        # Auto-detect Xilinx tool paths from xilinx_path if not explicitly set
        if self.xilinx_path and self.xilinx_path.exists():
            if not self.vivado_path:
                vivado_path = self.xilinx_path / "Vivado" / self.xilinx_version
                if vivado_path.exists():
                    self.vivado_path = vivado_path
            
            if not self.vitis_path:
                vitis_path = self.xilinx_path / "Vitis" / self.xilinx_version
                if vitis_path.exists():
                    self.vitis_path = vitis_path
            
            if not self.vitis_hls_path:
                hls_path = self.xilinx_path / "Vitis_HLS" / self.xilinx_version
                if hls_path.exists():
                    self.vitis_hls_path = hls_path
        
        # Auto-compute Vivado IP cache if not set
        if not self.vivado_ip_cache and self.vivado_path:
            self.vivado_ip_cache = self.bsmith_build_dir / "vivado_ip_cache"
    
    def _auto_detect_bsmith_dir(self) -> Optional[Path]:
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
    
    @field_validator('bsmith_dir')
    @classmethod
    def validate_bsmith_dir(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate that BSMITH_DIR exists and contains pyproject.toml."""
        if v is None:
            # Will be handled in model_post_init
            return v
        if not v.exists():
            raise ValueError(f"BSMITH_DIR {v} does not exist")
        if not (v / "pyproject.toml").exists():
            raise ValueError(f"BSMITH_DIR {v} does not contain pyproject.toml")
        return v.absolute()
    
    @field_validator('bsmith_build_dir', 'bsmith_deps_dir')
    @classmethod
    def resolve_paths(cls, v: Path, info) -> Path:
        """Resolve relative paths against BSMITH_DIR."""
        if not v.is_absolute() and 'bsmith_dir' in info.data and info.data['bsmith_dir'] is not None:
            return (info.data['bsmith_dir'] / v).absolute()
        return v.absolute()
    
    @field_validator('vivado_path', 'vitis_path', 'vitis_hls_path')
    @classmethod
    def validate_tool_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate Xilinx tool installation paths."""
        if v is not None:
            if not v.exists():
                raise ValueError(f"Xilinx tool path does not exist: {v}")
            
            # Check for settings64.sh
            settings_file = v / "settings64.sh"
            if not settings_file.exists():
                raise ValueError(f"Invalid Xilinx installation at {v} - missing settings64.sh")
            
            # Check for key executables (without executing them)
            if "Vivado" in str(v):
                if not (v / "bin" / "vivado").exists():
                    raise ValueError(f"Invalid Vivado installation - missing bin/vivado")
            elif "Vitis" in str(v) and "HLS" not in str(v):
                if not (v / "bin" / "vitis").exists():
                    raise ValueError(f"Invalid Vitis installation - missing bin/vitis")
            elif "HLS" in str(v):
                if not (v / "bin" / "vitis_hls").exists() and not (v / "bin" / "vivado_hls").exists():
                    raise ValueError(f"Invalid Vitis HLS installation - missing HLS executable")
        
        return v
    
    
    def to_env_dict(self) -> Dict[str, str]:
        """Convert configuration to environment variable dictionary."""
        env_dict = {}
        
        # Core paths
        env_dict["BSMITH_DIR"] = str(self.bsmith_dir)
        env_dict["BSMITH_BUILD_DIR"] = str(self.bsmith_build_dir)
        env_dict["BSMITH_DEPS_DIR"] = str(self.bsmith_deps_dir)
        
        # Plugins
        env_dict["BSMITH_PLUGINS_STRICT"] = "true" if self.plugins_strict else "false"
        
        # Xilinx tools
        if self.vivado_path:
            env_dict["XILINX_VIVADO"] = str(self.vivado_path)
            env_dict["VIVADO_PATH"] = str(self.vivado_path)
        if self.vitis_path:
            env_dict["XILINX_VITIS"] = str(self.vitis_path)
            env_dict["VITIS_PATH"] = str(self.vitis_path)
        if self.vitis_hls_path:
            env_dict["XILINX_HLS"] = str(self.vitis_hls_path)
            env_dict["HLS_PATH"] = str(self.vitis_hls_path)
        
        # Tools  
        # OHMYXILINX is required by FINN - always use deps/oh-my-xilinx
        env_dict["OHMYXILINX"] = str(self.bsmith_deps_dir / "oh-my-xilinx")
        env_dict["PLATFORM_REPO_PATHS"] = self.platform_repo_paths
        
        # Debug
        env_dict["BSMITH_DEBUG"] = "1" if self.debug else "0"
        
        # Vivado IP cache
        if self.vivado_ip_cache:
            env_dict["VIVADO_IP_CACHE"] = str(self.vivado_ip_cache)
        
        # Netron port
        env_dict["NETRON_PORT"] = str(self.netron_port)
        
        # FINN compatibility
        env_dict["FINN_ROOT"] = str(self.finn.finn_root) if self.finn.finn_root else str(self.bsmith_dir)
        env_dict["FINN_BUILD_DIR"] = str(self.finn.finn_build_dir) if self.finn.finn_build_dir else str(self.bsmith_build_dir)
        env_dict["FINN_DEPS_DIR"] = str(self.finn.finn_deps_dir) if self.finn.finn_deps_dir else str(self.bsmith_deps_dir)
        
        if self.finn.num_default_workers:
            env_dict["NUM_DEFAULT_WORKERS"] = str(self.finn.num_default_workers)
        
        return env_dict