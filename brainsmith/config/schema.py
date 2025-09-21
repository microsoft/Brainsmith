"""Brainsmith configuration schema using Pydantic.

This module defines the complete configuration schema for Brainsmith,
providing type safety, validation, and clear priority resolution.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Type
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
import yaml


class ConfigPriority(str, Enum):
    """Configuration priority levels."""
    CORE = "core"           # Required - error if missing
    OPTIONAL = "optional"   # Warn if missing  
    TERTIARY = "tertiary"   # Silent if missing


class PythonConfig(BaseModel):
    """Python environment configuration."""
    version: str = Field(default="python3.10", description="Python executable version")
    unbuffered: bool = Field(default=True, description="Python unbuffered output")
    
    model_config = ConfigDict(validate_assignment=True)


class XilinxConfig(BaseModel):
    """Xilinx/AMD tool configuration."""
    vivado_path: Optional[Path] = Field(default=None, description="Path to Vivado installation")
    vitis_path: Optional[Path] = Field(default=None, description="Path to Vitis installation") 
    hls_path: Optional[Path] = Field(default=None, description="Path to Vitis HLS installation")
    version: str = Field(default="2024.2", description="Default Xilinx version")
    # Legacy support
    xilinx_path: Optional[Path] = Field(default=None, description="Legacy Xilinx root path (deprecated)")
    
    @field_validator('vivado_path', 'vitis_path', 'hls_path')
    @classmethod
    def validate_tool_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate Xilinx tool installation paths."""
        if v is not None and v.exists():
            settings_file = v / "settings64.sh"
            if not settings_file.exists():
                raise ValueError(f"Invalid Xilinx installation at {v} - missing settings64.sh")
        return v
    
    model_config = ConfigDict(validate_assignment=True)


class ToolConfig(BaseModel):
    """Tool path configuration."""
    platform_repo_paths: str = Field(default="/opt/xilinx/platforms", description="Platform repository paths")
    ohmyxilinx_path: Optional[Path] = Field(default=None, description="oh-my-xilinx path override")
    
    model_config = ConfigDict(validate_assignment=True)


class DependencyConfig(BaseModel):
    """Dependency fetching configuration."""
    fetch_boards: bool = Field(default=True, description="Fetch board files")
    fetch_experimental: bool = Field(default=False, description="Fetch experimental dependencies")
    
    model_config = ConfigDict(validate_assignment=True)



class DebugConfig(BaseModel):
    """Debug configuration."""
    enabled: bool = Field(default=False, description="Enable debug output")
    
    model_config = ConfigDict(validate_assignment=True)


class FinnConfig(BaseModel):
    """FINN-specific path configuration."""
    finn_root: Optional[Path] = Field(default=None, description="FINN root directory")
    finn_build_dir: Optional[Path] = Field(default=None, description="FINN build directory")
    finn_deps_dir: Optional[Path] = Field(default=None, description="FINN dependencies directory")
    num_default_workers: Optional[int] = Field(default=None, description="Default number of workers")
    
    model_config = ConfigDict(validate_assignment=True)


# Priority metadata for fields
FIELD_PRIORITIES = {
    "bsmith_dir": ConfigPriority.CORE,
    "bsmith_build_dir": ConfigPriority.OPTIONAL,
    "bsmith_deps_dir": ConfigPriority.OPTIONAL,
    "python": ConfigPriority.TERTIARY,
    "xilinx": ConfigPriority.OPTIONAL,
    "xilinx.vivado_path": ConfigPriority.OPTIONAL,
    "xilinx.vitis_path": ConfigPriority.OPTIONAL,
    "xilinx.hls_path": ConfigPriority.OPTIONAL,
    "tools": ConfigPriority.TERTIARY,
    "compiler": ConfigPriority.TERTIARY,
    "dependencies": ConfigPriority.TERTIARY,
    "debug": ConfigPriority.TERTIARY,
    "finn": ConfigPriority.OPTIONAL,
}


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
        
        # Handle nested fields
        if field_name in data:
            return data[field_name], field_name, True
        
        # Not found
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
        validation_alias="BSMITH_DIR"
    )
    bsmith_build_dir: Path = Field(
        default=Path("/tmp/brainsmith_build"),
        description="Build directory for artifacts",
        validation_alias="BSMITH_BUILD_DIR"
    )
    bsmith_deps_dir: Path = Field(
        default=Path("deps"),
        description="Dependencies directory",
        validation_alias="BSMITH_DEPS_DIR"
    )
    
    # Nested configurations
    python: PythonConfig = Field(default_factory=PythonConfig)
    xilinx: XilinxConfig = Field(default_factory=XilinxConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    dependencies: DependencyConfig = Field(default_factory=DependencyConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    finn: FinnConfig = Field(default_factory=FinnConfig)
    
    # Additional top-level fields
    hw_compiler: str = Field(
        default="finn", 
        description="Hardware compiler backend",
        validation_alias="BSMITH_HW_COMPILER"
    )
    plugins_strict: bool = Field(
        default=True, 
        description="Strict plugin loading",
        validation_alias="BSMITH_PLUGINS_STRICT"
    )
    
    model_config = SettingsConfigDict(
        env_prefix='BSMITH_',
        env_nested_delimiter='__',
        validate_assignment=True,
        extra="allow",  # For forward compatibility
        case_sensitive=False,
        # Auto-detect BSMITH_DIR if not set
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
        
        # Handle Xilinx legacy paths
        if self.xilinx.xilinx_path and not self.xilinx.vivado_path and self.xilinx.xilinx_path.exists():
            xilinx_version = self.xilinx.version
            vivado_path = self.xilinx.xilinx_path / "Vivado" / xilinx_version
            if vivado_path.exists():
                self.xilinx.vivado_path = vivado_path
            
            vitis_path = self.xilinx.xilinx_path / "Vitis" / xilinx_version
            if vitis_path.exists():
                self.xilinx.vitis_path = vitis_path
                
            hls_path = self.xilinx.xilinx_path / "Vitis_HLS" / xilinx_version
            if hls_path.exists():
                self.xilinx.hls_path = hls_path
    
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
    
    def validate_by_priority(self) -> Dict[str, List[str]]:
        """Validate configuration based on priority levels.
        
        Returns:
            Dict with 'errors', 'warnings', and 'info' lists
        """
        results = {"errors": [], "warnings": [], "info": []}
        
        # Check core fields
        if not self.bsmith_dir or not self.bsmith_dir.exists():
            results["errors"].append("BSMITH_DIR is required and must exist")
        
        # Check optional fields (Xilinx tools)
        if not self.xilinx.vivado_path:
            results["warnings"].append("Vivado not found - hardware compilation will not be available")
        if not self.xilinx.vitis_path:
            results["warnings"].append("Vitis not found - some features may be limited")
        if not self.xilinx.hls_path:
            results["warnings"].append("Vitis HLS not found - HLS compilation will not be available")
        
        # Check FINN paths
        if not self.finn.finn_root:
            results["info"].append("FINN_ROOT not set - will be auto-detected if needed")
        
        return results
    
    def get_priority(self, field_path: str) -> ConfigPriority:
        """Get the priority level for a field."""
        return FIELD_PRIORITIES.get(field_path, ConfigPriority.TERTIARY)
    
    def to_env_dict(self) -> Dict[str, str]:
        """Convert configuration to environment variable dictionary."""
        env_dict = {}
        
        # Core paths
        env_dict["BSMITH_DIR"] = str(self.bsmith_dir)
        env_dict["BSMITH_BUILD_DIR"] = str(self.bsmith_build_dir)
        env_dict["BSMITH_DEPS_DIR"] = str(self.bsmith_deps_dir)
        
        # Python
        env_dict["PYTHON"] = self.python.version
        env_dict["PYTHONUNBUFFERED"] = "1" if self.python.unbuffered else "0"
        env_dict["BSMITH_PLUGINS_STRICT"] = "true" if self.plugins_strict else "false"
        
        # Xilinx tools
        if self.xilinx.vivado_path:
            env_dict["XILINX_VIVADO"] = str(self.xilinx.vivado_path)
            env_dict["VIVADO_PATH"] = str(self.xilinx.vivado_path)
        if self.xilinx.vitis_path:
            env_dict["XILINX_VITIS"] = str(self.xilinx.vitis_path)
            env_dict["VITIS_PATH"] = str(self.xilinx.vitis_path)
        if self.xilinx.hls_path:
            env_dict["XILINX_HLS"] = str(self.xilinx.hls_path)
            env_dict["HLS_PATH"] = str(self.xilinx.hls_path)
        
        # Tools  
        # OHMYXILINX is required by FINN - use override or default to deps/oh-my-xilinx
        ohmyxilinx = self.tools.ohmyxilinx_path or (self.bsmith_deps_dir / "oh-my-xilinx")
        env_dict["OHMYXILINX"] = str(ohmyxilinx)
        env_dict["PLATFORM_REPO_PATHS"] = self.tools.platform_repo_paths
        
        # Compiler
        env_dict["BSMITH_HW_COMPILER"] = self.hw_compiler
        
        # Dependencies
        env_dict["BSMITH_FETCH_BOARDS"] = "true" if self.dependencies.fetch_boards else "false"
        env_dict["BSMITH_FETCH_EXPERIMENTAL"] = "true" if self.dependencies.fetch_experimental else "false"
        
        # Debug
        env_dict["BSMITH_DEBUG"] = "1" if self.debug.enabled else "0"
        
        # FINN compatibility
        env_dict["FINN_ROOT"] = str(self.finn.finn_root) if self.finn.finn_root else str(self.bsmith_dir)
        env_dict["FINN_BUILD_DIR"] = str(self.finn.finn_build_dir) if self.finn.finn_build_dir else str(self.bsmith_build_dir)
        env_dict["FINN_DEPS_DIR"] = str(self.finn.finn_deps_dir) if self.finn.finn_deps_dir else str(self.bsmith_deps_dir)
        
        if self.finn.num_default_workers:
            env_dict["NUM_DEFAULT_WORKERS"] = str(self.finn.num_default_workers)
        
        return env_dict