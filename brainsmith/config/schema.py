"""Brainsmith configuration schema using Pydantic.

This module defines the complete configuration schema for Brainsmith,
providing type safety, validation, and clear priority resolution.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class PathConfig(BaseModel):
    """Build and dependency path configuration."""
    deps_dir: Path = Field(default=Path("deps"), description="Dependencies directory")
    build_dir: str = Field(default="/tmp/{user}_brainsmith", description="Build directory template")
    
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



class NetworkConfig(BaseModel):
    """Network configuration."""
    netron_port: str = Field(default="8082", description="Default Netron visualization port")
    
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
    "paths": ConfigPriority.OPTIONAL,
    "xilinx": ConfigPriority.OPTIONAL,
    "xilinx.vivado_path": ConfigPriority.OPTIONAL,
    "xilinx.vitis_path": ConfigPriority.OPTIONAL,
    "xilinx.hls_path": ConfigPriority.OPTIONAL,
    "tools": ConfigPriority.TERTIARY,
    "compiler": ConfigPriority.TERTIARY,
    "network": ConfigPriority.TERTIARY,
    "dependencies": ConfigPriority.TERTIARY,
    "network": ConfigPriority.TERTIARY,
    "debug": ConfigPriority.TERTIARY,
    "finn": ConfigPriority.OPTIONAL,
}


class BrainsmithConfig(BaseSettings):
    """The complete truth about Brainsmith configuration.
    
    Configuration priority:
    1. Built-in defaults
    2. Environment variables
    3. Project settings (brainsmith_settings.yaml)
    4. CLI arguments
    """
    
    # Core paths - required
    bsmith_dir: Path = Field(
        ..., 
        description="Root directory of Brainsmith",
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
    paths: PathConfig = Field(default_factory=PathConfig)
    xilinx: XilinxConfig = Field(default_factory=XilinxConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    dependencies: DependencyConfig = Field(default_factory=DependencyConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    finn: FinnConfig = Field(default_factory=FinnConfig)
    
    # Additional top-level fields
    hw_compiler: str = Field(default="finn", validation_alias="BSMITH_HW_COMPILER")
    plugins_strict: bool = Field(default=True, validation_alias="BSMITH_PLUGINS_STRICT")
    fetch_boards: bool = Field(default=True, validation_alias="BSMITH_FETCH_BOARDS")
    fetch_experimental: bool = Field(default=False, validation_alias="BSMITH_FETCH_EXPERIMENTAL")
    
    model_config = SettingsConfigDict(
        env_prefix="BSMITH_",
        env_nested_delimiter="__",
        validate_assignment=True,
        case_sensitive=False,
        extra="allow"
    )
    
    @field_validator('bsmith_dir')
    @classmethod
    def validate_bsmith_dir(cls, v: Path) -> Path:
        """Validate that BSMITH_DIR exists and contains pyproject.toml."""
        if not v.exists():
            raise ValueError(f"BSMITH_DIR {v} does not exist")
        if not (v / "pyproject.toml").exists():
            raise ValueError(f"BSMITH_DIR {v} does not contain pyproject.toml")
        return v.absolute()
    
    @field_validator('bsmith_build_dir', 'bsmith_deps_dir')
    @classmethod
    def resolve_paths(cls, v: Path, info) -> Path:
        """Resolve relative paths against BSMITH_DIR."""
        if not v.is_absolute() and 'bsmith_dir' in info.data:
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
        
        # Network
        env_dict["NETRON_PORT"] = self.network.netron_port
        
        # Dependencies
        env_dict["BSMITH_FETCH_BOARDS"] = "true" if self.fetch_boards else "false"
        env_dict["BSMITH_FETCH_EXPERIMENTAL"] = "true" if self.fetch_experimental else "false"
        
        # Debug
        env_dict["BSMITH_DEBUG"] = "1" if self.debug.enabled else "0"
        
        # FINN compatibility
        env_dict["FINN_ROOT"] = str(self.finn.finn_root) if self.finn.finn_root else str(self.bsmith_dir)
        env_dict["FINN_BUILD_DIR"] = str(self.finn.finn_build_dir) if self.finn.finn_build_dir else str(self.bsmith_build_dir)
        env_dict["FINN_DEPS_DIR"] = str(self.finn.finn_deps_dir) if self.finn.finn_deps_dir else str(self.bsmith_deps_dir)
        
        if self.finn.num_default_workers:
            env_dict["NUM_DEFAULT_WORKERS"] = str(self.finn.num_default_workers)
        
        return env_dict