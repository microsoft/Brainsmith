# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith configuration schema using Pydantic.

This module defines the complete configuration schema for Brainsmith,
providing type safety and validation.
"""

import os
import warnings
import yaml
from pathlib import Path
from functools import cached_property
from typing import Annotated, Optional, Dict, Any, List, Tuple, Type, Callable
from pydantic import BaseModel, Field, field_validator, ConfigDict, BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
from brainsmith._internal.io.yaml import load_yaml, _deep_merge

from .constants import (
    get_user_config_path,
    PROJECT_CONFIG_FILE,
    PROJECT_CONFIG_ALT_DIR,
    PROJECT_CONFIG_ALT_FILE,
)


# Pydantic validators for Path conversion
def _validate_path(v: Any) -> Optional[Path]:
    """Convert string/Path-like to Path object, handling None for optional fields."""
    if v is None:
        return None
    if not isinstance(v, Path):
        return Path(v)
    return v


# Type aliases for path fields with automatic validation
PathField = Annotated[Path, BeforeValidator(_validate_path)]
OptionalPathField = Annotated[Optional[Path], BeforeValidator(_validate_path)]


# FinnConfig nested model
class FinnConfig(BaseModel):
    """FINN-specific path configuration.

    These paths default to sensible values derived from the parent config:
    - finn_root: defaults to deps_dir/finn
    - finn_build_dir: defaults to build_dir
    - finn_deps_dir: defaults to deps_dir
    """
    finn_root: OptionalPathField = Field(default=None, description="FINN root directory (defaults to deps_dir/finn)")
    finn_build_dir: OptionalPathField = Field(default=None, description="FINN build directory (defaults to build_dir)")
    finn_deps_dir: OptionalPathField = Field(default=None, description="FINN dependencies directory (defaults to deps_dir)")

    model_config = ConfigDict(validate_assignment=True)


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for YAML files with support for user and project configs."""
    
    def __init__(self, settings_cls: Type[BaseSettings], yaml_file: Optional[Path] = None):
        super().__init__(settings_cls)
        self.yaml_files = []
        
        # Load user file first (lower priority)
        if '_BRAINSMITH_USER_FILE' in os.environ:
            self.yaml_files.append(Path(os.environ['_BRAINSMITH_USER_FILE']))
        else:
            # Check default user config location
            user_file = get_user_config_path()
            if user_file.exists():
                self.yaml_files.append(user_file)
        
        # Load project file second (higher priority - will override user config)
        if '_BRAINSMITH_PROJECT_FILE' in os.environ:
            self.yaml_files.append(Path(os.environ['_BRAINSMITH_PROJECT_FILE']))
        else:
            # Find project file in standard locations
            project_file = self._find_project_yaml_file()
            if project_file:
                self.yaml_files.append(project_file)
        
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
            Path.cwd() / PROJECT_CONFIG_FILE,
            Path.cwd() / PROJECT_CONFIG_ALT_DIR / PROJECT_CONFIG_ALT_FILE,
        ]:
            if location.exists():
                return location

        # Then check brainsmith project root
        try:
            import brainsmith
            bsmith_root = Path(brainsmith.__file__).parent.parent
            root_settings = bsmith_root / PROJECT_CONFIG_FILE
            if root_settings.exists():
                return root_settings
        except Exception:
            pass

        return None
    
    def _load_and_merge_yaml_files(self) -> Dict[str, Any]:
        """Load and merge multiple YAML files with proper priority.
        
        Files are loaded in order with later files having higher priority.
        Since we load user config first and project config second,
        project config will override user config values.
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
                        schema_class=self.settings_cls  # Pass the SystemConfig class to extract path fields
                    )
                    # Deep merge the data
                    merged_data = _deep_merge(merged_data, data)
                except yaml.YAMLError as e:
                    # Extract useful error information
                    error_msg = str(e)
                    if hasattr(e, 'problem_mark'):
                        mark = e.problem_mark
                        error_msg = f"line {mark.line + 1}, column {mark.column + 1}: {e.problem or 'invalid syntax'}"

                    warnings.warn(
                        f"Failed to parse config file {yaml_file}: "
                        f"YAML syntax error at {error_msg}. "
                        f"Skipping this configuration file.",
                        UserWarning,
                        stacklevel=2
                    )
                except Exception as e:
                    # For other unexpected errors, still warn but less specifically
                    warnings.warn(
                        f"Failed to read config file {yaml_file}: {e}. "
                        f"Skipping this configuration file.",
                        UserWarning,
                        stacklevel=2
                    )
        
        return merged_data

    def get_field_value(self, field_name: str, field_info: Any) -> Tuple[Any, str, bool]:
        """Get field value from YAML source."""
        if field_name in self._data:
            return self._data[field_name], field_name, True
        
        return None, field_name, False
    
    def __call__(self) -> Dict[str, Any]:
        """Return all settings from YAML file."""
        return self._data


class SystemConfig(BaseSettings):
    """The complete truth about Brainsmith configuration.

    Configuration priority (following pydantic-settings convention):
    1. CLI arguments (passed to constructor) - HIGHEST
    2. Environment variables (BSMITH_* prefix)
    3. Project config (brainsmith_config.yaml)
    4. User config (~/.brainsmith/config.yaml)
    5. Built-in defaults (Field defaults) - LOWEST
    
    Note: We only read BSMITH_* env vars, and only export FINN_* vars
    to avoid configuration feedback loops.
    """

    # ========================================================================
    # User Configuration (Input)
    # ========================================================================
    # These are the Pydantic fields that users can set via:
    # - YAML files (brainsmith_config.yaml)
    # - Environment variables (BSMITH_* prefix)
    # - Constructor arguments (programmatic usage)
    # All path fields accept relative paths in config; they are resolved
    # to absolute paths by the full_* properties below.
    # ========================================================================

    # Core paths
    # NOTE: bsmith_dir is now a cached property, not a configurable field
    build_dir: PathField = Field(
        default=Path("build"),
        description="Build directory for artifacts"
    )
    deps_dir: PathField = Field(
        default=Path("deps"),
        description="Dependencies directory"
    )
    # project_dir is computed (not user-configurable) - always set to brainsmith root
    # It's set in model_post_init as a regular attribute
    plugin_sources: Dict[str, Path] = Field(
        default_factory=lambda: {
            'brainsmith': Path('<builtin>'),
            'finn': Path('<builtin>'),
            'project': Path('<project>'),
            'user': Path.home() / '.brainsmith' / 'plugins'
        },
        description="Plugin source mappings. Built-in sources (brainsmith, finn, project, user) are protected and cannot be overridden."
    )
    default_source: str = Field(
        default='brainsmith',
        description="Default source when component reference has no prefix"
    )

    # Xilinx configuration (flattened)
    xilinx_path: PathField = Field(
        default=Path("/tools/Xilinx"),
        description="Xilinx root installation path"
    )
    xilinx_version: str = Field(
        default="2024.2",
        description="Xilinx tool version"
    )
    # These are auto-derived from xilinx_path but can be overridden
    vivado_path: OptionalPathField = Field(
        default=None,
        description="Path to Vivado (auto-detected from xilinx_path)"
    )
    vitis_path: OptionalPathField = Field(
        default=None,
        description="Path to Vitis (auto-detected from xilinx_path)"
    )
    vitis_hls_path: OptionalPathField = Field(
        default=None,
        description="Path to Vitis HLS (auto-detected from xilinx_path)"
    )
    
    # Tool paths (flattened)
    vendor_platform_paths: str = Field(
        default="/opt/xilinx/platforms",
        description="Vendor platform repository paths (Xilinx/Intel FPGA platforms)"
    )

    # Plugin settings
    plugins_strict: bool = Field(
        default=True,
        description="Strict plugin loading"
    )
    eager_plugin_discovery: bool = Field(
        default=False,
        description="Regenerate all plugin manifests on startup. Enable for development to ensure fresh cache."
    )

    # Vivado-specific settings
    vivado_ip_cache: OptionalPathField = Field(
        default=None,
        description="Vivado IP cache directory (auto-computed from build_dir if not set)"
    )
    
    # Network/visualization settings
    netron_port: int = Field(
        default=8080,
        description="Port for Netron neural network visualization"
    )

    # Worker configuration
    default_workers: int = Field(
        default=4,
        description="Default number of workers for parallel operations (exports to NUM_DEFAULT_WORKERS for FINN)"
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
        # 3. YAML files (custom source) - project config overrides user config
        # 4. Field defaults (built into pydantic)
        return (
            init_settings,
            env_settings,
            YamlSettingsSource(settings_cls),
        )
    
    def model_post_init(self, __context: Any) -> None:
        """Resolve all paths to absolute at init time."""

        # 1. Set project_dir (always brainsmith root, not user-configurable)
        self.project_dir = self.bsmith_dir

        # 2. Resolve core paths
        self.build_dir = self._resolve(self.build_dir, base=self.project_dir)
        self.deps_dir = self._resolve(self.deps_dir, base=self.bsmith_dir)

        # 3. Auto-detect and resolve Xilinx tools (eager)
        if not self.vivado_path:
            self.vivado_path = self._detect_xilinx_tool("Vivado")
        elif self.vivado_path:
            self.vivado_path = self._resolve(self.vivado_path, base=self.project_dir)

        if not self.vitis_path:
            self.vitis_path = self._detect_xilinx_tool("Vitis")
        elif self.vitis_path:
            self.vitis_path = self._resolve(self.vitis_path, base=self.project_dir)

        if not self.vitis_hls_path:
            self.vitis_hls_path = self._detect_xilinx_tool("Vitis_HLS")
        elif self.vitis_hls_path:
            self.vitis_hls_path = self._resolve(self.vitis_hls_path, base=self.project_dir)

        # 4. Resolve derived path: vivado_ip_cache
        if self.vivado_ip_cache:
            self.vivado_ip_cache = self._resolve(self.vivado_ip_cache, base=self.project_dir)
        else:
            self.vivado_ip_cache = self.build_dir / "vivado_ip_cache"

        # 5. Resolve FINN paths (with defaults)
        if self.finn.finn_root:
            self.finn.finn_root = self._resolve(self.finn.finn_root, base=self.project_dir)
        else:
            self.finn.finn_root = self.deps_dir / "finn"

        if self.finn.finn_build_dir:
            self.finn.finn_build_dir = self._resolve(self.finn.finn_build_dir, base=self.project_dir)
        else:
            self.finn.finn_build_dir = self.build_dir

        if self.finn.finn_deps_dir:
            self.finn.finn_deps_dir = self._resolve(self.finn.finn_deps_dir, base=self.project_dir)
        else:
            self.finn.finn_deps_dir = self.deps_dir

        # 6. Resolve plugin sources (expand markers)
        resolved_sources = {}
        for name, path in self.plugin_sources.items():
            resolved_sources[name] = self._resolve_plugin_source(name, path)
        self.plugin_sources = resolved_sources
    
    @cached_property
    def bsmith_dir(self) -> Path:
        """Brainsmith repository root containing pyproject.toml.

        This is the parent of the brainsmith package directory.
        """
        try:
            import brainsmith
            bsmith_dir = Path(brainsmith.__file__).parent.parent

            if not (bsmith_dir / "pyproject.toml").exists():
                raise ValueError(f"Auto-detected directory {bsmith_dir} does not contain pyproject.toml")

            return bsmith_dir
        except ImportError:
            raise ValueError("Cannot determine BSMITH_DIR: brainsmith package not importable")



    @field_validator('plugin_sources', mode='before')
    @classmethod
    def validate_plugin_sources(cls, v: Any) -> Dict[str, Path]:
        """Convert plugin_sources to dict of str -> Path and protect built-in sources."""
        # Protected sources that cannot be overridden
        PROTECTED_SOURCES = {'brainsmith', 'finn', 'project', 'user'}

        # Default protected values
        protected_defaults = {
            'brainsmith': Path('<builtin>'),
            'finn': Path('<builtin>'),
            'project': Path('<project>'),
            'user': Path.home() / '.brainsmith' / 'plugins'
        }

        if v is None:
            return protected_defaults

        if not isinstance(v, dict):
            raise ValueError("plugin_sources must be a dictionary mapping source names to paths")

        # Start with protected defaults
        result = protected_defaults.copy()

        # Add user-provided sources, but reject attempts to override protected ones
        for key, value in v.items():
            # Convert value to Path if needed
            if isinstance(value, str):
                value = Path(value)
            elif not isinstance(value, Path):
                raise ValueError(f"plugin_sources['{key}'] must be a string or Path, got {type(value)}")

            if key in PROTECTED_SOURCES:
                # Check if they're trying to override with a different value
                # We need to accept both sentinel paths AND their resolved forms
                # (model_post_init resolves '<builtin>' -> actual paths)
                value_str = str(value)
                default_str = str(protected_defaults[key])

                # Allow if it matches the sentinel value
                if value_str == default_str:
                    result[key] = value  # Use the provided value (might be Path vs PosixPath)
                    continue

                # Also allow if it's a resolved builtin/project path
                # (happens during model_post_init path resolution)
                if default_str == '<builtin>' or default_str == '<project>':
                    # Accept any path - these are being resolved by model_post_init
                    # We trust the resolution logic, not re-validating here
                    result[key] = value
                    continue

                # Otherwise reject the override
                raise ValueError(
                    f"Cannot override protected plugin source '{key}'. "
                    f"Protected sources are: {', '.join(sorted(PROTECTED_SOURCES))}"
                )
            else:
                # Add custom (non-protected) source
                result[key] = value

        return result

    # ========================================================================
    # Path Resolution Helpers
    # ========================================================================

    @staticmethod
    def _resolve(path: Path, base: Path) -> Path:
        """Resolve relative paths to absolute using base."""
        return path if path.is_absolute() else (base / path).resolve()

    def _detect_xilinx_tool(self, tool_name: str) -> Optional[Path]:
        """Auto-detect Xilinx tool path from xilinx_path/tool_name/version."""
        if not self.xilinx_path or not self.xilinx_path.exists():
            return None

        tool_path = self.xilinx_path / tool_name / self.xilinx_version
        return tool_path if tool_path.exists() else None

    def _resolve_plugin_source(self, name: str, path: Path) -> Path:
        """Resolve plugin source path, expanding <builtin> and <project> markers."""
        path_str = str(path)

        # Expand <builtin> marker
        if path_str == '<builtin>':
            if name == 'brainsmith':
                return self.bsmith_dir / 'brainsmith'
            elif name in ('finn', 'qonnx'):
                return self.deps_dir / name
            else:
                raise ValueError(
                    f"Unknown builtin source '{name}'. "
                    f"Builtin sources: brainsmith, finn, qonnx"
                )

        # Expand <project> marker
        if path_str == '<project>':
            return self.project_dir / 'plugins'

        # Regular path resolution
        return self._resolve(path, base=self.project_dir)

    # Environment Export (delegated to EnvironmentExporter for separation of concerns)

    def export_to_environment(self, **kwargs) -> Dict[str, str]:
        """Export configuration to environment variables.

        Delegates to EnvironmentExporter for actual export logic.
        See EnvironmentExporter.export_to_environment() for full documentation.

        Args:
            **kwargs: Passed to EnvironmentExporter.export_to_environment()
                     (include_internal, verbose, export)

        Returns:
            Dict of exported environment variables
        """
        from .env_export import EnvironmentExporter
        return EnvironmentExporter(self).export_to_environment(**kwargs)
