# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith configuration schema using Pydantic.

This module defines the complete configuration schema for Brainsmith,
providing type safety and validation.

Path Resolution Rules
---------------------
The settings system follows one simple principle for paths:
"Paths resolve relative to where they're specified"

1. **Full paths**: Always used as-is
2. **Relative paths from CLI**: Resolve to current working directory
3. **Relative paths from YAML/env/defaults**: Resolve to project directory

Examples:
    CLI:  brainsmith --build-dir build           → resolves to $PWD/build
    YAML: build_dir: build                       → resolves to $PROJECT_DIR/build
    Env:  BSMITH_BUILD_DIR=build                → resolves to $PROJECT_DIR/build

Project directory is detected by walking up from CWD to find .brainsmith/config.yaml.

Internal paths (deps_dir) always resolve to brainsmith installation, ignoring user input.

Configuration Priority
----------------------
Settings are loaded from multiple sources with the following priority (highest to lowest):
1. CLI arguments (passed to SystemConfig constructor)
2. Environment variables (BSMITH_* prefix)
3. Project config file (.brainsmith/config.yaml)
4. User config file (~/.brainsmith/config.yaml)
5. Built-in defaults (Field defaults in SystemConfig)

Path Resolution Flow
--------------------
1. CLI paths are resolved to CWD in load_config() before SystemConfig is created
2. All other paths stay relative through Pydantic's settings sources
3. model_post_init() resolves any remaining relative paths to project_dir
4. Special paths (deps_dir) are forced to specific locations regardless of input
"""

import os
import yaml
from pathlib import Path
from functools import cached_property
from typing import (
    Annotated, Optional, Dict, Any, List, Tuple, Type, Callable, Union
)
from pydantic import (
    BaseModel, Field, field_validator, ConfigDict, BeforeValidator
)
from pydantic_settings import (
    BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
)
from brainsmith._internal.io.yaml import deep_merge, expand_env_vars
from brainsmith.registry.constants import (
    CORE_NAMESPACE,
    KNOWN_ENTRY_POINTS,
    PROTECTED_SOURCES,
    SOURCE_BRAINSMITH,
    SOURCE_FINN,
    SOURCE_PROJECT,
    SOURCE_USER,
    DEFAULT_SOURCE_PRIORITY,
)


# Private constants for config file discovery
# These are internal implementation details - CLI tools should inline values as needed
_USER_CONFIG_PATH = Path.home() / ".brainsmith" / "config.yaml"
_PROJECT_CONFIG_DIR = ".brainsmith"
_PROJECT_CONFIG_FILE = "config.yaml"


def _find_project_config() -> Optional[Path]:
    """Find project configuration file with upward directory walk.

    Search order:
    1. If BSMITH_PROJECT_DIR is set, check that directory only
    2. Otherwise, walk up from CWD to find .brainsmith/config.yaml

    Project directory is always the parent of .brainsmith/

    Returns:
        Path to config file, or None if not found
    """
    # Priority 1: Explicit project directory override
    if project_dir_override := os.environ.get('BSMITH_PROJECT_DIR'):
        project_dir = Path(project_dir_override).resolve()
        candidate = project_dir / _PROJECT_CONFIG_DIR / _PROJECT_CONFIG_FILE

        if candidate.exists():
            return candidate

        # If BSMITH_PROJECT_DIR is set but no config found, return None
        # (don't fall through to upward walk - user explicitly set the location)
        return None

    # Priority 2: Walk up from CWD to find .brainsmith/config.yaml
    current = Path.cwd().resolve()

    # Walk up until we hit filesystem root
    while current != current.parent:
        candidate = current / _PROJECT_CONFIG_DIR / _PROJECT_CONFIG_FILE

        # Skip if this is the user config location (not a project config)
        if candidate.resolve() == _USER_CONFIG_PATH.resolve():
            current = current.parent
            continue

        if candidate.exists():
            return candidate

        current = current.parent

    # Reached filesystem root without finding config
    return None


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for YAML files with support for user and project configs."""

    def __init__(
        self,
        settings_cls: Type[BaseSettings],
        user_file: Optional[Path] = None,
        project_file: Optional[Path] = None
    ):
        super().__init__(settings_cls)
        self.yaml_files = []
        self.project_file_used = None  # Track which project file was actually loaded

        # Load user file first (lower priority)
        if user_file and user_file.exists():
            # Explicit user file provided and exists
            self.yaml_files.append(user_file)
        elif not user_file:
            # Check default user config location only if no explicit file provided
            default_user_file = _USER_CONFIG_PATH
            if default_user_file.exists():
                self.yaml_files.append(default_user_file)

        # Load project file second (higher priority - will override user config)
        if project_file and project_file.exists():
            # Explicit project file provided and exists
            self.yaml_files.append(project_file)
            self.project_file_used = project_file
        elif not project_file:
            # Find project file in standard locations only if no explicit file provided
            found_project_file = _find_project_config()
            if found_project_file:
                self.yaml_files.append(found_project_file)
                self.project_file_used = found_project_file

        # Load and merge all YAML files
        self._data = self._load_and_merge_yaml_files()

    def _load_and_merge_yaml_files(self) -> Dict[str, Any]:
        """Load and merge multiple YAML files with proper priority.

        Files are loaded in order with later files having higher priority.
        Since we load user config first and project config second,
        project config will override user config values.

        Raises:
            yaml.YAMLError: If any config file has syntax errors
            FileNotFoundError: If a specified config file doesn't exist
        """
        merged_data = {}

        for yaml_file in self.yaml_files:
            try:
                # Load and expand env vars inline (no path resolution)
                with open(yaml_file) as f:
                    data = yaml.safe_load(f) or {}
                data = expand_env_vars(data)

                # Deep merge the data
                merged_data = deep_merge(merged_data, data)
            except yaml.YAMLError as e:
                # Extract location info if available
                if hasattr(e, 'problem_mark'):
                    mark = e.problem_mark
                    location = f"line {mark.line + 1}, column {mark.column + 1}"
                else:
                    location = "unknown location"

                # Fail fast with clear error message
                raise yaml.YAMLError(
                    f"\n\nInvalid YAML in config file: {yaml_file}\n"
                    f"Error at {location}: {e.problem or str(e)}\n\n"
                    f"Fix the syntax error and try again."
                ) from e

        return merged_data

    def get_field_value(self, field_name: str, field_info: Any) -> Tuple[Any, str, bool]:
        """Get field value from YAML source."""
        if field_name in self._data:
            return self._data[field_name], field_name, True

        # Special handling for _project_file_used
        if field_name == '_project_file_used' and self.project_file_used:
            return self.project_file_used, field_name, True

        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        """Return all settings from YAML file."""
        data = self._data.copy()
        # Include project file info for project_dir detection
        if self.project_file_used:
            data['_project_file_used'] = self.project_file_used
        return data


class SystemConfig(BaseSettings):
    """The complete truth about Brainsmith configuration.

    Configuration priority (following pydantic-settings convention):
    1. CLI arguments (passed to constructor) - HIGHEST
    2. Environment variables (BSMITH_* prefix)
    3. Project config (.brainsmith/config.yaml)
    4. User config (~/.brainsmith/config.yaml)
    5. Built-in defaults (Field defaults) - LOWEST

    Note: We only read BSMITH_* env vars, and only export FINN_* vars
    to avoid configuration feedback loops.
    """

    # NOTE: bsmith_dir is now a cached property, not a configurable field
    build_dir: Path = Field(
        default=Path("build"),
        description="Build directory for artifacts"
    )
    deps_dir: Path = Field(
        default=Path("deps"),
        description="Dependencies directory"
    )
    # project_dir is computed (not user-configurable) - always set to brainsmith root
    # It's set in model_post_init as a regular attribute
    component_sources: Dict[str, Path | None] = Field(
        default_factory=lambda: {
            SOURCE_PROJECT: None,  # Resolved to project_dir / "plugins"
            SOURCE_USER: None      # Resolved to ~/.brainsmith/plugins
        },
        description=(
            "Filesystem-based component source paths. Maps source name to directory path. "
            "Standard sources 'project' and 'user' have default paths. "
            "Core namespace 'brainsmith' and entry point sources (e.g., 'finn') are "
            "loaded automatically and cannot be configured here."
        )
    )
    source_priority: List[str] = Field(
        default=list(DEFAULT_SOURCE_PRIORITY),
        description=(
            "Component source resolution priority. First source with matching "
            "component wins. Custom sources are auto-appended if not explicitly listed."
        )
    )

    # Xilinx configuration (flattened)
    xilinx_path: Path = Field(
        default=Path("/tools/Xilinx"),
        description="Xilinx root installation path"
    )
    xilinx_version: str = Field(
        default="2024.2",
        description="Xilinx tool version"
    )
    vivado_path: Path | None = Field(
        default=None,
        description="Path to Vivado (auto-detected from xilinx_path)"
    )
    vitis_path: Path | None = Field(
        default=None,
        description="Path to Vitis (auto-detected from xilinx_path)"
    )
    vitis_hls_path: Path | None = Field(
        default=None,
        description="Path to Vitis HLS (auto-detected from xilinx_path)"
    )

    vendor_platform_paths: str = Field(
        default="/opt/xilinx/platforms",
        description="Vendor platform repository paths (Xilinx/Intel FPGA platforms)"
    )

    components_strict: bool = Field(
        default=True,
        description="Strict component loading"
    )

    cache_components: bool = Field(
        default=True,
        description=(
            "Enable manifest caching for component discovery. When True, generates and "
            "uses a manifest cache in .brainsmith/ to speed up startup. Cache "
            "auto-invalidates when component files are modified. Set to False to always "
            "perform eager discovery."
        )
    )

    vivado_ip_cache: Path | None = Field(
        default=None,
        description="Vivado IP cache directory (auto-computed from build_dir if not set)"
    )

    netron_port: int = Field(
        default=8080,
        description="Port for Netron neural network visualization"
    )

    default_workers: int = Field(
        default=4,
        description=(
            "Default number of workers for parallel operations "
            "(exports to NUM_DEFAULT_WORKERS for FINN)"
        )
    )

    finn_root: Path | None = Field(
        default=None,
        description="FINN root directory (defaults to deps_dir/finn)"
    )
    finn_build_dir: Path | None = Field(
        default=None,
        description="FINN build directory (defaults to build_dir)"
    )
    finn_deps_dir: Path | None = Field(
        default=None,
        description="FINN dependencies directory (defaults to deps_dir)"
    )

    model_config = SettingsConfigDict(
        env_prefix="BSMITH_",
        env_nested_delimiter="__",
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
        """Customize settings sources.

        Priority order (first source wins):
        1. Init settings (CLI/constructor args) - paths already resolved to CWD
        2. Environment variables (BSMITH_*) - paths stay relative, resolved in model_post_init
        3. YAML files (custom source) - paths stay relative, resolved in model_post_init
        4. Field defaults (built into pydantic)

        Path Resolution:
        - CLI paths are resolved to CWD in load_config() before reaching here
        - All other paths stay relative and are resolved in model_post_init()
        """
        # Extract file paths from init_settings if provided
        init_dict = init_settings()
        user_file = init_dict.get('_user_file')
        project_file = init_dict.get('_project_file')

        return (
            init_settings,  # CLI args (already resolved to CWD in load_config)
            env_settings,   # Env vars (stay relative, resolved in model_post_init)
            YamlSettingsSource(settings_cls, user_file=user_file, project_file=project_file),
        )

    def model_post_init(self, __context: Any) -> None:
        """Resolve all paths to absolute.

        At this point:
        - CLI paths are already absolute (resolved to CWD in load_config)
        - YAML/env/default paths may be relative (need resolution to project_dir)

        Resolution steps:
        1. Detect project_dir (where config file is, or CWD)
        2. Resolve user-facing paths (relative → project_dir)
        3. Force internal paths (deps_dir → bsmith_dir)
        4. Set defaults for unset paths
        5. Validate everything is sane
        """
        self.project_dir = self._detect_project_root()
        self._resolve_core_paths()
        self._resolve_xilinx_tools()
        self._resolve_finn_paths()
        self._resolve_component_sources()
        self._resolve_source_priority()

    def _resolve_core_paths(self) -> None:
        """Resolve build_dir and force deps_dir to internal location."""
        self.build_dir = self._resolve(self.build_dir, self.project_dir)

        # deps_dir: ALWAYS internal to brainsmith installation (ignore user input)
        # This ensures dependencies (FINN, brevitas, etc.) stay with brainsmith package
        self.deps_dir = self.bsmith_dir / "deps"

    def _resolve_xilinx_tools(self) -> None:
        """Auto-detect or resolve Xilinx tool paths."""
        if self.vivado_path is None:
            self.vivado_path = self._detect_xilinx_tool("Vivado")
        else:
            self.vivado_path = self._resolve(self.vivado_path, self.project_dir)

        if self.vitis_path is None:
            self.vitis_path = self._detect_xilinx_tool("Vitis")
        else:
            self.vitis_path = self._resolve(self.vitis_path, self.project_dir)

        if self.vitis_hls_path is None:
            self.vitis_hls_path = self._detect_xilinx_tool("Vitis_HLS")
        else:
            self.vitis_hls_path = self._resolve(self.vitis_hls_path, self.project_dir)

        if self.vivado_ip_cache:
            self.vivado_ip_cache = self._resolve(self.vivado_ip_cache, self.project_dir)
        else:
            self.vivado_ip_cache = self.build_dir / "vivado_ip_cache"

    def _resolve_finn_paths(self) -> None:
        """Resolve FINN paths with sensible defaults."""
        if self.finn_root:
            self.finn_root = self._resolve(self.finn_root, self.project_dir)
        else:
            self.finn_root = self.deps_dir / "finn"

        if self.finn_build_dir:
            self.finn_build_dir = self._resolve(self.finn_build_dir, self.project_dir)
        else:
            self.finn_build_dir = self.build_dir

        if self.finn_deps_dir:
            self.finn_deps_dir = self._resolve(self.finn_deps_dir, self.project_dir)
        else:
            self.finn_deps_dir = self.deps_dir

    def _resolve_component_sources(self) -> None:
        """Resolve component source paths for filesystem-based sources.

        Only project, user, and custom sources are filesystem-based and configurable.
        Core namespace (brainsmith) and entry points (finn) are discovered automatically.
        """
        # Standard filesystem sources with default paths
        if self.component_sources.get(SOURCE_PROJECT) is None:
            self.component_sources[SOURCE_PROJECT] = self.project_dir / 'plugins'

        if self.component_sources.get(SOURCE_USER) is None:
            self.component_sources[SOURCE_USER] = Path.home() / '.brainsmith' / 'plugins'

        # Custom sources: resolve relative paths
        # Standard sources (project, user) are resolved above
        standard_sources = {SOURCE_PROJECT, SOURCE_USER}
        for name, path in list(self.component_sources.items()):
            if name not in standard_sources and path is not None:
                self.component_sources[name] = self._resolve(path, self.project_dir)

    def _resolve_source_priority(self) -> None:
        """Auto-append custom component sources to source_priority if not already listed.

        This ensures custom sources work automatically while allowing users to
        explicitly position them in the priority list if desired.
        """
        for source_name in self.component_sources.keys():
            if source_name not in self.source_priority:
                self.source_priority.append(source_name)

    @cached_property
    def bsmith_dir(self) -> Path:
        """Brainsmith repository root containing pyproject.toml.

        This is the parent of the brainsmith package directory.
        """
        try:
            import brainsmith
            bsmith_dir = Path(brainsmith.__file__).parent.parent

            if not (bsmith_dir / "pyproject.toml").exists():
                raise ValueError(
                    f"Auto-detected directory {bsmith_dir} does not contain pyproject.toml"
                )

            return bsmith_dir
        except ImportError:
            raise ValueError("Cannot determine BSMITH_DIR: brainsmith package not importable")

    def _detect_project_root(self) -> Path:
        """Detect user project root with upward walk from CWD.

        Project root detection priority:
        1. BSMITH_PROJECT_DIR env var (explicit override, no config check)
        2. Custom project file location (from load_config project_file param)
        3. Walk up from CWD to find config file, return its parent directory
        4. Fallback to CWD

        Returns the same project_dir regardless of which subdirectory you're in,
        providing stable path resolution.

        Returns:
            Absolute path to detected project root
        """
        if 'BSMITH_PROJECT_DIR' in os.environ:
            return Path(os.environ['BSMITH_PROJECT_DIR']).resolve()

        project_file_used = getattr(self, '_project_file_used', None)
        if project_file_used:
            return Path(project_file_used).parent.parent.resolve()

        config_file = _find_project_config()
        if config_file:
            return config_file.parent.parent

        return self.bsmith_dir

    @field_validator('component_sources', mode='before')
    @classmethod
    def validate_component_sources(cls, v: Any) -> Dict[str, Path | None]:
        """Validate component sources and warn about reserved source names.

        Reserved source names:
        - Core namespace ('brainsmith'): Internal components loaded via direct import
        - Entry points (e.g., 'finn'): Discovered via pip package entry points

        Users can configure filesystem-based sources (project, user, custom) but
        cannot override reserved names.
        """
        import logging
        logger = logging.getLogger(__name__)

        default_sources = cls.model_fields['component_sources'].default_factory()

        if v is None or not isinstance(v, dict):
            return default_sources

        result = default_sources.copy()

        for key, value in v.items():
            # Warn if trying to override core namespace
            if key == CORE_NAMESPACE and value is not None:
                logger.warning(
                    f"Component source '{key}' is a reserved core namespace and cannot be "
                    f"configured. The core brainsmith components are loaded from the package "
                    f"installation automatically. This configuration will be ignored."
                )
                continue  # Skip, don't add to result

            # Warn if trying to override known entry point sources
            if key in KNOWN_ENTRY_POINTS and value is not None:
                logger.warning(
                    f"Component source '{key}' is a registered entry point and cannot be "
                    f"configured. Entry point sources are discovered automatically from "
                    f"installed packages. This configuration will be ignored."
                )
                continue  # Skip, don't add to result

            # Add custom or standard filesystem sources
            result[key] = Path(value) if isinstance(value, str) else value

        return result

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
