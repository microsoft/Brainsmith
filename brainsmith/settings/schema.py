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

Project directory is detected by walking up from CWD to find brainsmith.yaml.

Internal paths (deps_dir) always resolve to brainsmith installation, ignoring user input.

Configuration Priority
----------------------
Settings are loaded from multiple sources with the following priority (highest to lowest):
1. CLI arguments (passed to SystemConfig constructor)
2. Environment variables (BSMITH_* prefix)
3. Project config file (brainsmith.yaml)
4. Built-in defaults (Field defaults in SystemConfig)

Path Resolution Flow
--------------------
1. CLI paths are resolved to CWD in load_config() before SystemConfig is created
2. All other paths stay relative through Pydantic's settings sources
3. model_post_init() resolves any remaining relative paths to project_dir
4. Special paths (deps_dir) are forced to specific locations regardless of input
"""

import logging
import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from brainsmith._internal.io.yaml import deep_merge, expand_env_vars
from brainsmith.registry.constants import (
    CORE_NAMESPACE,
    DEFAULT_SOURCE_PRIORITY,
    KNOWN_ENTRY_POINTS,
    SOURCE_MODULE_PREFIXES,
    SOURCE_PROJECT,
)

# Private constants for config file discovery
# These are internal implementation details - CLI tools should inline values as needed
_PROJECT_CONFIG_FILE = "brainsmith.yaml"


def _find_project_config() -> Path | None:
    """Find project configuration file with upward directory walk.

    Search order:
    1. If BSMITH_PROJECT_DIR is set, check that directory only
    2. Otherwise, walk up from CWD to find brainsmith.yaml

    Project directory is where brainsmith.yaml is located.

    Returns:
        Path to config file, or None if not found
    """
    # Priority 1: Explicit project directory override
    if project_dir_override := os.environ.get("BSMITH_PROJECT_DIR"):
        project_dir = Path(project_dir_override).resolve()
        candidate = project_dir / _PROJECT_CONFIG_FILE

        if candidate.exists():
            return candidate

        # If BSMITH_PROJECT_DIR is set but no config found, return None
        # (don't fall through to upward walk - user explicitly set the location)
        return None

    # Priority 2: Walk up from CWD to find brainsmith.yaml
    current = Path.cwd().resolve()

    # Walk up until we hit filesystem root
    while current != current.parent:
        candidate = current / _PROJECT_CONFIG_FILE

        if candidate.exists():
            return candidate

        current = current.parent

    # Reached filesystem root without finding config
    return None


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for YAML files with support for project configs."""

    def __init__(self, settings_cls: type[BaseSettings], project_file: Path | None = None):
        super().__init__(settings_cls)
        self.yaml_files = []
        self.project_file_used = None  # Track which project file was actually loaded

        # Load project file
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

    def _load_and_merge_yaml_files(self) -> dict[str, Any]:
        """Load and merge YAML configuration files.

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
                if hasattr(e, "problem_mark"):
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

    def get_field_value(self, field_name: str, field_info: Any) -> tuple[Any, str, bool]:
        """Get field value from YAML source."""
        if field_name in self._data:
            return self._data[field_name], field_name, True

        # Special handling for _project_file_used
        if field_name == "_project_file_used" and self.project_file_used:
            return self.project_file_used, field_name, True

        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return all settings from YAML file."""
        data = self._data.copy()
        # Include project file info for project_dir detection
        if self.project_file_used:
            data["_project_file_used"] = self.project_file_used
        return data


class LoggingConfig(BaseModel):
    """Logging configuration with progressive disclosure.

    Simple defaults for CLI use, advanced customization via config file.
    """

    # Simple (exposed in CLI)
    level: str = Field(
        default="normal", description="Console verbosity level: quiet | normal | verbose | debug"
    )

    # Advanced (config file only)
    finn_tools: dict[str, str] | None = Field(
        default=None,
        description="Per-tool log levels for FINN tools (e.g., {'vivado': 'WARNING', 'hls': 'INFO'})",
    )

    suppress_patterns: list[str] | None = Field(
        default=None,
        description="Regex patterns to suppress from console output (file logs unaffected)",
    )

    max_log_size_mb: int = Field(
        default=0, description="Maximum log file size in MB (0 = no rotation)"
    )

    keep_backups: int = Field(default=3, description="Number of rotated log backups to keep")

    model_config = ConfigDict(extra="forbid")


class SystemConfig(BaseSettings):
    """Configuration schema with hierarchical priority.

    Priority order (highest to lowest):
    1. CLI arguments (passed to constructor)
    2. Environment variables (BSMITH_* prefix)
    3. Project config (brainsmith.yaml)
    4. Built-in defaults
    """

    # NOTE: bsmith_dir is now a cached property, not a configurable field
    build_dir: Path = Field(default=Path("build"), description="Build directory for artifacts")
    deps_dir: Path = Field(default=Path("deps"), description="Dependencies directory")
    # project_dir is computed (not user-configurable) - always set to brainsmith root
    # It's set in model_post_init as a regular attribute
    component_sources: dict[str, Path | None] = Field(
        default_factory=lambda: {
            SOURCE_PROJECT: None,  # Resolved to project_dir (supports kernels/ and steps/ subdirectories)
        },
        description=(
            "Filesystem-based component source paths. Maps source name to directory path. "
            "'project' source defaults to project_dir (supports kernels/ and steps/ subdirectories). "
            "Core namespace 'brainsmith' and entry point sources (e.g., 'finn') are "
            "loaded automatically and cannot be configured here."
        ),
    )
    source_priority: list[str] = Field(
        default=list(DEFAULT_SOURCE_PRIORITY),
        description=(
            "Component source resolution priority. First source with matching "
            "component wins. Custom filesystem sources from component_sources are "
            "auto-appended if not explicitly listed. The 'custom' source (ephemeral "
            "runtime components) is always included at the end by default but can be "
            "repositioned by users."
        ),
    )
    source_module_prefixes: dict[str, str] = Field(
        default_factory=lambda: SOURCE_MODULE_PREFIXES.copy(),
        description=(
            "**DEPRECATED:** This field is no longer needed and will be removed in a future release. "
            "Source detection now uses component_sources keys directly for hierarchical prefix matching. "
            "Setting this field will emit a deprecation warning during configuration loading."
        ),
    )

    # Xilinx configuration (flattened)
    xilinx_path: Path = Field(
        default=Path("/tools/Xilinx"), description="Xilinx root installation path"
    )
    xilinx_version: str = Field(default="2024.2", description="Xilinx tool version")
    vivado_path: Path | None = Field(
        default=None, description="Path to Vivado (auto-detected from xilinx_path)"
    )
    vitis_path: Path | None = Field(
        default=None, description="Path to Vitis (auto-detected from xilinx_path)"
    )
    vitis_hls_path: Path | None = Field(
        default=None, description="Path to Vitis HLS (auto-detected from xilinx_path)"
    )

    vendor_platform_paths: str = Field(
        default="/opt/xilinx/platforms",
        description="Vendor platform repository paths (Xilinx/Intel FPGA platforms)",
    )

    components_strict: bool = Field(default=True, description="Strict component loading")

    cache_components: bool = Field(
        default=True,
        description=(
            "Enable manifest caching for component discovery. When True, generates and "
            "uses a manifest cache in .brainsmith/ to speed up startup. Cache "
            "auto-invalidates when component files are modified. Set to False to always "
            "perform eager discovery."
        ),
    )

    vivado_ip_cache: Path | None = Field(
        default=None,
        description="Vivado IP cache directory (auto-computed from build_dir if not set)",
    )

    netron_port: int = Field(
        default=8080, description="Port for Netron neural network visualization"
    )

    default_workers: int = Field(
        default=4,
        description=(
            "Default number of workers for parallel operations "
            "(exports to NUM_DEFAULT_WORKERS for FINN)"
        ),
    )

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration (verbosity, filters, rotation)",
    )

    finn_root: Path | None = Field(
        default=None, description="FINN root directory (defaults to deps_dir/finn)"
    )
    finn_build_dir: Path | None = Field(
        default=None, description="FINN build directory (defaults to build_dir)"
    )
    finn_deps_dir: Path | None = Field(
        default=None, description="FINN dependencies directory (defaults to deps_dir)"
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
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
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
        project_file = init_dict.get("_project_file")

        return (
            init_settings,  # CLI args (already resolved to CWD in load_config)
            env_settings,  # Env vars (stay relative, resolved in model_post_init)
            YamlSettingsSource(settings_cls, project_file=project_file),
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
        6. Check for deprecated configuration
        """
        self.project_dir = self._detect_project_root()
        self._resolve_core_paths()
        self._resolve_xilinx_tools()
        self._resolve_finn_paths()
        self._resolve_component_sources()
        self._resolve_source_priority()
        self._check_deprecations()

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

        Only project and custom sources are filesystem-based and configurable.
        Core namespace (brainsmith) and entry points (finn) are discovered automatically.
        """
        # Standard filesystem source with default path (project root with optional kernels/steps subdirs)
        if self.component_sources.get(SOURCE_PROJECT) is None:
            self.component_sources[SOURCE_PROJECT] = self.project_dir

        # Custom sources: resolve relative paths
        # Standard source (project) is resolved above
        standard_sources = {SOURCE_PROJECT}
        for name, path in list(self.component_sources.items()):
            if name not in standard_sources and path is not None:
                self.component_sources[name] = self._resolve(path, self.project_dir)

    def _resolve_source_priority(self) -> None:
        """Auto-append custom component sources to source_priority if not already listed.

        This ensures custom sources work automatically while allowing users to
        explicitly position them in the priority list if desired.

        Also ensures 'custom' source (ephemeral runtime components) is present
        at the end by default, but users can reorder it if needed.
        """
        from brainsmith.registry.constants import SOURCE_CUSTOM

        # Auto-append filesystem sources from component_sources
        for source_name in self.component_sources.keys():
            if source_name not in self.source_priority:
                self.source_priority.append(source_name)

        # Ensure 'custom' source is present (append at end if not explicitly configured)
        if SOURCE_CUSTOM not in self.source_priority:
            self.source_priority.append(SOURCE_CUSTOM)

    def _check_deprecations(self) -> None:
        """Check for deprecated configuration options and emit warnings."""
        logger = logging.getLogger(__name__)

        # Check if user has customized source_module_prefixes
        # Suppress any Pydantic deprecation warnings when accessing the field
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            default_prefixes = SOURCE_MODULE_PREFIXES
            if self.source_module_prefixes != default_prefixes:
                warnings.warn(
                    "The 'source_module_prefixes' configuration field is deprecated and will be removed in a future release. "
                    "Source detection now uses component_sources keys directly for hierarchical prefix matching. "
                    "Please remove this field from your configuration.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                logger.warning(
                    "DEPRECATED: source_module_prefixes in configuration. "
                    "Use component_sources keys for domain prefix matching instead."
                )

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
        1. Custom project file location (from load_config project_file param) - HIGHEST
        2. BSMITH_PROJECT_DIR env var (explicit override for runtime)
        3. Walk up from CWD to find brainsmith.yaml, return its parent directory

        Priority 1 ensures that when project init explicitly creates a config file,
        that file's location determines the project root, regardless of any sourced
        environment variables from other projects.

        Returns the same project_dir regardless of which subdirectory you're in,
        providing stable path resolution.

        Returns:
            Absolute path to detected project root

        Raises:
            ValueError: If no project can be detected
        """
        # Check for explicit project file first (e.g., from project init)
        project_file_used = getattr(self, "_project_file_used", None)
        if project_file_used:
            return Path(project_file_used).parent.resolve()

        # Then check environment variable override
        if "BSMITH_PROJECT_DIR" in os.environ:
            return Path(os.environ["BSMITH_PROJECT_DIR"]).resolve()

        # Walk up from CWD to find config
        config_file = _find_project_config()
        if config_file:
            return config_file.parent

        # No project found - fail with helpful error
        raise ValueError(
            "No Brainsmith project detected.\n\n"
            "To fix this, either:\n"
            "  1. Run 'brainsmith project init' to create a new project\n"
            "  2. Navigate to an existing project directory (containing brainsmith.yaml)\n"
            "  3. Source the project environment: source .brainsmith/env.sh\n"
            f"\nCurrent directory: {Path.cwd()}"
        )

    @field_validator("component_sources", mode="before")
    @classmethod
    def validate_component_sources(cls, v: Any) -> dict[str, Path | None]:
        """Validate component sources and warn about reserved source names.

        Reserved source names:
        - Core namespace ('brainsmith'): Internal components loaded via direct import
        - Entry points (e.g., 'finn'): Discovered via pip package entry points

        Users can configure filesystem-based sources (project, user, custom) but
        cannot override reserved names.
        """
        import logging

        logger = logging.getLogger(__name__)

        default_sources = cls.model_fields["component_sources"].default_factory()

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

    def _detect_xilinx_tool(self, tool_name: str) -> Path | None:
        """Auto-detect Xilinx tool path from xilinx_path/tool_name/version."""
        if not self.xilinx_path or not self.xilinx_path.exists():
            return None

        tool_path = self.xilinx_path / tool_name / self.xilinx_version
        return tool_path if tool_path.exists() else None

    # Activation Script Generation

    def generate_activation_script(self, output_path: Path) -> Path:
        """Generate bash activation script from current configuration.

        The script can be sourced multiple times safely:
        - Cleans up old Xilinx/brainsmith paths before adding new ones
        - Sources Xilinx settings64.sh files for complete tool environment

        Args:
            output_path: Where to write the activation script

        Returns:
            Path to generated script

        Example:
            >>> config = get_config()
            >>> config.generate_activation_script(Path("~/.brainsmith/env.sh"))
            >>> # User runs: source ~/.brainsmith/env.sh
        """
        from .env_export import EnvironmentExporter

        env_dict = EnvironmentExporter(self).to_env_dict()

        script_lines = [
            "#!/bin/bash",
            "# Auto-generated by brainsmith",
            "# Source this file to set up environment:",
            "#   source .brainsmith/env.sh",
            "",
            "# This script is idempotent - safe to source multiple times",
            "",
            self._generate_cleanup_code(),
            "",
            "# Export fresh environment variables",
        ]

        for key, value in sorted(env_dict.items()):
            # Skip internal markers
            if key.startswith("_BRAINSMITH") or key.startswith("_OLD_"):
                continue

            # Skip PATH - we'll add Xilinx tool paths separately
            if key == "PATH":
                continue

            # Properly escape quotes in values
            escaped_value = str(value).replace('"', '\\"')
            script_lines.append(f'export {key}="{escaped_value}"')

        # Add Xilinx tool paths to PATH
        script_lines.extend(
            [
                "",
                "# Add Xilinx tools to PATH",
                'if [ -n "$VIVADO_PATH" ]; then',
                '    export PATH="$VIVADO_PATH/bin:$PATH"',
                "fi",
                "",
                'if [ -n "$VITIS_PATH" ]; then',
                '    export PATH="$VITIS_PATH/bin:$PATH"',
                "fi",
                "",
                'if [ -n "$HLS_PATH" ]; then',
                '    export PATH="$HLS_PATH/bin:$PATH"',
                "fi",
            ]
        )

        # Source Xilinx settings64.sh for complete environment
        script_lines.extend(
            [
                "",
                "# Source Xilinx tool settings for full environment",
                "# Vitis includes Vivado, so check it first",
                'if [ -n "$VITIS_PATH" ] && [ -f "$VITIS_PATH/settings64.sh" ]; then',
                '    source "$VITIS_PATH/settings64.sh" 2>/dev/null',
                'elif [ -n "$VIVADO_PATH" ] && [ -f "$VIVADO_PATH/settings64.sh" ]; then',
                '    source "$VIVADO_PATH/settings64.sh" 2>/dev/null',
                "fi",
                "",
                "# Source HLS separately (not included in Vitis)",
                'if [ -n "$HLS_PATH" ] && [ -f "$HLS_PATH/settings64.sh" ]; then',
                '    source "$HLS_PATH/settings64.sh" 2>/dev/null',
                "fi",
            ]
        )

        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(script_lines))
        output_path.chmod(0o755)

        return output_path

    def generate_direnv_file(self, output_path: Path) -> Path:
        """Generate .envrc file for direnv integration.

        Creates a direnv configuration that:
        - Watches brainsmith.yaml for changes
        - Auto-regenerates environment when config changes
        - Sources .brainsmith/env.sh for all environment variables
        - Activates virtualenv automatically

        User must run 'direnv allow' to trust the file.

        Args:
            output_path: Where to write the .envrc file (typically project root)

        Returns:
            Path to generated .envrc file

        Example:
            >>> config = get_config()
            >>> config.generate_direnv_file(Path(".envrc"))
            >>> # User runs: direnv allow
        """
        output_path.parent / ".brainsmith"

        envrc_content = """#!/usr/bin/env bash
# Auto-generated by brainsmith
# Enable with: direnv allow

# Watch config file - direnv will reload when it changes
watch_file brainsmith.yaml

# Activate virtualenv first (required for brainsmith command)
if [ -d .venv ]; then
    export VIRTUAL_ENV="$PWD/.venv"
    PATH_add "$VIRTUAL_ENV/bin"
fi

# Auto-regenerate environment if config is newer than env.sh
if [ brainsmith.yaml -nt .brainsmith/env.sh ]; then
    echo "Config changed, regenerating environment..."
    if command -v brainsmith &> /dev/null; then
        brainsmith project init > /dev/null 2>&1 || {
            echo -e "\033[33mFailed to regenerate. Run: brainsmith project init\033[0m"
        }
    else
        echo -e "\033[33mbrainsmith command not found. Check venv activation.\033[0m"
    fi
fi

# Source Brainsmith environment (sets all variables, sources Xilinx settings64.sh)
source_env .brainsmith/env.sh
"""

        output_path = Path(output_path).expanduser()
        output_path.write_text(envrc_content)
        output_path.chmod(0o644)  # Readable but not executable (direnv sources it)

        return output_path

    def _generate_cleanup_code(self) -> str:
        """Generate bash code to remove old Xilinx/brainsmith paths.

        This ensures idempotency - old paths are removed before new ones
        are added, preventing duplicates when re-sourcing.

        Returns:
            Bash code as string
        """
        return """# Cleanup function - removes paths matching pattern from PATH-like variable
_cleanup_path_var() {
    local var_name=$1
    local pattern=$2
    # Use eval for shell-agnostic indirect variable expansion (works in bash and zsh)
    eval "local current_value=\\${$var_name}"

    # Remove matching paths, preserving order
    local new_value=$(echo "$current_value" | tr ':' '\\n' | grep -v "$pattern" | tr '\\n' ':' | sed 's/:$//')

    eval "export $var_name=\"$new_value\""
}

# Remove old Xilinx/Brainsmith paths to avoid duplicates
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _cleanup_path_var "LD_LIBRARY_PATH" "/tools/Xilinx"
    _cleanup_path_var "LD_LIBRARY_PATH" "brainsmith"
fi

if [ -n "${PATH:-}" ]; then
    _cleanup_path_var "PATH" "/.brainsmith/"
fi

# Clean up the helper function
unset -f _cleanup_path_var"""


# CLI helper for Poe task
def generate_activation_scripts():
    """CLI helper for Poe task to generate activation scripts.

    Called by: poetry run poe generate-activation-scripts
    or automatically via post_install hook
    """
    import os

    from brainsmith.settings import get_config, reset_config

    # Clear any cached config and polluted environment to ensure clean generation
    reset_config()

    # Temporarily unset LD_LIBRARY_PATH to prevent accumulation from previous runs
    old_ld_path = os.environ.pop("LD_LIBRARY_PATH", None)
    old_path = os.environ.pop("PATH", None)

    try:
        config = get_config()

        project_dir = config.project_dir
        brainsmith_dir = project_dir / ".brainsmith"
        brainsmith_dir.mkdir(parents=True, exist_ok=True)

        # Generate activation scripts
        config.generate_activation_script(brainsmith_dir / "env.sh")
        config.generate_deactivation_script(brainsmith_dir / "deactivate.sh")

        # Generate direnv integration file
        config.generate_direnv_file(project_dir / ".envrc")
    finally:
        # Restore original environment
        if old_ld_path:
            os.environ["LD_LIBRARY_PATH"] = old_ld_path
        if old_path:
            os.environ["PATH"] = old_path

    print(f"✅ Generated activation scripts in {brainsmith_dir}")
    print("   - env.sh (manual activation)")
    print("   - deactivate.sh (deactivation)")
    print("   - ../.envrc (direnv integration)")
    print()
    print("To enable direnv (recommended):")
    print("  direnv allow")
    print()
    print("Or use manual activation:")
    print(f"  source {brainsmith_dir / 'env.sh'}")
