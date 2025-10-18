# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import shlex
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from functools import reduce

import click

from brainsmith.settings import SystemConfig, load_config
from .constants import USER_CONFIG_DIR, USER_CONFIG_FILE

logger = logging.getLogger(__name__)


def _set_nested(dictionary: dict[str, Any], key_path: str, value: Any) -> None:
    """Example:
        >>> d = {}
        >>> _set_nested(d, "finn.build_dir", "/tmp")
        >>> d
        {'finn': {'build_dir': '/tmp'}}
    """
    keys = key_path.split('.')
    parent = reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], dictionary)
    parent[keys[-1]] = value


@dataclass
class ApplicationContext:
    """Manages CLI execution context and configuration loading.

    Loads configuration via SystemConfig with the following precedence:
    1. CLI arguments (--config, --build-dir) - highest priority
    2. Environment variables (BSMITH_*)
    3. Project config (./brainsmith_settings.yaml)
    4. User config (~/.brainsmith/settings.yaml)
    5. Built-in defaults - lowest priority

    Stores loaded configuration and provides access to Click context.
    """

    # Core settings
    no_progress: bool = False
    config_file: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    # Loaded configuration
    config: SystemConfig | None = None

    # User config path
    user_config_path: Path = field(default_factory=lambda: USER_CONFIG_DIR / USER_CONFIG_FILE)
    
    def load_configuration(self) -> None:
        logger.debug(f"Loading configuration from project_file={self.config_file}")
        
        # Load with user config support
        self.config = load_config(
            project_file=self.config_file,
            user_file=self.user_config_path if self.user_config_path.exists() else None
        )

        if self.overrides:
            config_dict = self.config.model_dump()

            for key, value in self.overrides.items():
                _set_nested(config_dict, key, value)

            # Recreate config instance with overrides applied
            self.config = SystemConfig(**config_dict)
    
    def get_effective_config(self) -> SystemConfig:
        if not self.config:
            self.load_configuration()
        return self.config
    
    
    def export_environment(self, shell: str = "bash") -> str:
        if not self.config:
            return ""

        env_vars = self.config.export_to_environment(verbose=False, export=False)

        lines = []
        if shell in ["bash", "zsh", "sh"]:
            for key, value in env_vars.items():
                escaped_value = shlex.quote(str(value))
                lines.append(f"export {key}={escaped_value}")
        elif shell == "fish":
            for key, value in env_vars.items():
                escaped_value = shlex.quote(str(value))
                lines.append(f"set -x {key} {escaped_value}")
        elif shell == "powershell":
            for key, value in env_vars.items():
                lines.append(f"$env:{key} = '{value}'")
        else:
            raise ValueError(f"Unsupported shell: {shell}")

        return "\n".join(lines)
