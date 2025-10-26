# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations  # PEP 563: Postponed evaluation of annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import click

# Lazy imports for settings module (PEP 562)
# This defers the expensive Pydantic import (~685ms) until actually needed.
# For `--help` commands, settings are never imported, making CLI 76% faster.
if TYPE_CHECKING:
    # Type hints only - not evaluated at runtime
    from brainsmith.settings import SystemConfig, load_config

logger = logging.getLogger(__name__)

# Use shared lazy loader for consistency
from brainsmith._internal.lazy_imports import LazyModuleLoader

_lazy_loader = LazyModuleLoader({
    "SystemConfig": "brainsmith.settings",
    "load_config": "brainsmith.settings",
})


def __getattr__(name):
    """Lazy import settings module on first access (defers expensive Pydantic ~685ms import)."""
    return _lazy_loader.get_attribute(name)


def __dir__():
    """Support dir() and IDE autocomplete."""
    return list(globals().keys()) + _lazy_loader.dir()


@dataclass
class ApplicationContext:
    """CLI execution context with SystemConfig loading and CLI argument handling."""

    # Core settings
    no_progress: bool = False
    config_file: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    # Loaded configuration
    config: SystemConfig | None = None

    # User config path (~/.brainsmith/config.yaml)
    user_config_path: Path = field(default_factory=lambda: Path.home() / ".brainsmith" / "config.yaml")

    @classmethod
    def from_cli_args(
        cls,
        config_file: Path | None,
        build_dir_override: Path | None,
        log_level: str,
        no_progress: bool,
        cli_name: str
    ) -> "ApplicationContext":
        """Create context from CLI arguments and perform all initialization.

        Args:
            config_file: Path to config file override
            build_dir_override: Path to build directory override
            log_level: Logging level
            no_progress: Disable progress indicators
            cli_name: Name of CLI (for logging)

        Returns:
            Initialized ApplicationContext with loaded configuration
        """
        import logging
        from brainsmith._internal.logging import setup_logging

        logger = logging.getLogger(__name__)

        context = cls(config_file=config_file, no_progress=no_progress)

        if build_dir_override:
            context.overrides["build_dir"] = str(build_dir_override)

        setup_logging(level=log_level)
        logger.debug(f"{cli_name} CLI initialized with logs={log_level}, no_progress={no_progress}")

        context.load_configuration()
        context.get_effective_config().export_to_environment(verbose=False)

        return context

    def load_configuration(self) -> None:
        # Lazy import settings at runtime
        from brainsmith.settings import load_config

        # Load with user config support and CLI overrides
        # Pydantic handles validation and priority (CLI overrides > env > file > defaults)
        self.config = load_config(
            project_file=self.config_file,
            user_file=self.user_config_path if self.user_config_path.exists() else None,
            **self.overrides  # Pass overrides directly to Pydantic
        )

    def get_effective_config(self) -> SystemConfig:
        if not self.config:
            self.load_configuration()
        return self.config
