# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations  # PEP 563: Postponed evaluation of annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import click

# Type hints only - settings imported lazily inside methods
if TYPE_CHECKING:
    from brainsmith.settings import SystemConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class ApplicationContext:
    """CLI execution context with SystemConfig loading and CLI argument handling."""

    # Core settings
    no_progress: bool = False
    config_file: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    # Loaded configuration
    config: "SystemConfig | None" = None

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
        context.get_effective_config().export_to_environment()

        return context

    def load_configuration(self) -> None:
        from brainsmith.settings import load_config

        # Load config with CLI overrides
        # Pydantic handles validation and priority (CLI overrides > env > file > defaults)
        self.config = load_config(
            project_file=self.config_file,
            **self.overrides  # Pass overrides directly to Pydantic
        )

    def get_effective_config(self) -> "SystemConfig":
        if not self.config:
            self.load_configuration()
        return self.config
