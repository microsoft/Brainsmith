# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
DSE Configuration.

Contains the DSEConfig class and utilities for extracting configuration
from blueprint YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from brainsmith.dse.types import OutputType


@dataclass
class DSEConfig:
    """Configuration for Design Space Exploration."""
    # Always required
    clock_ns: float  # Required field, mapped to synth_clk_period_ns in FINN config

    # Output type (clear names)
    output: OutputType = OutputType.ESTIMATES

    # Target (required for rtl/bitfile)
    board: Optional[str] = None

    # Step range control (optional overrides)
    start_step: Optional[str] = None
    stop_step: Optional[str] = None

    # Direct FINN parameter overrides
    finn_overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        # Validate clock_ns is positive
        if self.clock_ns <= 0:
            raise ValueError(f"clock_ns must be positive, got {self.clock_ns}")

        # Validate output type dependencies
        if self.output != OutputType.ESTIMATES and not self.board:
            raise ValueError(f"{self.output.value} requires board specification")


def _parse_output_type(output_str: str) -> OutputType:
    """Parse output type string to enum."""
    try:
        return OutputType(output_str)
    except ValueError:
        valid = ', '.join(t.value for t in OutputType)
        raise ValueError(f"Invalid output '{output_str}'. Must be one of: {valid}")


def extract_config(data: Dict[str, Any]) -> DSEConfig:
    """Extract DSEConfig from blueprint data.

    Config fields are expected at the blueprint top level (flat structure).
    Validation happens in DSEConfig.__post_init__.

    Args:
        data: Merged blueprint YAML data

    Returns:
        DSEConfig instance

    Raises:
        KeyError: If required field 'clock_ns' is missing
        ValueError: If validation fails (from DSEConfig.__post_init__)
    """
    return DSEConfig(
        clock_ns=float(data['clock_ns']),
        output=_parse_output_type(data.get('output', 'estimates')),
        board=data.get('board'),
        start_step=data.get('start_step'),
        stop_step=data.get('stop_step'),
        finn_overrides=data.get('finn_config', {})
    )


