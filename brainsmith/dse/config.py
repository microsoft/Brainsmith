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
    """Configuration extracted from blueprint YAML files.

    Renamed from BlueprintConfig for clarity - this config controls
    Design Space Exploration behavior.
    """
    # Always required
    clock_ns: float  # Required field, mapped to synth_clk_period_ns in FINN config

    # Output type (clear names)
    output: OutputType = OutputType.ESTIMATES

    # Target (required for rtl/bitfile)
    board: Optional[str] = None

    # Everything else has sensible defaults
    verify: bool = False
    verify_data: Optional[Path] = None
    parallel_builds: int = 4
    debug: bool = False
    save_intermediate_models: bool = False

    # Step range control for testing/debugging
    start_step: Optional[str] = None
    stop_step: Optional[str] = None

    # Direct FINN parameter overrides
    finn_overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output != OutputType.ESTIMATES and not self.board:
            raise ValueError(f"{self.output.value} requires board specification")


def _parse_output_type(output_str: str) -> OutputType:
    """Parse output type string to enum.

    Args:
        output_str: String representation of output type

    Returns:
        OutputType enum value

    Raises:
        ValueError: If output string is invalid
    """
    try:
        return OutputType(output_str)
    except ValueError:
        valid_values = ', '.join(t.value for t in OutputType)
        raise ValueError(
            f"Invalid output type '{output_str}'. "
            f"Must be one of: {valid_values}"
        )


def extract_config(data: Dict[str, Any]) -> DSEConfig:
    """Extract DSEConfig from blueprint data.

    Args:
        data: Merged blueprint YAML data

    Returns:
        DSEConfig instance

    Raises:
        ValueError: If required fields are missing
    """
    # Extract config - check both flat and global_config
    config_data = {**data.get('global_config', {}), **data}

    # Validate required field
    if 'clock_ns' not in config_data:
        raise ValueError("Missing required field 'clock_ns' in blueprint")

    return DSEConfig(
        clock_ns=float(config_data['clock_ns']),
        output=_parse_output_type(config_data.get('output', 'estimates')),
        board=config_data.get('board'),
        verify=config_data.get('verify', False),
        verify_data=Path(config_data['verify_data']) if 'verify_data' in config_data else None,
        parallel_builds=config_data.get('parallel_builds', 4),
        debug=config_data.get('debug', False),
        save_intermediate_models=config_data.get('save_intermediate_models', False),
        start_step=config_data.get('start_step'),
        stop_step=config_data.get('stop_step'),
        finn_overrides=data.get('finn_config', {})
    )


