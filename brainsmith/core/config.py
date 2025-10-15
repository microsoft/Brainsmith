# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from brainsmith.core.types import OutputType


@dataclass
class BlueprintConfig:
    """Configuration extracted from blueprint YAML files."""
    # Always required
    clock_ns: float

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
    
    def __post_init__(self):
        if self.output != OutputType.ESTIMATES and not self.board:
            raise ValueError(f"{self.output.value} requires board specification")