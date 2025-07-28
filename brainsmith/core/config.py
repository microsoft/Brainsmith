# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Clean Configuration Module

Arete: Simple, correct configuration with no hidden complexity.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional


@dataclass
class ForgeConfig:
    """Configuration that actually works."""
    # Always required
    clock_ns: float
    
    # Output type (clear names)
    output: Literal["estimates", "rtl", "bitfile"] = "estimates"
    
    # Target (required for rtl/bitfile)
    board: Optional[str] = None
    
    # Everything else has sensible defaults
    verify: bool = False
    verify_data: Optional[Path] = None
    parallel_builds: int = 4
    debug: bool = False
    save_intermediate_models: bool = False
    
    # Direct FINN parameter overrides
    finn_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.output != "estimates" and not self.board:
            raise ValueError(f"{self.output} requires board specification")