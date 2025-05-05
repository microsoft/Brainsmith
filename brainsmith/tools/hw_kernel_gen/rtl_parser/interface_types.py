############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Core data structures for interface analysis."""

from dataclasses import dataclass, field
from enum import Enum
# Use TYPE_CHECKING to avoid runtime import cycle, but keep type hints for checkers
from typing import Dict, Any, Optional, TYPE_CHECKING

# Use forward reference string hint for Port to break cycle
if TYPE_CHECKING:
    from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port

class InterfaceType(Enum):
    """Enumeration of supported interface types."""
    GLOBAL_CONTROL = "global"
    AXI_STREAM = "axis"
    AXI_LITE = "axilite"
    UNKNOWN = "unknown" # For ports not part of a recognized interface

@dataclass
class ValidationResult:
    """Represents the result of a validation check."""
    valid: bool
    message: Optional[str] = None

@dataclass
class PortGroup:
    """Group of related ports potentially forming an interface."""
    interface_type: InterfaceType
    name: Optional[str] = None # e.g., "in0" for AXI-Stream, "config" for AXI-Lite
    # Use string hint 'Port'
    ports: Dict[str, 'Port'] = field(default_factory=dict) # Maps signal suffix (e.g., TDATA) or full name to Port object
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., data width for AXI

    # Use string hint 'Port'
    def add_port(self, port: 'Port', key: Optional[str] = None) -> None:
        """Adds a port to the group, using a specific key or the port name."""
        if key is None:
            key = port.name
        if key in self.ports:
            # Handle potential duplicates or decide on override/error logic
            # For now, let's assume override is okay, but maybe log a warning
            pass
        self.ports[key] = port

@dataclass
class Interface:
    """Represents a validated and identified interface."""
    name: str # e.g., "global", "in0", "config"
    type: InterfaceType
    # Use string hint 'Port'
    ports: Dict[str, 'Port'] # Maps signal suffix/name to Port object
    validation_result: ValidationResult
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., data width, address width
