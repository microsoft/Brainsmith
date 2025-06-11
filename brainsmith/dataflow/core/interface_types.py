"""
Unified interface types for dataflow modeling.

This module defines the canonical interface type system used throughout
the entire codebase. RTL parser identifies interfaces using these types,
and dataflow components consume them directly.
"""

from enum import Enum


class InterfaceType(Enum):
    """Unified interface types with inherent protocol-role relationships"""
    
    # AXI-Stream interfaces (dataflow)
    INPUT = "input"      # AXI-Stream input for activation data
    OUTPUT = "output"    # AXI-Stream output for result data  
    WEIGHT = "weight"    # AXI-Stream input for weight/parameter data
    
    # AXI-Lite interfaces (configuration)
    CONFIG = "config"    # AXI-Lite for runtime configuration
    
    # Global control signals
    CONTROL = "control"  # Global control signals (clk, rst, etc.)
    
    # Unknown/fallback
    UNKNOWN = "unknown"  # Unrecognized interfaces
    
    @property
    def protocol(self) -> str:
        """Get the hardware protocol for this interface type"""
        protocol_map = {
            InterfaceType.INPUT: "axi_stream",
            InterfaceType.OUTPUT: "axi_stream", 
            InterfaceType.WEIGHT: "axi_stream",
            InterfaceType.CONFIG: "axi_lite",
            InterfaceType.CONTROL: "global_control",
            InterfaceType.UNKNOWN: "unknown"
        }
        return protocol_map[self]
    
    @property
    def is_dataflow(self) -> bool:
        """Check if this interface participates in dataflow"""
        return self in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]
    
    @property
    def is_axi_stream(self) -> bool:
        """Check if this interface uses AXI-Stream protocol"""
        return self.protocol == "axi_stream"
    
    @property
    def is_axi_lite(self) -> bool:
        """Check if this interface uses AXI-Lite protocol"""
        return self.protocol == "axi_lite"
    
    @property
    def is_configuration(self) -> bool:
        """Check if this interface is for configuration"""
        return self in [InterfaceType.CONFIG, InterfaceType.CONTROL]
    
    @property
    def direction(self) -> str:
        """Get the expected direction for this interface type"""
        direction_map = {
            InterfaceType.INPUT: "input",
            InterfaceType.WEIGHT: "input", 
            InterfaceType.OUTPUT: "output",
            InterfaceType.CONFIG: "bidirectional",
            InterfaceType.CONTROL: "input",
            InterfaceType.UNKNOWN: "unknown"
        }
        return direction_map[self]
    
    def __str__(self) -> str:
        """String representation"""
        return f"InterfaceType.{self.name}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"InterfaceType.{self.name}('{self.value}', protocol='{self.protocol}')"