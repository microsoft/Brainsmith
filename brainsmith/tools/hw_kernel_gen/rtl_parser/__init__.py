"""RTL Parser for Hardware Kernel Generator.

This package provides functionality to parse SystemVerilog RTL files and extract
information needed by the Hardware Kernel Generator to create FINN-compatible
hardware kernels.

Key Components:
    - Parser: Main entry point for RTL parsing
    - Interface Analysis: Extracts module parameters and ports
    - Pragma Processing: Handles @brainsmith pragma directives
    - Data Structures: Core data models for parsed information

Example Usage:
    from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser
    
    parser = RTLParser()
    kernel = parser.parse_file("my_kernel.sv")
    
    # Access parsed information
    print(f"Module name: {kernel.name}")
    print(f"Parameters: {kernel.parameters}")
    print(f"Ports: {kernel.ports}")
    print(f"Pragmas: {kernel.pragmas}")
"""

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    HWKernel,
    Parameter,
    Port,
    Pragma
)

__version__ = "0.1.0"
__all__ = ["RTLParser", "HWKernel", "Parameter", "Port", "Pragma"]