############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
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
"""

# Expose key classes and functions for easier import
from .data import (
    Direction,
    InterfaceType,
    Parameter,
    Port,
    PortGroup,
    Interface,
    HWKernel,
    Pragma,
    ValidationResult,
)
from .parser import RTLParser, ParserError
from .protocol_validator import ProtocolValidator

__all__ = [
    "RTLParser",
    "ParserError",
    "ProtocolValidator",
    "HWKernel",
    "Parameter",
    "Port",
    "PortGroup",
    "Interface",
    "InterfaceType",
    "Direction",
    "Pragma",
    "ValidationResult",
]