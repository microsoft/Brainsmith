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

from typing import Dict, List, Any

# Import unified InterfaceType from dataflow module
from brainsmith.dataflow.core.interface_types import InterfaceType

# Expose key classes and functions for easier import
from .data import (
    Direction,
    Parameter,
    Port,
    PortGroup,
    Interface,
    ParsedKernelData,
    TemplateDatatype,
    SimpleKernel,
    Pragma,
    ValidationResult,
)
from .parser import RTLParser, ParserError
from .protocol_validator import ProtocolValidator

__all__ = [
    "RTLParser",
    "ParserError",
    "ProtocolValidator",
    "ParsedKernelData",
    "TemplateDatatype",
    "SimpleKernel",
    "Parameter",
    "Port",
    "PortGroup",
    "Interface",
    "InterfaceType",
    "Direction",
    "Pragma",
    "ValidationResult",
    "parse_rtl_file",
]


def parse_rtl_file(rtl_file, advanced_pragmas: bool = False) -> ParsedKernelData:
    """
    Parse RTL file and return parsed kernel data.
    
    This function provides a clean interface for RTL parsing that returns
    ParsedKernelData for direct template generation and dataflow integration.
    
    Args:
        rtl_file: Path to SystemVerilog RTL file or Path object
        advanced_pragmas: Enable enhanced BDIM pragma processing (deprecated parameter)
        
    Returns:
        ParsedKernelData: Parsed kernel data with interfaces and metadata
        
    Raises:
        RTLParsingError: If RTL parsing fails
    """
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        from ..errors import RTLParsingError
        
        # Ensure rtl_file is a Path object
        if isinstance(rtl_file, str):
            rtl_file = Path(rtl_file)
        
        # Create RTL parser instance
        parser = RTLParser(debug=advanced_pragmas)
        
        # Parse the RTL file and return ParsedKernelData directly
        parsed_data = parser.parse_file(str(rtl_file))
        
        logger.info(f"Successfully parsed RTL file {rtl_file} → ParsedKernelData '{parsed_data.name}'")
        return parsed_data
        
    except Exception as e:
        logger.error(f"Failed to parse RTL file {rtl_file}: {e}")
        # Re-raise as RTLParsingError for consistent error handling
        from ..errors import RTLParsingError
        raise RTLParsingError(f"RTL parsing failed for {rtl_file}: {e}") from e


# Legacy functions removed - ParsedKernelData handles all functionality directly