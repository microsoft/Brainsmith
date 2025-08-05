############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""RTL Parser for Kernel Integrator.

This package provides functionality to parse SystemVerilog RTL files and extract
information needed by the Kernel Integrator to create FINN-compatible
hardware kernels.

Key Components:
    - Parser: Main entry point for RTL parsing
    - Interface Analysis: Extracts module parameters and ports
    - Pragma Processing: Handles @brainsmith pragma directives
    - Data Structures: Core data models for parsed information

Example Usage:
    from brainsmith.tools.kernel_integrator.rtl_parser import RTLParser
"""

from typing import Dict, List, Any

# Import shared types
from brainsmith.core.dataflow.types import InterfaceType
# Import RTL-specific types
from .rtl_data import (
    Parameter,
    Port,
    PortGroup,
    ProtocolValidationResult,
    PragmaType,
)
from .pragmas import Pragma
from .parser import RTLParser, ParserError
from .protocol_validator import ProtocolValidator

__all__ = [
    "RTLParser",
    "ParserError",
    "ProtocolValidator",
    "Parameter",
    "Port",
    "PortGroup",
    "InterfaceType",
    "Pragma",
    "PragmaType",
    "ProtocolValidationResult",
    "parse_rtl_file",
]


def parse_rtl_file(rtl_file, advanced_pragmas: bool = False):
    """
    Parse RTL file and return kernel metadata.
    
    This function provides a clean interface for RTL parsing that returns
    KernelMetadata for direct template generation and dataflow integration.
    
    Args:
        rtl_file: Path to SystemVerilog RTL file or Path object
        advanced_pragmas: Enable enhanced BDIM pragma processing (deprecated parameter)
        
    Returns:
        KernelMetadata: Parsed kernel metadata with InterfaceMetadata objects
        
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
        
        # Parse the RTL file and return KernelMetadata directly
        parsed_data = parser.parse_file(str(rtl_file))
        
        logger.info(f"Successfully parsed RTL file {rtl_file} → KernelMetadata '{parsed_data.name}'")
        return parsed_data
        
    except Exception as e:
        logger.error(f"Failed to parse RTL file {rtl_file}: {e}")
        # Re-raise as RTLParsingError for consistent error handling
        from ..errors import RTLParsingError
        raise RTLParsingError(f"RTL parsing failed for {rtl_file}: {e}") from e


# Legacy functions removed - ParsedKernelData handles all functionality directly