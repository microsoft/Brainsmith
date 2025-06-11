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
    "parse_rtl_file",
]


def parse_rtl_file(rtl_file, advanced_pragmas: bool = False):
    """
    Parse RTL file and return HWKernel with enhanced features.
    
    This function provides the interface expected by the CLI,
    returning a HWKernel with all enhanced features enabled.
    
    Args:
        rtl_file: Path to SystemVerilog RTL file or Path object
        advanced_pragmas: Enable enhanced BDIM pragma processing
        
    Returns:
        HWKernel: Kernel object with enhanced functionality
        
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
        
        # Parse the RTL file - this now returns the HWKernel directly
        hw_kernel = parser.parse_file(str(rtl_file))
        
        # Enhance the kernel with additional features
        _enhance_hw_kernel_for_enhanced_mode(hw_kernel, rtl_file, advanced_pragmas)
        
        logger.info(f"Successfully parsed RTL file {rtl_file} → HWKernel '{hw_kernel.name}'")
        return hw_kernel
        
    except Exception as e:
        logger.error(f"Failed to parse RTL file {rtl_file}: {e}")
        # Re-raise as RTLParsingError for consistent error handling
        from ..errors import RTLParsingError
        raise RTLParsingError(f"RTL parsing failed for {rtl_file}: {e}") from e


def _enhance_hw_kernel_for_enhanced_mode(hw_kernel, source_file, advanced_pragmas: bool):
    """
    Enhance HWKernel with additional features and metadata.
    
    Args:
        hw_kernel: HWKernel object to enhance (modified in place)
        source_file: Path to source RTL file
        advanced_pragmas: Whether advanced pragma processing was enabled
    """
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Set enhanced fields
    hw_kernel.source_file = Path(source_file)
    hw_kernel.pragma_sophistication_level = "advanced" if advanced_pragmas else "simple"
    hw_kernel.compiler_data = {}  # Will be populated by CLI
    
    # Extract BDIM metadata if advanced pragmas enabled
    if advanced_pragmas:
        bdim_metadata, warnings = _extract_bdim_metadata(hw_kernel.pragmas, hw_kernel.interfaces)
        hw_kernel.bdim_metadata = bdim_metadata
        hw_kernel.parsing_warnings.extend(warnings)
    
    logger.debug(f"Enhanced HWKernel '{hw_kernel.name}' with additional features (level: {hw_kernel.pragma_sophistication_level})")




def _extract_bdim_metadata(pragmas: List[Pragma], interfaces: Dict[str, Interface]) -> tuple:
    """
    Extract BDIM metadata from pragmas for advanced processing.
    
    Args:
        pragmas: List of parsed pragmas
        interfaces: Dictionary of parsed interfaces
        
    Returns:
        tuple: (bdim_metadata_dict, warnings_list)
    """
    bdim_metadata = {
        'tensor_dims': {},
        'block_dims': {},
        'stream_dims': {},
        'chunking_strategies': {}
    }
    warnings = []
    
    # Look for BDIM and TDIM pragmas
    for pragma in pragmas:
        pragma_type = getattr(pragma, 'type', None)
        if pragma_type and hasattr(pragma_type, 'value'):
            pragma_type_str = pragma_type.value
        else:
            pragma_type_str = str(pragma_type) if pragma_type else 'unknown'
            
        if pragma_type_str in ['bdim', 'tdim']:
            try:
                # Extract pragma data
                parsed_data = getattr(pragma, 'parsed_data', {})
                interface_name = parsed_data.get('interface_name', '')
                
                if interface_name:
                    if pragma_type_str == 'bdim':
                        # Enhanced BDIM pragma processing
                        if parsed_data.get('format') == 'enhanced':
                            bdim_metadata['chunking_strategies'][interface_name] = {
                                'type': 'index',
                                'chunk_index': parsed_data.get('chunk_index'),
                                'chunk_sizes': parsed_data.get('chunk_sizes', [])
                            }
                        else:
                            # Legacy format
                            bdim_metadata['block_dims'][interface_name] = {
                                'expressions': parsed_data.get('dimension_expressions', [])
                            }
                    else:
                        # TDIM (deprecated)
                        warnings.append(f"TDIM pragma found for {interface_name} - consider migrating to BDIM")
                        bdim_metadata['tensor_dims'][interface_name] = {
                            'expressions': parsed_data.get('dimension_expressions', [])
                        }
                        
            except Exception as e:
                warnings.append(f"Failed to process {pragma_type_str} pragma: {e}")
    
    return bdim_metadata if any(bdim_metadata.values()) else None, warnings