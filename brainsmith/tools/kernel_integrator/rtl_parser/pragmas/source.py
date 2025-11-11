############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################
"""Source-related pragma implementations.

This module contains pragmas that manage RTL source files and module selection:
- TopModulePragma: Selects the target module when multiple modules exist
- IncludeRTLPragma: Specifies additional RTL source files to include
"""

import logging
from dataclasses import dataclass
from typing import Any

from brainsmith.tools.kernel_integrator.metadata import KernelMetadata

from .base import Pragma, PragmaError

logger = logging.getLogger(__name__)


@dataclass
class TopModulePragma(Pragma):
    """TOP_MODULE pragma for specifying the target module.
    
    Format: @brainsmith top_module <module_name>
    
    Used when multiple modules exist in a file to specify which one
    should be processed by the Kernel Integrator.
    """
    
    def __post_init__(self):
        # Ensure base class __post_init__ is called
        super().__post_init__()

    def _parse_inputs(self) -> dict:
        """Handles TOP_MODULE pragma: @brainsmith top_module <module_name>"""
        logger.debug(f"Parsing TOP_MODULE pragma: {self.inputs} at line {self.line_number}")
        
        pos = self.inputs['positional']
        
        if len(pos) != 1:
            raise PragmaError("TOP_MODULE pragma requires exactly one argument: <module_name>")
        return {"module_name": pos[0]}

    def apply_to_kernel(self, kernel: KernelMetadata) -> None:
        """Apply TOP_MODULE pragma to kernel metadata."""
        # TOP_MODULE is handled during parsing to select the correct module
        # By the time we have KernelMetadata, the module has already been selected
        # This is a no-op but included for completeness
        logger.debug("TOP_MODULE pragma already processed during module selection")


class IncludeRTLPragma(Pragma):
    """Pragma for including additional RTL source files.
    
    Syntax:
        // @brainsmith INCLUDE_RTL <rtl_file_path>
        
    Examples:
        // @brainsmith INCLUDE_RTL helper_modules.sv
        // @brainsmith INCLUDE_RTL ../common/utilities.sv
        // @brainsmith INCLUDE_RTL /absolute/path/to/module.sv
    
    The specified file paths can be:
    - Absolute paths
    - Relative to the main RTL file's directory
    - Relative to the current working directory
    
    Path resolution follows this precedence order.
    """
    
    def _parse_inputs(self) -> dict[str, Any]:
        """Parse the RTL file path from pragma arguments.
        
        Returns:
            Dict containing the parsed RTL file path.
            
        Raises:
            PragmaError: If no file path is provided.
        """
        # Get raw arguments
        raw_args = self.inputs.get('raw', [])
        if not raw_args:
            raise PragmaError(
                f"INCLUDE_RTL pragma requires a file path argument at line {self.line_number}"
            )
        
        # Join all arguments to handle paths with spaces
        rtl_file_path = ' '.join(str(arg) for arg in raw_args)
        
        if not rtl_file_path.strip():
            raise PragmaError(
                f"INCLUDE_RTL pragma has empty file path at line {self.line_number}"
            )
        
        logger.debug(f"Parsed INCLUDE_RTL pragma with file: {rtl_file_path}")
        
        return {
            'rtl_file': rtl_file_path.strip()
        }
    
    def apply_to_kernel(self, kernel: KernelMetadata) -> None:
        """Apply this pragma to the kernel metadata.
        
        Adds the specified RTL file to the kernel's included_rtl_files list.
        
        Args:
            kernel: KernelMetadata object to modify.
        """
        rtl_file = self.parsed_data.get('rtl_file')
        if not rtl_file:
            logger.warning(f"INCLUDE_RTL pragma has no file path at line {self.line_number}")
            return
        
        # Initialize included_rtl_files if it doesn't exist
        if not hasattr(kernel, 'included_rtl_files'):
            kernel.included_rtl_files = []
        
        # Add the file if not already present
        if rtl_file not in kernel.included_rtl_files:
            kernel.included_rtl_files.append(rtl_file)
            logger.info(f"Added RTL file '{rtl_file}' to included files list")
        else:
            logger.debug(f"RTL file '{rtl_file}' already in included files list")