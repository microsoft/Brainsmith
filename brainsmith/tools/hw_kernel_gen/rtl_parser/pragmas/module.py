############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Module-level pragma implementations.

This module contains pragmas that affect module selection and configuration.
"""

from dataclasses import dataclass
from typing import Dict
import logging

from .base import Pragma, PragmaError
from ..rtl_data import PragmaType

logger = logging.getLogger(__name__)


@dataclass
class TopModulePragma(Pragma):
    """TOP_MODULE pragma for specifying the target module.
    
    Format: @brainsmith top_module <module_name>
    
    Used when multiple modules exist in a file to specify which one
    should be processed by the Hardware Kernel Generator.
    """
    
    def __post_init__(self):
        # Ensure base class __post_init__ is called
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """Handles TOP_MODULE pragma: @brainsmith top_module <module_name>"""
        logger.debug(f"Parsing TOP_MODULE pragma: {self.inputs} at line {self.line_number}")
        
        pos = self.inputs['positional']
        
        if len(pos) != 1:
            raise PragmaError("TOP_MODULE pragma requires exactly one argument: <module_name>")
        return {"module_name": pos[0]}

    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply TOP_MODULE pragma to kernel metadata."""
        # TOP_MODULE is handled during parsing to select the correct module
        # By the time we have KernelMetadata, the module has already been selected
        # This is a no-op but included for completeness
        logger.debug(f"TOP_MODULE pragma already processed during module selection")