############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Parameter-related pragma implementations.

This module contains pragmas that modify parameter handling including
aliases and derived parameters.
"""

from dataclasses import dataclass
from typing import Dict, List
import logging

from .base import Pragma, PragmaError
from brainsmith.tools.kernel_integrator.types.rtl import PragmaType, Parameter

logger = logging.getLogger(__name__)


@dataclass
class AliasPragma(Pragma):
    """ALIAS pragma for exposing RTL parameters with different names.
    
    Format: @brainsmith alias <rtl_param> <nodeattr_name>
    
    This pragma allows an RTL parameter to be exposed as a node attribute with
    a different name, improving the API for users.
    
    Examples:
    - @brainsmith alias PE parallelism_factor
    - @brainsmith alias C num_channels
    
    Validation:
    - The nodeattr_name cannot match any existing module parameter name
    """
    
    def __post_init__(self):
        super().__post_init__()
    
    def _parse_inputs(self) -> Dict:
        """Parse ALIAS pragma: @brainsmith ALIAS <rtl_param> <nodeattr_name>"""
        logger.debug(f"Parsing ALIAS pragma: {self.inputs} at line {self.line_number}")
        
        pos = self.inputs['positional']
        
        if len(pos) != 2:
            raise PragmaError(f"ALIAS pragma at line {self.line_number} requires exactly 2 arguments: <rtl_param> <nodeattr_name>. Got: {len(pos)} arguments")
        
        rtl_param = pos[0]
        nodeattr_name = pos[1]
        
        # Validate both names are valid identifiers
        if not rtl_param.isidentifier():
            raise PragmaError(f"ALIAS pragma RTL parameter name '{rtl_param}' is not a valid identifier")
        if not nodeattr_name.isidentifier():
            raise PragmaError(f"ALIAS pragma nodeattr name '{nodeattr_name}' is not a valid identifier")
        
        return {"rtl_param": rtl_param, "nodeattr_name": nodeattr_name}
        
    def validate_against_parameters(self, all_parameters: List[Parameter]) -> None:
        """
        Validate that the nodeattr_name doesn't conflict with existing parameters.
        
        Args:
            all_parameters: List of all module parameters
            
        Raises:
            PragmaError: If nodeattr_name conflicts with an existing parameter
        """
        nodeattr_name = self.parsed_data.get("nodeattr_name")
        rtl_param = self.parsed_data.get("rtl_param")
        
        # Check if nodeattr_name matches any existing parameter name
        for param in all_parameters:
            if param.name == nodeattr_name:
                raise PragmaError(
                    f"ALIAS pragma nodeattr name '{nodeattr_name}' conflicts with existing parameter. "
                    f"Choose a different alias name."
                )
        
        # Check if rtl_param exists
        param_exists = any(p.name == rtl_param for p in all_parameters)
        if not param_exists:
            logger.warning(
                f"ALIAS pragma references non-existent parameter '{rtl_param}'. "
                f"This alias will have no effect unless the parameter is defined."
            )
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply ALIAS pragma to kernel metadata."""
        rtl_param = self.parsed_data.get("rtl_param")
        nodeattr_name = self.parsed_data.get("nodeattr_name")
        
        # Validate against other parameters
        self.validate_against_parameters(kernel.parameters)
        
        # Remove from exposed parameters (it will be exposed with the alias name)
        if rtl_param in kernel.exposed_parameters:
            kernel.exposed_parameters.remove(rtl_param)
        
        # Add the alias name to exposed parameters
        if nodeattr_name not in kernel.exposed_parameters:
            kernel.exposed_parameters.append(nodeattr_name)
        
        # Add to linked parameters
        if not hasattr(kernel, 'linked_parameters') or kernel.linked_parameters is None:
            kernel.linked_parameters = {"aliases": {}, "derived": {}, "axilite": {}}
        if "aliases" not in kernel.linked_parameters:
            kernel.linked_parameters["aliases"] = {}
        
        kernel.linked_parameters["aliases"][rtl_param] = nodeattr_name
        
        logger.debug(f"Applied ALIAS pragma: {rtl_param} -> {nodeattr_name}")


@dataclass
class DerivedParameterPragma(Pragma):
    """DERIVED_PARAMETER pragma for computing parameters from Python expressions.
    
    Format: @brainsmith derived_parameter <rtl_param> <python_expression>
    
    This pragma prevents a parameter from being exposed as a node attribute and
    instead assigns it to a Python expression or function call in the RTLBackend.
    
    Examples:
    - @brainsmith derived_parameter SIMD self.get_input_datatype().bitwidth()
    - @brainsmith derived_parameter MEM_SIZE self.calc_wmem()
    """
    
    def __post_init__(self):
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """Parse DERIVED_PARAMETER pragma: @brainsmith DERIVED_PARAMETER <rtl_param> <python_expression>"""
        logger.debug(f"Parsing DERIVED_PARAMETER pragma: {self.inputs} at line {self.line_number}")
        
        pos = self.inputs['positional']
        
        if len(pos) < 2:
            raise PragmaError(f"DERIVED_PARAMETER pragma at line {self.line_number} requires parameter name and Python expression. Got: {len(pos)} arguments")
        
        param_name = pos[0]
        # Join remaining inputs as the Python expression (allows spaces)
        python_expression = " ".join(pos[1:])
        
        # Validate parameter name
        if not param_name.isidentifier():
            raise PragmaError(f"DERIVED_PARAMETER pragma parameter name '{param_name}' is not a valid identifier")
        
        return {"param_name": param_name, "python_expression": python_expression}
    
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply DERIVED_PARAMETER pragma to kernel metadata."""
        param_name = self.parsed_data.get("param_name")
        python_expression = self.parsed_data.get("python_expression")
        
        # Remove from exposed parameters
        if param_name in kernel.exposed_parameters:
            kernel.exposed_parameters.remove(param_name)
        
        # Add to linked parameters
        if not hasattr(kernel, 'linked_parameters') or kernel.linked_parameters is None:
            kernel.linked_parameters = {"aliases": {}, "derived": {}, "axilite": {}}
        if "derived" not in kernel.linked_parameters:
            kernel.linked_parameters["derived"] = {}
        
        kernel.linked_parameters["derived"][param_name] = python_expression
        
        logger.debug(f"Applied DERIVED_PARAMETER pragma: {param_name} -> {python_expression}")


@dataclass
class AxiLiteParamPragma(Pragma):
    """AXILITE_PARAM pragma for linking parameters to AXI-Lite configuration interfaces.
    
    Format: @brainsmith axilite_param <interface_name> <param_name>
    
    This pragma links a parameter to a specific AXI-Lite interface, marking it as
    a configuration control parameter and specifying which interface it controls.
    
    Examples:
    - @brainsmith axilite_param s_axilite_config USE_AXILITE
    - @brainsmith axilite_param s_axilite_ctrl CONFIG_EN
    - @brainsmith axilite_param potato MY_CUSTOM_CONFIG
    """
    
    def __post_init__(self):
        super().__post_init__()
    
    def _parse_inputs(self) -> Dict:
        """Parse AXILITE_PARAM pragma: @brainsmith axilite_param <interface_name> <param_name>"""
        logger.debug(f"Parsing AXILITE_PARAM pragma: {self.inputs} at line {self.line_number}")
        
        pos = self.inputs['positional']
        
        if len(pos) != 2:
            raise PragmaError(f"AXILITE_PARAM pragma at line {self.line_number} requires exactly 2 arguments: <interface_name> <param_name>. Got: {len(pos)} arguments")
        
        interface_name = pos[0]
        param_name = pos[1]
        
        # Validate parameter name
        if not param_name.isidentifier():
            raise PragmaError(f"AXILITE_PARAM pragma parameter name '{param_name}' is not a valid identifier")
        
        # Validate interface name
        if not interface_name.replace('_', '').isalnum():
            raise PragmaError(f"AXILITE_PARAM pragma interface name '{interface_name}' contains invalid characters")
        
        return {"param_name": param_name, "interface_name": interface_name}
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply AXILITE_PARAM pragma to kernel metadata."""
        param_name = self.parsed_data.get("param_name")
        interface_name = self.parsed_data.get("interface_name")
        
        # Initialize linked_parameters if not present
        if not hasattr(kernel, 'linked_parameters') or kernel.linked_parameters is None:
            kernel.linked_parameters = {"aliases": {}, "derived": {}, "axilite": {}}
        if "axilite" not in kernel.linked_parameters:
            kernel.linked_parameters["axilite"] = {}
        
        # Add parameter -> interface mapping
        kernel.linked_parameters["axilite"][param_name] = interface_name
        logger.debug(f"Linked parameter '{param_name}' to AXI-Lite interface '{interface_name}'")