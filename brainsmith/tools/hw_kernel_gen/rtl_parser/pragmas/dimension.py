############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Dimension-related pragma implementations.

This module contains pragmas for block and stream dimension configuration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

from .base import InterfacePragma, PragmaError
from ..data import PragmaType, Parameter
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType

logger = logging.getLogger(__name__)


@dataclass
class BDimPragma(InterfacePragma):
    """BDIM pragma for block dimension chunking strategies.
    
    Format: @brainsmith bdim <interface_name> <param_name> [SHAPE=<shape>] [RINDEX=<n>]
    
    IMPORTANT: 
    - Only allowed on INPUT or WEIGHT interfaces (not OUTPUT or CONTROL)
    - Only parameter names and ":" are allowed in shapes - NO magic numbers!
    - This ensures parameterizability which is a key design goal.
    
    Examples:
    - @brainsmith bdim in0 IN0_BDIM SHAPE=[DIM1]              # DIM1 parameter, RINDEX=0 (default)
    - @brainsmith bdim in0 IN0_BDIM SHAPE=[DIM2,DIM1]         # DIM2,DIM1 parameters, RINDEX=0
    - @brainsmith bdim in0 IN0_BDIM SHAPE=[:,:] RINDEX=1    # Full dimensions, starting from RINDEX=1
    - @brainsmith bdim weights WGT_BDIM SHAPE=[TILE_SIZE]   # TILE_SIZE parameter
    
    Invalid examples:
    - @brainsmith bdim output0 OUT_BDIM      # ERROR: Cannot use on OUTPUT interface
    - @brainsmith bdim in0 IN0_BDIM SHAPE=[16]   # ERROR: Magic number 16
    - @brainsmith bdim in0 IN0_BDIM SHAPE=[8,4]  # ERROR: Magic numbers 8,4
    """
    
    def __post_init__(self):
        super().__post_init__()
    
    # Class variable to store module parameters for validation (not a dataclass field)
    _module_parameters = {}
    
    @classmethod
    def set_module_parameters(cls, parameters: List['Parameter']) -> None:
        """Set module parameters for BDIM pragma validation."""
        cls._module_parameters = {p.name: p for p in parameters}
    
    def _get_module_parameters(self) -> Dict[str, 'Parameter']:
        """Get module parameters for validation."""
        return self._module_parameters
    
    def _validate_parameters(self) -> None:
        """Validate that all parameter names in block_shape exist in module parameters."""
        logger.debug(f"BDIM pragma parameter validation starting")
        if not self.parsed_data:
            logger.debug(f"BDIM pragma has no parsed_data, skipping validation")
            return
        
        block_shape = self.parsed_data.get("block_shape", [])
        module_parameters = self._get_module_parameters()
        param_names = set(module_parameters.keys())
        
        logger.debug(f"BDIM pragma validating block_shape: {block_shape}")
        logger.debug(f"Available module parameters: {sorted(param_names)}")
        
        for element in block_shape:
            if isinstance(element, str) and element != ':' and element.isidentifier():
                logger.debug(f"BDIM pragma validating parameter: '{element}'")
                # This is a parameter name - validate it exists
                if element not in param_names:
                    error_msg = (f"BDIM pragma at line {self.line_number} references unknown parameter '{element}'. "
                               f"Available parameters: {sorted(param_names) if param_names else 'none'}")
                    logger.error(f"BDIM parameter validation failed: {error_msg}")
                    raise PragmaError(error_msg)
                else:
                    logger.debug(f"BDIM pragma parameter '{element}' validated successfully")
        
        logger.debug(f"BDIM pragma parameter validation completed successfully")

    def _parse_inputs(self) -> Dict:
        """
        Parse new BDIM pragma format.
        
        Format: @brainsmith BDIM <interface_name> <param_name> [SHAPE=<shape>] [RINDEX=<n>]
        
        Returns:
            Dict with parsed data including interface name, parameter name, optional shape, and rindex
        """
        logger.debug(f"Parsing BDIM pragma: {self.inputs} at line {self.line_number}")
        
        if len(self.inputs) < 2:
            raise PragmaError("BDIM pragma requires interface name and parameter name")
        
        interface_name = self.inputs[0]
        param_name = self.inputs[1]
        
        # Validate interface name
        if not interface_name.replace('_', '').replace('V', '').isalnum():
            raise PragmaError(f"BDIM pragma interface name '{interface_name}' contains invalid characters")
        
        # Validate parameter name
        if not param_name.isidentifier():
            raise PragmaError(f"BDIM pragma parameter name '{param_name}' is not a valid identifier")
        
        # Parse optional arguments
        block_shape = None
        rindex = 0
        
        for i in range(2, len(self.inputs)):
            arg = self.inputs[i]
            
            if arg.startswith('SHAPE='):
                # Parse shape specification
                shape_spec = arg[6:]  # Remove 'SHAPE='
                if not (shape_spec.startswith('[') and shape_spec.endswith(']')):
                    raise PragmaError("BDIM pragma SHAPE must be in [shape1,shape2,...] format")
                
                shape_str = shape_spec[1:-1].strip()
                if not shape_str:
                    raise PragmaError("BDIM pragma SHAPE cannot be empty")
                
                # Parse shape elements - only allow ':' and parameter names (NO magic numbers)
                block_shape = []
                for element in shape_str.split(','):
                    element = element.strip()
                    if element == ':':
                        block_shape.append(':')
                    elif element.isdigit():
                        raise PragmaError(f"Magic numbers not allowed in BDIM pragma SHAPE. Use parameter names instead of '{element}'.")
                    elif element.isidentifier():
                        # Parameter name - defer validation until apply phase when parameters are available
                        block_shape.append(element)
                    else:
                        raise PragmaError(f"Invalid shape element '{element}'. Must be ':' (full dimension) or parameter name.")
            
            elif arg.startswith('RINDEX='):
                # Parse RINDEX
                try:
                    rindex = int(arg[7:])
                    if rindex < 0:
                        raise PragmaError(f"RINDEX must be non-negative, got {rindex}")
                except ValueError:
                    raise PragmaError(f"Invalid RINDEX value: {arg}")
            
            else:
                raise PragmaError(f"Unknown BDIM pragma parameter '{arg}'. Expected 'SHAPE=[...]' or 'RINDEX=n'")
        
        return {
            "interface_name": interface_name,
            "param_name": param_name,
            "block_shape": block_shape,
            "rindex": rindex
        }

    def apply_to_interface(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply BDIM pragma to modify chunking strategy and set parameter linkage."""
        logger.debug(f"Attempting to apply BDIM pragma to interface '{metadata.name}'")
        
        # Validate that BDIM pragma is only applied to INPUT or WEIGHT interfaces
        if metadata.interface_type not in [InterfaceType.INPUT, InterfaceType.WEIGHT]:
            error_msg = (f"BDIM pragma at line {self.line_number} cannot be applied to "
                        f"interface '{metadata.name}' of type '{metadata.interface_type.value}'. "
                        f"BDIM pragmas are only allowed on INPUT or WEIGHT interfaces.")
            logger.error(f"BDIM interface type validation failed: {error_msg}")
            raise PragmaError(error_msg)
        
        logger.debug(f"BDIM pragma applying to interface '{metadata.name}' - validating parameters")
        # Validate parameters now that module parameters are available
        self._validate_parameters()
        
        # Create chunking strategy from pragma data
        new_strategy = self._create_chunking_strategy()
        
        # Create shape parameters dict if shape was specified
        shape_params = None
        if self.parsed_data.get("block_shape") is not None:
            shape_params = {
                "shape": self.parsed_data.get("block_shape"),
                "rindex": self.parsed_data.get("rindex", 0)
            }
        
        logger.debug(f"BDIM pragma successfully applied to interface '{metadata.name}'")
        
        return metadata.update_attributes(
            chunking_strategy=new_strategy,
            bdim_param=self.parsed_data.get("param_name"),
            shape_params=shape_params
        )

    def _create_chunking_strategy(self):
        """Create chunking strategy from pragma data."""
        if not self.parsed_data:
            from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy
            return DefaultChunkingStrategy()
        
        # Create BlockChunkingStrategy with parsed data
        from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy
        
        block_shape = self.parsed_data.get("block_shape", [":"])
        rindex = self.parsed_data.get("rindex", 0)
        
        return BlockChunkingStrategy(
            block_shape=block_shape,
            rindex=rindex
        )


@dataclass
class SDimPragma(InterfacePragma):
    """SDIM pragma for linking stream dimension parameters.
    
    Format: @brainsmith sdim <interface_name> <param_name>
    
    This pragma links an interface to its stream dimension RTL parameter.
    The stream shape is inferred from the corresponding BDIM configuration.
    
    IMPORTANT: Only allowed on INPUT or WEIGHT interfaces (not OUTPUT or CONTROL)
    
    Examples:
    - @brainsmith sdim s_axis_input0 INPUT0_SDIM
    - @brainsmith sdim weights_V WEIGHTS_STREAM_SIZE
    
    Invalid examples:
    - @brainsmith sdim m_axis_output0 OUTPUT0_SDIM  # ERROR: Cannot use on OUTPUT interface
    """
    
    def __post_init__(self):
        super().__post_init__()
    
    # Class variable to store module parameters for validation (not a dataclass field)
    _module_parameters = {}
    
    @classmethod
    def set_module_parameters(cls, parameters: List['Parameter']) -> None:
        """Set module parameters for SDIM pragma validation."""
        cls._module_parameters = {p.name: p for p in parameters}
    
    def _get_module_parameters(self) -> Dict[str, 'Parameter']:
        """Get module parameters for validation."""
        return self._module_parameters
    
    def _validate_parameter(self) -> None:
        """Validate that the parameter name exists in module parameters."""
        logger.debug(f"SDIM pragma parameter validation starting")
        if not self.parsed_data:
            logger.debug(f"SDIM pragma has no parsed_data, skipping validation")
            return
        
        param_name = self.parsed_data.get("param_name")
        if not param_name:
            logger.debug(f"SDIM pragma has no param_name, skipping validation")
            return
            
        module_parameters = self._get_module_parameters()
        param_names = set(module_parameters.keys())
        
        logger.debug(f"SDIM pragma validating parameter: '{param_name}'")
        logger.debug(f"Available module parameters: {sorted(param_names)}")
        
        if param_name not in param_names:
            error_msg = (f"SDIM pragma at line {self.line_number} references unknown parameter '{param_name}'. "
                       f"Available parameters: {sorted(param_names) if param_names else 'none'}")
            logger.error(f"SDIM parameter validation failed: {error_msg}")
            raise PragmaError(error_msg)
        else:
            logger.debug(f"SDIM pragma parameter '{param_name}' validated successfully")
        
        logger.debug(f"SDIM pragma parameter validation completed successfully")

    def _parse_inputs(self) -> Dict:
        """
        Parse SDIM pragma format.
        
        Format: @brainsmith SDIM <interface_name> <param_name>
        
        Returns:
            Dict with parsed data including interface name and parameter name
        """
        logger.debug(f"Parsing SDIM pragma: {self.inputs} at line {self.line_number}")
        
        if len(self.inputs) != 2:
            raise PragmaError("SDIM pragma requires exactly two arguments: interface name and parameter name")
        
        interface_name = self.inputs[0]
        param_name = self.inputs[1]
        
        # Validate interface name
        if not interface_name.replace('_', '').replace('V', '').isalnum():
            raise PragmaError(f"SDIM pragma interface name '{interface_name}' contains invalid characters")
        
        # Validate parameter name
        if not param_name.isidentifier():
            raise PragmaError(f"SDIM pragma parameter name '{param_name}' is not a valid identifier")
        
        return {
            "interface_name": interface_name,
            "param_name": param_name
        }

    def apply_to_interface(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply SDIM pragma to set stream dimension parameter."""
        # Validate that SDIM pragma is only applied to INPUT or WEIGHT interfaces
        if metadata.interface_type not in [InterfaceType.INPUT, InterfaceType.WEIGHT]:
            error_msg = (f"SDIM pragma at line {self.line_number} cannot be applied to "
                        f"interface '{metadata.name}' of type '{metadata.interface_type.value}'. "
                        f"SDIM pragmas are only allowed on INPUT or WEIGHT interfaces.")
            logger.error(f"SDIM interface type validation failed: {error_msg}")
            raise PragmaError(error_msg)
        
        logger.debug(f"SDIM pragma applying to interface '{metadata.name}' - validating parameter")
        # Validate parameter now that module parameters are available
        self._validate_parameter()
        
        logger.debug(f"SDIM pragma successfully applied to interface '{metadata.name}'")
        
        # Update interface with SDIM parameter
        return metadata.update_attributes(
            sdim_param=self.parsed_data.get("param_name")
        )