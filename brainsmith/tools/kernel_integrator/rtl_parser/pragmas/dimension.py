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
from ...data import InterfaceType
from ...metadata import InterfaceMetadata
from ..rtl_data import PragmaType, Parameter

logger = logging.getLogger(__name__)


@dataclass
class BDimPragma(InterfacePragma):
    """BDIM pragma for specifying block dimension parameters.
    
    Formats:
    - @brainsmith BDIM <interface_name> <param_name>              # Single dimension
    - @brainsmith BDIM <interface_name> [<p1>, <p2>, ...]         # Multi-dimensional
    
    The parameters ARE the RTL parameters that define the block dimensions
    for the specified interface. These parameters will be removed from exposed_parameters.
    
    Special values:
    - "1" for singleton dimension (only allowed in lists with at least one parameter)
    - Parameter names for actual block dimensions
    
    IMPORTANT: 
    - Can be used on INPUT, OUTPUT, or WEIGHT interfaces (not CONTROL)
    - Only parameter names and "1" are allowed - NO other magic numbers!
    - Lists containing "1" must have at least one actual parameter
    - This ensures parameterizability which is a key design goal.
    
    Examples:
    - @brainsmith BDIM input0 TILE_SIZE               # Single block dimension
    - @brainsmith BDIM input0 [TILE_H, TILE_W]        # 2D block dimensions
    - @brainsmith BDIM weights [1, KERNEL_SIZE]       # Singleton + parameter dimension
    - @brainsmith BDIM bias [1, 1, BIAS_SIZE]         # Multiple singleton dimensions
    
    Invalid examples:
    - @brainsmith BDIM input0 [16, 32]        # ERROR: Magic numbers other than "1"
    - @brainsmith BDIM input0 1               # ERROR: Single parameter cannot be "1"
    - @brainsmith BDIM input0 [1]             # ERROR: List must have real parameters
    - @brainsmith BDIM input0 [H, W] RINDEX=0 # ERROR: RINDEX no longer supported
    """
    
    def __post_init__(self):
        super().__post_init__()
    
    def validate_with_kernel(self, kernel: 'KernelMetadata') -> None:
        """Validate that all parameter names in bdim_params exist in module parameters.
        
        This is called after the kernel metadata is available, allowing us to check
        parameter existence without class-level state.
        """
        logger.debug(f"BDIM pragma parameter validation starting")
        if not self.parsed_data:
            logger.debug(f"BDIM pragma has no parsed_data, skipping validation")
            return
        
        bdim_params = self.parsed_data.get("bdim_params", [])
        param_names = {p.name for p in kernel.parameters}
        
        logger.debug(f"BDIM pragma validating bdim_params: {bdim_params}")
        logger.debug(f"Available module parameters: {sorted(param_names)}")
        
        for element in bdim_params:
            if isinstance(element, str) and element != '1' and element.isidentifier():
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

    def _validate_shape_expression(self, shape_expr: List[str], expected_length: int) -> List:
        """
        Validate SHAPE expression for BDIM pragma.
        
        Args:
            shape_expr: List of shape elements from SHAPE=[]
            expected_length: Expected number of elements (must match param count)
            
        Returns:
            Validated shape expression ready for TilingSpec
            
        Raises:
            PragmaError: If shape expression is invalid
        """
        logger.debug(f"Validating BDIM SHAPE expression: {shape_expr}")
        
        if not isinstance(shape_expr, list):
            raise PragmaError(f"SHAPE parameter must be a list, got {type(shape_expr)}")
        
        if len(shape_expr) != expected_length:
            raise PragmaError(f"SHAPE length {len(shape_expr)} does not match parameter count {expected_length}")
        
        validated_shape = []
        for i, element in enumerate(shape_expr):
            if element == "1":
                # Singleton dimension - store as integer
                validated_shape.append(1)
            elif element == ":":
                # Full slice dimension - store as string
                validated_shape.append(":")
            elif isinstance(element, str) and element.isidentifier():
                # Parameter alias name - store as string
                validated_shape.append(element)
            else:
                raise PragmaError(f"Invalid SHAPE element '{element}' at position {i}. "
                                f"Must be '1' (singleton), ':' (full slice), or parameter name.")
        
        logger.debug(f"BDIM SHAPE expression validated: {validated_shape}")
        return validated_shape

    def _parse_inputs(self) -> Dict:
        """
        Parse BDIM pragma format with optional SHAPE parameter.
        
        Formats:
        - @brainsmith BDIM <interface_name> <param_name>              # Single dimension
        - @brainsmith BDIM <interface_name> [<p1>, <p2>, ...]         # Multi-dimensional
        - @brainsmith BDIM <interface_name> [<p1>, <p2>, ...] SHAPE=[<expr1>, <expr2>, ...]  # With shape
        
        The parameters ARE the RTL parameters that define the block dimensions.
        SHAPE expressions define how these map to the new tiling system:
        - 1: Singleton dimension
        - ":": Full slice dimension
        - "param_name": Parameter alias for node attributes
        
        Examples:
        - @brainsmith BDIM input0 TILE_SIZE               # Single parameter → shape=[":"]
        - @brainsmith BDIM input0 [TILE_H, TILE_W]        # Multi-dimensional → shape=[":", ":"]
        - @brainsmith BDIM input0 [BDIM0, BDIM1, 1] SHAPE=[TILE_SIZE, :, 1]  # With explicit shape
        
        Returns:
            Dict with parsed data including interface name, parameters, and shape
        """
        logger.debug(f"Parsing BDIM pragma: {self.inputs} at line {self.line_number}")
        
        pos = self.inputs['positional']
        named = self.inputs['named']
        
        if len(pos) < 2:
            raise PragmaError("BDIM pragma requires interface name and parameter(s)")
        
        interface_name = pos[0]
        
        # Validate interface name
        if not interface_name.replace('_', '').replace('V', '').isalnum():
            raise PragmaError(f"BDIM pragma interface name '{interface_name}' contains invalid characters")
        
        # Check if second argument is a parsed list
        if isinstance(pos[1], list):
            bdim_params = pos[1]
            
            if not bdim_params:
                raise PragmaError("BDIM pragma parameter list cannot be empty")
            
            # Validate list contents - allow '1' and parameter names
            has_real_param = False
            for element in bdim_params:
                if element == '1':
                    continue  # Allow literal "1" for singleton dimension
                elif element.isdigit():
                    raise PragmaError(f"Magic numbers not allowed in BDIM pragma except '1'. Use parameter names instead of '{element}'.")
                elif element.isidentifier():
                    has_real_param = True
                else:
                    raise PragmaError(f"Invalid parameter '{element}'. Must be '1' (singleton) or parameter name.")
            
            # Ensure at least one real parameter if using singletons
            if not has_real_param:
                raise PragmaError("BDIM pragma list must contain at least one parameter name (not just '1's)")
        else:
            # Single parameter - no brackets
            param = pos[1]
            if param == '1':
                raise PragmaError("Single BDIM parameter cannot be '1'. Use a parameter name or list syntax [1, param] for singleton dimensions.")
            elif not param.isidentifier():
                raise PragmaError(f"BDIM pragma parameter name '{param}' is not a valid identifier")
            
            bdim_params = [param]  # Store as single-element list for uniform handling
        
        # Check for any unexpected additional arguments
        if len(pos) > 2:
            raise PragmaError(f"Unexpected extra arguments in BDIM pragma: {pos[2:]}. RINDEX is no longer supported.")
        
        # Parse SHAPE parameter if provided
        shape_expr = named.get('SHAPE')
        if shape_expr:
            bdim_shape = self._validate_shape_expression(shape_expr, len(bdim_params))
        else:
            # Default: all parameters become full slices
            bdim_shape = [":"] * len(bdim_params)
        
        return {
            "interface_name": interface_name,
            "bdim_params": bdim_params,  # Always a list - for CodegenBinding
            "bdim_shape": bdim_shape     # Always a list - for TilingSpec
        }

    def apply_to_interface(self, metadata: InterfaceMetadata) -> None:
        """Apply BDIM pragma to set block dimension parameters and shape."""
        logger.debug(f"Attempting to apply BDIM pragma to interface '{metadata.name}'")
        
        # Validate that BDIM pragma is only applied to INPUT, OUTPUT, or WEIGHT interfaces (not CONTROL)
        if metadata.interface_type not in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
            error_msg = (f"BDIM pragma at line {self.line_number} cannot be applied to "
                        f"interface '{metadata.name}' of type '{metadata.interface_type.value}'. "
                        f"BDIM pragmas are only allowed on INPUT, OUTPUT, or WEIGHT interfaces.")
            logger.error(f"BDIM interface type validation failed: {error_msg}")
            raise PragmaError(error_msg)
        
        logger.debug(f"BDIM pragma applying to interface '{metadata.name}'")
        
        # Get block dimension parameters and shape
        bdim_params = self.parsed_data.get("bdim_params", [])
        bdim_shape = self.parsed_data.get("bdim_shape", [])
        
        # Update interface metadata in-place
        metadata.bdim_params = bdim_params      # RTL parameters for CodegenBinding
        metadata.bdim_shape = bdim_shape        # Shape expressions for TilingSpec
        
        logger.debug(f"BDIM pragma successfully applied to interface '{metadata.name}' "
                    f"with params={bdim_params} and shape={bdim_shape}")

    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply BDIM pragma to kernel metadata with validation."""
        # First validate parameters exist
        self.validate_with_kernel(kernel)
        
        # Then apply normally
        super().apply_to_kernel(kernel)


@dataclass
class SDimPragma(InterfacePragma):
    """SDIM pragma for specifying stream dimension parameters.
    
    Formats:
    - @brainsmith SDIM <interface_name> <param_name>              # Single dimension
    - @brainsmith SDIM <interface_name> [<p1>, <p2>, ...]         # Multi-dimensional
    
    The specified parameters ARE the RTL parameters that define the stream dimensions
    for the interface. These parameters will be removed from exposed_parameters.
    
    Special values:
    - "1" for singleton dimension (only allowed in lists with at least one parameter)
    - Parameter names for actual stream dimensions
    
    IMPORTANT: 
    - SDIM can only be used on INPUT or WEIGHT interfaces
    - Stream dimensions do not apply to OUTPUT or CONFIG interfaces
    - Lists containing "1" must have at least one actual parameter
    
    Examples:
    - @brainsmith SDIM s_axis_input0 INPUT0_SDIM               # Single stream dimension
    - @brainsmith SDIM weights_V WEIGHTS_STREAM_SIZE           # Weight interface
    - @brainsmith SDIM input0 [SDIM_H, SDIM_W, SDIM_C]        # 3D streaming (H×W×C)
    - @brainsmith SDIM weights [1, STREAM_SIZE]               # Singleton + parameter
    
    Invalid examples:
    - @brainsmith SDIM m_axis_output0 OUTPUT_SDIM             # ERROR: SDIM not allowed on OUTPUT
    - @brainsmith SDIM s_axilite_config CONFIG_SDIM           # ERROR: SDIM not allowed on CONFIG
    - @brainsmith SDIM input0 1                               # ERROR: Single parameter cannot be "1"
    - @brainsmith SDIM input0 [1]                             # ERROR: List must have real parameters
    """
    
    def __post_init__(self):
        super().__post_init__()
    
    def validate_with_kernel(self, kernel: 'KernelMetadata') -> None:
        """Validate that all parameter names exist in module parameters.
        
        This is called after the kernel metadata is available, allowing us to check
        parameter existence without class-level state.
        """
        logger.debug(f"SDIM pragma parameter validation starting")
        if not self.parsed_data:
            logger.debug(f"SDIM pragma has no parsed_data, skipping validation")
            return
        
        # Get parameter list - always a list now
        sdim_params = self.parsed_data.get("sdim_params", [])
            
        param_names = {p.name for p in kernel.parameters}
        
        logger.debug(f"SDIM pragma validating parameters: {sdim_params}")
        logger.debug(f"Available module parameters: {sorted(param_names)}")
        
        # Validate each parameter
        for param in sdim_params:
            if param != '1' and param not in param_names:
                error_msg = (f"SDIM pragma at line {self.line_number} references unknown parameter '{param}'. "
                           f"Available parameters: {sorted(param_names) if param_names else 'none'}")
                logger.error(f"SDIM parameter validation failed: {error_msg}")
                raise PragmaError(error_msg)
            elif param != '1':
                logger.debug(f"SDIM pragma parameter '{param}' validated successfully")
        
        logger.debug(f"SDIM pragma parameter validation completed successfully")

    def _validate_shape_expression(self, shape_expr: List[str], expected_length: int) -> List:
        """
        Validate SHAPE expression for SDIM pragma.
        
        Args:
            shape_expr: List of shape elements from SHAPE=[]
            expected_length: Expected number of elements (must match param count)
            
        Returns:
            Validated shape expression ready for TilingSpec
            
        Raises:
            PragmaError: If shape expression is invalid
        """
        logger.debug(f"Validating SDIM SHAPE expression: {shape_expr}")
        
        if not isinstance(shape_expr, list):
            raise PragmaError(f"SHAPE parameter must be a list, got {type(shape_expr)}")
        
        if len(shape_expr) != expected_length:
            raise PragmaError(f"SHAPE length {len(shape_expr)} does not match parameter count {expected_length}")
        
        validated_shape = []
        for i, element in enumerate(shape_expr):
            if element == "1":
                # Singleton dimension - store as integer
                validated_shape.append(1)
            elif element == ":":
                # Full slice dimension - store as string (unusual for SDIM but allowed)
                validated_shape.append(":")
            elif isinstance(element, str) and element.isidentifier():
                # Parameter alias name - store as string
                validated_shape.append(element)
            else:
                raise PragmaError(f"Invalid SHAPE element '{element}' at position {i}. "
                                f"Must be '1' (singleton), ':' (full slice), or parameter name.")
        
        logger.debug(f"SDIM SHAPE expression validated: {validated_shape}")
        return validated_shape

    def _parse_inputs(self) -> Dict:
        """
        Parse SDIM pragma format with optional SHAPE parameter.
        
        Formats:
        - @brainsmith SDIM <interface_name> <param_name>              # Single dimension
        - @brainsmith SDIM <interface_name> [<p1>, <p2>, ...]         # Multi-dimensional
        - @brainsmith SDIM <interface_name> [<p1>, <p2>, ...] SHAPE=[<expr1>, <expr2>, ...]  # With shape
        
        The parameters ARE the RTL parameters that define the stream dimensions.
        SHAPE expressions define how these map to the new tiling system:
        - 1: Singleton dimension
        - ":": Full slice dimension (not common for SDIM)
        - "param_name": Parameter alias for node attributes
        
        Examples:
        - @brainsmith SDIM input0 STREAM_SIZE               # Single parameter → shape=["STREAM_SIZE"]
        - @brainsmith SDIM input0 [SDIM_H, SDIM_W]          # Multi-dimensional → shape=["SDIM_H", "SDIM_W"]
        - @brainsmith SDIM input0 [SDIM0, SDIM1, 1] SHAPE=[SIMD, PARALLEL, 1]  # With explicit shape
        
        Returns:
            Dict with parsed data including interface name, parameters, and shape
        """
        logger.debug(f"Parsing SDIM pragma: {self.inputs} at line {self.line_number}")
        
        pos = self.inputs['positional']
        named = self.inputs['named']
        
        if len(pos) < 2:
            raise PragmaError("SDIM pragma requires at least two arguments: interface name and parameter(s)")
        
        interface_name = pos[0]
        
        # Validate interface name
        if not interface_name.replace('_', '').replace('V', '').isalnum():
            raise PragmaError(f"SDIM pragma interface name '{interface_name}' contains invalid characters")
        
        # Check if second argument is a parsed list
        if isinstance(pos[1], list):
            sdim_params = pos[1]
            
            if not sdim_params:
                raise PragmaError("SDIM pragma parameter list cannot be empty")
            
            # Validate list contents - allow '1' and parameter names
            has_real_param = False
            for element in sdim_params:
                if element == '1':
                    continue  # Allow literal "1" for singleton dimension
                elif element.isdigit():
                    raise PragmaError(f"Magic numbers not allowed in SDIM pragma except '1'. Use parameter names instead of '{element}'.")
                elif element.isidentifier():
                    has_real_param = True
                else:
                    raise PragmaError(f"Invalid parameter '{element}'. Must be '1' (singleton) or parameter name.")
            
            # Ensure at least one real parameter if using singletons
            if not has_real_param:
                raise PragmaError("SDIM pragma list must contain at least one parameter name (not just '1's)")
        else:
            # Single parameter - no brackets
            param = pos[1]
            if param == '1':
                raise PragmaError("Single SDIM parameter cannot be '1'. Use a parameter name or list syntax [1, param] for singleton dimensions.")
            elif not param.isidentifier():
                raise PragmaError(f"SDIM pragma parameter name '{param}' is not a valid identifier")
            
            sdim_params = [param]  # Store as single-element list for uniform handling
        
        # Parse SHAPE parameter if provided
        shape_expr = named.get('SHAPE')
        if shape_expr:
            sdim_shape = self._validate_shape_expression(shape_expr, len(sdim_params))
        else:
            # Default: parameters become direct node attributes (use RTL parameter names)
            sdim_shape = sdim_params[:]  # Copy the parameter names as-is
        
        return {
            "interface_name": interface_name,
            "sdim_params": sdim_params,  # Always a list - for CodegenBinding
            "sdim_shape": sdim_shape     # Always a list - for TilingSpec
        }

    def apply_to_interface(self, metadata: InterfaceMetadata) -> None:
        """Apply SDIM pragma to set stream dimension parameters and shape."""
        logger.debug(f"Attempting to apply SDIM pragma to interface '{metadata.name}'")
        
        # Validate that SDIM pragma is only applied to INPUT or WEIGHT interfaces
        if metadata.interface_type not in [InterfaceType.INPUT, InterfaceType.WEIGHT]:
            error_msg = (f"SDIM pragma at line {self.line_number} cannot be applied to "
                        f"interface '{metadata.name}' of type '{metadata.interface_type.value}'. "
                        f"SDIM pragmas are only allowed on INPUT or WEIGHT interfaces.")
            logger.error(f"SDIM interface type validation failed: {error_msg}")
            raise PragmaError(error_msg)
        
        logger.debug(f"SDIM pragma applying to interface '{metadata.name}'")
        
        # Get stream dimension parameters and shape
        sdim_params = self.parsed_data.get("sdim_params", [])
        sdim_shape = self.parsed_data.get("sdim_shape", [])
        
        # Update interface metadata in-place
        metadata.sdim_params = sdim_params      # RTL parameters for CodegenBinding
        metadata.sdim_shape = sdim_shape        # Shape expressions for TilingSpec
        
        logger.debug(f"SDIM pragma successfully applied to interface '{metadata.name}' "
                    f"with params={sdim_params} and shape={sdim_shape}")
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply SDIM pragma to kernel metadata with validation."""
        # First validate parameters exist
        self.validate_with_kernel(kernel)
        
        # Then apply normally
        super().apply_to_kernel(kernel)