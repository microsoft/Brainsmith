from __future__ import annotations
############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Data structures for RTL Parser.

This module defines the core data structures used by the RTL Parser to represent
parsed SystemVerilog modules, their components (ports, parameters, pragmas),
and the identified hardware interfaces (Global Control, AXI-Stream, AXI-Lite).

Includes:
- Enums for Port Direction and Interface Type.
- Dataclasses for Parameter, Port, Pragma, ValidationResult, PortGroup, etc.

Each class uses Python's dataclass decorator for clean initialization and
representation, along with type hints for better IDE support and runtime
validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
import logging
from datetime import datetime

# Import unified interface types from dataflow module
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup

# Set up logger for this module
logger = logging.getLogger(__name__)


class PragmaError(Exception):
    """Custom exception for errors during pragma parsing or validation."""
    pass

# --- Enums ---

class Direction(Enum):
    """Port direction enumeration."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"

class PragmaType(Enum):
    """Valid pragma types recognized by the parser."""
    TOP_MODULE = "top_module"          # Specify the top module if multiple exist
    DATATYPE = "datatype"              # Restrict datatype for an interface
    DERIVED_PARAMETER = "derived_parameter" # Link module param to python function
    WEIGHT = "weight"                  # Specify interface as a weight
    BDIM = "bdim"                      # Override block dimensions for an interface
    SDIM = "sdim"                      # Override stream dimensions for an interface
    DATATYPE_PARAM = "datatype_param"  # Map interface datatype properties to RTL parameters
    ALIAS = "alias"                    # Expose RTL parameter with different name in nodeattr

# --- Simple Data Structures ---

@dataclass
class ValidationResult:
    """Represents the result of a protocol validation check."""
    valid: bool
    message: Optional[str] = None

@dataclass
class Parameter:
    """SystemVerilog parameter representation.
    
    Attributes:
        name: Parameter identifier
        param_type: Parameter datatype (e.g., "int", "logic", "derived")
        default_value: Default value if specified
        description: Optional documentation from RTL comments
        template_param_name: Name used in the wrapper template (e.g., $NAME$).
    """
    name: str
    param_type: Optional[str] = None  # Parameter datatype (can be None for typeless parameters)
    default_value: Optional[str] = None
    description: Optional[str] = None
    template_param_name: str = field(init=False)  # Computed template parameter name

    def __post_init__(self):
        """Validate parameter attributes after initialization."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid parameter name: {self.name}")
        self.template_param_name = f"${self.name.upper()}$"

@dataclass
class Port:
    """SystemVerilog port representation.
    
    Attributes:
        name: Port identifier
        direction: Port direction (input/output/inout)
        width: Bit width expression (preserved as string)
        description: Optional documentation from RTL comments
    """
    name: str
    direction: Direction
    width: str = "1"  # Default to single bit
    description: Optional[str] = None

    def __post_init__(self):
        """Validate port attributes, converting string direction to Enum if needed."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid port name: {self.name}")
        if not isinstance(self.direction, Direction):
            if isinstance(self.direction, str):
                try:
                    self.direction = Direction(self.direction.lower())
                except ValueError:
                    raise ValueError(f"Invalid port direction string: {self.direction}")
            else:
                raise ValueError(f"Invalid port direction type: {type(self.direction)}")

# --- Intermediate Structures ---

@dataclass
class PortGroup:
    """Represents a group of related ports potentially forming an interface.

    This is an intermediate structure created by the InterfaceScanner based on
    naming conventions, before protocol validation.
    """
    interface_type: InterfaceType
    name: Optional[str] = None # e.g., "in0" for AXI-Stream, "config" for AXI-Lite
    ports: Dict[str, Port] = field(default_factory=dict) # Maps signal suffix (e.g., TDATA) or full name to Port object
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., data width for AXI

    def add_port(self, port: Port, key: Optional[str] = None) -> None:
        """Adds a port to the group, using a specific key or the port name.

        If a key (e.g., signal suffix like 'TDATA') is provided, it's used.
        Otherwise, the full port name is used as the key.
        Warns when overriding existing keys.
        """
        if key is None:
            key = port.name
        if key in self.ports:
            logger.warning(f"Overwriting port key '{key}' in PortGroup '{self.name}'")
        self.ports[key] = port

# --- Validated/Complex Structures ---

# Interface class has been removed - use InterfaceMetadata instead

# --- Pragma Structure ---

@dataclass
class Pragma:
    """Brainsmith pragma representation.
    
    Pragmas are special comments that provide additional information to the
    Hardware Kernel Generator. They follow the format:
        // @brainsmith <type> <inputs...>
    
    Attributes:
        type: Pragma type identifier (using PragmaType enum)
        inputs: List of space-separated inputs
        line_number: Source line number for error reporting
        parsed_data: Optional processed data from pragma handler
    """
    type: PragmaType
    inputs: List[str]
    line_number: int
    parsed_data: Dict = field(init=False) # Stores the result of _parse_inputs

    def __post_init__(self):
        try:
            self.parsed_data = self._parse_inputs()
        except PragmaError as e:
            logger.error(f"Error processing pragma {self.type.name} at line {self.line_number} with inputs {self.inputs}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing pragma {self.type.name} at line {self.line_number} with inputs {self.inputs}: {e}")
            # Wrap unexpected errors in PragmaError to ensure consistent error handling upstream
            raise PragmaError(f"Unexpected error during pragma {self.type.name} processing: {e}")

    def _parse_inputs(self) -> Dict:
        """
        Abstract method to parse pragma inputs.
        Subclasses must implement this method.
        """
        raise NotImplementedError(f"Pragma type {self.type.name} must implement _parse_inputs.")

    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """
        Apply this pragma to kernel metadata.
        
        Subclasses must implement this method to modify the kernel metadata
        as appropriate for their pragma type.
        
        Args:
            kernel: KernelMetadata object to modify
        """
        raise NotImplementedError(f"Pragma type {self.type.name} must implement apply_to_kernel.")

    def applies_to_interface_metadata(self, metadata: InterfaceMetadata) -> bool:
        """
        Check if this pragma applies to the given interface metadata.
        
        Base implementation returns False. Subclasses should override this method
        to implement their specific applicability logic.
        
        Args:
            metadata: InterfaceMetadata to check against
            
        Returns:
            bool: True if pragma applies to this interface, False otherwise
            
        Examples:
            >>> pragma = DatatypePragma(...)
            >>> metadata = InterfaceMetadata(name="in0", ...)
            >>> if pragma.applies_to_interface_metadata(metadata):
            ...     # Apply pragma effects
            ...     pass
        """
        return False

    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """
        Apply pragma effects to InterfaceMetadata.
        
        Base implementation returns metadata unchanged. Subclasses should override
        this method to implement their specific effects on interface metadata.
        
        This method enables a clean chain-of-responsibility pattern where each
        pragma can modify the interface metadata independently and composably.
        
        Args:
            metadata: Current InterfaceMetadata to modify
            
        Returns:
            InterfaceMetadata: Modified metadata with pragma effects applied.
                              Should return a new InterfaceMetadata instance.
        """
        return metadata


    def __str__(self):
        return f"@brainsmith {self.type.value} " + " ".join(map(str, self.inputs))

# --- Interface Pragma Base Class ---

@dataclass
class InterfacePragma(Pragma):
    """Base class for pragmas that modify interface metadata.
    
    This class provides common functionality for pragmas that target specific
    interfaces, including interface name matching and base application logic.
    """
    
    def apply_to_interface_by_name(self, interface_name: str, kernel: 'KernelMetadata') -> bool:
        """
        Find interface by name in KernelMetadata and apply pragma if found.
        
        This method centralizes the interface-finding logic that was previously
        duplicated across all interface pragma subclasses.
        
        Args:
            interface_name: Name of target interface
            kernel: KernelMetadata containing interfaces
            
        Returns:
            bool: True if interface was found and pragma applied, False otherwise
        """
        # Find the target interface
        for interface in kernel.interfaces:
            if interface.name == interface_name:
                # Apply pragma-specific logic via apply_to_metadata
                updated_metadata = self.apply_to_metadata(interface)
                
                # Update the interface in the kernel with the modified metadata
                # Since InterfaceMetadata objects are mutable, we update in place
                interface.datatype_constraints = updated_metadata.datatype_constraints
                interface.chunking_strategy = updated_metadata.chunking_strategy
                interface.datatype_metadata = updated_metadata.datatype_metadata
                interface.bdim_param = updated_metadata.bdim_param
                interface.sdim_param = updated_metadata.sdim_param
                interface.shape_params = updated_metadata.shape_params
                interface.interface_type = updated_metadata.interface_type
                interface.description = updated_metadata.description
                
                logger.debug(f"Applied {self.type.value} pragma to interface '{interface_name}'")
                return True
        
        logger.warning(f"{self.type.value} pragma target interface '{interface_name}' not found")
        return False
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """
        Apply interface pragma to kernel metadata.
        
        Default implementation handles single interface pragmas.
        Subclasses can override for more complex behavior (e.g., WeightPragma
        for multiple interfaces, DatatypeParamPragma for internal datatypes).
        """
        interface_name = self.parsed_data.get("interface_name")
        if interface_name:
            self.apply_to_interface_by_name(interface_name, kernel)

    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """
        Apply pragma effects to InterfaceMetadata.
        
        Subclasses must override this method to implement their specific effects.
        
        Args:
            metadata: Current InterfaceMetadata to modify
            
        Returns:
            InterfaceMetadata: Modified metadata with pragma effects applied
        """
        # Subclasses must override this method
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply_to_metadata()")


# --- Pragma Subclasses ---

@dataclass
class TopModulePragma(Pragma):
    def __post_init__(self): # Ensure base class __post_init__ is called if overridden
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """Handles TOP_MODULE pragma: @brainsmith top_module <module_name>"""
        logger.debug(f"Parsing TOP_MODULE pragma: {self.inputs} at line {self.line_number}")
        if len(self.inputs) != 1:
            raise PragmaError("TOP_MODULE pragma requires exactly one argument: <module_name>")
        return {"module_name": self.inputs[0]}

    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply TOP_MODULE pragma to kernel metadata."""
        # TOP_MODULE is handled during parsing to select the correct module
        # By the time we have KernelMetadata, the module has already been selected
        # This is a no-op but included for completeness
        logger.debug(f"TOP_MODULE pragma already processed during module selection")


@dataclass
class DatatypePragma(InterfacePragma):
    def __post_init__(self):
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """
        Handles DATATYPE pragma with constraint groups:
        @brainsmith DATATYPE <interface_name> <base_type> <min_bits> <max_bits>
        
        Example: @brainsmith DATATYPE in0 UINT 8 16
        Example: @brainsmith DATATYPE weights FIXED 8 8
        """
        logger.debug(f"Parsing DATATYPE pragma: {self.inputs} at line {self.line_number}")
        
        if len(self.inputs) != 4:
            raise PragmaError("DATATYPE pragma requires interface_name, base_type, min_bits, max_bits")
        
        interface_name = self.inputs[0]
        base_type = self.inputs[1].strip()
        
        try:
            min_bits = int(self.inputs[2])
            max_bits = int(self.inputs[3])
        except ValueError:
            raise PragmaError(f"DATATYPE pragma min_bits and max_bits must be integers, got: {self.inputs[2]}, {self.inputs[3]}")
        
        if min_bits <= 0:
            raise PragmaError(f"DATATYPE pragma min_bits must be positive, got: {min_bits}")
        
        if min_bits > max_bits:
            raise PragmaError(f"DATATYPE pragma min_bits ({min_bits}) cannot be greater than max_bits ({max_bits})")
        
        # Validate base type using DatatypeConstraintGroup validation
        try:
            # Test constraint group creation to validate base type
            DatatypeConstraintGroup(base_type, min_bits, max_bits)
        except ValueError as e:
            raise PragmaError(f"DATATYPE pragma invalid base type or constraints: {e}")
        
        return {
            "interface_name": interface_name,
            "base_type": base_type,
            "min_width": min_bits,
            "max_width": max_bits
        }

    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply DATATYPE pragma to modify datatype constraints."""
        # Create new datatype constraint group based on pragma
        new_constraint_group = self._create_constraint_group()
        
        # Combine with existing constraints (pragma adds to constraints, doesn't replace)
        existing_constraints = getattr(metadata, 'datatype_constraints', []) or []
        new_constraints = existing_constraints + [new_constraint_group]
        
        return metadata.update_attributes(datatype_constraints=new_constraints)

    def _create_constraint_group(self) -> DatatypeConstraintGroup:
        """Create DatatypeConstraintGroup from pragma data."""
        if not self.parsed_data:
            # Default constraint if no pragma data
            return DatatypeConstraintGroup("UINT", 8, 32)
        
        base_type = self.parsed_data.get("base_type", "UINT")
        min_width = self.parsed_data.get("min_width", 8)
        max_width = self.parsed_data.get("max_width", 32)
        
        return DatatypeConstraintGroup(base_type, min_width, max_width)


@dataclass
class BDimPragma(InterfacePragma):
    """
    BDIM pragma for block dimension chunking strategies.
    
    New simplified format:
    @brainsmith bdim <interface_name> <param_name> [SHAPE=<shape>] [RINDEX=<n>]
    
    IMPORTANT: 
    - Only allowed on INPUT or WEIGHT interfaces (not OUTPUT or CONTROL)
    - Only parameter names and ":" are allowed in shapes - NO magic numbers!
    - This ensures parameterizability which is a key design goal.
    
    Examples:
    - @brainsmith bdim in0 IN0_BDIM SHAPE=[PE]              # PE parameter, RINDEX=0 (default)
    - @brainsmith bdim in0 IN0_BDIM SHAPE=[SIMD,PE]         # SIMD,PE parameters, RINDEX=0
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
    



    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply BDIM pragma to modify chunking strategy and set parameter linkage."""
        logger.debug(f"Attempting to apply BDIM pragma to interface '{metadata.name}'")
        
        # Validate that BDIM pragma is only applied to INPUT or WEIGHT interfaces
        from brainsmith.dataflow.core.interface_types import InterfaceType
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
    """
    SDIM pragma for linking stream dimension parameters.
    
    Format: @brainsmith SDIM <interface_name> <param_name>
    
    This pragma links an interface to its stream dimension RTL parameter.
    The stream shape is inferred from the corresponding BDIM configuration.
    
    IMPORTANT: Only allowed on INPUT or WEIGHT interfaces (not OUTPUT or CONTROL)
    
    Examples:
    - @brainsmith SDIM s_axis_input0 INPUT0_SDIM
    - @brainsmith SDIM weights_V WEIGHTS_STREAM_SIZE
    
    Invalid examples:
    - @brainsmith SDIM m_axis_output0 OUTPUT0_SDIM  # ERROR: Cannot use on OUTPUT interface
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


    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply SDIM pragma to set stream dimension parameter linkage."""
        logger.debug(f"Attempting to apply SDIM pragma to interface '{metadata.name}'")
        if not self.applies_to_interface_metadata(metadata):
            logger.debug(f"SDIM pragma does not apply to interface '{metadata.name}'")
            return metadata
        
        # Validate that SDIM pragma is only applied to INPUT or WEIGHT interfaces
        from brainsmith.dataflow.core.interface_types import InterfaceType
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
        
        return metadata.update_attributes(
            sdim_param=self.parsed_data.get("param_name")
        )

    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply SDIM pragma to set stream dimension parameter."""
        # Validate that SDIM pragma is only applied to INPUT or WEIGHT interfaces
        from brainsmith.dataflow.core.interface_types import InterfaceType
        if metadata.interface_type not in [InterfaceType.INPUT, InterfaceType.WEIGHT]:
            error_msg = (f"SDIM pragma at line {self.line_number} cannot be applied to "
                        f"interface '{metadata.name}' of type '{metadata.interface_type.value}'. "
                        f"SDIM pragmas are only allowed on INPUT or WEIGHT interfaces.")
            logger.error(f"SDIM interface type validation failed: {error_msg}")
            raise PragmaError(error_msg)
        
        # Update interface with SDIM parameter
        return metadata.update_attributes(
            sdim_param=self.parsed_data.get("param_name")
        )


# --- Parameter Pragma Base Class ---

@dataclass
class ParameterPragma(Pragma):
    """Base class for pragmas that modify parameter handling.
    
    This class provides common functionality for pragmas that affect how
    parameters are exposed or processed, such as ALIAS and DERIVED_PARAMETER.
    """
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply parameter pragma to kernel metadata.
        
        Base implementation for parameter pragmas.
        Subclasses should override this method.
        """
        raise NotImplementedError(f"ParameterPragma subclass {self.__class__.__name__} must implement apply_to_kernel.")
    
    def applies_to_parameter(self, param_name: str) -> bool:
        """
        Check if this pragma applies to the given parameter.
        
        Base implementation returns False. Subclasses should override this method
        to implement their specific applicability logic.
        
        Args:
            param_name: Parameter name to check against
            
        Returns:
            bool: True if pragma applies to this parameter, False otherwise
        """
        return False


@dataclass
class DerivedParameterPragma(ParameterPragma):
    """
    Refactored DERIVED_PARAMETER pragma that assigns parameters to Python expressions.
    
    Format: @brainsmith DERIVED_PARAMETER <rtl_param> <python_expression>
    
    This pragma prevents a parameter from being exposed as a node attribute and
    instead assigns it to a Python expression or function call in the RTLBackend.
    
    Examples:
    - @brainsmith DERIVED_PARAMETER SIMD self.get_input_datatype().bitwidth()
    - @brainsmith DERIVED_PARAMETER MEM_SIZE self.calc_wmem()
    """
    
    def __post_init__(self):
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """Parse DERIVED_PARAMETER pragma: @brainsmith DERIVED_PARAMETER <rtl_param> <python_expression>"""
        logger.debug(f"Parsing DERIVED_PARAMETER pragma: {self.inputs} at line {self.line_number}")
        if len(self.inputs) < 2:
            raise PragmaError(f"DERIVED_PARAMETER pragma at line {self.line_number} requires parameter name and Python expression. Got: {self.inputs}")
        
        param_name = self.inputs[0]
        # Join remaining inputs as the Python expression (allows spaces)
        python_expression = " ".join(self.inputs[1:])
        
        # Validate parameter name
        if not param_name.isidentifier():
            raise PragmaError(f"DERIVED_PARAMETER pragma parameter name '{param_name}' is not a valid identifier")
        
        return {"param_name": param_name, "python_expression": python_expression}
    
    def applies_to_parameter(self, param_name: str) -> bool:
        """Check if this pragma applies to the given parameter."""
        return self.parsed_data.get("param_name") == param_name
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply DERIVED_PARAMETER pragma to kernel metadata."""
        param_name = self.parsed_data.get("param_name")
        python_expression = self.parsed_data.get("python_expression")
        
        # Remove from exposed parameters
        if param_name in kernel.exposed_parameters:
            kernel.exposed_parameters.remove(param_name)
        
        # Add to parameter pragma data
        if kernel.parameter_pragma_data is None:
            kernel.parameter_pragma_data = {"aliases": {}, "derived": {}}
        if "derived" not in kernel.parameter_pragma_data:
            kernel.parameter_pragma_data["derived"] = {}
        
        kernel.parameter_pragma_data["derived"][param_name] = python_expression
        
        logger.debug(f"Applied DERIVED_PARAMETER pragma: {param_name} -> {python_expression}")



@dataclass
class AliasPragma(ParameterPragma):
    """
    ALIAS pragma for exposing RTL parameters with different names in nodeattr.
    
    Format: @brainsmith ALIAS <rtl_param> <nodeattr_name>
    
    This pragma allows an RTL parameter to be exposed as a node attribute with
    a different name, improving the API for users.
    
    Examples:
    - @brainsmith ALIAS PE parallelism_factor
    - @brainsmith ALIAS C num_channels
    
    Validation:
    - The nodeattr_name cannot match any existing module parameter name
    """
    
    def __post_init__(self):
        super().__post_init__()
    
    def _parse_inputs(self) -> Dict:
        """Parse ALIAS pragma: @brainsmith ALIAS <rtl_param> <nodeattr_name>"""
        logger.debug(f"Parsing ALIAS pragma: {self.inputs} at line {self.line_number}")
        if len(self.inputs) != 2:
            raise PragmaError(f"ALIAS pragma at line {self.line_number} requires exactly 2 arguments: <rtl_param> <nodeattr_name>. Got: {self.inputs}")
        
        rtl_param = self.inputs[0]
        nodeattr_name = self.inputs[1]
        
        # Validate both names are valid identifiers
        if not rtl_param.isidentifier():
            raise PragmaError(f"ALIAS pragma RTL parameter name '{rtl_param}' is not a valid identifier")
        if not nodeattr_name.isidentifier():
            raise PragmaError(f"ALIAS pragma nodeattr name '{nodeattr_name}' is not a valid identifier")
        
        return {"rtl_param": rtl_param, "nodeattr_name": nodeattr_name}
    
    def applies_to_parameter(self, param_name: str) -> bool:
        """Check if this pragma applies to the given parameter."""
        return self.parsed_data.get("rtl_param") == param_name
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply ALIAS pragma to kernel metadata."""
        rtl_param = self.parsed_data.get("rtl_param")
        nodeattr_name = self.parsed_data.get("nodeattr_name")
        
        # Validate against other parameters
        self.validate_against_parameters(kernel.parameters)
        
        # Remove from exposed parameters (it will be exposed with the alias name)
        if rtl_param in kernel.exposed_parameters:
            kernel.exposed_parameters.remove(rtl_param)
        
        # Add to parameter pragma data
        if kernel.parameter_pragma_data is None:
            kernel.parameter_pragma_data = {"aliases": {}, "derived": {}}
        if "aliases" not in kernel.parameter_pragma_data:
            kernel.parameter_pragma_data["aliases"] = {}
        
        kernel.parameter_pragma_data["aliases"][rtl_param] = nodeattr_name
        
        logger.debug(f"Applied ALIAS pragma: {rtl_param} -> {nodeattr_name}")
    
    def validate_against_parameters(self, all_parameters: List[Parameter]) -> None:
        """
        Validate that the nodeattr_name doesn't conflict with existing parameters.
        
        Args:
            all_parameters: List of all module parameters
            
        Raises:
            PragmaError: If nodeattr_name matches an existing parameter name
        """
        nodeattr_name = self.parsed_data.get("nodeattr_name")
        rtl_param = self.parsed_data.get("rtl_param")
        
        # Check if nodeattr_name conflicts with any other parameter
        for param in all_parameters:
            if param.name == nodeattr_name and param.name != rtl_param:
                raise PragmaError(
                    f"ALIAS pragma at line {self.line_number}: nodeattr name '{nodeattr_name}' "
                    f"conflicts with existing parameter '{param.name}'. "
                    f"Choose a different alias name."
                )


@dataclass
class WeightPragma(InterfacePragma):
    def __post_init__(self):
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """Handles WEIGHT pragma: @brainsmith WEIGHT <interface_name_0> [<interface_name_1> ...]"""
        logger.debug(f"Parsing WEIGHT pragma: {self.inputs} at line {self.line_number}")
        if not self.inputs: # Equivalent to len(self.inputs) < 1
            raise PragmaError(f"WEIGHT pragma at line {self.line_number} requires at least one argument: <interface_name_0> [...]. Got: {self.inputs}")
        
        # All inputs are interface names
        interface_names = self.inputs
        return {"interface_names": interface_names}


    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply WEIGHT pragma to kernel metadata."""
        interface_names = self.parsed_data.get("interface_names", [])
        
        # WeightPragma handles multiple interfaces, so we apply to each one
        for interface_name in interface_names:
            self.apply_to_interface_by_name(interface_name, kernel)
    
    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply WEIGHT pragma to mark interface as weight type."""
        from brainsmith.dataflow.core.interface_types import InterfaceType
        return metadata.update_attributes(
            interface_type=InterfaceType.WEIGHT  # Override type
        )


@dataclass 
class DatatypeParamPragma(InterfacePragma):
    """
    Maps specific RTL parameters to interface datatype properties.
    
    Syntax: @brainsmith DATATYPE_PARAM <interface_name> <property_type> <parameter_name>
    
    Examples:
    // @brainsmith DATATYPE_PARAM s_axis_input0 width INPUT0_WIDTH
    // @brainsmith DATATYPE_PARAM s_axis_input0 signed SIGNED_INPUT0
    // @brainsmith DATATYPE_PARAM s_axis_query width QUERY_WIDTH
    // @brainsmith DATATYPE_PARAM s_axis_wq width WQ_WIDTH
    """
    
    def _parse_inputs(self) -> Dict:
        if len(self.inputs) != 3:
            raise PragmaError("DATATYPE_PARAM pragma requires interface_name, property_type, parameter_name")
        
        interface_name = self.inputs[0]
        property_type = self.inputs[1].lower()
        parameter_name = self.inputs[2]
        
        # Validate property type
        valid_properties = ['width', 'signed', 'format', 'bias', 'fractional_width']
        if property_type not in valid_properties:
            raise PragmaError(f"Invalid property_type '{property_type}'. Must be one of: {valid_properties}")
        
        return {
            "interface_name": interface_name,
            "property_type": property_type, 
            "parameter_name": parameter_name
        }
    
    
    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply DATATYPE_PARAM pragma to set datatype parameter mapping."""
        property_type = self.parsed_data['property_type']
        parameter_name = self.parsed_data['parameter_name']
        
        # Import DatatypeMetadata
        from brainsmith.dataflow.core.datatype_metadata import DatatypeMetadata
        
        # Get or create DatatypeMetadata
        if metadata.datatype_metadata is None:
            # Create new DatatypeMetadata with interface name
            # For now, we'll create with just width and update it
            if property_type == 'width':
                new_dt = DatatypeMetadata(name=metadata.name, width=parameter_name)
            else:
                # Need a width parameter first - use default naming
                new_dt = DatatypeMetadata(
                    name=metadata.name, 
                    width=f"{metadata.name}_WIDTH",
                    **{property_type: parameter_name}
                )
            metadata = metadata.update_attributes(datatype_metadata=new_dt)
        else:
            # Update existing DatatypeMetadata
            updated_dt_metadata = metadata.datatype_metadata.update(**{property_type: parameter_name})
            metadata = metadata.update_attributes(datatype_metadata=updated_dt_metadata)
        
        return metadata
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply DATATYPE_PARAM pragma to kernel metadata."""
        interface_name = self.parsed_data.get("interface_name")
        property_type = self.parsed_data.get("property_type")
        parameter_name = self.parsed_data.get("parameter_name")
        
        # Try to apply to interface first using unified method
        if self.apply_to_interface_by_name(interface_name, kernel):
            # Successfully applied to interface, also remove from exposed parameters
            if parameter_name in kernel.exposed_parameters:
                kernel.exposed_parameters.remove(parameter_name)
            return
        
        # If interface not found, this might be an internal datatype
        # Add to internal datatypes
        internal_dt = self.create_standalone_datatype()
        
        if kernel.internal_datatypes is None:
            kernel.internal_datatypes = []
        
        # Check if we already have this datatype and merge
        existing_dt = None
        for dt in kernel.internal_datatypes:
            if dt.name == interface_name:
                existing_dt = dt
                break
        
        if existing_dt:
            # Update existing datatype
            setattr(existing_dt, property_type, parameter_name)
        else:
            # Add new internal datatype
            kernel.internal_datatypes.append(internal_dt)
        
        # Remove parameter from exposed list
        if parameter_name in kernel.exposed_parameters:
            kernel.exposed_parameters.remove(parameter_name)
        
        logger.debug(f"Applied DATATYPE_PARAM pragma to internal datatype '{interface_name}': {property_type}={parameter_name}")
    
    def create_standalone_datatype(self) -> 'DatatypeMetadata':
        """Create a standalone DatatypeMetadata for internal mechanisms."""
        from brainsmith.dataflow.core.datatype_metadata import DatatypeMetadata
        
        interface_name = self.parsed_data['interface_name']
        property_type = self.parsed_data['property_type']
        parameter_name = self.parsed_data['parameter_name']
        
        # Create datatype with only the specified property
        return DatatypeMetadata(
            name=interface_name,
            **{property_type: parameter_name},
            description=f"Internal datatype binding from pragma at line {self.line_number}"
        )


# Note: Interface class has been removed in favor of InterfaceMetadata
