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
- Dataclasses for Parameter, Port, Pragma, ValidationResult, PortGroup, and HWKernel.

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

class InterfaceNameMatcher:
    """Mixin providing interface name matching utilities for pragma classes."""
    
    @staticmethod
    def _interface_names_match(pragma_name: str, interface_name: str) -> bool:
        """
        Check if pragma interface name matches actual interface name.
        
        Supports multiple matching patterns to handle variations in interface
        naming conventions, including AXI naming patterns.
        
        Args:
            pragma_name: Interface name specified in pragma
            interface_name: Actual interface name from RTL parsing
            
        Returns:
            bool: True if names match according to any supported pattern
            
        Examples:
            >>> InterfaceNameMatcher._interface_names_match("in0", "in0")
            True
            >>> InterfaceNameMatcher._interface_names_match("in0", "in0_V_data_V")
            True
            >>> InterfaceNameMatcher._interface_names_match("in0_V_data_V", "in0")
            True
            >>> InterfaceNameMatcher._interface_names_match("weights", "bias")
            False
        """
        # Exact match
        if pragma_name == interface_name:
            return True
        
        # Prefix match (e.g., "in0" matches "in0_V_data_V")
        if interface_name.startswith(pragma_name):
            return True
        
        # Reverse prefix match (e.g., "in0_V_data_V" matches "in0")
        if pragma_name.startswith(interface_name):
            return True
        
        # Base name matching (remove common suffixes)
        pragma_base = pragma_name.replace('_V_data_V', '').replace('_data', '')
        interface_base = interface_name.replace('_V_data_V', '').replace('_data', '')
        
        return pragma_base == interface_base


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

    def apply(self, **kwargs) -> Any:
        """
        Abstract method to apply the pragma's effects.
        Subclasses must implement this method and can return any relevant data.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments. Subclasses will expect specific
                      keys like 'interfaces', 'parameters', 'hw_kernel'.
        """
        raise NotImplementedError(f"Pragma type {self.type.name} must implement apply.")

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

    def applies_to_interface(self, interface) -> bool:
        """
        DEPRECATED: Use applies_to_interface_metadata instead.
        
        Check if this pragma applies to the given interface.
        This method is deprecated and will be removed in a future version.
        Use applies_to_interface_metadata() for new code.
        """
        import warnings
        warnings.warn(
            "applies_to_interface() is deprecated. Use applies_to_interface_metadata() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Create temporary InterfaceMetadata for backward compatibility
        from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
        temp_metadata = InterfaceMetadata(
            name=interface.name,
            interface_type=interface.type,
            datatype_constraints=[],
            chunking_strategy=None
        )
        return self.applies_to_interface_metadata(temp_metadata)

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

    def apply_to_interface_metadata(self, interface, 
                                  metadata: InterfaceMetadata) -> InterfaceMetadata:
        """
        DEPRECATED: Use apply_to_metadata instead.
        
        Apply pragma effects to InterfaceMetadata with Interface compatibility.
        This method is deprecated and will be removed in a future version.
        Use apply_to_metadata() for new code.
        """
        import warnings
        warnings.warn(
            "apply_to_interface_metadata() is deprecated. Use apply_to_metadata() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.apply_to_metadata(metadata)

    def __str__(self):
        return f"@brainsmith {self.type.value} " + " ".join(map(str, self.inputs))

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

    def apply(self, **kwargs) -> Any:
        """Applies the TOP_MODULE pragma."""
        hw_kernel: Optional[HWKernel] = kwargs.get('hw_kernel')
        # The primary effect of TOP_MODULE (identifying the main module) is typically
        # handled by the Parser when it first processes the list of all pragmas
        # to find the target module name before full HWKernel construction.
        if hw_kernel and self.parsed_data.get("module_name"):
            current_kernel_name = hw_kernel.name
            new_kernel_name = self.parsed_data["module_name"]
            if current_kernel_name and current_kernel_name != new_kernel_name:
                logger.warning(
                    f"TOP_MODULE pragma at line {self.line_number} trying to change HWKernel name "
                    f"from '{current_kernel_name}' to '{new_kernel_name}'. This might be an issue "
                    f"if the kernel was already identified differently. Sticking to '{new_kernel_name}'."
                )
            hw_kernel.name = new_kernel_name
            logger.info(f"TOP_MODULE pragma applied: HWKernel name set to '{hw_kernel.name}' based on pragma at line {self.line_number}.")
        elif not hw_kernel and self.parsed_data.get("module_name"):
            logger.debug(f"TOP_MODULE pragma at line {self.line_number} processed. Module name '{self.parsed_data.get('module_name')}' is available. HWKernel object not provided for immediate update.")
        else:
            logger.debug(f"TOP_MODULE pragma at line {self.line_number} processed. No module name in parsed_data or no HWKernel provided.")


@dataclass
class DatatypePragma(Pragma, InterfaceNameMatcher):
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


    def applies_to_interface_metadata(self, metadata: InterfaceMetadata) -> bool:
        """Check if this DATATYPE pragma applies to the given interface metadata."""
        if not self.parsed_data:
            return False
        
        pragma_interface_name = self.parsed_data.get('interface_name')
        if not pragma_interface_name:
            return False
        
        return self._interface_names_match(pragma_interface_name, metadata.name)

    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply DATATYPE pragma to modify datatype constraints."""
        if not self.applies_to_interface_metadata(metadata):
            return metadata
        
        # Create new datatype constraint group based on pragma
        new_constraint_group = self._create_constraint_group()
        
        # Combine with existing constraints (pragma adds to constraints, doesn't replace)
        existing_constraints = getattr(metadata, 'datatype_constraints', [])
        new_constraints = existing_constraints + [new_constraint_group]
        
        return InterfaceMetadata(
            name=metadata.name,
            interface_type=metadata.interface_type,
            datatype_constraints=new_constraints,
            chunking_strategy=metadata.chunking_strategy
        )

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
class BDimPragma(Pragma, InterfaceNameMatcher):
    """
    BDIM pragma for block dimension chunking strategies.
    
    New simplified format:
    @brainsmith bdim <interface_name> [<shape>] [RINDEX=<n>]
    
    IMPORTANT: Only parameter names and ":" are allowed in shapes - NO magic numbers!
    This ensures parameterizability which is a key design goal.
    
    Examples:
    - @brainsmith bdim in0 [PE]              # PE parameter, RINDEX=0 (default)
    - @brainsmith bdim in0 [SIMD,PE]         # SIMD,PE parameters, RINDEX=0
    - @brainsmith bdim in0 [:,:] RINDEX=1    # Full dimensions, starting from RINDEX=1
    - @brainsmith bdim in0 [PE,:]            # PE parameter and full dimension
    - @brainsmith bdim weights [TILE_SIZE]   # TILE_SIZE parameter
    
    Invalid (magic numbers not allowed):
    - @brainsmith bdim in0 [16]              # ERROR: Magic number 16
    - @brainsmith bdim in0 [8,4]             # ERROR: Magic numbers 8,4
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
        
        Format: @brainsmith bdim <interface_name> [<shape>] [RINDEX=<n>]
        
        Returns:
            Dict with parsed data including interface name, block shape, and rindex
        """
        logger.debug(f"Parsing BDIM pragma: {self.inputs} at line {self.line_number}")
        
        if len(self.inputs) < 2:
            raise PragmaError("BDIM pragma requires interface name and block shape")
        
        interface_name = self.inputs[0]
        
        # Validate interface name
        if not interface_name.replace('_', '').replace('V', '').isalnum():
            raise PragmaError(f"BDIM pragma interface name '{interface_name}' contains invalid characters")
        
        # Parse block shape [shape1,shape2,...]
        if len(self.inputs) < 2 or not (self.inputs[1].startswith('[') and self.inputs[1].endswith(']')):
            raise PragmaError("BDIM pragma requires block shape in [shape1,shape2,...] format")
        
        shape_str = self.inputs[1][1:-1].strip()
        if not shape_str:
            raise PragmaError("BDIM pragma block shape cannot be empty")
        
        # Parse shape elements - only allow ':' and parameter names (NO magic numbers)
        block_shape = []
        
        for element in shape_str.split(','):
            element = element.strip()
            if element == ':':
                block_shape.append(':')
            elif element.isdigit():
                raise PragmaError(f"Magic numbers not allowed in BDIM pragma shape. Use parameter names instead of '{element}'.")
            elif element.isidentifier():
                # Parameter name - defer validation until apply phase when parameters are available
                block_shape.append(element)
            else:
                raise PragmaError(f"Invalid shape element '{element}'. Must be ':' (full dimension) or parameter name.")
        
        # Parse optional RINDEX
        rindex = 0
        if len(self.inputs) > 2:
            rindex_str = self.inputs[2]
            if rindex_str.startswith('RINDEX='):
                try:
                    rindex = int(rindex_str[7:])
                    if rindex < 0:
                        raise PragmaError(f"RINDEX must be non-negative, got {rindex}")
                except ValueError:
                    raise PragmaError(f"Invalid RINDEX value: {rindex_str}")
            else:
                raise PragmaError(f"Unknown parameter '{rindex_str}'. Expected 'RINDEX=n'")
        
        return {
            "interface_name": interface_name,
            "block_shape": block_shape,
            "rindex": rindex
        }
    


    def applies_to_interface_metadata(self, metadata: InterfaceMetadata) -> bool:
        """Check if this BDIM pragma applies to the given interface metadata."""
        if not self.parsed_data:
            logger.debug(f"BDIM pragma has no parsed_data")
            return False
        
        pragma_interface_name = self.parsed_data.get('interface_name')
        if not pragma_interface_name:
            logger.debug(f"BDIM pragma has no interface_name in parsed_data")
            return False
        
        match_result = self._interface_names_match(pragma_interface_name, metadata.name)
        logger.debug(f"BDIM pragma interface matching: '{pragma_interface_name}' <-> '{metadata.name}' = {match_result}")
        return match_result

    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply BDIM pragma to modify chunking strategy."""
        logger.debug(f"Attempting to apply BDIM pragma to interface '{metadata.name}'")
        if not self.applies_to_interface_metadata(metadata):
            logger.debug(f"BDIM pragma does not apply to interface '{metadata.name}'")
            return metadata
        
        logger.debug(f"BDIM pragma applying to interface '{metadata.name}' - validating parameters")
        # Validate parameters now that module parameters are available
        self._validate_parameters()
        
        # Create chunking strategy from pragma data
        new_strategy = self._create_chunking_strategy()
        logger.debug(f"BDIM pragma successfully applied to interface '{metadata.name}'")
        
        return InterfaceMetadata(
            name=metadata.name,
            interface_type=metadata.interface_type,
            datatype_constraints=metadata.datatype_constraints,
            chunking_strategy=new_strategy
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
class DerivedParameterPragma(Pragma):
    def __post_init__(self):
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """Handles DERIVED_PARAMETER pragma: @brainsmith DERIVED_PARAMETER <python_function_name> <param_name_0> [<param_name_1> ...]"""
        logger.debug(f"Parsing DERIVED_PARAMETER pragma: {self.inputs} at line {self.line_number}")
        if len(self.inputs) < 2:
            raise PragmaError(f"DERIVED_PARAMETER pragma at line {self.line_number} requires at least two arguments: <python_function_name> <param_name_0> [...]. Got: {self.inputs}")
        
        python_function_name = self.inputs[0]
        param_names = self.inputs[1:]
        return {"python_function_name": python_function_name, "param_names": param_names}

    def apply(self, **kwargs) -> Any:
        """Applies the DERIVED_PARAMETER pragma by adding a new parameter to the HWKernel."""
        hw_kernel: Optional[HWKernel] = kwargs.get('hw_kernel')
        if not hw_kernel:
            logger.warning(f"DERIVED_PARAMETER pragma at line {self.line_number}: hw_kernel not provided. Cannot apply.")
            return

        param_name = self.parsed_data.get("param_name")
        param_value = self.parsed_data.get("param_value")

        if not param_name or param_value is None: # Check param_value is not None explicitly
            logger.warning(f"DERIVED_PARAMETER pragma at line {self.line_number}: Missing param_name or param_value in parsed_data. Cannot apply. Data: {self.parsed_data}")
            return

        # Check if a parameter with the same name already exists from the module definition (non-derived)
        existing_module_param = next((p for p in hw_kernel.parameters if p.name == param_name and p.param_type != "derived"), None)
        if existing_module_param:
            logger.error(f"DERIVED_PARAMETER pragma at line {self.line_number}: Parameter '{param_name}' already exists in the module definition with type '{existing_module_param.param_type}'. Derived parameters cannot override module parameters. Skipping.")
            return

        # Check if this derived parameter (by name) has already been added by another pragma
        existing_derived_param = next((p for p in hw_kernel.parameters if p.name == param_name and p.param_type == "derived"), None)
        if existing_derived_param:
            if existing_derived_param.default_value == param_value:
                logger.info(f"DERIVED_PARAMETER pragma at line {self.line_number}: Parameter '{param_name}' with value '{param_value}' (type: derived) already added by a previous pragma. Skipping duplicate.")
            else:
                logger.error(f"DERIVED_PARAMETER pragma at line {self.line_number}: Parameter '{param_name}' (type: derived) already added by a previous pragma with a different value ('{existing_derived_param.default_value}' vs '{param_value}'). Conflicting pragmas. Skipping.")
            return

        try:
            new_param = Parameter(
                name=param_name,
                param_type="derived",  # Mark this parameter as 'derived'
                default_value=param_value
            )
            hw_kernel.parameters.append(new_param)
            logger.info(f"Applied DERIVED_PARAMETER pragma from line {self.line_number}: Added parameter '{param_name}' = '{param_value}' (type: derived) to HWKernel '{hw_kernel.name}'.")
        except ValueError as e:  # Catch potential errors from Parameter constructor (e.g., invalid name)
            logger.error(f"DERIVED_PARAMETER pragma at line {self.line_number}: Error creating Parameter object for '{param_name}': {e}. Skipping.")
            # Optionally, re-raise as PragmaError to halt processing if critical
            # raise PragmaError(f"Error creating derived parameter '{param_name}': {e}") from e
        return # Explicitly return None or Any relevant data if needed in future


@dataclass
class WeightPragma(Pragma, InterfaceNameMatcher):
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

    def applies_to_interface_metadata(self, metadata: InterfaceMetadata) -> bool:
        """Check if this WEIGHT pragma applies to the given interface metadata."""
        if not self.parsed_data:
            return False
        
        interface_names = self.parsed_data.get('interface_names', [])
        for pragma_name in interface_names:
            if self._interface_names_match(pragma_name, metadata.name):
                return True
        return False

    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply WEIGHT pragma to mark interface as weight type."""
        if not self.applies_to_interface_metadata(metadata):
            return metadata
        
        return InterfaceMetadata(
            name=metadata.name,
            interface_type=InterfaceType.WEIGHT,  # Override type
            datatype_constraints=metadata.datatype_constraints,
            chunking_strategy=metadata.chunking_strategy
        )


# Note: Interface class has been removed in favor of InterfaceMetadata
