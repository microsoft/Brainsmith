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
- Dataclasses for Parameter, Port, Pragma, ValidationResult, PortGroup, Interface, and HWKernel.

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
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint

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

# Legacy InterfaceType enum removed - now using unified types from dataflow module
# All interface type references now use brainsmith.dataflow.core.interface_types.InterfaceType

class PragmaType(Enum):
    """Valid pragma types recognized by the parser."""
    TOP_MODULE = "top_module"          # Specify the top module if multiple exist
    DATATYPE = "datatype"              # Restrict datatype for an interface
    DERIVED_PARAMETER = "derived_parameter" # Link module param to python function
    WEIGHT = "weight"                  # Specify interface as a weight
    BDIM = "bdim"                      # Override block dimensions for an interface (new preferred)
    TDIM = "tdim"                      # Override tensor dimensions for an interface (deprecated, use BDIM)

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

@dataclass
class Interface:
    """DEPRECATED: Represents a fully validated and identified interface.
    
    This class is being phased out in favor of InterfaceMetadata for direct
    dataflow integration. Currently maintained for pragma system compatibility.
    
    Created by the InterfaceBuilder after a PortGroup successfully passes
    validation by the ProtocolValidator.
    
    TODO: Remove this class after pragma system is refactored to work with
    InterfaceMetadata objects directly.
    """
    name: str # e.g., "global", "in0", "config"
    type: InterfaceType
    ports: Dict[str, Port] # Maps signal suffix/name to Port object
    validation_result: ValidationResult
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., data width, address width
    wrapper_name: Optional[str] = None  # New attribute to store wrapper name

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

    def applies_to_interface(self, interface: 'Interface') -> bool:
        """
        Check if this pragma applies to the given interface.
        
        Base implementation returns False. Subclasses should override this method
        to implement their specific applicability logic.
        
        Args:
            interface: Interface to check against (temporary compatibility)
            
        Returns:
            bool: True if pragma applies to this interface, False otherwise
            
        Examples:
            >>> pragma = DatatypePragma(...)
            >>> interface = Interface(name="in0", ...)
            >>> if pragma.applies_to_interface(interface):
            ...     # Apply pragma effects
            ...     pass
        """
        return False

    def apply_to_interface_metadata(self, interface: 'Interface', 
                                  metadata: InterfaceMetadata) -> InterfaceMetadata:
        """
        Apply pragma effects to InterfaceMetadata.
        
        Base implementation returns metadata unchanged. Subclasses should override
        this method to implement their specific effects on interface metadata.
        
        This method enables a clean chain-of-responsibility pattern where each
        pragma can modify the interface metadata independently and composably.
        
        Args:
            interface: Interface this pragma applies to
            metadata: Current InterfaceMetadata to modify
            
        Returns:
            InterfaceMetadata: Modified metadata with pragma effects applied.
                              Should return a new InterfaceMetadata instance.
            
        Examples:
            >>> pragma = DatatypePragma(...)
            >>> interface = Interface(name="in0", ...)
            >>> base_metadata = InterfaceMetadata(...)
            >>> updated_metadata = pragma.apply_to_interface_metadata(interface, base_metadata)
        """
        return metadata

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
        Handles enhanced DATATYPE pragma:
        @brainsmith DATATYPE <interface_name> <base_types> <min_bits> <max_bits>
        
        Example: @brainsmith DATATYPE in0 INT,UINT 1 16
        Example: @brainsmith DATATYPE weights FIXED 8 8
        """
        logger.debug(f"Parsing enhanced DATATYPE pragma: {self.inputs} at line {self.line_number}")
        
        if len(self.inputs) != 4:
            raise PragmaError("DATATYPE pragma requires interface_name, base_types, min_bits, max_bits")
        
        interface_name = self.inputs[0]
        base_types = [t.strip() for t in self.inputs[1].split(',')]
        
        try:
            min_bits = int(self.inputs[2])
            max_bits = int(self.inputs[3])
        except ValueError:
            raise PragmaError(f"DATATYPE pragma min_bits and max_bits must be integers, got: {self.inputs[2]}, {self.inputs[3]}")
        
        if min_bits > max_bits:
            raise PragmaError(f"DATATYPE pragma min_bits ({min_bits}) cannot be greater than max_bits ({max_bits})")
        
        # Validate base types
        valid_base_types = {'INT', 'UINT', 'FLOAT', 'FIXED'}
        for base_type in base_types:
            if base_type not in valid_base_types:
                raise PragmaError(f"DATATYPE pragma invalid base type '{base_type}'. Valid types: {valid_base_types}")
        
        return {
            "interface_name": interface_name,
            "base_types": base_types,
            "min_bitwidth": min_bits,
            "max_bitwidth": max_bits
        }


    def applies_to_interface(self, interface: 'Interface') -> bool:
        """Check if this DATATYPE pragma applies to the given interface."""
        if not self.parsed_data:
            return False
        
        pragma_interface_name = self.parsed_data.get('interface_name')
        if not pragma_interface_name:
            return False
        
        return self._interface_names_match(pragma_interface_name, interface.name)

    def apply_to_interface_metadata(self, interface: 'Interface', 
                                  metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply DATATYPE pragma to modify allowed datatypes."""
        if not self.applies_to_interface(interface):
            return metadata
        
        # Create new datatype constraints based on pragma
        new_constraints = self._create_datatype_constraints()
        
        return InterfaceMetadata(
            name=metadata.name,
            interface_type=metadata.interface_type,
            allowed_datatypes=new_constraints,
            chunking_strategy=metadata.chunking_strategy
        )

    def _create_datatype_constraints(self) -> List[DataTypeConstraint]:
        """Create DataTypeConstraint objects from pragma data."""
        if not self.parsed_data:
            return []
        
        base_types = self.parsed_data.get("base_types", ["UINT"])
        min_bits = self.parsed_data.get("min_bitwidth", 8)
        max_bits = self.parsed_data.get("max_bitwidth", 32)
        
        constraints = []
        for base_type in base_types:
            # Create constraints for the bitwidth range
            if min_bits == max_bits:
                # Single bitwidth
                constraints.append(DataTypeConstraint(
                    finn_type=f"{base_type}{min_bits}",
                    bit_width=min_bits,
                    signed=(base_type == "INT")
                ))
            else:
                # Range of bitwidths - create constraints for min and max
                constraints.extend([
                    DataTypeConstraint(
                        finn_type=f"{base_type}{min_bits}",
                        bit_width=min_bits,
                        signed=(base_type == "INT")
                    ),
                    DataTypeConstraint(
                        finn_type=f"{base_type}{max_bits}",
                        bit_width=max_bits,
                        signed=(base_type == "INT")
                    )
                ])
        
        return constraints


@dataclass
class BDimPragma(Pragma, InterfaceNameMatcher):
    """
    BDIM pragma for block dimension chunking strategies.
    
    Supports two formats:
    1. Legacy: @brainsmith BDIM <interface_name> <dim1_expr> <dim2_expr> ... <dimN_expr>
    2. Enhanced: @brainsmith BDIM <interface_name> <chunk_index> [<chunk_sizes>]
    
    Examples:
    - Legacy: @brainsmith BDIM in0 PE*CHANNELS 1
    - Enhanced: @brainsmith BDIM in0_V_data_V -1 [16]
    - Enhanced: @brainsmith BDIM out0_V_data_V 2 [4,8]
    """
    
    def __post_init__(self):
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """
        Handles both legacy and enhanced BDIM pragma formats.
        
        Enhanced format: @brainsmith BDIM <interface_name> <chunk_index> [<chunk_sizes>]
        Legacy format: @brainsmith BDIM <interface_name> <dim1_expr> <dim2_expr> ... <dimN_expr>
        
        Returns:
            Dict with parsed data including format type, interface name, and strategy/expressions
        """
        logger.debug(f"Parsing BDIM pragma: {self.inputs} at line {self.line_number}")
        
        if len(self.inputs) < 2:
            raise PragmaError("BDIM pragma requires interface name and at least one additional argument")
        
        interface_name = self.inputs[0]
        
        # Validate interface name
        if not interface_name.replace('_', '').replace('V', '').isalnum():
            raise PragmaError(f"BDIM pragma interface name '{interface_name}' contains invalid characters")
        
        # Detect enhanced format vs legacy format
        # Enhanced format: second argument is index AND third argument has brackets [...]
        # Legacy format: multiple dimension expressions without brackets
        
        if len(self.inputs) >= 3:
            try:
                chunk_index = int(self.inputs[1])
                # Check if third argument looks like enhanced format with brackets
                if len(self.inputs) == 3 and self.inputs[2].startswith('[') and self.inputs[2].endswith(']'):
                    # Enhanced format: @brainsmith BDIM interface_name -1 [16]
                    return self._parse_enhanced_format(interface_name, chunk_index, self.inputs[2:])
                else:
                    # Legacy format: @brainsmith BDIM interface_name 8 1
                    return self._parse_legacy_format(interface_name, self.inputs[1:])
            except ValueError:
                # Second argument is not an integer, must be legacy
                return self._parse_legacy_format(interface_name, self.inputs[1:])
        else:
            # Only two arguments, legacy format
            return self._parse_legacy_format(interface_name, self.inputs[1:])
    
    def _parse_enhanced_format(self, interface_name: str, chunk_index: int, remaining_inputs: List[str]) -> Dict:
        """Parse enhanced BDIM format with chunking strategy."""
        if len(remaining_inputs) != 1:
            raise PragmaError("Enhanced BDIM pragma requires exactly one chunk_sizes argument in [size1,size2,...] format")
        
        chunk_sizes_str = remaining_inputs[0]
        
        # Parse chunk sizes from [size1,size2,...] format
        if not (chunk_sizes_str.startswith('[') and chunk_sizes_str.endswith(']')):
            raise PragmaError(f"Enhanced BDIM pragma chunk_sizes must be in [size1,size2,...] format, got: {chunk_sizes_str}")
        
        sizes_str = chunk_sizes_str[1:-1].strip()
        if not sizes_str:
            raise PragmaError("Enhanced BDIM pragma chunk_sizes cannot be empty")
        
        # Parse chunk sizes - allow ':' (full dimension), parameter names, or numeric values
        chunk_sizes = []
        for s in sizes_str.split(','):
            s = s.strip()
            if s == ':':
                # Full dimension reference
                chunk_sizes.append(':')
            elif s.isidentifier():
                # Module parameter reference (must be valid identifier)
                chunk_sizes.append(s)
            elif s.isdigit():
                # Numeric value (parameter value or constant)
                chunk_sizes.append(s)
            else:
                raise PragmaError(f"Enhanced BDIM pragma chunk_sizes must be ':' (full dimension), parameter names, or numeric values, got: '{s}'.")
        
        return {
            "format": "enhanced",
            "interface_name": interface_name,
            "chunk_index": chunk_index,
            "chunk_sizes": chunk_sizes,
            "chunking_strategy_type": "index"  # Enhanced format always uses index-based chunking
        }
    
    def _parse_legacy_format(self, interface_name: str, dimension_expressions: List[str]) -> Dict:
        """Parse legacy BDIM format with dimension expressions."""
        # Basic validation of dimension expressions (more detailed validation happens during evaluation)
        for i, expr in enumerate(dimension_expressions):
            if not expr.strip():
                raise PragmaError(f"BDIM pragma dimension expression {i+1} is empty")
        
        return {
            "format": "legacy",
            "interface_name": interface_name,
            "dimension_expressions": dimension_expressions
        }
    
    def _evaluate_expression(self, expression: str, parameters: Dict[str, Any]) -> int:
        """
        Safely evaluate a dimension expression using module parameters.
        
        Args:
            expression: Mathematical expression string (e.g., "PE*CHANNELS", "BATCH_SIZE+1")
            parameters: Dictionary of parameter names to values
            
        Returns:
            Evaluated integer result
            
        Raises:
            PragmaError: If expression cannot be evaluated or result is invalid
        """
        try:
            # Create a safe evaluation context with only parameters and basic operations
            safe_dict = {"__builtins__": {}}
            safe_dict.update(parameters)
            
            # Add common mathematical functions if needed
            import math
            safe_dict.update({
                'abs': abs, 'min': min, 'max': max,
                'pow': pow, 'round': round,
                'ceil': math.ceil, 'floor': math.floor
            })
            
            result = eval(expression, safe_dict)
            
            # Validate result
            if not isinstance(result, (int, float)):
                raise PragmaError(f"BDIM expression '{expression}' must evaluate to a number, got {type(result)}")
            
            result = int(result)
            if result <= 0:
                raise PragmaError(f"BDIM expression '{expression}' must evaluate to a positive integer, got {result}")
                
            return result
            
        except NameError as e:
            raise PragmaError(f"BDIM expression '{expression}' references undefined parameter: {e}")
        except (SyntaxError, ValueError) as e:
            raise PragmaError(f"BDIM expression '{expression}' has invalid syntax: {e}")
        except Exception as e:
            raise PragmaError(f"BDIM expression '{expression}' evaluation failed: {e}")


    def applies_to_interface(self, interface: 'Interface') -> bool:
        """Check if this BDIM pragma applies to the given interface."""
        if not self.parsed_data:
            return False
        
        pragma_interface_name = self.parsed_data.get('interface_name')
        if not pragma_interface_name:
            return False
        
        return self._interface_names_match(pragma_interface_name, interface.name)

    def apply_to_interface_metadata(self, interface: 'Interface', 
                                  metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply BDIM pragma to modify chunking strategy."""
        if not self.applies_to_interface(interface):
            return metadata
        
        # Create chunking strategy from pragma data
        new_strategy = self._create_chunking_strategy()
        
        return InterfaceMetadata(
            name=metadata.name,
            interface_type=metadata.interface_type,
            allowed_datatypes=metadata.allowed_datatypes,
            chunking_strategy=new_strategy
        )

    def _create_chunking_strategy(self):
        """Create chunking strategy from pragma data."""
        if not self.parsed_data:
            from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy
            return DefaultChunkingStrategy()
        
        pragma_format = self.parsed_data.get("format", "legacy")
        
        if pragma_format == "enhanced":
            # Enhanced format: create index-based chunking strategy
            try:
                from brainsmith.dataflow.core.block_chunking import IndexBasedChunkingStrategy
                chunk_index = self.parsed_data.get("chunk_index", -1)
                chunk_sizes = self.parsed_data.get("chunk_sizes", [":"]) 
                
                return IndexBasedChunkingStrategy(
                    start_index=chunk_index,
                    shape=chunk_sizes
                )
            except ImportError:
                # Fallback to DefaultChunkingStrategy if IndexChunkingStrategy not available
                from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy
                logger.warning(f"IndexBasedChunkingStrategy not available for BDIM pragma at line {self.line_number}. Using DefaultChunkingStrategy.")
                return DefaultChunkingStrategy()
        else:
            # Legacy format: use dimension expressions
            try:
                from brainsmith.dataflow.core.block_chunking import ExpressionChunkingStrategy
                dimension_expressions = self.parsed_data.get("dimension_expressions", [])
                
                return ExpressionChunkingStrategy(
                    dimension_expressions=dimension_expressions
                )
            except ImportError:
                # Fallback to DefaultChunkingStrategy if ExpressionChunkingStrategy not available
                from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy
                logger.warning(f"ExpressionChunkingStrategy not available for BDIM pragma at line {self.line_number}. Using DefaultChunkingStrategy.")
                return DefaultChunkingStrategy()


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
    


    def applies_to_interface(self, interface: 'Interface') -> bool:
        """Check if this WEIGHT pragma applies to the given interface."""
        if not self.parsed_data:
            return False
        
        interface_names = self.parsed_data.get('interface_names', [])
        for pragma_name in interface_names:
            if self._interface_names_match(pragma_name, interface.name):
                return True
        return False

    def apply_to_interface_metadata(self, interface: 'Interface', 
                                  metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply WEIGHT pragma to mark interface as weight type."""
        if not self.applies_to_interface(interface):
            return metadata
        
        return InterfaceMetadata(
            name=metadata.name,
            interface_type=InterfaceType.WEIGHT,  # Override type
            allowed_datatypes=metadata.allowed_datatypes,
            chunking_strategy=metadata.chunking_strategy
        )


# Note: Helper methods for Interface class would be added here if needed
