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
import logging # Added import

# Set up logger for this module - typically at the module level
logger = logging.getLogger(__name__)

class PragmaError(Exception): # Added PragmaError definition
    """Custom exception for errors during pragma parsing or validation."""
    pass

# --- Enums ---

class Direction(Enum):
    """Port direction enumeration."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"

class InterfaceType(Enum):
    """Enumeration of supported interface types."""
    GLOBAL_CONTROL = "global"
    AXI_STREAM = "axistream"
    AXI_LITE = "axilite"
    UNKNOWN = "unknown" # For ports not part of a recognized interface

class PragmaType(Enum):
    """Valid pragma types recognized by the parser."""
    TOP_MODULE = "top_module"          # Specify the top module if multiple exist
    DATATYPE = "datatype"              # Restrict datatype for an interface
    DERIVED_PARAMETER = "derived_parameter" # Link module param to python function
    WEIGHT = "weight"                  # Specify interface as a weight

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
    param_type: Optional[str] = None # Changed from str to Optional[str]
    default_value: Optional[str] = None
    description: Optional[str] = None
    template_param_name: str = field(init=False) # Added field

    def __post_init__(self):
        """Validate parameter attributes after initialization."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid parameter name: {self.name}")
        # Parameter type can be None for typeless parameters
        self.template_param_name = f"${self.name.upper()}$" # Initialize template name

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
        Currently overrides existing keys without warning.
        """
        if key is None:
            key = port.name
    # if key in self.ports:
        # logger.warning(f"Overwriting port key '{key}' in PortGroup '{self.name}'")
        self.ports[key] = port

# --- Validated/Complex Structures ---

@dataclass
class Interface:
    """Represents a fully validated and identified interface.

    Created by the InterfaceBuilder after a PortGroup successfully passes
    validation by the ProtocolValidator.
    """
    name: str # e.g., "global", "in0", "config"
    type: InterfaceType
    ports: Dict[str, Port] # Maps signal suffix/name to Port object
    validation_result: ValidationResult
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., data width, address width
    wrapper_name: Optional[str] = None  # New attribute to store wrapper name

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

    def apply(self, **kwargs) -> Any: # Changed return type to Any
        """
        Abstract method to apply the pragma's effects.
        Subclasses must implement this method and can return any relevant data.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments. Subclasses will expect specific
                      keys like 'interfaces', 'parameters', 'hw_kernel'.
        """
        raise NotImplementedError(f"Pragma type {self.type.name} must implement apply.")

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

    def apply(self, **kwargs) -> Any: # Changed return type to Any
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
class DatatypePragma(Pragma):
    def __post_init__(self):
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """Handles DATATYPE pragma: @brainsmith datatype <if_name> <size> OR <if_name> <min> <max>"""
        logger.debug(f"Parsing DATATYPE pragma: {self.inputs} at line {self.line_number}")

        if len(self.inputs) == 2:
            interface_name = self.inputs[0]
            size = self.inputs[1]
            # TODO: Validate size format (e.g., ensure it's numeric or a valid type string)
            return {
                "interface_name": interface_name,
                "min_size": size,
                "max_size": size,
                "is_fixed_size": True
            }
        elif len(self.inputs) == 3:
            interface_name = self.inputs[0]
            min_size = self.inputs[1]
            max_size = self.inputs[2]
            # TODO: Validate size formats
            # TODO: Validate min_size <= max_size (if numeric)
            return {
                "interface_name": interface_name,
                "min_size": min_size,
                "max_size": max_size,
                "is_fixed_size": False
            }
        else:
            raise PragmaError("DATATYPE pragma requires <interface_name> <size> OR <interface_name> <min_size> <max_size>")

    def apply(self, **kwargs) -> Any: # Changed return type to Any
        """Applies the DATATYPE pragma to the specified interface."""
        interfaces: Optional[Dict[str, Interface]] = kwargs.get('interfaces')

        if not self.parsed_data:
            logger.warning(f"DATATYPE pragma at line {self.line_number} has no parsed_data. Skipping application.")
            return

        if interfaces is None:
            logger.warning(f"DATATYPE pragma at line {self.line_number} requires 'interfaces' keyword argument to apply. Skipping.")
            return

        interface_name = self.parsed_data.get("interface_name")
        min_size = self.parsed_data.get("min_size")
        max_size = self.parsed_data.get("max_size")
        is_fixed_size = self.parsed_data.get("is_fixed_size")

        if not interface_name:
            logger.warning(f"DATATYPE pragma at line {self.line_number} missing 'interface_name' in parsed_data. Skipping.")
            return

        applied_to_interface = False
        for iface_key, iface in interfaces.items():
            if iface.name == interface_name or iface.name.startswith(interface_name):
                iface.metadata["datatype_min_size"] = min_size
                iface.metadata["datatype_max_size"] = max_size
                iface.metadata["datatype_is_fixed"] = is_fixed_size
                
                datatype_str = f"{min_size}" if is_fixed_size else f"{min_size}..{max_size}"
                iface.metadata["datatype_raw_str"] = datatype_str
                
                logger.info(f"Applied DATATYPE pragma from line {self.line_number} to interface '{iface.name}'. Datatype set to: {datatype_str}")
                applied_to_interface = True
        
        if not applied_to_interface:
            logger.warning(f"DATATYPE pragma from line {self.line_number} for interface '{interface_name}' did not match any existing interfaces.")


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
class WeightPragma(Pragma):
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
    

    def apply(self, **kwargs) -> Any: # Changed return type to Any
        """Applies the WEIGHT pragma to the specified interface."""
        interfaces: Optional[Dict[str, Interface]] = kwargs.get('interfaces')

        if not self.parsed_data:
            logger.warning(f"WEIGHT pragma at line {self.line_number} has no parsed_data. Skipping application.")
            return

        if interfaces is None:
            logger.warning(f"WEIGHT pragma at line {self.line_number} requires 'interfaces' keyword argument to apply. Skipping.")
            return

        interface_name = self.parsed_data.get("interface_name")
        type_name = self.parsed_data.get("type_name")
        depth = self.parsed_data.get("depth")

        if not interface_name: # type_name and depth could be empty strings if allowed, but interface_name is crucial
            logger.warning(f"WEIGHT pragma at line {self.line_number} missing 'interface_name' in parsed_data. Skipping.")
            return
            
        applied_to_interface = False
        for iface_key, iface in interfaces.items():
            # Match if the interface name is exactly the one specified,
            # or if the pragma specifies a base name and the interface is e.g. iface_name_0, iface_name_1 etc.
            # Current InterfaceBuilder names are exact like "in0", "s_axi_control".
            # So, exact match should be sufficient for now.
            if iface.name == interface_name: # Consider iface.name.startswith(interface_name) if needed
                iface.metadata["is_weight"] = True
                iface.metadata["weight_type"] = type_name
                iface.metadata["weight_depth"] = depth
                logger.info(f"Applied WEIGHT pragma from line {self.line_number} to interface '{iface.name}'. Marked as weight, type='{type_name}', depth='{depth}'.")
                applied_to_interface = True
                # break # Assuming interface names are unique and we only apply to the first match.
        
        if not applied_to_interface:
            logger.warning(f"WEIGHT pragma from line {self.line_number} for interface '{interface_name}' did not match any existing interfaces.")

# --- Top-Level Structure ---

@dataclass
class HWKernel:
    """Top-level representation of a parsed hardware kernel.
    
    This structure holds the consolidated information extracted from an RTL file,
    focusing on a single target module (often specified by a pragma).
    
    Attributes:
        name: Kernel (module) name
        parameters: List of parameters
        interfaces: Dictionary of detected interfaces (e.g., AXI-Lite, AXI-Stream)
        pragmas: List of Brainsmith pragmas found
        metadata: Optional dictionary for additional info (e.g., source file)
    """
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    interfaces: Dict[str, Interface] = field(default_factory=dict)
    pragmas: List[Pragma] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing for HWKernel."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid kernel name: {self.name}")
        # Additional validation or processing can be added here if needed
