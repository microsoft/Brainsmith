############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Hardware Kernel Generator Metadata Classes.

This module contains the metadata classes used to describe hardware kernels,
interfaces, and their relationships. These classes bridge the RTL parser
output with the template generation system.

Classes:
- DatatypeMetadata: RTL parameter to datatype property mappings
- InterfaceMetadata: Complete interface description with constraints
- KernelMetadata: Complete kernel description for code generation
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING

# Import shared types from main data module
from .data import InterfaceType, BaseDataType
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup, validate_datatype_against_constraints

if TYPE_CHECKING:
    from .rtl_parser.rtl_data import Parameter
    from .rtl_parser.pragmas.base import Pragma

# Import relationship types that are used in pragma implementation
from brainsmith.core.dataflow.relationships import DimensionRelationship, RelationType

import logging

logger = logging.getLogger(__name__)


@dataclass
class DatatypeMetadata:
    """
    Explicit binding between RTL parameters and datatype properties.
    
    This class provides a structured way to map RTL parameter names to
    specific datatype properties used in code generation. All properties
    are optional RTL parameter names, allowing flexible datatype definitions.
    
    Attributes:
        name: Identifier for this datatype (e.g., "in", "accumulator", "threshold")
        width: RTL parameter name for bit width
        signed: RTL parameter name for signedness (0/1)
        format: RTL parameter name for format selection (INT/UINT/FIXED/FLOAT)
        bias: RTL parameter name for bias/offset value
        fractional_width: RTL parameter name for fractional bits (FIXED types)
        exponent_width: RTL parameter name for exponent bits (FLOAT types)
        mantissa_width: RTL parameter name for mantissa bits (FLOAT types)
        description: Optional human-readable description
    """
    name: str  # Required - identifier for this datatype
    width: Optional[str] = None
    signed: Optional[str] = None
    format: Optional[str] = None
    bias: Optional[str] = None
    fractional_width: Optional[str] = None
    exponent_width: Optional[str] = None
    mantissa_width: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate metadata parameters."""
        if not self.name:
            raise ValueError("DatatypeMetadata name cannot be empty")
    
    def get_all_parameters(self) -> list[str]:
        """
        Get list of all RTL parameter names referenced by this metadata.
        
        Returns:
            List of parameter names (non-None values only)
        """
        params = []
        
        # Add all non-None parameters
        if self.width is not None:
            params.append(self.width)
        if self.signed is not None:
            params.append(self.signed)
        if self.format is not None:
            params.append(self.format)
        if self.bias is not None:
            params.append(self.bias)
        if self.fractional_width is not None:
            params.append(self.fractional_width)
        if self.exponent_width is not None:
            params.append(self.exponent_width)
        if self.mantissa_width is not None:
            params.append(self.mantissa_width)
            
        return params
    
    def update(self, **kwargs) -> 'DatatypeMetadata':
        """
        Create a new DatatypeMetadata with updated fields.
        
        Args:
            **kwargs: Field values to update
            
        Returns:
            New DatatypeMetadata instance with updated fields
        """
        # Get current values
        current = {
            'name': self.name,
            'width': self.width,
            'signed': self.signed,
            'format': self.format,
            'bias': self.bias,
            'fractional_width': self.fractional_width,
            'exponent_width': self.exponent_width,
            'mantissa_width': self.mantissa_width,
            'description': self.description
        }
        
        # Update with provided values
        current.update(kwargs)
        
        return DatatypeMetadata(**current)


@dataclass
class InterfaceMetadata:
    """
    Interface metadata with QONNX constraint groups - no default datatypes.
    
    This class encapsulates all information needed to create and configure
    a dataflow interface with runtime datatype validation.
    """
    name: str
    interface_type: InterfaceType
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    description: Optional[str] = None
    
    # Parameter linkage mappings
    datatype_metadata: Optional[DatatypeMetadata] = None
    """
    Optional datatype metadata binding RTL parameters to QONNX datatype properties.
    If None, defaults will be created based on interface name patterns.
    
    Example: DatatypeMetadata(name="in0", datatype_params={"width": "INPUT0_WIDTH", "signed": "SIGNED_INPUT0"})
    """
    
    bdim_params: Optional[List[str]] = None
    """
    RTL parameter names that define the block dimensions.
    These are the actual parameters used in the RTL (e.g., TILE_H, TILE_W).
    Always stored as a list, even for single parameters.
    
    Example: ["TILE_SIZE"] or ["TILE_H", "TILE_W", "TILE_C"]
    """
    
    sdim_params: Optional[List[str]] = None
    """
    RTL parameter names for stream dimensions.
    Always stored as a list, even for single parameters.
    
    Example: ["STREAM_SIZE"] or ["SDIM_D0", "SDIM_D1", "SDIM_D2"]
    """
    
    
    
    bdim_shape: Optional[List[Union[str, int]]] = None
    """
    Shape expressions for block dimensions in TilingSpec format.
    Elements can be:
    - 1: Singleton dimension
    - ":": Full slice dimension  
    - "param_name": Parameter alias for node attributes
    
    Example: [1, "TILE_SIZE", ":"] means [singleton, parameter, full_slice]
    """
    
    sdim_shape: Optional[List[Union[str, int]]] = None
    """
    Shape expressions for stream dimensions in TilingSpec format.
    Elements can be:
    - 1: Singleton dimension
    - ":": Full slice dimension (unusual for SDIM)
    - "param_name": Parameter alias for node attributes
    
    Example: [1, "SIMD", 1] means [singleton, parameter, singleton]
    """
    
    
    def __post_init__(self):
        """Validate metadata parameters."""
        if not self.name:
            raise ValueError("Interface name cannot be empty")
        
        # Allow empty datatype_constraints - these can be populated by pragmas
    
    
    def validates_datatype(self, datatype: BaseDataType) -> bool:
        """
        Check if a QONNX datatype satisfies constraint groups.
        
        Args:
            datatype: QONNX BaseDataType instance to validate
            
        Returns:
            bool: True if datatype satisfies at least one constraint group
        """
        if not self.datatype_constraints:
            return True  # No constraints = allow anything
        return validate_datatype_against_constraints(datatype, self.datatype_constraints)
    
    def validate_parameters(self, module_param_names: set) -> List[str]:
        """
        Validate that expected parameters exist in the module.
        
        This method checks if the interface's expected parameters (BDIM, SDIM, 
        datatype parameters) actually exist in the RTL module's parameter list.
        
        Args:
            module_param_names: Set of parameter names from the RTL module
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Skip non-dataflow interfaces
        if self.interface_type not in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
            return errors
        
        # CRITICAL: INPUT and WEIGHT interfaces MUST have BDIM, OUTPUT interfaces are optional
        has_bdim = bool(self.bdim_params)
        if not has_bdim and self.interface_type.value in ['input', 'weight']:
            errors.append(
                f"AXI Stream interface '{self.name}' ({self.interface_type.value.upper()}) "
                f"is missing required BDIM parameter. Either add a parameter named "
                f"'{self.name}_BDIM' or use '@brainsmith BDIM {self.name} <param_name>' pragma."
            )
        elif self.bdim_params:
            # Check BDIM parameters exist (skip '1' singletons)
            missing_params = [p for p in self.bdim_params if p != "1" and p not in module_param_names]
            if missing_params:
                errors.append(
                    f"Interface '{self.name}' references BDIM parameters {missing_params} "
                    f"which do not exist in the module."
                )
        
        # CRITICAL: INPUT and WEIGHT interfaces MUST have SDIM
        if self.interface_type.value in ['input', 'weight']:
            has_sdim = bool(self.sdim_params)
            if not has_sdim:
                errors.append(
                    f"Interface '{self.name}' ({self.interface_type.value.upper()}) "
                    f"is missing required SDIM parameter. Either add a parameter named "
                    f"'{self.name}_SDIM' or use '@brainsmith SDIM {self.name} <param_name>' pragma."
                )
            elif self.sdim_params:
                # Check SDIM parameters exist (skip '1' singletons)
                missing_params = [p for p in self.sdim_params if p != "1" and p not in module_param_names]
                if missing_params:
                    errors.append(
                        f"Interface '{self.name}' references SDIM parameters {missing_params} "
                        f"which do not exist in the module."
                    )
        
        # Check datatype parameters - validate DatatypeMetadata exists and has width
        if not self.datatype_metadata:
            # No datatype metadata at all - this is a warning for dataflow interfaces
            errors.append(
                f"Interface '{self.name}' has no datatype metadata. "
                f"Use @brainsmith DATATYPE_PARAM pragma to specify parameter mappings."
            )
        elif self.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
            # For streaming interfaces, width is required
            if not self.datatype_metadata.width:
                errors.append(
                    f"Interface '{self.name}' datatype has no width parameter specified. "
                    f"Streaming interfaces require a width parameter for data formatting."
                )
        
        
        return errors
    
    def get_constraint_description(self) -> str:
        """
        Get human-readable description of constraints.
        
        Returns:
            str: Description like "UINT8-16, INT4-8" or "No datatype constraints"
        """
        if not self.datatype_constraints:
            return "No datatype constraints"
        
        descriptions = []
        for group in self.datatype_constraints:
            if group.min_width == group.max_width:
                descriptions.append(f"{group.base_type}{group.min_width}")
            else:
                descriptions.append(f"{group.base_type}{group.min_width}-{group.max_width}")
        return ", ".join(descriptions)
    
    def get_datatype_parameter_name(self, property_type: str) -> str:
        """
        Get RTL parameter name for a datatype property.
        
        Args:
            property_type: 'width', 'signed', 'format', 'bias', 'fractional_width'
            
        Returns:
            RTL parameter name (e.g., 'potato_WIDTH', 'SIGNED_potato')
        """
        if self.datatype_metadata:
            # Get from explicit attributes
            if property_type == 'width':
                return self.datatype_metadata.width
            elif property_type == 'signed' and self.datatype_metadata.signed:
                return self.datatype_metadata.signed
            elif property_type == 'format' and self.datatype_metadata.format:
                return self.datatype_metadata.format
            elif property_type == 'bias' and self.datatype_metadata.bias:
                return self.datatype_metadata.bias
            elif property_type == 'fractional_width' and self.datatype_metadata.fractional_width:
                return self.datatype_metadata.fractional_width
            elif property_type == 'exponent_width' and self.datatype_metadata.exponent_width:
                return self.datatype_metadata.exponent_width
            elif property_type == 'mantissa_width' and self.datatype_metadata.mantissa_width:
                return self.datatype_metadata.mantissa_width
        
        # Default naming convention: use consistent suffix pattern
        if property_type == 'width':
            return f"{self.name}_WIDTH"
        elif property_type == 'signed':
            return f"{self.name}_SIGNED"  # Changed from SIGNED_{self.name} for consistency
        elif property_type == 'format':
            return f"{self.name}_FORMAT"
        elif property_type == 'bias':
            return f"{self.name}_BIAS"
        elif property_type == 'fractional_width':
            return f"{self.name}_FRACTIONAL_WIDTH"
        elif property_type == 'exponent_width':
            return f"{self.name}_EXPONENT_WIDTH"
        elif property_type == 'mantissa_width':
            return f"{self.name}_MANTISSA_WIDTH"
        else:
            return f"{self.name}_{property_type.upper()}"
    
    def get_bdim_parameter_name(self) -> str:
        """
        Get RTL parameter name for block dimensions.
        For backward compatibility, returns the first parameter if multiple exist.
        
        Returns:
            RTL parameter name (e.g., 'potato_BDIM', 'weights_V_BDIM')
        """
        if self.bdim_params and self.bdim_params[0] != '1':
            return self.bdim_params[0]
        
        # Default naming convention: use actual interface name directly
        return f"{self.name}_BDIM"
    
    def get_sdim_parameter_name(self) -> str:
        """
        Get RTL parameter name for stream dimensions.
        For backward compatibility, returns the first parameter if multiple exist.
        
        Returns:
            RTL parameter name (e.g., 'potato_SDIM', 'weights_V_SDIM')
        """
        if self.sdim_params and self.sdim_params[0] != '1':
            return self.sdim_params[0]
        
        # Default naming convention: use actual interface name directly
        return f"{self.name}_SDIM"
    
    def has_shape_linkage(self) -> bool:
        """
        Check if interface has complete shape/size parameter linkage.
        
        Returns:
            bool: True if both BDIM and SDIM parameters are available
        """
        return bool(self.bdim_params) and bool(self.sdim_params)


@dataclass
class KernelMetadata:
    """
    Complete metadata for AutoHWCustomOp generation from RTL.
    
    This unified structure contains all information needed to generate
    an AutoHWCustomOp subclass, with the RTL parser creating InterfaceMetadata
    directly instead of requiring post-processing transformation.
    """
    
    # === Core Identification ===
    
    name: str
    """
    The RTL module name (e.g., 'thresholding_axi').
    
    Reason: Essential identifier for the kernel.
    Use: 
    - Generate class name (ThresholdingAxi)
    - Set kernel_name attribute in generated class
    - Reference in error messages and logging
    """
    
    source_file: Path
    """
    Absolute path to the source RTL file.
    
    Reason: Traceability and debugging.
    Use:
    - Document generation source in comments
    - Error messages can reference source location
    - Future re-parsing or validation needs
    """
    
    # === Interface Definitions (The Key Innovation) ===
    
    interfaces: List[InterfaceMetadata]
    """
    List of InterfaceMetadata objects created directly by RTL parser.
    
    Reason: This is the KEY UNIFICATION - instead of creating intermediate
    Interface objects that need conversion, the RTL parser directly creates
    the InterfaceMetadata objects that AutoHWCustomOp needs.
    
    Use:
    - Passed directly to AutoHWCustomOp.__init__()
    - No transformation needed in templates
    - Contains interface types, datatype constraints, chunking strategies
    """
    
    # === RTL Parameters ===
    
    parameters: List['Parameter']
    """
    SystemVerilog parameters from the RTL module.
    
    Reason: RTL parameters represent configurable aspects of the hardware
    that may need to be exposed as FINN node attributes.
    
    Use:
    - Generate node attributes in get_nodeattr_types()
    - Provide default values for kernel configuration
    - Templates decide which parameters to expose
    """
    
    exposed_parameters: List[str]
    """
    Names of parameters that should be exposed as FINN node attributes.
    
    Reason: After pragma application, some parameters are linked to interfaces
    (BDIM, SDIM, datatype) and should not be exposed as nodeattr. This list
    contains only the remaining parameters that need explicit node attributes.
    
    Use:
    - Template generation for get_nodeattr_types()
    - Filter out parameters that are linked to interface metadata
    - Ensure only truly configurable parameters become nodeattr
    """
    
    # === Pragma Information ===
    
    pragmas: List['Pragma']
    """
    All @brainsmith pragmas found in the RTL source.
    
    Reason: Pragmas provide metadata not inferrable from RTL structure.
    
    Use:
    - Already processed into InterfaceMetadata for chunking/datatypes
    - Kept for completeness and future extensibility
    - May contain directives for advanced features
    """
    
    # === Parsing Diagnostics ===
    
    parsing_warnings: List[str] = field(default_factory=list)
    """
    Non-fatal warnings encountered during RTL parsing.
    
    Reason: RTL parsing may encounter ambiguities or make assumptions
    that should be communicated to developers.
    
    Use:
    - Include as comments in generated code
    - Log during generation process
    - Help developers understand parsing decisions
    """
    
    linked_parameters: Dict[str, Any] = field(default_factory=dict)
    """
    Linked parameter data from pragmas (ALIAS, DERIVED_PARAMETER, AXILITE_PARAM).
    
    Structure:
    {
        "aliases": {"rtl_param": "nodeattr_name", ...},
        "derived": {"param_name": "python_expression", ...},
        "axilite": {"param_name": "interface_name", ...}
    }
    
    Reason: Parameter pragmas affect how parameters are exposed, computed,
    and categorized in the generated code.
    
    Use:
    - ALIAS: Maps RTL parameter names to different nodeattr names
    - DERIVED: Parameters assigned via Python expressions instead of nodeattr
    - AXILITE: Links control parameters to their AXI-Lite interfaces
    - Template generation for prepare_codegen_rtl_values() and categorization
    """
    
    internal_datatypes: List[DatatypeMetadata] = field(default_factory=list)
    """
    Datatype metadata for internal kernel mechanisms (accumulator, threshold, etc.).
    
    These are created from DATATYPE_PARAM pragmas that don't match any interface name.
    They enable documentation and construction of internal computation datatypes.
    
    Example:
    - DatatypeMetadata(name="accumulator", datatype_params={"width": "ACC_WIDTH", "signed": "ACC_SIGNED"})
    - DatatypeMetadata(name="threshold", datatype_params={"width": "THRESH_WIDTH"})
    
    Use:
    - Documentation of internal precision requirements
    - Future datatype inference and validation
    - Template generation for internal datatype handling
    """
    
    relationships: List['DimensionRelationship'] = field(default_factory=list)
    """
    Interface relationships defined via RELATIONSHIP pragmas.
    
    These capture dependencies between interface dimensions for the Kernel Modeling system.
    
    Example:
    - EQUAL: All dimensions must match between interfaces
    - DEPENDENT: Specific dimension relationships (copy, scaled, min)
    - MULTIPLE/DIVISIBLE: Dimension constraints between interfaces
    
    Use:
    - Kernel modeling constraint validation
    - Automatic dimension inference
    - Documentation of interface dependencies
    """
    
    def get_class_name(self) -> str:
        """
        Generate Python class name from module name.
        
        Converts snake_case or kebab-case to PascalCase.
        Examples: 
        - thresholding_axi → ThresholdingAxi
        - matrix-multiply → MatrixMultiply
        """
        from .utils import pascal_case
        return pascal_case(self.name)
    
    def validate(self) -> List[str]:
        """
        Validate kernel metadata for consistency and completeness.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        warnings = []
        
        # Check required fields
        if not self.name:
            errors.append("Kernel name is required")
            
        if not self.interfaces:
            errors.append("At least one interface is required")
            
        # Check for at least one input and one output
        has_input = any(iface.interface_type.value == 'input' for iface in self.interfaces)
        has_output = any(iface.interface_type.value == 'output' for iface in self.interfaces)
        
        if not has_input:
            errors.append("Kernel must have at least one input interface")
        if not has_output:
            errors.append("Kernel must have at least one output interface")
        
        # Validate Global Control interface exists
        has_global_control = any(
            iface.interface_type.value == 'control' for iface in self.interfaces
        )
        if not has_global_control:
            errors.append(f"Module '{self.name}' is missing a valid Global Control interface (ap_clk, ap_rst_n)")
            
        # Validate interface names are unique
        interface_names = [iface.name for iface in self.interfaces]
        if len(interface_names) != len(set(interface_names)):
            errors.append("Duplicate interface names found")
            
        # Check that exposed parameters exist
        param_names = {p.name for p in self.parameters}
        for exp_param in self.exposed_parameters:
            if exp_param not in param_names:
                errors.append(f"Exposed parameter '{exp_param}' not found in parameters")
        
        # Validate interface parameters
        for interface in self.interfaces:
            interface_errors = interface.validate_parameters(param_names)
            errors.extend(interface_errors)
        
        return errors


# Module exports
__all__ = [
    "DatatypeMetadata",
    "InterfaceMetadata", 
    "KernelMetadata",
]