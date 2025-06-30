"""
Enhanced template context for Phase 2 RTL to AutoHWCustomOp generation.

This module defines the TemplateContext dataclass that carries all information
needed to generate AutoHWCustomOp subclass code from RTL modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
from pathlib import Path

from ..data import InterfaceType
from ..metadata import InterfaceMetadata, DatatypeMetadata
from ..rtl_parser.rtl_data import Parameter

if TYPE_CHECKING:
    from ..codegen_binding import CodegenBinding
    from brainsmith.core.dataflow.relationships import DimensionRelationship


@dataclass
class ParameterDefinition:
    """Enhanced parameter definition for template generation."""
    name: str                              # Parameter name (e.g., "PE", "SIMD")
    param_type: Optional[str] = None       # Type hint (e.g., "int", "logic")
    default_value: Optional[int] = None    # Default value from RTL (if whitelisted)
    description: Optional[str] = None      # Description from comments
    line_number: int = 0                   # Line number in RTL for error reporting
    template_param_name: Optional[str] = None  # Template placeholder (e.g., "$PE$")
    is_whitelisted: bool = False          # Whether parameter can have defaults
    is_required: bool = True              # Whether FINN must provide value


@dataclass
class TemplateContext:
    """
    Enhanced template context for generating AutoHWCustomOp subclasses.
    
    This context includes all information needed to generate code that:
    1. Extracts runtime parameters from ONNX nodes
    2. Defines static interface metadata with validated symbolic BDIM
    3. Creates proper node attribute definitions
    4. Integrates seamlessly with FINN
    """
    
    # Core module information
    module_name: str                       # RTL module name
    class_name: str                        # Generated Python class name
    source_file: Path                      # Source RTL file path
    
    # Interface metadata with validated symbolic BDIM shapes
    interface_metadata: List[InterfaceMetadata]  # All interfaces with chunking
    
    # Enhanced parameter definitions
    parameter_definitions: List[ParameterDefinition]  # All module parameters
    exposed_parameters: List[str] = field(default_factory=list)  # Parameters that should be nodeattr
    whitelisted_defaults: Dict[str, int] = field(default_factory=dict)  # Default values for whitelisted params
    required_attributes: List[str] = field(default_factory=list)  # Parameters without defaults
    
    # Categorized interfaces for template compatibility
    input_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    output_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    weight_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    config_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    control_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    
    # Template generation helpers
    base_imports: List[str] = field(default_factory=lambda: [
        "from typing import List, Dict, Tuple, Any",
        "import numpy as np",
        "from qonnx.core.datatype import DataType",
        "from brainsmith.dataflow.core import AutoHWCustomOp",
        "from brainsmith.tools.hw_kernel_gen.data import InterfaceMetadata, InterfaceType",
        "from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy"
    ])
    
    # Additional context from existing template system
    parallelism_info: Dict[str, Any] = field(default_factory=dict)
    algorithm_info: Dict[str, Any] = field(default_factory=dict)
    node_attributes: Dict[str, Any] = field(default_factory=dict)
    # Note: datatype_mappings, shape_calculation_methods, stream_width_methods removed
    # These are now handled by AutoHWCustomOp parent class automatically
    resource_estimation_methods: Dict[str, Any] = field(default_factory=dict)
    
    # Datatype parameter information for new architecture
    datatype_linked_params: List[str] = field(default_factory=list)
    datatype_param_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)  
    interface_datatype_attributes: List[Dict[str, Any]] = field(default_factory=list)
    datatype_derivation_methods: Dict[str, str] = field(default_factory=dict)
    
    # Internal datatype metadata from kernel
    internal_datatypes: List['DatatypeMetadata'] = field(default_factory=list)
    
    # Template-time parameter assignments (replaces runtime logic)
    datatype_parameter_assignments: List[Dict[str, str]] = field(default_factory=list)
    
    # Linked parameter data (ALIAS, DERIVED_PARAMETER, AXILITE_PARAM)
    linked_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Categorized parameters for organized RTL wrapper generation
    categorized_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Template flags
    has_inputs: bool = False
    has_outputs: bool = False
    has_weights: bool = False
    has_config: bool = False
    
    # Kernel analysis
    kernel_complexity: str = "medium"
    kernel_type: str = "generic"
    
    # Unified CodegenBinding
    codegen_binding: Optional['CodegenBinding'] = None
    
    # Relationships between interfaces
    relationships: List['DimensionRelationship'] = field(default_factory=list)
    
    def get_node_attribute_definitions(self) -> Dict[str, Tuple[str, bool, Any]]:
        """
        Generate FINN node attribute definitions for all parameters.
        
        Returns:
            Dict mapping attribute names to (type, required, default) tuples
        """
        attrs = {}
        
        # Add all RTL parameters as node attributes
        for param in self.parameter_definitions:
            if param.is_whitelisted and param.default_value is not None:
                # Optional attribute with default
                attrs[param.name] = ("i", False, param.default_value)
            else:
                # Required attribute (no default)
                attrs[param.name] = ("i", True, None)
        
        # Add datatype attributes if interfaces exist
        if self.has_inputs:
            attrs["inputDataType"] = ("s", True, "")
        if self.has_outputs:
            attrs["outputDataType"] = ("s", True, "")
        if self.has_weights:
            attrs["weightDataType"] = ("s", True, "")
            
        # Add standard AutoHWCustomOp attributes
        attrs["runtime_writeable_weights"] = ("i", False, 0)
        attrs["numInputVectors"] = ("ints", False, [1])
        
        return attrs
    
    def get_runtime_parameter_extraction(self) -> List[str]:
        """
        Generate code lines for extracting runtime parameters from ONNX node.
        
        Returns:
            List of Python code lines for parameter extraction
        """
        lines = []
        lines.append("runtime_parameters = {}")
        
        for param in self.parameter_definitions:
            lines.append(f'runtime_parameters["{param.name}"] = self.get_nodeattr("{param.name}")')
        
        return lines
    
    def get_interface_metadata_code(self) -> List[str]:
        """
        Generate code for the get_interface_metadata() method.
        
        Returns:
            List of Python code lines for interface metadata
        """
        lines = []
        lines.append("return [")
        
        for interface in self.interface_metadata:
            lines.append("    InterfaceMetadata(")
            lines.append(f'        name="{interface.name}",')
            lines.append(f'        interface_type=InterfaceType.{interface.interface_type.name},')
            
            # Allowed datatypes
            if interface.allowed_datatypes:
                dt_strs = []
                for dt in interface.allowed_datatypes:
                    dt_strs.append(f'DataTypeConstraint(finn_type="{dt.finn_type}", '
                                 f'bit_width={dt.bit_width}, signed={dt.signed})')
                lines.append(f'        allowed_datatypes=[{", ".join(dt_strs)}],')
            else:
                lines.append('        allowed_datatypes=[],')
            
            # Chunking strategy with validated symbolic BDIM
            if interface.chunking_strategy:
                cs = interface.chunking_strategy
                # Convert block_shape to proper representation
                shape_repr = repr(cs.block_shape)  # Handles both strings and integers
                lines.append(f'        chunking_strategy=BlockChunkingStrategy(')
                lines.append(f'            block_shape={shape_repr},')
                lines.append(f'            rindex={cs.rindex}')
                lines.append('        )')
            else:
                lines.append('        chunking_strategy=None')
                
            lines.append("    ),")
        
        lines.append("]")
        return lines
    
    def validate(self) -> List[str]:
        """
        Validate the template context for consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check module and class names
        if not self.module_name:
            errors.append("Module name is required")
        if not self.class_name:
            errors.append("Class name is required")
            
        # Check interface metadata
        if not self.interface_metadata:
            errors.append("At least one interface is required")
            
        # Validate parameter definitions
        param_names = set()
        for param in self.parameter_definitions:
            if param.name in param_names:
                errors.append(f"Duplicate parameter name: {param.name}")
            param_names.add(param.name)
            
            # Check whitelisted parameters have defaults
            if param.is_whitelisted and param.name not in self.whitelisted_defaults:
                errors.append(f"Whitelisted parameter {param.name} missing default value")
        
        # Validate required attributes match non-whitelisted parameters
        expected_required = {p.name for p in self.parameter_definitions if not p.is_whitelisted}
        actual_required = set(self.required_attributes)
        if expected_required != actual_required:
            errors.append(f"Required attributes mismatch. Expected: {expected_required}, Got: {actual_required}")
        
        return errors