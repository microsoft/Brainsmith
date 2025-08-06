"""
Enhanced template context for Phase 2 RTL to AutoHWCustomOp generation.

This module defines the TemplateContext dataclass that carries all information
needed to generate AutoHWCustomOp subclass code from RTL modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
from pathlib import Path

from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.types.metadata import InterfaceMetadata, DatatypeMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Parameter

if TYPE_CHECKING:
    from ..codegen_binding import CodegenBinding
    from brainsmith.core.dataflow.relationships import DimensionRelationship


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
    parameter_definitions: List[Parameter]  # All module parameters (unified Parameter type)
    exposed_parameters: List[str] = field(default_factory=list)  # Parameters that should be nodeattr
    required_attributes: List[str] = field(default_factory=list)  # Parameters without defaults
    
    # Categorized interfaces for template compatibility
    input_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    output_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    weight_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    config_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    control_interfaces: List[InterfaceMetadata] = field(default_factory=list)
    
    
    # Additional context
    parallelism_info: Dict[str, Any] = field(default_factory=dict)
    
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
    
    
    
    # Unified CodegenBinding
    codegen_binding: Optional['CodegenBinding'] = None
    
    # Relationships between interfaces
    relationships: List['DimensionRelationship'] = field(default_factory=list)
    
    # SHAPE parameters for HWCustomOp node attributes
    shape_nodeattrs: List[Dict[str, str]] = field(default_factory=list)
    
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
            
        
        # Validate required attributes match parameters without defaults
        expected_required = {p.name for p in self.parameter_definitions if p.default_value is None}
        actual_required = set(self.required_attributes)
        if expected_required != actual_required:
            errors.append(f"Required attributes mismatch. Expected: {expected_required}, Got: {actual_required}")
        
        return errors