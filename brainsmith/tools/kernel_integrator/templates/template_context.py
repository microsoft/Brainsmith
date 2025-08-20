"""
Enhanced template context for Phase 2 RTL to AutoHWCustomOp generation.

This module defines the TemplateContext dataclass that carries all information
needed to generate AutoHWCustomOp subclass code from RTL modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
from pathlib import Path

from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.types.metadata import InterfaceMetadata, KernelMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Parameter

if TYPE_CHECKING:
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
    
    # Core reference to KernelMetadata
    kernel_metadata: KernelMetadata        # Source of truth for all kernel data
    
    # Active transformations still used by templates
    datatype_linked_params: List[str] = field(default_factory=list)  # Used by _categorize_parameters()
    categorized_parameters: Dict[str, Any] = field(default_factory=dict)  # Used by rtl_wrapper_minimal.v.j2
    shape_nodeattrs: List[Dict[str, str]] = field(default_factory=list)  # Used by hw_custom_op.py.j2
    
    # Simple field properties that delegate to KernelMetadata
    @property
    def module_name(self) -> str:
        """Module name from KernelMetadata."""
        return self.kernel_metadata.name
    
    @property
    def class_name(self) -> str:
        """PascalCase class name from KernelMetadata."""
        return self.kernel_metadata.class_name
    
    @property
    def source_file(self) -> Path:
        """Source file path from KernelMetadata."""
        return Path(self.kernel_metadata.source_file)
    
    @property
    def required_attributes(self) -> List[str]:
        """Required parameter names from KernelMetadata."""
        return self.kernel_metadata.required_attributes
    
    @property
    def relationships(self) -> List:
        """Relationships from KernelMetadata."""
        return self.kernel_metadata.relationships if hasattr(self.kernel_metadata, 'relationships') else []
    
    @property
    def internal_datatypes(self) -> List['DatatypeMetadata']:
        """Internal datatypes from KernelMetadata."""
        return self.kernel_metadata.internal_datatypes
    
    @property
    def linked_parameters(self) -> Dict[str, Any]:
        """Linked parameters from KernelMetadata."""
        return self.kernel_metadata.linked_parameters
    
    # Interface properties that delegate to KernelMetadata
    @property
    def interface_metadata(self) -> List[InterfaceMetadata]:
        """All interfaces from KernelMetadata."""
        return self.kernel_metadata.interfaces
    
    @property
    def input_interfaces(self) -> List[InterfaceMetadata]:
        """Input interfaces from KernelMetadata."""
        return self.kernel_metadata.input_interfaces
    
    @property
    def output_interfaces(self) -> List[InterfaceMetadata]:
        """Output interfaces from KernelMetadata."""
        return self.kernel_metadata.output_interfaces
    
    @property
    def weight_interfaces(self) -> List[InterfaceMetadata]:
        """Weight interfaces from KernelMetadata."""
        return self.kernel_metadata.weight_interfaces
    
    @property
    def config_interfaces(self) -> List[InterfaceMetadata]:
        """Config interfaces from KernelMetadata."""
        return self.kernel_metadata.config_interfaces
    
    @property
    def control_interfaces(self) -> List[InterfaceMetadata]:
        """Control interfaces from KernelMetadata."""
        return self.kernel_metadata.control_interfaces
    
    # Parameter properties that delegate to KernelMetadata
    @property
    def parameter_definitions(self) -> List[Parameter]:
        """All parameters from KernelMetadata."""
        return self.kernel_metadata.parameters
    
    @property
    def exposed_parameters(self) -> List[str]:
        """Exposed parameters from KernelMetadata."""
        return self.kernel_metadata.exposed_parameters
    
    # Convenience properties that delegate to KernelMetadata
    @property
    def has_inputs(self) -> bool:
        """Check if kernel has input interfaces."""
        return self.kernel_metadata.has_inputs
    
    @property
    def has_outputs(self) -> bool:
        """Check if kernel has output interfaces."""
        return self.kernel_metadata.has_outputs
    
    @property
    def has_weights(self) -> bool:
        """Check if kernel has weight interfaces."""
        return self.kernel_metadata.has_weights
    
    @property
    def has_config(self) -> bool:
        """Check if kernel has config interfaces."""
        return self.kernel_metadata.has_config
    
    def validate(self) -> List[str]:
        """
        Validate the template context for consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate KernelMetadata reference
        if not self.kernel_metadata:
            errors.append("KernelMetadata reference is required")
            return errors  # Can't validate further without kernel_metadata
        
        # Delegate most validation to KernelMetadata
        kernel_errors = self.kernel_metadata.validate()
        if kernel_errors:
            errors.extend(kernel_errors)
        
        # Template-specific validations can be added here if needed
        
        return errors