"""
Interface metadata classes for enhanced AutoHWCustomOp architecture.

This module provides the object-oriented metadata system that replaces
static dictionaries in generated AutoHWCustomOp classes.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging
from .interface_types import InterfaceType
from .qonnx_types import DatatypeConstraintGroup, validate_datatype_against_constraints, BaseDataType
from .datatype_metadata import DatatypeMetadata

logger = logging.getLogger(__name__)


@dataclass
class InterfaceMetadata:
    """
    Interface metadata with QONNX constraint groups - no default datatypes.
    
    This class encapsulates all information needed to create and configure
    a dataflow interface with runtime datatype validation.
    """
    name: str
    interface_type: InterfaceType
    compiler_name: Optional[str] = None
    """
    Standardized compiler name for consistent code generation.
    
    Format based on interface type and discovery order:
    - CONTROL: 'global'
    - INPUT: 'input0', 'input1', 'input2', ...
    - WEIGHT: 'weight0', 'weight1', 'weight2', ...
    - OUTPUT: 'output0', 'output1', 'output2', ...
    
    Used for consistent variable naming in generated AutoHWCustomOp and AutoRTLBackend.
    """
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    chunking_strategy: Any = field(default=None)  # DEPRECATED: Legacy dataflow field, to be removed with AutoHWCustomOp refactoring
    default_layout: Optional[str] = None
    description: Optional[str] = None
    
    # Parameter linkage mappings
    datatype_metadata: Optional[DatatypeMetadata] = None
    """
    Optional datatype metadata binding RTL parameters to QONNX datatype properties.
    If None, defaults will be created based on interface name patterns.
    
    Example: DatatypeMetadata(name="in0", datatype_params={"width": "INPUT0_WIDTH", "signed": "SIGNED_INPUT0"})
    """
    
    bdim_param: Optional[str] = None
    """
    RTL parameter name for block dimensions (legacy single parameter).
    If None, defaults to {clean_interface_name}_BDIM pattern.
    
    Example: "INPUT0_BDIM", "WEIGHTS_BDIM"
    """
    
    bdim_params: Optional[List[str]] = None
    """
    RTL parameter names that define the block dimensions.
    These are the actual parameters used in the RTL (e.g., TILE_H, TILE_W).
    For compatibility, bdim_param can still be used for single parameter case.
    
    Example: ["TILE_H", "TILE_W", "TILE_C"]
    """
    
    sdim_param: Optional[str] = None
    """
    RTL parameter name for stream dimensions.
    If None, defaults to {clean_interface_name}_SDIM pattern.
    
    Example: "INPUT0_SDIM", "WEIGHTS_SDIM"
    """
    
    # Multi-dimensional support
    sdim_params: Optional[List[str]] = None
    """
    RTL parameter names for multi-dimensional stream dimensions.
    For single dimension, use sdim_param. For multi-dimensional, use this list.
    
    Example: ["SDIM_D0", "SDIM_D1", "SDIM_D2"]
    """
    
    block_shape: Optional[List[Any]] = None
    """
    Multi-dimensional block shape specification.
    Elements can be parameter names (str) or literals (int) or ':' for full dimension.
    
    Example: ["TILE_M", 64, ":"]
    """
    
    block_rindex: int = 0
    """
    Starting index for block dimensions (from BDIM pragma RINDEX).
    """
    
    shape_params: Optional[Dict[str, Any]] = None
    """
    Optional shape specification parameters from BDIM pragma.
    Contains 'shape' (List[str]) and 'rindex' (int) if specified.
    
    Example: {"shape": ["C", "PE"], "rindex": 0}
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
    
    def validate_parameters(self, module_param_names: set) -> Tuple[List[str], List[str]]:
        """
        Validate that expected parameters exist in the module.
        
        This method checks if the interface's expected parameters (BDIM, SDIM, 
        datatype parameters) actually exist in the RTL module's parameter list.
        
        Args:
            module_param_names: Set of parameter names from the RTL module
            
        Returns:
            Tuple of (errors, warnings) - errors are critical validation failures
        """
        errors = []
        warnings = []
        
        # Skip non-dataflow interfaces
        if self.interface_type not in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
            return errors, warnings
        
        # CRITICAL: All stream interfaces MUST have BDIM
        has_bdim = bool(self.bdim_param or self.bdim_params)
        if not has_bdim:
            errors.append(
                f"AXI Stream interface '{self.name}' ({self.interface_type.value.upper()}) "
                f"is missing required BDIM parameter. Either add a parameter named "
                f"'{self.name}_BDIM' or use '@brainsmith BDIM {self.name} <param_name>' pragma."
            )
        elif self.bdim_param and self.bdim_param not in module_param_names:
            # BDIM is set but parameter doesn't exist
            errors.append(
                f"Interface '{self.name}' references BDIM parameter '{self.bdim_param}' "
                f"which does not exist in the module."
            )
        elif self.bdim_params:
            # Check multi-dimensional BDIM parameters
            missing_params = [p for p in self.bdim_params if p != "1" and p not in module_param_names]
            if missing_params:
                errors.append(
                    f"Interface '{self.name}' references BDIM parameters {missing_params} "
                    f"which do not exist in the module."
                )
        
        # CRITICAL: INPUT and WEIGHT interfaces MUST have SDIM
        if self.interface_type.value in ['input', 'weight']:
            has_sdim = bool(self.sdim_param or self.sdim_params)
            if not has_sdim:
                errors.append(
                    f"Interface '{self.name}' ({self.interface_type.value.upper()}) "
                    f"is missing required SDIM parameter. Either add a parameter named "
                    f"'{self.name}_SDIM' or use '@brainsmith SDIM {self.name} <param_name>' pragma."
                )
            elif self.sdim_param and self.sdim_param not in module_param_names:
                # SDIM is set but parameter doesn't exist
                errors.append(
                    f"Interface '{self.name}' references SDIM parameter '{self.sdim_param}' "
                    f"which does not exist in the module."
                )
            elif self.sdim_params:
                # Check multi-dimensional SDIM parameters
                missing_params = [p for p in self.sdim_params if p != "1" and p not in module_param_names]
                if missing_params:
                    errors.append(
                        f"Interface '{self.name}' references SDIM parameters {missing_params} "
                        f"which do not exist in the module."
                    )
        
        # Check datatype parameters - validate DatatypeMetadata exists and has width
        if not self.datatype_metadata:
            # No datatype metadata at all - this is a warning for dataflow interfaces
            warnings.append(
                f"Interface '{self.name}' has no datatype metadata. "
                f"Use @brainsmith DATATYPE_PARAM pragma to specify parameter mappings."
            )
        elif self.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
            # For streaming interfaces, width is required
            if not self.datatype_metadata.width:
                warnings.append(
                    f"Interface '{self.name}' datatype has no width parameter specified. "
                    f"Streaming interfaces require a width parameter for data formatting."
                )
        
        # Check shape parameters reference valid module parameters
        if self.shape_params:
            shape = self.shape_params.get('shape', [])
            for shape_element in shape:
                if shape_element != ':' and shape_element not in module_param_names:
                    warnings.append(
                        f"Interface '{self.name}' BDIM shape references parameter '{shape_element}' "
                        f"that was not found in module parameters."
                    )
        
        return errors, warnings
    
    def update_attributes(self, **kwargs) -> 'InterfaceMetadata':
        """
        Create a new InterfaceMetadata instance with updated attributes.
        
        This method provides a clean way to update specific fields while preserving
        all other attributes. Any field not specified in kwargs will retain its
        current value.
        
        Args:
            **kwargs: Keyword arguments for fields to update
            
        Returns:
            InterfaceMetadata: New instance with updated attributes
            
        Example:
            >>> metadata = InterfaceMetadata(name="in0", interface_type=InterfaceType.INPUT)
            >>> updated = metadata.update_attributes(
            ...     datatype_constraints=[new_constraint],
            ...     bdim_param="IN0_BDIM"
            ... )
        """
        # Get current values as a dict
        current_values = {
            'name': self.name,
            'interface_type': self.interface_type,
            'compiler_name': self.compiler_name,
            'datatype_constraints': self.datatype_constraints,
            'chunking_strategy': self.chunking_strategy,
            'default_layout': self.default_layout,
            'description': self.description,
            'datatype_metadata': self.datatype_metadata,
            'bdim_param': self.bdim_param,
            'bdim_params': self.bdim_params,
            'sdim_param': self.sdim_param,
            'sdim_params': self.sdim_params,
            'block_shape': self.block_shape,
            'block_rindex': self.block_rindex,
            'shape_params': self.shape_params
        }
        
        # Update with provided kwargs
        current_values.update(kwargs)
        
        # Create and return new instance
        return InterfaceMetadata(**current_values)
    
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
        
        Returns:
            RTL parameter name (e.g., 'potato_BDIM', 'weights_V_BDIM')
        """
        if self.bdim_param:
            return self.bdim_param
        
        # Default naming convention: use actual interface name directly
        return f"{self.name}_BDIM"
    
    def get_sdim_parameter_name(self) -> str:
        """
        Get RTL parameter name for stream dimensions.
        
        Returns:
            RTL parameter name (e.g., 'potato_SDIM', 'weights_V_SDIM')
        """
        if self.sdim_param:
            return self.sdim_param
        
        # Default naming convention: use actual interface name directly
        return f"{self.name}_SDIM"
    
    def has_shape_linkage(self) -> bool:
        """
        Check if interface has complete shape/size parameter linkage.
        
        Returns:
            bool: True if both BDIM and SDIM parameters are available
        """
        return bool(self.bdim_param or self._has_default_bdim_param()) and \
               bool(self.sdim_param or self._has_default_sdim_param())
    
    def _has_default_bdim_param(self) -> bool:
        """Check if interface follows default BDIM parameter naming."""
        # This will be validated during interface building
        return True  # Assume available for now
    
    def _has_default_sdim_param(self) -> bool:
        """Check if interface follows default SDIM parameter naming."""
        # This will be validated during interface building
        return True  # Assume available for now
    


@dataclass
class InterfaceMetadataCollection:
    """
    Collection of interface metadata for an AutoHWCustomOp.
    
    Provides convenient access patterns and validation.
    """
    interfaces: List[InterfaceMetadata]
    
    def __post_init__(self):
        """Validate the collection."""
        interface_names = [iface.name for iface in self.interfaces]
        if len(interface_names) != len(set(interface_names)):
            raise ValueError("Duplicate interface names found")
    
    def get_by_name(self, name: str) -> Optional[InterfaceMetadata]:
        """Get interface metadata by name."""
        for iface in self.interfaces:
            if iface.name == name:
                return iface
        return None
    
    def get_by_type(self, interface_type: InterfaceType) -> List[InterfaceMetadata]:
        """Get all interfaces of a specific type."""
        return [iface for iface in self.interfaces if iface.interface_type == interface_type]
    
    def get_input_interfaces(self) -> List[InterfaceMetadata]:
        """Get all input interfaces."""
        return self.get_by_type(InterfaceType.INPUT)
    
    def get_output_interfaces(self) -> List[InterfaceMetadata]:
        """Get all output interfaces."""
        return self.get_by_type(InterfaceType.OUTPUT)
    
    def get_weight_interfaces(self) -> List[InterfaceMetadata]:
        """Get all weight interfaces."""
        return self.get_by_type(InterfaceType.WEIGHT)
    
    def get_config_interfaces(self) -> List[InterfaceMetadata]:
        """Get all config interfaces."""
        return self.get_by_type(InterfaceType.CONFIG)
    
    def interface_names(self) -> List[str]:
        """Get list of all interface names."""
        return [iface.name for iface in self.interfaces]
    
    def validate_datatype_for_interface(self, interface_name: str, datatype_string: str) -> bool:
        """Validate a datatype for a specific interface."""
        iface = self.get_by_name(interface_name)
        if not iface:
            raise KeyError(f"Interface '{interface_name}' not found")
        return iface.validates_datatype(datatype_string)
