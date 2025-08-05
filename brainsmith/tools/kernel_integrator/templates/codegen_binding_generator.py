"""
CodegenBinding generator for template context.

Creates unified CodegenBinding from KernelMetadata for use in templates.
"""

from typing import Dict, List, Set, TYPE_CHECKING
from ..metadata import KernelMetadata, InterfaceMetadata
from brainsmith.core.dataflow.types import InterfaceType

if TYPE_CHECKING:
    from ..codegen_binding import CodegenBinding


def generate_codegen_binding(kernel_metadata: KernelMetadata) -> 'CodegenBinding':
    """
    Generate unified CodegenBinding from KernelMetadata.
    
    This consolidates all parameter binding information into a single
    object for use in code generation.
    
    Args:
        kernel_metadata: Complete kernel metadata from RTL parser
        
    Returns:
        CodegenBinding with all parameter linkage information
    """
    # Import inside function to avoid circular dependency
    from ..codegen_binding import (
        CodegenBinding, ParameterBinding, ParameterSource,
        InterfaceBinding, InternalBinding, SourceType, ParameterCategory
    )
    
    codegen = CodegenBinding()
    
    # Track which parameters we've seen
    processed_params = set()
    
    # 1. Process exposed algorithm parameters
    # First, identify which parameters are BDIM/SDIM from interface metadata
    shape_params = set()
    for interface in kernel_metadata.interfaces:
        if interface.bdim_params:
            shape_params.update(p for p in interface.bdim_params if p != '1')
        if interface.sdim_params:
            shape_params.update(p for p in interface.sdim_params if p != '1')
    
    for param_name in kernel_metadata.exposed_parameters:
        # Determine category based on parameter usage
        if param_name in shape_params:
            # This is a shape parameter (BDIM/SDIM)
            category = ParameterCategory.SHAPE
        else:
            # This is an algorithm parameter
            category = ParameterCategory.ALGORITHM
            
        codegen.add_parameter_binding(
            param_name,
            ParameterSource(type=SourceType.NODEATTR),
            category
        )
        processed_params.add(param_name)
    
    # 2. Process parameter aliases
    aliases = kernel_metadata.linked_parameters.get("aliases", {})
    for rtl_param, nodeattr_name in aliases.items():
        codegen.add_parameter_binding(
            rtl_param,
            ParameterSource(
                type=SourceType.NODEATTR_ALIAS,
                nodeattr_name=nodeattr_name
            ),
            ParameterCategory.ALGORITHM
        )
        processed_params.add(rtl_param)
    
    # 3. Process derived parameters
    derived = kernel_metadata.linked_parameters.get("derived", {})
    for param_name, expression in derived.items():
        codegen.add_parameter_binding(
            param_name,
            ParameterSource(
                type=SourceType.DERIVED,
                expression=expression
            ),
            ParameterCategory.ALGORITHM
        )
        processed_params.add(param_name)
    
    # 4. Process AXI-Lite parameters
    axilite = kernel_metadata.linked_parameters.get("axilite", {})
    for param_name, interface_name in axilite.items():
        codegen.add_parameter_binding(
            param_name,
            ParameterSource(
                type=SourceType.NODEATTR,
                interface_name=interface_name
            ),
            ParameterCategory.CONTROL
        )
        processed_params.add(param_name)
    
    # 5. Process interface bindings
    for interface in kernel_metadata.interfaces:
        if interface.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
            # Check if interface has any parameters
            has_params = False
            
            # Datatype parameters
            datatype_params = {}
            if interface.datatype_metadata:
                dt_meta = interface.datatype_metadata
                if dt_meta.width:
                    datatype_params["width"] = dt_meta.width
                    has_params = True
                if dt_meta.signed:
                    datatype_params["signed"] = dt_meta.signed
                    has_params = True
                if dt_meta.format:
                    datatype_params["format"] = dt_meta.format
                    has_params = True
                if dt_meta.bias:
                    datatype_params["bias"] = dt_meta.bias
                    has_params = True
                if dt_meta.fractional_width:
                    datatype_params["fractional_width"] = dt_meta.fractional_width
                    has_params = True
            
            # Shape parameters
            bdim_params = interface.bdim_params if interface.bdim_params else []
            sdim_params = interface.sdim_params if interface.sdim_params else []
            
            if bdim_params or sdim_params:
                has_params = True
            
            # Only add interface binding if it has parameters
            if has_params:
                codegen.add_interface_binding(
                    interface.name,
                    datatype_params=datatype_params if datatype_params else None,
                    bdim_params=bdim_params,
                    sdim_params=sdim_params
                )
                
                # Add parameter bindings for interface parameters
                _add_interface_parameter_bindings(
                    codegen, interface, processed_params
                )
    
    # 6. Process internal datatype bindings
    for internal_dt in kernel_metadata.internal_datatypes:
        datatype_params = {}
        
        if internal_dt.width:
            datatype_params["width"] = internal_dt.width
        if internal_dt.signed:
            datatype_params["signed"] = internal_dt.signed
        if internal_dt.format:
            datatype_params["format"] = internal_dt.format
        
        if datatype_params:
            codegen.add_internal_binding(
                internal_dt.name,
                datatype_params=datatype_params
            )
            
            # Add parameter bindings for internal datatype parameters
            _add_internal_parameter_bindings(
                codegen, internal_dt, processed_params
            )
    
    # 7. Identify any remaining parameters as hidden
    all_params = {p.name for p in kernel_metadata.parameters}
    codegen.hidden_parameters = list(all_params - processed_params)
    
    return codegen


def _add_interface_parameter_bindings(
    codegen: 'CodegenBinding',
    interface: InterfaceMetadata,
    processed_params: Set[str]
) -> None:
    """Add parameter bindings for interface parameters."""
    # Import inside function to avoid circular dependency
    from ..codegen_binding import ParameterSource, SourceType, ParameterCategory
    
    # Datatype parameters
    if interface.datatype_metadata:
        dt_meta = interface.datatype_metadata
        
        if dt_meta.width and dt_meta.width not in processed_params:
            codegen.add_parameter_binding(
                dt_meta.width,
                ParameterSource(
                    type=SourceType.INTERFACE_DATATYPE,
                    interface_name=interface.name,
                    property_name="width"
                ),
                ParameterCategory.DATATYPE
            )
            processed_params.add(dt_meta.width)
        
        if dt_meta.signed and dt_meta.signed not in processed_params:
            codegen.add_parameter_binding(
                dt_meta.signed,
                ParameterSource(
                    type=SourceType.INTERFACE_DATATYPE,
                    interface_name=interface.name,
                    property_name="signed"
                ),
                ParameterCategory.DATATYPE
            )
            processed_params.add(dt_meta.signed)
        
        if dt_meta.format and dt_meta.format not in processed_params:
            codegen.add_parameter_binding(
                dt_meta.format,
                ParameterSource(
                    type=SourceType.INTERFACE_DATATYPE,
                    interface_name=interface.name,
                    property_name="format"
                ),
                ParameterCategory.DATATYPE
            )
            processed_params.add(dt_meta.format)
    
    # BDIM parameters
    if interface.bdim_params:
        for i, param_name in enumerate(interface.bdim_params):
            if param_name != '1' and param_name not in processed_params:
                codegen.add_parameter_binding(
                    param_name,
                    ParameterSource(
                        type=SourceType.INTERFACE_SHAPE,
                        interface_name=interface.name,
                        property_name="bdim",
                        dimension_index=i
                    ),
                    ParameterCategory.SHAPE
                )
                processed_params.add(param_name)
    
    # SDIM parameters
    if interface.sdim_params:
        for i, param_name in enumerate(interface.sdim_params):
            if param_name != '1' and param_name not in processed_params:
                codegen.add_parameter_binding(
                    param_name,
                    ParameterSource(
                        type=SourceType.INTERFACE_SHAPE,
                        interface_name=interface.name,
                        property_name="sdim",
                        dimension_index=i
                    ),
                    ParameterCategory.SHAPE
                )
                processed_params.add(param_name)


def _add_internal_parameter_bindings(
    codegen: 'CodegenBinding',
    internal_dt,
    processed_params: Set[str]
) -> None:
    """Add parameter bindings for internal datatype parameters."""
    # Import inside function to avoid circular dependency
    from ..codegen_binding import ParameterSource, SourceType, ParameterCategory
    
    if internal_dt.width and internal_dt.width not in processed_params:
        codegen.add_parameter_binding(
            internal_dt.width,
            ParameterSource(
                type=SourceType.INTERNAL_DATATYPE,
                interface_name=internal_dt.name,  # Using interface_name field
                property_name="width"
            ),
            ParameterCategory.INTERNAL
        )
        processed_params.add(internal_dt.width)
    
    if internal_dt.signed and internal_dt.signed not in processed_params:
        codegen.add_parameter_binding(
            internal_dt.signed,
            ParameterSource(
                type=SourceType.INTERNAL_DATATYPE,
                interface_name=internal_dt.name,
                property_name="signed"
            ),
            ParameterCategory.INTERNAL
        )
        processed_params.add(internal_dt.signed)