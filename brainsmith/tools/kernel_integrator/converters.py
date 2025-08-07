"""
Converters for bridging kernel_integrator types with dataflow types.

This module provides conversion functions to transform kernel metadata
into dataflow-compatible structures while maintaining data integrity.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from brainsmith.core.dataflow.types import InterfaceType, ShapeSpec, Shape
from brainsmith.core.dataflow.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.input_definition import InputDefinition
from brainsmith.core.dataflow.output_definition import OutputDefinition
from brainsmith.core.dataflow.base import ParameterBinding
from brainsmith.core.dataflow.qonnx_types import BaseDataType, DataType
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup
from brainsmith.core.dataflow.relationships import DimensionRelationship

from .types.metadata import KernelMetadata, InterfaceMetadata
from .constraint_builder import (
    build_datatype_constraints,
    build_dimension_constraints,
    build_parameter_constraints
)


def metadata_to_kernel_definition(
    kernel_metadata: KernelMetadata,
    kernel_path: Optional[Path] = None
) -> KernelDefinition:
    """
    Convert KernelMetadata to dataflow KernelDefinition.
    
    This is the main entry point for converting parsed RTL metadata
    into the dataflow system's kernel representation.
    
    Args:
        kernel_metadata: Parsed kernel metadata from RTL
        kernel_path: Optional path to kernel implementation
        
    Returns:
        KernelDefinition ready for dataflow system
    """
    # Create kernel definition
    kernel_def = KernelDefinition(
        name=kernel_metadata.name,
        metadata={
            "source_file": str(kernel_metadata.source_file),
            "kernel_path": str(kernel_path) if kernel_path else None,
            "kernel_type": "rtl",
            "parameters": _convert_parameters_to_dict(kernel_metadata),
            "exposed_parameters": list(kernel_metadata.exposed_parameters),
            "internal_datatypes": _convert_internal_datatypes(kernel_metadata),
            "kernel_metadata": kernel_metadata  # Store original for full access
        }
    )
    
    # Convert interfaces to input/output definitions
    for interface in kernel_metadata.interfaces:
        if interface.interface_type == InterfaceType.INPUT:
            input_def = _convert_to_input_definition(interface, kernel_metadata)
            kernel_def.add_input(input_def)
        elif interface.interface_type == InterfaceType.OUTPUT:
            output_def = _convert_to_output_definition(interface, kernel_metadata)
            kernel_def.add_output(output_def)
        elif interface.interface_type == InterfaceType.WEIGHT:
            # Weights are treated as inputs in dataflow
            input_def = _convert_to_input_definition(interface, kernel_metadata, is_weight=True)
            kernel_def.add_input(input_def)
        # CONFIG and CONTROL interfaces stored in metadata
        elif interface.interface_type in [InterfaceType.CONFIG, InterfaceType.CONTROL]:
            if "special_interfaces" not in kernel_def.metadata:
                kernel_def.metadata["special_interfaces"] = {}
            kernel_def.metadata["special_interfaces"][interface.name] = {
                "type": interface.interface_type.value,
                "description": interface.description,
                "interface": interface  # Store full interface for access
            }
    
    # Convert relationships
    kernel_def.relationships = list(kernel_metadata.relationships)
    
    return kernel_def


def _convert_to_input_definition(
    interface: InterfaceMetadata,
    kernel_metadata: KernelMetadata,
    is_weight: bool = False
) -> InputDefinition:
    """Convert interface metadata to input definition."""
    # Extract tiling from interface
    block_tiling = interface.bdim_shape if interface.bdim_shape else None
    stream_tiling = interface.sdim_shape if interface.sdim_shape else None
    
    # Build datatype constraints
    datatype_constraints = interface.datatype_constraints or []
    
    # Create input definition
    input_def = InputDefinition(
        name=interface.name,
        datatype_constraints=datatype_constraints,
        block_tiling=block_tiling,
        stream_tiling=stream_tiling,
        is_weight=is_weight or interface.is_weight,
        optional=False  # RTL interfaces are typically required
    )
    
    # Store additional metadata as attributes
    input_def.metadata = {
        "rtl_name": interface.name,
        "description": interface.description,
        "weight_file": interface.weight_file,
        "bdim_params": interface.bdim_params,
        "sdim_params": interface.sdim_params,
        "datatype_metadata": interface.datatype_metadata,
        "interface": interface  # Store full interface
    }
    
    return input_def


def _convert_to_output_definition(
    interface: InterfaceMetadata,
    kernel_metadata: KernelMetadata
) -> OutputDefinition:
    """Convert interface metadata to output definition."""
    # Extract tiling from interface
    block_tiling = interface.bdim_shape if interface.bdim_shape else None
    stream_tiling = interface.sdim_shape if interface.sdim_shape else None
    
    # Build datatype constraints
    datatype_constraints = interface.datatype_constraints or []
    
    # Create output definition
    output_def = OutputDefinition(
        name=interface.name,
        datatype_constraints=datatype_constraints,
        block_tiling=block_tiling
        # Note: OutputDefinition doesn't have stream_tiling
    )
    
    # Store additional metadata as attributes
    output_def.metadata = {
        "rtl_name": interface.name,
        "description": interface.description,
        "bdim_params": interface.bdim_params,
        "sdim_params": interface.sdim_params,
        "datatype_metadata": interface.datatype_metadata,
        "interface": interface  # Store full interface
    }
    
    return output_def


def _convert_parameters_to_dict(kernel_metadata: KernelMetadata) -> Dict[str, Any]:
    """Convert parameter list to dictionary format."""
    param_dict = {}
    for param in kernel_metadata.parameters:
        param_dict[param.name] = {
            "type": param.rtl_type or "integer",
            "default": param.default_value,
            "description": None  # Parameter descriptions not stored
        }
    return param_dict


def _convert_internal_datatypes(kernel_metadata: KernelMetadata) -> List[Dict[str, Any]]:
    """Convert internal datatype metadata to serializable format."""
    result = []
    for dt_meta in kernel_metadata.internal_datatypes:
        result.append({
            "name": dt_meta.name,
            "parameters": dt_meta.get_all_parameters(),
            "description": dt_meta.description
        })
    return result


def kernel_definition_to_metadata(
    kernel_def: KernelDefinition,
    source_file: Path
) -> KernelMetadata:
    """
    Convert dataflow KernelDefinition back to KernelMetadata.
    
    This enables round-trip conversion for testing and compatibility.
    
    Args:
        kernel_def: Dataflow kernel definition
        source_file: Source file path for metadata
        
    Returns:
        KernelMetadata compatible with kernel_integrator
    """
    # If we stored the original metadata, use it
    if kernel_def.metadata and "kernel_metadata" in kernel_def.metadata:
        return kernel_def.metadata["kernel_metadata"]
    
    from .types.rtl import Parameter
    
    # Convert input/output definitions to interfaces
    interfaces = []
    
    # Convert inputs
    for input_def in kernel_def.input_definitions:
        interface = _convert_input_to_interface(input_def)
        interfaces.append(interface)
    
    # Convert outputs
    for output_def in kernel_def.output_definitions:
        interface = _convert_output_to_interface(output_def)
        interfaces.append(interface)
    
    # Extract special interfaces from metadata
    if kernel_def.metadata and "special_interfaces" in kernel_def.metadata:
        for name, info in kernel_def.metadata["special_interfaces"].items():
            # Use stored interface if available
            if "interface" in info:
                interfaces.append(info["interface"])
            else:
                interface = InterfaceMetadata(
                    name=name,
                    interface_type=InterfaceType(info["type"]),
                    description=info.get("description")
                )
                interfaces.append(interface)
    
    # Convert parameters from metadata
    parameters = []
    if kernel_def.metadata and "parameters" in kernel_def.metadata:
        for name, param_info in kernel_def.metadata["parameters"].items():
            param = Parameter(
                name=name,
                rtl_type=param_info.get("type", "integer"),
                default_value=param_info.get("default")
            )
            parameters.append(param)
    
    # Extract exposed parameters
    exposed_parameters = []
    if kernel_def.metadata and "exposed_parameters" in kernel_def.metadata:
        exposed_parameters = kernel_def.metadata["exposed_parameters"]
    
    # Create kernel metadata
    metadata = KernelMetadata(
        name=kernel_def.name,
        interfaces=interfaces,
        parameters=parameters,
        source_file=str(source_file),
        exposed_parameters=exposed_parameters,
        internal_datatypes=[],  # Would need reconstruction
        linked_parameters={},  # Would need reconstruction
        relationships=kernel_def.relationships
    )
    
    return metadata


def _convert_input_to_interface(input_def: InputDefinition) -> InterfaceMetadata:
    """Convert InputDefinition to InterfaceMetadata."""
    # Check for stored interface
    if hasattr(input_def, 'metadata') and input_def.metadata and "interface" in input_def.metadata:
        return input_def.metadata["interface"]
    
    # Determine interface type
    interface_type = InterfaceType.WEIGHT if input_def.is_weight else InterfaceType.INPUT
    
    # Extract shape information from tiling
    bdim_shape = input_def.block_tiling
    sdim_shape = input_def.stream_tiling
    
    # Extract metadata
    rtl_name = input_def.name
    description = None
    weight_file = None
    
    if hasattr(input_def, 'metadata') and input_def.metadata:
        rtl_name = input_def.metadata.get("rtl_name", input_def.name)
        description = input_def.metadata.get("description")
        weight_file = input_def.metadata.get("weight_file")
    
    return InterfaceMetadata(
        name=rtl_name,
        interface_type=interface_type,
        datatype_constraints=input_def.datatype_constraints,
        description=description,
        bdim_shape=bdim_shape,
        sdim_shape=sdim_shape,
        is_weight=input_def.is_weight,
        weight_file=weight_file
    )


def _convert_output_to_interface(output_def: OutputDefinition) -> InterfaceMetadata:
    """Convert OutputDefinition to InterfaceMetadata."""
    # Check for stored interface
    if hasattr(output_def, 'metadata') and output_def.metadata and "interface" in output_def.metadata:
        return output_def.metadata["interface"]
    
    # Extract shape information from tiling
    bdim_shape = output_def.block_tiling
    sdim_shape = None  # OutputDefinition doesn't have stream_tiling
    
    # Extract metadata
    rtl_name = output_def.name
    description = None
    
    if hasattr(output_def, 'metadata') and output_def.metadata:
        rtl_name = output_def.metadata.get("rtl_name", output_def.name)
        description = output_def.metadata.get("description")
    
    return InterfaceMetadata(
        name=rtl_name,
        interface_type=InterfaceType.OUTPUT,
        datatype_constraints=output_def.datatype_constraints,
        description=description,
        bdim_shape=bdim_shape,
        sdim_shape=sdim_shape
    )


def validate_conversion(
    original: KernelMetadata,
    converted: KernelDefinition
) -> Tuple[bool, List[str]]:
    """
    Validate that conversion preserves essential information.
    
    Args:
        original: Original kernel metadata
        converted: Converted kernel definition
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check name preservation
    if original.name != converted.name:
        errors.append(f"Name mismatch: {original.name} != {converted.name}")
    
    # Check interface count (considering special interfaces)
    original_count = len(original.interfaces)
    converted_count = len(converted.input_definitions) + len(converted.output_definitions)
    if converted.metadata and "special_interfaces" in converted.metadata:
        converted_count += len(converted.metadata["special_interfaces"])
    
    if original_count != converted_count:
        errors.append(
            f"Interface count mismatch: {original_count} != {converted_count}"
        )
    
    # Check interface types preserved
    original_types = {i.interface_type for i in original.interfaces}
    converted_types = set()
    
    # Add input types
    for inp in converted.input_definitions:
        if inp.is_weight:
            converted_types.add(InterfaceType.WEIGHT)
        else:
            converted_types.add(InterfaceType.INPUT)
    
    # Add output types
    for _ in converted.output_definitions:
        converted_types.add(InterfaceType.OUTPUT)
    
    # Add special interface types
    if converted.metadata and "special_interfaces" in converted.metadata:
        for info in converted.metadata["special_interfaces"].values():
            converted_types.add(InterfaceType(info["type"]))
    
    if original_types != converted_types:
        errors.append(f"Interface types mismatch: {original_types} != {converted_types}")
    
    # Check parameter preservation
    if converted.metadata and "parameters" in converted.metadata:
        if len(original.parameters) != len(converted.metadata["parameters"]):
            errors.append(
                f"Parameter count mismatch: {len(original.parameters)} != {len(converted.metadata['parameters'])}"
            )
    
    return len(errors) == 0, errors