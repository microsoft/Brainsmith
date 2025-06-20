############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Interface-related pragma implementations.

This module contains pragmas that modify interface metadata including
datatype constraints, parameter mappings, and interface types.
"""

from dataclasses import dataclass
from typing import Dict
import logging

from .base import InterfacePragma, PragmaError
from ..data import PragmaType
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
from brainsmith.dataflow.core.interface_types import InterfaceType

logger = logging.getLogger(__name__)


@dataclass
class DatatypePragma(InterfacePragma):
    """DATATYPE pragma for constraining interface datatypes.
    
    Format: @brainsmith datatype <interface_name> <base_type> <min_bits> <max_bits>
    
    This pragma adds datatype constraints to an interface, specifying the
    allowed base types and bit widths.
    
    Examples:
    - @brainsmith datatype in0 UINT 8 16
    - @brainsmith datatype weights FIXED 8 8
    """
    
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

    def apply_to_interface(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply DATATYPE pragma to modify datatype constraints."""
        # Create new datatype constraint group based on pragma
        new_constraint_group = self._create_constraint_group()
        
        # Combine with existing constraints (pragma adds to constraints, doesn't replace)
        existing_constraints = getattr(metadata, 'datatype_constraints', []) or []
        new_constraints = existing_constraints + [new_constraint_group]
        
        return metadata.update_attributes(datatype_constraints=new_constraints)

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
class WeightPragma(InterfacePragma):
    """WEIGHT pragma for marking interfaces as weight type.
    
    Format: @brainsmith weight <interface_name_0> [<interface_name_1> ...]
    
    This pragma marks one or more interfaces as weight (parameter) interfaces,
    which have special handling in the dataflow model.
    
    Examples:
    - @brainsmith weight weights
    - @brainsmith weight weights0 weights1 bias
    """
    
    def __post_init__(self):
        super().__post_init__()

    def _parse_inputs(self) -> Dict:
        """Handles WEIGHT pragma: @brainsmith WEIGHT <interface_name_0> [<interface_name_1> ...]"""
        logger.debug(f"Parsing WEIGHT pragma: {self.inputs} at line {self.line_number}")
        if not self.inputs:
            raise PragmaError(f"WEIGHT pragma at line {self.line_number} requires at least one argument: <interface_name_0> [...]. Got: {self.inputs}")
        
        # All inputs are interface names
        interface_names = self.inputs
        return {"interface_names": interface_names}

    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply WEIGHT pragma to kernel metadata."""
        interface_names = self.parsed_data.get("interface_names", [])
        
        # WeightPragma handles multiple interfaces, so we apply to each one
        for interface_name in interface_names:
            self.apply_to_interface_by_name(interface_name, kernel)
    
    def apply_to_interface(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply WEIGHT pragma to mark interface as weight type."""
        return metadata.update_attributes(
            interface_type=InterfaceType.WEIGHT  # Override type
        )


@dataclass 
class DatatypeParamPragma(InterfacePragma):
    """Maps specific RTL parameters to interface datatype properties.
    
    Format: @brainsmith datatype_param <interface_name> <property_type> <parameter_name>
    
    This pragma links RTL parameters to datatype properties like width, signed, etc.
    Can be used for both interfaces and internal datatypes.
    
    Examples:
    - @brainsmith datatype_param s_axis_input0 width INPUT0_WIDTH
    - @brainsmith datatype_param s_axis_input0 signed SIGNED_INPUT0
    - @brainsmith datatype_param accumulator width ACC_WIDTH
    """
    
    def _parse_inputs(self) -> Dict:
        if len(self.inputs) != 3:
            raise PragmaError("DATATYPE_PARAM pragma requires interface_name, property_type, parameter_name")
        
        interface_name = self.inputs[0]
        property_type = self.inputs[1].lower()
        parameter_name = self.inputs[2]
        
        # Validate property type
        valid_properties = ['width', 'signed', 'format', 'bias', 'fractional_width']
        if property_type not in valid_properties:
            raise PragmaError(f"Invalid property_type '{property_type}'. Must be one of: {valid_properties}")
        
        return {
            "interface_name": interface_name,
            "property_type": property_type, 
            "parameter_name": parameter_name
        }
    
    def apply_to_interface(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply DATATYPE_PARAM pragma to set datatype parameter mapping."""
        property_type = self.parsed_data['property_type']
        parameter_name = self.parsed_data['parameter_name']
        
        # Import DatatypeMetadata
        from brainsmith.dataflow.core.datatype_metadata import DatatypeMetadata
        
        # Get or create DatatypeMetadata
        if metadata.datatype_metadata is None:
            # Create new DatatypeMetadata with interface name
            # For now, we'll create with just width and update it
            if property_type == 'width':
                new_dt = DatatypeMetadata(name=metadata.name, width=parameter_name)
            else:
                # Need a width parameter first - use default naming
                new_dt = DatatypeMetadata(
                    name=metadata.name, 
                    width=f"{metadata.name}_WIDTH",
                    **{property_type: parameter_name}
                )
            metadata = metadata.update_attributes(datatype_metadata=new_dt)
        else:
            # Update existing DatatypeMetadata
            updated_dt_metadata = metadata.datatype_metadata.update(**{property_type: parameter_name})
            metadata = metadata.update_attributes(datatype_metadata=updated_dt_metadata)
        
        return metadata
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply DATATYPE_PARAM pragma to kernel metadata."""
        interface_name = self.parsed_data.get("interface_name")
        property_type = self.parsed_data.get("property_type")
        parameter_name = self.parsed_data.get("parameter_name")
        
        # Try to apply to interface first using unified method
        if self.apply_to_interface_by_name(interface_name, kernel):
            # Successfully applied to interface, also remove from exposed parameters
            if parameter_name in kernel.exposed_parameters:
                kernel.exposed_parameters.remove(parameter_name)
            return
        
        # If interface not found, this might be an internal datatype
        # Add to internal datatypes
        internal_dt = self.create_standalone_datatype()
        
        if kernel.internal_datatypes is None:
            kernel.internal_datatypes = []
        
        # Check if we already have this datatype and merge
        existing_dt = None
        for dt in kernel.internal_datatypes:
            if dt.name == interface_name:
                existing_dt = dt
                break
        
        if existing_dt:
            # Update existing datatype
            setattr(existing_dt, property_type, parameter_name)
        else:
            # Add new internal datatype
            kernel.internal_datatypes.append(internal_dt)
        
        # Remove parameter from exposed list
        if parameter_name in kernel.exposed_parameters:
            kernel.exposed_parameters.remove(parameter_name)
        
        logger.debug(f"Applied DATATYPE_PARAM pragma to internal datatype '{interface_name}': {property_type}={parameter_name}")
    
    def create_standalone_datatype(self) -> 'DatatypeMetadata':
        """Create a standalone DatatypeMetadata for internal mechanisms."""
        from brainsmith.dataflow.core.datatype_metadata import DatatypeMetadata
        
        interface_name = self.parsed_data['interface_name']
        property_type = self.parsed_data['property_type']
        parameter_name = self.parsed_data['parameter_name']
        
        # Create datatype with only the specified property
        return DatatypeMetadata(
            name=interface_name,
            **{property_type: parameter_name},
            description=f"Internal datatype binding from pragma at line {self.line_number}"
        )