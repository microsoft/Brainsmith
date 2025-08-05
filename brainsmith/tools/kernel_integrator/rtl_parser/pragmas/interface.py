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
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup
from brainsmith.core.dataflow.types import InterfaceType
from ...metadata import InterfaceMetadata, DatatypeMetadata
from ..rtl_data import PragmaType

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
        @brainsmith DATATYPE <interface_name> [<type1>, <type2>, ...] <min_bits> <max_bits>
        @brainsmith DATATYPE <interface_name> * <min_bits> <max_bits>
        
        Example: @brainsmith DATATYPE in0 UINT 8 16
        Example: @brainsmith DATATYPE weights FIXED 8 8
        Example: @brainsmith DATATYPE in0 [INT, UINT, FIXED] 1 32
        Example: @brainsmith DATATYPE in0 * 8 32  # Any type from 8-32 bits
        """
        logger.debug(f"Parsing DATATYPE pragma: {self.inputs} at line {self.line_number}")
        
        pos = self.inputs['positional']
        
        if len(pos) != 4:
            raise PragmaError("DATATYPE pragma requires interface_name, base_type(s), min_bits, max_bits")
        
        interface_name = pos[0]
        base_types_input = pos[1]
        
        # Handle both single type (string) and list of types
        if isinstance(base_types_input, list):
            # List of types provided
            base_types = base_types_input
            if not base_types:
                raise PragmaError("DATATYPE pragma base type list cannot be empty")
            # Handle wildcard in list - if * is present, use ANY
            if "*" in base_types:
                base_types = ["ANY"]
        else:
            # Single type provided - convert to list for consistent handling
            if base_types_input.strip() == "*":
                base_types = ["ANY"]
            else:
                base_types = [base_types_input.strip()]
        
        try:
            min_bits = int(pos[2])
            max_bits = int(pos[3])
        except ValueError:
            raise PragmaError(f"DATATYPE pragma min_bits and max_bits must be integers, got: {pos[2]}, {pos[3]}")
        
        if min_bits <= 0:
            raise PragmaError(f"DATATYPE pragma min_bits must be positive, got: {min_bits}")
        
        if min_bits > max_bits:
            raise PragmaError(f"DATATYPE pragma min_bits ({min_bits}) cannot be greater than max_bits ({max_bits})")
        
        # Validate each base type using DatatypeConstraintGroup validation
        for base_type in base_types:
            # ANY type is always valid - skip additional validation
            if base_type == "ANY":
                continue
            try:
                # Test constraint group creation to validate base type
                DatatypeConstraintGroup(base_type, min_bits, max_bits)
            except ValueError as e:
                raise PragmaError(f"DATATYPE pragma invalid base type '{base_type}' or constraints: {e}")
        
        return {
            "interface_name": interface_name,
            "base_types": base_types,  # Now always a list
            "min_width": min_bits,
            "max_width": max_bits
        }

    def apply_to_interface(self, metadata: InterfaceMetadata) -> None:
        """Apply DATATYPE pragma to modify datatype constraints."""
        logger.debug(f"Attempting to apply DATATYPE pragma to interface '{metadata.name}'")
        
        # Validate interface type - exclude CONTROL
        if metadata.interface_type == InterfaceType.CONTROL:
            error_msg = (f"DATATYPE pragma at line {self.line_number} cannot be applied to "
                        f"CONTROL interface '{metadata.name}'. DATATYPE pragmas are not "
                        f"applicable to clock/reset signals.")
            logger.error(f"DATATYPE interface type validation failed: {error_msg}")
            raise PragmaError(error_msg)
        
        # Create new datatype constraint groups based on pragma
        new_constraint_groups = []
        base_types = self.parsed_data.get("base_types", ["UINT"])
        min_width = self.parsed_data.get("min_width", 8)
        max_width = self.parsed_data.get("max_width", 32)
        
        for base_type in base_types:
            constraint_group = DatatypeConstraintGroup(base_type, min_width, max_width)
            new_constraint_groups.append(constraint_group)
        
        # Combine with existing constraints (pragma adds to constraints, doesn't replace)
        existing_constraints = metadata.datatype_constraints or []
        metadata.datatype_constraints = existing_constraints + new_constraint_groups
        
        logger.debug(f"DATATYPE pragma successfully applied to interface '{metadata.name}' with {len(new_constraint_groups)} constraint groups")



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
        
        pos = self.inputs['positional']
        
        if not pos:
            raise PragmaError(f"WEIGHT pragma at line {self.line_number} requires at least one argument: <interface_name_0> [...]")
        
        # All inputs are interface names
        interface_names = pos
        return {"interface_names": interface_names}
    
    def apply_to_interface(self, metadata: InterfaceMetadata) -> None:
        """Apply WEIGHT pragma to mark interface as weight type."""
        metadata.interface_type = InterfaceType.WEIGHT  # Override type

    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply WEIGHT pragma to kernel metadata."""
        interface_names = self.parsed_data.get("interface_names", [])
        
        # WeightPragma handles multiple interfaces, so we apply to each one
        for interface_name in interface_names:
            found = False
            for interface in kernel.interfaces:
                if interface.name == interface_name:
                    self.apply_to_interface(interface)
                    logger.debug(f"Applied WEIGHT pragma to interface '{interface_name}'")
                    found = True
                    break
            
            if not found:
                logger.warning(f"WEIGHT pragma target interface '{interface_name}' not found")



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
        pos = self.inputs['positional']
        
        if len(pos) != 3:
            raise PragmaError("DATATYPE_PARAM pragma requires interface_name, property_type, parameter_name")
        
        interface_name = pos[0]
        property_type = pos[1].lower()
        parameter_name = pos[2]
        
        # Validate property type
        valid_properties = ['width', 'signed', 'format', 'bias', 'fractional_width']
        if property_type not in valid_properties:
            raise PragmaError(f"Invalid property_type '{property_type}'. Must be one of: {valid_properties}")
        
        return {
            "interface_name": interface_name,
            "property_type": property_type, 
            "parameter_name": parameter_name
        }
    
    def apply_to_interface(self, metadata: InterfaceMetadata) -> None:
        """Apply DATATYPE_PARAM pragma to set datatype parameter mapping."""
        logger.debug(f"Attempting to apply DATATYPE_PARAM pragma to interface '{metadata.name}'")
        
        # Validate interface type - exclude CONTROL
        if metadata.interface_type == InterfaceType.CONTROL:
            error_msg = (f"DATATYPE_PARAM pragma at line {self.line_number} cannot be applied to "
                        f"CONTROL interface '{metadata.name}'. Datatype parameters are not "
                        f"applicable to clock/reset signals.")
            logger.error(f"DATATYPE_PARAM interface type validation failed: {error_msg}")
            raise PragmaError(error_msg)
        
        property_type = self.parsed_data['property_type']
        parameter_name = self.parsed_data['parameter_name']
        
        # DatatypeMetadata is already imported at module level
        
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
            metadata.datatype_metadata = new_dt
        else:
            # Update existing DatatypeMetadata in-place
            setattr(metadata.datatype_metadata, property_type, parameter_name)
    
    def apply_to_kernel(self, kernel: 'KernelMetadata') -> None:
        """Apply DATATYPE_PARAM pragma to kernel metadata."""
        interface_name = self.parsed_data.get("interface_name")
        property_type = self.parsed_data.get("property_type")
        parameter_name = self.parsed_data.get("parameter_name")
        
        # Try to find and apply to interface
        found = False
        for interface in kernel.interfaces:
            if interface.name == interface_name:
                self.apply_to_interface(interface)
                logger.debug(f"Applied DATATYPE_PARAM pragma to interface '{interface_name}'")
                found = True
                # Remove from exposed parameters
                if parameter_name in kernel.exposed_parameters:
                    kernel.exposed_parameters.remove(parameter_name)
                break
        
        if found:
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
        
        interface_name = self.parsed_data['interface_name']
        property_type = self.parsed_data['property_type']
        parameter_name = self.parsed_data['parameter_name']
        
        # Create datatype with only the specified property
        kwargs = {
            'name': interface_name,
            'description': f"Internal datatype binding from pragma at line {self.line_number}"
        }
        
        # Add the specific property
        if property_type in ['width', 'signed', 'format', 'bias', 'fractional_width', 
                             'exponent_width', 'mantissa_width']:
            kwargs[property_type] = parameter_name
        
        return DatatypeMetadata(**kwargs)