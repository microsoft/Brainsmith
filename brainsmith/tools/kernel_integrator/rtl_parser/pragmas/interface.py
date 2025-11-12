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

import logging
from dataclasses import dataclass

from brainsmith.dataflow.constraint_types import DatatypeConstraintGroup
from brainsmith.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.metadata import KernelMetadata

from .base import InterfacePragma, PragmaError

logger = logging.getLogger(__name__)


@dataclass
class DatatypeConstraintPragma(InterfacePragma):
    """DATATYPE_CONSTRAINT pragma for constraining interface datatypes.

    Format: @brainsmith datatype_constraint <interface_name> <base_type> <min_bits> <max_bits>

    This pragma adds datatype constraints to an interface, specifying the
    allowed base types and bit widths.

    Examples:
    - @brainsmith datatype_constraint in0 UINT 8 16
    - @brainsmith datatype_constraint weights FIXED 8 8
    """

    def __post_init__(self):
        super().__post_init__()

    def _parse_inputs(self) -> dict:
        """
        Handles DATATYPE_CONSTRAINT pragma with constraint groups:
        @brainsmith DATATYPE_CONSTRAINT <interface_name> <base_type> <min_bits> <max_bits>
        @brainsmith DATATYPE_CONSTRAINT <interface_name> [<type1>, <type2>, ...] <min_bits> <max_bits>
        @brainsmith DATATYPE_CONSTRAINT <interface_name> * <min_bits> <max_bits>

        Example: @brainsmith DATATYPE_CONSTRAINT in0 UINT 8 16
        Example: @brainsmith DATATYPE_CONSTRAINT weights FIXED 8 8
        Example: @brainsmith DATATYPE_CONSTRAINT in0 [INT, UINT, FIXED] 1 32
        Example: @brainsmith DATATYPE_CONSTRAINT in0 * 8 32  # Any type from 8-32 bits
        """
        logger.debug(
            f"Parsing DATATYPE_CONSTRAINT pragma: {self.inputs} at line {self.inputs.get('line_number', 'unknown')}"
        )

        pos = self.inputs["positional"]

        if len(pos) != 4:
            raise PragmaError(
                "DATATYPE_CONSTRAINT pragma requires interface_name, base_type(s), min_bits, max_bits"
            )

        interface_name = pos[0]
        base_types_input = pos[1]

        # Handle both single type (string) and list of types
        if isinstance(base_types_input, list):
            # List of types provided
            base_types = base_types_input
            if not base_types:
                raise PragmaError("DATATYPE_CONSTRAINT pragma base type list cannot be empty")
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
            raise PragmaError(
                f"DATATYPE_CONSTRAINT pragma min_bits and max_bits must be integers, got: {pos[2]}, {pos[3]}"
            )

        if min_bits <= 0:
            raise PragmaError(
                f"DATATYPE_CONSTRAINT pragma min_bits must be positive, got: {min_bits}"
            )

        if min_bits > max_bits:
            raise PragmaError(
                f"DATATYPE_CONSTRAINT pragma min_bits ({min_bits}) cannot be greater than max_bits ({max_bits})"
            )

        # Validate each base type using DatatypeConstraintGroup validation
        for base_type in base_types:
            # ANY type is always valid - skip additional validation
            if base_type == "ANY":
                continue
            try:
                # Test constraint group creation to validate base type
                DatatypeConstraintGroup(base_type, min_bits, max_bits)
            except ValueError as e:
                raise PragmaError(
                    f"DATATYPE_CONSTRAINT pragma invalid base type '{base_type}' or constraints: {e}"
                )

        return {
            "interface_name": interface_name,
            "base_types": base_types,  # Now always a list
            "min_width": min_bits,
            "max_width": max_bits,
        }

    def apply_to_kernel(self, kernel: KernelMetadata) -> None:
        """Apply DATATYPE_CONSTRAINT pragma to kernel metadata."""
        interface_name = self.parsed_data.get("interface_name")

        # Find the interface using helper
        interface = self.find_interface(kernel, interface_name)
        if interface is None:
            logger.warning(
                f"DATATYPE_CONSTRAINT pragma target interface '{interface_name}' not found"
            )
            return

        # Validate interface type - exclude CONTROL
        if (
            hasattr(interface, "interface_type")
            and interface.interface_type == InterfaceType.CONTROL
        ):
            error_msg = (
                f"DATATYPE_CONSTRAINT pragma at line {self.inputs.get('line_number', 'unknown')} cannot be applied to "
                f"CONTROL interface '{interface_name}'. DATATYPE_CONSTRAINT pragmas are not "
                f"applicable to clock/reset signals."
            )
            logger.error(f"DATATYPE_CONSTRAINT interface type validation failed: {error_msg}")
            raise PragmaError(error_msg)

        # Create new datatype constraint groups based on pragma
        new_constraint_groups = []
        base_types = self.parsed_data.get("base_types", ["UINT"])
        min_width = self.parsed_data.get("min_width", 8)
        max_width = self.parsed_data.get("max_width", 32)

        for base_type in base_types:
            constraint_group = DatatypeConstraintGroup(base_type, min_width, max_width)
            new_constraint_groups.append(constraint_group)

        # Check if interface supports dataflow properties
        if hasattr(interface, "supports_dataflow") and interface.supports_dataflow():
            # Create DataflowMetadata if it doesn't exist
            if not hasattr(interface, "dataflow") or interface.dataflow is None:
                from brainsmith.tools.kernel_integrator.metadata import DataflowMetadata

                interface.dataflow = DataflowMetadata()

            # Combine with existing constraints (pragma adds to constraints, doesn't replace)
            existing_constraints = interface.dataflow.datatype_constraints or []
            interface.dataflow.datatype_constraints = existing_constraints + new_constraint_groups
        else:
            # Fallback for interfaces without dataflow support (shouldn't happen with current design)
            if not hasattr(interface, "datatype_constraints"):
                interface.datatype_constraints = []
            existing_constraints = interface.datatype_constraints or []
            interface.datatype_constraints = existing_constraints + new_constraint_groups

        logger.debug(
            f"DATATYPE_CONSTRAINT pragma successfully applied to interface '{interface_name}' with {len(new_constraint_groups)} constraint groups"
        )


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

    def _parse_inputs(self) -> dict:
        """Handles WEIGHT pragma: @brainsmith WEIGHT <interface_name_0> [<interface_name_1> ...]"""
        logger.debug(
            f"Parsing WEIGHT pragma: {self.inputs} at line {self.inputs.get('line_number', 'unknown')}"
        )

        pos = self.inputs["positional"]

        if not pos:
            raise PragmaError(
                f"WEIGHT pragma at line {self.inputs.get('line_number', 'unknown')} requires at least one argument: <interface_name_0> [...]"
            )

        # All inputs are interface names
        interface_names = pos
        return {"interface_names": interface_names}

    def apply_to_kernel(self, kernel: "KernelMetadata") -> None:
        """Apply WEIGHT pragma to kernel metadata."""
        interface_names = self.parsed_data.get("interface_names", [])

        # WeightPragma handles multiple interfaces, so we apply to each one
        for interface_name in interface_names:
            interface = self.find_interface(kernel, interface_name)
            if interface is not None:
                # Check if interface supports dataflow properties
                if hasattr(interface, "supports_dataflow") and interface.supports_dataflow():
                    # Create DataflowMetadata if it doesn't exist
                    if not hasattr(interface, "dataflow") or interface.dataflow is None:
                        from brainsmith.tools.kernel_integrator.metadata import DataflowMetadata

                        interface.dataflow = DataflowMetadata()

                    # Mark interface as weight type
                    interface.dataflow.is_weight = True
                    logger.debug(f"Applied WEIGHT pragma to interface '{interface_name}'")
                else:
                    logger.warning(f"Interface '{interface_name}' does not support weight marking")
            else:
                logger.warning(f"WEIGHT pragma target interface '{interface_name}' not found")


@dataclass
class DatatypePragma(InterfacePragma):
    """Maps specific RTL parameters to interface datatype properties.

    Format: @brainsmith datatype <interface_name> <property_type> <parameter_name>

    This pragma links RTL parameters to datatype properties like width, signed, etc.
    Can be used for both interfaces and internal datatypes.

    Examples:
    - @brainsmith datatype s_axis_input0 width INPUT0_WIDTH
    - @brainsmith datatype s_axis_input0 signed SIGNED_INPUT0
    - @brainsmith datatype accumulator width ACC_WIDTH
    """

    def _parse_inputs(self) -> dict:
        pos = self.inputs["positional"]

        if len(pos) != 3:
            raise PragmaError(
                "DATATYPE pragma requires interface_name, property_type, parameter_name"
            )

        interface_name = pos[0]
        property_type = pos[1].lower()
        parameter_name = pos[2]

        # Validate property type
        valid_properties = [
            "width",
            "signed",
            "format",
            "bias",
            "fractional_width",
            "exponent_width",
            "mantissa_width",
        ]
        if property_type not in valid_properties:
            raise PragmaError(
                f"Invalid property_type '{property_type}'. Must be one of: {valid_properties}"
            )

        return {
            "interface_name": interface_name,
            "property_type": property_type,
            "parameter_name": parameter_name,
        }

    def apply_to_kernel(self, kernel: "KernelMetadata") -> None:
        """Apply DATATYPE pragma to kernel metadata - moves parameter to interface."""
        interface_name = self.parsed_data.get("interface_name")
        property_type = self.parsed_data.get("property_type")
        parameter_name = self.parsed_data.get("parameter_name")

        # Find the interface using helper method
        interface = self.find_interface(kernel, interface_name)

        if interface is None:
            # If interface not found, this might be an internal datatype
            # For now, just log a warning - internal datatypes handling can be added later if needed
            logger.warning(f"DATATYPE pragma target interface '{interface_name}' not found")
            return

        # Find and remove parameter from kernel.parameters
        param_index = None
        for i, param in enumerate(kernel.parameters):
            if param.name == parameter_name:
                param_index = i
                break

        if param_index is not None:
            # Move parameter to interface
            param = kernel.parameters.pop(param_index)

            # Store the property type in the parameter's kernel_value
            param.kernel_value = property_type

            # Create DatatypeParameters if needed
            if not hasattr(interface, "dtype_params") or interface.dtype_params is None:
                from brainsmith.tools.kernel_integrator.metadata import DatatypeParameters

                interface.dtype_params = DatatypeParameters()

            # Assign to the appropriate property
            if property_type == "width":
                interface.dtype_params.width = param
            elif property_type == "signed":
                interface.dtype_params.signed = param
            elif property_type == "format":
                interface.dtype_params.format = param
            elif property_type == "bias":
                interface.dtype_params.bias = param
            elif property_type == "fractional_width":
                interface.dtype_params.fractional_width = param
            elif property_type == "exponent_width":
                interface.dtype_params.exponent_width = param
            elif property_type == "mantissa_width":
                interface.dtype_params.mantissa_width = param

            logger.debug(
                f"Moved parameter '{param.name}' from kernel to interface '{interface_name}' dtype_params.{property_type}"
            )
        else:
            logger.warning(
                f"DATATYPE pragma references parameter '{parameter_name}' which is not in kernel.parameters"
            )

        logger.debug(f"DATATYPE pragma completed for interface '{interface_name}'")
