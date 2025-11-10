############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Validation contexts for constraint checking in dataflow system.

This module provides focused validation contexts for the two phases of
kernel construction:

1. **Design Space Validation** (structural constraints during build())
   - Uses DesignSpaceValidationContext
   - Has tensor shapes, block shapes, datatypes
   - No stream shapes yet (those are optimization parameters)

2. **Configuration Validation** (optimization constraints during configure())
   - Uses ConfigurationValidationContext
   - Has all shape hierarchies (tensor, block, stream)
   - Has specific parallelization parameters

Example usage (Design Space):
    ctx = DesignSpaceValidationContext(
        inputs=interfaces_input,
        outputs=interfaces_output,
        internal_datatypes=internal_datatypes,
        param_getter=get_nodeattr
    )
    for constraint in structural_constraints:
        if error := constraint.check(ctx):
            raise ValueError(error)

Example usage (Configuration):
    ctx = ConfigurationValidationContext(
        configured_model=instance,
        params=params
    )
    for constraint in parametric_constraints:
        if error := constraint.check(ctx):
            raise ValueError(error)
"""

import logging
from typing import Any, List, Optional, Callable, Dict
from dataclasses import dataclass, field

from qonnx.core.datatype import DataType

from .types import ShapeHierarchy

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def _find_interface_in_dicts(name: str, inputs: Dict, outputs: Dict) -> Any:
    """Find interface by name from input/output dicts.

    Args:
        name: Interface name
        inputs: Input interfaces dict
        outputs: Output interfaces dict

    Returns:
        Interface object (InterfaceDesignSpace or InterfaceDesignPoint)

    Raises:
        KeyError: If interface not found
    """
    if name in inputs:
        return inputs[name]
    if name in outputs:
        return outputs[name]

    available = list(inputs.keys()) + list(outputs.keys())
    available_str = ", ".join(available) if available else "none"
    raise KeyError(f"Interface '{name}' not found. Available: {available_str}")


# =============================================================================
# ValidationError (Structured Error Type)
# =============================================================================

class ValidationError(ValueError):
    """Validation error with context and suggestions.

    Args:
        message: Error message
        location: Optional context (e.g., "input.stream[1]")
        suggestions: Optional list of suggestions

    Examples:
        >>> raise ValidationError("Invalid parameter")
        >>> raise ValidationError("PE must divide 768", location="output.stream[1]")
    """

    def __init__(self, message: str, location: str = "", suggestions: list = None):
        self.message = message
        self.location = location
        self.suggestions = suggestions or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with location and suggestions."""
        if self.location:
            msg = f"{self.location}: {self.message}"
        else:
            msg = self.message

        if self.suggestions:
            suggestions_str = ", ".join(self.suggestions)
            msg += f"\n  Suggestions: {suggestions_str}"

        return msg

    def __str__(self) -> str:
        """Return formatted message (delegates to parent Exception)."""
        return super().__str__()


# =============================================================================
# Design Space Validation Context (Structural Constraints)
# =============================================================================

@dataclass
class DesignSpaceValidationContext:
    """Validation context for structural constraints during design space build.

    Used during KernelDesignSpace construction to validate tensor shapes,
    block shapes, and datatypes. Stream shapes not available until configure().

    Attributes:
        inputs: Input interfaces (InterfaceDesignSpace)
        outputs: Output interfaces (InterfaceDesignSpace)
        internal_datatypes: Internal datatypes
        param_getter: Optional nodeattr getter

    Example:
        ctx = DesignSpaceValidationContext(
            inputs=interfaces_input,
            outputs=interfaces_output,
            internal_datatypes=internal_datatypes,
            param_getter=get_nodeattr
        )
        for constraint in structural_constraints:
            if error := constraint.check(ctx):
                raise ValueError(error)
    """

    inputs: Dict[str, Any]  # InterfaceDesignSpace
    outputs: Dict[str, Any]  # InterfaceDesignSpace
    internal_datatypes: Dict[str, DataType]
    param_getter: Optional[Callable[[str], Any]] = None

    def _find_interface(self, name: str) -> Any:
        """Find interface from inputs or outputs."""
        return _find_interface_in_dicts(name, self.inputs, self.outputs)

    def get_datatype(self, name: str) -> DataType:
        """Get datatype from interface or internal datatypes.

        Args:
            name: Interface or internal datatype name

        Returns:
            DataType

        Raises:
            KeyError: If interface/datatype not found
        """
        # Check internal datatypes first
        if name in self.internal_datatypes:
            return self.internal_datatypes[name]

        # Check interfaces
        return self._find_interface(name).datatype

    def get_shape(
        self,
        name: str,
        hierarchy: ShapeHierarchy = ShapeHierarchy.TENSOR
    ) -> tuple[int, ...]:
        """Get shape at hierarchy level.

        Args:
            name: Interface name
            hierarchy: Which level of hierarchy (TENSOR or BLOCK only)

        Returns:
            Shape tuple

        Raises:
            KeyError: If interface not found
            RuntimeError: If STREAM hierarchy requested (not available in design space)
        """
        if hierarchy == ShapeHierarchy.STREAM:
            raise RuntimeError(
                "Stream shapes not available in design space validation context. "
                "Stream-level constraints are optimization constraints and should be "
                "validated during configure(), not during build()."
            )

        interface = self._find_interface(name)

        if hierarchy == ShapeHierarchy.BLOCK:
            return interface.block_shape
        elif hierarchy == ShapeHierarchy.TENSOR:
            return interface.tensor_shape
        else:
            raise ValueError(f"Unknown hierarchy: {hierarchy}")

    def is_dynamic(self, name: str) -> bool:
        """Check if interface is dynamic (no initializer).

        Args:
            name: Interface name

        Returns:
            True if dynamic (activations), False if static (weights)
        """
        if name in self.inputs:
            return not self.inputs[name].is_weight

        if name in self.outputs:
            return True  # Outputs always dynamic

        # Not found
        raise KeyError(f"Interface '{name}' not found")

    def get_param(self, name: str) -> Any:
        """Get kernel parameter value.

        Note: Primarily for rare block_tiling parameters. Most parameters
        are stream_tiling and only available during configure().

        Args:
            name: Parameter name

        Returns:
            Parameter value

        Raises:
            RuntimeError: If no param_getter provided
            KeyError: If parameter not found
        """
        if self.param_getter is None:
            raise RuntimeError(
                f"No param_getter available. Cannot get parameter '{name}'."
            )

        try:
            return self.param_getter(name)
        except (AttributeError, KeyError) as e:
            raise KeyError(f"Parameter '{name}' not found in nodeattrs") from e


# =============================================================================
# Configuration Validation Context (Parametric Constraints)
# =============================================================================

@dataclass
class ConfigurationValidationContext:
    """Validation context for optimization constraints during configure().

    Used during KernelDesignSpace.configure() to validate constraints on
    stream shapes and parallelization parameters.

    Attributes:
        configured_model: KernelDesignPoint with configured interfaces
        params: Parallelization parameters

    Example:
        ctx = ConfigurationValidationContext(
            configured_model=instance,
            params=params
        )
        for constraint in parametric_constraints:
            if error := constraint.check(ctx):
                raise ValueError(error)
    """

    configured_model: Any  # KernelDesignPoint
    params: Dict[str, int]

    def _find_interface(self, name: str) -> Any:
        """Find interface from configured model."""
        return _find_interface_in_dicts(
            name,
            self.configured_model.inputs,
            self.configured_model.outputs
        )

    def get_datatype(self, name: str) -> DataType:
        """Get datatype from interface or internal datatypes.

        Args:
            name: Interface or internal datatype name

        Returns:
            DataType

        Raises:
            KeyError: If interface/datatype not found
        """
        # Check internal datatypes
        if name in self.configured_model.internal_datatypes:
            return self.configured_model.internal_datatypes[name]

        # Check interfaces
        return self._find_interface(name).datatype

    def get_shape(
        self,
        name: str,
        hierarchy: ShapeHierarchy = ShapeHierarchy.TENSOR
    ) -> tuple[int, ...]:
        """Get shape at hierarchy level.

        All hierarchies available (TENSOR, BLOCK, STREAM).

        Args:
            name: Interface name
            hierarchy: Which level of hierarchy

        Returns:
            Shape tuple

        Raises:
            KeyError: If interface not found
        """
        interface = self._find_interface(name)
        return interface.get_shape(hierarchy)

    def is_dynamic(self, name: str) -> bool:
        """Check if interface is dynamic (no initializer).

        Args:
            name: Interface name

        Returns:
            True if dynamic (activations), False if static (weights)
        """
        if name in self.configured_model.inputs:
            return not self.configured_model.inputs[name].is_weight

        if name in self.configured_model.outputs:
            return True  # Outputs always dynamic

        # Not found
        raise KeyError(f"Interface '{name}' not found")

    def get_param(self, name: str) -> Any:
        """Get kernel parameter value.

        Args:
            name: Parameter name (e.g., "SIMD", "PE", "epsilon")

        Returns:
            Parameter value

        Raises:
            KeyError: If parameter not found
        """
        if name not in self.params:
            raise KeyError(
                f"Parameter '{name}' not found in configuration. "
                f"Available: {list(self.params.keys())}"
            )
        return self.params[name]


__all__ = [
    'ValidationError',
    'DesignSpaceValidationContext',
    'ConfigurationValidationContext',
]
