############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unified validation context system for constraints.

This module provides the core abstraction for validating constraints across
different contexts (ONNX inference, kernel build-time).

The ValidationContext protocol defines a unified interface for accessing
tensor/interface properties regardless of whether we're validating an ONNX
node or a kernel model.

Example usage:
    # ONNX context (inference-time)
    onnx_ctx = OnnxValidationContext(node, model)
    datatype = onnx_ctx.get_datatype("input0")

    # Kernel context (build-time)
    kernel_ctx = KernelValidationContext(kernel_instance, get_nodeattr)
    datatype = kernel_ctx.get_datatype("input0")

    # Same interface, different contexts!
"""

import logging
from typing import Any, List, Optional, Callable, Protocol, Dict
from dataclasses import dataclass, field
from enum import Enum

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
import qonnx.core.data_layout as DataLayout

from .types import ShapeHierarchy

logger = logging.getLogger(__name__)


# =============================================================================
# ValidationError (Structured Error Type)
# =============================================================================

@dataclass(frozen=True)
class ValidationError:
    """Simple validation error with context and suggestions.

    Replaces string-based error returns with structured type.
    """
    message: str
    location: str  # e.g. "input.stream[1]", "output.block[0]"
    suggestions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        msg = f"{self.location}: {self.message}"
        if self.suggestions:
            suggestions_str = ", ".join(self.suggestions)
            msg += f"\n  Suggestions: {suggestions_str}"
        return msg


# =============================================================================
# ValidationContext Protocol
# =============================================================================

class ValidationContext(Protocol):
    """Protocol for accessing tensor/interface properties across ONNX and kernel contexts.

    Implementations: OnnxValidationContext, KernelValidationContext
    Context-specific operations (is_dynamic, get_layout) gracefully degrade when not applicable.
    """

    def get_datatype(self, name: str) -> DataType:
        """Get datatype (raises KeyError if not found)."""
        ...

    def get_shape(
        self,
        name: str,
        hierarchy: ShapeHierarchy = ShapeHierarchy.TENSOR
    ) -> tuple[int, ...]:
        """Get shape at hierarchy level (ONNX supports TENSOR only)."""
        ...

    def is_dynamic(self, name: str) -> bool:
        """Check if tensor is dynamic - no initializer (always True for kernel contexts)."""
        ...

    def get_layout(self, name: str) -> Optional[Any]:
        """Get data layout - NCHW/NHWC (None for kernel contexts)."""
        ...

    def get_param(self, name: str) -> Any:
        """Get kernel parameter (raises RuntimeError on ONNX contexts, KeyError if not found)."""
        ...

    def get_interfaces(self) -> List[str]:
        """Get all interface/tensor names in scope."""
        ...

    def get_node_attribute(self, name: str, default: Any = None) -> Any:
        """Get ONNX node attribute (only available in ONNX context).

        Args:
            name: Attribute name
            default: Default value if attribute not found

        Returns:
            Attribute value or default

        Raises:
            RuntimeError: In kernel context (not applicable)
            KeyError: If attribute not found and no default provided
        """
        ...


# =============================================================================
# ONNX Validation Context
# =============================================================================

@dataclass
class OnnxValidationContext:
    """Validation context for ONNX nodes (inference-time).

    Adapts ONNX NodeProto + ModelWrapper to ValidationContext protocol.
    Used during kernel inference pattern matching to validate constraints
    on ONNX nodes before creating hardware kernels.

    Resolves schema interface names (e.g., "input0") to ONNX tensor names
    (e.g., "act_in") by searching the schema and using positional correspondence.
    This allows constraints to use self-documenting schema names while validating
    actual ONNX tensor properties.

    Example:
        # With schema (interface name resolution)
        ctx = OnnxValidationContext(add_node, model, schema)
        dt = ctx.get_datatype("input0")  # Finds schema.inputs[0].name == "input0"
                                         # Returns datatype of node.input[0]

        # Without schema (direct ONNX names)
        ctx = OnnxValidationContext(add_node, model)
        dt = ctx.get_datatype("act_in")  # Uses ONNX tensor name directly
    """

    node: NodeProto
    model: ModelWrapper
    schema: Any = None  # Optional KernelSchema for interface name resolution

    def _resolve_name(self, name: str) -> str:
        """Resolve schema interface name to ONNX tensor name by position.

        Searches schema.inputs and schema.outputs to find interface with matching
        name, then returns the corresponding node.input[i] or node.output[i].

        If no schema provided or name not found, returns name unchanged.

        Performance: O(n) where n = # interfaces, typically 3-4, so ~50ns.

        Args:
            name: Schema interface name (e.g., "input0") or direct ONNX tensor name

        Returns:
            ONNX tensor name (e.g., "act_in")
        """
        if self.schema is None:
            return name

        # Search inputs: schema.inputs[i].name → node.input[i]
        for i, inp_schema in enumerate(self.schema.inputs):
            if inp_schema.name == name and i < len(self.node.input):
                return self.node.input[i]

        # Search outputs: schema.outputs[i].name → node.output[i]
        for i, out_schema in enumerate(self.schema.outputs):
            if out_schema.name == name and i < len(self.node.output):
                return self.node.output[i]

        # Not found in schema, assume direct ONNX tensor name
        return name

    def get_datatype(self, name: str) -> DataType:
        """Get datatype from ONNX tensor."""
        tensor_name = self._resolve_name(name)
        dt = self.model.get_tensor_datatype(tensor_name)
        if dt is None:
            raise KeyError(f"Tensor '{name}' (mapped to '{tensor_name}') not found in ONNX model")
        return dt

    def get_shape(
        self,
        name: str,
        hierarchy: ShapeHierarchy = ShapeHierarchy.TENSOR
    ) -> tuple[int, ...]:
        """Get shape from ONNX tensor.

        Note:
            ONNX only has tensor shapes. Block/stream hierarchies not applicable.
            If non-TENSOR hierarchy requested, returns tensor shape (graceful degradation).
        """
        tensor_name = self._resolve_name(name)
        shape = self.model.get_tensor_shape(tensor_name)
        if shape is None:
            raise KeyError(f"Tensor '{name}' (mapped to '{tensor_name}') not found in ONNX model")

        if hierarchy != ShapeHierarchy.TENSOR:
            logger.debug(
                f"ONNX context only supports TENSOR hierarchy, "
                f"requested {hierarchy.value} for '{name}', returning tensor shape"
            )

        return tuple(shape)

    def is_dynamic(self, name: str) -> bool:
        """Check if tensor is dynamic (no initializer)."""
        tensor_name = self._resolve_name(name)
        return self.model.get_initializer(tensor_name) is None

    def get_layout(self, name: str) -> Optional[Any]:
        """Get data layout from ONNX tensor."""
        tensor_name = self._resolve_name(name)
        try:
            return self.model.get_tensor_layout(tensor_name)
        except (AttributeError, KeyError):
            return None

    def get_param(self, name: str) -> Any:
        """Get kernel parameter - NOT SUPPORTED in ONNX context.

        Raises:
            RuntimeError: ONNX context has no kernel parameters
        """
        raise RuntimeError(
            f"ONNX validation context does not support kernel parameters. "
            f"Cannot get parameter '{name}'. "
            f"Use KernelValidationContext for parameter access."
        )

    def get_interfaces(self) -> List[str]:
        """Get all tensor names (inputs + outputs)."""
        return list(self.node.input) + list(self.node.output)

    def get_node_attribute(self, name: str, default: Any = None) -> Any:
        """Get attribute from ONNX node.

        Args:
            name: Attribute name
            default: Default value if attribute not found

        Returns:
            Attribute value or default

        Raises:
            KeyError: If attribute not found and no default provided
        """
        from onnx import helper

        try:
            return helper.get_node_attr_value(self.node, name)
        except AttributeError:
            if default is not None:
                return default
            raise KeyError(f"Attribute '{name}' not found on node '{self.node.name}'")


# =============================================================================
# Kernel Validation Context
# =============================================================================

@dataclass
class KernelValidationContext:
    """Validation context for kernel models (build-time).

    Adapts KernelConfiguration + nodeattr getter to ValidationContext protocol.
    Used during kernel model building to validate constraints on
    kernel interfaces and parameters.

    Example:
        ctx = KernelValidationContext(kernel_instance, op.get_nodeattr)
        dt = ctx.get_datatype("input0")
        stream_shape = ctx.get_shape("input0", ShapeHierarchy.STREAM)
        pe = ctx.get_param("PE")
    """

    kernel_instance: Any  # KernelInstance (avoid circular import)
    param_getter: Callable[[str], Any]

    def get_datatype(self, name: str) -> DataType:
        """Get datatype from kernel interface."""
        # Try inputs first
        if name in self.kernel_instance.inputs:
            return self.kernel_instance.inputs[name].datatype
        # Try outputs
        if name in self.kernel_instance.outputs:
            return self.kernel_instance.outputs[name].datatype
        # Not found - helpful error
        available = (
            list(self.kernel_instance.inputs.keys()) +
            list(self.kernel_instance.outputs.keys())
        )
        raise KeyError(
            f"Interface '{name}' not found in kernel model. "
            f"Available: {', '.join(available)}"
        )

    def get_shape(
        self,
        name: str,
        hierarchy: ShapeHierarchy = ShapeHierarchy.TENSOR
    ) -> tuple[int, ...]:
        """Get shape from kernel interface at specified hierarchy."""
        # Try inputs first
        if name in self.kernel_instance.inputs:
            return tuple(self.kernel_instance.inputs[name].get_shape(hierarchy))
        # Try outputs
        if name in self.kernel_instance.outputs:
            return tuple(self.kernel_instance.outputs[name].get_shape(hierarchy))
        # Not found - helpful error
        available = (
            list(self.kernel_instance.inputs.keys()) +
            list(self.kernel_instance.outputs.keys())
        )
        raise KeyError(
            f"Interface '{name}' not found in kernel model. "
            f"Available: {', '.join(available)}"
        )

    def is_dynamic(self, name: str) -> bool:
        """Check if interface is dynamic (not a weight).

        For inputs, returns NOT is_weight (inferred from ONNX initializers).
        For outputs and internals, always returns True (no weights).

        Returns:
            True if dynamic (activations), False if static (weights)
        """
        # Try inputs first
        if name in self.kernel_instance.inputs:
            return not self.kernel_instance.inputs[name].is_weight
        # Try outputs (always dynamic)
        if name in self.kernel_instance.outputs:
            return True
        # Not found - assume dynamic
        return True

    def get_layout(self, name: str) -> Optional[Any]:
        """Get data layout - NOT TRACKED in kernel context.

        Note:
            Kernel models don't track layout (layout handled during inference).
            Always returns None. This allows layout-specific constraints like
            HasLayout to gracefully pass in kernel contexts (legitimate
            graceful degradation - layout info genuinely doesn't exist).
        """
        return None

    def get_param(self, name: str) -> Any:
        """Get kernel parameter via nodeattr getter."""
        try:
            return self.param_getter(name)
        except (AttributeError, KeyError) as e:
            raise KeyError(f"Parameter '{name}' not found in nodeattrs") from e

    def get_interfaces(self) -> List[str]:
        """Get all interface names (inputs + outputs)."""
        return list(self.kernel_instance.inputs.keys()) + list(self.kernel_instance.outputs.keys())

    def get_node_attribute(self, name: str, default: Any = None) -> Any:
        """Get ONNX node attribute - NOT AVAILABLE in kernel context.

        Node attributes are only available during ONNX inference validation.
        In kernel build context, the node is already converted to a hardware kernel.

        Args:
            name: Attribute name
            default: Default value (ignored - always raises RuntimeError)

        Raises:
            RuntimeError: Always (node attributes don't exist in kernel context)
        """
        raise RuntimeError(
            f"Node attributes not available in kernel build context. "
            f"Cannot get attribute '{name}'. "
            f"Node attribute constraints are only checked during ONNX inference."
        )


# =============================================================================
# Design Space Validation Context (Two-Phase Construction)
# =============================================================================

@dataclass
class DesignSpaceValidationContext:
    """Validation context for KernelDesignSpace (no stream shapes yet).

    Used to validate structural constraints before stream shape resolution.
    Adapts lists of InterfaceDesignSpace to ValidationContext protocol.

    Builds dict cache from lists for O(1) lookup. This context is used during
    design space construction (build()) to validate constraints that don't
    depend on stream shapes (e.g., datatype constraints, tensor/block shape
    constraints). STREAM hierarchy queries will fail since stream shapes are
    not resolved yet.

    Example:
        ctx = DesignSpaceValidationContext(
            inputs=[inv_input1, inv_input2],
            outputs=[inv_output],
            internal_datatypes={"accumulator": DataType["INT32"]},
            param_getter=get_nodeattr,
        )
        dt = ctx.get_datatype("input0")  # Works
        tensor_shape = ctx.get_shape("input0", ShapeHierarchy.TENSOR)  # Works
        block_shape = ctx.get_shape("input0", ShapeHierarchy.BLOCK)  # Works
        stream_shape = ctx.get_shape("input0", ShapeHierarchy.STREAM)  # RuntimeError
    """
    inputs: Dict[str, Any]  # Dict[str, InterfaceDesignSpace]
    outputs: Dict[str, Any]  # Dict[str, InterfaceDesignSpace]
    internal_datatypes: Dict[str, DataType]
    param_getter: Callable[[str], Any]

    def get_datatype(self, name: str) -> DataType:
        """Get datatype from design space interface."""
        if name in self.inputs:
            return self.inputs[name].datatype
        if name in self.outputs:
            return self.outputs[name].datatype
        available = list(self.inputs.keys()) + list(self.outputs.keys())
        raise KeyError(
            f"Interface '{name}' not found in design space. "
            f"Available: {', '.join(available)}"
        )

    def get_shape(
        self,
        name: str,
        hierarchy: ShapeHierarchy = ShapeHierarchy.TENSOR
    ) -> tuple[int, ...]:
        """Get shape at hierarchy level (STREAM not available).

        Args:
            name: Interface name
            hierarchy: Shape hierarchy level

        Returns:
            Shape tuple for requested hierarchy

        Raises:
            KeyError: If interface not found
            RuntimeError: If STREAM hierarchy requested (not available yet)
            ValueError: If unknown hierarchy
        """
        # Find interface
        if name in self.inputs:
            interface = self.inputs[name]
        elif name in self.outputs:
            interface = self.outputs[name]
        else:
            available = list(self.inputs.keys()) + list(self.outputs.keys())
            raise KeyError(
                f"Interface '{name}' not found in design space. "
                f"Available: {', '.join(available)}"
            )

        # Return requested hierarchy
        if hierarchy == ShapeHierarchy.STREAM:
            raise RuntimeError(
                "Stream shapes not available in design space validation context. "
                "Stream-level constraints are parametric constraints and should be "
                "validated during configure(), not during build()."
            )
        elif hierarchy == ShapeHierarchy.BLOCK:
            return interface.block_shape
        elif hierarchy == ShapeHierarchy.TENSOR:
            return interface.tensor_shape
        else:
            raise ValueError(f"Unknown hierarchy: {hierarchy}")

    def is_dynamic(self, name: str) -> bool:
        """Check if interface is dynamic (not a weight).

        For inputs, returns NOT is_weight (inferred from ONNX initializers).
        For outputs, always returns True (no weights).

        Returns:
            True if dynamic (activations), False if static (weights)
        """
        if name in self.inputs:
            return not self.inputs[name].is_weight
        if name in self.outputs:
            return True  # Outputs always dynamic
        # Not found - assume dynamic
        return True

    def get_layout(self, name: str) -> Optional[Any]:
        """Get data layout - NOT TRACKED in kernel context.

        Note:
            Kernel models don't track layout (layout handled during inference).
            Always returns None. This allows layout-specific constraints like
            HasLayout to gracefully pass in kernel contexts.
        """
        return None

    def get_param(self, name: str) -> Any:
        """Get kernel parameter via param_getter.

        Args:
            name: Parameter name

        Returns:
            Parameter value

        Raises:
            KeyError: If parameter not found
        """
        try:
            return self.param_getter(name)
        except (AttributeError, KeyError) as e:
            raise KeyError(f"Parameter '{name}' not found in nodeattrs") from e

    def get_interfaces(self) -> List[str]:
        """Get all interface names."""
        return list(self.inputs.keys()) + list(self.outputs.keys())

    def get_node_attribute(self, name: str, default: Any = None) -> Any:
        """Get ONNX node attribute - NOT AVAILABLE in design space context.

        Node attributes are only available during ONNX inference validation.
        In design space build context, the node is already being converted.

        Args:
            name: Attribute name
            default: Default value (ignored - always raises RuntimeError)

        Raises:
            RuntimeError: Always (node attributes don't exist in design space context)
        """
        raise RuntimeError(
            f"Node attributes not available in design space validation context. "
            f"Cannot get attribute '{name}'. "
            f"Node attribute constraints are only checked during ONNX inference."
        )


# =============================================================================
# Configured Validation Context (Two-Phase Construction)
# =============================================================================

@dataclass
class ConfigurationValidationContext:
    """Validation context for KernelInstance (two-phase construction).

    Adapts KernelInstance to ValidationContext protocol.
    Similar to KernelValidationContext but works with InterfaceConfiguration
    which uses a flyweight pattern (references design space + resolved stream_shape).

    Used during KernelDesignSpace.configure() to validate parametric constraints
    after stream shapes are resolved.

    Example:
        ctx = ConfigurationValidationContext(configured_model, {"SIMD": 64})
        dt = ctx.get_datatype("input0")
        stream_shape = ctx.get_shape("input0", ShapeHierarchy.STREAM)
        simd = ctx.get_param("SIMD")
    """
    configured_model: Any  # KernelInstance (avoid circular import)
    param_getter_dict: Dict[str, Any]  # Parallelization params

    def get_datatype(self, name: str) -> DataType:
        """Get datatype from configured interface."""
        # Try inputs first
        if name in self.configured_model.inputs:
            return self.configured_model.inputs[name].datatype
        # Try outputs
        if name in self.configured_model.outputs:
            return self.configured_model.outputs[name].datatype
        # Not found - helpful error
        available = (
            list(self.configured_model.inputs.keys()) +
            list(self.configured_model.outputs.keys())
        )
        raise KeyError(
            f"Interface '{name}' not found in configured model. "
            f"Available: {', '.join(available)}"
        )

    def get_shape(
        self,
        name: str,
        hierarchy: ShapeHierarchy = ShapeHierarchy.TENSOR
    ) -> tuple[int, ...]:
        """Get shape from configured interface at hierarchy.

        InterfaceConfiguration supports all hierarchies:
        - TENSOR: from design_space.tensor_shape
        - BLOCK: from design_space.block_shape
        - STREAM: from stream_shape (resolved)
        """
        # Try inputs first
        if name in self.configured_model.inputs:
            return self.configured_model.inputs[name].get_shape(hierarchy)
        # Try outputs
        if name in self.configured_model.outputs:
            return self.configured_model.outputs[name].get_shape(hierarchy)
        # Not found - helpful error
        available = (
            list(self.configured_model.inputs.keys()) +
            list(self.configured_model.outputs.keys())
        )
        raise KeyError(
            f"Interface '{name}' not found in configured model. "
            f"Available: {', '.join(available)}"
        )

    def is_dynamic(self, name: str) -> bool:
        """Check if interface is dynamic (not a weight).

        For inputs, returns NOT is_weight (inferred from ONNX initializers).
        For outputs, always returns True (no weights).

        Returns:
            True if dynamic (activations), False if static (weights)
        """
        # Try inputs first
        if name in self.configured_model.inputs:
            return not self.configured_model.inputs[name].is_weight
        # Try outputs (always dynamic)
        if name in self.configured_model.outputs:
            return True
        # Not found - assume dynamic
        return True

    def get_layout(self, name: str) -> Optional[Any]:
        """Get data layout - NOT TRACKED in kernel context.

        Note:
            Kernel models don't track layout (layout handled during inference).
            Always returns None. This allows layout-specific constraints like
            HasLayout to gracefully pass in kernel contexts.
        """
        return None

    def get_param(self, name: str) -> Any:
        """Get parallelization parameter from configuration.

        Args:
            name: Parameter name (e.g., "SIMD", "PE")

        Returns:
            Parameter value

        Raises:
            KeyError: If parameter not found in configuration
        """
        if name not in self.param_getter_dict:
            raise KeyError(f"Parameter '{name}' not found in configuration")
        return self.param_getter_dict[name]

    def get_interfaces(self) -> List[str]:
        """Get all interface names (inputs + outputs)."""
        return (
            list(self.configured_model.inputs.keys()) +
            list(self.configured_model.outputs.keys())
        )

    def get_node_attribute(self, name: str, default: Any = None) -> Any:
        """Get ONNX node attribute - NOT AVAILABLE in configured context.

        Node attributes are only available during ONNX inference validation.
        In configured kernel context, the node is already converted and configured.

        Args:
            name: Attribute name
            default: Default value (ignored - always raises RuntimeError)

        Raises:
            RuntimeError: Always (node attributes don't exist in configured context)
        """
        raise RuntimeError(
            f"Node attributes not available in configured kernel context. "
            f"Cannot get attribute '{name}'. "
            f"Node attribute constraints are only checked during ONNX inference."
        )


__all__ = [
    'ValidationError',
    'ValidationContext',
    'OnnxValidationContext',
    'KernelValidationContext',
    'DesignSpaceValidationContext',
    'ConfigurationValidationContext',
]
