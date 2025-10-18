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
    kernel_ctx = KernelValidationContext(kernel_model, get_nodeattr)
    datatype = kernel_ctx.get_datatype("input0")

    # Same interface, different contexts!
"""

import logging
from typing import Any, List, Optional, Callable, Protocol
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


# =============================================================================
# ONNX Validation Context
# =============================================================================

@dataclass
class OnnxValidationContext:
    """Validation context for ONNX nodes (inference-time).

    Adapts ONNX NodeProto + ModelWrapper to ValidationContext protocol.
    Used during kernel inference pattern matching to validate constraints
    on ONNX nodes before creating hardware kernels.

    Maps schema interface names (e.g., "input", "input0", "output") to actual
    ONNX tensor names using the node's input/output lists.

    Example:
        # With schema mapping
        ctx = OnnxValidationContext(add_node, model, schema)
        dt = ctx.get_datatype("input0")  # Maps to node.input[0]

        # Without schema (direct ONNX names)
        ctx = OnnxValidationContext(add_node, model)
        dt = ctx.get_datatype(node.input[0])  # Uses ONNX tensor name directly
    """

    node: NodeProto
    model: ModelWrapper
    schema: Any = None  # Optional KernelSchema for interface name mapping

    def __post_init__(self):
        """Build mapping from schema interface names to ONNX tensor names."""
        self._name_map = {}
        if self.schema is not None:
            # Map input interfaces to ONNX tensors
            for i, inp_schema in enumerate(self.schema.inputs):
                if i < len(self.node.input):
                    self._name_map[inp_schema.name] = self.node.input[i]
            # Map output interfaces to ONNX tensors
            for i, out_schema in enumerate(self.schema.outputs):
                if i < len(self.node.output):
                    self._name_map[out_schema.name] = self.node.output[i]

    def _resolve_name(self, name: str) -> str:
        """Resolve schema interface name to ONNX tensor name."""
        return self._name_map.get(name, name)

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


# =============================================================================
# Kernel Validation Context
# =============================================================================

@dataclass
class KernelValidationContext:
    """Validation context for kernel models (build-time).

    Adapts KernelModel + nodeattr getter to ValidationContext protocol.
    Used during kernel model building to validate constraints on
    kernel interfaces and parameters.

    Example:
        ctx = KernelValidationContext(kernel_model, op.get_nodeattr)
        dt = ctx.get_datatype("input0")
        stream_shape = ctx.get_shape("input0", ShapeHierarchy.STREAM)
        pe = ctx.get_param("PE")
    """

    kernel_model: Any  # KernelModel (avoid circular import)
    param_getter: Callable[[str], Any]

    def get_datatype(self, name: str) -> DataType:
        """Get datatype from kernel interface."""
        try:
            interface = self.kernel_model.get_interface(name)
            return interface.datatype
        except (AttributeError, KeyError) as e:
            raise KeyError(f"Interface '{name}' not found in kernel model") from e

    def get_shape(
        self,
        name: str,
        hierarchy: ShapeHierarchy = ShapeHierarchy.TENSOR
    ) -> tuple[int, ...]:
        """Get shape from kernel interface at specified hierarchy."""
        try:
            interface = self.kernel_model.get_interface(name)
            shape = interface.get_shape(hierarchy)
            return tuple(shape)
        except (AttributeError, KeyError) as e:
            raise KeyError(f"Interface '{name}' not found in kernel model") from e

    def is_dynamic(self, name: str) -> bool:
        """Check if interface is dynamic (not a weight).

        For inputs, returns NOT is_weight (inferred from ONNX initializers).
        For outputs and internals, always returns True (no weights).

        Returns:
            True if dynamic (activations), False if static (weights)
        """
        try:
            interface = self.kernel_model.get_interface(name)
            # Inputs have is_weight flag (inferred from ONNX)
            if hasattr(interface, 'is_weight'):
                return not interface.is_weight
            # Outputs and internals are always dynamic
            return True
        except (AttributeError, KeyError):
            # Interface not found - assume dynamic
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
        input_names = [inp.name for inp in self.kernel_model.inputs]
        output_names = [out.name for out in self.kernel_model.outputs]
        return input_names + output_names


__all__ = [
    'ValidationError',
    'ValidationContext',
    'OnnxValidationContext',
    'KernelValidationContext',
]
