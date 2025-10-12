############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Immutable kernel models for the dataflow system.

These models represent runtime instances created from schemas. They are
immutable snapshots that reflect the current state of nodeattrs at creation
time. Models are cached and refreshed when nodeattrs change.

Key principles:
- All models are immutable (frozen dataclasses)
- Models are cached for performance
- Refresh cached models when nodeattrs change
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from abc import ABC
import math

from .types import Shape, ShapeHierarchy, prod
from qonnx.core.datatype import BaseDataType


@dataclass(frozen=True)
class InterfaceModel(ABC):
    """Base class for input/output interface models.

    Provides common fields and methods for all interface model types.

    Shape hierarchy:
    - tensor_shape: Full logical tensor dimensions
    - block_shape: Block tiling dimensions
    - stream_shape: Streaming dimensions per cycle
    """

    name: str
    tensor_shape: Shape
    block_shape: Shape
    datatype: BaseDataType
    stream_shape: Tuple[Optional[int], ...]

    @property
    def tensor_folding_factor(self) -> int:
        """Number of blocks to cover full tensor (tensor_shape / block_shape)."""
        num_blocks = 1
        for t, b in zip(self.tensor_shape, self.block_shape):
            num_blocks *= math.ceil(t / b)
        return num_blocks

    @property
    def block_folding_factor(self) -> int:
        """Cycles to stream one block (block_shape / stream_shape)."""
        cycles = 1
        for b, s in zip(self.block_shape, self.stream_shape):
            cycles *= math.ceil(b / s)
        return cycles

    @property
    def streaming_bandwidth(self) -> int:
        """Elements streamed per cycle."""
        return prod(self.stream_shape)

    @property
    def stream_width_bits(self) -> int:
        """Stream width in bits."""
        return self.streaming_bandwidth * self.datatype.bitwidth()

    def get_shape(self, hierarchy: ShapeHierarchy) -> Tuple[Optional[int], ...]:
        """Get shape at specified hierarchy level.

        Args:
            hierarchy: Which level of the shape hierarchy to retrieve

        Returns:
            Shape at the specified level

        Raises:
            ValueError: If hierarchy is invalid
        """
        if hierarchy == ShapeHierarchy.STREAM:
            return self.stream_shape
        elif hierarchy == ShapeHierarchy.BLOCK:
            return self.block_shape
        elif hierarchy == ShapeHierarchy.TENSOR:
            return self.tensor_shape
        else:
            raise ValueError(f"Invalid hierarchy: {hierarchy}")


@dataclass(frozen=True)
class InputModel(InterfaceModel):
    """Immutable input interface model.

    Represents a concrete input with resolved dimensions and datatypes.
    """

    is_weight: bool = False

    @property
    def initiation_interval(self) -> int:
        """Cycles to stream entire tensor."""
        return self.tensor_folding_factor * self.block_folding_factor


@dataclass(frozen=True)
class OutputModel(InterfaceModel):
    """Immutable output interface model.

    Pure specification of output tensor characteristics.
    Stream shape may contain None values during construction (resolved by KernelModel).
    """

    def has_unset_dims(self) -> bool:
        """Check if any stream dimensions are unset (None)."""
        return any(d is None for d in self.stream_shape)

    @property
    def block_folding_factor(self) -> int:
        """Cycles to stream one block (block_shape / stream_shape)."""
        if self.has_unset_dims():
            raise ValueError(
                f"Cannot compute block_folding_factor for '{self.name}' with unset stream dimensions"
            )
        return super().block_folding_factor

    @property
    def streaming_bandwidth(self) -> int:
        """Elements streamed per cycle."""
        if self.has_unset_dims():
            raise ValueError(
                f"Cannot compute streaming_bandwidth for '{self.name}' with unset stream dimensions"
            )
        return super().streaming_bandwidth

    @property
    def stream_width_bits(self) -> int:
        """Stream width in bits."""
        if self.has_unset_dims():
            raise ValueError(
                f"Cannot compute stream_width_bits for '{self.name}' with unset stream dimensions"
            )
        return super().stream_width_bits


@dataclass(frozen=True)
class KernelModel:
    """Immutable kernel model representing a configured kernel instance.

    This model is a snapshot of a kernel's configuration at a point in time.
    Models are cached and refreshed when nodeattrs change.

    Resolution flow in __post_init__:
    1. Derive any remaining unset output dimensions (default logic)
    """

    # Kernel identity
    name: str

    # Interface models
    inputs: Tuple[InputModel, ...]
    outputs: Tuple[OutputModel, ...]

    # Performance characteristics
    latency_cycles: Tuple[int, int] = (1, 1)
    pipeline_depth: int = 1
    clock_freq_mhz: float = 100.0

    def __post_init__(self):
        """Resolve output dimensions."""
        # Convert lists to tuples for immutability
        if isinstance(self.inputs, list):
            object.__setattr__(self, 'inputs', tuple(self.inputs))
        if isinstance(self.outputs, list):
            object.__setattr__(self, 'outputs', tuple(self.outputs))

        # Derive any remaining unset output dimensions
        resolved_outputs = self._derive_unset_dimensions(list(self.outputs))
        object.__setattr__(self, 'outputs', tuple(resolved_outputs))

    def _derive_unset_dimensions(self, outputs: List[OutputModel]) -> List[OutputModel]:
        """Fill in unset dimensions using default derivation logic.

        Default: max(input block_folding_factors) along last dimension.
        This preserves current KernelModel.output_stream_shape() behavior.

        Args:
            outputs: List of OutputModels (may have unset dimensions)

        Returns:
            List of OutputModels with all dimensions set
        """
        if not self.inputs:
            default_value = 1
        else:
            default_value = max(inp.block_folding_factor for inp in self.inputs)

        resolved = []
        for output in outputs:
            if output.has_unset_dims():
                stream_shape = list(output.stream_shape)
                for i, dim in enumerate(stream_shape):
                    if dim is None:
                        # Default: stream along last dimension
                        stream_shape[i] = default_value if i == len(stream_shape) - 1 else 1

                output = OutputModel(
                    name=output.name,
                    tensor_shape=output.tensor_shape,
                    block_shape=output.block_shape,
                    stream_shape=tuple(stream_shape),
                    datatype=output.datatype
                )

            resolved.append(output)

        return resolved

    def get_input(self, name: str) -> Optional[InputModel]:
        """Get input model by name."""
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return None

    def get_output(self, name: str) -> Optional[OutputModel]:
        """Get output model by name."""
        for out in self.outputs:
            if out.name == name:
                return out
        return None
    
    @property
    def initiation_interval(self) -> int:
        """Kernel initiation interval in cycles."""
        if not self.inputs:
            return 1
        return max(inp.initiation_interval for inp in self.inputs)

    @property
    def max_block_folding_factor(self) -> int:
        """Maximum block folding factor across all inputs."""
        if not self.inputs:
            return 1
        return max(inp.block_folding_factor for inp in self.inputs)

    @property
    def max_tensor_folding_factor(self) -> int:
        """Maximum tensor folding factor across all inputs."""
        if not self.inputs:
            return 1
        return max(inp.tensor_folding_factor for inp in self.inputs)

    @property
    def total_output_values(self) -> int:
        """Total output values across all outputs."""
        return sum(prod(out.tensor_shape) for out in self.outputs)

    def output_streaming_rate(self, output_idx: int = 0) -> int:
        """Streaming rate for output (elements per cycle).

        Derived from the slowest input's block folding factor.
        """
        if not self.inputs:
            return 1
        return max(inp.block_folding_factor for inp in self.inputs)

    def output_stream_width_bits(self, output_idx: int = 0) -> int:
        """Stream width in bits for output.

        Returns the actual stream width based on the output's stream_shape,
        not derived from input block folding factors.
        """
        output = self.outputs[output_idx]
        return output.stream_width_bits

    def output_stream_shape(self, output_idx: int = 0) -> Shape:
        """Stream shape for output.

        Now simply returns the OutputModel's stream_shape attribute
        (resolved during construction).
        """
        return self.outputs[output_idx].stream_shape

    # =========================================================================
    # Unified Accessor Methods
    # =========================================================================

    def input_datatype(self, ind: int = 0) -> BaseDataType:
        """Get input datatype."""
        return self.inputs[ind].datatype

    def output_datatype(self, ind: int = 0) -> BaseDataType:
        """Get output datatype."""
        return self.outputs[ind].datatype

    def input_tensor_shape(self, ind: int = 0) -> Shape:
        """Get input tensor shape."""
        return self.inputs[ind].tensor_shape

    def output_tensor_shape(self, ind: int = 0) -> Shape:
        """Get output tensor shape."""
        return self.outputs[ind].tensor_shape

    def input_block_shape(self, ind: int = 0) -> Shape:
        """Get input block shape."""
        return self.inputs[ind].block_shape

    def output_block_shape(self, ind: int = 0) -> Shape:
        """Get output block shape."""
        return self.outputs[ind].block_shape

    def input_stream_shape(self, ind: int = 0) -> Shape:
        """Get input stream shape."""
        return self.inputs[ind].stream_shape

    def input_stream_width_bits(self, ind: int = 0) -> int:
        """Get input stream width in bits."""
        return self.inputs[ind].stream_width_bits

    @property
    def throughput_fps(self) -> float:
        """Throughput in inferences per second."""
        cycles_per_inf = self.initiation_interval
        clock_hz = self.clock_freq_mhz * 1e6
        return clock_hz / cycles_per_inf


