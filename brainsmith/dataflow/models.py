############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Immutable kernel models for the dataflow system.

These models represent runtime instances built by KernelModelBuilder from:
- ModelWrapper context (tensor shapes, datatypes from ONNX graph)
- KernelSchema (structural definition, tiling templates)
- Nodeattrs (user parameters like SIMD, PE)

Key principles:
- All models are immutable (frozen dataclasses)
- Builder constructs fully resolved models (no post-processing)
- Models are cached, rebuilt on context/parameter changes
- Shapes are NEVER stored in nodeattrs (extracted from context)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from abc import ABC
import math

from .types import Shape, ShapeHierarchy, prod
from qonnx.core.datatype import BaseDataType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InternalDatatypeModel:
    """Minimal interface model for internal datatypes.

    Internal datatypes (e.g., accumulator) represent intermediate computation
    datatypes that don't correspond to ONNX tensors, so they have no associated
    tensor shape.

    This provides the minimal interface (.datatype) needed for DatatypeSource
    derivation patterns to reference internal datatypes during model building.

    Unlike InputModel/OutputModel which have full shape hierarchies, internal
    datatypes are datatype-only since they represent scalar computation values
    (e.g., accumulator precision) rather than tensors.
    """
    datatype: BaseDataType


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
    stream_shape: Shape
    datatype: BaseDataType

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

    Represents a concrete output with fully resolved dimensions and datatypes.
    All dimensions are resolved by KernelModelBuilder before construction.
    """
    pass


@dataclass(frozen=True)
class KernelModel:
    """Immutable kernel model representing a configured kernel instance.

    This model is a runtime snapshot built by KernelModelBuilder from:
    - ModelWrapper context (tensor shapes, datatypes from ONNX graph)
    - KernelSchema (structural definition, tiling templates)
    - Nodeattrs (user parameters like SIMD, PE)

    All dimensions are fully resolved by the builder before model construction.
    Models are cached and rebuilt when:
    - User parameters change (via set_nodeattr())
    - Context updates (new call to get_kernel_model(ctx))
    """

    # Kernel identity
    name: str

    # Interface models (fully resolved)
    inputs: Tuple[InputModel, ...]
    outputs: Tuple[OutputModel, ...]

    # Performance characteristics
    latency_cycles: Tuple[int, int] = (1, 1)
    pipeline_depth: int = 1
    clock_freq_mhz: float = 100.0

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

    def get_interface(self, name: str) -> Optional[Union[InputModel, OutputModel]]:
        """Get interface model by name (searches inputs then outputs)."""
        # Try inputs first
        interface = self.get_input(name)
        if interface is not None:
            return interface
        # Try outputs
        return self.get_output(name)

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

    def output_stream_width_bits(self, output_idx: int = 0) -> int:
        """Stream width in bits for output.

        Returns the actual stream width based on the output's stream_shape,
        not derived from input block folding factors.
        """
        output = self.outputs[output_idx]
        return output.stream_width_bits

    def output_stream_shape(self, output_idx: int = 0) -> Shape:
        """Stream shape for output.

        Returns the OutputModel's stream_shape attribute (resolved by builder).
        """
        return self.outputs[output_idx].stream_shape

    @property
    def throughput_fps(self) -> float:
        """Throughput in inferences per second."""
        cycles_per_inf = self.initiation_interval
        clock_hz = self.clock_freq_mhz * 1e6
        return clock_hz / cycles_per_inf
