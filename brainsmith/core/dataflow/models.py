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
import math

from .types import Shape, prod
from qonnx.core.datatype import BaseDataType


@dataclass(frozen=True)
class InputModel:
    """Immutable input interface model.

    Represents a concrete input with resolved dimensions and datatypes.

    Shape hierarchy:
    - tensor_shape: Full logical tensor dimensions
    - block_shape: Block tiling dimensions (tensor_shape / block_shape = tensor_folding_factor)
    - stream_shape: Streaming dimensions per cycle (block_shape / stream_shape = block_folding_factor)
    """

    # Core properties
    name: str
    tensor_shape: Shape
    block_shape: Shape
    datatype: BaseDataType
    stream_shape: Shape
    is_weight: bool = False

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
    def initiation_interval(self) -> int:
        """Cycles to stream entire tensor."""
        return self.tensor_folding_factor * self.block_folding_factor

    @property
    def streaming_bandwidth(self) -> int:
        """Elements streamed per cycle."""
        return prod(self.stream_shape)

    @property
    def stream_width_bits(self) -> int:
        """Stream width in bits."""
        return self.streaming_bandwidth * self.datatype.bitwidth()


@dataclass(frozen=True)
class OutputModel:
    """Immutable output interface model.

    Pure specification of output tensor characteristics.
    Streaming behavior is computed by KernelModel.

    Shape hierarchy:
    - tensor_shape: Full logical tensor dimensions
    - block_shape: Block tiling dimensions
    """

    # Core properties
    name: str
    tensor_shape: Shape
    block_shape: Shape
    datatype: BaseDataType

    @property
    def tensor_folding_factor(self) -> int:
        """Number of blocks to cover full tensor."""
        num_blocks = 1
        for t, b in zip(self.tensor_shape, self.block_shape):
            num_blocks *= math.ceil(t / b)
        return num_blocks


@dataclass(frozen=True)
class KernelModel:
    """Immutable kernel model representing a configured kernel instance.

    This model is a snapshot of a kernel's configuration at a point in time.
    Models are cached and refreshed when nodeattrs change.
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
        """Ensure proper types after construction."""
        # Convert lists to tuples for immutability
        if isinstance(self.inputs, list):
            object.__setattr__(self, 'inputs', tuple(self.inputs))
        if isinstance(self.outputs, list):
            object.__setattr__(self, 'outputs', tuple(self.outputs))
    
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
        """Stream width in bits for output."""
        output = self.outputs[output_idx]
        return self.output_streaming_rate(output_idx) * output.datatype.bitwidth()

    def output_stream_shape(self, output_idx: int = 0) -> Shape:
        """Stream shape for output (inferred from block processing).

        Stream along last dimension with streaming_rate parallelism.
        """
        output = self.outputs[output_idx]
        rate = self.output_streaming_rate(output_idx)

        # Stream along last dimension with streaming_rate parallelism
        stream_shape = list(output.block_shape)
        if stream_shape:  # Guard against empty shapes
            stream_shape[-1] = rate
        return tuple(stream_shape)

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


