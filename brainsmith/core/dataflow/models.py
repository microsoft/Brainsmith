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
time. Models should always be created fresh to avoid staleness.

Key principles:
- All models are immutable (frozen dataclasses)
- Create fresh models for each use via factory functions
- Never store model instances - always recreate from current state
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

    # Streaming configuration
    stream_shape: Shape

    # Optional properties
    is_weight: bool = False
    
    @property
    def streaming_bandwidth(self) -> int:
        """Elements streamed per cycle."""
        return prod(self.stream_shape)

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
    It should be created fresh whenever needed to ensure it reflects current
    nodeattr values.
    
    Create via create_kernel_model() factory function.
    """
    
    # Kernel identity
    name: str
    
    # Interface models
    inputs: Tuple[InputModel, ...]
    outputs: Tuple[OutputModel, ...]
    
    # Resolved parameter values
    parameters: Dict[str, Any] = field(default_factory=dict)
    
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

    @property
    def throughput_fps(self) -> float:
        """Throughput in inferences per second."""
        cycles_per_inf = self.initiation_interval
        clock_hz = self.clock_freq_mhz * 1e6
        return clock_hz / cycles_per_inf


# Factory functions for creating models

def create_input_model(
    name: str,
    tensor_shape: Tuple[int, ...],
    block_shape: Tuple[int, ...],
    datatype: BaseDataType,
    stream_shape: Tuple[int, ...],
    is_weight: bool = False
) -> InputModel:
    """Factory function to create an InputModel.

    Args:
        name: Interface name
        tensor_shape: Full tensor dimensions
        block_shape: Block tiling dimensions
        datatype: Data type
        stream_shape: Streaming dimensions per cycle
        is_weight: Whether this is a weight input

    Returns:
        Immutable InputModel instance
    """
    return InputModel(
        name=name,
        tensor_shape=tensor_shape,
        block_shape=block_shape,
        datatype=datatype,
        stream_shape=stream_shape,
        is_weight=is_weight
    )


def create_output_model(
    name: str,
    tensor_shape: Tuple[int, ...],
    block_shape: Tuple[int, ...],
    datatype: BaseDataType,
    streaming_rate: int = None  # Deprecated, ignored
) -> OutputModel:
    """Factory function to create an OutputModel.

    Args:
        name: Interface name
        tensor_shape: Full tensor dimensions
        block_shape: Block tiling dimensions
        datatype: Data type
        streaming_rate: DEPRECATED - Ignored. Streaming rate is computed by KernelModel.

    Returns:
        Immutable OutputModel instance
    """
    if streaming_rate is not None:
        import warnings
        warnings.warn(
            "streaming_rate parameter is deprecated. Output streaming is computed by KernelModel.",
            DeprecationWarning,
            stacklevel=2
        )

    return OutputModel(
        name=name,
        tensor_shape=tensor_shape,
        block_shape=block_shape,
        datatype=datatype
    )


def create_kernel_model(
    name: str,
    inputs: List[InputModel],
    outputs: List[OutputModel],
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> KernelModel:
    """Factory function to create a KernelModel.

    Args:
        name: Kernel name
        inputs: List of InputModel instances
        outputs: List of OutputModel instances (will NOT be mutated)
        parameters: Resolved parameter values (e.g., CHANNELS, PE)
        **kwargs: Performance characteristics (clock_freq_mhz, etc.)

    Returns:
        Immutable KernelModel instance
    """
    return KernelModel(
        name=name,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        parameters=parameters or {},
        **kwargs
    )


def update_kernel_stream_config(
    kernel: KernelModel,
    stream_config: Dict[str, Union[int, List[int]]]
) -> KernelModel:
    """Create new KernelModel with updated streaming configuration.

    This function creates a new immutable KernelModel with updated stream_shape
    values for the specified inputs.

    Args:
        kernel: Original KernelModel
        stream_config: Maps input names to new stream dimensions

    Returns:
        New KernelModel with updated configuration
    """
    updated_inputs = []

    for inp in kernel.inputs:
        if inp.name in stream_config:
            stream_spec = stream_config[inp.name]

            # Convert spec to shape
            if isinstance(stream_spec, int):
                # Uniform streaming
                new_stream_shape = tuple(
                    min(stream_spec, b) for b in inp.block_shape
                )
            else:
                new_stream_shape = tuple(stream_spec)

            # Validate dimensions
            if len(new_stream_shape) != len(inp.block_shape):
                raise ValueError(
                    f"Stream shape {len(new_stream_shape)} must match "
                    f"block shape {len(inp.block_shape)} for input '{inp.name}'"
                )

            for i, (s, b) in enumerate(zip(new_stream_shape, inp.block_shape)):
                if s <= 0:
                    raise ValueError(f"Stream shape[{i}]={s} must be positive")
                if s > b:
                    raise ValueError(
                        f"Stream shape[{i}]={s} exceeds block shape {b} "
                        f"for input '{inp.name}'"
                    )

            # Create new input with updated stream shape
            updated_inp = create_input_model(
                inp.name,
                inp.tensor_shape,
                inp.block_shape,
                inp.datatype,
                new_stream_shape,
                is_weight=inp.is_weight
            )
            updated_inputs.append(updated_inp)
        else:
            updated_inputs.append(inp)

    # Create new kernel with updated inputs
    return create_kernel_model(
        kernel.name,
        updated_inputs,
        list(kernel.outputs),
        kernel.parameters,
        latency_cycles=kernel.latency_cycles,
        pipeline_depth=kernel.pipeline_depth,
        clock_freq_mhz=kernel.clock_freq_mhz
    )