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
    Create via create_input_model() factory function.
    """
    
    # Core properties
    name: str
    tensor_dims: Shape
    block_dims: Shape  
    datatype: BaseDataType
    
    # Streaming configuration (SDIM)
    stream_dims: Shape
    
    # Optional properties
    is_weight: bool = False
    
    @property
    def streaming_bandwidth(self) -> int:
        """Elements streamed per cycle."""
        return prod(self.stream_dims)
    
    @property
    def initiation_interval(self) -> int:
        """Cycles to stream entire tensor."""
        total_blocks = 1
        for t, b in zip(self.tensor_dims, self.block_dims):
            total_blocks *= math.ceil(t / b)
        
        cycles_per_block = 1
        for b, s in zip(self.block_dims, self.stream_dims):
            cycles_per_block *= math.ceil(b / s)
        
        return total_blocks * cycles_per_block
    
    @property
    def bandwidth_bits(self) -> int:
        """Interface bandwidth in bits per cycle."""
        return self.streaming_bandwidth * self.datatype.bitwidth()
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        return {
            "name": self.name,
            "tensor_dims": self.tensor_dims,
            "block_dims": self.block_dims,
            "stream_dims": self.stream_dims,
            "datatype": self.datatype.name,
            "streaming_bandwidth": self.streaming_bandwidth,
            "initiation_interval": self.initiation_interval,
            "bandwidth_bits": self.bandwidth_bits,
            "is_weight": self.is_weight
        }


@dataclass(frozen=True)
class OutputModel:
    """Immutable output interface model.
    
    Represents a concrete output with resolved dimensions and datatypes.
    Create via create_output_model() factory function.
    """
    
    # Core properties
    name: str
    tensor_dims: Shape
    block_dims: Shape
    datatype: BaseDataType
    
    # Output streaming rate (elements per cycle)
    streaming_rate: int
    
    @property
    def bandwidth_bits(self) -> int:
        """Interface bandwidth in bits per cycle."""
        return self.streaming_rate * self.datatype.bitwidth()
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        return {
            "name": self.name,
            "tensor_dims": self.tensor_dims,
            "block_dims": self.block_dims,
            "datatype": self.datatype.name,
            "streaming_rate": self.streaming_rate,
            "bandwidth_bits": self.bandwidth_bits
        }


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
    def throughput_fps(self) -> float:
        """Throughput in inferences per second."""
        cycles_per_inf = self.initiation_interval
        clock_hz = self.clock_freq_mhz * 1e6
        return clock_hz / cycles_per_inf
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        total_input_bw = sum(inp.bandwidth_bits for inp in self.inputs)
        total_output_bw = sum(out.bandwidth_bits for out in self.outputs)
        
        return {
            "kernel": self.name,
            "parameters": self.parameters,
            "initiation_interval": self.initiation_interval,
            "throughput_fps": self.throughput_fps,
            "clock_freq_mhz": self.clock_freq_mhz,
            "inputs": {inp.name: inp.calculate_metrics() for inp in self.inputs},
            "outputs": {out.name: out.calculate_metrics() for out in self.outputs},
            "total_input_bandwidth_bits": total_input_bw,
            "total_output_bandwidth_bits": total_output_bw,
            "total_bandwidth_mbps": (total_input_bw + total_output_bw) * self.clock_freq_mhz / 8.0
        }

def create_model

# Factory functions for creating models

def create_input_model(
    name: str,
    tensor_dims: Shape,
    block_dims: Shape,
    datatype: BaseDataType,
    stream_dims: Optional[Shape] = None,
    **kwargs
) -> InputModel:
    """Factory function to create an InputModel.
    
    Args:
        name: Interface name
        tensor_dims: Full tensor dimensions
        block_dims: Block dimensions for tiling
        datatype: QONNX datatype
        stream_dims: Streaming dimensions (default: all 1s)
        **kwargs: Additional properties (is_weight, etc.)
        
    Returns:
        Immutable InputModel instance
    """
    if stream_dims is None:
        stream_dims = tuple(1 for _ in block_dims)
    
    return InputModel(
        name=name,
        tensor_dims=tuple(tensor_dims),
        block_dims=tuple(block_dims),
        datatype=datatype,
        stream_dims=tuple(stream_dims),
        is_weight=kwargs.get('is_weight', False)
    )


def create_output_model(
    name: str,
    tensor_dims: Shape,
    block_dims: Shape,
    datatype: BaseDataType,
    streaming_rate: int = 1
) -> OutputModel:
    """Factory function to create an OutputModel.
    
    Args:
        name: Interface name
        tensor_dims: Full tensor dimensions
        block_dims: Block dimensions for tiling
        datatype: QONNX datatype
        streaming_rate: Elements per cycle (default: 1)
        
    Returns:
        Immutable OutputModel instance
    """
    return OutputModel(
        name=name,
        tensor_dims=tuple(tensor_dims),
        block_dims=tuple(block_dims),
        datatype=datatype,
        streaming_rate=streaming_rate
    )


def create_kernel_model(
    name: str,
    inputs: List[InputModel],
    outputs: List[OutputModel],
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> KernelModel:
    """Factory function to create a KernelModel.
    
    This function also computes output streaming rates based on input
    configurations to ensure consistency.
    
    Args:
        name: Kernel name
        inputs: List of InputModel instances
        outputs: List of OutputModel instances  
        parameters: Resolved parameter values (e.g., CHANNELS, PE)
        **kwargs: Performance characteristics (clock_freq_mhz, etc.)
        
    Returns:
        Immutable KernelModel instance
    """
    # Compute output rates based on first input
    if inputs and outputs:
        first_input_rate = inputs[0].streaming_bandwidth
        
        # Update output models with computed rates
        updated_outputs = []
        for out in outputs:
            if out.streaming_rate != first_input_rate:
                # Create new output with updated rate
                updated_out = create_output_model(
                    out.name,
                    out.tensor_dims,
                    out.block_dims,
                    out.datatype,
                    first_input_rate
                )
                updated_outputs.append(updated_out)
            else:
                updated_outputs.append(out)
        outputs = updated_outputs
    
    return KernelModel(
        name=name,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        parameters=parameters or {},
        latency_cycles=kwargs.get('latency_cycles', (1, 1)),
        pipeline_depth=kwargs.get('pipeline_depth', 1),
        clock_freq_mhz=kwargs.get('clock_freq_mhz', 100.0)
    )


def update_kernel_stream_config(
    kernel: KernelModel,
    stream_config: Dict[str, Union[int, List[int]]]
) -> KernelModel:
    """Create new KernelModel with updated streaming configuration.
    
    This function creates a new immutable KernelModel with updated SDIM
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
                new_stream_dims = tuple(
                    min(stream_spec, b) for b in inp.block_dims
                )
            else:
                new_stream_dims = tuple(stream_spec)
            
            # Validate dimensions
            if len(new_stream_dims) != len(inp.block_dims):
                raise ValueError(
                    f"Stream dims {len(new_stream_dims)} must match "
                    f"block dims {len(inp.block_dims)} for input '{inp.name}'"
                )
            
            for i, (s, b) in enumerate(zip(new_stream_dims, inp.block_dims)):
                if s <= 0:
                    raise ValueError(f"Stream dim[{i}]={s} must be positive")
                if s > b:
                    raise ValueError(
                        f"Stream dim[{i}]={s} exceeds block dim {b} "
                        f"for input '{inp.name}'"
                    )
            
            # Create new input with updated stream dims
            updated_inp = create_input_model(
                inp.name,
                inp.tensor_dims,
                inp.block_dims,
                inp.datatype,
                new_stream_dims,
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