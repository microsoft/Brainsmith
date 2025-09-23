############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Kernel Model V3 - Simplified Factory Pattern

Uses module-level factory functions instead of complex metaclass magic.
Models are still immutable, but creation is controlled by convention.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
import math

from .types import Shape, RaggedShape, prod
from .qonnx_types import BaseDataType


@dataclass(frozen=True)
class InputInterfaceV3:
    """Immutable input interface model"""
    
    # Core properties
    tensor_dims: Shape
    block_dims: RaggedShape
    datatype: BaseDataType
    
    # Streaming configuration  
    stream_dims: Shape
    
    # Schema reference
    schema_name: str
    
    # Performance properties
    skip_prob: Tuple[float, ...] = field(default_factory=tuple)
    actual_utilization: float = 1.0
    
    @property
    def streaming_bandwidth(self) -> int:
        """Elements streamed per cycle"""
        return prod(self.stream_dims)
    
    @property
    def initiation_interval(self) -> int:
        """Cycles to stream entire tensor"""
        total_blocks = 1
        for t, b in zip(self.tensor_dims, self.block_dims):
            total_blocks *= math.ceil(t / b)
        
        cycles_per_block = 1
        for b, s in zip(self.block_dims, self.stream_dims):
            cycles_per_block *= math.ceil(b / s)
        
        return total_blocks * cycles_per_block
    
    @property
    def bandwidth_bits(self) -> int:
        """Interface bandwidth in bits per cycle"""
        return self.streaming_bandwidth * self.datatype.bitwidth()


@dataclass(frozen=True)
class OutputInterfaceV3:
    """Immutable output interface model"""
    
    # Core properties
    tensor_dims: Shape
    block_dims: RaggedShape
    datatype: BaseDataType
    
    # Output streaming rate
    streaming_rate: int
    
    # Schema reference
    schema_name: str
    
    @property
    def bandwidth_bits(self) -> int:
        """Interface bandwidth in bits per cycle"""
        return self.streaming_rate * self.datatype.bitwidth()


@dataclass(frozen=True)
class KernelModelV3:
    """Immutable kernel model"""
    
    # Interface collections
    input_models: Tuple[InputInterfaceV3, ...]
    output_models: Tuple[OutputInterfaceV3, ...]
    
    # Schema reference
    schema_name: str
    
    # Parameter values
    parameter_values: Dict[str, Any] = field(default_factory=dict)
    
    # Performance characteristics
    latency_cycles: Tuple[int, int] = (1, 1)
    pipeline_depth: int = 1
    clock_freq_mhz: float = 100.0
    
    def __post_init__(self):
        """Finalize initialization"""
        # Ensure tuples
        if isinstance(self.input_models, list):
            object.__setattr__(self, 'input_models', tuple(self.input_models))
        if isinstance(self.output_models, list):
            object.__setattr__(self, 'output_models', tuple(self.output_models))
    
    def get_input(self, name: str) -> Optional[InputInterfaceV3]:
        """Get input model by name"""
        for inp in self.input_models:
            if inp.schema_name == name:
                return inp
        return None
    
    def get_output(self, name: str) -> Optional[OutputInterfaceV3]:
        """Get output model by name"""
        for out in self.output_models:
            if out.schema_name == name:
                return out
        return None
    
    @property
    def initiation_interval(self) -> int:
        """Kernel initiation interval"""
        if not self.input_models:
            return 1
        return max(inp.initiation_interval for inp in self.input_models)
    
    @property
    def throughput_fps(self) -> float:
        """Throughput in inferences per second"""
        cycles_per_inf = self.initiation_interval
        clock_hz = self.clock_freq_mhz * 1e6
        return clock_hz / cycles_per_inf


# Factory functions - these are the ONLY way to create models

def create_input_model(
    tensor_dims: Shape,
    block_dims: RaggedShape,
    datatype: BaseDataType,
    stream_dims: Optional[Shape],
    schema_name: str,
    **kwargs
) -> InputInterfaceV3:
    """Factory for creating input models
    
    This is the ONLY supported way to create InputInterfaceV3.
    """
    if stream_dims is None:
        stream_dims = tuple(1 for _ in block_dims)
    
    return InputInterfaceV3(
        tensor_dims=tuple(tensor_dims),
        block_dims=tuple(block_dims),
        datatype=datatype,
        stream_dims=tuple(stream_dims),
        schema_name=schema_name,
        skip_prob=kwargs.get('skip_prob', ()),
        actual_utilization=kwargs.get('actual_utilization', 1.0)
    )


def create_output_model(
    tensor_dims: Shape,
    block_dims: RaggedShape,
    datatype: BaseDataType,
    streaming_rate: int,
    schema_name: str
) -> OutputInterfaceV3:
    """Factory for creating output models"""
    return OutputInterfaceV3(
        tensor_dims=tuple(tensor_dims),
        block_dims=tuple(block_dims),
        datatype=datatype,
        streaming_rate=streaming_rate,
        schema_name=schema_name
    )


def create_kernel_model(
    input_models: List[InputInterfaceV3],
    output_models: List[OutputInterfaceV3],
    schema_name: str,
    **kwargs
) -> KernelModelV3:
    """Factory for creating kernel models"""
    
    # Compute output rates based on inputs
    if input_models and output_models:
        first_input_rate = input_models[0].streaming_bandwidth
        
        # Update output models with computed rates
        new_outputs = []
        for out in output_models:
            if out.streaming_rate != first_input_rate:
                # Create new with updated rate
                new_out = create_output_model(
                    out.tensor_dims,
                    out.block_dims,
                    out.datatype,
                    first_input_rate,
                    out.schema_name
                )
                new_outputs.append(new_out)
            else:
                new_outputs.append(out)
        output_models = new_outputs
    
    return KernelModelV3(
        input_models=tuple(input_models),
        output_models=tuple(output_models),
        schema_name=schema_name,
        parameter_values=kwargs.get('parameter_values', {}),
        latency_cycles=kwargs.get('latency_cycles', (1, 1)),
        pipeline_depth=kwargs.get('pipeline_depth', 1),
        clock_freq_mhz=kwargs.get('clock_freq_mhz', 100.0)
    )


def update_kernel_stream_config(
    kernel: KernelModelV3,
    config: Dict[str, Union[int, List[int]]]
) -> KernelModelV3:
    """Create new kernel model with updated streaming configuration"""
    
    new_inputs = []
    for inp in kernel.input_models:
        if inp.schema_name in config:
            stream_spec = config[inp.schema_name]
            
            # Convert to shape
            if isinstance(stream_spec, int):
                # Uniform across all dimensions
                new_stream_dims = tuple(
                    min(stream_spec, b) for b in inp.block_dims
                )
            else:
                new_stream_dims = tuple(stream_spec)
            
            # Validate
            if len(new_stream_dims) != len(inp.block_dims):
                raise ValueError(
                    f"Stream dims {len(new_stream_dims)} must match "
                    f"block dims {len(inp.block_dims)}"
                )
            
            for i, (s, b) in enumerate(zip(new_stream_dims, inp.block_dims)):
                if s <= 0:
                    raise ValueError(f"Stream dim[{i}]={s} must be positive")
                if s > b:
                    raise ValueError(f"Stream dim[{i}]={s} exceeds block dim {b}")
            
            # Create new input
            new_inp = create_input_model(
                inp.tensor_dims,
                inp.block_dims,
                inp.datatype,
                new_stream_dims,
                inp.schema_name,
                skip_prob=inp.skip_prob,
                actual_utilization=inp.actual_utilization
            )
            new_inputs.append(new_inp)
        else:
            new_inputs.append(inp)
    
    # Create new kernel
    return create_kernel_model(
        new_inputs,
        list(kernel.output_models),
        kernel.schema_name,
        parameter_values=kernel.parameter_values,
        latency_cycles=kernel.latency_cycles,
        pipeline_depth=kernel.pipeline_depth,
        clock_freq_mhz=kernel.clock_freq_mhz
    )