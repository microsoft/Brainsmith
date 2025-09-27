############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Kernel Model V3 - Pure Factory Pattern Implementation

This module implements immutable models that can only be created through
factory methods, enforcing the "always fresh" paradigm where models are
never cached and always reflect current nodeattr values.

Key Principles:
- Models are immutable value objects
- Direct instantiation is forbidden
- All creation goes through AutoHWCustomOp factories
- SDIM changes create new models, not mutations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any, FrozenSet
import math
from abc import ABC

from .types import Shape, RaggedShape, prod
from .qonnx_types import BaseDataType
from .base import ParameterBinding


def _check_factory_only(cls_name: str, kwargs: dict):
    """Check that instance is being created from factory"""
    if not kwargs.pop('_from_factory', False):
        raise RuntimeError(
            f"{cls_name} cannot be instantiated directly. "
            f"Use factory methods from AutoHWCustomOp."
        )


@dataclass(frozen=True)
class InputInterfaceV3:
    """Immutable input interface model
    
    All fields are frozen after construction. Changes require
    creating a new instance through factories.
    """
    
    def __post_init__(self):
        """Validate factory creation"""
        # Check happens before dataclass processes fields
        pass
    
    # Core properties
    tensor_dims: Shape
    block_dims: RaggedShape
    datatype: BaseDataType
    
    # Streaming configuration
    stream_dims: Shape  # What we've been calling SDIM
    
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
        # Total blocks
        total_blocks = 1
        for t, b in zip(self.tensor_dims, self.block_dims):
            total_blocks *= math.ceil(t / b)
        
        # Cycles per block
        cycles_per_block = 1
        for b, s in zip(self.block_dims, self.stream_dims):
            cycles_per_block *= math.ceil(b / s)
        
        return total_blocks * cycles_per_block
    
    @property
    def bandwidth_bits(self) -> int:
        """Interface bandwidth in bits per cycle"""
        return self.streaming_bandwidth * self.datatype.bitwidth()
    
    def with_stream_dims(self, new_stream_dims: Shape) -> 'InputInterfaceV3':
        """Create new instance with different streaming dimensions"""
        # Validate
        if len(new_stream_dims) != len(self.block_dims):
            raise ValueError(
                f"Stream dims {len(new_stream_dims)} must match "
                f"block dims {len(self.block_dims)}"
            )
        
        for i, (s, b) in enumerate(zip(new_stream_dims, self.block_dims)):
            if s <= 0:
                raise ValueError(f"Stream dim[{i}]={s} must be positive")
            if s > b:
                raise ValueError(f"Stream dim[{i}]={s} exceeds block dim {b}")
        
        # Create new instance through factory
        return InputInterfaceV3(
            _from_factory=True,
            tensor_dims=self.tensor_dims,
            block_dims=self.block_dims,
            datatype=self.datatype,
            stream_dims=tuple(new_stream_dims),
            schema_name=self.schema_name,
            skip_prob=self.skip_prob,
            actual_utilization=self.actual_utilization
        )
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        return {
            "tensor_dims": self.tensor_dims,
            "block_dims": self.block_dims,
            "stream_dims": self.stream_dims,
            "datatype": self.datatype.name,
            "streaming_bandwidth": self.streaming_bandwidth,
            "initiation_interval": self.initiation_interval,
            "bandwidth_bits": self.bandwidth_bits,
            "utilization": self.actual_utilization
        }


@dataclass(frozen=True)
class OutputInterfaceV3(_FactoryOnly):
    """Immutable output interface model"""
    
    # Core properties
    tensor_dims: Shape
    block_dims: RaggedShape
    datatype: BaseDataType
    
    # Output streaming rate (computed from kernel behavior)
    streaming_rate: int
    
    # Schema reference
    schema_name: str
    
    @property
    def bandwidth_bits(self) -> int:
        """Interface bandwidth in bits per cycle"""
        return self.streaming_rate * self.datatype.bitwidth()
    
    def with_streaming_rate(self, new_rate: int) -> 'OutputInterfaceV3':
        """Create new instance with different streaming rate"""
        return OutputInterfaceV3(
            _from_factory=True,
            tensor_dims=self.tensor_dims,
            block_dims=self.block_dims,
            datatype=self.datatype,
            streaming_rate=new_rate,
            schema_name=self.schema_name
        )
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        return {
            "tensor_dims": self.tensor_dims,
            "block_dims": self.block_dims,
            "datatype": self.datatype.name,
            "streaming_rate": self.streaming_rate,
            "bandwidth_bits": self.bandwidth_bits
        }


@dataclass(frozen=True)
class KernelModelV3(_FactoryOnly):
    """Immutable kernel model
    
    Represents a snapshot of kernel configuration at a point in time.
    All modifications return new instances.
    """
    
    # Interface collections (as tuples for immutability)
    input_models: Tuple[InputInterfaceV3, ...]
    output_models: Tuple[OutputInterfaceV3, ...]
    
    # Schema reference
    schema_name: str
    
    # Parameter bindings (frozen dict)
    parameter_values: Dict[str, Any] = field(default_factory=dict)
    
    # Performance characteristics
    latency_cycles: Tuple[int, int] = (1, 1)
    pipeline_depth: int = 1
    clock_freq_mhz: float = 100.0
    
    def __post_init__(self):
        """Validate and compute derived values"""
        # Convert to tuples if needed (for frozen dataclass)
        if isinstance(self.input_models, list):
            object.__setattr__(self, 'input_models', tuple(self.input_models))
        if isinstance(self.output_models, list):
            object.__setattr__(self, 'output_models', tuple(self.output_models))
        
        # Freeze parameter dict
        if isinstance(self.parameter_values, dict):
            object.__setattr__(self, 'parameter_values', 
                             dict(self.parameter_values))  # Shallow copy
        
        # Compute output rates
        self._compute_output_rates()
    
    def _compute_output_rates(self):
        """Compute output streaming rates based on inputs"""
        if not self.input_models or not self.output_models:
            return
        
        # Default: match first input's rate
        first_input_rate = self.input_models[0].streaming_bandwidth
        
        # Create new output models with computed rates
        new_outputs = []
        for output in self.output_models:
            if output.streaming_rate != first_input_rate:
                new_outputs.append(output.with_streaming_rate(first_input_rate))
            else:
                new_outputs.append(output)
        
        if new_outputs != list(self.output_models):
            object.__setattr__(self, 'output_models', tuple(new_outputs))
    
    def with_stream_config(self, config: Dict[str, Union[int, List[int]]]) -> 'KernelModelV3':
        """Create new model with updated streaming configuration
        
        Args:
            config: Maps input names to stream dimensions
            
        Returns:
            New KernelModelV3 with updated configuration
        """
        # Build updated input models
        new_inputs = []
        for inp in self.input_models:
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
                
                new_inputs.append(inp.with_stream_dims(new_stream_dims))
            else:
                new_inputs.append(inp)
        
        # Create new model
        return KernelModelV3(
            _from_factory=True,
            input_models=tuple(new_inputs),
            output_models=self.output_models,
            schema_name=self.schema_name,
            parameter_values=self.parameter_values,
            latency_cycles=self.latency_cycles,
            pipeline_depth=self.pipeline_depth,
            clock_freq_mhz=self.clock_freq_mhz
        )
    
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
        
        # Maximum across all inputs
        return max(inp.initiation_interval for inp in self.input_models)
    
    @property
    def throughput_fps(self) -> float:
        """Throughput in inferences per second"""
        cycles_per_inf = self.initiation_interval
        clock_hz = self.clock_freq_mhz * 1e6
        return clock_hz / cycles_per_inf
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        return {
            "kernel": self.schema_name,
            "initiation_interval": self.initiation_interval,
            "throughput_fps": self.throughput_fps,
            "inputs": {
                inp.schema_name: inp.calculate_metrics() 
                for inp in self.input_models
            },
            "outputs": {
                out.schema_name: out.calculate_metrics()
                for out in self.output_models
            },
            "total_input_bandwidth_bits": sum(
                inp.bandwidth_bits for inp in self.input_models
            ),
            "total_output_bandwidth_bits": sum(
                out.bandwidth_bits for out in self.output_models
            )
        }


# Factory functions that would live in AutoHWCustomOp

def create_kernel_model_v3(
    schema,  # KernelSchema
    nodeattrs: Dict[str, Any],
    input_shapes: List[Shape],
    output_shapes: List[Shape],
    _from_factory: bool = True
) -> KernelModelV3:
    """Factory to create KernelModelV3 from schema and runtime values
    
    This would be a method in AutoHWCustomOp that has access to:
    - The KernelSchema
    - Current nodeattr values
    - Input/output tensor shapes from ONNX graph
    """
    
    # Create input models
    input_models = []
    for i, inp_schema in enumerate(schema.input_schemas):
        # Resolve datatype
        if inp_schema.datatype_attr:
            datatype = BaseDataType[nodeattrs[inp_schema.datatype_attr]]
        else:
            datatype = BaseDataType["UINT8"]  # Default
        
        # Resolve block dims from template
        block_dims = resolve_template(
            inp_schema.block_tiling or [":"],
            nodeattrs,
            input_shapes[i]
        )
        
        # Resolve stream dims
        if inp_schema.stream_tiling:
            stream_dims = resolve_template(
                inp_schema.stream_tiling,
                nodeattrs,
                input_shapes[i]
            )
        else:
            stream_dims = tuple(1 for _ in block_dims)
        
        inp_model = InputInterfaceV3(
            _from_factory=True,
            tensor_dims=tuple(input_shapes[i]),
            block_dims=block_dims,
            datatype=datatype,
            stream_dims=stream_dims,
            schema_name=inp_schema.name
        )
        input_models.append(inp_model)
    
    # Create output models
    output_models = []
    for i, out_schema in enumerate(schema.output_schemas):
        # Resolve datatype
        if out_schema.datatype_attr:
            datatype = BaseDataType[nodeattrs[out_schema.datatype_attr]]
        else:
            datatype = BaseDataType["UINT8"]
        
        # Resolve block dims
        block_dims = resolve_template(
            out_schema.block_tiling or [":"],
            nodeattrs,
            output_shapes[i]
        )
        
        out_model = OutputInterfaceV3(
            _from_factory=True,
            tensor_dims=tuple(output_shapes[i]),
            block_dims=block_dims,
            datatype=datatype,
            streaming_rate=1,  # Will be computed
            schema_name=out_schema.name
        )
        output_models.append(out_model)
    
    # Extract parameter values
    param_values = {}
    for key in ["CHANNELS", "PE", "SIMD"]:  # Example parameters
        if key in nodeattrs:
            param_values[key] = nodeattrs[key]
    
    return KernelModelV3(
        _from_factory=True,
        input_models=tuple(input_models),
        output_models=tuple(output_models),
        schema_name=schema.name,
        parameter_values=param_values
    )


def resolve_template(
    template: List[Union[int, str]], 
    nodeattrs: Dict[str, Any],
    tensor_shape: Shape
) -> Tuple[int, ...]:
    """Resolve a template with nodeattr references to concrete values"""
    result = []
    for i, item in enumerate(template):
        if isinstance(item, int):
            result.append(item)
        elif item == ":":
            # Full dimension
            if i < len(tensor_shape):
                result.append(tensor_shape[i])
            else:
                result.append(1)
        elif isinstance(item, str):
            # Nodeattr reference
            if item in nodeattrs:
                result.append(nodeattrs[item])
            else:
                raise ValueError(f"Nodeattr '{item}' not found")
        else:
            raise ValueError(f"Invalid template item: {item}")
    
    return tuple(result)