############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Input interface model with streaming configuration support"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Tuple
import math
from .base_interface import BaseInterface
from .base import ParameterBinding
from .types import Shape, RaggedShape, prod
from .qonnx_types import BaseDataType

@dataclass
class InputInterface(BaseInterface):
    """Model for input interfaces with streaming configuration
    
    Input interfaces support:
    - Multi-dimensional streaming (SDIM)
    - Performance modeling
    - Sparsity and utilization tracking
    """
    
    # Streaming configuration
    _sdim: Optional[Shape] = field(default=None, init=False)
    
    # Runtime behavior
    skip_prob: List[float] = field(default_factory=list)  # Sparsity per phase
    actual_utilization: float = 1.0  # Actual vs theoretical utilization
    
    def __init__(self, 
                 tensor_dims: Shape, 
                 block_dims: RaggedShape,
                 datatype: BaseDataType,
                 stream_dims: Optional[Shape] = None,
                 skip_prob: List[float] = None,
                 actual_utilization: float = 1.0,
                 definition: Optional['InputDefinition'] = None,
                 parameter_binding: Optional[ParameterBinding] = None,
                 **kwargs):
        """Initialize input interface"""
        super().__init__(definition)
        
        self.tensor_dims = tensor_dims
        self.block_dims = block_dims
        self.datatype = datatype
        self.skip_prob = skip_prob or []
        self.actual_utilization = actual_utilization
        self.parameter_binding = parameter_binding
        
        # Initialize streaming dimensions from constructor parameter
        if stream_dims is not None:
            self._sdim = tuple(stream_dims)
        else:
            self._sdim = None
        
        # Call parent post_init
        super().__post_init__()
        
        # Default skip_prob if not provided
        if not self.skip_prob:
            self.skip_prob = [0.0] * self.n_phases
    
    @property
    def sdim(self) -> Shape:
        """Streaming dimensions (elements per cycle per dimension)"""
        if self._sdim is None:
            # Default: minimal streaming [1, 1, ...] for each dimension
            return tuple(1 for _ in self.block_dims)
        return self._sdim
    
    @sdim.setter
    def sdim(self, value: Union[int, List[int], Tuple[int, ...]]):
        """Set streaming dimensions with validation
        
        Args:
            value: Can be:
                - int: Uniform streaming across all dimensions
                - List/Tuple: Per-dimension streaming
        """
        # Convert to tuple
        if isinstance(value, int):
            # Uniform for all dimensions in the first phase
            value = tuple(value for _ in self.block_dims)
        else:
            value = tuple(value)
        
        # Validate against dimensions in the first phase
        if len(value) != len(self.block_dims):
            raise ValueError(
                f"SDIM dimensionality {len(value)} must match "
                f"block dimensionality {len(self.block_dims)}"
            )
        
        # Validate each dimension
        for i, (s, b) in enumerate(zip(value, self.block_dims)):
            if s <= 0:
                raise ValueError(f"SDIM[{i}]={s} must be positive")
            if s > b:
                raise ValueError(f"SDIM[{i}]={s} exceeds block dim {b}")
        
        self._sdim = value
        self._invalidate_performance_cache()
    
    @property
    def streaming_bandwidth(self) -> int:
        """Total elements streamed per cycle"""
        return prod(self.sdim)
    
    @property
    def initiation_interval(self) -> int:
        """Total cycles to stream entire tensor"""
        if "initiation_interval" not in self._cached_metrics:
            # Total blocks needed
            total_blocks = 1
            for t, b in zip(self.tensor_dims, self.block_dims):
                total_blocks *= math.ceil(t / b)
            
            # Cycles per block
            cycles_per_block = 1
            for b, s in zip(self.block_dims, self.sdim):
                cycles_per_block *= math.ceil(b / s)
            
            self._cached_metrics["initiation_interval"] = total_blocks * cycles_per_block
        
        return self._cached_metrics["initiation_interval"]
    
    @property
    def bandwidth_bits(self) -> int:
        """Interface bandwidth in bits per cycle"""
        if "bandwidth_bits" not in self._cached_metrics:
            # Use QONNX bitwidth() method
            self._cached_metrics["bandwidth_bits"] = self.streaming_bandwidth * self.datatype.bitwidth()
        return self._cached_metrics["bandwidth_bits"]
    
    def effective_bandwidth(self, clock_freq_mhz: float = 100.0) -> float:
        """Compute effective bandwidth in MB/s
        
        Overrides base to include utilization factor.
        """
        cycles_per_second = clock_freq_mhz * 1e6
        bytes_per_cycle = self.bandwidth_bytes * self.actual_utilization
        return bytes_per_cycle * cycles_per_second / 1e6
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        return {
            "tensor_dims": self.tensor_dims,
            "block_dims": self.block_dims,
            "sdim": self.sdim,
            "streaming_bandwidth": self.streaming_bandwidth,
            "initiation_interval": self.initiation_interval,
            "bandwidth_bits": self.bandwidth_bits,
            "bandwidth_mbps": self.effective_bandwidth(100.0),
            "n_phases": self.n_phases,
            "is_csdf": self.is_csdf,
            "utilization": self.actual_utilization,
            "skip_probabilities": self.skip_prob
        }
    
    def validate_connection(self, other: 'InputInterface') -> List[str]:
        """Validate this input can connect to another (for relationships)"""
        errors = []
        
        # Check tensor dimensions match
        if self.tensor_dims != other.tensor_dims:
            errors.append(
                f"Tensor dimension mismatch: {self.tensor_dims} != {other.tensor_dims}"
            )
        
        # Check data types match
        if self.datatype != other.datatype:
            errors.append(
                f"Data type mismatch: {self.datatype.get_canonical_name()} != {other.datatype.get_canonical_name()}"
            )
        
        return errors
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"InputInterface("
            f"tensor={self.tensor_dims}, "
            f"block={self.block_dims}, "
            f"sdim={self.sdim}, "
            f"bandwidth={self.streaming_bandwidth})"
        )