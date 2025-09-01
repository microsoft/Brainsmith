############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Output interface model - simplified without SDIM configuration"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math
from .base_interface import BaseInterface
from .base import ParameterBinding
from .types import Shape, RaggedShape, prod
from .qonnx_types import BaseDataType

@dataclass
class OutputInterface(BaseInterface):
    """Model for output interfaces
    
    Output interfaces:
    - Do NOT have configurable SDIM
    - Streaming rate is determined by kernel computation
    - Only track tensor and block dimensions
    """
    
    # Computed streaming rate (set by kernel)
    _streaming_rate: Optional[int] = field(default=None, init=False)
    
    def __init__(self,
                 tensor_dims: Shape,
                 block_dims: RaggedShape,
                 datatype: BaseDataType,
                 definition: Optional['OutputDefinition'] = None,
                 parameter_binding: Optional[ParameterBinding] = None,
                 **kwargs):
        """Initialize output interface"""
        super().__init__(definition)
        
        self.tensor_dims = tensor_dims
        self.block_dims = block_dims
        self.datatype = datatype
        self.parameter_binding = parameter_binding
        
        # Initialize fields
        self._streaming_rate = None
        
        # Call parent post_init
        super().__post_init__()
    
    @property
    def streaming_rate(self) -> int:
        """Elements produced per cycle (computed by kernel)"""
        return self._streaming_rate if self._streaming_rate is not None else 1
    
    def set_streaming_rate(self, rate: int):
        """Set the computed streaming rate
        
        This should only be called by the kernel based on input rates
        and computation pattern.
        """
        if rate <= 0:
            raise ValueError(f"Streaming rate must be positive, got {rate}")
        self._streaming_rate = rate
        self._invalidate_performance_cache()
    
    @property
    def production_interval(self) -> int:
        """Cycles to produce entire tensor (based on streaming rate)"""
        if "production_interval" not in self._cached_metrics:
            total_elements = prod(self.tensor_dims)
            self._cached_metrics["production_interval"] = math.ceil(
                total_elements / self.streaming_rate
            )
        return self._cached_metrics["production_interval"]
    
    @property
    def bandwidth_bits(self) -> int:
        """Interface bandwidth in bits per cycle"""
        if "bandwidth_bits" not in self._cached_metrics:
            # Use QONNX bitwidth() method
            self._cached_metrics["bandwidth_bits"] = self.streaming_rate * self.datatype.bitwidth()
        return self._cached_metrics["bandwidth_bits"]
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        return {
            "tensor_dims": self.tensor_dims,
            "block_dims": self.block_dims[0],
            "streaming_rate": self.streaming_rate,
            "production_interval": self.production_interval,
            "bandwidth_bits": self.bandwidth_bits,
            "bandwidth_mbps": self.effective_bandwidth(100.0),
            "n_phases": self.n_phases,
            "is_csdf": self.is_csdf
        }
    
    def validate_dimensions(self, expected_dims: Shape) -> List[str]:
        """Validate output dimensions match expected"""
        errors = []
        
        if self.tensor_dims != expected_dims:
            errors.append(
                f"Output dimension mismatch: {self.tensor_dims} != expected {expected_dims}"
            )
        
        return errors
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"OutputInterface("
            f"tensor={self.tensor_dims}, "
            f"block={self.block_dims[0] if len(self.block_dims) == 1 else self.block_dims}, "
            f"rate={self.streaming_rate} elem/cycle)"
        )