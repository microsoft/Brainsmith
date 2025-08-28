############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Base interface class shared by input and output interfaces"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from abc import abstractmethod
import math

from .base import BaseModel, ParameterBinding
from .types import Shape, RaggedShape, prod
from .qonnx_types import BaseDataType


@dataclass
class BaseInterface(BaseModel):
    """Base class for input and output interfaces
    
    Provides common functionality:
    - Tensor and block dimensions
    - Data type handling
    - CSDF phase tracking
    - Bandwidth calculations
    - Performance metric caching
    
    Note on CSDF support:
    - RaggedShape allows different block sizes per phase
    - This provides forward compatibility for cyclo-static dataflow
    - Currently most kernels use single-phase (SDF) behavior
    """
    
    # Core dimensions
    tensor_dims: Shape              # Full tensor shape
    block_dims: RaggedShape        # Block decomposition (can be CSDF)
    datatype: BaseDataType         # Concrete QONNX datatype
    
    # Runtime behavior
    parameter_binding: Optional[ParameterBinding] = None
    
    # Internal state
    _cached_metrics: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize with optimized setup"""
        # Normalize block_dims to list format
        if isinstance(self.block_dims, tuple):
            self.block_dims = [self.block_dims]
        
        self._cached_metrics = {}
    
    @property
    def n_phases(self) -> int:
        """Number of CSDF phases
        
        For cyclo-static dataflow, different phases can have different
        block sizes. Single-phase = standard synchronous dataflow (SDF).
        """
        return len(self.block_dims)
    
    @property
    def is_csdf(self) -> bool:
        """Check if interface has cyclo-static behavior
        
        CSDF allows different token production/consumption rates
        across a repeating cycle of phases.
        """
        return self.n_phases > 1
    
    @property
    @abstractmethod
    def bandwidth_bits(self) -> int:
        """Interface bandwidth in bits per cycle
        
        Must be implemented by subclasses based on their
        streaming characteristics (SDIM for input, rate for output).
        """
        pass
    
    @property
    def bandwidth_bytes(self) -> float:
        """Bandwidth in bytes per cycle"""
        return self.bandwidth_bits / 8.0
    
    def effective_bandwidth(self, clock_freq_mhz: float = 100.0) -> float:
        """Compute effective bandwidth in MB/s
        
        Args:
            clock_freq_mhz: Clock frequency in MHz
            
        Returns:
            Effective bandwidth in MB/s
        """
        cycles_per_second = clock_freq_mhz * 1e6
        bytes_per_cycle = self.bandwidth_bytes
        return bytes_per_cycle * cycles_per_second / 1e6
    
    def _invalidate_performance_cache(self):
        """Invalidate cached performance metrics"""
        self._cached_metrics.clear()
    
    @abstractmethod
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics
        
        Must be implemented by subclasses to include their
        specific metrics.
        """
        pass