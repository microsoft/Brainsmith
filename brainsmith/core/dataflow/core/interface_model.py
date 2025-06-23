############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Interface model for performance calculations and runtime behavior"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import math
from .base import BaseModel
from .types import Shape, RaggedShape, prod, is_valid_tiling


@dataclass
class InterfaceModel(BaseModel):
    """Runtime model of an interface optimized for performance calculations
    
    Focuses on actual dimensions and performance metrics. Assumes valid
    configuration and skips validation for performance.
    """
    
    # Runtime dimensions (the actual "what is")
    tensor_dims: Shape
    block_dims: RaggedShape  # Can be single shape or list for CSDF
    # stream_dims managed as property, not field
    
    # Runtime behavior modeling
    skip_prob: List[float] = field(default_factory=list)  # Sparsity per phase
    actual_utilization: float = 1.0  # Actual vs theoretical utilization
    
    # Dynamic parallelism support
    _ipar: Optional[int] = field(default=None, init=False)  # Interface parallelism value
    _stream_dims: Optional[Shape] = field(default=None, init=False)  # Cached stream dimensions
    parameter_binding: Optional[Dict[str, int]] = None  # Parameter values from definition
    
    def __init__(self, tensor_dims: Shape, block_dims: RaggedShape, 
                 stream_dims: Optional[Shape] = None,
                 skip_prob: List[float] = None,
                 actual_utilization: float = 1.0,
                 definition: Optional['InterfaceDefinition'] = None,
                 parameter_binding: Optional[Dict[str, int]] = None,
                 **kwargs):
        """Initialize interface model"""
        # Call parent constructor
        super().__init__(definition)
        
        # Set fields
        self.tensor_dims = tensor_dims
        self.block_dims = block_dims
        self.skip_prob = skip_prob or []
        self.actual_utilization = actual_utilization
        self.parameter_binding = parameter_binding
        
        # Initialize dynamic fields first
        self._ipar = None
        self._stream_dims = None
        self._cached_metrics = {}  # Initialize early to avoid AttributeError
        
        # Call post init
        self.__post_init__()
        
        # Now set stream_dims if provided (will use setter)
        if stream_dims is not None:
            self.stream_dims = stream_dims
    
    def __post_init__(self):
        """Initialize model with optimized setup (minimal validation)"""
        # Normalize block_dims to list format for consistency
        if isinstance(self.block_dims, tuple):
            self.block_dims = [self.block_dims]
        
        # Default skip_prob if not provided
        if not self.skip_prob:
            self.skip_prob = [0.0] * self.n_phases
        
        # Store computed values for performance
        self._cached_metrics = {}
    
    @property
    def ipar(self) -> int:
        """Interface parallelism (flattened stream width)"""
        if self._ipar is not None:
            return self._ipar
        return 1  # Default parallelism
    
    @ipar.setter
    def ipar(self, value: int):
        """Set iPar and invalidate cached stream dimensions"""
        if value <= 0:
            raise ValueError(f"iPar must be positive, got {value}")
        self._ipar = value
        self._stream_dims = None  # Invalidate cache
        self._invalidate_performance_cache()
    
    @property
    def stream_dims(self) -> Shape:
        """Get stream dimensions (calculated from iPar if needed)"""
        if self._stream_dims is None:
            self._stream_dims = self._calculate_stream_dims()
        return self._stream_dims
    
    @stream_dims.setter
    def stream_dims(self, value: Shape):
        """Set stream dimensions and update iPar"""
        self._stream_dims = value
        self._ipar = prod(value) if value else 1
        self._invalidate_performance_cache()
    
    @property
    def n_phases(self) -> int:
        """Number of CSDF phases"""
        return len(self.block_dims)
    
    @property
    def is_csdf(self) -> bool:
        """Check if interface has cyclo-static behavior"""
        return self.n_phases > 1
    
    @property
    def rate_pattern(self) -> List[int]:
        """CSDF rate pattern for ADFG analysis
        
        Returns number of tokens (stream slices) per block for each phase.
        """
        if "rate_pattern" not in self._cached_metrics:
            self._cached_metrics["rate_pattern"] = [
                prod(bd) // prod(self.stream_dims) for bd in self.block_dims
            ]
        return self._cached_metrics["rate_pattern"]
    
    @property
    def ii_pattern(self) -> List[int]:
        """Initiation interval per phase
        
        Returns cycles needed to stream one block for each phase.
        """
        if "ii_pattern" not in self._cached_metrics:
            self._cached_metrics["ii_pattern"] = [
                math.ceil(prod(bd) / prod(self.stream_dims)) for bd in self.block_dims
            ]
        return self._cached_metrics["ii_pattern"]
    
    @property
    def tokens_per_inference(self) -> int:
        """Total tokens (blocks) per tensor"""
        if "tokens_per_inference" not in self._cached_metrics:
            total = 0
            for bd in self.block_dims:
                n_blocks = 1
                for t, b in zip(self.tensor_dims, bd):
                    n_blocks *= math.ceil(t / b)
                total += n_blocks
            self._cached_metrics["tokens_per_inference"] = total // self.n_phases
        return self._cached_metrics["tokens_per_inference"]
    
    @property
    def bandwidth_bits(self) -> int:
        """Bandwidth in bits per cycle"""
        if "bandwidth_bits" not in self._cached_metrics:
            dtype_bits = self.definition.dtype.bits if self.definition else 8
            self._cached_metrics["bandwidth_bits"] = self.ipar * dtype_bits
        return self._cached_metrics["bandwidth_bits"]
    
    @property
    def bandwidth_bytes(self) -> float:
        """Bandwidth in bytes per cycle"""
        return self.bandwidth_bits / 8.0
    
    def get_phase_info(self, phase: int) -> Dict[str, Any]:
        """Get information for a specific CSDF phase"""
        if phase >= self.n_phases:
            raise ValueError(f"Phase {phase} >= n_phases {self.n_phases}")
        
        bd = self.block_dims[phase]
        return {
            "block_dims": bd,
            "block_size": prod(bd),
            "rate": prod(bd) // prod(self.stream_dims),
            "ii": math.ceil(prod(bd) / prod(self.stream_dims)),
            "skip_prob": self.skip_prob[phase] if phase < len(self.skip_prob) else 0.0
        }
    
    def effective_rate(self) -> float:
        """Compute effective rate accounting for sparsity and utilization"""
        if "effective_rate" not in self._cached_metrics:
            if not any(self.skip_prob):
                base_rate = float(sum(self.rate_pattern)) / self.n_phases
            else:
                # Weight by (1 - skip_prob) for each phase
                total = 0.0
                for rate, skip in zip(self.rate_pattern, self.skip_prob):
                    total += rate * (1.0 - skip)
                base_rate = total / self.n_phases
            
            # Apply actual utilization factor
            self._cached_metrics["effective_rate"] = base_rate * self.actual_utilization
        
        return self._cached_metrics["effective_rate"]
    
    def effective_bandwidth(self, clock_freq_mhz: float = 100.0) -> float:
        """Compute effective bandwidth in MB/s accounting for utilization
        
        Args:
            clock_freq_mhz: Clock frequency in MHz
            
        Returns:
            Effective bandwidth in MB/s
        """
        cycles_per_second = clock_freq_mhz * 1e6
        bytes_per_cycle = self.bandwidth_bytes * self.actual_utilization
        return bytes_per_cycle * cycles_per_second / 1e6  # Convert to MB/s
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for this interface
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            # Basic properties
            "tensor_dims": self.tensor_dims,
            "block_dims": self.block_dims,
            "stream_dims": self.stream_dims,
            "n_phases": self.n_phases,
            "is_csdf": self.is_csdf,
            
            # Parallelism and rates
            "interface_parallelism": self.ipar,
            "rate_pattern": self.rate_pattern,
            "ii_pattern": self.ii_pattern,
            "effective_rate": self.effective_rate(),
            
            # Bandwidth
            "bandwidth_bits_per_cycle": self.bandwidth_bits,
            "bandwidth_bytes_per_cycle": self.bandwidth_bytes,
            "effective_bandwidth_100mhz_mbps": self.effective_bandwidth(100.0),
            
            # Volume
            "tokens_per_inference": self.tokens_per_inference,
            "total_tensor_size": prod(self.tensor_dims),
            
            # Utilization and sparsity
            "actual_utilization": self.actual_utilization,
            "skip_probabilities": self.skip_prob,
            "has_sparsity": any(self.skip_prob),
        }
        
        # Add phase-specific info for CSDF
        if self.is_csdf:
            metrics["phase_info"] = [
                self.get_phase_info(i) for i in range(self.n_phases)
            ]
        
        return metrics
    
    def calculate_memory_requirements(self) -> Dict[str, int]:
        """Calculate memory requirements for this interface
        
        Returns:
            Dictionary with memory requirements in bytes
        """
        dtype_bits = self.definition.dtype.bits if self.definition else 8
        bytes_per_element = dtype_bits // 8
        
        return {
            "tensor_memory_bytes": prod(self.tensor_dims) * bytes_per_element,
            "max_block_memory_bytes": max(prod(bd) for bd in self.block_dims) * bytes_per_element,
            "stream_buffer_bytes": prod(self.stream_dims) * bytes_per_element,
        }
    
    def simulate_tokens(self, n_inferences: int = 1) -> List[Dict[str, Any]]:
        """Simulate token generation for this interface
        
        Args:
            n_inferences: Number of inferences to simulate
            
        Returns:
            List of token information (one per generated token)
        """
        tokens = []
        
        for inf in range(n_inferences):
            for phase in range(self.n_phases):
                phase_info = self.get_phase_info(phase)
                
                # Calculate number of blocks for this phase
                bd = self.block_dims[phase]
                n_blocks = 1
                for t, b in zip(self.tensor_dims, bd):
                    n_blocks *= math.ceil(t / b)
                
                # Generate tokens for each block
                for block_idx in range(n_blocks):
                    tokens.append({
                        "inference": inf,
                        "phase": phase,
                        "block_index": block_idx,
                        "block_dims": bd,
                        "tokens": phase_info["rate"],
                        "cycles": phase_info["ii"],
                        "skip_prob": phase_info["skip_prob"]
                    })
        
        return tokens
    
    def validate_connection(self, other: 'InterfaceModel') -> List[str]:
        """Validate this interface can connect to another (for graph construction)
        
        Args:
            other: Target interface model
            
        Returns:
            List of connection errors (empty if valid)
        """
        errors = []
        
        # Check tensor dimensions match
        if self.tensor_dims != other.tensor_dims:
            errors.append(
                f"Tensor dimension mismatch: {self.tensor_dims} != {other.tensor_dims}"
            )
        
        # Check data types match (if we have definitions)
        if (self.definition and other.definition and 
            self.definition.dtype != other.definition.dtype):
            errors.append(
                f"Data type mismatch: {self.definition.dtype} != {other.definition.dtype}"
            )
        
        # Check rate compatibility for CSDF
        if self.is_csdf or other.is_csdf:
            if self.rate_pattern != other.rate_pattern:
                errors.append(
                    f"Rate pattern mismatch: {self.rate_pattern} != {other.rate_pattern}"
                )
        
        return errors
    
    def _calculate_stream_dims(self) -> Shape:
        """Calculate stream dimensions from iPar value
        
        Algorithm:
        1. Find first divisible dimension in block_dims
        2. Apply as much parallelism as possible to that dimension
        3. Remaining dimensions get stream_dim = 1
        """
        if self._ipar is None or self._ipar == 1:
            # No parallelism or unit parallelism
            return tuple(1 for _ in self.block_dims[0])
        
        # Use first block dims for calculation (phase 0)
        block_dims = self.block_dims[0]
        stream_dims = []
        remaining_par = self._ipar
        
        for i, block_dim in enumerate(block_dims):
            if remaining_par > 1 and block_dim > 1:
                # Apply as much parallelism as possible to this dimension
                dim_par = min(remaining_par, block_dim)
                
                # Find best divisor
                best_div = 1
                for div in range(dim_par, 0, -1):
                    if block_dim % div == 0 and remaining_par % div == 0:
                        best_div = div
                        break
                
                stream_dims.append(best_div)
                remaining_par //= best_div
            else:
                stream_dims.append(1)
        
        # Validate we used all parallelism
        if remaining_par > 1:
            # Could not fully apply parallelism - apply remainder to first viable dimension
            for i, (sd, bd) in enumerate(zip(stream_dims, block_dims)):
                if bd >= sd * remaining_par:
                    stream_dims[i] *= remaining_par
                    break
        
        return tuple(stream_dims)
    
    def _invalidate_performance_cache(self):
        """Invalidate cached performance metrics"""
        self._cached_metrics.clear()
    
    def clear_cache(self):
        """Clear cached performance metrics (for dynamic recalculation)"""
        self._cached_metrics.clear()
    
    def update_utilization(self, utilization: float):
        """Update actual utilization and clear dependent caches"""
        self.actual_utilization = max(0.0, min(1.0, utilization))
        # Clear caches that depend on utilization
        self._cached_metrics.pop("effective_rate", None)
    
    def update_sparsity(self, skip_probabilities: List[float]):
        """Update sparsity pattern and clear dependent caches"""
        if len(skip_probabilities) != self.n_phases:
            raise ValueError(f"skip_probabilities length must match n_phases ({self.n_phases})")
        
        self.skip_prob = [max(0.0, min(1.0, p)) for p in skip_probabilities]
        # Clear caches that depend on sparsity
        self._cached_metrics.pop("effective_rate", None)
    
    def __repr__(self) -> str:
        """String representation optimized for debugging"""
        return (
            f"InterfaceModel("
            f"tensor={self.tensor_dims}, "
            f"block={self.block_dims[0] if len(self.block_dims) == 1 else self.block_dims}, "
            f"stream={self.stream_dims}, "
            f"ipar={self.ipar}, "
            f"util={self.actual_utilization:.2f})"
        )