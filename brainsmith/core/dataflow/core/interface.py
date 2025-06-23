############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unified interface definition"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Set
import math
from .types import (
    Shape, RaggedShape, InterfaceDirection, DataType,
    prod, is_valid_tiling
)


@dataclass
class Interface:
    """Hardware interface with data hierarchy
    
    Represents a streaming interface on a hardware kernel with:
    - Tensor level: Full data for one inference/execution
    - Block level: Data for one kernel calculation/firing
    - Stream level: Data transferred per clock cycle
    
    Supports both uniform and ragged (CSDF) tiling patterns.
    """
    
    # Core properties
    name: str
    direction: InterfaceDirection
    dtype: DataType
    
    # Data hierarchy
    tensor_dims: Shape
    block_dims: RaggedShape  # Can be single shape or list for CSDF
    stream_dims: Optional[Shape] = None  # Will default based on tensor_dims
    
    # Advanced features
    skip_prob: List[float] = field(default_factory=list)  # Sparsity per phase
    optional: bool = False  # For conditional interfaces
    
    # Native constraints
    alignment: Optional[int] = None          # Memory alignment requirement in bytes
    min_dims: Optional[Shape] = None         # Minimum dimension sizes
    max_dims: Optional[Shape] = None         # Maximum dimension sizes
    granularity: Optional[Shape] = None      # Dimension granularity (must be multiples)
    
    # Dataflow metadata
    produces: Set[str] = field(default_factory=set)      # Interfaces this feeds
    consumes: Set[str] = field(default_factory=set)      # Interfaces this reads from
    synchronized_with: Set[str] = field(default_factory=set)  # Must process together
    
    def __post_init__(self):
        """Validate and normalize inputs"""
        # Validate types
        if not isinstance(self.name, str):
            raise TypeError(f"Interface name must be string, got {type(self.name)}")
        
        if not isinstance(self.direction, InterfaceDirection):
            raise TypeError(f"Direction must be InterfaceDirection, got {type(self.direction)}")
        
        if not isinstance(self.dtype, DataType):
            raise TypeError(f"Dtype must be DataType, got {type(self.dtype)}")
        
        # Default stream_dims to all 1s matching tensor dimensions
        if self.stream_dims is None:
            self.stream_dims = tuple(1 for _ in self.tensor_dims)
        
        # Normalize block_dims to list format
        if isinstance(self.block_dims, tuple):
            self.block_dims = [self.block_dims]
        elif not isinstance(self.block_dims, list):
            raise TypeError(f"block_dims must be Shape or List[Shape], got {type(self.block_dims)}")
        
        # Validate dimensions
        if len(self.tensor_dims) == 0:
            raise ValueError("Tensor dimensions cannot be empty")
        
        for i, bd in enumerate(self.block_dims):
            if len(bd) != len(self.tensor_dims):
                raise ValueError(
                    f"Block dims[{i}] has {len(bd)} dimensions, "
                    f"but tensor has {len(self.tensor_dims)}"
                )
            if not is_valid_tiling(self.tensor_dims, bd):
                raise ValueError(
                    f"Block dims[{i}] {bd} cannot tile tensor dims {self.tensor_dims}"
                )
        
        if len(self.stream_dims) != len(self.tensor_dims):
            raise ValueError(
                f"Stream dims has {len(self.stream_dims)} dimensions, "
                f"but tensor has {len(self.tensor_dims)}"
            )
        
        # Validate stream fits in smallest block
        min_block = self._get_min_block_dims()
        for i, (s, b) in enumerate(zip(self.stream_dims, min_block)):
            if s > b:
                raise ValueError(
                    f"Stream dim[{i}]={s} exceeds min block dim[{i}]={b}"
                )
        
        # Normalize skip_prob
        if self.skip_prob:
            if len(self.skip_prob) != self.n_phases:
                raise ValueError(
                    f"skip_prob length {len(self.skip_prob)} != n_phases {self.n_phases}"
                )
            for p in self.skip_prob:
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"skip_prob values must be in [0,1], got {p}")
        else:
            # Default: no sparsity
            self.skip_prob = [0.0] * self.n_phases
        
        # Validate native constraints
        self._validate_native_constraints()
    
    @property
    def ipar(self) -> int:
        """Interface parallelism (flattened stream width)"""
        return prod(self.stream_dims)
    
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
        return [prod(bd) // prod(self.stream_dims) 
                for bd in self.block_dims]
    
    @property
    def ii_pattern(self) -> List[int]:
        """Initiation interval per phase
        
        Returns cycles needed to stream one block for each phase.
        """
        return [math.ceil(prod(bd) / prod(self.stream_dims))
                for bd in self.block_dims]
    
    @property
    def tokens_per_inference(self) -> int:
        """Total tokens (blocks) per tensor"""
        total = 0
        for bd in self.block_dims:
            n_blocks = 1
            for t, b in zip(self.tensor_dims, bd):
                n_blocks *= math.ceil(t / b)
            total += n_blocks
        return total // self.n_phases  # Average if CSDF
    
    @property
    def bandwidth_bits(self) -> int:
        """Bandwidth in bits per cycle"""
        return self.ipar * self.dtype.bits
    
    def _get_min_block_dims(self) -> Shape:
        """Get minimum block dimensions across all phases"""
        if not self.block_dims:
            return self.tensor_dims
        
        min_dims = list(self.block_dims[0])
        for bd in self.block_dims[1:]:
            for i, d in enumerate(bd):
                min_dims[i] = min(min_dims[i], d)
        
        return tuple(min_dims)
    
    def get_phase_info(self, phase: int) -> dict:
        """Get information for a specific CSDF phase"""
        if phase >= self.n_phases:
            raise ValueError(f"Phase {phase} >= n_phases {self.n_phases}")
        
        bd = self.block_dims[phase]
        return {
            "block_dims": bd,
            "block_size": prod(bd),
            "rate": prod(bd) // prod(self.stream_dims),
            "ii": math.ceil(prod(bd) / prod(self.stream_dims)),
            "skip_prob": self.skip_prob[phase]
        }
    
    def effective_rate(self) -> float:
        """Compute effective rate accounting for sparsity"""
        if not any(self.skip_prob):
            return float(sum(self.rate_pattern)) / self.n_phases
        
        # Weight by (1 - skip_prob) for each phase
        total = 0.0
        for rate, skip in zip(self.rate_pattern, self.skip_prob):
            total += rate * (1.0 - skip)
        
        return total / self.n_phases
    
    def validate_connection(self, other: "Interface") -> None:
        """Validate this interface can connect to another
        
        Checks:
        - Compatible directions (output -> input)
        - Same tensor dimensions
        - Same data type
        """
        # Check directions
        valid_connections = [
            (InterfaceDirection.OUTPUT, InterfaceDirection.INPUT),
            (InterfaceDirection.OUTPUT, InterfaceDirection.WEIGHT),
        ]
        
        if (self.direction, other.direction) not in valid_connections:
            raise ValueError(
                f"Cannot connect {self.direction.value} to {other.direction.value}"
            )
        
        # Check tensor dimensions
        if self.tensor_dims != other.tensor_dims:
            raise ValueError(
                f"Tensor dimension mismatch: {self.tensor_dims} != {other.tensor_dims}"
            )
        
        # Check data types
        if self.dtype != other.dtype:
            raise ValueError(
                f"Data type mismatch: {self.dtype} != {other.dtype}"
            )
    
    def _validate_native_constraints(self):
        """Validate native interface constraints during construction"""
        errors = self.validate_constraints()
        if errors:
            raise ValueError(f"Interface constraint violations:\n" + "\n".join(errors))
    
    def validate_constraints(self) -> List[str]:
        """Validate interface-level constraints
        
        Returns:
            List of constraint violation messages (empty if valid)
        """
        errors = []
        
        # Check alignment constraint
        if self.alignment is not None:
            if self.alignment <= 0:
                errors.append(f"Alignment must be positive, got {self.alignment}")
            
            total_size = prod(self.tensor_dims) * self.dtype.bits // 8  # Size in bytes
            if total_size % self.alignment != 0:
                errors.append(
                    f"Interface {self.name} total size {total_size} bytes "
                    f"not aligned to {self.alignment} bytes"
                )
        
        # Check dimension bounds
        if self.min_dims is not None:
            if len(self.min_dims) != len(self.tensor_dims):
                errors.append(
                    f"min_dims length {len(self.min_dims)} != "
                    f"tensor dims length {len(self.tensor_dims)}"
                )
            else:
                for i, (actual, min_val) in enumerate(zip(self.tensor_dims, self.min_dims)):
                    if actual < min_val:
                        errors.append(
                            f"Interface {self.name} dim[{i}]={actual} < min={min_val}"
                        )
        
        if self.max_dims is not None:
            if len(self.max_dims) != len(self.tensor_dims):
                errors.append(
                    f"max_dims length {len(self.max_dims)} != "
                    f"tensor dims length {len(self.tensor_dims)}"
                )
            else:
                for i, (actual, max_val) in enumerate(zip(self.tensor_dims, self.max_dims)):
                    if actual > max_val:
                        errors.append(
                            f"Interface {self.name} dim[{i}]={actual} > max={max_val}"
                        )
        
        # Check granularity constraints
        if self.granularity is not None:
            if len(self.granularity) != len(self.tensor_dims):
                errors.append(
                    f"granularity length {len(self.granularity)} != "
                    f"tensor dims length {len(self.tensor_dims)}"
                )
            else:
                for i, (actual, gran) in enumerate(zip(self.tensor_dims, self.granularity)):
                    if gran is not None and gran > 0 and actual % gran != 0:
                        errors.append(
                            f"Interface {self.name} dim[{i}]={actual} "
                            f"not divisible by granularity {gran}"
                        )
        
        return errors
    
    def add_produces(self, interface_name: str):
        """Add interface to produces set"""
        self.produces.add(interface_name)
    
    def add_consumes(self, interface_name: str):
        """Add interface to consumes set"""
        self.consumes.add(interface_name)
    
    def add_synchronized_with(self, interface_name: str):
        """Add interface to synchronized_with set"""
        self.synchronized_with.add(interface_name)
    
    def has_constraint(self, constraint_type: str) -> bool:
        """Check if interface has a specific type of constraint"""
        if constraint_type == "alignment":
            return self.alignment is not None
        elif constraint_type == "bounds":
            return self.min_dims is not None or self.max_dims is not None
        elif constraint_type == "granularity":
            return self.granularity is not None
        else:
            return False
    
    def __repr__(self) -> str:
        """String representation"""
        constraints = []
        if self.alignment is not None:
            constraints.append(f"align={self.alignment}")
        if self.min_dims is not None:
            constraints.append(f"min={self.min_dims}")
        if self.max_dims is not None:
            constraints.append(f"max={self.max_dims}")
        if self.granularity is not None:
            constraints.append(f"gran={self.granularity}")
        
        constraint_str = f", {', '.join(constraints)}" if constraints else ""
        
        return (
            f"Interface(name='{self.name}', "
            f"dir={self.direction.value}, "
            f"dtype={self.dtype}, "
            f"tensor={self.tensor_dims}, "
            f"block={self.block_dims if self.is_csdf else self.block_dims[0]}, "
            f"stream={self.stream_dims}"
            f"{constraint_str})"
        )