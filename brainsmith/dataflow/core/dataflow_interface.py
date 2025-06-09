"""
DataflowInterface: Core interface abstraction for hardware kernel interfaces

This module provides the foundational DataflowInterface class and supporting
data structures for representing hardware kernel interfaces with datatype
constraints and validation.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union

from .validation import (
    ValidationError, 
    ValidationResult, 
    ValidationSeverity,
    create_validation_result,
    create_divisibility_error,
    create_range_error,
    create_datatype_error
)

class DataflowInterfaceType(Enum):
    """Interface type hierarchy for dataflow modeling"""
    INPUT = "input"      # AXI-Stream input for activation data
    OUTPUT = "output"    # AXI-Stream output for result data  
    WEIGHT = "weight"    # AXI-Stream input for weight/parameter data
    CONFIG = "config"    # AXI-Lite for runtime configuration
    CONTROL = "control"  # Global control signals (clk, rst, etc.)

@dataclass
class DataflowDataType:
    """Enhanced datatype specification supporting FINN compatibility"""
    base_type: str       # INT, UINT, FLOAT, FIXED
    bitwidth: int        # Bit precision
    signed: bool         # Sign specification
    finn_type: str       # FINN DataType string representation
    
    def __post_init__(self):
        """Validate datatype specification after initialization"""
        valid_base_types = ["INT", "UINT", "FLOAT", "FIXED"]
        if self.base_type not in valid_base_types:
            raise ValueError(f"Invalid base_type '{self.base_type}'. Must be one of {valid_base_types}")
        
        if self.bitwidth <= 0:
            raise ValueError(f"Invalid bitwidth {self.bitwidth}. Must be positive integer")
        
        # Validate consistency between base_type and signed
        if self.base_type == "UINT" and self.signed:
            raise ValueError("UINT base_type cannot be signed")
        
        # Generate FINN type if not provided
        if not self.finn_type:
            self.finn_type = self._generate_finn_type()
    
    def _generate_finn_type(self) -> str:
        """Generate FINN DataType string representation"""
        if self.base_type in ["INT", "UINT"]:
            sign_prefix = "INT" if self.signed else "UINT"
            return f"{sign_prefix}{self.bitwidth}"
        elif self.base_type == "FIXED":
            sign_prefix = "INT" if self.signed else "UINT"
            return f"FIXED<{self.bitwidth},{sign_prefix}>"
        elif self.base_type == "FLOAT":
            return f"FLOAT{self.bitwidth}"
        else:
            return f"{self.base_type}{self.bitwidth}"

@dataclass
class DataTypeConstraint:
    """Constraint on allowed datatypes for an interface"""
    base_types: List[str]        # Allowed base types (INT, UINT, FLOAT, FIXED)
    min_bitwidth: int            # Minimum allowed bitwidth
    max_bitwidth: int            # Maximum allowed bitwidth
    signed_allowed: bool = True  # Allow signed types
    unsigned_allowed: bool = True # Allow unsigned types
    
    def __post_init__(self):
        """Validate constraint specification"""
        if self.min_bitwidth <= 0:
            raise ValueError("min_bitwidth must be positive")
        if self.max_bitwidth < self.min_bitwidth:
            raise ValueError("max_bitwidth must be >= min_bitwidth")
        if not self.signed_allowed and not self.unsigned_allowed:
            raise ValueError("At least one of signed_allowed or unsigned_allowed must be True")
    
    def is_valid_datatype(self, dtype: DataflowDataType) -> bool:
        """Check if a datatype satisfies this constraint"""
        # Check base type
        if dtype.base_type not in self.base_types:
            return False
        
        # Check bitwidth range
        if not (self.min_bitwidth <= dtype.bitwidth <= self.max_bitwidth):
            return False
        
        # Check sign constraints
        if dtype.signed and not self.signed_allowed:
            return False
        if not dtype.signed and not self.unsigned_allowed:
            return False
        
        return True

class ConstraintType(Enum):
    """Types of constraints that can be applied to interfaces"""
    DIVISIBILITY = "divisibility"      # Mathematical divisibility requirements
    RANGE = "range"                    # Parameter value ranges
    DEPENDENCY = "dependency"          # Inter-parameter dependencies
    RESOURCE = "resource"              # Resource limit constraints
    DATATYPE = "datatype"              # Datatype constraints

@dataclass
class Constraint:
    """Base constraint representation"""
    name: str
    constraint_type: ConstraintType
    parameters: Dict[str, Any]
    error_message: str

@dataclass
class DivisibilityConstraint(Constraint):
    """Divisibility constraint (e.g., tDim % sDim == 0)"""
    dividend: str      # Parameter name or expression
    divisor: str       # Parameter name or expression
    
    def __post_init__(self):
        self.constraint_type = ConstraintType.DIVISIBILITY

@dataclass
class RangeConstraint(Constraint):
    """Range constraint (e.g., 1 <= iPar <= tDim)"""
    parameter: str
    min_value: Union[int, str]  # Literal or parameter reference
    max_value: Union[int, str]  # Literal or parameter reference
    
    def __post_init__(self):
        self.constraint_type = ConstraintType.RANGE

@dataclass
class DataflowInterface:
    """
    Unified abstraction for hardware kernel interfaces providing 
    standardized representation of interface characteristics.
    
    Three-tier dimension system:
    - qDim: Query dimension (original tensor shape, e.g., 768 for BERT hidden size)
    - tDim: Tensor processing dimension (chunk size, e.g., 96 elements per chunk)  
    - sDim: Stream dimension (hardware parallelism, e.g., 8 elements per cycle)
    - num_tensors: Computed as qDim ÷ tDim (number of chunks to process)
    """
    name: str                           # Interface identifier (e.g., "in0", "out0", "weights")
    interface_type: DataflowInterfaceType  # INPUT, OUTPUT, WEIGHT, CONFIG, CONTROL
    qDim: List[int]                     # Query dimensions (original tensor shape)
    tDim: List[int]                     # Tensor chunk dimensions (size per chunk)
    sDim: List[int]                     # Stream dimensions (elements per clock cycle)
    dtype: DataflowDataType             # Element data type specification
    allowed_datatypes: List[DataTypeConstraint] = field(default_factory=list)  # Allowed datatypes from DATATYPE pragma
    axi_metadata: Dict[str, Any] = field(default_factory=dict)        # Protocol-specific metadata
    constraints: List[Constraint] = field(default_factory=list)       # Interface-specific constraints
    pragma_metadata: Dict[str, Any] = field(default_factory=dict)    # Pragma-derived information
    
    def __post_init__(self):
        """Validate interface after initialization"""
        self._validate_dimensions()
        self._set_default_constraints()
    
    def _validate_dimensions(self):
        """Validate dimension consistency with correct tensor chunking relationships"""
        if len(self.qDim) != len(self.tDim) or len(self.tDim) != len(self.sDim):
            raise ValueError(f"Interface {self.name}: qDim, tDim, and sDim must have same length")
        
        for i, (q, t, s) in enumerate(zip(self.qDim, self.tDim, self.sDim)):
            if q <= 0 or t <= 0 or s <= 0:
                raise ValueError(f"Interface {self.name}: All dimensions must be positive")
            
            # Validate tensor chunking relationship: qDim must be divisible by tDim
            if q % t != 0:
                raise ValueError(f"Interface {self.name}: qDim[{i}] ({q}) must be divisible by tDim[{i}] ({t}) for valid chunking")
            
            # Validate streaming relationship: tDim must be divisible by sDim
            if t % s != 0:
                raise ValueError(f"Interface {self.name}: tDim[{i}] ({t}) must be divisible by sDim[{i}] ({s}) for valid streaming")
    
    def _set_default_constraints(self):
        """Set default datatype constraints if none provided"""
        if not self.allowed_datatypes:
            # Default: allow common integer types from 1 to 32 bits
            default_constraint = DataTypeConstraint(
                base_types=["INT", "UINT"],
                min_bitwidth=1,
                max_bitwidth=32,
                signed_allowed=True,
                unsigned_allowed=True
            )
            self.allowed_datatypes = [default_constraint]
    
    def get_num_tensors(self) -> List[int]:
        """Calculate number of tensor chunks (qDim ÷ tDim) for each dimension"""
        return [q // t for q, t in zip(self.qDim, self.tDim)]
    
    def calculate_total_elements(self) -> int:
        """Calculate total number of elements across all dimensions"""
        return np.prod(self.qDim)
    
    def calculate_elements_per_chunk(self) -> int:
        """Calculate number of elements processed per chunk"""
        return np.prod(self.tDim)
    
    def calculate_total_chunks(self) -> int:
        """Calculate total number of chunks to process"""
        return np.prod(self.get_num_tensors())
    
    def calculate_stream_width(self) -> int:
        """Calculate AXI stream width based on sDim and dtype"""
        elements_per_cycle = np.prod(self.sDim)
        raw_width = elements_per_cycle * self.dtype.bitwidth
        # Align to 8-bit boundaries for AXI compatibility
        return ((raw_width + 7) // 8) * 8
    
    def validate_constraints(self) -> ValidationResult:
        """Validate interface constraints and configuration"""
        result = create_validation_result()
        
        # Validate dimension relationships
        for i, (q, t, s) in enumerate(zip(self.qDim, self.tDim, self.sDim)):
            # Check qDim divisibility by tDim for valid chunking
            if q % t != 0:
                error = create_divisibility_error(
                    self.name, f"qDim[{i}]", q, t
                )
                result.add_error(error)
            
            # Check tDim divisibility by sDim for valid streaming
            if t % s != 0:
                error = create_divisibility_error(
                    self.name, f"tDim[{i}]", t, s
                )
                result.add_error(error)
        
        # Validate datatype against constraints
        datatype_valid = self.validate_datatype(self.dtype)
        if not datatype_valid:
            allowed_types = []
            for constraint in self.allowed_datatypes:
                allowed_types.extend(constraint.base_types)
            error = create_datatype_error(
                self.name,
                f"{self.dtype.base_type}{self.dtype.bitwidth}",
                list(set(allowed_types))
            )
            result.add_error(error)
        
        return result
    
    def apply_parallelism(self, iPar: Optional[int] = None, wPar: Optional[int] = None) -> None:
        """Update sDim based on parallelism parameters"""
        if self.interface_type == DataflowInterfaceType.INPUT and iPar is not None:
            if len(self.sDim) > 0:
                self.sDim[0] = iPar
        elif self.interface_type == DataflowInterfaceType.WEIGHT and wPar is not None:
            if len(self.sDim) > 0:
                self.sDim[0] = wPar
        elif self.interface_type == DataflowInterfaceType.OUTPUT:
            # Output parallelism typically derived from input parallelism
            if iPar is not None and len(self.sDim) > 0:
                self.sDim[0] = iPar
    
    def get_axi_signals(self) -> Dict[str, Dict[str, Any]]:
        """Generate AXI signal specifications for this interface"""
        signals = {}
        
        if self.interface_type in [DataflowInterfaceType.INPUT, DataflowInterfaceType.WEIGHT]:
            # Slave AXI-Stream interface
            signals[f"{self.name}_TDATA"] = {
                "direction": "input",
                "width": self.calculate_stream_width(),
                "description": f"Input data stream for {self.name}"
            }
            signals[f"{self.name}_TVALID"] = {
                "direction": "input", 
                "width": 1,
                "description": f"Valid signal for {self.name}"
            }
            signals[f"{self.name}_TREADY"] = {
                "direction": "output",
                "width": 1, 
                "description": f"Ready signal for {self.name}"
            }
        elif self.interface_type == DataflowInterfaceType.OUTPUT:
            # Master AXI-Stream interface
            signals[f"{self.name}_TDATA"] = {
                "direction": "output",
                "width": self.calculate_stream_width(),
                "description": f"Output data stream for {self.name}"
            }
            signals[f"{self.name}_TVALID"] = {
                "direction": "output",
                "width": 1,
                "description": f"Valid signal for {self.name}"
            }
            signals[f"{self.name}_TREADY"] = {
                "direction": "input",
                "width": 1,
                "description": f"Ready signal for {self.name}"
            }
        
        return signals
    
    def validate_datatype(self, target_dtype: DataflowDataType) -> bool:
        """Validate that target datatype is allowed for this interface"""
        for constraint in self.allowed_datatypes:
            if constraint.is_valid_datatype(target_dtype):
                return True
        return False
    
    def validate_datatype_string(self, dtype_string: str) -> bool:
        """
        Validate if a datatype string is allowed for this interface.
        
        Args:
            dtype_string: FINN datatype string (e.g., "UINT8", "INT16")
            
        Returns:
            bool: True if datatype is valid for this interface
        """
        try:
            # Parse the datatype string to extract components
            import re
            
            # Extract base type and bitwidth from FINN datatype string
            match = re.match(r'([A-Z]+)(\d+)', dtype_string)
            if not match:
                return False
            
            base_type = match.group(1)
            bitwidth = int(match.group(2))
            
            # Determine if signed based on base type
            signed = base_type.startswith('INT') and not base_type.startswith('UINT')
            
            # Create DataflowDataType for validation
            target_dtype = DataflowDataType(
                base_type=base_type,
                bitwidth=bitwidth,
                signed=signed,
                finn_type=dtype_string
            )
            
            return self.validate_datatype(target_dtype)
        except Exception:
            return False
    
    def get_memory_footprint(self) -> int:
        """Calculate total memory footprint in bits"""
        total_elements = np.prod(self.get_num_tensors()) * np.prod(self.tDim)
        return total_elements * self.dtype.bitwidth
    
    def get_transfer_cycles(self) -> int:
        """Calculate number of transfer cycles needed"""
        total_elements = np.prod(self.tDim)
        elements_per_cycle = np.prod(self.sDim)
        return (total_elements + elements_per_cycle - 1) // elements_per_cycle
    
    def reconstruct_tensor_shape(self) -> List[int]:
        """
        Reconstruct original tensor shape from num_tensors and tDim using broadcasting
        
        Mathematical relationship: original_tensor_shape = num_tensors × tDim
        
        Returns:
            List[int]: Original tensor shape before chunking
        """
        return self._broadcast_tensor_shape(self.get_num_tensors(), self.tDim)
    
    def _broadcast_tensor_shape(self, num_tensors: List[int], tDim: List[int]) -> List[int]:
        """
        Broadcast num_tensors and tDim to reconstruct original tensor shape
        
        Args:
            num_tensors: Number of tensor chunks
            tDim: Tensor chunk dimensions
            
        Returns:
            List[int]: Broadcasted tensor shape
        """
        if len(num_tensors) == 1 and len(tDim) == 1:
            # Simple case: [num_tensors[0] * tDim[0]]
            return [num_tensors[0] * tDim[0]]
        elif len(num_tensors) == len(tDim):
            # Element-wise multiplication
            return [n * t for n, t in zip(num_tensors, tDim)]
        else:
            # More complex broadcasting - concatenate dimensions
            return num_tensors + tDim
    
    def validate_tensor_chunking(self, original_shape: List[int]) -> ValidationResult:
        """
        Validate that num_tensors/tDim correctly chunk the original tensor shape
        
        Args:
            original_shape: Original tensor shape to validate against
            
        Returns:
            ValidationResult with any chunking errors
        """
        result = create_validation_result()
        
        try:
            reconstructed = self.reconstruct_tensor_shape()
            
            # Check if reconstruction matches original (allowing for different representations)
            total_elements_original = np.prod(original_shape)
            total_elements_reconstructed = np.prod(reconstructed)
            
            if total_elements_original != total_elements_reconstructed:
                error = ValidationError(
                    component=f"interface.{self.name}",
                    error_type="tensor_chunking_mismatch",
                    message=f"Tensor chunking mismatch: original shape {original_shape} "
                           f"has {total_elements_original} elements, but num_tensors={self.get_num_tensors()} × tDim={self.tDim} "
                           f"gives {total_elements_reconstructed} elements",
                    severity=ValidationSeverity.ERROR,
                    context={
                        "original_shape": original_shape,
                        "num_tensors": self.get_num_tensors(),
                        "tDim": self.tDim,
                        "reconstructed_shape": reconstructed
                    }
                )
                result.add_error(error)
        except Exception as e:
            error = ValidationError(
                component=f"interface.{self.name}",
                error_type="tensor_reconstruction_error",
                message=f"Failed to reconstruct tensor shape: {str(e)}",
                severity=ValidationSeverity.ERROR,
                context={"exception": str(e)}
            )
            result.add_error(error)
        
        return result
    
    @classmethod
    def from_tensor_chunking(cls, name: str, interface_type: DataflowInterfaceType,
                            original_shape: List[int], tDim: List[int],
                            dtype: DataflowDataType, chunking_mode: str = "broadcast",
                            **kwargs) -> 'DataflowInterface':
        """
        Factory method to create DataflowInterface from tensor chunking specification
        
        Args:
            name: Interface name
            interface_type: Type of interface
            original_shape: Original tensor shape before chunking
            tDim: Desired tensor dimensions per calculation
            dtype: Data type specification
            chunking_mode: How to compute qDim ("broadcast", "divide", "explicit")
            **kwargs: Additional interface parameters
            
        Returns:
            DataflowInterface with computed qDim
        """
        qDim = cls._compute_qDim_from_chunking(original_shape, tDim, chunking_mode)
        
        # Ensure all dimensions have the same length
        if len(qDim) != len(tDim):
            if len(qDim) == 1 and len(tDim) > 1:
                # Expand qDim to match tDim length with 1s
                qDim = qDim + [1] * (len(tDim) - 1)
            elif len(tDim) == 1 and len(qDim) > 1:
                # Keep qDim as is, expand tDim
                tDim = tDim + [1] * (len(qDim) - 1)
            else:
                # For other mismatches, make them all the same length
                max_len = max(len(qDim), len(tDim))
                qDim = qDim + [1] * (max_len - len(qDim))
                tDim = tDim + [1] * (max_len - len(tDim))
        
        # Initialize sDim to match tDim (will be updated by parallelism)
        sDim = tDim.copy()
        
        return cls(
            name=name,
            interface_type=interface_type,
            qDim=qDim,
            tDim=tDim,
            sDim=sDim,
            dtype=dtype,
            **kwargs
        )
    
    @staticmethod
    def _compute_qDim_from_chunking(original_shape: List[int], tDim: List[int],
                                   chunking_mode: str = "broadcast") -> List[int]:
        """
        Compute qDim from original tensor shape and desired tDim
        
        Args:
            original_shape: Original tensor shape
            tDim: Desired tensor dimensions
            chunking_mode: Computation method
            
        Returns:
            List[int]: Computed qDim
            
        Examples:
            original_shape=[30, 50], tDim=[50] → qDim=[30] (chunking along last dim)
            original_shape=[10, 30, 50], tDim=[3, 5] → qDim=[1000] (where total=15000, tDim=15)
        """
        if chunking_mode == "broadcast":
            total_elements = np.prod(original_shape)
            tDim_elements = np.prod(tDim)
            
            if len(tDim) == 1:
                # Single tDim: compute qDim such that qDim * tDim = total_elements
                # For original_shape=[30, 50] with tDim=[50], qDim should be [30]
                if len(original_shape) == 2 and tDim[0] == original_shape[-1]:
                    return [original_shape[0]]  # Return first dimension
                elif len(original_shape) == 1:
                    qDim_val = total_elements // tDim[0]
                    return [qDim_val]
                else:
                    # Find matching dimension and compute qDim from remaining
                    for i, dim in enumerate(original_shape):
                        if dim == tDim[0]:
                            # Remove this dimension and compute product of others
                            remaining_dims = original_shape[:i] + original_shape[i+1:]
                            if remaining_dims:
                                return [np.prod(remaining_dims)]
                            else:
                                return [1]
                    # If no exact match, compute total division
                    qDim_val = max(1, total_elements // tDim[0])
                    return [qDim_val]
            else:
                # Multiple tDim: compute qDim such that qDim × tDim = original_shape elements
                qDim_elements = max(1, total_elements // tDim_elements)
                return [qDim_elements]
        
        elif chunking_mode == "divide":
            # Direct division of original_shape by tDim
            if len(original_shape) != len(tDim):
                raise ValueError(f"For divide mode, original_shape and tDim must have same length")
            return [max(1, orig // t) for orig, t in zip(original_shape, tDim)]
        
        else:
            raise ValueError(f"Unknown chunking_mode: {chunking_mode}")
    
    def __str__(self) -> str:
        """String representation of interface"""
        return (f"DataflowInterface(name='{self.name}', "
                f"type={self.interface_type.value}, "
                f"qDim={self.qDim}, tDim={self.tDim}, sDim={self.sDim}, "
                f"dtype={self.dtype.finn_type})")
