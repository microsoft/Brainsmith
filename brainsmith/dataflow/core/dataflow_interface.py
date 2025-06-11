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

from .interface_types import InterfaceType
from .validation import (
    ValidationError, 
    ValidationResult, 
    ValidationSeverity,
    create_validation_result,
    create_divisibility_error,
    create_range_error,
    create_datatype_error,
    validate_positive_integers,
    validate_dimension_relationships
)

# Legacy InterfaceType removed - use InterfaceType directly

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
    """Divisibility constraint (e.g., block_dims % stream_dims == 0)"""
    dividend: str      # Parameter name or expression
    divisor: str       # Parameter name or expression
    
    def __post_init__(self):
        self.constraint_type = ConstraintType.DIVISIBILITY

@dataclass
class RangeConstraint(Constraint):
    """Range constraint (e.g., 1 <= iPar <= block_dims)"""
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
    
    Three-tier dimension system for Interface-Wise Dataflow Modeling:
    - tensor_dims: Full tensor dimensions (original input tensor shape before any processing)
      Example: BERT input [1,128,768] where 1=batch, 128=sequence length, 768=hidden dimension
    - block_dims: Block dimensions (processing chunk shape for each operation) 
      Example: [1,8,96] meaning process 8 sequence elements with 96 features per chunk
    - stream_dims: Stream dimensions (hardware parallelism - elements processed per clock cycle)
      Example: [1,1,8] meaning 8 features processed in parallel each cycle
    - num_blocks: Computed as tensor_dims ÷ block_dims (number of chunks to process)
      Example: [1,128,768] ÷ [1,8,96] = [1,16,8] chunks total
    """
    name: str                           # Interface identifier (e.g., "in0", "out0", "weights")
    interface_type: InterfaceType  # INPUT, OUTPUT, WEIGHT interface type
    tensor_dims: List[int]              # Full tensor dimensions: original input tensor shape
    block_dims: List[int]               # Block dimensions: processing chunk shape
    stream_dims: List[int]                     # Stream dimensions: elements per clock cycle (hardware parallelism)
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
        """
        Validate dimension consistency with flexible length requirements.
        
        tensor_dims, block_dims, and stream_dims can have different lengths to support various tensor shapes:
        - tensor_dims: Can be multi-dimensional (e.g., [1,128,768] for BERT)
        - block_dims: Can have fewer dimensions (e.g., [128] for 1D processing)  
        - stream_dims: Can be single dimension (e.g., [8] for parallelism)
        """
        # Basic validation - all dimensions must be non-empty and positive
        for dim_list, name in [(self.tensor_dims, "tensor_dims"), (self.block_dims, "block_dims"), (self.stream_dims, "stream_dims")]:
            if not dim_list:
                raise ValueError(f"Interface {self.name}: {name} cannot be empty")
            for i, dim in enumerate(dim_list):
                if dim <= 0:
                    raise ValueError(f"Interface {self.name}: {name}[{i}] ({dim}) must be positive")
        
        # Validate chunking relationship between tensor_dims and block_dims
        # Only validate dimensions that exist in both tensor_dims and block_dims
        min_dims = min(len(self.tensor_dims), len(self.block_dims))
        for i in range(min_dims):
            tensor_dim = self.tensor_dims[i]
            block_dim = self.block_dims[i]
            if tensor_dim % block_dim != 0:
                raise ValueError(f"Interface {self.name}: tensor_dims[{i}] ({tensor_dim}) must be divisible by block_dims[{i}] ({block_dim}) for valid chunking")
        
        # Validate streaming relationship between block_dims and stream_dims  
        # Only validate dimensions that exist in both block_dims and stream_dims
        min_stream_dims = min(len(self.block_dims), len(self.stream_dims))
        for i in range(min_stream_dims):
            block_dim = self.block_dims[i]
            stream_dim = self.stream_dims[i]
            if block_dim % stream_dim != 0:
                raise ValueError(f"Interface {self.name}: block_dims[{i}] ({block_dim}) must be divisible by stream_dims[{i}] ({stream_dim}) for valid streaming")
    
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
    
    def get_num_blocks(self) -> List[int]:
        """
        Calculate number of processing blocks (tensor_dims ÷ block_dims) for each dimension.
        
        This represents how many tensor blocks must be processed to handle
        the complete tensor (original input tensor).
        
        Only calculates for dimensions that exist in both tensor_dims and block_dims.
        
        Returns:
            List[int]: Number of blocks per dimension (length = min(len(tensor_dims), len(block_dims)))
        """
        min_dims = min(len(self.tensor_dims), len(self.block_dims))
        return [self.tensor_dims[i] // self.block_dims[i] for i in range(min_dims)]
    
    def calculate_total_elements(self) -> int:
        """Calculate total number of elements in original input tensor (tensor_dims)"""
        return np.prod(self.tensor_dims)
    
    def calculate_elements_per_block(self) -> int:
        """Calculate number of elements in each processing block (block_dims)"""
        return np.prod(self.block_dims)
    
    def calculate_total_blocks(self) -> int:
        """Calculate total number of processing blocks (num_blocks)"""
        return np.prod(self.get_num_blocks())
    
    def calculate_stream_width(self) -> int:
        """Calculate AXI stream width based on stream_dims and dtype"""
        elements_per_cycle = np.prod(self.stream_dims)
        raw_width = elements_per_cycle * self.dtype.bitwidth
        # Align to 8-bit boundaries for AXI compatibility
        return ((raw_width + 7) // 8) * 8
    
    def validate_constraints(self) -> ValidationResult:
        """Validate interface constraints and configuration"""
        result = create_validation_result()
        
        # Validate dimension relationships
        for i, (tensor_dim, block_dim, stream_dim) in enumerate(zip(self.tensor_dims, self.block_dims, self.stream_dims)):
            # Check tensor_dims divisibility by block_dims for valid chunking
            if tensor_dim % block_dim != 0:
                error = create_divisibility_error(
                    self.name, f"tensor_dims[{i}]", tensor_dim, block_dim
                )
                result.add_error(error)
            
            # Check block_dims divisibility by stream_dims for valid streaming
            if block_dim % stream_dim != 0:
                error = create_divisibility_error(
                    self.name, f"block_dims[{i}]", block_dim, stream_dim
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
    
    def validate(self) -> ValidationResult:
        """Comprehensive validation of interface configuration.
        
        Validates all aspects of the interface including:
        - Dimension validity and relationships
        - Datatype constraints
        - Interface-specific constraints
        - Mathematical axiom compliance
        
        Returns:
            ValidationResult with detailed validation feedback
        """
        result = ValidationResult(True)
        
        # Validate that all dimensions are positive integers
        tensor_result = validate_positive_integers(self.tensor_dims, "tensor_dims")
        result.merge(tensor_result)
        
        block_result = validate_positive_integers(self.block_dims, "block_dims")
        result.merge(block_result)
        
        stream_result = validate_positive_integers(self.stream_dims, "stream_dims")
        result.merge(stream_result)
        
        # Validate dimension relationships follow axioms
        if result.is_valid:  # Only if basic validation passed
            dim_result = validate_dimension_relationships(
                self.tensor_dims, self.block_dims, self.stream_dims
            )
            result.merge(dim_result)
        
        # Validate datatype constraints
        if self.allowed_datatypes:
            datatype_valid = False
            for constraint in self.allowed_datatypes:
                if constraint.is_valid_datatype(self.dtype):
                    datatype_valid = True
                    break
            
            if not datatype_valid:
                allowed_types = []
                for constraint in self.allowed_datatypes:
                    allowed_types.extend(constraint.base_types)
                result.add_error(
                    f"Datatype {self.dtype.finn_type} not allowed. Must be one of: {list(set(allowed_types))}",
                    {"current_datatype": self.dtype.finn_type, "allowed_types": list(set(allowed_types))}
                )
        
        # Validate interface-specific constraints
        constraint_result = self.validate_constraints()
        result.merge(constraint_result)
        
        # Add interface-specific validations based on type
        if self.interface_type == InterfaceType.WEIGHT:
            # Weight interfaces should have reasonable dimensions for hardware
            total_elements = self.calculate_total_elements()
            if total_elements > 1000000:  # 1M elements threshold
                result.add_warning(
                    f"Weight interface has {total_elements} elements, which may require significant memory",
                    {"total_elements": total_elements, "memory_bits": self.get_memory_footprint()}
                )
        
        # Validate stream width is reasonable for AXI
        stream_width = self.calculate_stream_width()
        if stream_width > 1024:  # Common AXI width limit
            result.add_warning(
                f"Stream width of {stream_width} bits may exceed typical AXI limits",
                {"stream_width": stream_width, "elements_per_cycle": np.prod(self.stream_dims)}
            )
        
        return result
    
    def apply_parallelism(self, iPar: Optional[int] = None, wPar: Optional[int] = None) -> None:
        """Update stream_dims based on parallelism parameters"""
        if self.interface_type == InterfaceType.INPUT and iPar is not None:
            if len(self.stream_dims) > 0:
                self.stream_dims[0] = iPar
        elif self.interface_type == InterfaceType.WEIGHT and wPar is not None:
            if len(self.stream_dims) > 0:
                self.stream_dims[0] = wPar
        elif self.interface_type == InterfaceType.OUTPUT:
            # Output parallelism typically derived from input parallelism
            if iPar is not None and len(self.stream_dims) > 0:
                self.stream_dims[0] = iPar
    
    def calculate_cII(self) -> int:
        """
        Calculate calculation initiation interval (cII) for this interface.
        
        cII represents the number of cycles between consecutive calculations
        for this interface: cII = ∏(block_dims_i / stream_dims_i)
        
        Only calculates for dimensions that exist in both block_dims and stream_dims.
        
        Returns:
            int: Calculation initiation interval in cycles
        """
        cII = 1
        min_dims = min(len(self.block_dims), len(self.stream_dims))
        for i in range(min_dims):
            bdim = self.block_dims[i]
            sdim = self.stream_dims[i]
            if sdim > 0:
                cII *= bdim // sdim
        return max(cII, 1)
    
    def get_axi_signals(self) -> Dict[str, Dict[str, Any]]:
        """Generate AXI signal specifications for this interface"""
        signals = {}
        
        if self.interface_type in [InterfaceType.INPUT, InterfaceType.WEIGHT]:
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
        elif self.interface_type == InterfaceType.OUTPUT:
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
        total_elements = np.prod(self.get_num_blocks()) * np.prod(self.block_dims)
        return total_elements * self.dtype.bitwidth
    
    def get_transfer_cycles(self) -> int:
        """Calculate number of transfer cycles needed"""
        total_elements = np.prod(self.block_dims)
        elements_per_cycle = np.prod(self.stream_dims)
        return (total_elements + elements_per_cycle - 1) // elements_per_cycle
    
    def calculate_memory_footprint(self) -> Dict[str, int]:
        """Calculate memory requirements for this interface.
        
        Returns:
            Dict containing memory analysis:
            - total_bits: Total memory required in bits
            - total_bytes: Total memory required in bytes  
            - buffers: Buffer requirements breakdown
        """
        total_bits = self.get_memory_footprint()
        total_bytes = (total_bits + 7) // 8  # Round up to bytes
        
        # Calculate buffer requirements
        block_buffer_bits = np.prod(self.block_dims) * self.dtype.bitwidth
        stream_buffer_bits = np.prod(self.stream_dims) * self.dtype.bitwidth
        
        return {
            "total_bits": total_bits,
            "total_bytes": total_bytes,
            "buffers": {
                "block_buffer_bits": block_buffer_bits,
                "stream_buffer_bits": stream_buffer_bits,
                "num_blocks": self.calculate_total_blocks()
            }
        }
    
    def analyze_resource_requirements(self) -> Dict[str, Any]:
        """Analyze resource requirements using ResourceAnalyzer.
        
        Returns:
            Dict containing comprehensive resource analysis
        """
        # Import ResourceAnalyzer here to avoid circular imports
        from .resource_analysis import ResourceAnalyzer
        
        analyzer = ResourceAnalyzer()
        requirements = analyzer.analyze_interface(self)
        
        return requirements.get_summary()
    
    
    def reconstruct_tensor_shape(self) -> List[int]:
        """
        Reconstruct original tensor shape from num_blocks and block_dims using broadcasting
        
        Mathematical relationship: original_tensor_shape = num_blocks × block_dims
        
        Returns:
            List[int]: Original tensor shape before chunking
        """
        return self._broadcast_tensor_shape(self.get_num_blocks(), self.block_dims)
    
    def _broadcast_tensor_shape(self, num_blocks: List[int], block_dims: List[int]) -> List[int]:
        """
        Broadcast num_blocks and block_dims to reconstruct original tensor shape
        
        Args:
            num_blocks: Number of tensor blocks
            block_dims: Tensor block dimensions
            
        Returns:
            List[int]: Broadcasted tensor shape
        """
        if len(num_blocks) == 1 and len(block_dims) == 1:
            # Simple case: [num_blocks[0] * block_dims[0]]
            return [num_blocks[0] * block_dims[0]]
        elif len(num_blocks) == len(block_dims):
            # Element-wise multiplication
            return [n * t for n, t in zip(num_blocks, block_dims)]
        else:
            # More complex broadcasting - concatenate dimensions
            return num_blocks + block_dims
    
    def validate_tensor_chunking(self, original_shape: List[int]) -> ValidationResult:
        """
        Validate that tensor chunking correctly chunks the original tensor shape
        
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
                           f"has {total_elements_original} elements, but num_blocks={self.get_num_blocks()} × block_dims={self.block_dims} "
                           f"gives {total_elements_reconstructed} elements",
                    severity=ValidationSeverity.ERROR,
                    context={
                        "original_shape": original_shape,
                        "num_blocks": self.get_num_blocks(),
                        "block_dims": self.block_dims,
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
    def from_tensor_chunking(cls, name: str, interface_type: InterfaceType,
                            original_shape: List[int], block_dims: List[int],
                            dtype: DataflowDataType, chunking_mode: str = "broadcast",
                            **kwargs) -> 'DataflowInterface':
        """
        Factory method to create DataflowInterface from tensor chunking specification
        
        Args:
            name: Interface name
            interface_type: Type of interface
            original_shape: Original tensor shape before chunking
            block_dims: Desired block dimensions per calculation
            dtype: Data type specification
            chunking_mode: How to compute tensor_dims ("broadcast", "divide", "explicit")
            **kwargs: Additional interface parameters
            
        Returns:
            DataflowInterface with computed tensor_dims
        """
        # In the new architecture, tensor_dims should be the original tensor shape
        tensor_dims = list(original_shape)
        
        # Initialize stream_dims to default 1 for each block_dims dimension (can be updated by parallelism)
        stream_dims = [1] * len(block_dims)
        
        return cls(
            name=name,
            interface_type=interface_type,
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            stream_dims=stream_dims,
            dtype=dtype,
            **kwargs
        )
    
    @staticmethod
    def _compute_tensor_dims_from_chunking(original_shape: List[int], block_dims: List[int],
                                   chunking_mode: str = "broadcast") -> List[int]:
        """
        Compute tensor_dims from original tensor shape and desired block_dims
        
        Args:
            original_shape: Original tensor shape
            block_dims: Desired block dimensions
            chunking_mode: Computation method
            
        Returns:
            List[int]: Computed tensor_dims
            
        Examples:
            original_shape=[30, 50], block_dims=[50] → tensor_dims=[30] (chunking along last dim)
            original_shape=[10, 30, 50], block_dims=[3, 5] → tensor_dims=[1000] (where total=15000, block_dims=15)
        """
        if chunking_mode == "broadcast":
            total_elements = np.prod(original_shape)
            block_dims_elements = np.prod(block_dims)
            
            if len(block_dims) == 1:
                # Single block_dims: compute tensor_dims such that tensor_dims * block_dims = total_elements
                # For original_shape=[30, 50] with block_dims=[50], tensor_dims should be [30]
                if len(original_shape) == 2 and block_dims[0] == original_shape[-1]:
                    return [original_shape[0]]  # Return first dimension
                elif len(original_shape) == 1:
                    tensor_dims_val = total_elements // block_dims[0]
                    return [tensor_dims_val]
                else:
                    # Find matching dimension and compute tensor_dims from remaining
                    for i, dim in enumerate(original_shape):
                        if dim == block_dims[0]:
                            # Remove this dimension and compute product of others
                            remaining_dims = original_shape[:i] + original_shape[i+1:]
                            if remaining_dims:
                                return [np.prod(remaining_dims)]
                            else:
                                return [1]
                    # If no exact match, compute total division
                    tensor_dims_val = max(1, total_elements // block_dims[0])
                    return [tensor_dims_val]
            else:
                # Multiple block_dims: compute tensor_dims such that tensor_dims × block_dims = original_shape elements
                tensor_dims_elements = max(1, total_elements // block_dims_elements)
                return [tensor_dims_elements]
        
        elif chunking_mode == "divide":
            # Direct division of original_shape by block_dims
            if len(original_shape) != len(block_dims):
                raise ValueError(f"For divide mode, original_shape and block_dims must have same length")
            return [max(1, orig // block_dim) for orig, block_dim in zip(original_shape, block_dims)]
        
        else:
            raise ValueError(f"Unknown chunking_mode: {chunking_mode}")
    
    def __str__(self) -> str:
        """String representation of interface"""
        return (f"DataflowInterface(name='{self.name}', "
                f"type={self.interface_type.value}, "
                f"tensor_dims={self.tensor_dims}, block_dims={self.block_dims}, stream_dims={self.stream_dims}, "
                f"dtype={self.dtype.finn_type})")
