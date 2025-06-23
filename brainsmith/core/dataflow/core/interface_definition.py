############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Interface definition for constraint specification and validation"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union, Dict, Any, Callable
from .base import BaseDefinition, ValidationContext, ParameterBinding
from .types import Shape, DataType, InterfaceDirection


@dataclass
class InterfaceDefinition(BaseDefinition):
    """Definition of an interface with constraints and validation rules
    
    Specifies the constraints, bounds, and relationships for an interface type.
    Does not contain actual runtime dimensions - those are in InterfaceModel.
    """
    
    # Core specification
    name: str
    direction: InterfaceDirection
    dtype: DataType
    
    # Constraint specifications
    alignment: Optional[int] = None          # Memory alignment requirement in bytes
    min_dims: Optional[Shape] = None         # Minimum dimension sizes
    max_dims: Optional[Shape] = None         # Maximum dimension sizes
    granularity: Optional[Shape] = None      # Dimension granularity (must be multiples)
    
    # Dataflow relationships
    produces: Set[str] = field(default_factory=set)      # Interfaces this feeds
    consumes: Set[str] = field(default_factory=set)      # Interfaces this reads from
    synchronized_with: Set[str] = field(default_factory=set)  # Must process together
    
    # Advanced constraint specifications
    sparsity_pattern: Optional[str] = None   # Expected sparsity pattern
    optional: bool = False                   # Interface is optional in some modes
    conditional_constraints: Dict[str, Any] = field(default_factory=dict)  # Context-dependent constraints
    
    # Block dimension specification
    block_dims_expr: Optional[Union[List[Union[str, int]], Callable]] = None  # Block dimension specification
    onnx_layout: Optional[str] = None  # ONNX layout hint (e.g., "NCHW", "NHWC", "NLC")
    
    def validate(self) -> List[str]:
        """Validate the interface definition for internal consistency
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate basic properties
        if not isinstance(self.name, str) or not self.name:
            errors.append("Interface name must be non-empty string")
        
        if not isinstance(self.direction, InterfaceDirection):
            errors.append(f"Direction must be InterfaceDirection, got {type(self.direction)}")
        
        if not isinstance(self.dtype, DataType):
            errors.append(f"Dtype must be DataType, got {type(self.dtype)}")
        
        # Validate constraints are self-consistent
        if self.alignment is not None and self.alignment <= 0:
            errors.append(f"Alignment must be positive, got {self.alignment}")
        
        if self.min_dims is not None and self.max_dims is not None:
            if len(self.min_dims) != len(self.max_dims):
                errors.append("min_dims and max_dims must have same length")
            else:
                for i, (min_val, max_val) in enumerate(zip(self.min_dims, self.max_dims)):
                    if min_val > max_val:
                        errors.append(f"min_dims[{i}]={min_val} > max_dims[{i}]={max_val}")
        
        if self.granularity is not None:
            for i, gran in enumerate(self.granularity):
                if gran is not None and gran <= 0:
                    errors.append(f"granularity[{i}] must be positive, got {gran}")
        
        # Validate dataflow relationships don't reference self
        if self.name in self.produces:
            errors.append("Interface cannot produce to itself")
        if self.name in self.consumes:
            errors.append("Interface cannot consume from itself")
        if self.name in self.synchronized_with:
            errors.append("Interface cannot be synchronized with itself")
        
        return errors
    
    def derive_block_dims(self, tensor_dims: Shape, 
                         param_binding: Dict[str, int],
                         config: Optional[Dict[str, Any]] = None) -> Shape:
        """Derive block dimensions from tensor dimensions and parameters
        
        Args:
            tensor_dims: Full tensor dimensions
            param_binding: Parameter values for substitution
            config: Optional configuration for adaptive strategies
            
        Returns:
            Block dimensions derived from specification
        """
        if callable(self.block_dims_expr):
            # Function-based specification
            result = self.block_dims_expr(tensor_dims, param_binding, config or {})
            # Process result to handle ":" and ensure tuple
            processed = []
            for i, dim in enumerate(result):
                if dim == ":":
                    processed.append(tensor_dims[i])
                else:
                    processed.append(dim)
            return tuple(processed)
        elif isinstance(self.block_dims_expr, list):
            # Expression list specification
            from .expressions import evaluate_expression
            result = []
            context = {
                'tensor': tensor_dims,
                'params': param_binding,
                'config': config or {},
                'interfaces': {},  # No interface context needed
                'parameters': param_binding,  # For backward compatibility
                'constants': {}
            }
            
            for i, expr in enumerate(self.block_dims_expr):
                if isinstance(expr, int):
                    # Literal integer
                    result.append(expr)
                elif expr == ":":
                    # Full dimension
                    if i < len(tensor_dims):
                        result.append(tensor_dims[i])
                    else:
                        raise ValueError(f"Expression ':' at index {i} exceeds tensor dimensions")
                elif isinstance(expr, str):
                    # Expression to evaluate
                    # Support tensor[i] syntax
                    expr_with_tensor = expr.replace('tensor[', 'tensor_')
                    for j in range(len(tensor_dims)):
                        expr_with_tensor = expr_with_tensor.replace(f'tensor_{j}]', str(tensor_dims[j]))
                    
                    # Support params['key'] syntax
                    for key, value in param_binding.items():
                        expr_with_tensor = expr_with_tensor.replace(f"params['{key}']", str(value))
                        expr_with_tensor = expr_with_tensor.replace(f'params["{key}"]', str(value))
                    
                    # Evaluate the expression
                    try:
                        value = evaluate_expression(expr_with_tensor, context)
                        result.append(int(value))
                    except Exception as e:
                        raise ValueError(f"Error evaluating block dimension expression '{expr}': {str(e)}")
                else:
                    raise ValueError(f"Invalid block dimension specification at index {i}: {expr}")
            
            return tuple(result)
        else:
            # Use default chunking strategy
            return self._default_block_chunking(tensor_dims)
    
    def _default_block_chunking(self, tensor_dims: Shape) -> Shape:
        """Default block chunking strategy
        
        Simple and predictable:
        - Default: full tensor (all dimensions as ":")
        - With ONNX layout hint: documented behavior per layout
        
        Args:
            tensor_dims: Full tensor dimensions
            
        Returns:
            Default block dimensions
        """
        if not self.onnx_layout:
            # No layout specified: use full tensor
            return tensor_dims
        
        # Layout-specific defaults (explicit, no guessing)
        if self.onnx_layout == "NCHW":
            # Batch and channels, full spatial dimensions
            if len(tensor_dims) == 4:
                return (tensor_dims[0], tensor_dims[1], tensor_dims[2], tensor_dims[3])
            else:
                return tensor_dims
        elif self.onnx_layout == "NHWC":
            # Batch and channels, full spatial dimensions
            if len(tensor_dims) == 4:
                return (tensor_dims[0], tensor_dims[1], tensor_dims[2], tensor_dims[3])
            else:
                return tensor_dims
        elif self.onnx_layout == "NLC":
            # Batch and channels, full sequence
            if len(tensor_dims) == 3:
                return (tensor_dims[0], tensor_dims[1], tensor_dims[2])
            else:
                return tensor_dims
        elif self.onnx_layout == "NCL":
            # Batch and channels, full sequence
            if len(tensor_dims) == 3:
                return (tensor_dims[0], tensor_dims[1], tensor_dims[2])
            else:
                return tensor_dims
        else:
            # Unknown layout: use full tensor
            return tensor_dims
    
    def validate_model_dimensions(self, tensor_dims: Shape) -> List[str]:
        """Validate that model dimensions satisfy this definition's constraints
        
        Args:
            tensor_dims: Actual tensor dimensions from model
            
        Returns:
            List of constraint violations (empty if valid)
        """
        errors = []
        
        # Check dimension bounds
        if self.min_dims is not None:
            if len(self.min_dims) != len(tensor_dims):
                errors.append(
                    f"min_dims length {len(self.min_dims)} != "
                    f"tensor dims length {len(tensor_dims)}"
                )
            else:
                for i, (actual, min_val) in enumerate(zip(tensor_dims, self.min_dims)):
                    if actual < min_val:
                        errors.append(
                            f"Interface {self.name} dim[{i}]={actual} < min={min_val}"
                        )
        
        if self.max_dims is not None:
            if len(self.max_dims) != len(tensor_dims):
                errors.append(
                    f"max_dims length {len(self.max_dims)} != "
                    f"tensor dims length {len(tensor_dims)}"
                )
            else:
                for i, (actual, max_val) in enumerate(zip(tensor_dims, self.max_dims)):
                    if actual > max_val:
                        errors.append(
                            f"Interface {self.name} dim[{i}]={actual} > max={max_val}"
                        )
        
        # Check granularity constraints
        if self.granularity is not None:
            if len(self.granularity) != len(tensor_dims):
                errors.append(
                    f"granularity length {len(self.granularity)} != "
                    f"tensor dims length {len(tensor_dims)}"
                )
            else:
                for i, (actual, gran) in enumerate(zip(tensor_dims, self.granularity)):
                    if gran is not None and gran > 0 and actual % gran != 0:
                        errors.append(
                            f"Interface {self.name} dim[{i}]={actual} "
                            f"not divisible by granularity {gran}"
                        )
        
        return errors
    
    def validate_model_alignment(self, tensor_dims: Shape) -> List[str]:
        """Validate alignment constraint for model dimensions
        
        Args:
            tensor_dims: Actual tensor dimensions from model
            
        Returns:
            List of alignment violations (empty if valid)
        """
        errors = []
        
        if self.alignment is not None:
            from .types import prod
            total_size = prod(tensor_dims) * self.dtype.bits // 8  # Size in bytes
            if total_size % self.alignment != 0:
                errors.append(
                    f"Interface {self.name} total size {total_size} bytes "
                    f"not aligned to {self.alignment} bytes"
                )
        
        return errors
    
    def create_model(self, tensor_dims: Shape, 
                    block_dims: Optional[Union[Shape, List[Shape]]] = None,
                    stream_dims: Optional[Shape] = None,
                    parameter_binding: Optional[Dict[str, int]] = None,
                    config: Optional[Dict[str, Any]] = None,
                    **kwargs) -> 'InterfaceModel':
        """Create an interface model from this definition
        
        Args:
            tensor_dims: Actual tensor dimensions
            block_dims: Actual block dimensions (optional, will derive if not provided)
            stream_dims: Actual stream dimensions (optional, will be calculated from iPar)
            parameter_binding: Parameter values for block dimension derivation
            config: Configuration for adaptive strategies
            **kwargs: Additional model parameters
            
        Returns:
            InterfaceModel instance
            
        Raises:
            ValueError: If dimensions violate definition constraints
        """
        # Validate dimensions against constraints
        errors = self.validate_model_dimensions(tensor_dims)
        errors.extend(self.validate_model_alignment(tensor_dims))
        
        if errors:
            raise ValueError(f"Model dimensions violate definition constraints:\n" + 
                           "\n".join(errors))
        
        # Derive block dimensions if not provided
        if block_dims is None:
            if parameter_binding is None:
                parameter_binding = {}
            block_dims = self.derive_block_dims(tensor_dims, parameter_binding, config)
        
        # Import here to avoid circular dependency
        from .interface_model import InterfaceModel
        
        return InterfaceModel(
            definition=self,
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            stream_dims=stream_dims,
            parameter_binding=parameter_binding,
            **kwargs
        )
    
    def add_produces(self, interface_name: str):
        """Add interface to produces set"""
        if interface_name != self.name:
            self.produces.add(interface_name)
    
    def add_consumes(self, interface_name: str):
        """Add interface to consumes set"""
        if interface_name != self.name:
            self.consumes.add(interface_name)
    
    def add_synchronized_with(self, interface_name: str):
        """Add interface to synchronized_with set"""
        if interface_name != self.name:
            self.synchronized_with.add(interface_name)
    
    def has_constraint(self, constraint_type: str) -> bool:
        """Check if definition has a specific type of constraint"""
        if constraint_type == "alignment":
            return self.alignment is not None
        elif constraint_type == "bounds":
            return self.min_dims is not None or self.max_dims is not None
        elif constraint_type == "granularity":
            return self.granularity is not None
        elif constraint_type == "dataflow":
            return len(self.produces) > 0 or len(self.consumes) > 0
        elif constraint_type == "synchronization":
            return len(self.synchronized_with) > 0
        else:
            return False
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of all constraints defined"""
        summary = {}
        
        if self.alignment is not None:
            summary["alignment"] = self.alignment
        if self.min_dims is not None:
            summary["min_dims"] = self.min_dims
        if self.max_dims is not None:
            summary["max_dims"] = self.max_dims
        if self.granularity is not None:
            summary["granularity"] = self.granularity
        if self.produces:
            summary["produces"] = list(self.produces)
        if self.consumes:
            summary["consumes"] = list(self.consumes)
        if self.synchronized_with:
            summary["synchronized_with"] = list(self.synchronized_with)
        
        return summary
    
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
        if self.produces:
            constraints.append(f"produces={list(self.produces)}")
        
        constraint_str = f", {', '.join(constraints)}" if constraints else ""
        
        return (
            f"InterfaceDefinition(name='{self.name}', "
            f"dir={self.direction.value}, "
            f"dtype={self.dtype}"
            f"{constraint_str})"
        )