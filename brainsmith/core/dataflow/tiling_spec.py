############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tiling specification for shape expressions"""

from dataclasses import dataclass
from typing import List, Union, Optional, Set
from enum import Enum


class TilingExprType(Enum):
    """Type of tiling expression"""
    SINGLETON = "singleton"  # Fixed size of 1
    FULL = "full"           # Take full dimension (:)
    LITERAL = "literal"     # Fixed integer value
    PARAMETER = "parameter" # Named parameter to be resolved at runtime


@dataclass
class TilingExpr:
    """Single tiling expression"""
    expr_type: TilingExprType
    value: Optional[Union[int, str]] = None
    
    @classmethod
    def from_value(cls, value: Union[int, str]) -> 'TilingExpr':
        """Create TilingExpr from a value"""
        if value == 1:
            return cls(TilingExprType.SINGLETON, 1)
        elif value == ":":
            return cls(TilingExprType.FULL, None)
        elif isinstance(value, int):
            if value <= 0:
                raise ValueError(f"Tiling expression must be positive, got {value}")
            return cls(TilingExprType.LITERAL, value)
        elif isinstance(value, str):
            if not value or value.isspace():
                raise ValueError("Parameter name cannot be empty")
            return cls(TilingExprType.PARAMETER, value)
        else:
            raise TypeError(f"Invalid tiling expression type: {type(value)}")
    
    @property
    def is_static(self) -> bool:
        """Check if expression has a static value"""
        return self.expr_type in [TilingExprType.SINGLETON, TilingExprType.LITERAL]
    
    @property
    def is_parameter(self) -> bool:
        """Check if expression is a parameter"""
        return self.expr_type == TilingExprType.PARAMETER
    
    @property
    def parameter_name(self) -> Optional[str]:
        """Get parameter name if this is a parameter expression"""
        if self.expr_type == TilingExprType.PARAMETER:
            return self.value
        return None
    
    def __repr__(self) -> str:
        if self.expr_type == TilingExprType.SINGLETON:
            return "1"
        elif self.expr_type == TilingExprType.FULL:
            return ":"
        elif self.expr_type == TilingExprType.LITERAL:
            return str(self.value)
        elif self.expr_type == TilingExprType.PARAMETER:
            return f'"{self.value}"'
        else:
            return f"TilingExpr({self.expr_type}, {self.value})"


@dataclass
class TilingSpec:
    """Specification for tiling dimensions
    
    Encapsulates a list of tiling expressions that define how
    tensor dimensions should be tiled into blocks or streams.
    
    Examples:
        # Fixed tiling with parameters
        TilingSpec([1, "CH_TILES", ":", ":"])
        
        # All literals
        TilingSpec([32, 64, 14, 14])
        
        # Mixed expressions
        TilingSpec([1, "SIMD", 1, 1])
    """
    
    expressions: List[TilingExpr]
    
    def __init__(self, values: List[Union[int, str]]):
        """Initialize from a list of values
        
        Args:
            values: List of tiling expressions (1, ":", "<param>", or integer)
        """
        if not values:
            raise ValueError("TilingSpec cannot be empty")
        
        self.expressions = []
        for i, val in enumerate(values):
            try:
                expr = TilingExpr.from_value(val)
                self.expressions.append(expr)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid tiling expression at index {i}: {e}")
    
    @property
    def ndim(self) -> int:
        """Number of dimensions in the tiling spec"""
        return len(self.expressions)
    
    def get_parameters(self) -> Set[str]:
        """Get all parameter names used in expressions
        
        Returns:
            Set of parameter names
        """
        params = set()
        for expr in self.expressions:
            if expr.is_parameter:
                params.add(expr.parameter_name)
        return params
    
    def validate_against_shape(self, shape: List[int]) -> List[str]:
        """Validate tiling spec against a tensor shape
        
        Args:
            shape: Tensor shape to validate against
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Allow specs with fewer dimensions than tensor (will be left-padded)
        if len(self.expressions) > len(shape):
            errors.append(
                f"Tiling spec has {len(self.expressions)} dimensions "
                f"but tensor only has {len(shape)} dimensions"
            )
            return errors  # Can't validate further
        
        # Check each expression against corresponding dimension from the right
        # If tensor has more dims than spec, leftmost tensor dims are untiled (singleton)
        offset = len(shape) - len(self.expressions)
        for i, expr in enumerate(self.expressions):
            shape_idx = offset + i
            dim_size = shape[shape_idx]
            
            if expr.expr_type == TilingExprType.LITERAL:
                if dim_size % expr.value != 0:
                    errors.append(
                        f"Dimension {shape_idx}: tile size {expr.value} does not "
                        f"evenly divide tensor dimension {dim_size}"
                    )
        
        return errors
    
    def resolve(self, shape: List[int], parameters: dict) -> List[int]:
        """Resolve expressions to concrete tile sizes
        
        Supports adaptive behavior: if tensor has more dimensions than the
        tiling spec, the spec is left-padded with singletons (1) to match.
        This allows RTL to specify tiling only for dimensions it cares about.
        
        Args:
            shape: Tensor shape
            parameters: Parameter values
            
        Returns:
            List of resolved tile sizes (same length as shape)
            
        Raises:
            ValueError: If resolution fails
        """
        # Handle dimension mismatch with adaptive left-padding
        if len(self.expressions) > len(shape):
            raise ValueError(
                f"Cannot resolve: tiling spec has {len(self.expressions)} dims "
                f"but shape only has {len(shape)} dims"
            )
        
        # Left-pad with singletons if shape has more dimensions
        padding_needed = len(shape) - len(self.expressions)
        result = [1] * padding_needed  # Start with singleton padding
        
        # Resolve expressions for the rightmost dimensions
        offset = len(shape) - len(self.expressions)
        for i, expr in enumerate(self.expressions):
            shape_idx = offset + i
            dim_size = shape[shape_idx]
            
            if expr.expr_type == TilingExprType.SINGLETON:
                result.append(1)
            elif expr.expr_type == TilingExprType.FULL:
                result.append(dim_size)
            elif expr.expr_type == TilingExprType.LITERAL:
                result.append(expr.value)
            elif expr.expr_type == TilingExprType.PARAMETER:
                if expr.value not in parameters:
                    raise ValueError(
                        f"Parameter '{expr.value}' not found in parameter binding"
                    )
                param_value = parameters[expr.value]
                if not isinstance(param_value, int) or param_value <= 0:
                    raise ValueError(
                        f"Parameter '{expr.value}' must be a positive integer, "
                        f"got {param_value}"
                    )
                result.append(param_value)
        
        return result
    
    def __repr__(self) -> str:
        expr_strs = []
        for expr in self.expressions:
            if expr.expr_type == TilingExprType.SINGLETON:
                expr_strs.append("1")
            elif expr.expr_type == TilingExprType.FULL:
                expr_strs.append(":")
            elif expr.expr_type == TilingExprType.LITERAL:
                expr_strs.append(str(expr.value))
            elif expr.expr_type == TilingExprType.PARAMETER:
                expr_strs.append(f'"{expr.value}"')
        
        return f"TilingSpec([{', '.join(expr_strs)}])"
    
    def to_list(self) -> List[Union[int, str]]:
        """Convert back to list representation"""
        result = []
        for expr in self.expressions:
            if expr.expr_type == TilingExprType.SINGLETON:
                result.append(1)
            elif expr.expr_type == TilingExprType.FULL:
                result.append(":")
            elif expr.expr_type == TilingExprType.LITERAL:
                result.append(expr.value)
            elif expr.expr_type == TilingExprType.PARAMETER:
                result.append(expr.value)
        return result