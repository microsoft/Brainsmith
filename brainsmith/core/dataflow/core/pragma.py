############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Constraint system for kernel modeling"""

import ast
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
from .interface import Interface
from .types import prod


class Pragma(ABC):
    """Base class for all pragmas"""
    
    @abstractmethod
    def evaluate(self, interfaces: Dict[str, Interface], 
                env: Dict[str, int]) -> bool:
        """Evaluate pragma given interfaces and environment"""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert to human-readable string"""
        pass
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class TiePragma(Pragma):
    """Equality constraint between interface expressions
    
    Examples:
        TIE mat[1] vec       # Matrix columns match vector size
        TIE input output     # Interfaces have same total size
    """
    left_expr: str
    right_expr: str
    
    def evaluate(self, interfaces: Dict[str, Interface], 
                env: Dict[str, int]) -> bool:
        """Check if left and right expressions evaluate to same value"""
        left_val = evaluate_expr(self.left_expr, interfaces, env)
        right_val = evaluate_expr(self.right_expr, interfaces, env)
        return left_val == right_val
    
    def to_string(self) -> str:
        return f"TIE {self.left_expr} {self.right_expr}"


@dataclass
class ConstrPragma(Pragma):
    """Unary constraint on interface expression
    
    Examples:
        CONSTR vec % BURST    # Vector aligned to burst size
        CONSTR mat[0] >= 16   # Minimum parallelism
    """
    expr: str
    op: str  # "=", "<=", ">=", "%", "!=", ">"
    value: Union[int, str]  # int constant or symbol name
    
    def evaluate(self, interfaces: Dict[str, Interface],
                env: Dict[str, int]) -> bool:
        """Check if expression satisfies constraint"""
        expr_val = evaluate_expr(self.expr, interfaces, env)
        
        # Resolve value (could be symbol or literal)
        if isinstance(self.value, str):
            constraint_val = env.get(self.value)
            if constraint_val is None:
                raise ValueError(f"Unknown symbol in pragma: {self.value}")
        else:
            constraint_val = self.value
        
        # Apply operator
        if self.op == "=":
            return expr_val == constraint_val
        elif self.op == "<=":
            return expr_val <= constraint_val
        elif self.op == ">=":
            return expr_val >= constraint_val
        elif self.op == "%":
            return expr_val % constraint_val == 0
        elif self.op == "!=":
            return expr_val != constraint_val
        elif self.op == ">":
            return expr_val > constraint_val
        elif self.op == "<":
            return expr_val < constraint_val
        else:
            raise ValueError(f"Unknown operator: {self.op}")
    
    def to_string(self) -> str:
        return f"CONSTR {self.expr} {self.op} {self.value}"


# Expression evaluation
class InterfaceExprVisitor(ast.NodeVisitor):
    """AST visitor for evaluating interface expressions"""
    
    def __init__(self, interfaces: Dict[str, Interface], env: Dict[str, int]):
        self.interfaces = interfaces
        self.env = env
        self.result = None
    
    def visit_BinOp(self, node):
        """Handle binary operations"""
        left = self.evaluate_node(node.left)
        right = self.evaluate_node(node.right)
        
        op_map = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow
        }
        
        op_func = op_map.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        
        return int(op_func(left, right))
    
    def visit_UnaryOp(self, node):
        """Handle unary operations"""
        operand = self.evaluate_node(node.operand)
        
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    
    def visit_Name(self, node):
        """Handle variable names (interface names or env symbols)"""
        name = node.id
        
        # Check if it's an interface
        if name in self.interfaces:
            # Return total interface size
            intf = self.interfaces[name]
            return prod(intf.tensor_dims)
        
        # Check environment
        if name in self.env:
            return self.env[name]
        
        raise ValueError(f"Unknown identifier: {name}")
    
    def visit_Subscript(self, node):
        """Handle interface[dim] subscript access"""
        if not isinstance(node.value, ast.Name):
            raise ValueError("Can only subscript interface names")
        
        intf_name = node.value.id
        if intf_name not in self.interfaces:
            raise ValueError(f"Unknown interface: {intf_name}")
        
        intf = self.interfaces[intf_name]
        
        # Get index
        if isinstance(node.slice, ast.Constant):
            idx = node.slice.value
        elif isinstance(node.slice, ast.Index):  # Python < 3.9 compatibility
            idx = node.slice.value.value
        else:
            raise ValueError("Only constant indices supported")
        
        # Return dimension size
        if not isinstance(idx, int):
            raise ValueError(f"Index must be integer, got {type(idx)}")
        
        if idx < 0 or idx >= len(intf.tensor_dims):
            raise ValueError(f"Index {idx} out of range for interface {intf_name}")
        
        return intf.tensor_dims[idx]
    
    def visit_Constant(self, node):
        """Handle literal constants"""
        if isinstance(node.value, (int, float)):
            return int(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    
    def visit_Num(self, node):  # Python < 3.8 compatibility
        """Handle numeric literals"""
        return int(node.n)
    
    def evaluate_node(self, node):
        """Evaluate a single AST node"""
        visitor = InterfaceExprVisitor(self.interfaces, self.env)
        return visitor.visit(node)
    
    def visit(self, node):
        """Visit and return result"""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)


def evaluate_expr(expr: str, interfaces: Dict[str, Interface], 
                 env: Dict[str, int]) -> int:
    """Evaluate expression with interface references
    
    Args:
        expr: Expression string (e.g., "mat[0] * vec")
        interfaces: Dictionary of interface name -> Interface
        env: Environment dictionary with symbol values
    
    Returns:
        Integer result of expression
    
    Examples:
        "mat[0]" -> first dimension of mat interface
        "vec" -> total size of vec interface
        "mat[0] * SIMD" -> first dim of mat times SIMD value
    """
    try:
        # Parse expression
        tree = ast.parse(expr, mode='eval')
        
        # Evaluate
        visitor = InterfaceExprVisitor(interfaces, env)
        result = visitor.visit(tree.body)
        
        return result
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {e}")


def parse_pragma(pragma_str: str) -> Pragma:
    """Parse pragma from string representation
    
    Examples:
        "TIE mat[1] vec" -> TiePragma("mat[1]", "vec")
        "CONSTR vec % BURST" -> ConstrPragma("vec", "%", "BURST")
    """
    parts = pragma_str.strip().split()
    
    if not parts:
        raise ValueError("Empty pragma string")
    
    pragma_type = parts[0].upper()
    
    if pragma_type == "TIE":
        if len(parts) != 3:
            raise ValueError(f"TIE pragma needs exactly 2 expressions, got {len(parts)-1}")
        return TiePragma(parts[1], parts[2])
    
    elif pragma_type == "CONSTR":
        if len(parts) < 4:
            raise ValueError(f"CONSTR pragma needs expression, operator, and value")
        
        # Find operator position
        operators = ["<=", ">=", "!=", "=", ">", "<", "%"]
        op_idx = None
        op = None
        
        for i in range(1, len(parts)-1):
            if parts[i] in operators:
                op_idx = i
                op = parts[i]
                break
        
        if op_idx is None:
            raise ValueError(f"No valid operator found in CONSTR pragma")
        
        # Join expression parts
        expr = " ".join(parts[1:op_idx])
        value_str = " ".join(parts[op_idx+1:])
        
        # Try to parse value as int
        try:
            value = int(value_str)
        except ValueError:
            value = value_str  # Keep as symbol
        
        return ConstrPragma(expr, op, value)
    
    else:
        raise ValueError(f"Unknown pragma type: {pragma_type}")