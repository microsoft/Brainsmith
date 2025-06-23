############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Expression evaluation for kernel constraints and dependencies"""

import ast
import operator
import math
from typing import Dict, Any, Union, List
from .types import prod


class ExpressionEvaluator(ast.NodeVisitor):
    """Safe expression evaluator for kernel constraints
    
    Supports:
    - Interface dimension access: input[0], output[1]
    - Total interface size: input.total, output.total
    - Stream dimensions: input.stream[0]
    - Mathematical operations: +, -, *, /, //, %, **
    - Built-in functions: max, min, abs, ceil, floor
    - Constants and parameters
    """
    
    def __init__(self, context: Dict[str, Any]):
        """Initialize evaluator with context
        
        Args:
            context: Dictionary containing:
                - interfaces: Dict[str, Interface]
                - parameters: Dict[str, Union[int, float]]
                - constants: Dict[str, Union[int, float]]
                - tensor: Optional tensor dimensions for tensor[i] syntax
                - params: Optional parameter dict for params['key'] syntax
                - config: Optional config dict for config['key'] syntax
        """
        self.interfaces = context.get('interfaces', {})
        self.parameters = context.get('parameters', {})
        self.constants = context.get('constants', {})
        self.tensor = context.get('tensor', None)
        self.params = context.get('params', {})
        self.config = context.get('config', {})
        
        # Built-in functions
        self.functions = {
            'max': max,
            'min': min,
            'abs': abs,
            'ceil': math.ceil,
            'floor': math.floor,
            'sqrt': math.sqrt,
            'log': math.log,
            'log2': math.log2,
            'pow': pow
        }
        
        # Operator mapping
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos
        }
    
    def evaluate(self, expression: str) -> Union[int, float]:
        """Evaluate expression and return result
        
        Args:
            expression: String expression to evaluate
            
        Returns:
            Evaluated result
            
        Raises:
            ValueError: If expression is invalid or references unknown symbols
        """
        try:
            tree = ast.parse(expression, mode='eval')
            result = self.visit(tree.body)
            
            # Ensure numeric result
            if not isinstance(result, (int, float)):
                raise ValueError(f"Expression must evaluate to a number, got {type(result)}")
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")
    
    def visit_BinOp(self, node):
        """Handle binary operations"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        op_func = self.operators.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        
        # Handle division by zero
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("Division by zero")
        
        return op_func(left, right)
    
    def visit_UnaryOp(self, node):
        """Handle unary operations"""
        operand = self.visit(node.operand)
        
        op_func = self.operators.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        
        return op_func(operand)
    
    def visit_Name(self, node):
        """Handle variable names (parameters, constants, interface names)"""
        name = node.id
        
        # Check special names first
        if name == "tensor" and self.tensor is not None:
            return self.tensor
        elif name == "params":
            return self.params
        elif name == "config":
            return self.config
        
        # Check parameters
        if name in self.parameters:
            return self.parameters[name]
        
        # Check constants
        if name in self.constants:
            return self.constants[name]
        
        # Check if it's an interface name (total size)
        if name in self.interfaces:
            intf = self.interfaces[name]
            return prod(intf.tensor_dims)
        
        raise ValueError(f"Unknown identifier: {name}")
    
    def visit_Attribute(self, node):
        """Handle attribute access (e.g., input.total, input.stream)"""
        if not isinstance(node.value, ast.Name):
            raise ValueError("Only simple attribute access supported")
        
        obj_name = node.value.id
        attr_name = node.attr
        
        if obj_name not in self.interfaces:
            raise ValueError(f"Unknown interface: {obj_name}")
        
        intf = self.interfaces[obj_name]
        
        if attr_name == "total":
            return prod(intf.tensor_dims)
        elif attr_name == "stream":
            # Return stream dimensions as object for subscript access
            return intf.stream_dims
        elif attr_name == "block":
            # Return block dimensions (first phase for CSDF)
            return intf.block_dims[0] if intf.block_dims else intf.tensor_dims
        elif attr_name == "bandwidth":
            return intf.bandwidth_bits
        elif attr_name == "ipar":
            return intf.ipar
        else:
            raise ValueError(f"Unknown attribute: {attr_name}")
    
    def visit_Subscript(self, node):
        """Handle subscript access (e.g., input[0], input.stream[1], params['key'])"""
        # Get the object being subscripted
        obj = self.visit(node.value)
        
        # Get the index/key
        if isinstance(node.slice, ast.Constant):
            idx = node.slice.value
        elif isinstance(node.slice, ast.Index):  # Python < 3.9 compatibility
            idx = self.visit(node.slice.value)
        else:
            idx = self.visit(node.slice)
        
        # Handle dictionary access
        if isinstance(obj, dict):
            if isinstance(idx, (str, int)):
                if idx in obj:
                    return obj[idx]
                else:
                    raise ValueError(f"Key '{idx}' not found in dictionary")
            else:
                raise ValueError(f"Dictionary key must be string or int, got {type(idx)}")
        
        # Handle numeric indexing
        if not isinstance(idx, int):
            raise ValueError(f"Index must be integer, got {type(idx)}")
        
        # Handle different object types
        if isinstance(obj, tuple):  # Dimensions tuple
            if idx < 0 or idx >= len(obj):
                raise ValueError(f"Index {idx} out of range for dimensions {obj}")
            return obj[idx]
        elif isinstance(node.value, ast.Name):
            # Direct interface subscript (e.g., input[0])
            intf_name = node.value.id
            if intf_name not in self.interfaces:
                raise ValueError(f"Unknown interface: {intf_name}")
            
            intf = self.interfaces[intf_name]
            if idx < 0 or idx >= len(intf.tensor_dims):
                raise ValueError(f"Index {idx} out of range for interface {intf_name}")
            
            return intf.tensor_dims[idx]
        else:
            raise ValueError(f"Cannot subscript object of type {type(obj)}")
    
    def visit_Call(self, node):
        """Handle function calls"""
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls supported")
        
        func_name = node.func.id
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
        
        # Evaluate arguments
        args = [self.visit(arg) for arg in node.args]
        
        # Handle special cases
        if func_name in ['max', 'min'] and len(args) == 0:
            raise ValueError(f"{func_name} requires at least one argument")
        
        try:
            return self.functions[func_name](*args)
        except Exception as e:
            raise ValueError(f"Error calling {func_name}: {str(e)}")
    
    def visit_Constant(self, node):
        """Handle literal constants"""
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
    
    def visit_Num(self, node):  # Python < 3.8 compatibility
        """Handle numeric literals"""
        return node.n
    
    def generic_visit(self, node):
        """Handle unsupported node types"""
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def evaluate_expression(expression: str, context: Dict[str, Any]) -> Union[int, float]:
    """Convenience function for expression evaluation
    
    Args:
        expression: String expression to evaluate
        context: Context dictionary with interfaces, parameters, constants
        
    Returns:
        Evaluated result
    """
    evaluator = ExpressionEvaluator(context)
    return evaluator.evaluate(expression)


def validate_expression(expression: str, context: Dict[str, Any]) -> List[str]:
    """Validate expression without evaluating
    
    Args:
        expression: String expression to validate
        context: Context dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    try:
        # Try to parse the expression
        ast.parse(expression, mode='eval')
    except SyntaxError as e:
        errors.append(f"Syntax error: {str(e)}")
        return errors
    
    try:
        # Try to evaluate with dummy context
        evaluator = ExpressionEvaluator(context)
        evaluator.evaluate(expression)
    except ValueError as e:
        errors.append(str(e))
    
    return errors


def extract_dependencies(expression: str) -> List[str]:
    """Extract variable dependencies from expression
    
    Args:
        expression: String expression to analyze
        
    Returns:
        List of variable names referenced in expression
    """
    dependencies = []
    
    try:
        tree = ast.parse(expression, mode='eval')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                dependencies.append(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                dependencies.append(node.value.id)
    
    except SyntaxError:
        pass  # Return empty list for invalid expressions
    
    return list(set(dependencies))  # Remove duplicates