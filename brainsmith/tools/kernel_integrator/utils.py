############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Utility functions for Kernel Integrator.

This module contains common utility functions used throughout the Kernel
Integrator to reduce code duplication and provide consistent behavior.
"""

import re
from typing import Dict, List, Optional, Any, Tuple


def pascal_case(name: str) -> str:
    """
    Convert snake_case or kebab-case to PascalCase.
    
    Args:
        name: String to convert (e.g., "my_module_name" or "my-module-name")
        
    Returns:
        PascalCase string (e.g., "MyModuleName")
        
    Examples:
        >>> pascal_case("thresholding_axi")
        "ThresholdingAxi"
        >>> pascal_case("matrix-multiply")
        "MatrixMultiply"
        >>> pascal_case("my_custom_op")
        "MyCustomOp"
    """
    # Replace hyphens with underscores
    name = name.replace('-', '_')
    
    # Split on underscores and capitalize each part
    parts = name.split('_')
    return ''.join(word.capitalize() for word in parts if word)


def snake_case(name: str) -> str:
    """
    Convert PascalCase or kebab-case to snake_case.
    
    Args:
        name: String to convert (e.g., "MyModuleName" or "my-module-name")
        
    Returns:
        snake_case string (e.g., "my_module_name")
        
    Examples:
        >>> snake_case("ThresholdingAxi")
        "thresholding_axi"
        >>> snake_case("MatrixMultiply")
        "matrix_multiply"
    """
    # Replace hyphens with underscores
    name = name.replace('-', '_')
    
    # Insert underscores before capitals and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def is_valid_identifier(name: str) -> bool:
    """
    Check if a string is a valid Python/SystemVerilog identifier.
    
    Args:
        name: String to check
        
    Returns:
        True if valid identifier, False otherwise
        
    Examples:
        >>> is_valid_identifier("my_var")
        True
        >>> is_valid_identifier("123abc")
        False
        >>> is_valid_identifier("my-var")
        False
    """
    return name.isidentifier()


def extract_dimensions_from_list(dim_list: List[Any]) -> Tuple[List[int], List[str]]:
    """
    Extract numeric dimensions and parameter names from a mixed list.
    
    Args:
        dim_list: List containing integers, "1", and parameter names
        
    Returns:
        Tuple of (numeric_dims, param_names)
        
    Examples:
        >>> extract_dimensions_from_list([1, "TILE_SIZE", 16])
        ([1, 16], ["TILE_SIZE"])
        >>> extract_dimensions_from_list(["1", "PE", "SIMD"])
        ([1], ["PE", "SIMD"])
    """
    numeric_dims = []
    param_names = []
    
    for item in dim_list:
        if isinstance(item, int):
            numeric_dims.append(item)
        elif isinstance(item, str):
            if item == "1" or item.isdigit():
                numeric_dims.append(int(item))
            elif item.isidentifier():
                param_names.append(item)
    
    return numeric_dims, param_names


def validate_parameter_name(name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a parameter name and return detailed error if invalid.
    
    Args:
        name: Parameter name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        >>> validate_parameter_name("PE")
        (True, None)
        >>> validate_parameter_name("123abc")
        (False, "Parameter name cannot start with a digit")
        >>> validate_parameter_name("")
        (False, "Parameter name cannot be empty")
    """
    if not name:
        return False, "Parameter name cannot be empty"
    
    if name[0].isdigit():
        return False, "Parameter name cannot start with a digit"
    
    if not name.isidentifier():
        return False, f"Parameter name '{name}' contains invalid characters"
    
    # Check for reserved words (common ones)
    reserved = {"if", "else", "for", "while", "module", "input", "output", 
                "reg", "wire", "parameter", "localparam", "def", "class", 
                "import", "from", "return", "pass", "break", "continue"}
    
    if name.lower() in reserved:
        return False, f"Parameter name '{name}' is a reserved keyword"
    
    return True, None


def format_template_variable(param_name: str) -> str:
    """
    Format parameter name as template variable in FINN style.
    
    Args:
        param_name: Parameter name (e.g., "PE")
        
    Returns:
        Template variable format (e.g., "$PE$")
        
    Examples:
        >>> format_template_variable("PE")
        "$PE$"
        >>> format_template_variable("input_width")
        "$INPUT_WIDTH$"
    """
    return f"${param_name.upper()}$"


def parse_template_variable(template_var: str) -> Optional[str]:
    """
    Extract parameter name from template variable format.
    
    Args:
        template_var: Template variable (e.g., "$PE$")
        
    Returns:
        Parameter name or None if not valid format
        
    Examples:
        >>> parse_template_variable("$PE$")
        "PE"
        >>> parse_template_variable("$INPUT_WIDTH$")
        "INPUT_WIDTH"
        >>> parse_template_variable("not_a_template")
        None
    """
    if template_var.startswith("$") and template_var.endswith("$"):
        return template_var[1:-1]
    return None




def group_parameters_by_interface(
    parameters: List[str],
    interface_mappings: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Group parameters by their associated interfaces.
    
    Args:
        parameters: List of all parameter names
        interface_mappings: Dict mapping interface names to their parameters
        
    Returns:
        Dict with "interface_params" and "general_params" keys
        
    Examples:
        >>> group_parameters_by_interface(
        ...     ["PE", "SIMD", "INPUT_WIDTH", "OUTPUT_WIDTH"],
        ...     {"input": ["INPUT_WIDTH"], "output": ["OUTPUT_WIDTH"]}
        ... )
        {
            "interface_params": {"input": ["INPUT_WIDTH"], "output": ["OUTPUT_WIDTH"]},
            "general_params": ["PE", "SIMD"]
        }
    """
    # Collect all interface-associated parameters
    interface_params = {}
    all_interface_params = set()
    
    for interface, params in interface_mappings.items():
        interface_params[interface] = params
        all_interface_params.update(params)
    
    # Remaining parameters are general
    general_params = [p for p in parameters if p not in all_interface_params]
    
    return {
        "interface_params": interface_params,
        "general_params": general_params
    }


def validate_shape_expression(
    shape_expr: List[Any],
    available_params: set,
    allow_full_slice: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate a shape expression (BDIM/SDIM SHAPE parameter).
    
    Args:
        shape_expr: List of shape elements
        available_params: Set of available parameter names
        allow_full_slice: Whether to allow ":" for full slice
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        >>> validate_shape_expression(
        ...     [1, "PE", ":"],
        ...     {"PE", "SIMD"},
        ...     True
        ... )
        (True, None)
        >>> validate_shape_expression(
        ...     [1, "UNKNOWN", ":"],
        ...     {"PE", "SIMD"},
        ...     True
        ... )
        (False, "Unknown parameter 'UNKNOWN' in shape expression")
    """
    if not isinstance(shape_expr, list):
        return False, "Shape expression must be a list"
    
    if not shape_expr:
        return False, "Shape expression cannot be empty"
    
    for i, element in enumerate(shape_expr):
        if element == 1 or element == "1":
            # Singleton dimension - always valid
            continue
        elif element == ":" and allow_full_slice:
            # Full slice dimension
            continue
        elif isinstance(element, str) and element.isidentifier():
            # Parameter name - check if it exists
            if element not in available_params:
                return False, f"Unknown parameter '{element}' in shape expression"
        else:
            return False, f"Invalid shape element '{element}' at position {i}"
    
    return True, None




def create_parameter_assignment(
    param_name: str,
    assignment_expr: str,
    comment: str
) -> Dict[str, str]:
    """
    Create a standardized parameter assignment dictionary.
    
    Args:
        param_name: Parameter name
        assignment_expr: Python expression to compute value
        comment: Human-readable comment
        
    Returns:
        Dictionary with param, template_var, assignment, and comment
        
    Examples:
        >>> create_parameter_assignment("PE", 'str(self.get_nodeattr("PE"))', "Processing elements")
        {'param': 'PE', 'template_var': '$PE$', 'assignment': 'str(self.get_nodeattr("PE"))', 'comment': 'Processing elements'}
    """
    return {
        "param": param_name,
        "template_var": format_template_variable(param_name),
        "assignment": assignment_expr,
        "comment": comment
    }


# Module exports
__all__ = [
    "pascal_case",
    "snake_case",
    "is_valid_identifier",
    "extract_dimensions_from_list",
    "validate_parameter_name",
    "format_template_variable",
    "parse_template_variable",
    "group_parameters_by_interface",
    "validate_shape_expression",
    "create_parameter_assignment",
]