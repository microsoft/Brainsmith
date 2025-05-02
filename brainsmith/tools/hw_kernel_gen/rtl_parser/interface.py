"""Interface analysis for SystemVerilog modules.

This module provides specialized parsing functions for extracting and analyzing
module interface components like parameters and ports.
"""

import logging
from typing import List, Optional, Tuple
from tree_sitter import Node

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    Parameter,
    Port,
    Direction
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parse_parameter_declaration(node: Node) -> Optional[Parameter]:
    """Parse a parameter declaration node.
    
    Args:
        node: parameter_declaration AST node
        
    Returns:
        Parameter instance if valid, None otherwise
        
    Example SystemVerilog:
        parameter int WIDTH = 32;
        parameter logic SIGNED = 1;
    """
    if node is None:
        return None

    logger.debug("Parsing parameter declaration:")
    _debug_node(node)

    # Skip local parameters
    if node.type == "localparam_declaration" or _has_text(node, "localparam"):
        return None
    
    # Get parameter type (may be optional)
    param_type = "logic"  # Default type
    type_node = _find_child(node, ["data_type", "type_identifier", "simple_type"])
    if type_node is not None:
        param_type = type_node.text.decode('utf8')
    
    # Get parameter name
    name_node = _find_child(node, ["simple_identifier", "identifier", "parameter_identifier"])
    if name_node is None:
        return None
    name = name_node.text.decode('utf8')
    
    # Get default value if present
    default_value = None
    equals = _find_child(node, "=")
    if equals is not None:
        # Get text after equals
        current = equals.next_sibling
        if current is not None:
            # For both simple numbers and expressions
            value_text = []
            while current and current.type != ";":
                value_text.append(current.text.decode('utf8'))
                current = current.next_sibling
            if value_text:
                default_value = "".join(value_text).strip()
    
    return Parameter(
        name=name,
        param_type=param_type,
        default_value=default_value
    )

def parse_port_declaration(node: Node) -> Optional[Port]:
    """Parse a port declaration node.
    
    Args:
        node: port_declaration AST node
        
    Returns:
        Port instance if valid, None otherwise
        
    Example SystemVerilog:
        input logic clk,
        output logic [WIDTH-1:0] data
    """
    if node is None:
        return None

    logger.debug("Parsing port declaration:")
    _debug_node(node)
    
    direction = None
    direction_types = ["input", "output", "inout"]
    
    # Check if the node is a data_declaration with direction keyword
    node_text = node.text.decode('utf8')
    for dir_type in direction_types:
        if node_text.startswith(dir_type):
            direction = Direction(dir_type)
            break
    
    if direction is None:
        # Try to find direction node
        dir_node = _find_child(node, ["port_direction"] + direction_types)
        if dir_node is not None:
            if dir_node.type == "port_direction":
                # For ANSI style, use text
                dir_text = dir_node.text.decode('utf8')
                direction = Direction(dir_text)
            else:
                # For non-ANSI style, use node type
                direction = Direction(dir_node.type)
        else:
            # Try text search
            for dir_type in direction_types:
                if _has_text(node, dir_type):
                    direction = Direction(dir_type)
                    break
    
    if direction is None:
        return None
    
    # Get port name - check all identifiers and take the last one
    name_node = None
    for child in node.children:
        if child.type in ["simple_identifier", "identifier", "port_identifier"]:
            name_node = child
    if name_node is None:
        return None
    name = name_node.text.decode('utf8')
    
    # Get width expression
    width = "1"  # Default to single bit
    range_nodes = []
    
    # Look for different types of range expressions
    for range_type in ["packed_dimension", "dimension", "range_expression"]:
        range_node = _find_child(node, range_type)
        if range_node is not None:
            range_nodes.append(range_node)
    
    if range_nodes:
        # Extract range expression text
        for range_node in range_nodes:
            width_parts = []
            for child in range_node.children:
                if child.type not in ["[", "]", ":"]:
                    width_parts.append(child.text.decode('utf8'))
            if width_parts:
                width = "".join(width_parts).strip()
                break
    
    return Port(
        name=name,
        direction=direction,
        width=width
    )

def extract_module_header(node: Node) -> Tuple[str, List[Node], List[Node]]:
    """Extract key components from module header.
    
    Args:
        node: module_declaration AST node
        
    Returns:
        Tuple of:
        - Module name
        - List of parameter declaration nodes
        - List of port declaration nodes
        
    Raises:
        ValueError: If module header is invalid
    """
    if node is None:
        raise ValueError("Invalid module node")

    logger.debug("Extracting module header:")
    _debug_node(node)

    # Get module name
    name_node = _find_child(node, ["module_identifier", "simple_identifier", "identifier"])
    if name_node is None:
        raise ValueError("Module name not found")
    name = name_node.text.decode('utf8')
    
    # Find parameter declarations
    param_nodes = []
    param_list = _find_child(node, ["parameter_port_list", "list_of_parameter_declarations"])
    if param_list is not None:
        for child in param_list.children:
            if child.type in ["parameter_declaration", "parameter_port_declaration"]:
                param_nodes.append(child)
    
    # Find port declarations
    port_nodes = []
    for type_name in ["port_list", "list_of_port_declarations", "ansi_port_list"]:
        port_list = _find_child(node, type_name)
        if port_list is not None:
            for child in port_list.children:
                if child.type in ["port_declaration", "ansi_port_declaration", "net_declaration", "data_declaration"]:
                    port_nodes.append(child)
    
    return name, param_nodes, port_nodes

def _find_child(node: Node, type_names: str | List[str], recursive: bool = True) -> Optional[Node]:
    """Find first child node matching any of the given types.
    
    Args:
        node: Parent node
        type_names: Single type name or list of type names to match 
        recursive: Whether to search recursively through children
        
    Returns:
        Matching node if found, None otherwise
    """
    if node is None:
        return None

    if isinstance(type_names, str):
        type_names = [type_names]
    
    for child in node.children:
        if child.type in type_names:
            return child
        # Recursively search child's children if requested
        if recursive:
            result = _find_child(child, type_names)
            if result is not None:
                return result
    return None

def _has_text(node: Node, text: str) -> bool:
    """Check if node or any child contains the exact text.
    
    Args:
        node: Node to check
        text: Text to find
        
    Returns:
        True if text found, False otherwise
    """
    if node is None:
        return False
    
    node_text = node.text.decode('utf8')
    if text in node_text.split():
        return True
    
    for child in node.children:
        if _has_text(child, text):
            return True
    
    return False

def _debug_node(node: Node, prefix: str = "") -> None:
    """Debug helper to print node structure."""
    if node is None:
        return
    logger.debug(f"{prefix}Node type: {node.type}, text: {node.text.decode('utf8')}")
    for child in node.children:
        _debug_node(child, prefix + "  ")