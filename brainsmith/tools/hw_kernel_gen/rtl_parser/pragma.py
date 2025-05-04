"""Pragma processing for Hardware Kernel Generator.

This module handles the parsing and validation of @brainsmith pragmas found
in SystemVerilog comments. Pragmas provide additional information to guide
the Hardware Kernel Generator.

Example pragma:
    // @brainsmith interface AXI_STREAM
    // @brainsmith parameter STATIC WIDTH
"""

import logging
from enum import Enum
from typing import List, Optional, Dict, Callable
from tree_sitter import Node

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Pragma

# Set up logger for this module
logger = logging.getLogger(__name__)

class PragmaType(Enum):
    """Valid pragma types based on RTL_Parser-Data-Analysis.md."""
    TOP_MODULE = "top_module"          # Specify the top module if multiple exist
    DATATYPE = "datatype"              # Restrict datatype for an interface
    DERIVED_PARAMETER = "derived_parameter" # Link module param to python function

class PragmaError(Exception):
    """Base class for pragma-related errors."""
    pass

class PragmaParser:
    """Parser for @brainsmith pragmas.
    
    This class handles the extraction and validation of pragmas from
    SystemVerilog comments.
    
    Attributes:
        debug: Enable debug output
        handlers: Dict of pragma type to handler functions
    """
    
    def __init__(self, debug: bool = False):
        """Initialize pragma parser.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        # Update handlers to match the required PragmaTypes
        self.handlers: Dict[str, Callable[[List[str], int], Dict]] = {
            PragmaType.TOP_MODULE.value: self._handle_top_module,
            PragmaType.DATATYPE.value: self._handle_datatype,
            PragmaType.DERIVED_PARAMETER.value: self._handle_derived_parameter,
        }
    
    def parse_comment(self, node: Node, line_number: int) -> Optional[Pragma]:
        """Parse a comment node for @brainsmith pragmas.
        
        Args:
            node: Comment AST node
            line_number: Source line number
            
        Returns:
            Pragma instance if valid pragma found, None otherwise
        """
        text = node.text.decode('utf8').strip('/ ')
        
        # Check for pragma prefix
        if not text.startswith('@brainsmith'):
            return None
        
        # Split into components
        parts = text.split()
        if len(parts) < 2:
            logger.warning(f"Invalid pragma format at line {line_number}: {text}")
            return None
        
        # Extract type and inputs
        pragma_type_str = parts[1]
        inputs = parts[2:] if len(parts) > 2 else []

        # Normalize pragma type to lowercase for case-insensitive matching
        pragma_type_lower = pragma_type_str.lower()

        # Validate type against the *new* set of handlers (using lowercase)
        if pragma_type_lower not in self.handlers:
            # Check if it's one of the *old* types to provide a better warning
            old_types = {"interface", "parameter", "resource", "timing", "feature"}
            if pragma_type_lower in old_types: # Check lowercase against old types too
                 logger.warning(f"Obsolete pragma type '{pragma_type_str}' used at line {line_number}. This type is no longer supported.")
            else:
                 logger.warning(f"Unknown pragma type '{pragma_type_str}' at line {line_number}: {text}")
            return None # Ignore unknown or obsolete pragmas
        
        # Create pragma instance
        pragma = Pragma(
            type=pragma_type_lower, # Store the lowercase string value
            inputs=inputs,
            line_number=line_number
        )

        # Process with appropriate handler (using lowercase key)
        try:
            # Pass line number for better error context
            pragma.processed_data = self.handlers[pragma_type_lower](inputs, line_number)
        except PragmaError as e:
            logger.warning(f"Pragma Error: {e} (Line: {line_number})")
            return None # Ignore pragmas that fail validation in their handler
        except Exception as e:
            logger.error(f"Unexpected error processing pragma at line {line_number}: {e}")
            return None # Ignore unexpected errors

        return pragma
    
    # --- Pragma Handlers (Updated for required types) ---

    def _handle_top_module(self, inputs: List[str], line_number: int) -> Dict:
        """Handle TOP_MODULE pragma. Expected: <module_name>."""
        logger.debug(f"Processing TOP_MODULE pragma: {inputs} at line {line_number}")
        if len(inputs) != 1:
            raise PragmaError("TOP_MODULE pragma requires exactly one argument: <module_name>")
        return {"module_name": inputs[0]}

    def _handle_datatype(self, inputs: List[str], line_number: int) -> Dict:
        """Handle DATATYPE pragma.
        Expected: <interface_name> <size>
        OR
        Expected: <interface_name> <min_size> <max_size>

        If only <size> is provided, the interface supports *only* that specific size.
        If <min_size> and <max_size> are provided, the interface supports sizes within that inclusive range.
        """
        logger.debug(f"Processing DATATYPE pragma: {inputs} at line {line_number}")

        if len(inputs) == 2:
            # Case: Fixed size
            interface_name = inputs[0]
            size = inputs[1]
            # TODO: Validate size format (e.g., ensure it's numeric or a valid type string)
            processed = {
                "interface_name": interface_name,
                "min_size": size,
                "max_size": size, # Explicitly set max == min for fixed size
                "is_fixed_size": True
            }
        elif len(inputs) == 3:
            # Case: Size range
            interface_name = inputs[0]
            min_size = inputs[1]
            max_size = inputs[2]
            # TODO: Validate size formats
            # TODO: Validate min_size <= max_size (if numeric)
            processed = {
                "interface_name": interface_name,
                "min_size": min_size,
                "max_size": max_size,
                "is_fixed_size": False
            }
        else:
            raise PragmaError("DATATYPE pragma requires <interface_name> <size> OR <interface_name> <min_size> <max_size>")

        return processed

    def _handle_derived_parameter(self, inputs: List[str], line_number: int) -> Dict:
        """Handle DERIVED_PARAMETER pragma. Expected: <module_param_name> <python_function_name>."""
        logger.debug(f"Processing DERIVED_PARAMETER pragma: {inputs} at line {line_number}")
        if len(inputs) != 2:
            raise PragmaError("DERIVED_PARAMETER pragma requires <module_param_name> <python_function_name>")
        return {
            "module_param_name": inputs[0],
            "python_function_name": inputs[1]
        }

# --- Extraction Function ---

def extract_pragmas(root_node: Node) -> List[Pragma]:
    """Extracts all valid @brainsmith pragmas from comments in the AST."""
    pragmas = []
    parser = PragmaParser() # Can add debug=True here if needed

    # Simple BFS traversal to find all comment nodes
    queue = [root_node]
    visited = {root_node.id}
    comment_nodes = []

    while queue:
        node = queue.pop(0)
        if node.type == 'comment':
            comment_nodes.append(node)

        for child in node.children:
            if child.id not in visited:
                visited.add(child.id)
                queue.append(child)

    # Sort comments by line number
    comment_nodes.sort(key=lambda n: n.start_point[0])

    # Parse each comment
    for node in comment_nodes:
        line_number = node.start_point[0] + 1 # tree-sitter is 0-based, humans 1-based
        pragma = parser.parse_comment(node, line_number)
        if pragma:
            pragmas.append(pragma)
            logger.debug(f"Found valid pragma: {pragma}")

    return pragmas