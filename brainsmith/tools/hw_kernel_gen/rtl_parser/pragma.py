############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Pragma processing for Hardware Kernel Generator.

This module handles the parsing and validation of @brainsmith pragmas found
in SystemVerilog comments. Pragmas provide additional information to guide
the Hardware Kernel Generator.

Example pragma:
    // @brainsmith top module_name
    // @brainsmith supported_dtype in0 INT 4 8
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
            logger.debug(f"Ignoring comment at line {line_number}: Unknown pragma type '{pragma_type_str}'")
            return None

        # Convert lowercase string back to uppercase enum member name
        try:
            enum_member_name = pragma_type_lower.upper()
            pragma_enum_type = getattr(PragmaType, enum_member_name)
        except AttributeError:
            # This case should ideally be caught by the handler check earlier
            logger.error(f"Internal error: Pragma type '{pragma_type_lower}' has a handler but no matching PragmaType enum member.")
            return None

        # Create pragma instance using the enum member
        pragma = Pragma(
            type=pragma_enum_type, # Assign the enum member
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
        """Handle DERIVED_PARAMETER pragma. Expected: <python_function_name> <module_param_name1> [<module_param_name2> ...]."""
        logger.debug(f"Processing DERIVED_PARAMETER pragma: {inputs} at line {line_number}")
        # Expect at least one function name and one parameter name
        if len(inputs) < 2:
            raise PragmaError("DERIVED_PARAMETER pragma requires at least <python_function_name> <module_param_name>")

        function_name = inputs[0]
        module_param_names = inputs[1:] # Get all elements after the first one

        logger.debug(f"Derived Parameter: Function='{function_name}', Params={module_param_names}")

        return {
            "python_function_name": function_name,
            "module_param_names": module_param_names # Return list of names
        }

# --- Extraction Function ---

def extract_pragmas(root_node: Node) -> List[Pragma]:
    """Extracts all Brainsmith pragmas from the AST."""
    pragmas = []
    parser = PragmaParser()
    comments_found_count = 0 # Add counter

    # Simple recursive walk for comments - might need optimization for large files
    def find_comments(node: Node):
        nonlocal comments_found_count # Allow modification
        if node.type == 'comment':
            comments_found_count += 1 # Increment counter
            # Use INFO level for visibility during testing
            logger.info(f"Found 'comment' node at line {node.start_point[0]+1}: {node.text.decode('utf8')[:60]}...")
            # Get line number (0-based)
            line_number = node.start_point[0]
            pragma = parser.parse_comment(node, line_number + 1) # Pass 1-based line number
            if pragma:
                logger.info(f"Found valid pragma: {pragma}")
                pragmas.append(pragma)
            else: # Add logging if parse_comment returns None
                 logger.info(f"Node at line {line_number+1} was a comment but not a valid pragma.")

        for child in node.children:
            find_comments(child)

    # Use INFO level for visibility during testing
    logger.info(">>> Starting pragma extraction from AST root.")
    find_comments(root_node)
    logger.info(f"<<< Finished pragma extraction. Found {comments_found_count} comment nodes and {len(pragmas)} valid pragmas.")
    return pragmas