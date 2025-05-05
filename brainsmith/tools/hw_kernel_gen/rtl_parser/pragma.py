############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Pragma processing for Hardware Kernel Generator.

Handles the extraction, parsing, and validation of @brainsmith pragmas
found within SystemVerilog comments (e.g., // @brainsmith top my_module).
"""

import logging
from typing import List, Optional, Dict, Callable

from tree_sitter import Node

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Pragma, PragmaType

# Set up logger for this module
logger = logging.getLogger(__name__)

class PragmaError(Exception):
    """Custom exception for errors during pragma parsing or validation."""
    pass

class PragmaParser:
    """Extracts and validates @brainsmith pragmas from comment nodes."""

    def __init__(self, debug: bool = False):
        """Initializes the PragmaParser and registers pragma handlers."""
        self.debug = debug
        # Use PragmaType enum members as keys
        self.handlers: Dict[PragmaType, Callable[[List[str], int], Dict]] = {
            PragmaType.TOP_MODULE: self._handle_top_module,
            PragmaType.DATATYPE: self._handle_datatype,
            PragmaType.DERIVED_PARAMETER: self._handle_derived_parameter,
        }

    def parse_comment(self, node: Node, line_number: int) -> Optional[Pragma]:
        """Parses a comment AST node to find and validate a @brainsmith pragma.

        Checks for the '@brainsmith' prefix, extracts the type and inputs,
        validates the type, and calls the appropriate handler function.

        Args:
            node: The tree-sitter comment node.
            line_number: The 1-based line number where the comment starts.

        Returns:
            A validated Pragma object if a valid pragma is found, otherwise None.
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

        # --- Find matching PragmaType enum member ---
        pragma_enum_type: Optional[PragmaType] = None
        pragma_type_lower = pragma_type_str.lower()
        for member in PragmaType:
            if member.value == pragma_type_lower:
                pragma_enum_type = member
                break
        
        # Validate type against the *enum members* in the handlers dict keys
        if pragma_enum_type is None or pragma_enum_type not in self.handlers:
            # Log as debug, as many comments might not be intended pragmas
            logger.debug(f"Ignoring comment at line {line_number}: Unknown or unsupported pragma type '@brainsmith {pragma_type_str}'")
            return None
        
        # --- PragmaType is now validated and exists in handlers ---

        # Create pragma instance using the enum member
        pragma = Pragma(
            type=pragma_enum_type,
            inputs=inputs,
            line_number=line_number
        )

        # Process with appropriate handler using the enum member as the key
        try:
            pragma.processed_data = self.handlers[pragma_enum_type](inputs, line_number)
        except PragmaError as e:
            logger.warning(f"Pragma Error: {e} (Line: {line_number})")
            return None # Ignore pragmas that fail validation in their handler
        except Exception as e:
            logger.error(f"Unexpected error processing pragma at line {line_number}: {e}")
            return None # Ignore unexpected errors

        return pragma

    # --- Pragma Handlers ---

    def _handle_top_module(self, inputs: List[str], line_number: int) -> Dict:
        """Handles TOP_MODULE pragma: @brainsmith top_module <module_name>"""
        logger.debug(f"Processing TOP_MODULE pragma: {inputs} at line {line_number}")
        if len(inputs) != 1:
            raise PragmaError("TOP_MODULE pragma requires exactly one argument: <module_name>")
        return {"module_name": inputs[0]}

    def _handle_datatype(self, inputs: List[str], line_number: int) -> Dict:
        """Handles DATATYPE pragma: @brainsmith datatype <if_name> <size> OR <if_name> <min> <max>"""
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
        """Handles DERIVED_PARAMETER pragma: @brainsmith derived_parameter <func> <param1> [<param2>...]"""
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
    """Extracts all valid @brainsmith pragmas from an AST by walking comment nodes.

    Uses PragmaParser to parse and validate comments found during the AST traversal.

    Args:
        root_node: The root node of the tree-sitter AST.

    Returns:
        A list of validated Pragma objects found in the AST.
    """
    pragmas = []
    parser = PragmaParser()
    comments_found_count = 0 # Add counter

    # Simple recursive walk for comments - might need optimization for large files
    def find_comments(node: Node):
        nonlocal comments_found_count # Allow modification
        if node.type == 'comment':
            comments_found_count += 1 # Increment counter
            # Log comment finding at DEBUG level
            logger.debug(f"Found 'comment' node at line {node.start_point[0]+1}: {node.text.decode('utf8')[:60]}...")
            # Get line number (0-based)
            line_number = node.start_point[0]
            pragma = parser.parse_comment(node, line_number + 1) # Pass 1-based line number
            if pragma:
                # Log valid pragma finding at INFO level
                logger.info(f"Found valid pragma: {pragma}")
                pragmas.append(pragma)
            # No need for else log here, parse_comment logs failures/ignores

        for child in node.children:
            find_comments(child)

    # Log start/end at INFO level
    logger.info(">>> Starting pragma extraction from AST root.")
    find_comments(root_node)
    logger.info(f"<<< Finished pragma extraction. Found {comments_found_count} comment nodes and {len(pragmas)} valid pragmas.")
    return pragmas