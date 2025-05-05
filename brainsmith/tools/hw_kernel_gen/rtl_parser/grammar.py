"""Handles SystemVerilog grammar loading and node type constants for tree-sitter.

This module centralizes the tree-sitter grammar loading logic using ctypes
and defines constants for SystemVerilog node types used by the parser.

Grammar Source: Assumes a pre-compiled tree-sitter grammar library (e.g., sv.so)
based on a grammar like tree-sitter-verilog. The exact version compatibility
might depend on the tree-sitter library version used during compilation.

Why ctypes?: Tree-sitter's Python binding typically loads grammars via language
files (.so, .dll, .dylib) containing a specific C function (e.g., tree_sitter_verilog).
Using ctypes allows direct loading of this shared library and accessing the
language function pointer, which is then wrapped into a Python capsule that the
tree-sitter Python library understands. This avoids needing the grammar source
code at runtime, only the compiled library.
"""

import os
import ctypes
import logging
from ctypes import c_void_p, c_char_p, py_object, pythonapi
from typing import Optional
from tree_sitter import Language

logger = logging.getLogger(__name__)

# --- Node Type Constants ---
# Based on common tree-sitter-verilog grammar node names.
# Adjust these if your specific grammar version uses different names.

# Declarations
MODULE_DECLARATION = "module_declaration"
PARAMETER_PORT_DECLARATION = "parameter_port_declaration"
LOCAL_PARAMETER_DECLARATION = "local_parameter_declaration"
PARAMETER_DECLARATION = "parameter_declaration"
TYPE_PARAMETER_DECLARATION = "type_parameter_declaration"
PORT_DECLARATION = "port_declaration"
ANSI_PORT_DECLARATION = "ansi_port_declaration"
NON_ANSI_PORT_DECLARATION = "non_ansi_port_declaration" # May vary by grammar
LIST_OF_PORT_DECLARATIONS = "list_of_port_declarations" # May vary by grammar
LIST_OF_PORTS = "list_of_ports" # May vary by grammar
PORT = "port" # May vary by grammar

# Parameters
PARAMETER_PORT_LIST = "parameter_port_list"
LIST_OF_PARAM_ASSIGNMENTS = "list_of_param_assignments"
PARAM_ASSIGNMENT = "param_assignment"
LIST_OF_TYPE_ASSIGNMENTS = "list_of_type_assignments"
TYPE_ASSIGNMENT = "type_assignment"

# Ports
PORT_DIRECTION = "port_direction"
INPUT_DECLARATION = "input_declaration" # Non-ANSI style
OUTPUT_DECLARATION = "output_declaration" # Non-ANSI style
INOUT_DECLARATION = "inout_declaration" # Non-ANSI style
NET_DECLARATION = "net_declaration" # Often contains Non-ANSI port types/widths
DATA_DECLARATION = "data_declaration" # Often contains Non-ANSI port types/widths

# Types and Expressions
DATA_TYPE = "data_type"
DATA_TYPE_OR_IMPLICIT = "data_type_or_implicit"
PACKED_DIMENSION = "packed_dimension"
UNPACKED_DIMENSION = "unpacked_dimension" # Less common in ports/params, but possible
CONSTANT_EXPRESSION = "constant_expression"
CONSTANT_PARAM_EXPRESSION = "constant_param_expression"
CONSTANT_MINTYPMAX_EXPRESSION = "constant_mintypmax_expression"
EXPRESSION = "expression"
PRIMARY_LITERAL = "primary_literal"
BINARY_EXPRESSION = "binary_expression"
INTEGER_VECTOR_TYPE = "integer_vector_type" # e.g., logic, bit, reg
INTEGER_ATOM_TYPE = "integer_atom_type" # e.g., int, byte, integer
NON_INTEGER_TYPE = "non_integer_type" # e.g., real, shortreal
STRING_LITERAL = "string_literal"
NUMBER = "number"

# Identifiers
SIMPLE_IDENTIFIER = "simple_identifier"
IDENTIFIER = "identifier" # General identifier, often used as fallback

# Keywords (used for specific checks)
TYPE_KEYWORD = "type"
INPUT_KEYWORD = "input"
OUTPUT_KEYWORD = "output"
INOUT_KEYWORD = "inout"

# Comments / Pragmas
COMMENT = "comment"

# Errors
ERROR = "ERROR"


# --- Language Loading ---

def load_language(grammar_path: str) -> Language:
    """Loads the tree-sitter grammar from the specified path using ctypes.

    Args:
        grammar_path: Absolute path to the compiled grammar library (.so, .dll, .dylib).

    Returns:
        A tree-sitter Language object.

    Raises:
        FileNotFoundError: If the grammar file does not exist.
        AttributeError: If the expected language function (tree_sitter_verilog) is not found.
        RuntimeError: For other ctypes or tree-sitter initialization errors.
    """
    if not os.path.exists(grammar_path):
        raise FileNotFoundError(f"Grammar library not found at: {grammar_path}")

    try:
        # 1. Load the shared library
        lib = ctypes.cdll.LoadLibrary(grammar_path)
        logger.debug(f"Loaded grammar library: {grammar_path}")

        # 2. Get the language function pointer (adjust 'tree_sitter_verilog' if needed)
        language_function_name = "tree_sitter_verilog"
        if not hasattr(lib, language_function_name):
             raise AttributeError(f"Language function '{language_function_name}' not found in '{grammar_path}'. Check grammar compilation.")
        lang_ptr_func = getattr(lib, language_function_name)
        lang_ptr_func.restype = c_void_p
        lang_ptr = lang_ptr_func()
        logger.debug(f"Obtained language function pointer from '{language_function_name}'")

        # 3. Create a Python capsule for the language pointer
        #    The capsule name "tree_sitter.Language" is expected by the tree-sitter Python library.
        PyCapsule_New = pythonapi.PyCapsule_New
        PyCapsule_New.restype = py_object
        PyCapsule_New.argtypes = (c_void_p, c_char_p, c_void_p)
        capsule = PyCapsule_New(lang_ptr, b"tree_sitter.Language", None)
        logger.debug("Created Python capsule for language pointer")

        # 4. Create the tree-sitter Language object from the capsule
        language = Language(capsule)
        logger.info(f"Successfully created Language object from '{grammar_path}'")
        return language

    except FileNotFoundError: # Re-raise specific error
        raise
    except AttributeError as e:
        logger.error(f"Attribute error during grammar loading: {e}")
        raise # Re-raise specific error
    except Exception as e:
        logger.exception(f"Failed to load grammar from '{grammar_path}' using ctypes: {e}")
        raise RuntimeError(f"Failed to load grammar: {e}")

