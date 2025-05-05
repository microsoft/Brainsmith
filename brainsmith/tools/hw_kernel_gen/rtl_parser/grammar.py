############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Handles SystemVerilog grammar loading and node type constants for tree-sitter.

This module centralizes the tree-sitter grammar loading logic using ctypes
and defines constants for SystemVerilog node types used by the parser.

Grammar Source: Assumes a pre-compiled tree-sitter grammar library (e.g., sv.so)
based on a grammar like tree-sitter-verilog. The exact version compatibility
might depend on the tree-sitter library version used during compilation.


Why ctypes?: Tree-sitter's Python binding typically loads grammars via language
files (.so, .dll, .dylib) containing a specific C function (e.g., tree_sitter_verilog).
Since version 0.23.0 tree-sitter removed the ability directly initialize a 
Language from .so files: https://github.com/tree-sitter/py-tree-sitter/discussions/251
Using ctypes allows direct loading of this shared library and accessing the
language function pointer, which is then wrapped into a Python capsule that the
tree-sitter Python library understands. This avoids needing the grammar source
code at runtime, only the compiled library.
"""

import os
import ctypes
import logging
import inspect # Add inspect import
from ctypes import c_void_p, c_char_p, py_object, pythonapi
from typing import Optional
from tree_sitter import Language

logger = logging.getLogger(__name__)

# --- ADDED: Default grammar filename ---
DEFAULT_GRAMMAR_FILENAME = "sv.so"
# --- END ADDED ---

def load_language(grammar_path: Optional[str]) -> Language: # Changed type hint to Optional[str]
    """Loads the tree-sitter grammar from the specified path using ctypes.
       If grammar_path is None, attempts to load 'sv.so' from the same directory as this file.

    Args:
        grammar_path: Absolute path to the compiled grammar library (.so, .dll, .dylib),
                      or None to use the default path.

    Returns:
        A tree-sitter Language object.

    Raises:
        FileNotFoundError: If the grammar file does not exist.
        AttributeError: If the expected language function (tree_sitter_verilog) is not found.
        RuntimeError: For other ctypes or tree-sitter initialization errors.
    """
    # --- MODIFIED: Determine default path if None ---
    if grammar_path is None:
        # Get the directory containing this grammar.py file
        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        grammar_path = os.path.join(current_dir, DEFAULT_GRAMMAR_FILENAME)
        logger.info(f"Grammar path not provided, defaulting to: {grammar_path}")
    # --- END MODIFICATION ---

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

