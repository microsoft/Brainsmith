"""SystemVerilog RTL parser implementation.

This module implements the main RTL parser using tree-sitter to parse
SystemVerilog files and extract module interfaces, parameters, and pragmas.
"""

import os
import logging
import ctypes
from ctypes import c_void_p, c_char_p, py_object, pythonapi
from typing import Optional, List
from tree_sitter import Language, Parser, Tree, Node

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import HWKernel
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface import (
    extract_module_header,
    parse_parameter_declaration,
    parse_port_declaration,
    _debug_node
)
from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import extract_pragmas

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ParserError(Exception):
    """Base class for parser errors."""
    pass

class SyntaxError(ParserError):
    """Raised when SystemVerilog syntax is invalid."""
    pass

class RTLParser:
    """Parser for SystemVerilog RTL files.
    
    This class uses tree-sitter to parse SystemVerilog files and extract
    the information needed by the Hardware Kernel Generator.
    
    Attributes:
        parser: tree-sitter Parser instance
        debug: Enable debug output
    """
    
    def __init__(self, grammar_path: Optional[str] = None, debug: bool = False):
        """Initialize the RTL parser.
        
        Args:
            grammar_path: Path to SystemVerilog grammar .so file
                        If None, tries to find sv.so in the rtl_parser directory
            debug: Enable debug output
        
        Raises:
            FileNotFoundError: If SystemVerilog grammar file not found
            RuntimeError: If parser initialization fails
        """
        self.debug = debug
        
        # Find grammar file
        if grammar_path is None:
            grammar_path = os.path.join(
                os.path.dirname(__file__),
                "sv.so"
            )
        
        if not os.path.exists(grammar_path):
            raise FileNotFoundError(f"SystemVerilog grammar file not found at: {grammar_path}")
        
        # Initialize parser
        try:
            # 1. Load the shared object
            lib = ctypes.cdll.LoadLibrary(grammar_path)

            # 2. Get language pointer
            lang_ptr = lib.tree_sitter_verilog
            lang_ptr.restype = c_void_p
            lang_ptr = lang_ptr()

            # 3. Create Python capsule
            PyCapsule_New = pythonapi.PyCapsule_New
            PyCapsule_New.restype = py_object
            PyCapsule_New.argtypes = (c_void_p, c_char_p, c_void_p)
            capsule = PyCapsule_New(lang_ptr, b"tree_sitter.Language", None)

            # 4. Create parser with language
            language = Language(capsule)
            self.parser = Parser(language)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize parser: {e}")
    
    def parse_file(self, file_path: str) -> HWKernel:
        """Parse a SystemVerilog file.
        
        Args:
            file_path: Path to SystemVerilog file
            
        Returns:
            HWKernel instance containing parsed information
            
        Raises:
            FileNotFoundError: If file not found
            SyntaxError: If SystemVerilog syntax is invalid
            ParserError: If parsing fails for other reasons
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse file
        tree = self.parser.parse(bytes(source, 'utf8'))
        
        if self.debug:
            logger.debug("Full parse tree:")
            _debug_node(tree.root_node)
        
        # Check for syntax errors
        if tree.root_node.has_error:
            raise SyntaxError("Invalid SystemVerilog syntax")
        
        # Find module definition (search through source_file if needed)
        module_node = None
        if tree.root_node.type == "source_file":
            for child in tree.root_node.children:
                if child.type == "module_declaration":
                    module_node = child
                    break
        elif tree.root_node.type == "module_declaration":
            module_node = tree.root_node
        
        if module_node is None:
            raise ParserError("No module definition found")
        
        try:
            # Extract module components
            name, param_nodes, port_nodes = extract_module_header(module_node)
            
            # Create kernel instance
            kernel = HWKernel(name=name)
            
            # Extract parameters
            for node in param_nodes:
                logger.debug(f"Processing parameter node:")
                _debug_node(node)
                param = parse_parameter_declaration(node)
                if param is not None:
                    kernel.add_parameter(param)
            
            # Extract ports
            for node in port_nodes:
                logger.debug(f"Processing port node:")
                _debug_node(node)
                port = parse_port_declaration(node)
                if port is not None:
                    kernel.add_port(port)
            
            # Extract pragmas
            pragmas = extract_pragmas(tree.root_node)
            for pragma in pragmas:
                kernel.add_pragma(pragma)
            
            return kernel
            
        except ValueError as e:
            raise ParserError(str(e))
    
    def _find_module_node(self, root: Node) -> Optional[Node]:
        """Find the first module definition node.
        
        Args:
            root: Root AST node
            
        Returns:
            Module node if found, None otherwise
        """
        if root.type == "module_declaration":
            return root
        
        for child in root.children:
            if child.type == "module_declaration":
                return child
            result = self._find_module_node(child)
            if result is not None:
                return result
        
        return None