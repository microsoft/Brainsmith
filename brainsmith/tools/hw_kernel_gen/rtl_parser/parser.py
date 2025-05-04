"""SystemVerilog RTL parser implementation.

This module implements the main RTL parser using tree-sitter to parse
SystemVerilog files and extract module interfaces, parameters, and pragmas.
"""

import os
import logging
import ctypes
import re # Added import for regex
from ctypes import c_void_p, c_char_p, py_object, pythonapi
from typing import Optional, List, Tuple, Union # Added List
from tree_sitter import Language, Parser, Tree, Node
import collections

# Corrected import: Ensure ModuleSummary is imported here
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import HWKernel, Port, Parameter, Direction, ModuleSummary
from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import extract_pragmas, PragmaType, Pragma # Added Pragma import
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder
# Import Interface types for checking counts
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import InterfaceType

# Configure root logger level based on environment or config if needed
# logging.basicConfig(level=logging.INFO) # Example: Default to INFO
logger = logging.getLogger(__name__) # Get logger for this module

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
        # Set logger level based on debug flag
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

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
            # Instantiate InterfaceBuilder
            self.interface_builder = InterfaceBuilder(debug=debug)
            logger.info("RTLParser initialized successfully.")
        except FileNotFoundError as e:
            logger.error(f"Grammar file error: {e}")
            raise # Re-raise specific error
        except Exception as e:
            logger.exception(f"Failed to initialize parser: {e}") # Use exception for stack trace
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
            logger.error(f"Input file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Starting parsing for: {file_path}")
        # Read file
        try:
            with open(file_path, 'r') as f:
                source = f.read()
        except Exception as e:
            logger.exception(f"Failed to read file {file_path}: {e}")
            raise ParserError(f"Failed to read file {file_path}: {e}")

        # Parse file
        try:
            tree = self.parser.parse(bytes(source, 'utf8'))
        except Exception as e:
            # Catch potential errors during the parse call itself
            logger.exception(f"Tree-sitter parsing failed for {file_path}: {e}")
            raise ParserError(f"Core parsing failed for {file_path}: {e}")

        if self.debug:
            logger.debug(f"Raw parse tree for {file_path}:")
            # _debug_node(tree.root_node) # Keep this commented unless deep debugging needed

        # Check for syntax errors more specifically
        if tree.root_node.has_error:
            # Find first error node if possible (simple approach)
            error_node = self._find_first_error_node(tree.root_node)
            line = error_node.start_point[0] + 1 if error_node else 'unknown'
            col = error_node.start_point[1] + 1 if error_node else 'unknown'
            error_msg = f"Invalid SystemVerilog syntax near line {line}, column {col}."
            logger.error(f"Syntax error in {file_path} near line {line}:{col}")
            raise SyntaxError(error_msg)

        # Find module definition(s)
        module_nodes = self._find_module_nodes(tree.root_node)
        if not module_nodes:
            logger.error(f"No module definitions found in {file_path}")
            raise ParserError(f"No module definition found in {file_path}")

        # Extract pragmas first to check for TOP_MODULE
        logger.debug("Extracting pragmas...")
        try:
            pragmas = extract_pragmas(tree.root_node)
        except Exception as e:
            logger.exception(f"Error during pragma extraction in {file_path}: {e}")
            raise ParserError(f"Failed during pragma extraction: {e}")
        logger.debug(f"Found {len(pragmas)} potential pragmas.")

        # Select the target module node
        try:
            module_node = self._select_target_module(module_nodes, pragmas, file_path)
        except ParserError as e:
            logger.error(e) # Log the specific error from selection logic
            raise

        # Corrected line 170
        # logger.info(f"Target module identified: '{module_node.child_by_field_name("name").text.decode()}'")

        # --- Start processing the selected module ---
        try:
            # Extract module components
            logger.debug("Extracting module header...")
            # Use the new private method
            name, param_nodes, port_nodes = self._extract_module_header(module_node)
            logger.debug(f"Extracted header for module '{name}'")

            # Create kernel instance
            kernel = HWKernel(name=name)
            # kernel.metadata['source_file'] = file_path # Consider adding metadata if needed

            kernel.pragmas = pragmas # Assign extracted pragmas

            # Extract parameters
            logger.debug("Extracting parameters...")
            for node in param_nodes:
                # logger.debug(f"Processing parameter node:")
                # self._debug_node(node) # Use the new private method if debugging
                # Use the new private method
                param = self._parse_parameter_declaration(node)
                if param is not None:
                    kernel.parameters.append(param)
            logger.debug(f"Extracted {len(kernel.parameters)} parameters.")

            # Extract ports
            logger.debug("Extracting ports...")
            extracted_ports: List[Port] = []
            for node in port_nodes:
                # logger.debug(f"Processing port node:")
                # self._debug_node(node) # Use the new private method if debugging
                # Use the new private method
                parsed_port_list = self._parse_port_declaration(node) # Returns List[Port]
                if parsed_port_list: # Check if the list is not empty
                    extracted_ports.extend(parsed_port_list) # Use extend to add elements
            kernel.ports = extracted_ports
            # Correctly log the total number of Port objects extracted
            logger.debug(f"Extracted {len(kernel.ports)} individual port objects.")

            # --- Integrate Interface Analysis ---
            logger.info(f"Starting interface analysis for module {kernel.name}...")
            validated_interfaces, unassigned_ports = self.interface_builder.build_interfaces(kernel.ports)
            kernel.interfaces = validated_interfaces
            logger.info(f"Interface analysis complete. Found {len(kernel.interfaces)} valid interfaces.")

            # --- Modify Post-Analysis Validation ---
            # 1. Check for unassigned ports - Log as warning instead of error
            if unassigned_ports:
                 unassigned_names = [p.name for p in unassigned_ports]
                 # Changed from error to warning
                 warning_msg = f"Module '{kernel.name}' has {len(unassigned_ports)} ports not assigned to any standard interface: {unassigned_names}"
                 logger.warning(warning_msg)
                 # Store unassigned ports for potential later use (optional, requires adding to HWKernel)
                 # kernel.unassigned_ports = unassigned_ports
                 # Temporarily disabled: raise ParserError(error_msg)

            # 2. Check interface counts (Keep existing logic if needed)
            # Example: Ensure at least one AXI-Stream if required by project spec
            has_axi_stream = any(iface.type == InterfaceType.AXI_STREAM for iface in kernel.interfaces.values())
            if not has_axi_stream:
                 # Keep this error if it's a hard requirement
                 error_msg = f"Module '{kernel.name}' requires at least one AXI-Stream interface, but found none after analysis."
                 logger.error(error_msg)
                 raise ParserError(error_msg)

            # --- Add Placeholders for Future Data Processing ---
            # TODO: Implement Kernel Parameter formatting (using kernel.parameters)
            #       Result should be stored in kernel.kernel_parameters

            # TODO: Implement Compiler Flag inference (using kernel.pragmas)
            #       Result should be stored in kernel.compiler_flags

            logger.info(f"Successfully parsed and validated module '{kernel.name}' from {file_path}")
            return kernel

        # Consolidate exception handling for different stages
        except FileNotFoundError as e:
            # This specific error is already handled before the main try block
            logger.error(f"File not found error during parsing: {e}") # Should not happen here ideally
            raise ParserError(f"File not found: {file_path}")
        except SyntaxError as e:
            # Already logged, just re-raise
            raise
        except ParserError as e:
            # Catch specific errors raised during processing (module selection, validation etc.)
            logger.error(f"Parser processing error for {file_path}: {e}")
            raise # Re-raise the specific parser error
        except Exception as e:
            # Catch unexpected errors during module processing
            logger.exception(f"An unexpected error occurred during module processing of {file_path}: {e}")
            raise ParserError(f"An unexpected error occurred during module processing: {e}")

    def _find_first_error_node(self, node: Node) -> Optional[Node]:
        """Simple BFS to find the first node marked with an error."""
        queue = [node]
        visited = {node.id}
        while queue:
            current = queue.pop(0)
            if current.has_error or current.is_missing:
                # Try to find a more specific child error first
                for child in current.children:
                     if child.has_error or child.is_missing:
                         return child # Return first child error
                return current # Return parent if no child has specific error

            for child in current.children:
                if child.id not in visited:
                    visited.add(child.id)
                    queue.append(child)
        return None # No error node found

    def _find_module_nodes(self, root: Node) -> List[Node]:
        """Find all top-level module definition nodes using BFS."""
        nodes = []
        if root.type == "source_file":
             for child in root.children:
                 if child.type == "module_declaration":
                     nodes.append(child)
        elif root.type == "module_declaration": # Handle case where root is the module
             nodes.append(root)
        return nodes

    def _select_target_module(self, module_nodes: List[Node], pragmas: List["Pragma"], file_path: str) -> Node:
        """Select the target module based on count and TOP_MODULE pragma."""
        top_module_pragmas = [p for p in pragmas if p.type == PragmaType.TOP_MODULE.value]

        # Extract module names using the helper function
        module_names_map = {}
        for node in module_nodes:
            name, _, _ = self._extract_module_header(node)
            if name:
                module_names_map[name] = node
            else:
                # Log or handle cases where name extraction fails for a node
                logger.warning(f"Could not extract module name from node: {node.text.decode()[:50]}...")

        if len(module_nodes) == 1 and not top_module_pragmas:
            logger.debug("Found single module, selecting it as target.")
            return module_nodes[0]
        elif len(module_nodes) > 1:
            if len(top_module_pragmas) == 1:
                target_name = top_module_pragmas[0].processed_data.get("module_name")
                logger.info(f"Found TOP_MODULE pragma, searching for module '{target_name}'.")
                if target_name in module_names_map:
                     logger.debug(f"Found matching module '{target_name}'.")
                     return module_names_map[target_name]
                else:
                     raise ParserError(f"TOP_MODULE pragma specified '{target_name}', but no such module found in {file_path}.")
            elif len(top_module_pragmas) > 1:
                raise ParserError(f"Multiple TOP_MODULE pragmas found in {file_path}. Only one is allowed.")
            else: # Multiple modules, no pragma
                raise ParserError(f"Multiple modules ({list(module_names_map.keys())}) found in {file_path}, but no TOP_MODULE pragma specified.")
        elif len(module_nodes) == 1 and top_module_pragmas:
             # Single module, but pragma exists - check if it matches
             target_name = top_module_pragmas[0].processed_data.get("module_name")
             # Get the actual name from the single node using the helper
             actual_name, _, _ = self._extract_module_header(module_nodes[0])
             if not actual_name:
                 # This case should be less likely now, but handle it
                 raise ParserError(f"Could not determine module name for comparison with TOP_MODULE pragma '{target_name}'.")

             if actual_name == target_name:
                 logger.debug(f"Found single module '{actual_name}' matching TOP_MODULE pragma.")
                 return module_nodes[0]
             else:
                 # Now uses extracted name
                 raise ParserError(f"TOP_MODULE pragma specifies '{target_name}', but the only module found is '{actual_name}'.")
        else:
             # Should not happen if _find_module_nodes works correctly
             raise ParserError("Internal error: Inconsistent module node state.")

    # --- Re-introduced Helper Function ---
    def _extract_module_header(self, module_node: Node) -> Tuple[Optional[str], Optional[List[Node]], Optional[List[Node]]]:
        """
        Extracts module name, parameter nodes, and port nodes from a module_declaration node.
        Handles both ANSI and potentially non-ANSI header structures recognized by tree-sitter.

        Returns:
            Tuple: (module_name, parameter_nodes, port_nodes)
                   Returns (None, None, None) if essential parts are missing.
        """
        if not module_node or module_node.type != "module_declaration":
            logger.error("Invalid node passed to _extract_module_header. Expected 'module_declaration'.")
            return None, None, None

        module_name: Optional[str] = None
        param_nodes: Optional[List[Node]] = []
        port_nodes: Optional[List[Node]] = []
        header_node: Optional[Node] = None # Keep track of the header node

        # Find module identifier (name)
        name_node = self._find_child(module_node, ["simple_identifier", "identifier"]) # Direct child?
        if not name_node:
             header_node = self._find_child(module_node, ["module_ansi_header", "module_nonansi_header"]) # Look in headers
             if header_node:
                  name_node = self._find_child(header_node, ["simple_identifier", "identifier"])

        if name_node:
            module_name = name_node.text.decode('utf8')
            logger.debug(f"Extracted module name: {module_name}")
        else:
            logger.warning(f"Could not find module name identifier within node: {module_node.text.decode()[:50]}...")

        # --- Search for lists within the appropriate node ---
        # If header_node was found during name search, use it. Otherwise, search module_node directly.
        search_parent_node = header_node if header_node else module_node
        logger.debug(f"Searching for parameter/port lists within node type: {search_parent_node.type}")

        # Find parameter list node within the search_parent_node
        param_list_node = self._find_child(search_parent_node, ["parameter_port_list"])
        if param_list_node:
            # Extract individual parameter declarations within the list
            param_nodes = self._find_children(param_list_node, ["parameter_port_declaration"])
            logger.debug(f"Found parameter list node containing {len(param_nodes)} declarations.")
        else:
            logger.debug("No parameter list node found.")

        # Find port list node (ANSI style) within the search_parent_node
        port_list_node = self._find_child(search_parent_node, ["list_of_port_declarations"])
        if port_list_node:
            # Extract individual port declarations within the list
            port_nodes = self._find_children(port_list_node, ["ansi_port_declaration"]) # Specific to ANSI
            logger.debug(f"Found ANSI port list node containing {len(port_nodes)} declarations.")
        else:
            # TODO: Add logic for non-ANSI ports if needed (search module body items)
            logger.debug("No ANSI port list node found. Non-ANSI port extraction not yet implemented.")


        return module_name, param_nodes, port_nodes

    # --- Existing Helper Functions ---
    def _debug_node(self, node: Node, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> None:
        """Debug helper to print node structure with depth limit."""
        if node is None or current_depth > max_depth:
            return
        indent = "  " * current_depth
        node_text_raw = node.text.decode('utf8')
        # Limit displayed text and escape newlines for cleaner logging
        node_text_display = node_text_raw.replace('\n', '\\n')[:80]
        if len(node_text_raw) > 80:
             node_text_display += "..."

        logger.debug(f"{prefix}{indent}Node type: {node.type}, text: '{node_text_display}' (ID: {node.id})")
        for i, child in enumerate(node.children):
            # Pass max_depth and increment current_depth in recursive call
            self._debug_node(child, prefix=f"{prefix}Child {i}: ", max_depth=max_depth, current_depth=current_depth + 1)

    def _extract_direction(self, node: Node) -> Optional[Direction]:
        """Extract the direction (input, output, inout) from a port declaration node."""
        if node is None:
            return None

        direction = None
        direction_types = ["input", "output", "inout"]
        direction_node = self._find_child(node, ["port_direction"] + direction_types)
        if direction_node:
            dir_text = direction_node.text.decode('utf8')
            # Handle cases where the node type itself is the direction (e.g., 'input')
            if dir_text in direction_types:
                direction = Direction(dir_text)
            elif direction_node.type == "port_direction":
                # Find the actual keyword within the port_direction node
                for child in direction_node.children:
                    if child.text.decode('utf8') in direction_types:
                        direction = Direction(child.text.decode('utf8'))
                        break

        if direction is None: # Fallback for simpler structures if needed
            node_text = node.text.decode('utf8')
            first_word = node_text.split()[0] if node_text else ""
            if first_word in direction_types: direction = Direction(first_word)

        return direction

    def _find_identifiers_recursive(self, node: Node) -> List[str]:
        """Recursively find all simple_identifier or port_identifier texts under a node."""
        identifiers = []
        node_type = node.type
        node_text = node.text.decode('utf8').strip()

        # Base case: If it's an identifier node, add its text
        # Exclude common keywords that might appear as identifiers in the AST
        # Also exclude known type names that might be parsed as identifiers in some contexts
        keywords_to_exclude = [d.value for d in Direction] + \
                              ['logic', 'reg', 'wire', 'bit', 'integer', 'input', 'output', 'inout', 'signed', 'unsigned', 'parameter', 'localparam', 'module', 'endmodule', 'interface', 'endinterface'] # Common types/modifiers/keywords

        if node_type in ["simple_identifier", "identifier", "port_identifier"] and node_text not in keywords_to_exclude:
             # Check parent type to avoid grabbing module name identifier if node is module_identifier
             if not (node.parent and node.parent.type in ["module_declaration", "module_identifier", "interface_identifier"]):
                 identifiers.append(node_text)


        # Recursive step: Traverse children
        for child in node.children:
            # Avoid recursing into the data type definition itself if it looks like an identifier
            # This prevents extracting 'logic' from 'input logic clk' if 'logic' is parsed as an identifier within the type node
            # Also skip recursing into parameter declarations if we are looking for ports
            if child.type not in ['data_type', 'parameter_port_list', 'parameter_declaration']: # Simple check, might need refinement
                 identifiers.extend(self._find_identifiers_recursive(child))

        # Return unique identifiers found in this subtree
        # Using dict.fromkeys preserves order and ensures uniqueness efficiently
        return list(dict.fromkeys(identifiers))

    def _parse_port_declaration(self, node: Node) -> List[Port]:
        """
        Parses a port declaration node (ANSI or non-ANSI) and returns a list of Port objects.
        Refined based on detailed AST analysis.
        """
        logger.debug(f"Parsing port declaration node: {node.text.decode()}")
        # self._debug_node(node, "PortDecl", max_depth=5) # Uncomment for deep debug

        direction = self._extract_direction(node)
        if direction is None:
            logger.warning(f"Could not determine direction for port declaration: {node.text.decode()}")
            return [] # Cannot proceed without direction

        data_type = "logic" # Default
        final_width = "1"   # Default
        port_names = []

        # --- Determine Header Type (Variable, Net, Interface, Implicit) ---
        variable_port_header = self._find_child(node, ["variable_port_header"])
        net_port_header = self._find_child(node, ["net_port_header"])
        interface_port_header = self._find_child(node, ["interface_port_header"]) # Check for interface header

        # --- Extract Type and Width based on Header ---
        if variable_port_header:
            logger.debug("Parsing as Variable Port Header")
            variable_port_type = self._find_child(variable_port_header, ["variable_port_type"])
            if variable_port_type:
                # Data Type: First child (usually data_type node)
                dt_node = self._find_child(variable_port_type, ["data_type"])
                if dt_node:
                    # Extract base type from within data_type
                    core_type_node = self._find_child(dt_node, ["signing", "integer_vector_type", "integer_atom_type", "non_integer_type", "simple_identifier", "ps_identifier", "identifier", "data_type_identifier"])
                    if core_type_node: data_type = core_type_node.text.decode('utf8').strip()
                    else: data_type = dt_node.text.decode('utf8').strip() # Fallback
                else: # Should have a data_type child based on AST
                     logger.warning("No data_type node found within variable_port_type")

                # Width: Sibling of data_type within variable_port_type
                width_node = None
                # 1. Search directly within variable_port_type first
                width_node = self._find_child(variable_port_type, ["packed_dimension", "unpacked_dimension"])
                if width_node:
                    logger.debug("Found width node directly within variable_port_type.")
                elif dt_node: # 2. Fallback: Check siblings of dt_node (original logic)
                    logger.debug("Width node not direct child of variable_port_type, checking siblings of data_type.")
                    sibling = dt_node.next_sibling
                    if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]:
                        width_node = sibling
                        logger.debug("Found width node as next sibling of data_type in variable_port_type")
                    else: # Check previous just in case
                        sibling = dt_node.prev_sibling
                        if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]:
                             width_node = sibling
                             logger.debug("Found width node as prev sibling of data_type in variable_port_type")
                # 3. ADDED Fallback: Check as child of dt_node
                elif dt_node:
                    logger.debug("Width node not sibling, checking as child of data_type.")
                    width_node = self._find_child(dt_node, ["packed_dimension", "unpacked_dimension"])
                    if width_node:
                        logger.debug("Found width node as child of data_type.")


                # Process the found width_node (if any)
                if width_node:
                    logger.debug(f"Found potential width node (variable): Type={width_node.type}, Text='{width_node.text.decode()}'")
                    final_width = self._extract_width_from_dimension(width_node)
                else:
                    logger.debug("No width node found associated with variable_port_type.")
            else:
                logger.warning("No variable_port_type found within variable_port_header")

        elif net_port_header:
            logger.debug("Parsing as Net Port Header")
            net_port_type = self._find_child(net_port_header, ["net_port_type"])
            if net_port_type:
                # Data Type: First child (usually net_type node)
                nt_node = self._find_child(net_port_type, ["net_type"])
                if nt_node: data_type = nt_node.text.decode('utf8').strip()
                else: logger.warning("No net_type node found within net_port_type")

                # Width: Nested under data_type_or_implicit -> implicit_data_type
                dtoi_node = self._find_child(net_port_type, ["data_type_or_implicit"])
                if dtoi_node:
                    idt_node = self._find_child(dtoi_node, ["implicit_data_type"])
                    if idt_node:
                        width_node = self._find_child(idt_node, ["packed_dimension", "unpacked_dimension"])
                        if width_node:
                            # ADDED LOG
                            logger.debug(f"Found potential width node (net): Type={width_node.type}, Text='{width_node.text.decode()}'")
                            final_width = self._extract_width_from_dimension(width_node)
                        else:
                            # ADDED LOG
                             logger.debug("No width node found nested in net_port_type.")
                    else:
                        # ADDED LOG
                         logger.debug("No implicit_data_type node found, cannot search for width.")
            elif self._find_child(net_port_header, ["port_direction"]):
                 # Handle implicit type (e.g., "input enable_in") - header only has direction
                 data_type = "wire" # Default implicit type
                 logger.debug("Parsing as Implicit Net Port (defaulting type to wire)")
            else:
                 logger.warning("No net_port_type or direction found within net_port_header")


        elif interface_port_header:
             logger.debug("Parsing as Interface Port Header")
             # Extract interface type name (e.g., 'axi_if')
             if_identifier_node = self._find_child(interface_port_header, ["interface_identifier"])
             if if_identifier_node:
                  data_type = if_identifier_node.text.decode('utf8').strip()
                  # Modport might be a sibling or child depending on grammar details
                  modport_node = self._find_child(interface_port_header, ["modport_identifier"])
                  if modport_node:
                       data_type += "." + modport_node.text.decode('utf8').strip()
                  logger.debug(f"Interface type extracted as: {data_type}")
             else:
                  logger.warning("Could not find interface_identifier within interface_port_header")
             # Width is typically not applicable or '1' for interface ports themselves
             final_width = "1"

        else: # Fallback/Non-ANSI (might need more robust handling if mixed styles occur)
            logger.warning(f"Could not identify standard ANSI header type for: {node.text.decode()}. Attempting fallback.")
            # Basic fallback: look for type and dimension directly under the node
            dt_node = self._find_child(node, ["data_type"])
            if dt_node: data_type = dt_node.text.decode('utf8').strip()
            width_node = self._find_child(node, ["packed_dimension", "unpacked_dimension"])
            if width_node: final_width = self._extract_width_from_dimension(width_node)


        # --- Extract Port Name(s) ---
        # Name is usually the last simple_identifier sibling within the ansi_port_declaration
        # Or search recursively if it's a list
        list_of_ids_node = self._find_child(node, ["list_of_port_identifiers", "list_of_variable_identifiers"])
        if list_of_ids_node:
             potential_names = self._find_identifiers_recursive(list_of_ids_node)
        else:
             # Find last identifier sibling as primary candidate
             last_identifier = None
             for child in reversed(node.children):
                  if child.type == "simple_identifier":
                       last_identifier = child
                       break
                  # Handle ERROR node for interface ports - name might be after ERROR
                  if child.type == "ERROR" and child.prev_sibling and child.prev_sibling.type == "simple_identifier":
                       last_identifier = child.prev_sibling
                       logger.debug("Adjusting name search due to ERROR node (interface port).")
                       break


             if last_identifier:
                  potential_names = [last_identifier.text.decode('utf8').strip()]
             else: # Absolute fallback: recursive search on the whole node
                  potential_names = self._find_identifiers_recursive(node)

        logger.debug(f"Potential names found: {potential_names}")

        # --- Filter and Deduplicate Names ---
        filtered_names = []
        seen_names = set()
        keywords_to_exclude_set = set([d.value for d in Direction] + [t.strip() for t in data_type.split('.')]) # Exclude base type and modport if present

        for name in potential_names:
            if name and name not in keywords_to_exclude_set and name not in seen_names:
                filtered_names.append(name)
                seen_names.add(name)
        port_names = filtered_names

        logger.debug(f"Filtered port names: {port_names}")

        if not port_names:
             logger.warning(f"Failed to extract any valid port names from node: {node.text.decode()}")
             return []

        # --- Create Port objects ---
        parsed_ports = []
        for name in port_names:
            logger.info(f"Successfully parsed port: Name='{name}', Direction='{direction.value}', Width='{final_width}', Type='{data_type}'")
            parsed_ports.append(Port(name=name, direction=direction, width=final_width)) # Assuming type isn't stored in Port object

        return parsed_ports

    def _extract_width_from_dimension(self, width_node: Node) -> str:
        """Helper to extract text content from dimension nodes."""
        if not width_node: return "1"
        logger.debug(f"Extracting width from node: Type={width_node.type}, Text='{width_node.text.decode()}'")

        # Prioritize finding the range or expression node within the dimension
        expr_node = self._find_child(width_node, ["constant_range", "range_expression", "constant_expression", "expression", "primary_literal", "number"])

        if expr_node:
            # ADDED LOG
            logger.debug(f"Found expression node: Type={expr_node.type}, Text='{expr_node.text.decode()}'")
            width_text = expr_node.text.decode('utf8').strip()
            logger.debug(f"Width expression text found: '{width_text}'")
            # Check if the found expression is the full content between brackets
            full_node_text = width_node.text.decode('utf8').strip()
            if full_node_text.startswith('[') and full_node_text.endswith(']'):
                expected_inner_text = full_node_text[1:-1].strip()
                 # ADDED LOG
                logger.debug(f"Full node inner text: '{expected_inner_text}'")
                if width_text == expected_inner_text:
                     # ADDED LOG
                    logger.debug("Expression node text matches full inner text.")
                    return width_text # Perfect match
                else:
                    # Sometimes the expr_node might be nested deeper, use the full inner text
                    logger.debug(f"Expression node text ('{width_text}') differs from node inner text ('{expected_inner_text}'), using inner text.")
                    return expected_inner_text if expected_inner_text else "1"
            else:
                 # If original node wasn't bracketed (less common), use expr_node text
                  # ADDED LOG
                 logger.debug("Original width node not bracketed, using expression node text.")
                 return width_text
        else:
             # ADDED LOG
            logger.debug("No specific expression node found within width_node.")
            # Fallback: Use cleaned text of the dimension node itself, removing brackets
            cleaned_width_text = width_node.text.decode('utf8').strip()
            if cleaned_width_text.startswith('[') and cleaned_width_text.endswith(']'):
                cleaned_width_text = cleaned_width_text[1:-1].strip()
             # ADDED LOG
            logger.debug(f"Using fallback cleaned text: '{cleaned_width_text}'")
            return cleaned_width_text if cleaned_width_text else "1" # Return cleaned text or default

    # Ensure _find_child, _extract_direction, _find_identifiers_recursive are present and correct
    def _find_child(self, node: Node, types: List[str]) -> Optional[Node]:
        """Find the first direct child node matching any of the given types."""
        if not node: return None
        for child in node.children:
            if child.type in types:
                return child
        return None

    def _find_children(self, node: Node, types: List[str]) -> List[Node]:
        """Find all direct children nodes matching any of the given types."""
        found_nodes = []
        if not node: return found_nodes
        for child in node.children:
            if child.type in types:
                found_nodes.append(child)
        return found_nodes

    # --- Add Parameter Parsing Method ---
    def _parse_parameter_declaration(self, node: Node) -> Optional[Parameter]:
        """
        Parses a parameter_port_declaration node and returns a Parameter object.
        """
        if not node or node.type != "parameter_port_declaration":
            logger.warning(f"Invalid node type passed to _parse_parameter_declaration: {node.type}")
            return None

        param_name: Optional[str] = None
        param_type: str = "parameter" # Default type if not specified
        default_value: Optional[str] = None

        # Find the core parameter_declaration or local_parameter_declaration
        param_decl_node = self._find_child(node, ["parameter_declaration", "local_parameter_declaration"])
        if not param_decl_node:
            logger.warning(f"Could not find parameter_declaration or local_parameter_declaration within: {node.text.decode()}")
            # Try finding assignment directly under parameter_port_declaration as fallback
            param_decl_node = node

        # Determine if localparam
        is_local = param_decl_node.type == "local_parameter_declaration" or "localparam" in param_decl_node.text.decode().split()

        # Extract type if present (often within data_type_or_implicit)
        dtoi_node = self._find_child(param_decl_node, ["data_type_or_implicit"])
        if dtoi_node:
            dt_node = self._find_child(dtoi_node, ["data_type"])
            if dt_node:
                # Extract specific type like 'int', 'integer', etc.
                core_type_node = self._find_child(dt_node, ["integer_atom_type", "integer_vector_type", "non_integer_type", "signing", "simple_identifier"])
                if core_type_node:
                    param_type = core_type_node.text.decode('utf8').strip()
                else:
                    param_type = dt_node.text.decode('utf8').strip() # Fallback to full data_type text
                logger.debug(f"Parameter type found: {param_type}")

        # Find the assignment part (list_of_param_assignments -> param_assignment)
        assignment_list_node = self._find_child(param_decl_node, ["list_of_param_assignments"])
        if assignment_list_node:
            assignment_node = self._find_child(assignment_list_node, ["param_assignment"])
            if assignment_node:
                # Extract name (simple_identifier)
                name_node = self._find_child(assignment_node, ["simple_identifier", "identifier"])
                if name_node:
                    param_name = name_node.text.decode('utf8').strip()
                else:
                    logger.warning(f"Could not find parameter name in assignment: {assignment_node.text.decode()}")
                    return None # Name is essential

                # Extract default value (constant_param_expression -> constant_expression)
                value_expr_node = self._find_child(assignment_node, ["constant_param_expression", "constant_expression", "expression"])
                if value_expr_node:
                    # Further drill down for cleaner expression text if possible
                    inner_expr = self._find_child(value_expr_node, ["constant_mintypmax_expression", "constant_expression", "primary_literal", "binary_expression"])
                    if inner_expr:
                         default_value = inner_expr.text.decode('utf8').strip()
                    else:
                         default_value = value_expr_node.text.decode('utf8').strip() # Fallback
                    logger.debug(f"Parameter '{param_name}' default value found: {default_value}")
            else:
                 logger.warning(f"Could not find param_assignment within list: {assignment_list_node.text.decode()}")
                 return None # Cannot get name/value without assignment
        else:
             logger.warning(f"Could not find list_of_param_assignments in: {param_decl_node.text.decode()}")
             # Maybe a declaration without assignment? Try finding name directly
             name_node = self._find_child(param_decl_node, ["simple_identifier", "identifier"])
             if name_node:
                  param_name = name_node.text.decode('utf8').strip()
             else:
                  return None # Still need a name


        if param_name:
            logger.info(f"Successfully parsed parameter: Name='{param_name}', Type='{param_type}', Default='{default_value}', Local={is_local}")
            # Corrected: Use 'default_value' and include 'param_type'
            return Parameter(name=param_name, param_type=param_type, default_value=default_value)
        else:
            return None

    # ... (rest of the class) ...