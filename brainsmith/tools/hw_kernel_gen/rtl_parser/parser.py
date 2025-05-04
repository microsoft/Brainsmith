"""SystemVerilog RTL parser implementation.

This module implements the main RTL parser using tree-sitter to parse
SystemVerilog files and extract module interfaces, parameters, and pragmas.
"""

import os
import logging
import ctypes
import re # Added import for regex
from ctypes import c_void_p, c_char_p, py_object, pythonapi
from typing import Optional, List, Tuple
from tree_sitter import Language, Parser, Tree, Node
import collections

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import HWKernel, Port, Parameter, Direction # Added Direction
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
                port = self._parse_port_declaration(node)
                if port is not None:
                    extracted_ports.append(port)
            kernel.ports = extracted_ports
            logger.debug(f"Extracted {len(kernel.ports)} ports.")

            # --- Integrate Interface Analysis ---
            logger.info(f"Starting interface analysis for module {kernel.name}...")
            validated_interfaces, unassigned_ports = self.interface_builder.build_interfaces(kernel.ports)
            kernel.interfaces = validated_interfaces
            logger.info(f"Interface analysis complete. Found {len(kernel.interfaces)} valid interfaces.")

            # --- Add Post-Analysis Validation ---
            # 1. Check for unassigned ports
            if unassigned_ports:
                 unassigned_names = [p.name for p in unassigned_ports]
                 error_msg = f"Module '{kernel.name}' has {len(unassigned_ports)} ports not assigned to any valid interface: {unassigned_names}"
                 logger.error(error_msg)
                 raise ParserError(error_msg)

            # 2. Check interface counts
            axi_stream_count = sum(1 for iface in kernel.interfaces.values() if iface.type == InterfaceType.AXI_STREAM)
            axi_lite_count = sum(1 for iface in kernel.interfaces.values() if iface.type == InterfaceType.AXI_LITE)

            if axi_stream_count == 0:
                error_msg = f"Module '{kernel.name}' requires at least one AXI-Stream interface, but found none."
                logger.error(error_msg)
                raise ParserError(error_msg)

            if axi_lite_count > 1:
                error_msg = f"Module '{kernel.name}' allows at most one AXI-Lite interface, but found {axi_lite_count}."
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

    # --- Moved Helper Functions from interface.py ---

    def _debug_node(self, node: Node, prefix: str = "") -> None:
        """Debug helper to print node structure."""
        if node is None:
            return
        logger.debug(f"{prefix}Node type: {node.type}, text: {node.text.decode('utf8')}")
        for child in node.children:
            self._debug_node(child, prefix + "  ") # Call recursively using self

    def _parse_port_declaration(self, node: Node) -> Optional[Port]:
        """Parse a port declaration node (handles ANSI and non-ANSI styles)."""
        if node is None: return None
        # logger.debug("\\nParsing port declaration:")
        # self._debug_node(node)

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

        if direction is None: return None # Cannot determine direction

        # --- Extract Data Type --- (Added)
        data_type = "logic" # Default if not specified
        type_node = self._find_child(node, ["data_type", "implicit_data_type", "simple_type", "type_identifier"])
        if type_node:
            # Extract text, handling potential nested structures in data_type
            type_parts = [tn.text.decode('utf8') for tn in type_node.children if tn.type != 'signing'] # Exclude 'signed'/'unsigned' for now
            if type_parts:
                data_type = " ".join(type_parts).strip()
            else:
                data_type = type_node.text.decode('utf8').strip()
            # Handle common case like 'logic signed' -> 'logic'
            if data_type.startswith("logic") or data_type.startswith("reg") or data_type.startswith("wire"):
                 data_type = data_type.split()[0]

        # --- Extract Width --- (Refined)
        width = "1" # Default width
        width_text_representation = None # Store the full text like "[WIDTH-1:0]"
        # Find the node representing the dimension/range, e.g., packed_dimension `[7:0]`
        range_node = self._find_child(node, ["packed_dimension", "vector_dimension", "range_expression"])
        if range_node:
            # Extract the text content of the range node itself
            range_text = range_node.text.decode('utf8').strip()
            # Remove the outer brackets if present
            if range_text.startswith('[') and range_text.endswith(']'):
                width = range_text[1:-1].strip()
                width_text_representation = range_text # Store with brackets
            else:
                # Fallback or warning if brackets aren't found as expected
                logger.warning(f"Could not parse width from range node text: '{range_text}'")
                width = range_text # Use the text as is, might be incorrect
                width_text_representation = f"[{width}]" # Assume brackets were intended

        # --- Extract Name --- (Revised Logic)
        name = None
        port_identifier_node = self._find_child(node, ["port_identifier"])

        if port_identifier_node:
            name = port_identifier_node.text.decode('utf8').strip()
            # Basic validation: ensure it's not a direction keyword if found as identifier
            if name in [d.value for d in Direction]:
                 logger.warning(f"Port identifier node text '{name}' matches a direction keyword. Discarding.")
                 name = None # Avoid using direction keywords as names

        # Fallback: If no specific port_identifier found, try the previous logic
        # (useful for potentially different non-ANSI structures if they reach here)
        if name is None:
            logger.debug(f"No 'port_identifier' child found for node: {node.text.decode()}. Trying fallback.")
            last_identifier_node = None
            # Iterate through direct children to find the last identifier
            for child in node.children:
                # Check for general identifier types, excluding type/direction keywords explicitly
                if child.type in ["simple_identifier", "identifier"] and \
                   child.text.decode('utf8') not in [d.value for d in Direction] and \
                   child.text.decode('utf8') != data_type: # Check against extracted data_type
                    last_identifier_node = child # Keep updating to get the last one

            if last_identifier_node:
                potential_name = last_identifier_node.text.decode('utf8')
                # Double-check it's not a keyword we already identified
                if potential_name != direction.value and potential_name != data_type:
                     name = potential_name
                # Handle edge case: Sometimes the type node itself is the last identifier
                elif type_node and last_identifier_node.id == type_node.id:
                     logger.warning(f"Fallback failed: Last identifier was the type node for: {node.text.decode()}")
                     name = None # Explicitly set to None
                else:
                     logger.warning(f"Fallback check failed for potential name '{potential_name}' from node: {node.text.decode()}")
                     name = None # Explicitly set to None


        if name is None:
             logger.debug(f"Failed to extract port name from node: {node.text.decode()}")
             # self._debug_node(node, prefix="Failed Node: ") # Uncomment for deep debug
             return None # Cannot determine name

        logger.debug(f"Parsed port: Name='{name}', Direction='{direction.value}', Width='{width}', Type='{data_type}'")
        # TODO: Consider adding data_type to the Port object if needed later
        return Port(name=name, direction=direction, width=width)


    def _parse_parameter_declaration(self, node: Node) -> Optional[Parameter]:
        """Parse a parameter declaration node."""
        if node is None: return None
        # logger.debug("\\nParsing parameter declaration:") # Optional: enable if needed
        # self._debug_node(node)

        if node.type == "localparam_declaration" or self._has_text(node, "localparam"): # Use self._has_text
            return None

        param_type = "logic"
        type_node = self._find_child(node, ["data_type", "type_identifier", "simple_type"]) # Use self._find_child
        if type_node is not None: param_type = type_node.text.decode('utf8')

        name_node = self._find_child(node, ["simple_identifier", "identifier", "parameter_identifier"]) # Use self._find_child
        if name_node is None: return None
        name = name_node.text.decode('utf8')

        default_value = None
        equals = self._find_child(node, "=") # Use self._find_child
        if equals is not None:
            current = equals.next_sibling
            if current is not None:
                value_text = []
                while current and current.type != ";":
                    value_text.append(current.text.decode('utf8'))
                    current = current.next_sibling
                if value_text: default_value = "".join(value_text).strip()

        return Parameter(name=name, param_type=param_type, default_value=default_value)

    def _find_nodes_recursively(self, node: Node, target_types: List[str]) -> List[Node]:
        """Recursively find all nodes of specified types using BFS."""
        found_nodes = []
        if node is None:
            return found_nodes

        queue = [node]
        # No need for visited set in a typical AST traversal unless cycles are possible

        while queue:
            current_node = queue.pop(0)
            if current_node.type in target_types:
                found_nodes.append(current_node)

            # Add children to the queue for further exploration
            queue.extend(current_node.children)
        return found_nodes

    def _extract_module_header(self, node: Node) -> Tuple[str, List[Node], List[Node]]:
        """Extract key components from module header."""
        if node is None: raise ValueError("Invalid module node")
        # logger.debug("\\nExtracting module header:") # Optional: enable if needed
        # self._debug_node(node)

        name_node = self._find_child(node, ["module_identifier", "simple_identifier", "identifier"]) # Use self._find_child
        if name_node is None: raise ValueError("Module name not found")
        name = name_node.text.decode('utf8')

        param_nodes = []
        param_list = self._find_child(node, ["parameter_port_list", "list_of_parameter_declarations"]) # Use self._find_child
        if param_list is not None:
            for child in param_list.children:
                if child.type in ["parameter_declaration", "parameter_port_declaration"]:
                    param_nodes.append(child)

        port_nodes = []
        processed_node_ids = set() # To avoid duplicates from ANSI/non-ANSI overlap

        # --- Debug: List direct children of module_declaration node ---
        # logger.debug(f"Direct children of module_declaration ({name}):") # Commented out
        # for child in node.children:
        #     logger.debug(f"  - Child type: {child.type}, Text: {child.text.decode()[:50]}...") # Commented out
        # --- End Debug ---

        # --- ANSI Port Extraction (Revised - Direct Recursive Search) ---
        module_header = self._find_child(node, "module_ansi_header", recursive=False)
        if module_header:
            # logger.debug("Found module_ansi_header node. Structure:") # Commented out
            # self._debug_node(module_header, prefix="  MH> ") # Commented out

            # logger.debug("Searching for ANSI port declarations recursively within module_ansi_header...") # Commented out
            ansi_port_types = ["port_declaration", "ansi_port_declaration", "net_port_header", "variable_port_header"]
            found_ansi_ports = self._find_nodes_recursively(module_header, ansi_port_types)
            # logger.debug(f"  Found {len(found_ansi_ports)} potential ANSI port nodes via recursive search.") # Commented out
            for port_node in found_ansi_ports:
                if port_node.id not in processed_node_ids:
                    is_container = False
                    for target_type in ansi_port_types:
                        if port_node.type != target_type and self._find_child(port_node, target_type, recursive=True):
                            # logger.debug(f"  Skipping likely container node: {port_node.type} - {port_node.text.decode()[:30]}...") # Commented out
                            is_container = True
                            break
                    if not is_container:
                        # logger.debug(f"  Appending ANSI port node: {port_node.type} - {port_node.text.decode()[:30]}...") # Commented out
                        port_nodes.append(port_node)
                        processed_node_ids.add(port_node.id)
        else:
            logger.debug("No module_ansi_header found, looking for non-ANSI ports.")

        # --- Non-ANSI Port Extraction (Fallback/Addition) ---
        logger.debug("Checking for non-ANSI port declarations in module body...")
        # Iterate through direct children of the module_declaration node
        for child in node.children:
             if child.type == "module_item":
                 # Check for port_declaration directly within module_item
                 port_decl_node = self._find_child(child, "port_declaration", recursive=False)
                 if port_decl_node and port_decl_node.id not in processed_node_ids:
                     logger.debug(f"  Appending non-ANSI port node (from module_item/port_declaration): {port_decl_node.text.decode()[:30]}...")
                     port_nodes.append(port_decl_node)
                     processed_node_ids.add(port_decl_node.id)
                     continue # Found port_declaration, move to next module_item

                 # Check for net_declaration within module_item (common for non-ANSI)
                 net_decl_node = self._find_child(child, "net_declaration", recursive=False)
                 if net_decl_node and net_decl_node.id not in processed_node_ids:
                     # Check if this net_declaration likely represents a port (e.g., contains input/output keyword)
                     # Use _has_text for potentially better checking within the node
                     if self._has_text(net_decl_node, "input") or self._has_text(net_decl_node, "output") or self._has_text(net_decl_node, "inout"):
                         logger.debug(f"  Appending non-ANSI port node (from module_item/net_declaration): {net_decl_node.text.decode()[:30]}...")
                         port_nodes.append(net_decl_node)
                         processed_node_ids.add(net_decl_node.id)

        logger.debug(f"Finished extracting header. Found {len(param_nodes)} param nodes, {len(port_nodes)} port nodes.")
        return name, param_nodes, port_nodes

    def _find_child(self, node: Node, type_names: str | List[str], recursive: bool = True) -> Optional[Node]:
        """Find first child node matching any of the given types.

        Args:
            node: The starting node.
            type_names: A string or list of strings representing the target node types.
            recursive: If True, performs a recursive BFS search. If False, only checks direct children.

        Returns:
            The first matching node found, or None.
        """
        if node is None: return None
        if isinstance(type_names, str): type_names = [type_names]

        if not recursive:
            # Non-recursive: Check only direct children
            for child in node.children:
                if child.type in type_names:
                    return child
            return None # Not found among direct children
        else:
            # Recursive: Perform Breadth-First Search (BFS)
            queue = collections.deque(node.children) # Start queue with direct children
            visited = {child.id for child in node.children} # Track visited nodes

            while queue:
                current_node = queue.popleft()

                # Check if the current node matches
                if current_node.type in type_names:
                    return current_node # Found the first match

                # Add its children to the queue if not visited
                for child in current_node.children:
                    if child.id not in visited:
                        visited.add(child.id)
                        queue.append(child)

            return None # Not found after traversing the subtree

    def _has_text(self, node: Node, text: str) -> bool:
        """Check if node or any child contains the exact text token."""
        if node is None: return False

        queue = [node]
        visited = {node.id}

        while queue:
            current = queue.pop(0)
            # Check if the text exists as a distinct token in the node's direct text
            # This avoids matching substrings within identifiers
            node_text_content = current.text.decode('utf8')
            # Simple split might be okay, but regex could be more robust
            # Using split for now, assuming space separation or common delimiters
            tokens = re.split(r'\\s+|(?=[;,()[\\]])|(?<=[;,()[\\]])', node_text_content)
            if text in tokens:
                 # Check if it's a direct child's type or specific node text
                 is_direct_match = False
                 if current.type == text: # e.g., node type is 'input'
                     is_direct_match = True
                 elif any(child.type == text for child in current.children): # e.g., child node type is 'input'
                      is_direct_match = True
                 # Add more specific checks if needed based on grammar
                 if is_direct_match:
                     return True


            for child in current.children:
                if child.id not in visited:
                    visited.add(child.id)
                    queue.append(child)
        return False

    # ... rest of RTLParser class ...