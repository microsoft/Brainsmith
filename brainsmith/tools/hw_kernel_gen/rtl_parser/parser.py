############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""SystemVerilog RTL parser implementation.

This module implements the main RTL parser using tree-sitter to parse
SystemVerilog files and extract module interfaces, parameters, and pragmas.
"""

import collections
import logging
from typing import Optional, List, Tuple, Dict

from tree_sitter import Parser, Node

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import HWKernel, Port, Parameter, Direction
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import InterfaceType, Interface
from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import extract_pragmas, PragmaType, Pragma
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder
from . import grammar

# Configure logger
logger = logging.getLogger(__name__)

# --- Error Classes (Keep them here for now) ---
class ParserError(Exception):
    """Base class for parser errors."""
    pass

class SyntaxError(ParserError):
    """Raised when SystemVerilog syntax is invalid."""
    pass
# --- End Error Classes ---

# --- Main RTLParser Class ---
class RTLParser:
    """Parser for SystemVerilog RTL files. (Original Implementation)

    This class uses tree-sitter to parse SystemVerilog files and extract
    the information needed by the Hardware Kernel Generator.

    Attributes:
        parser: tree-sitter Parser instance
        debug: Enable debug output
    """
    def __init__(self, grammar_path: Optional[str] = None, debug: bool = False):
        """Initializes the RTLParser.

        Loads the tree-sitter SystemVerilog grammar and initializes the parser
        and the InterfaceBuilder.

        Args:
            grammar_path: Optional path to the compiled tree-sitter grammar library.
                          If None, uses the default path configured in grammar.py.
            debug: If True, enables detailed debug logging.

        Raises:
            FileNotFoundError: If the grammar library cannot be found or loaded.
            RuntimeError: For other unexpected errors during grammar loading.
        """
        self.debug = debug
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

        try:
            language = grammar.load_language(grammar_path)
            self.parser = Parser(language)
            logger.info("SystemVerilog grammar loaded successfully.")
        except (FileNotFoundError, AttributeError, RuntimeError) as e:
            logger.error(f"Failed to load SystemVerilog grammar: {e}")
            raise FileNotFoundError(f"Failed to load SystemVerilog grammar: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred during grammar loading: {e}")
            raise RuntimeError(f"Unexpected error loading grammar: {e}")

        self.interface_builder = InterfaceBuilder(debug=self.debug)

        # Initialize state variables
        self.tree: Optional[Node] = None
        self.pragmas: List[Pragma] = []
        self.module_node: Optional[Node] = None

    def _initial_parse(self, file_path: str) -> None:
        """Performs Stage 1 of parsing: Initial AST generation and module selection.

        Reads the source file, parses it into an Abstract Syntax Tree (AST) using
        tree-sitter, checks for basic syntax errors, finds all module definitions,
        extracts `@brainsmith` pragmas, and selects the target module node based on
        the number of modules found and the presence of a `TOP_MODULE` pragma.

        Args:
            file_path: The absolute path to the SystemVerilog file to parse.

        Returns:
            None. Results are stored in instance variables:
            - `self.tree`: The root node of the parsed AST.
            - `self.pragmas`: A list of extracted Pragma objects.
            - `self.module_node`: The tree-sitter Node representing the selected target module.

        Raises:
            ParserError: If the file cannot be read, core parsing fails, no modules are found,
                         pragma extraction fails, or module selection logic fails (e.g., ambiguity).
            SyntaxError: If the input file contains SystemVerilog syntax errors detected by tree-sitter.
            FileNotFoundError: (Propagated) If the input file does not exist.
        """
        logger.info(f"Stage 1: Initial parsing for {file_path}")
        self.tree = None # Reset state
        self.pragmas = []
        self.module_node = None

        # 1. Read file
        try:
            with open(file_path, 'r') as f:
                source = f.read()
        except Exception as e:
            logger.exception(f"Failed to read file {file_path}: {e}")
            raise ParserError(f"Failed to read file {file_path}: {e}")
        
        # 2. Parse using self.parser
        try:
            self.tree = self.parser.parse(bytes(source, 'utf8'))
        except Exception as e:
            logger.exception(f"Tree-sitter parsing failed for {file_path}: {e}")
            raise ParserError(f"Core parsing failed for {file_path}: {e}")

        # 3. Check syntax
        if self.tree.root_node.has_error:
            error_node = self._find_first_error_node(self.tree.root_node)
            line = error_node.start_point[0] + 1 if error_node else 'unknown'
            col = error_node.start_point[1] + 1 if error_node else 'unknown'
            error_msg = f"Invalid SystemVerilog syntax near line {line}, column {col}."
            logger.error(f"Syntax error in {file_path} near line {line}:{col}")
            raise SyntaxError(error_msg)

        # 4. Find module nodes
        module_nodes = self._find_module_nodes(self.tree.root_node)
        if not module_nodes:
            logger.error(f"No module definitions found in {file_path}")
            raise ParserError(f"No module definition found in {file_path}")        

        # 5. Extract pragmas
        logger.debug("Extracting pragmas...")
        try:
            self.pragmas = extract_pragmas(self.tree.root_node)
        except Exception as e:
            logger.exception(f"Error during pragma extraction in {file_path}: {e}")
            raise ParserError(f"Failed during pragma extraction: {e}")
        logger.debug(f"Found {len(self.pragmas)} potential pragmas.")

        # 6. Select target module
        try:
            self.module_node = self._select_target_module(module_nodes, self.pragmas, file_path)
            logger.info(f"Selected target module node: {self.module_node.type}") # Log basic info
        except ParserError as e:
            logger.error(e) # Log the specific error from selection logic
            raise # Re-raise the selection error

    def _extract_kernel_components(self) -> Tuple[str, List[Parameter], List[Port]]:
        """Performs Stage 2 of parsing: Extraction of name, parameters, and ports.

        Processes the `self.module_node` (selected in Stage 1) to extract the
        module's name, its parameters (excluding localparams), and its ports
        (currently supporting ANSI-style declarations).

        Requires `_initial_parse` to have been run successfully first.

        Args:
            None. Operates on `self.module_node`.

        Returns:
            A tuple containing:
            - `name` (str): The name of the parsed hardware kernel module.
            - `parameters` (List[Parameter]): A list of extracted Parameter objects.
            - `ports` (List[Port]): A list of extracted Port objects.

        Raises:
            ParserError: If `self.module_node` is not set (Stage 1 was not run or failed),
                         or if extraction of the header, parameters, or ports fails.
        """
        if not self.module_node:
            raise ParserError("Cannot extract components: _initial_parse must be run first.")
        logger.info("Stage 2: Extracting kernel components (name, parameters, ports)")

        # 1. Extract header (name, param_nodes, port_nodes)
        try:
            name, param_nodes, port_nodes = self._extract_module_header(self.module_node)
            if name is None:
                raise ParserError("Failed to extract module name from header.")
            logger.debug(f"Extracted header for module '{name}'")
        except Exception as e:
            logger.exception(f"Error during module header extraction: {e}")
            raise ParserError(f"Failed during module header extraction: {e}")

        # 2. Parse parameters
        parameters: List[Parameter] = []
        logger.debug("Extracting parameters...")
        try:
            for node in param_nodes:
                param = self._parse_parameter_declaration(node)
                if param is not None: # Skips local params implicitly
                    parameters.append(param)
            logger.debug(f"Extracted {len(parameters)} parameters.")
        except Exception as e:
            logger.exception(f"Error during parameter parsing: {e}")
            raise ParserError(f"Failed during parameter parsing: {e}")

        # 3. Parse ports
        ports: List[Port] = []
        logger.debug("Extracting ports...")
        try:
            for node in port_nodes:
                parsed_port_list = self._parse_port_declaration(node) # Returns List[Port]
                if parsed_port_list: # Check if the list is not empty
                    ports.extend(parsed_port_list) # Use extend to add elements
            logger.debug(f"Extracted {len(ports)} individual port objects.")
        except Exception as e:
            logger.exception(f"Error during port parsing: {e}")
            raise ParserError(f"Failed during port parsing: {e}")

        # 4. Return name, parameters, ports
        logger.info("Stage 2: Component extraction complete.")
        return name, parameters, ports

    def _analyze_and_validate_interfaces(self, ports: List[Port], kernel_name: str) -> Dict[str, Interface]:
        """Performs Stage 3 of parsing: Interface building and validation.

        Takes the list of raw ports extracted in Stage 2 and uses the `InterfaceBuilder`
        to group them into logical interfaces (AXI-Stream, AXI-Lite, Global Control).
        It then performs critical validation checks:
        1. Ensures a Global Control interface (`ap_clk`, `ap_rst_n`) exists.
        2. Ensures at least one AXI-Stream interface exists.
        3. Ensures no ports were left unassigned to a standard interface.

        Args:
            ports: The list of Port objects extracted in Stage 2.
            kernel_name: The name of the kernel module (used for error messages).

        Returns:
            A dictionary mapping interface names (str) to validated Interface objects.

        Raises:
            ParserError: If the interface building process fails, or if any of the
                         post-analysis validation checks (Global Control, AXI-Stream,
                         Unassigned Ports) fail.
        """
        logger.info(f"Stage 3: Analyzing and validating interfaces for module {kernel_name}")

        # 1. Call self.interface_builder.build_interfaces(ports)
        try:
            validated_interfaces, unassigned_ports = self.interface_builder.build_interfaces(ports)
            logger.info(f"Interface analysis complete. Found {len(validated_interfaces)} valid interfaces.")
        except Exception as e:
            logger.exception(f"Error during interface building for module {kernel_name}: {e}")
            # Re-raise as ParserError to be consistent? Or let specific error propagate?
            # Let's wrap it for now. # TAFK TODO
            raise ParserError(f"Failed during interface building: {e}")

        # --- Post-Analysis Validation ---

        # 2. Perform Global Control check
        has_global_control = any(
            iface.type == InterfaceType.GLOBAL_CONTROL for iface in validated_interfaces.values()
        )
        if not has_global_control:
            error_msg = f"Module '{kernel_name}' is missing a valid Global Control interface (ap_clk, ap_rst_n)."
            logger.error(error_msg)
            raise ParserError(error_msg)

        # 3. Perform AXI-Stream check
        has_axi_stream = any(
            iface.type == InterfaceType.AXI_STREAM for iface in validated_interfaces.values()
        )
        if not has_axi_stream:
            error_msg = f"Module '{kernel_name}' requires at least one AXI-Stream interface, but found none after analysis."
            logger.error(error_msg)
            raise ParserError(error_msg)

        # 4. Perform Unassigned Ports check
        if unassigned_ports:
             unassigned_names = [p.name for p in unassigned_ports]
             error_msg = f"Module '{kernel_name}' has {len(unassigned_ports)} ports not assigned to any standard interface: {unassigned_names}"
             logger.error(error_msg)
             raise ParserError(error_msg)

        # 5. Return validated_interfaces
        logger.info("Stage 3: Interface analysis and validation complete.")
        return validated_interfaces

    def parse_file(self, file_path: str) -> HWKernel:
        """Orchestrates the multi-stage parsing process for a SystemVerilog file.

        This is the main public method to parse an RTL file. It calls the
        internal stage methods in sequence:
        1. `_initial_parse`: Reads file, parses AST, selects module.
        2. `_extract_kernel_components`: Extracts name, parameters, ports.
        3. `_analyze_and_validate_interfaces`: Builds and validates interfaces.
        Finally, it constructs and returns the `HWKernel` data object.

        Args:
            file_path: The absolute path to the SystemVerilog file to parse.

        Returns:
            An `HWKernel` object containing the parsed information (name, parameters,
            interfaces, pragmas).

        Raises:
            ParserError: If any stage of the parsing process fails due to logical errors,
                         ambiguity, or validation failures.
            SyntaxError: If the input file has SystemVerilog syntax errors.
            FileNotFoundError: If the input file cannot be found.
            Exception: Catches and wraps any other unexpected errors during orchestration
                       as a `ParserError`.
        """
        logger.info(f"Starting full parsing orchestration for: {file_path}")
        try:
            # 1. Call Stage 1: Initial Parse
            self._initial_parse(file_path)

            # 2. Call Stage 2: Extract Components
            name, parameters, ports = self._extract_kernel_components()

            # 3. Call Stage 3: Analyze and Validate Interfaces
            validated_interfaces = self._analyze_and_validate_interfaces(ports, name)

            # 4. Create HWKernel object
            # Ensure pragmas are available from self.pragmas (set in _initial_parse)
            kernel = HWKernel(
                name=name,
                parameters=parameters,
                interfaces=validated_interfaces,
                pragmas=self.pragmas
                # Decide if raw ports should still be stored on HWKernel.
                # If yes, add 'ports=ports' here. Currently, HWKernel doesn't store raw ports.
            )

            # 5. Return HWKernel
            logger.info(f"Successfully parsed module '{kernel.name}' from {file_path}")
            return kernel

        except (SyntaxError, ParserError) as e:
            # Log specific parser/syntax errors raised by stages
            # Error should already be logged by the stage method that raised it.
            logger.error(f"Parsing failed for {file_path}: {e}")
            raise # Re-raise the specific error
        except FileNotFoundError as e:
            # Handle file not found specifically if not caught earlier
            logger.error(f"File not found during parsing: {e}")
            raise
        except Exception as e:
            # Catch any other unexpected errors during orchestration
            logger.exception(f"An unexpected error occurred during parsing orchestration for {file_path}: {e}")
            # Wrap in ParserError for consistent error type from this function
            raise ParserError(f"An unexpected error occurred during parsing orchestration: {e}")

    # --- Helper Functions ---
    def _find_first_error_node(self, node: Node) -> Optional[Node]:
        """Finds the first AST node marked with an error using BFS."""
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
        """Finds all top-level 'module_declaration' nodes in the AST."""
        module_nodes = []
        queue = collections.deque([root])
        while queue:
            node = queue.popleft()
            if node.type == "module_declaration":
                module_nodes.append(node)
            # Avoid descending into nested modules if grammar supports them
            if node != root and node.type == "module_declaration":
                continue
            queue.extend(node.children)
        return module_nodes

    def _select_target_module(self, module_nodes: List[Node], pragmas: List["Pragma"], file_path: str) -> Node:
        """Selects the target module node based on count and TOP_MODULE pragma."""
        top_module_pragmas = [p for p in pragmas if p.type == PragmaType.TOP_MODULE]

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

    def _extract_module_header(self, module_node: Node) -> Tuple[Optional[str], Optional[List[Node]], Optional[List[Node]]]:
        """Extracts name, parameter nodes, and port nodes from a module_declaration node."""
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

    def _debug_node(self, node: Node, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> None:
        """Debug helper to print AST node structure recursively with a depth limit."""
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
        """Extracts the port direction (input, output, inout) from relevant AST nodes."""
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
        """Recursively finds all 'simple_identifier' or 'port_identifier' texts under a node, excluding keywords."""
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
        """Parses an 'ansi_port_declaration' node into a list of Port objects (one per identifier)."""
        logger.debug(f"Parsing port declaration node: {node.text.decode()}")

        final_width = "1" # Default
        data_type = "logic" # Default
        direction = Direction.INPUT # Default

        # --- Try finding header types ---
        variable_port_header = self._find_child(node, ["variable_port_header"])
        net_port_header = self._find_child(node, ["net_port_header"])
        interface_port_header = self._find_child(node, ["interface_port_header"])

        width_node = None # Initialize width_node

        if variable_port_header:
            logger.debug("Parsing as Variable Port Header")
            direction = self._extract_direction(self._find_child(variable_port_header, ["port_direction"]))
            variable_port_type = self._find_child(variable_port_header, ["variable_port_type"])
            if variable_port_type:
                dt_node = self._find_child(variable_port_type, ["data_type"])
                if dt_node:
                    data_type = dt_node.text.decode('utf8').strip()
                    # Search for width as sibling or child of data_type first
                    width_node = self._find_child(dt_node, ["packed_dimension", "unpacked_dimension"]) # Check child
                    if not width_node: # Check siblings
                         sibling = dt_node.next_sibling
                         if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]: width_node = sibling
                         else:
                              sibling = dt_node.prev_sibling
                              if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]: width_node = sibling
                # Fallback: Search directly within variable_port_type if not found near data_type
                if not width_node:
                    width_node = self._find_child(variable_port_type, ["packed_dimension", "unpacked_dimension"])

        elif net_port_header:
            logger.debug("Parsing as Net Port Header")
            direction = self._extract_direction(self._find_child(net_port_header, ["port_direction"]))
            net_port_type = self._find_child(net_port_header, ["net_port_type"])
            if net_port_type:
                # Data Type: Can be net_type or within data_type_or_implicit
                nt_node = self._find_child(net_port_type, ["net_type"])
                if nt_node: data_type = nt_node.text.decode('utf8').strip()

                dtoi_node = self._find_child(net_port_type, ["data_type_or_implicit"])
                if dtoi_node:
                    # If data_type exists here, it might override net_type
                    dt_node = self._find_child(dtoi_node, ["data_type"])
                    if dt_node: data_type = dt_node.text.decode('utf8').strip()

                    # Width is usually in implicit_data_type or sibling/child of data_type
                    idt_node = self._find_child(dtoi_node, ["implicit_data_type"])
                    if idt_node:
                        width_node = self._find_child(idt_node, ["packed_dimension", "unpacked_dimension"])
                    if not width_node and dt_node: # Check near data_type if present
                         width_node = self._find_child(dt_node, ["packed_dimension", "unpacked_dimension"]) # Check child
                         if not width_node: # Check siblings
                              sibling = dt_node.next_sibling
                              if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]: width_node = sibling
                              else:
                                   sibling = dt_node.prev_sibling
                                   if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]: width_node = sibling
                # Fallback: Search directly within net_port_type
                if not width_node:
                     width_node = self._find_child(net_port_type, ["packed_dimension", "unpacked_dimension"])

            elif self._find_child(net_port_header, ["port_direction"]): # Handle implicit type like "input enable;"
                 data_type = "wire"
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

        else: # Non-ANSI -> Raise Error
             port_text_preview = node.text.decode('utf8').strip().split('\n')[0][:80] # Get first line preview
             error_msg = (
                f"Port declaration '{port_text_preview}...' appears to be non-ANSI style "
                f"(e.g., missing type/width in header). Only ANSI-style port declarations are supported."
            )
             logger.error(error_msg)
             raise ParserError(error_msg)
             # --- REMOVED Fallback Logic ---

        # --- Process Width Node ---
        if width_node and not interface_port_header:
            logger.debug(f"Found potential width node: Type={width_node.type}, Text='{width_node.text.decode()}'")
            extracted = self._extract_width_from_dimension(width_node)
            if extracted: final_width = extracted
            else: logger.warning(f"Width extraction returned empty for node: {width_node.text.decode()}, keeping default '1'.")
        elif not interface_port_header: # Only log if not an interface
             logger.debug(f"No width node found. Final width: {final_width}")


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
        keywords_to_exclude_set = set([d.value for d in Direction])

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
        """Extracts the width string (e.g., '31:0', 'WIDTH-1:0') from a dimension node."""
        if not width_node: return "1"
        logger.debug(f"Extracting width from node: Type={width_node.type}, Text='{width_node.text.decode()}'")

        # Prioritize finding the range or expression node within the dimension
        expr_node = self._find_child(width_node, ["constant_range", "range_expression", "constant_expression", "expression", "primary_literal", "number"])

        if expr_node:
            logger.debug(f"Found expression node: Type={expr_node.type}, Text='{expr_node.text.decode()}'")
            width_text = expr_node.text.decode('utf8').strip()
            logger.debug(f"Width expression text found: '{width_text}'")
            # Check if the found expression is the full content between brackets
            full_node_text = width_node.text.decode('utf8').strip()
            if full_node_text.startswith('[') and full_node_text.endswith(']'):
                expected_inner_text = full_node_text[1:-1].strip()
                logger.debug(f"Full node inner text: '{expected_inner_text}'")
                if width_text == expected_inner_text:
                    logger.debug("Expression node text matches full inner text.")
                    return width_text # Perfect match
                else:
                    # Sometimes the expr_node might be nested deeper, use the full inner text
                    logger.debug(f"Expression node text ('{width_text}') differs from node inner text ('{expected_inner_text}'), using inner text.")
                    return expected_inner_text if expected_inner_text else "1"
            else:
                # If original node wasn't bracketed (less common), use expr_node text
                logger.debug("Original width node not bracketed, using expression node text.")
                return width_text
        else:
            logger.debug("No specific expression node found within width_node.")
            # Fallback: Use cleaned text of the dimension node itself, removing brackets
            cleaned_width_text = width_node.text.decode('utf8').strip()
            if cleaned_width_text.startswith('[') and cleaned_width_text.endswith(']'):
                cleaned_width_text = cleaned_width_text[1:-1].strip()
            logger.debug(f"Using fallback cleaned text: '{cleaned_width_text}'")
            return cleaned_width_text if cleaned_width_text else "1" # Return cleaned text or default

    def _find_child(self, node: Node, types: List[str]) -> Optional[Node]:
        """Finds the first direct child node matching any of the given types."""
        if not node: return None
        for child in node.children:
            if child.type in types:
                return child
        return None

    def _find_children(self, node: Node, types: List[str]) -> List[Node]:
        """Finds all direct child nodes matching any of the given types."""
        found_nodes = []
        if not node: return found_nodes
        for child in node.children:
            if child.type in types:
                found_nodes.append(child)
        return found_nodes

    def _parse_parameter_declaration(self, node: Node) -> Optional[Parameter]:
        """Parses a parameter declaration node into a Parameter object, skipping localparams."""
        param_name: Optional[str] = None
        param_type: str = "parameter" # Default type if not specified
        default_value: Optional[str] = None
 
        # Check if the node itself is local_parameter_declaration or contains it
        param_decl_node = self._find_child(node, ["parameter_declaration", "local_parameter_declaration"])
        if not param_decl_node:
             # If node is directly local_parameter_declaration (passed from body scan)
             if node.type == "local_parameter_declaration":
                 param_decl_node = node
             else:
                 logger.warning(f"Could not find parameter_declaration or local_parameter_declaration within: {node.text.decode()}")
                 # Try finding assignment directly under parameter_port_declaration as fallback
                 param_decl_node = node # Use the original node if specific decl not found

        # Determine if localparam and skip if true
        is_local = param_decl_node.type == "local_parameter_declaration"
        if is_local:
            logger.debug(f"Skipping local parameter: {param_decl_node.text.decode()[:50]}...")
            return None

        logger.debug(f"--- Entering _parse_parameter_declaration for node: {param_decl_node.type} | Text: '{param_decl_node.text.decode()[:60]}...'")

        # --- Extract Type ---
        param_type = None
        logger.debug("--- Starting type extraction ---")
        # Look for explicit type declaration first
        type_node = self._find_child(param_decl_node, ["data_type_or_implicit", "data_type"])
        logger.debug(f"Found type_node: {type_node.type if type_node else 'None'}")
        if type_node:
            # Previously might have only taken a sub-node's text
            param_type = type_node.text.decode('utf8').strip()
            # Special case: if the node is data_type_or_implicit and contains 'type', it's a type parameter
            if type_node.type == "data_type_or_implicit":
                 type_keyword_node = self._find_child(type_node, ["type"])
                 if type_keyword_node:
                      param_type = "type" # Override if 'type' keyword is present
            logger.debug(f"Explicit type found: '{param_type}'")
        else:
             # No explicit type node found, check for 'parameter type T' structure
             logger.debug("No explicit type_node found. Checking for type_parameter_declaration...")
             type_param_decl = self._find_child(param_decl_node, ["type_parameter_declaration"])
             logger.debug(f"Found type_param_decl: {type_param_decl.type if type_param_decl else 'None'}")
             if type_param_decl:
                  param_type = "type"
                  logger.debug("Found type_parameter_declaration, setting param_type='type'")
             else:
                  logger.debug("No type_parameter_declaration found, assuming implicit type.")
                  param_type = None # Default for implicit

        logger.debug(f"--- Type extraction complete. Final param_type: {param_type}")

        # --- Extract Name and Default Value ---
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
                    inner_expr = self._find_child(value_expr_node, ["constant_min_type_max_expression", "constant_expression", "primary_literal", "binary_expression"])
                    if inner_expr:
                         default_value = inner_expr.text.decode('utf8').strip()
                    else:
                         default_value = value_expr_node.text.decode('utf8').strip() # Fallback
                    logger.debug(f"Parameter '{param_name}' default value found: {default_value}")
            else:
                 logger.warning(f"Could not find param_assignment within list: {assignment_list_node.text.decode()}")
                 return None # Cannot get name/value without assignment
        else:
            logger.debug(f"No list_of_param_assignments found in: {param_decl_node.text.decode()[:50]}...")
            # Check if this is a 'parameter type' declaration
            if param_type == "type":
                logger.debug(f"Handling 'parameter type' specific structure: {param_decl_node.text.decode()[:50]}...")
                type_param_decl_node = self._find_child(param_decl_node, ["type_parameter_declaration"])
                if type_param_decl_node:
                    list_of_assignments = self._find_child(type_param_decl_node, ["list_of_type_assignments"])
                    if list_of_assignments:
                        assignment_node = self._find_child(list_of_assignments, ["type_assignment"])
                        if assignment_node:
                            # Extract name
                            name_node = self._find_child(assignment_node, ["simple_identifier", "identifier"])
                            if name_node:
                                param_name = name_node.text.decode('utf8').strip()
                            else:
                                logger.warning(f"Could not find parameter name in type_assignment: {assignment_node.text.decode()}")
                                return None
                            # Extract default value (assigned type)
                            value_node = self._find_child(assignment_node, ["data_type"])
                            if value_node:
                                default_value = value_node.text.decode('utf8').strip()
                                logger.debug(f"Type Parameter '{param_name}' default type found: {default_value}")
                            else:
                                logger.warning(f"Could not find default type (data_type) for type parameter '{param_name}'")
                                # Keep param_name, default_value remains None (or handle as error?)
                        else:
                            logger.warning(f"Could not find type_assignment within list: {list_of_assignments.text.decode()}")
                            return None
                    else:
                        logger.warning(f"Could not find list_of_type_assignments within type_parameter_declaration: {type_param_decl_node.text.decode()}")
                        return None
                else:
                    logger.warning(f"param_type is 'type' but could not find type_parameter_declaration node within: {param_decl_node.text.decode()}")
                    return None
            else:
                # Original fallback: Declaration without assignment? Try finding name directly
                # This case might be hit for implicit types if type extraction failed earlier
                name_node = self._find_child(param_decl_node, ["simple_identifier", "identifier"])
                if name_node:
                    param_name = name_node.text.decode('utf8').strip()
                    logger.debug(f"Found parameter '{param_name}' without assignment list (or type extraction failed).")
                    # For implicit types, param_type should be None here
                    if param_type is not None:
                         logger.warning(f"Parameter '{param_name}' has type '{param_type}' but no assignment list found?")
                else:
                    logger.warning(f"Could not determine parameter name: {param_decl_node.text.decode()}")
                    return None

        # --- Create and Return Parameter --- 
        if param_name:
            # Ensure param_type is set correctly (might be None for implicit)
            final_param_type = param_type if param_type else None # Explicitly use None if not found
            logger.info(f"Successfully parsed parameter: Name='{param_name}', Type='{final_param_type}', Default='{default_value}'")
            return Parameter(name=param_name, param_type=final_param_type, default_value=default_value)
        else:
            # This path should ideally not be reached if logic above is correct
            logger.error(f"Failed to extract parameter details from node: {param_decl_node.text.decode()}")
            return None
