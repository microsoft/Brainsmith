############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Module extraction and selection utilities for SystemVerilog RTL parser.

This module handles:
- Module selection based on pragmas or explicit targets
- Extraction of module components (name, parameters, ports) from AST nodes
"""

import logging
from typing import Optional, List, Tuple, Dict
from tree_sitter import Node

from .data import Port, Parameter, Direction, PragmaType
from .pragmas import Pragma
from .ast_parser import ASTParser

logger = logging.getLogger(__name__)


class ModuleExtractor:
    """Extracts and selects SystemVerilog modules from AST nodes.
    
    This class handles:
    - Module selection based on pragmas or explicit targets
    - Module name extraction
    - Parameter extraction (excluding localparams)
    - Port extraction with directions and widths
    """
    
    def __init__(self, ast_parser: ASTParser, debug: bool = False):
        """Initialize the component extractor.
        
        Args:
            ast_parser: ASTParser instance for node traversal utilities.
            debug: Enable debug logging.
        """
        self.ast_parser = ast_parser
        self.debug = debug
    
    def extract_module_header(self, module_node: Node) -> Tuple[Optional[str], Optional[List[Node]], Optional[List[Node]]]:
        """Extract name, parameter nodes, and port nodes from a module_declaration node.
        
        Args:
            module_node: The module_declaration node to process.
            
        Returns:
            Tuple of (module_name, parameter_nodes, port_nodes).
            Any element may be None if not found.
        """
        if not module_node or module_node.type != "module_declaration":
            logger.error("Invalid node passed to extract_module_header. Expected 'module_declaration'.")
            return None, None, None
        
        module_name: Optional[str] = None
        param_nodes: Optional[List[Node]] = []
        port_nodes: Optional[List[Node]] = []
        
        # Find the header node first
        header_node = self.ast_parser.find_child(module_node, ["module_ansi_header", "module_nonansi_header"])
        
        # Determine the node to search for name, parameters, and ports
        search_parent_node = header_node if header_node else module_node
        logger.debug(f"Determined search parent node type: {search_parent_node.type}")
        
        # Find module identifier (name)
        if header_node:
            name_node = self.ast_parser.find_child(header_node, ["simple_identifier", "identifier"])
        else:
            name_node = self.ast_parser.find_child(module_node, ["simple_identifier", "identifier"])
        
        if name_node:
            module_name = name_node.text.decode('utf8')
            logger.debug(f"Extracted module name: {module_name}")
        else:
            logger.warning(f"Could not find module name identifier within node: {module_node.text.decode()[:50]}...")
        
        # Search for parameter and port lists
        logger.debug(f"Searching for parameter/port lists within node type: {search_parent_node.type}")
        
        if self.debug:
            logger.debug(f"--- Children of '{search_parent_node.type}' node (runtime) ---")
            for i, child in enumerate(search_parent_node.children):
                child_text = child.text.decode('utf8').strip().replace('\\n', '\\\\n')
                if len(child_text) > 60:
                    child_text = child_text[:57] + "..."
                logger.debug(f"  Child {i}: Type='{child.type}', Text='{child_text}'")
            logger.debug(f"--- End Children of '{search_parent_node.type}' ---")
        
        # Find parameter list node
        param_list_node = self.ast_parser.find_child(search_parent_node, ["parameter_port_list"])
        if param_list_node:
            param_nodes = self.ast_parser.find_children(param_list_node, ["parameter_port_declaration"])
            logger.debug(f"Found parameter list node containing {len(param_nodes)} declarations.")
        else:
            logger.debug("No parameter list node found.")
        
        # Find port list node (ANSI style)
        port_list_node = self.ast_parser.find_child(search_parent_node, ["list_of_port_declarations"])
        if port_list_node:
            port_nodes = self.ast_parser.find_children(port_list_node, ["ansi_port_declaration"])
            logger.debug(f"Found ANSI port list node containing {len(port_nodes)} declarations.")
        else:
            logger.debug("No ANSI port list node found. Non-ANSI port extraction not yet implemented.")
        
        return module_name, param_nodes, port_nodes
    
    def extract_module_name(self, module_node: Node) -> Optional[str]:
        """Extract just the module name from a module_declaration node.
        
        Args:
            module_node: The module_declaration node.
            
        Returns:
            Module name or None if not found.
        """
        name, _, _ = self.extract_module_header(module_node)
        return name
    
    def extract_parameters(self, module_node: Node) -> List[Parameter]:
        """Extract all parameters from a module_declaration node.
        
        Args:
            module_node: The module_declaration node.
            
        Returns:
            List of Parameter objects.
        """
        _, param_nodes, _ = self.extract_module_header(module_node)
        
        if not param_nodes:
            return []
        
        parameters = []
        for node in param_nodes:
            param = self.parse_parameter_declaration(node)
            if param is not None:  # Skips local params
                parameters.append(param)
        
        logger.debug(f"Extracted {len(parameters)} parameters.")
        return parameters
    
    def extract_ports(self, module_node: Node) -> List[Port]:
        """Extract all ports from a module_declaration node.
        
        Args:
            module_node: The module_declaration node.
            
        Returns:
            List of Port objects.
        """
        _, _, port_nodes = self.extract_module_header(module_node)
        
        if not port_nodes:
            return []
        
        ports = []
        for node in port_nodes:
            parsed_port_list = self.parse_port_declaration(node)
            if parsed_port_list:
                ports.extend(parsed_port_list)
        
        logger.debug(f"Successfully parsed {len(ports)} individual port objects.")
        return ports
    
    def parse_parameter_declaration(self, node: Node) -> Optional[Parameter]:
        """Parse a parameter declaration node into a Parameter object.
        
        Skips localparam declarations.
        
        Args:
            node: The parameter declaration node.
            
        Returns:
            Parameter object or None if localparam or parsing fails.
        """
        param_name: Optional[str] = None
        param_type: str = "parameter"  # Default type
        default_value: Optional[str] = None
        
        # Check if the node itself is local_parameter_declaration
        param_decl_node = self.ast_parser.find_child(node, ["parameter_declaration", "local_parameter_declaration"])
        if not param_decl_node:
            if node.type == "local_parameter_declaration":
                param_decl_node = node
            else:
                logger.warning(f"Could not find parameter_declaration or local_parameter_declaration within: {node.text.decode()}")
                param_decl_node = node
        
        # Skip localparams
        is_local = param_decl_node.type == "local_parameter_declaration"
        if is_local:
            logger.debug(f"Skipping local parameter: {param_decl_node.text.decode()[:50]}...")
            return None
        
        logger.debug(f"--- Entering parse_parameter_declaration for node: {param_decl_node.type} | Text: '{param_decl_node.text.decode()[:60]}...'")
        
        # Extract Type
        param_type = None
        logger.debug("--- Starting type extraction ---")
        type_node = self.ast_parser.find_child(param_decl_node, ["data_type_or_implicit", "data_type"])
        logger.debug(f"Found type_node: {type_node.type if type_node else 'None'}")
        
        if type_node:
            param_type = type_node.text.decode('utf8').strip()
            # Special case for type parameters
            if type_node.type == "data_type_or_implicit":
                type_keyword_node = self.ast_parser.find_child(type_node, ["type"])
                if type_keyword_node:
                    param_type = "type"
            logger.debug(f"Explicit type found: '{param_type}'")
        else:
            # Check for type_parameter_declaration
            logger.debug("No explicit type_node found. Checking for type_parameter_declaration...")
            type_param_decl = self.ast_parser.find_child(param_decl_node, ["type_parameter_declaration"])
            logger.debug(f"Found type_param_decl: {type_param_decl.type if type_param_decl else 'None'}")
            if type_param_decl:
                param_type = "type"
                logger.debug("Found type_parameter_declaration, setting param_type='type'")
            else:
                logger.debug("No type_parameter_declaration found, assuming implicit type.")
                param_type = None
        
        logger.debug(f"--- Type extraction complete. Final param_type: {param_type}")
        
        # Extract Name and Default Value
        assignment_list_node = self.ast_parser.find_child(param_decl_node, ["list_of_param_assignments"])
        if assignment_list_node:
            assignment_node = self.ast_parser.find_child(assignment_list_node, ["param_assignment"])
            if assignment_node:
                # Extract name
                name_node = self.ast_parser.find_child(assignment_node, ["simple_identifier", "identifier"])
                if name_node:
                    param_name = name_node.text.decode('utf8').strip()
                else:
                    logger.warning(f"Could not find parameter name in assignment: {assignment_node.text.decode()}")
                    return None
                
                # Extract default value
                value_expr_node = self.ast_parser.find_child(assignment_node, ["constant_param_expression", "constant_expression", "expression"])
                if value_expr_node:
                    inner_expr = self.ast_parser.find_child(value_expr_node, ["constant_min_type_max_expression", "constant_expression", "primary_literal", "binary_expression"])
                    if inner_expr:
                        default_value = inner_expr.text.decode('utf8').strip()
                    else:
                        default_value = value_expr_node.text.decode('utf8').strip()
                    logger.debug(f"Parameter '{param_name}' default value found: {default_value}")
            else:
                logger.warning(f"Could not find param_assignment within list: {assignment_list_node.text.decode()}")
                return None
        else:
            logger.debug(f"No list_of_param_assignments found in: {param_decl_node.text.decode()[:50]}...")
            # Handle 'parameter type' declarations
            if param_type == "type":
                logger.debug(f"Handling 'parameter type' specific structure: {param_decl_node.text.decode()[:50]}...")
                type_param_decl_node = self.ast_parser.find_child(param_decl_node, ["type_parameter_declaration"])
                if type_param_decl_node:
                    list_of_assignments = self.ast_parser.find_child(type_param_decl_node, ["list_of_type_assignments"])
                    if list_of_assignments:
                        assignment_node = self.ast_parser.find_child(list_of_assignments, ["type_assignment"])
                        if assignment_node:
                            # Extract name
                            name_node = self.ast_parser.find_child(assignment_node, ["simple_identifier", "identifier"])
                            if name_node:
                                param_name = name_node.text.decode('utf8').strip()
                            else:
                                logger.warning(f"Could not find parameter name in type_assignment: {assignment_node.text.decode()}")
                                return None
                            # Extract default value
                            value_node = self.ast_parser.find_child(assignment_node, ["data_type"])
                            if value_node:
                                default_value = value_node.text.decode('utf8').strip()
                                logger.debug(f"Type Parameter '{param_name}' default type found: {default_value}")
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
                # Fallback: Try finding name directly
                name_node = self.ast_parser.find_child(param_decl_node, ["simple_identifier", "identifier"])
                if name_node:
                    param_name = name_node.text.decode('utf8').strip()
                    logger.debug(f"Found parameter '{param_name}' without assignment list.")
                    if param_type is not None:
                        logger.warning(f"Parameter '{param_name}' has type '{param_type}' but no assignment list found?")
                else:
                    logger.warning(f"Could not determine parameter name: {param_decl_node.text.decode()}")
                    return None
        
        # Create and return Parameter
        if param_name:
            final_param_type = param_type if param_type else None
            logger.info(f"Successfully parsed parameter: Name='{param_name}', Type='{final_param_type}', Default='{default_value}'")
            return Parameter(name=param_name, param_type=final_param_type, default_value=default_value)
        else:
            logger.error(f"Failed to extract parameter details from node: {param_decl_node.text.decode()}")
            return None
    
    def parse_port_declaration(self, node: Node) -> List[Port]:
        """Parse an 'ansi_port_declaration' node into a list of Port objects.
        
        Args:
            node: The port declaration node.
            
        Returns:
            List of Port objects (one per identifier in the declaration).
        """
        logger.debug(f"Parsing port declaration node: {node.text.decode()}")
        
        final_width = "1"  # Default
        data_type = "logic"  # Default
        direction = Direction.INPUT  # Default
        
        # Try finding header types
        variable_port_header = self.ast_parser.find_child(node, ["variable_port_header"])
        net_port_header = self.ast_parser.find_child(node, ["net_port_header"])
        interface_port_header = self.ast_parser.find_child(node, ["interface_port_header"])
        
        width_node = None
        
        if variable_port_header:
            logger.debug("Parsing as Variable Port Header")
            direction = self._extract_direction(self.ast_parser.find_child(variable_port_header, ["port_direction"]))
            variable_port_type = self.ast_parser.find_child(variable_port_header, ["variable_port_type"])
            if variable_port_type:
                dt_node = self.ast_parser.find_child(variable_port_type, ["data_type"])
                if dt_node:
                    data_type = dt_node.text.decode('utf8').strip()
                    # Search for width
                    width_node = self.ast_parser.find_child(dt_node, ["packed_dimension", "unpacked_dimension"])
                    if not width_node:
                        sibling = dt_node.next_sibling
                        if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]:
                            width_node = sibling
                        else:
                            sibling = dt_node.prev_sibling
                            if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]:
                                width_node = sibling
                if not width_node:
                    width_node = self.ast_parser.find_child(variable_port_type, ["packed_dimension", "unpacked_dimension"])
        
        elif net_port_header:
            logger.debug("Parsing as Net Port Header")
            direction = self._extract_direction(self.ast_parser.find_child(net_port_header, ["port_direction"]))
            net_port_type = self.ast_parser.find_child(net_port_header, ["net_port_type"])
            if net_port_type:
                # Data Type
                nt_node = self.ast_parser.find_child(net_port_type, ["net_type"])
                if nt_node:
                    data_type = nt_node.text.decode('utf8').strip()
                
                dtoi_node = self.ast_parser.find_child(net_port_type, ["data_type_or_implicit"])
                if dtoi_node:
                    dt_node = self.ast_parser.find_child(dtoi_node, ["data_type"])
                    if dt_node:
                        data_type = dt_node.text.decode('utf8').strip()
                    
                    # Width
                    idt_node = self.ast_parser.find_child(dtoi_node, ["implicit_data_type"])
                    if idt_node:
                        width_node = self.ast_parser.find_child(idt_node, ["packed_dimension", "unpacked_dimension"])
                    if not width_node and dt_node:
                        width_node = self.ast_parser.find_child(dt_node, ["packed_dimension", "unpacked_dimension"])
                        if not width_node:
                            sibling = dt_node.next_sibling
                            if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]:
                                width_node = sibling
                            else:
                                sibling = dt_node.prev_sibling
                                if sibling and sibling.type in ["packed_dimension", "unpacked_dimension"]:
                                    width_node = sibling
                if not width_node:
                    width_node = self.ast_parser.find_child(net_port_type, ["packed_dimension", "unpacked_dimension"])
            
            elif self.ast_parser.find_child(net_port_header, ["port_direction"]):
                data_type = "wire"
                logger.debug("Parsing as Implicit Net Port (defaulting type to wire)")
            else:
                logger.warning("No net_port_type or direction found within net_port_header")
        
        elif interface_port_header:
            logger.debug("Parsing as Interface Port Header")
            if_identifier_node = self.ast_parser.find_child(interface_port_header, ["interface_identifier"])
            if if_identifier_node:
                data_type = if_identifier_node.text.decode('utf8').strip()
                modport_node = self.ast_parser.find_child(interface_port_header, ["modport_identifier"])
                if modport_node:
                    data_type += "." + modport_node.text.decode('utf8').strip()
                logger.debug(f"Interface type extracted as: {data_type}")
            else:
                logger.warning("Could not find interface_identifier within interface_port_header")
            final_width = "1"
        
        else:
            # Non-ANSI style - raise error
            port_text_preview = node.text.decode('utf8').strip().split('\n')[0][:80]
            error_msg = (
                f"Port declaration '{port_text_preview}...' appears to be non-ANSI style "
                f"(e.g., missing type/width in header). Only ANSI-style port declarations are supported."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Process Width Node
        if width_node and not interface_port_header:
            logger.debug(f"Found potential width node: Type={width_node.type}, Text='{width_node.text.decode()}'")
            extracted = self._extract_width_from_dimension(width_node)
            if extracted:
                final_width = extracted
            else:
                logger.warning(f"Width extraction returned empty for node: {width_node.text.decode()}, keeping default '1'.")
        elif not interface_port_header:
            logger.debug(f"No width node found. Final width: {final_width}")
        
        # Extract Port Name(s)
        list_of_ids_node = self.ast_parser.find_child(node, ["list_of_port_identifiers", "list_of_variable_identifiers"])
        if list_of_ids_node:
            potential_names = self._find_identifiers_recursive(list_of_ids_node)
        else:
            # Find last identifier sibling
            last_identifier = None
            for child in reversed(node.children):
                if child.type == "simple_identifier":
                    last_identifier = child
                    break
                # Handle ERROR node for interface ports
                if child.type == "ERROR" and child.prev_sibling and child.prev_sibling.type == "simple_identifier":
                    last_identifier = child.prev_sibling
                    logger.debug("Adjusting name search due to ERROR node (interface port).")
                    break
            
            if last_identifier:
                potential_names = [last_identifier.text.decode('utf8').strip()]
            else:
                potential_names = self._find_identifiers_recursive(node)
        
        logger.debug(f"Potential names found: {potential_names}")
        
        # Filter and deduplicate names
        filtered_names = []
        seen_names = set()
        keywords_to_exclude = set([d.value for d in Direction])
        
        for name in potential_names:
            if name and name not in keywords_to_exclude and name not in seen_names:
                filtered_names.append(name)
                seen_names.add(name)
        port_names = filtered_names
        
        logger.debug(f"Filtered port names: {port_names}")
        
        if not port_names:
            logger.warning(f"Failed to extract any valid port names from node: {node.text.decode()}")
            return []
        
        # Create Port objects
        parsed_ports = []
        for name in port_names:
            logger.info(f"Successfully parsed port: Name='{name}', Direction='{direction.value}', Width='{final_width}', Type='{data_type}'")
            parsed_ports.append(Port(name=name, direction=direction, width=final_width))
        
        return parsed_ports
    
    def _extract_direction(self, node: Node) -> Optional[Direction]:
        """Extract the port direction from AST nodes.
        
        Args:
            node: Node potentially containing direction information.
            
        Returns:
            Direction enum value or None.
        """
        if node is None:
            return None
        
        direction = None
        direction_types = ["input", "output", "inout"]
        direction_node = self.ast_parser.find_child(node, ["port_direction"] + direction_types)
        
        if direction_node:
            dir_text = direction_node.text.decode('utf8')
            if dir_text in direction_types:
                direction = Direction(dir_text)
            elif direction_node.type == "port_direction":
                # Find the actual keyword within the port_direction node
                for child in direction_node.children:
                    if child.text.decode('utf8') in direction_types:
                        direction = Direction(child.text.decode('utf8'))
                        break
        
        if direction is None:
            node_text = node.text.decode('utf8')
            first_word = node_text.split()[0] if node_text else ""
            if first_word in direction_types:
                direction = Direction(first_word)
        
        return direction
    
    def _extract_width_from_dimension(self, width_node: Node) -> str:
        """Extract the width string from a dimension node.
        
        Args:
            width_node: Node containing width information.
            
        Returns:
            Width expression string (e.g., '31:0', 'WIDTH-1:0').
        """
        if not width_node:
            return "1"
        
        logger.debug(f"Extracting width from node: Type={width_node.type}, Text='{width_node.text.decode()}'")
        
        # Find the expression node within the dimension
        expr_node = self.ast_parser.find_child(width_node, ["constant_range", "range_expression", "constant_expression", "expression", "primary_literal", "number"])
        
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
                    return width_text
                else:
                    logger.debug(f"Expression node text ('{width_text}') differs from node inner text ('{expected_inner_text}'), using inner text.")
                    return expected_inner_text if expected_inner_text else "1"
            else:
                logger.debug("Original width node not bracketed, using expression node text.")
                return width_text
        else:
            logger.debug("No specific expression node found within width_node.")
            # Fallback: Use cleaned text of the dimension node itself
            cleaned_width_text = width_node.text.decode('utf8').strip()
            if cleaned_width_text.startswith('[') and cleaned_width_text.endswith(']'):
                cleaned_width_text = cleaned_width_text[1:-1].strip()
            logger.debug(f"Using fallback cleaned text: '{cleaned_width_text}'")
            return cleaned_width_text if cleaned_width_text else "1"
    
    def _find_identifiers_recursive(self, node: Node) -> List[str]:
        """Recursively find all identifier texts under a node.
        
        Args:
            node: Root node to search from.
            
        Returns:
            List of identifier strings.
        """
        identifiers = []
        node_type = node.type
        node_text = node.text.decode('utf8').strip()
        
        # Keywords to exclude
        keywords_to_exclude = [d.value for d in Direction] + [
            'logic', 'reg', 'wire', 'bit', 'integer', 'input', 'output', 'inout',
            'signed', 'unsigned', 'parameter', 'localparam', 'module', 'endmodule',
            'interface', 'endinterface'
        ]
        
        if node_type in ["simple_identifier", "identifier", "port_identifier"] and node_text not in keywords_to_exclude:
            # Check parent type to avoid module names
            if not (node.parent and node.parent.type in ["module_declaration", "module_identifier", "interface_identifier"]):
                identifiers.append(node_text)
        
        # Recursive step
        for child in node.children:
            # Skip certain node types
            if child.type not in ['data_type', 'parameter_port_list', 'parameter_declaration']:
                identifiers.extend(self._find_identifiers_recursive(child))
        
        # Return unique identifiers preserving order
        return list(dict.fromkeys(identifiers))
    
    def select_target_module(self, module_nodes: List[Node], pragmas: List[Pragma], 
                           source_name: str, target_module: Optional[str] = None) -> Node:
        """Select the target module based on pragmas or explicit target.
        
        Args:
            module_nodes: List of module nodes found in AST.
            pragmas: List of extracted pragmas.
            source_name: Name of source for error messages.
            target_module: Optional explicit module name to target.
            
        Returns:
            Selected module node.
            
        Raises:
            ValueError: If module selection fails.
        """
        top_module_pragmas = [p for p in pragmas if p.type == PragmaType.TOP_MODULE]
        
        # Extract module names
        module_names_map = {}
        for node in module_nodes:
            name = self.extract_module_name(node)
            if name:
                module_names_map[name] = node
            else:
                logger.warning(f"Could not extract module name from node: {node.text.decode()[:50]}...")
        
        # Priority 1: Explicit target_module parameter
        if target_module:
            if module_names_map and target_module in module_names_map:
                logger.info(f"Found explicitly requested module '{target_module}'.")
                return module_names_map[target_module]
            else:
                available = list(module_names_map.keys()) if module_names_map else []
                raise ValueError(
                    f"Requested module '{target_module}' not found in {source_name}. "
                    f"Available modules: {available}"
                )
        
        # Priority 2: Single module (with or without TOP_MODULE pragma)
        if len(module_nodes) == 1:
            if top_module_pragmas:
                # Verify the TOP_MODULE pragma matches
                target_name = top_module_pragmas[0].parsed_data.get("module_name")
                actual_name = self.extract_module_name(module_nodes[0])
                
                if not actual_name:
                    raise ValueError(
                        f"Could not determine module name for comparison "
                        f"with TOP_MODULE pragma '{target_name}'."
                    )
                
                if actual_name != target_name:
                    raise ValueError(
                        f"TOP_MODULE pragma specifies '{target_name}', "
                        f"but the only module found is '{actual_name}'."
                    )
                
                logger.debug(f"Found single module '{actual_name}' matching TOP_MODULE pragma.")
            else:
                logger.debug("Found single module, selecting it as target.")
            
            return module_nodes[0]
        
        # Priority 3: Multiple modules (requires TOP_MODULE pragma)
        elif len(module_nodes) > 1:
            if len(top_module_pragmas) == 1:
                target_name = top_module_pragmas[0].parsed_data.get("module_name")
                logger.info(f"Found TOP_MODULE pragma, searching for module '{target_name}'.")
                
                if module_names_map and target_name in module_names_map:
                    logger.debug(f"Found matching module '{target_name}'.")
                    return module_names_map[target_name]
                else:
                    raise ValueError(
                        f"TOP_MODULE pragma specified '{target_name}', "
                        f"but no such module found in {source_name}."
                    )
            elif len(top_module_pragmas) > 1:
                raise ValueError(
                    f"Multiple TOP_MODULE pragmas found in {source_name}. "
                    f"Only one is allowed."
                )
            else:
                available = list(module_names_map.keys()) if module_names_map else []
                raise ValueError(
                    f"Multiple modules ({available}) found in {source_name}, "
                    f"but no TOP_MODULE pragma specified."
                )
        
        else:
            raise ValueError("Internal error: Inconsistent module node state.")