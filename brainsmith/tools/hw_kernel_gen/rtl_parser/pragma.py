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
from typing import List, Optional, Dict, Callable, Any, Tuple

from tree_sitter import Node

from .data import (
    Pragma, PragmaType, TopModulePragma, DatatypePragma, BDimPragma, SDimPragma,
    DerivedParameterPragma, WeightPragma, DatatypeParamPragma, AliasPragma, 
    ParameterPragma, PragmaError
)
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.datatype_metadata import DatatypeMetadata
from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy
from brainsmith.dataflow.core.interface_types import InterfaceType

# Set up logger for this module
logger = logging.getLogger(__name__)

class PragmaHandler:
    """Extracts, validates, and applies @brainsmith pragmas from comment nodes."""

    def __init__(self, debug: bool = False):
        """Initializes the PragmaHandler and registers pragma handlers."""
        self.debug = debug
        self.pragmas: List[Pragma] = []  # List to store found pragmas
        # Map PragmaType to the corresponding Pragma subclass constructor
        self.pragma_constructors: Dict[PragmaType, Callable[..., Pragma]] = {
            PragmaType.TOP_MODULE: TopModulePragma,
            PragmaType.DATATYPE: DatatypePragma,
            PragmaType.BDIM: BDimPragma,
            PragmaType.SDIM: SDimPragma,
            PragmaType.DERIVED_PARAMETER: DerivedParameterPragma,
            PragmaType.WEIGHT: WeightPragma,
            PragmaType.DATATYPE_PARAM: DatatypeParamPragma,
            PragmaType.ALIAS: AliasPragma,
        }

    def _validate_pragma(self, node: Node, line_number: int) -> Optional[Pragma]:
        """Parses a comment AST node to find and validate a @brainsmith pragma.

        Checks for the '@brainsmith' prefix, extracts the type and inputs,
        validates the type, and instantiates the appropriate Pragma subclass.

        Args:
            node: The tree-sitter comment node.
            line_number: The 1-based line number where the comment starts.

        Returns:
            A validated Pragma subclass object if a valid pragma is found, otherwise None.
        """
        text = node.text.decode('utf8').strip('/ ')

        if not text.startswith('@brainsmith'):
            return None

        parts = text.split()
        if len(parts) < 2:
            logger.warning(f"Invalid pragma format at line {line_number}: {text}")
            return None

        pragma_type_str = parts[1]
        inputs = parts[2:] if len(parts) > 2 else []

        pragma_enum_type: Optional[PragmaType] = None
        pragma_type_lower = pragma_type_str.lower()
        for member in PragmaType:
            if member.value == pragma_type_lower:
                pragma_enum_type = member
                break
        
        if pragma_enum_type is None or pragma_enum_type not in self.pragma_constructors:
            logger.debug(f"Ignoring comment at line {line_number}: Unknown or unsupported pragma type '@brainsmith {pragma_type_str}'")
            return None

        # Get the correct Pragma subclass constructor
        pragma_class = self.pragma_constructors[pragma_enum_type]
        
        try:
            # Instantiate the specific Pragma subclass
            # The _parse_inputs logic is now handled in the Pragma subclass __post_init__
            return pragma_class(
                type=pragma_enum_type,
                inputs=inputs,
                line_number=line_number
            )
        except PragmaError as e:
            # Errors during _parse_inputs (called in __post_init__) will be caught here.
            # The Pragma subclasses already log these errors.
            logger.warning(f"Error instantiating pragma {pragma_enum_type.name} at line {line_number}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error instantiating pragma {pragma_enum_type.name} at line {line_number}: {e}")
            return None

    def extract_pragmas(self, root_node: Node) -> List[Pragma]:
        """Extracts all valid @brainsmith pragmas from an AST by walking comment nodes.

        Uses PragmaParser to parse and validate comments found during the AST traversal.

        Args:
            root_node: The root node of the tree-sitter AST.

        Returns:
            A list of validated Pragma objects found in the AST.
        """
        pragmas = []
        comments_found_count = 0 # Add counter

        # Simple recursive walk for comments - might need optimization for large files
        def find_comments(node: Node):
            nonlocal comments_found_count # Access outer scope variable
            if node.type == 'comment':
                comments_found_count += 1 # Increment counter
                logger.debug(f"Found 'comment' node at line {node.start_point[0]+1}: {node.text.decode('utf8')[:60]}...")
                # Get line number (0-based)
                line_number = node.start_point[0]
                pragma = self._validate_pragma(node, line_number + 1) # Pass 1-based line number
                if pragma:
                    logger.info(f"Found valid pragma: {pragma}")
                    pragmas.append(pragma)

            for child in node.children:
                find_comments(child)

        # Log start/end at INFO level
        logger.info(">>> Starting pragma extraction from AST root.")
        find_comments(root_node)
        logger.info(f"<<< Finished pragma extraction. Found {comments_found_count} comment nodes and {len(pragmas)} valid pragmas.")
        self.pragmas = pragmas # Store the extracted pragmas in the instance
        return pragmas

    def get_interface_pragmas(self) -> List['InterfacePragma']:
        """Get all interface-related pragmas.
        
        Returns:
            List of InterfacePragma instances
        """
        from .data import InterfacePragma
        return [pragma for pragma in self.pragmas if isinstance(pragma, InterfacePragma)]

    def get_parameter_pragmas(self) -> List['ParameterPragma']:
        """Get all parameter-related pragmas.
        
        Returns:
            List of ParameterPragma instances
        """
        return [pragma for pragma in self.pragmas if isinstance(pragma, ParameterPragma)]

    def get_pragmas_by_type(self, pragma_type: PragmaType) -> List[Pragma]:
        """Get all pragmas of a specific type.
        
        Args:
            pragma_type: The PragmaType to filter by
            
        Returns:
            List of Pragma instances of the specified type
        """
        return [pragma for pragma in self.pragmas if pragma.type == pragma_type]

    def apply_interface_pragmas(self, metadata_list: List[InterfaceMetadata], 
                              module_parameters: Optional[List] = None) -> List[InterfaceMetadata]:
        """Apply all interface pragmas to a list of InterfaceMetadata.
        
        This method applies all interface pragmas in sequence using a clean
        chain-of-responsibility pattern.
        
        Args:
            metadata_list: List of InterfaceMetadata to process
            module_parameters: Optional list of module parameters for pragma validation
            
        Returns:
            List[InterfaceMetadata]: List of InterfaceMetadata with all applicable pragmas applied
        """
        interface_pragmas = self.get_interface_pragmas()
        
        if not interface_pragmas:
            logger.debug("No interface pragmas found, returning metadata unchanged")
            return metadata_list
        
        # Set module parameters for BDIM and SDIM pragmas if provided
        if module_parameters is not None:
            from .data import BDimPragma, SDimPragma
            BDimPragma.set_module_parameters(module_parameters)
            SDimPragma.set_module_parameters(module_parameters)
            logger.debug(f"Set {len(module_parameters)} module parameters for pragma validation")
        
        logger.debug(f"Applying {len(interface_pragmas)} interface pragmas to {len(metadata_list)} interfaces")
        
        result_metadata = []
        for metadata in metadata_list:
            current_metadata = metadata
            
            # Apply each interface pragma in sequence
            for pragma in interface_pragmas:
                try:
                    if pragma.applies_to_interface_metadata(current_metadata):
                        logger.debug(f"Applying {pragma.type.value} pragma to interface '{current_metadata.name}'")
                        current_metadata = pragma.apply_to_metadata(current_metadata)
                                
                except Exception as e:
                    # Re-raise validation errors (PragmaError) as they indicate invalid pragma usage
                    from .data import PragmaError
                    if isinstance(e, PragmaError):
                        logger.error(f"Pragma validation failed: {e}")
                        raise
                    else:
                        logger.error(f"Failed to apply {pragma.type.value} pragma to interface '{current_metadata.name}': {e}")
                        # Continue with other pragmas for non-validation errors
            
            result_metadata.append(current_metadata)
        
        logger.debug(f"Applied interface pragmas to {len(result_metadata)} interfaces")
        return result_metadata
    
    def apply_parameter_pragmas(self, exposed_parameters: List[str], 
                               all_parameters: List) -> tuple[List[str], Dict[str, Any]]:
        """Apply all parameter pragmas and return updated exposed parameters and pragma data.
        
        This method processes ALIAS and DERIVED_PARAMETER pragmas, removing affected
        parameters from the exposed list and collecting pragma data for template generation.
        
        Args:
            exposed_parameters: List of parameter names currently exposed
            all_parameters: List of all Parameter objects for validation
            
        Returns:
            Tuple of (updated_exposed_parameters, parameter_pragma_data)
            - updated_exposed_parameters: List with pragma-affected parameters removed
            - parameter_pragma_data: Dict containing 'aliases' and 'derived' mappings
        """
        parameter_pragmas = self.get_parameter_pragmas()
        
        if not parameter_pragmas:
            logger.debug("No parameter pragmas found")
            return exposed_parameters, {"aliases": {}, "derived": {}}
        
        logger.debug(f"Applying {len(parameter_pragmas)} parameter pragmas to {len(exposed_parameters)} exposed parameters")
        
        # Create a copy to avoid modifying the input list
        remaining_exposed = exposed_parameters.copy()
        aliases = {}
        derived = {}
        
        for pragma in parameter_pragmas:
            try:
                if pragma.type == PragmaType.ALIAS:
                    # Validate ALIAS pragma
                    pragma.validate_against_parameters(all_parameters)
                    
                    rtl_param = pragma.parsed_data.get("rtl_param")
                    nodeattr_name = pragma.parsed_data.get("nodeattr_name")
                    
                    if rtl_param in remaining_exposed:
                        aliases[rtl_param] = nodeattr_name
                        remaining_exposed.remove(rtl_param)
                        logger.debug(f"Applied ALIAS pragma: '{rtl_param}' -> '{nodeattr_name}'")
                    else:
                        logger.warning(
                            f"ALIAS pragma at line {pragma.line_number}: Parameter '{rtl_param}' "
                            f"is not in exposed parameters list. Skipping."
                        )
                
                elif pragma.type == PragmaType.DERIVED_PARAMETER:
                    param_name = pragma.parsed_data.get("param_name")
                    python_expression = pragma.parsed_data.get("python_expression")
                    
                    if param_name in remaining_exposed:
                        derived[param_name] = python_expression
                        remaining_exposed.remove(param_name)
                        logger.debug(f"Applied DERIVED_PARAMETER pragma: '{param_name}' = {python_expression}")
                    else:
                        logger.warning(
                            f"DERIVED_PARAMETER pragma at line {pragma.line_number}: Parameter '{param_name}' "
                            f"is not in exposed parameters list. Skipping."
                        )
                        
            except PragmaError as e:
                logger.error(f"Failed to apply parameter pragma: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error applying parameter pragma at line {pragma.line_number}: {e}")
                raise
        
        parameter_pragma_data = {
            "aliases": aliases,
            "derived": derived
        }
        
        logger.debug(f"Applied parameter pragmas: {len(aliases)} aliases, {len(derived)} derived")
        logger.debug(f"Remaining exposed parameters: {len(remaining_exposed)}")
        
        return remaining_exposed, parameter_pragma_data
    
    def collect_internal_datatype_pragmas(self, interface_names: List[str]) -> List[DatatypeMetadata]:
        """
        Collect DATATYPE_PARAM pragmas that don't match any interface.
        
        These pragmas define datatype bindings for internal kernel mechanisms
        like accumulators, thresholds, etc.
        
        Args:
            interface_names: List of interface names to exclude
            
        Returns:
            List of DatatypeMetadata objects for internal mechanisms
        """
        # Get all DATATYPE_PARAM pragmas
        datatype_param_pragmas = self.get_pragmas_by_type(PragmaType.DATATYPE_PARAM)
        
        if not datatype_param_pragmas:
            logger.debug("No DATATYPE_PARAM pragmas found")
            return []
        
        # Group by target name (interface_name in pragma)
        internal_datatypes = {}
        
        for pragma in datatype_param_pragmas:
            target_name = pragma.parsed_data.get('interface_name')
            
            # Skip if it matches an interface
            if target_name in interface_names:
                continue
            
            # Create or update DatatypeMetadata for this internal mechanism
            if target_name not in internal_datatypes:
                internal_datatypes[target_name] = pragma.create_standalone_datatype()
            else:
                # Merge with existing metadata
                property_type = pragma.parsed_data['property_type']
                parameter_name = pragma.parsed_data['parameter_name']
                internal_datatypes[target_name] = internal_datatypes[target_name].update(
                    **{property_type: parameter_name}
                )
        
        logger.info(f"Collected {len(internal_datatypes)} internal datatype bindings: {list(internal_datatypes.keys())}")
        return list(internal_datatypes.values())
    