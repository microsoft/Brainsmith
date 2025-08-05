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

import logging
from typing import Optional, List
from pathlib import Path
from tree_sitter import Node, Tree

from ..metadata import KernelMetadata
from .rtl_data import PragmaType, Parameter, Port
from .pragmas import Pragma
from .pragma import PragmaHandler
from .interface_builder import InterfaceBuilder
from .ast_parser import ASTParser, SyntaxError
from .module_extractor import ModuleExtractor
from .parameter_linker import ParameterLinker

# Configure logger
logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Base class for parser errors."""
    pass


class ParsedData:
    """Container for parsed module data before KernelMetadata creation."""
    
    def __init__(self):
        self.module_name: Optional[str] = None
        self.parameters: List[Parameter] = []
        self.ports: List[Port] = []
        self.pragmas: List[Pragma] = []
        self.parsing_warnings: List[str] = []


class RTLParser:
    """Parser for SystemVerilog RTL files.

    This class orchestrates the parsing of SystemVerilog files using sub-components:
    - ASTParser: Handles tree-sitter operations
    - ModuleExtractor: Selects modules and extracts parameters/ports
    - InterfaceBuilder: Builds interface metadata
    - PragmaHandler: Processes pragmas
    - ParameterLinker: Auto-links parameters to interfaces

    Attributes:
        debug: Enable debug output
        auto_link_parameters: Enable automatic parameter linking
    """
    
    def __init__(self, debug: bool = False,
                 auto_link_parameters: bool = True, strict: bool = True):
        """Initializes the RTLParser.

        Creates sub-components for AST parsing, component extraction,
        and workflow orchestration.

        Args:
            debug: If True, enables detailed debug logging.
            auto_link_parameters: If True, enables automatic parameter linking based
                                on naming conventions. Default is True.
            strict: If True, enables strict validation of parsed metadata. When False,
                    validation is skipped allowing parsing of files that don't meet all
                    requirements. Default is True.

        Raises:
            RuntimeError: For unexpected errors during initialization.
        """
        self.debug = debug
        self.auto_link_parameters = auto_link_parameters
        self.strict = strict
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

        # Initialize sub-components
        self.ast_parser = ASTParser(debug=self.debug)
        self.module_extractor = ModuleExtractor(self.ast_parser, debug=self.debug)
        self.interface_builder = InterfaceBuilder(debug=self.debug)
        self.pragma_handler = PragmaHandler(debug=self.debug)

    def _parse_and_extract(self, source: str, source_name: str = "<string>", 
                          is_file: bool = True, target_module: Optional[str] = None) -> ParsedData:
        """Parse source and extract components using sub-components.

        Args:
            source: File path (if is_file=True) or SystemVerilog source code string
            source_name: Name for logging/error messages (file path or "<string>")
            is_file: If True, treat source as file path; if False, treat as source code
            target_module: Optional specific module name to target

        Returns:
            ParsedData containing extracted information.

        Raises:
            ParserError: If parsing fails, no modules found, or module selection fails
            SyntaxError: If SystemVerilog syntax errors detected by tree-sitter
            FileNotFoundError: If file does not exist
        """
        logger.info(f"Parsing and extracting from {source_name}")
        
        # Get source content
        if is_file:
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.exception(f"Failed to read file {source}: {e}")
                raise ParserError(f"Failed to read file {source}: {e}")
        else:
            content = source
        
        # Parse AST
        try:
            tree = self.ast_parser.parse_source(content)
        except Exception as e:
            raise ParserError(f"Core parsing failed for {source_name}: {e}")
        
        # Check syntax
        syntax_error = self.ast_parser.check_syntax_errors(tree)
        if syntax_error:
            raise syntax_error
        
        # Create parsed data container
        data = ParsedData()
        
        # Find modules
        module_nodes = self.ast_parser.find_modules(tree)
        if not module_nodes:
            raise ParserError(f"No module definitions found in {source_name}")
        
        # Extract pragmas
        logger.debug("Extracting pragmas...")
        try:
            data.pragmas = self.pragma_handler.extract_pragmas(tree.root_node)
        except Exception as e:
            logger.exception(f"Error during pragma extraction in {source_name}: {e}")
            raise ParserError(f"Failed during pragma extraction: {e}")
        logger.debug(f"Found {len(data.pragmas)} potential pragmas.")
        
        # Select target module
        try:
            module_node = self.module_extractor.select_target_module(
                module_nodes, data.pragmas, source_name, target_module
            )
            logger.info(f"Selected target module node: {module_node.type}")
        except ValueError as e:
            logger.error(e)
            raise ParserError(str(e))
        
        # Extract components
        logger.info("Extracting kernel components (name, parameters, ports)")
        
        # Extract module name
        data.module_name = self.module_extractor.extract_module_name(module_node)
        if not data.module_name:
            raise ParserError("Failed to extract module name from header.")
        logger.debug(f"Extracted module name: '{data.module_name}'")
        
        # Extract parameters
        try:
            data.parameters = self.module_extractor.extract_parameters(module_node)
            logger.debug(f"Extracted {len(data.parameters)} parameters.")
        except Exception as e:
            logger.exception(f"Error during parameter parsing: {e}")
            raise ParserError(f"Failed during parameter parsing: {e}")
        
        # Extract ports
        try:
            data.ports = self.module_extractor.extract_ports(module_node)
            logger.debug(f"Successfully parsed {len(data.ports)} individual port objects.")
        except Exception as e:
            logger.exception(f"Error during port parsing: {e}")
            raise ParserError(f"Failed during port parsing: {e}")
        
        logger.info("Component extraction complete.")
        return data

    def _apply_pragmas(self, kernel_metadata: KernelMetadata) -> None:
        """Apply all pragmas to kernel metadata.
        
        Args:
            kernel_metadata: KernelMetadata to modify.
        """
        logger.info(f"Applying {len(kernel_metadata.pragmas)} pragmas to kernel metadata")
        
        for pragma in kernel_metadata.pragmas:
            try:
                pragma.apply_to_kernel(kernel_metadata)
            except Exception as e:
                logger.warning(
                    f"Failed to apply pragma {pragma.type.value} "
                    f"at line {pragma.line_number}: {e}"
                )
        
        logger.info(f"Pragma application complete. Exposed parameters: {len(kernel_metadata.exposed_parameters)}")
    
    def _apply_autolinking(self, kernel_metadata: KernelMetadata) -> None:
        """Apply auto-linking to kernel metadata.
        
        Args:
            kernel_metadata: KernelMetadata to modify.
        """
        if not self.auto_link_parameters:
            return
        
        # Delegate all autolinking logic to ParameterLinker
        linker = ParameterLinker(enable_interface_linking=True, enable_internal_linking=True)
        linker.apply_to_kernel_metadata(kernel_metadata)

    def parse(self, systemverilog_code: str, source_name: str = "<string>", module_name: Optional[str] = None) -> KernelMetadata:
        """Core SystemVerilog string parser.
        
        Args:
            systemverilog_code: SystemVerilog module source code
            source_name: Name for logging/error messages (default: "<string>")
            module_name: Optional target module name (auto-detect if None)
            
        Returns:
            KernelMetadata: Parsed kernel metadata with InterfaceMetadata objects
            
        Raises:
            SyntaxError: Invalid SystemVerilog syntax
            ParserError: Parser configuration or runtime error
        """
        logger.info(f"Starting string-based parsing for: {source_name}")
        try:
            # Parse and extract components
            parsed_data = self._parse_and_extract(
                systemverilog_code, source_name, is_file=False, target_module=module_name
            )
            
            # Initialize exposed parameters - all parameters are initially exposed
            exposed_parameters = [p.name for p in parsed_data.parameters]
            logger.debug(f"Initialized {len(exposed_parameters)} exposed parameters: {exposed_parameters}")

            # Build initial InterfaceMetadata objects (without pragma application)
            base_metadata_list, unassigned_ports = self.interface_builder.build_interface_metadata(
                parsed_data.ports
            )
            logger.info(f"Built {len(base_metadata_list)} base interfaces from AST")

            # Build KernelMetadata with initial data
            kernel_metadata = KernelMetadata(
                name=parsed_data.module_name,
                source_file=Path(source_name),
                interfaces=base_metadata_list,
                parameters=parsed_data.parameters,
                exposed_parameters=exposed_parameters,
                pragmas=parsed_data.pragmas,
                parsing_warnings=parsed_data.parsing_warnings,
                linked_parameters={"aliases": {}, "derived": {}, "axilite": {}},
                internal_datatypes=[]
            )

            # Apply ALL pragmas to KernelMetadata
            self._apply_pragmas(kernel_metadata)
            
            # Auto-linking with remaining parameters
            self._apply_autolinking(kernel_metadata)
            
            # Validate the complete KernelMetadata only if strict mode is enabled
            if self.strict:
                try:
                    logger.info("Starting KernelMetadata validation...")
                    validation_errors = kernel_metadata.validate()
                    logger.info(f"Validation returned {len(validation_errors)} errors")
                    if validation_errors:
                        logger.error(f"Validation errors: {validation_errors}")
                        raise ValueError(f"KernelMetadata validation failed: {'; '.join(validation_errors)}")
                except ValueError as e:
                    logger.error(f"Validation raised ValueError: {e}")
                    raise ParserError(str(e))
            else:
                logger.info("Skipping validation (strict mode disabled)")
            
            logger.info(f"KernelMetadata object created for '{kernel_metadata.name}' with {len(kernel_metadata.parameters)} params ({len(kernel_metadata.exposed_parameters)} exposed), {len(kernel_metadata.interfaces)} interfaces.")
            logger.info(f"Successfully parsed and processed module '{kernel_metadata.name}' from {source_name}")
            return kernel_metadata

        except (SyntaxError, ParserError) as e:
            logger.error(f"String parsing failed for {source_name}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during string parsing for {source_name}: {e}")
            raise ParserError(f"Unexpected error during string parsing: {e}")

    def parse_file(self, file_path: str) -> KernelMetadata:
        """Parse a SystemVerilog file by reading it and calling the core parse method.

        Args:
            file_path: The absolute path to the SystemVerilog file to parse.

        Returns:
            A `KernelMetadata` object containing the parsed information (name, parameters,
            interfaces, pragmas).

        Raises:
            ParserError: If any stage of the parsing process fails due to logical errors,
                         ambiguity, or validation failures.
            SyntaxError: If the input file has SystemVerilog syntax errors.
            FileNotFoundError: If the input file cannot be found.
        """
        logger.info(f"Starting file parsing for: {file_path}")
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                systemverilog_code = f.read()
            
            # Delegate to core parse method
            return self.parse(systemverilog_code, file_path)
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise
        except (UnicodeDecodeError, IOError) as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise ParserError(f"Failed to read file {file_path}: {e}")
        except (SyntaxError, ParserError):
            # Re-raise parsing errors as-is (already logged by parse method)
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during file parsing for {file_path}: {e}")
            raise ParserError(f"Unexpected error during file parsing: {e}")