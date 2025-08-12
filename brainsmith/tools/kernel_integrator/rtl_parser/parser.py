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
from typing import Optional, List, Tuple
from pathlib import Path
from tree_sitter import Node, Tree

from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata
from brainsmith.tools.kernel_integrator.types.rtl import PragmaType, Parameter, Port
from .pragmas import Pragma
from .pragma import PragmaHandler
from .ast_parser import ASTParser, SyntaxError
from .kernel_builder import KernelBuilder
from .parameter_linker import ParameterLinker

# Configure logger
logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Base class for parser errors."""
    pass


class RTLParser:
    """Parser for SystemVerilog RTL files.

    This class orchestrates the parsing of SystemVerilog files using sub-components:
    - ASTParser: Handles tree-sitter operations
    - KernelBuilder: Selects modules, extracts components, and builds KernelMetadata
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
        self.kernel_builder = KernelBuilder(self.ast_parser, debug=self.debug)
        self.pragma_handler = PragmaHandler(debug=self.debug)

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
    
    def _sync_parameter_exposure(self, kernel_metadata: KernelMetadata) -> None:
        """Sync Parameter objects with exposed_parameters list.
        
        Updates source_type and source_detail on Parameter objects based on the 
        linked_parameters after pragma processing.
        
        Args:
            kernel_metadata: KernelMetadata with updated exposed_parameters.
        """
        # Import SourceType
        from brainsmith.tools.kernel_integrator.types.rtl import SourceType
        
        # Create a set for efficient lookup
        exposed_set = set(kernel_metadata.exposed_parameters)
        
        # Update each Parameter object
        for param in kernel_metadata.parameters:
            # Update source information based on linked_parameters
            if param.name in kernel_metadata.linked_parameters.get("aliases", {}):
                param.source_type = SourceType.NODEATTR_ALIAS
                param.source_detail = {"nodeattr_name": kernel_metadata.linked_parameters["aliases"][param.name]}
            elif param.name in kernel_metadata.linked_parameters.get("derived", {}):
                param.source_type = SourceType.DERIVED
                param.source_detail = {"expression": kernel_metadata.linked_parameters["derived"][param.name]}
            elif param.name in kernel_metadata.linked_parameters.get("axilite", {}):
                param.source_type = SourceType.AXILITE
                param.source_detail = {"interface_name": kernel_metadata.linked_parameters["axilite"][param.name]}
            # else: source_type remains SourceType.RTL (default)
            
        logger.debug(f"Synced parameter exposure: {len(exposed_set)} exposed out of {len(kernel_metadata.parameters)} total")
    
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
        
        # Assign parameter categories based on usage
        linker.assign_parameter_categories(kernel_metadata)

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
            # Parse AST
            try:
                tree = self.ast_parser.parse_source(systemverilog_code)
            except Exception as e:
                raise ParserError(f"Core parsing failed for {source_name}: {e}")
            
            # Check syntax
            syntax_error = self.ast_parser.check_syntax_errors(tree)
            if syntax_error:
                raise syntax_error
            
            # Build KernelMetadata (pragma extraction now happens inside)
            try:
                kernel_metadata = self.kernel_builder.build_kernel_metadata(
                    tree, source_name, module_name
                )
            except ValueError as e:
                raise ParserError(str(e))
            except Exception as e:
                logger.exception(f"Error during kernel building: {e}")
                raise ParserError(f"Failed during kernel building: {e}")

            # Set pragmas in handler for high-level operations
            self.pragma_handler.set_pragmas(kernel_metadata.pragmas)
            
            # Apply all pragmas to KernelMetadata
            self._apply_pragmas(kernel_metadata)
            
            # Update Parameter objects based on pragma results
            self._sync_parameter_exposure(kernel_metadata)
            
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