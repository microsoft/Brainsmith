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

from brainsmith.tools.kernel_integrator.metadata import KernelMetadata
from .types import ParsedModule, PragmaType, Parameter, Port
from .pragmas import Pragma
from .ast_parser import ASTParser, SyntaxError
from .kernel_builder import KernelBuilder
from .module_extractor import ModuleExtractor
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
    - ModuleExtractor: Selects modules and extracts components
    - KernelBuilder: Builds interfaces and assembles KernelMetadata
    - ParameterLinker: Auto-links parameters to interfaces

    Attributes:
        debug: Enable debug output
        strict: Enable strict validation (default: True)
    """
    
    def __init__(self, debug: bool = False, strict: bool = True):
        """Initializes the RTLParser.

        Creates sub-components for AST parsing, component extraction,
        and workflow orchestration.

        Args:
            debug: If True, enables detailed debug logging.
            strict: If True, enables strict validation (default: True).
        Raises:
            RuntimeError: For unexpected errors during initialization.
        """
        self.debug = debug
        self.strict = strict
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

        # Initialize sub-components
        self.ast_parser = ASTParser(debug=self.debug)
        self.module_extractor = ModuleExtractor(self.ast_parser, debug=self.debug)
        self.kernel_builder = KernelBuilder(self.ast_parser, debug=self.debug)
        self.linker = ParameterLinker()





    def parse(self, systemverilog_code: str, source_name: str = "<string>") -> KernelMetadata:
        """Core SystemVerilog string parser.
        
        Args:
            systemverilog_code: SystemVerilog module source code
            source_name: Name for logging/error messages (default: "<string>")
            
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
            
            # Extract module components
            parsed_module = self.module_extractor.extract_from_tree(
                tree, source_name
            )
            
            # Build KernelMetadata from parsed module
            kernel_metadata = self.kernel_builder.build(parsed_module)
            
            # Apply all pragmas to KernelMetadata
            self._apply_pragmas(kernel_metadata, parsed_module)

            # Auto-linking with remaining parameters            
            self.linker.link_parameters(kernel_metadata)

            # Format for compiler export
            self._format_for_compiler_export(kernel_metadata)

            # TODO: Add validation toggled by self.strict flag
            
            logger.info(f"KernelMetadata object created for '{kernel_metadata.name}' with {len(kernel_metadata.parameters)} params")
            logger.info(f"Successfully parsed and processed module '{kernel_metadata.name}' from {source_name}")
            return kernel_metadata

        except (SyntaxError, ParserError) as e:
            logger.error(f"String parsing failed for {source_name}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during string parsing for {source_name}: {e}")
            raise ParserError(f"Unexpected error during string parsing: {e}")



    def _apply_pragmas(self, kernel_metadata: KernelMetadata, parsed_module: ParsedModule) -> None:
        """Apply all pragmas to kernel metadata.
        
        Args:
            kernel_metadata: KernelMetadata to modify.
            parsed_module: ParsedModule containing pragmas to apply.
        """
        logger.info(f"Applying {len(parsed_module.pragmas)} pragmas to kernel metadata")

        for pragma in parsed_module.pragmas:
            try:
                pragma.apply_to_kernel(kernel_metadata)
            except Exception as e:
                logger.warning(
                    f"Failed to apply pragma {pragma.type.value} "
                    f"at line {pragma.line_number}: {e}"
                )
        
        logger.info(f"Pragma application complete.")
    
    def _format_for_compiler_export(self, kernel: KernelMetadata) -> None:
        """Format kernel metadata for compiler export.
        
        This method prepares the KernelMetadata for use by downstream compilers
        by applying necessary transformations and standardizations.
        
        Current formatting:
        - Assigns standardized compiler names to interfaces
        
        Future formatting may include:
        - Parameter name standardization
        - Datatype normalization
        - Additional compiler-specific requirements
        
        Args:
            kernel: KernelMetadata to format in-place
        """
        logger.info("Formatting kernel metadata for compiler export")
        
        # Separate inputs by type
        regular_inputs = [iface for iface in kernel.inputs if not iface.is_weight]
        weight_inputs = [iface for iface in kernel.inputs if iface.is_weight]
        
        # Separate AXI-Lite configs by type
        weight_configs = [iface for iface in kernel.config if iface.is_weight]
        regular_configs = [iface for iface in kernel.config if not iface.is_weight]
        
        # Count total weight interfaces (AXI-Stream weights + AXI-Lite weights)
        total_weight_interfaces = len(weight_inputs) + len(weight_configs)
        
        # Assign compiler names to regular inputs
        if len(regular_inputs) == 1:
            regular_inputs[0].compiler_name = "input"
        else:
            for idx, iface in enumerate(regular_inputs):
                iface.compiler_name = f"input{idx}"
        
        # Assign compiler names to all weight interfaces (both AXI-Stream and AXI-Lite)
        weight_idx = 0
        
        # First assign to AXI-Stream weight inputs
        if total_weight_interfaces == 1:
            # Single weight interface - no index needed
            if weight_inputs:
                weight_inputs[0].compiler_name = "weight"
        else:
            # Multiple weight interfaces - use indices
            for iface in weight_inputs:
                iface.compiler_name = f"weight{weight_idx}"
                weight_idx += 1
        
        # Then assign to AXI-Lite weight config interfaces
        if total_weight_interfaces == 1 and weight_configs:
            # Single weight interface and it's AXI-Lite - no index
            weight_configs[0].compiler_name = "weight"
        else:
            # Continue indexing for AXI-Lite weight interfaces
            for config in weight_configs:
                config.compiler_name = f"weight{weight_idx}"
                weight_idx += 1
        
        # Assign compiler names to non-weight AXI-Lite config interfaces
        if len(regular_configs) == 1:
            regular_configs[0].compiler_name = "config"
        else:
            for idx, config in enumerate(regular_configs):
                config.compiler_name = f"config{idx}"
        
        # Assign compiler names to outputs
        if len(kernel.outputs) == 1:
            kernel.outputs[0].compiler_name = "output"
        else:
            for idx, iface in enumerate(kernel.outputs):
                iface.compiler_name = f"output{idx}"
        
        # Log the assignments for debugging
        logger.debug("Compiler name assignments:")
        for iface in kernel.inputs:
            logger.debug(f"  Input '{iface.name}' -> '{iface.compiler_name}'")
        for iface in kernel.outputs:
            logger.debug(f"  Output '{iface.name}' -> '{iface.compiler_name}'")
        for iface in kernel.config:
            logger.debug(f"  Config '{iface.name}' -> '{iface.compiler_name}'")


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