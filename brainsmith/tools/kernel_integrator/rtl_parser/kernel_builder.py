############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Kernel metadata builder for SystemVerilog RTL parser.

This module handles:
- Building complete KernelMetadata objects with interfaces
- Coordinating module extraction (delegated to ModuleExtractor)
- Interface building and KernelMetadata assembly
"""

import logging
from typing import Optional, List, Tuple, Dict
from pathlib import Path
from tree_sitter import Node, Tree

from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Port, Parameter
from .pragmas import Pragma
from .ast_parser import ASTParser
from .module_extractor import ModuleExtractor
from .interface_builder import InterfaceBuilder

logger = logging.getLogger(__name__)


class KernelBuilder:
    """Builds KernelMetadata from SystemVerilog AST nodes.
    
    This class handles:
    - Coordinating module component extraction (via ModuleExtractor)
    - Interface building (via InterfaceBuilder)
    - KernelMetadata assembly from extracted components
    """
    
    def __init__(self, ast_parser: ASTParser, debug: bool = False):
        """Initialize the kernel builder.
        
        Args:
            ast_parser: ASTParser instance for node traversal utilities.
            debug: Enable debug logging.
        """
        self.ast_parser = ast_parser
        self.module_extractor = ModuleExtractor(ast_parser, debug=debug)
        self.interface_builder = InterfaceBuilder(debug=debug)
        self.debug = debug
    
    def build_kernel_metadata(self, tree: Tree, source_name: str = "<string>", 
                            target_module: Optional[str] = None) -> KernelMetadata:
        """Build complete KernelMetadata from AST tree.
        
        This method handles the complete workflow:
        1. Uses ModuleExtractor to get module components and pragmas
        2. Builds interfaces  
        3. Assembles KernelMetadata
        
        Args:
            tree: Parsed AST tree.
            source_name: Source file name for error messages.
            target_module: Explicit target module name (optional).
            
        Returns:
            Complete KernelMetadata object with interfaces built.
        """
        # Use ModuleExtractor to get all components including pragmas
        module_name, parameters, ports, pragmas = self.module_extractor.extract_from_tree(
            tree, source_name, target_module
        )
        
        # Then build metadata
        return self._build_from_components(
            module_name, parameters, ports, pragmas, source_name
        )
    
    def _build_from_components(self, module_name: str, parameters: List[Parameter], 
                               ports: List[Port], pragmas: List[Pragma], 
                               source_name: str) -> KernelMetadata:
        """Build KernelMetadata from already extracted components.
        
        Internal method to build KernelMetadata from components.
        """
        # Note: This is now a private method
        
        # Initialize parsing warnings
        parsing_warnings: List[str] = []
        
        # Initialize exposed parameters - all parameters are initially exposed
        exposed_parameters = [p.name for p in parameters]
        logger.debug(f"Initialized {len(exposed_parameters)} exposed parameters: {exposed_parameters}")
        
        # Build initial InterfaceMetadata objects
        base_metadata_list, unassigned_ports = self.interface_builder.build_interface_metadata(
            ports
        )
        logger.info(f"Built {len(base_metadata_list)} base interfaces from AST")
        
        # Build KernelMetadata with initial data
        kernel_metadata = KernelMetadata(
            name=module_name,
            source_file=Path(source_name),
            interfaces=base_metadata_list,
            parameters=parameters,
            exposed_parameters=exposed_parameters,
            pragmas=pragmas,
            parsing_warnings=parsing_warnings,
            linked_parameters={"aliases": {}, "derived": {}, "axilite": {}},
            internal_datatypes=[]
        )
        
        return kernel_metadata