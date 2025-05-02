"""Pragma processing for Hardware Kernel Generator.

This module handles the parsing and validation of @brainsmith pragmas found
in SystemVerilog comments. Pragmas provide additional information to guide
the Hardware Kernel Generator.

Example pragma:
    // @brainsmith interface AXI_STREAM
    // @brainsmith parameter STATIC WIDTH
"""

from enum import Enum
from typing import List, Optional, Dict, Callable
from tree_sitter import Node

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Pragma

class PragmaType(Enum):
    """Valid pragma types."""
    INTERFACE = "interface"    # Specify interface protocol
    PARAMETER = "parameter"    # Parameter configuration
    RESOURCE = "resource"     # Resource utilization hints
    TIMING = "timing"         # Timing constraints
    FEATURE = "feature"       # Optional features

class PragmaError(Exception):
    """Base class for pragma-related errors."""
    pass

class PragmaParser:
    """Parser for @brainsmith pragmas.
    
    This class handles the extraction and validation of pragmas from
    SystemVerilog comments.
    
    Attributes:
        debug: Enable debug output
        handlers: Dict of pragma type to handler functions
    """
    
    def __init__(self, debug: bool = False):
        """Initialize pragma parser.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.handlers: Dict[str, Callable[[List[str]], Dict]] = {
            PragmaType.INTERFACE.value: self._handle_interface,
            PragmaType.PARAMETER.value: self._handle_parameter,
            PragmaType.RESOURCE.value: self._handle_resource,
            PragmaType.TIMING.value: self._handle_timing,
            PragmaType.FEATURE.value: self._handle_feature
        }
    
    def parse_comment(self, node: Node, line_number: int) -> Optional[Pragma]:
        """Parse a comment node for @brainsmith pragmas.
        
        Args:
            node: Comment AST node
            line_number: Source line number
            
        Returns:
            Pragma instance if valid pragma found, None otherwise
        """
        text = node.text.decode('utf8').strip('/ ')
        
        # Check for pragma prefix
        if not text.startswith('@brainsmith'):
            return None
        
        # Split into components
        parts = text.split()
        if len(parts) < 2:
            if self.debug:
                print(f"Warning: Invalid pragma format at line {line_number}")
            return None
        
        # Extract type and inputs
        pragma_type = parts[1]
        inputs = parts[2:] if len(parts) > 2 else []
        
        # Validate type
        if pragma_type not in self.handlers:
            if self.debug:
                print(f"Warning: Unknown pragma type '{pragma_type}' at line {line_number}")
            return None
        
        # Create pragma instance
        pragma = Pragma(
            type=pragma_type,
            inputs=inputs,
            line_number=line_number
        )
        
        # Process with appropriate handler
        try:
            pragma.processed_data = self.handlers[pragma_type](inputs)
        except PragmaError as e:
            if self.debug:
                print(f"Warning: {e} at line {line_number}")
            return None
        
        return pragma
    
    def _handle_interface(self, inputs: List[str]) -> Dict:
        """Handle interface pragma.
        
        Format: @brainsmith interface <protocol>
        
        Args:
            inputs: Pragma inputs
            
        Returns:
            Processed interface data
            
        Raises:
            PragmaError: If inputs invalid
        """
        if not inputs:
            raise PragmaError("Interface protocol not specified")
        return {
            "protocol": inputs[0],
            "options": inputs[1:] if len(inputs) > 1 else []
        }
    
    def _handle_parameter(self, inputs: List[str]) -> Dict:
        """Handle parameter pragma.
        
        Format: @brainsmith parameter <type> <name>
        
        Args:
            inputs: Pragma inputs
            
        Returns:
            Processed parameter data
            
        Raises:
            PragmaError: If inputs invalid
        """
        if len(inputs) < 2:
            raise PragmaError("Parameter type and name required")
        return {
            "param_type": inputs[0],
            "name": inputs[1],
            "options": inputs[2:] if len(inputs) > 2 else []
        }
    
    def _handle_resource(self, inputs: List[str]) -> Dict:
        """Handle resource pragma.
        
        Format: @brainsmith resource <type> <value>
        
        Args:
            inputs: Pragma inputs
            
        Returns:
            Processed resource data
            
        Raises:
            PragmaError: If inputs invalid
        """
        if len(inputs) < 2:
            raise PragmaError("Resource type and value required")
        return {
            "resource_type": inputs[0],
            "value": inputs[1],
            "options": inputs[2:] if len(inputs) > 2 else []
        }
    
    def _handle_timing(self, inputs: List[str]) -> Dict:
        """Handle timing pragma.
        
        Format: @brainsmith timing <constraint> <value>
        
        Args:
            inputs: Pragma inputs
            
        Returns:
            Processed timing data
            
        Raises:
            PragmaError: If inputs invalid
        """
        if len(inputs) < 2:
            raise PragmaError("Timing constraint and value required")
        return {
            "constraint": inputs[0],
            "value": inputs[1],
            "options": inputs[2:] if len(inputs) > 2 else []
        }
    
    def _handle_feature(self, inputs: List[str]) -> Dict:
        """Handle feature pragma.
        
        Format: @brainsmith feature <name> [enabled|disabled]
        
        Args:
            inputs: Pragma inputs
            
        Returns:
            Processed feature data
            
        Raises:
            PragmaError: If inputs invalid
        """
        if not inputs:
            raise PragmaError("Feature name required")
        enabled = True
        if len(inputs) > 1:
            if inputs[1].lower() == "disabled":
                enabled = False
            elif inputs[1].lower() != "enabled":
                raise PragmaError("Feature must be 'enabled' or 'disabled'")
        return {
            "name": inputs[0],
            "enabled": enabled,
            "options": inputs[2:] if len(inputs) > 2 else []
        }

def extract_pragmas(root_node: Node) -> List[Pragma]:
    """Extract all pragmas from AST.
    
    Args:
        root_node: Root AST node
        
    Returns:
        List of found pragmas
    """
    parser = PragmaParser()
    pragmas = []
    
    def visit_comments(node: Node, line: int = 1) -> int:
        """Recursively visit nodes to find comments.
        
        Args:
            node: Current AST node
            line: Current line number
            
        Returns:
            Updated line number
        """
        # Update line count
        if node.start_point[0] > line:
            line = node.start_point[0]
        
        # Check for comment
        if node.type == "comment":
            pragma = parser.parse_comment(node, line)
            if pragma is not None:
                pragmas.append(pragma)
        
        # Visit children
        for child in node.children:
            line = visit_comments(child, line)
        
        return line
    
    visit_comments(root_node)
    return pragmas