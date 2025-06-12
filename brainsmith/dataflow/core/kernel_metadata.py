"""
Unified kernel metadata for direct RTL-to-AutoHWCustomOp generation.

This module defines KernelMetadata, which unifies RTL parsing results with
InterfaceMetadata creation, eliminating the need for intermediate transformations.
The RTL parser creates this directly, and templates use it without conversion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from .interface_metadata import InterfaceMetadata
from ...tools.hw_kernel_gen.rtl_parser.data import Parameter, Pragma


@dataclass
class KernelMetadata:
    """
    Complete metadata for AutoHWCustomOp generation from RTL.
    
    This unified structure contains all information needed to generate
    an AutoHWCustomOp subclass, with the RTL parser creating InterfaceMetadata
    directly instead of requiring post-processing transformation.
    """
    
    # === Core Identification ===
    
    name: str
    """
    The RTL module name (e.g., 'thresholding_axi').
    
    Reason: Essential identifier for the kernel.
    Use: 
    - Generate class name (ThresholdingAxi)
    - Set kernel_name attribute in generated class
    - Reference in error messages and logging
    """
    
    source_file: Path
    """
    Absolute path to the source RTL file.
    
    Reason: Traceability and debugging.
    Use:
    - Document generation source in comments
    - Error messages can reference source location
    - Future re-parsing or validation needs
    """
    
    # === Interface Definitions (The Key Innovation) ===
    
    interfaces: List[InterfaceMetadata]
    """
    List of InterfaceMetadata objects created directly by RTL parser.
    
    Reason: This is the KEY UNIFICATION - instead of creating intermediate
    Interface objects that need conversion, the RTL parser directly creates
    the InterfaceMetadata objects that AutoHWCustomOp needs.
    
    Use:
    - Passed directly to AutoHWCustomOp.__init__()
    - No transformation needed in templates
    - Contains interface types, datatype constraints, chunking strategies
    """
    
    # === RTL Parameters ===
    
    parameters: List[Parameter]
    """
    SystemVerilog parameters from the RTL module.
    
    Reason: RTL parameters represent configurable aspects of the hardware
    that may need to be exposed as FINN node attributes.
    
    Use:
    - Generate node attributes in get_nodeattr_types()
    - Provide default values for kernel configuration
    - Templates decide which parameters to expose
    """
    
    # === Pragma Information ===
    
    pragmas: List[Pragma]
    """
    All @brainsmith pragmas found in the RTL source.
    
    Reason: Pragmas provide metadata not inferrable from RTL structure.
    
    Use:
    - Already processed into InterfaceMetadata for chunking/datatypes
    - Kept for completeness and future extensibility
    - May contain directives for advanced features
    """
    
    # === Parsing Diagnostics ===
    
    parsing_warnings: List[str] = field(default_factory=list)
    """
    Non-fatal warnings encountered during RTL parsing.
    
    Reason: RTL parsing may encounter ambiguities or make assumptions
    that should be communicated to developers.
    
    Use:
    - Include as comments in generated code
    - Log during generation process
    - Help developers understand parsing decisions
    """
    
    def get_class_name(self) -> str:
        """
        Generate Python class name from module name.
        
        Converts snake_case or kebab-case to PascalCase.
        Examples: 
        - thresholding_axi → ThresholdingAxi
        - matrix-multiply → MatrixMultiply
        """
        parts = self.name.replace('-', '_').split('_')
        return ''.join(word.capitalize() for word in parts if word)
    
    def validate(self) -> List[str]:
        """
        Validate the metadata for completeness and consistency.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append("Kernel name is required")
            
        if not self.interfaces:
            errors.append("At least one interface is required")
            
        # Check for at least one input and one output
        has_input = any(iface.interface_type.value == 'input' for iface in self.interfaces)
        has_output = any(iface.interface_type.value == 'output' for iface in self.interfaces)
        
        if not has_input:
            errors.append("Kernel must have at least one input interface")
        if not has_output:
            errors.append("Kernel must have at least one output interface")
                
        return errors