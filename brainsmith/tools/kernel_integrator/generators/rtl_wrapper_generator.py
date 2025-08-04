"""
RTL wrapper generator for KI.

Generates SystemVerilog RTL wrappers with parameter validation and interface connections.
"""

from typing import Dict

from .base import GeneratorBase
from ..templates import TemplateContext


class RTLWrapperGenerator(GeneratorBase):
    """Generates SystemVerilog RTL wrapper with enhanced parameter validation."""
    
    name = "rtl_wrapper"
    template_file = "rtl_wrapper_minimal.v.j2"
    output_pattern = "{kernel_name}_wrapper.v"
    
    # Using base class process_context which now properly converts TemplateContext
    
