"""
HWCustomOp generator with CodegenBinding.

Generates AutoHWCustomOp subclasses using the modern template system with CodegenBinding.
This is the primary and only HWCustomOp generator.
"""

from typing import Dict

from .base import GeneratorBase
from ..templates.template_context import TemplateContext


class HWCustomOpGenerator(GeneratorBase):
    """Generates AutoHWCustomOp subclass with CodegenBinding."""
    
    name = "hw_custom_op"
    template_file = "hw_custom_op.py.j2"
    output_pattern = "{kernel_name}_hw_custom_op.py"
    
    # Using base class process_context which properly converts TemplateContext