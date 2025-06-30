"""
RTL Backend generator with CodegenBinding.

Generates AutoRTLBackend subclasses using the modern template system with CodegenBinding.
This is the primary and only RTL backend generator.
"""

from typing import Dict

from .base import GeneratorBase
from ..templates.template_context import TemplateContext


class RTLBackendGenerator(GeneratorBase):
    """Generates AutoRTLBackend subclass with CodegenBinding."""
    
    name = "rtl_backend"
    template_file = "rtl_backend.py.j2"
    output_pattern = "{kernel_name}_rtl.py"
    
    # Using base class process_context which properly converts TemplateContext