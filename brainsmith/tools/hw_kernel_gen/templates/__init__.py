"""
Templates module for HW Kernel Generator.

This module provides template context classes and code generation utilities
for the HWKG system.
"""

# Export main template utilities
from .template_context import TemplateContext
from .context_generator import TemplateContextGenerator

# Export code generation utilities  
from .codegen_binding_generator import generate_codegen_binding

__all__ = [
    "TemplateContext",
    "TemplateContextGenerator", 
    "generate_codegen_binding"
]