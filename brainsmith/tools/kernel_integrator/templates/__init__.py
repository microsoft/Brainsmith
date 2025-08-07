"""
Templates module for Kernel Integrator.

This module provides template context classes and code generation utilities
for the KI system.
"""

# Export main template utilities
from .template_context import TemplateContext
from .context_generator import TemplateContextGenerator

__all__ = [
    "TemplateContext",
    "TemplateContextGenerator"
]