"""
Base class for HWKG generators.

Generators are responsible for processing template context and rendering
Jinja2 templates to produce output files.
"""

from typing import Dict
from abc import ABC

try:
    from ..templates.template_context import TemplateContext
except ImportError:
    # Handle case when imported directly
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from templates.template_context import TemplateContext


class GeneratorBase(ABC):
    """
    Minimal base class for generator definitions.
    
    Each generator must specify:
    - name: Unique identifier for the generator
    - template_file: Jinja2 template filename
    - output_pattern: Format string for output filename
    
    Optionally override process_context() for custom context processing.
    """
    
    name: str = None
    template_file: str = None
    output_pattern: str = None
    
    def process_context(self, context: TemplateContext) -> TemplateContext:
        """
        Process template context for this generator.
        
        Default implementation passes through the TemplateContext directly.
        Override this method to add custom processing, transformations,
        or additional context data specific to this generator.
        
        Args:
            context: Full template context from TemplateContextGenerator
            
        Returns:
            TemplateContext for Jinja2 template rendering
        """
        return context
    
    def get_output_filename(self, kernel_name: str) -> str:
        """
        Generate output filename using the output pattern.
        
        Args:
            kernel_name: Name of the kernel being processed
            
        Returns:
            Formatted output filename
        """
        if not self.output_pattern:
            raise ValueError(f"Generator {self.name} has no output_pattern defined")
        
        return self.output_pattern.format(kernel_name=kernel_name)
    
    def validate(self) -> bool:
        """
        Validate that the generator is properly configured.
        
        Returns:
            True if generator is valid, False otherwise
        """
        if not self.name:
            return False
        if not self.template_file:
            return False
        if not self.output_pattern:
            return False
        return True
    
    def get_template_file(self, jinja_env) -> str:
        """
        Get the template file to use for rendering.
        
        Default implementation returns self.template_file.
        Override this method to implement template fallback logic.
        
        Args:
            jinja_env: Jinja2 environment to check template availability
            
        Returns:
            Template filename to use
            
        Raises:
            Exception: If no suitable template is found
        """
        return self.template_file
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', template='{self.template_file}')"