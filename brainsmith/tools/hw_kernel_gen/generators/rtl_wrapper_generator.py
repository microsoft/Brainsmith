"""
RTL wrapper generator for HWKG.

Generates SystemVerilog RTL wrappers with parameter validation and interface connections.
"""

from typing import Dict

try:
    from .base import GeneratorBase
    from ..templates.template_context import TemplateContext
except ImportError:
    # Handle case when imported directly
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(current_dir.parent))
    from base import GeneratorBase
    from templates.template_context import TemplateContext


class RTLWrapperGenerator(GeneratorBase):
    """Generates SystemVerilog RTL wrapper with enhanced parameter validation."""
    
    name = "rtl_wrapper"
    template_file = "rtl_wrapper_minimal.v.j2"  # Prefer minimal template
    output_pattern = "{kernel_name}_wrapper.v"
    
    # Template fallback order
    template_fallbacks = [
        "rtl_wrapper_minimal.v.j2",
        "rtl_wrapper_v2.v.j2", 
        "rtl_wrapper.v.j2"
    ]
    
    def process_context(self, context: TemplateContext) -> Dict:
        """
        Process context for RTL wrapper generation.
        
        Uses default pass-through behavior. Template fallback logic
        is handled by the get_template_file() method.
        
        Args:
            context: Full template context
            
        Returns:
            Context dictionary for template rendering
        """
        return self.context_to_dict(context)
    
    def get_template_file(self, jinja_env) -> str:
        """
        Get the best available template file with fallbacks.
        
        Args:
            jinja_env: Jinja2 environment to check template availability
            
        Returns:
            Template filename
            
        Raises:
            Exception: If no templates are available
        """
        for template in self.template_fallbacks:
            try:
                jinja_env.get_template(template)
                return template
            except Exception:
                continue
        
        raise Exception(f"No RTL wrapper templates found. Tried: {self.template_fallbacks}")