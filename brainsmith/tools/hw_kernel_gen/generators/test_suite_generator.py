"""
Test suite generator for HWKG.

Generates pytest test suites with Phase 2 parameter handling and validation.
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


class TestSuiteGenerator(GeneratorBase):
    """Generates pytest test suite with Phase 2 parameter handling."""
    
    name = "test_suite"
    template_file = "test_suite_v2.py.j2"  # Prefer v2 template
    output_pattern = "test_{kernel_name}.py"
    
    # Template fallback order
    template_fallbacks = [
        "test_suite_v2.py.j2",
        "test_suite.py.j2"
    ]
    
    def process_context(self, context: TemplateContext) -> Dict:
        """
        Process context for test suite generation.
        
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
        
        raise Exception(f"No test suite templates found. Tried: {self.template_fallbacks}")