"""
HWCustomOp generator for HWKG.

Generates AutoHWCustomOp subclasses using the Phase 2 template system.
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


class HWCustomOpGenerator(GeneratorBase):
    """Generates AutoHWCustomOp subclass with runtime parameter extraction."""
    
    name = "hw_custom_op"
    template_file = "hw_custom_op_phase2.py.j2"
    output_pattern = "{kernel_name}_hw_custom_op.py"
    
    def process_context(self, context: TemplateContext) -> Dict:
        """
        Process context for HWCustomOp generation.
        
        Uses default pass-through behavior since the Phase 2 template
        expects the full context structure.
        
        Args:
            context: Full template context
            
        Returns:
            Context dictionary for template rendering
        """
        return self.context_to_dict(context)