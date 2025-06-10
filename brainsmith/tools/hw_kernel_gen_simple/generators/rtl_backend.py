"""
RTL Backend generator implementation.

Generates RTL backend Python classes using the simplified pattern.
"""

from pathlib import Path
from .base import GeneratorBase
from ..data import HWKernel


class RTLBackendGenerator(GeneratorBase):
    """Generates RTL backend Python classes."""
    
    def __init__(self, template_dir: Path = None):
        super().__init__('rtl_backend.py.j2', template_dir)
    
    def _get_output_filename(self, hw_kernel: HWKernel) -> str:
        """Get output filename for RTL backend class."""
        return f"{hw_kernel.name.lower()}_rtlbackend.py"
    
    def _get_template_context(self, hw_kernel: HWKernel) -> dict:
        """Get template context for RTL backend generation."""
        context = super()._get_template_context(hw_kernel)
        
        # Add RTL backend-specific context
        context.update({
            'class_name': f"{hw_kernel.class_name}RTLBackend",
            'rtl_wrapper_file': f"{hw_kernel.name.lower()}_wrapper.v",
            'synthesis_directives': self._get_synthesis_directives(hw_kernel),
            'timing_constraints': self._get_timing_constraints(hw_kernel)
        })
        
        return context
    
    def _get_synthesis_directives(self, hw_kernel: HWKernel) -> list:
        """Get synthesis directives based on kernel type."""
        directives = []
        
        if hw_kernel.kernel_type in ['matmul', 'conv']:
            directives.extend([
                'set_directive -PIPELINE "compute_loop"',
                'set_directive -UNROLL "inner_loop"'
            ])
        elif hw_kernel.kernel_type == 'threshold':
            directives.append('set_directive -INLINE "threshold_compare"')
        
        return directives
    
    def _get_timing_constraints(self, hw_kernel: HWKernel) -> dict:
        """Get timing constraints based on kernel complexity."""
        if hw_kernel.kernel_complexity == 'high':
            return {'clock_period': '5ns', 'uncertainty': '0.5ns'}
        elif hw_kernel.kernel_complexity == 'medium':
            return {'clock_period': '4ns', 'uncertainty': '0.4ns'}
        else:
            return {'clock_period': '3ns', 'uncertainty': '0.3ns'}