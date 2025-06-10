"""
HWCustomOp generator implementation.

Generates HWCustomOp Python classes using the simplified pattern.
"""

from pathlib import Path
from .base import GeneratorBase
from ..data import HWKernel


class HWCustomOpGenerator(GeneratorBase):
    """Generates HWCustomOp Python classes."""
    
    def __init__(self, template_dir: Path = None):
        super().__init__('hw_custom_op_slim.py.j2', template_dir)
    
    def _get_output_filename(self, hw_kernel: HWKernel) -> str:
        """Get output filename for HWCustomOp class."""
        return f"{hw_kernel.name.lower()}_hwcustomop.py"
    
    def _get_template_context(self, hw_kernel: HWKernel) -> dict:
        """Get template context for HWCustomOp generation."""
        context = super()._get_template_context(hw_kernel)
        
        # Add HWCustomOp-specific context
        context.update({
            'class_name': f"{hw_kernel.class_name}HWCustomOp",
            'verification_required': hw_kernel.verification_required,
            'kernel_verifications': self._get_kernel_verifications(hw_kernel)
        })
        
        return context
    
    def _get_kernel_verifications(self, hw_kernel: HWKernel) -> list:
        """Get kernel-specific verification rules."""
        verifications = []
        
        # Add common verifications based on kernel type
        if hw_kernel.kernel_type == 'threshold':
            verifications.append({
                'description': 'Verify threshold parameters are within valid range',
                'code': 'assert self.get_nodeattr("threshold") > 0, "Threshold must be positive"'
            })
        elif hw_kernel.kernel_type in ['matmul', 'conv']:
            verifications.append({
                'description': 'Verify input/output dimension compatibility',
                'code': 'assert len(self.get_input_dataflow_interface().tDim) >= 2, "Need at least 2D tensors"'
            })
        
        return verifications