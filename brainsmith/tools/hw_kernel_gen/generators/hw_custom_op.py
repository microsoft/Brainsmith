"""
HWCustomOp generator (simplified version).

Based on hw_kernel_gen_simple pattern with template compatibility
for HWCustomOp generation.
"""

from pathlib import Path
from .base import GeneratorBase
from brainsmith.dataflow.core.kernel_metadata import KernelMetadata


class HWCustomOpGenerator(GeneratorBase):
    """HWCustomOp generator for FINN integration."""
    
    def __init__(self, template_dir: Path = None):
        super().__init__('hw_custom_op_slim.py.j2', template_dir)
    
    def _get_output_filename(self, kernel_metadata: KernelMetadata) -> str:
        """Get output filename for HWCustomOp class."""
        return f"{kernel_metadata.name.lower()}_hwcustomop.py"