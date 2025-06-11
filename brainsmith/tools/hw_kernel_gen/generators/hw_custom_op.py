"""
HWCustomOp generator (simplified version).

Based on hw_kernel_gen_simple pattern with template compatibility
for HWCustomOp generation.
"""

from pathlib import Path
from .base import GeneratorBase
from ..rtl_parser.data import ParsedKernelData


class HWCustomOpGenerator(GeneratorBase):
    """HWCustomOp generator for FINN integration."""
    
    def __init__(self, template_dir: Path = None):
        super().__init__('hw_custom_op_slim.py.j2', template_dir)
    
    def _get_output_filename(self, parsed_data: ParsedKernelData) -> str:
        """Get output filename for HWCustomOp class."""
        return f"{parsed_data.name.lower()}_hwcustomop.py"