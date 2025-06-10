"""
Unified RTL backend generator.

Based on hw_kernel_gen_simple pattern with template compatibility
for RTL backend component generation.
"""

from pathlib import Path
from .base import GeneratorBase
from ..data import UnifiedHWKernel


class UnifiedRTLBackendGenerator(GeneratorBase):
    """Unified RTL backend generator for FINN integration."""
    
    def __init__(self, template_dir: Path = None):
        super().__init__('rtl_backend.py.j2', template_dir)
    
    def _get_output_filename(self, hw_kernel: UnifiedHWKernel) -> str:
        """Get output filename for RTL backend class."""
        return f"{hw_kernel.name.lower()}_rtlbackend.py"