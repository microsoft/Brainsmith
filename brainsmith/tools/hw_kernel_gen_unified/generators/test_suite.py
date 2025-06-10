"""
Unified test suite generator.

Based on hw_kernel_gen_simple pattern with template compatibility
for test suite generation.
"""

from pathlib import Path
from .base import GeneratorBase
from ..data import UnifiedHWKernel


class UnifiedTestSuiteGenerator(GeneratorBase):
    """Unified test suite generator for comprehensive validation."""
    
    def __init__(self, template_dir: Path = None):
        super().__init__('test_suite.py.j2', template_dir)
    
    def _get_output_filename(self, hw_kernel: UnifiedHWKernel) -> str:
        """Get output filename for test suite."""
        return f"test_{hw_kernel.name.lower()}.py"