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
    template_file = "test_suite_v2.py.j2"
    output_pattern = "test_{kernel_name}.py"
    
    # Using base class process_context which now properly converts TemplateContext
    
