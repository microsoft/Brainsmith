"""
Test Suite generator implementation.

Generates test suites using the simplified pattern.
"""

from pathlib import Path
from .base import GeneratorBase
from ..data import HWKernel


class TestSuiteGenerator(GeneratorBase):
    """Generates test suites for hardware kernels."""
    
    def __init__(self, template_dir: Path = None):
        super().__init__('test_suite.py.j2', template_dir)
    
    def _get_output_filename(self, hw_kernel: HWKernel) -> str:
        """Get output filename for test suite."""
        return f"test_{hw_kernel.name.lower()}.py"
    
    def _get_template_context(self, hw_kernel: HWKernel) -> dict:
        """Get template context for test suite generation."""
        context = super()._get_template_context(hw_kernel)
        
        # Add test suite-specific context
        context.update({
            'test_class_name': f"Test{hw_kernel.class_name}",
            'hwcustomop_class_name': f"{hw_kernel.class_name}HWCustomOp",
            'test_cases': self._get_test_cases(hw_kernel),
            'test_data_generators': self._get_test_data_generators(hw_kernel)
        })
        
        return context
    
    def _get_test_cases(self, hw_kernel: HWKernel) -> list:
        """Get test cases based on kernel type."""
        test_cases = [
            {
                'name': 'test_initialization',
                'description': 'Test basic kernel initialization',
                'test_type': 'unit'
            },
            {
                'name': 'test_interface_validation',
                'description': 'Test interface metadata validation',
                'test_type': 'unit'
            }
        ]
        
        # Add kernel-specific test cases
        if hw_kernel.kernel_type == 'threshold':
            test_cases.extend([
                {
                    'name': 'test_threshold_values',
                    'description': 'Test various threshold values',
                    'test_type': 'functional'
                },
                {
                    'name': 'test_boundary_conditions',
                    'description': 'Test edge cases and boundary conditions',
                    'test_type': 'functional'
                }
            ])
        elif hw_kernel.kernel_type in ['matmul', 'conv']:
            test_cases.extend([
                {
                    'name': 'test_dimension_compatibility',
                    'description': 'Test input/output dimension matching',
                    'test_type': 'functional'
                },
                {
                    'name': 'test_parallelism_settings',
                    'description': 'Test different parallelism configurations',
                    'test_type': 'performance'
                }
            ])
        
        return test_cases
    
    def _get_test_data_generators(self, hw_kernel: HWKernel) -> list:
        """Get test data generators based on interfaces."""
        generators = []
        
        for interface in hw_kernel.interfaces:
            if interface.get('dataflow_type') == 'INPUT':
                generators.append({
                    'interface_name': interface['name'],
                    'data_type': 'random_tensor',
                    'shape_hint': '[1, 8, 32, 32]'  # Default shape
                })
            elif interface.get('dataflow_type') == 'WEIGHT':
                generators.append({
                    'interface_name': interface['name'],
                    'data_type': 'weight_tensor',
                    'shape_hint': '[8, 8]'  # Default weight shape
                })
        
        return generators