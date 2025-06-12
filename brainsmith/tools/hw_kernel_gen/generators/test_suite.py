"""
Test suite generator.

Based on hw_kernel_gen_simple pattern with template compatibility
for test suite generation.
"""

from pathlib import Path
from .base import GeneratorBase
from brainsmith.dataflow.core.kernel_metadata import KernelMetadata


class TestSuiteGenerator(GeneratorBase):
    """Test suite generator for comprehensive validation."""
    
    def __init__(self, template_dir: Path = None):
        super().__init__('test_suite.py.j2', template_dir)
    
    def _get_output_filename(self, parsed_data: KernelMetadata) -> str:
        """Get output filename for test suite."""
        return f"test_{parsed_data.name.lower()}.py"
    
    def _get_template_context(self, parsed_data: KernelMetadata) -> dict:
        """Get enhanced template context for test suite generation."""
        context = super()._get_template_context(parsed_data)
        
        # Enhance Interface objects with template-required attributes
        enhanced_interfaces = self._enhance_interfaces_for_template(context['interfaces'])
        context['interfaces'] = enhanced_interfaces
        
        # Also enhance the categorized interface lists
        context['input_interfaces'] = self._enhance_interfaces_for_template(context['input_interfaces'])
        context['output_interfaces'] = self._enhance_interfaces_for_template(context['output_interfaces'])
        context['weight_interfaces'] = self._enhance_interfaces_for_template(context['weight_interfaces'])
        context['config_interfaces'] = self._enhance_interfaces_for_template(context['config_interfaces'])
        context['control_interfaces'] = self._enhance_interfaces_for_template(context['control_interfaces'])
        
        return context
    
    def _enhance_interfaces_for_template(self, interfaces: list) -> list:
        """
        Enhance Interface objects with attributes required by templates.
        
        Templates expect Interface objects to have tensor_dims, block_dims, stream_dims.
        """
        enhanced_interfaces = []
        
        for iface in interfaces:
            # Create a copy-like object with additional attributes
            enhanced_iface = type('EnhancedInterface', (), {})()
            
            # Copy all original attributes
            for attr in dir(iface):
                if not attr.startswith('_'):
                    try:
                        setattr(enhanced_iface, attr, getattr(iface, attr))
                    except:
                        pass  # Skip attributes that can't be copied
            
            # Add BDIM attributes with defaults
            enhanced_iface.tensor_dims = iface.metadata.get('tensor_dims', [1]) if hasattr(iface, 'metadata') else [1]
            enhanced_iface.block_dims = iface.metadata.get('block_dims', [1]) if hasattr(iface, 'metadata') else [1] 
            enhanced_iface.stream_dims = iface.metadata.get('stream_dims', [1]) if hasattr(iface, 'metadata') else [1]
            
            # Add dtype attribute for test suite templates
            if not hasattr(enhanced_iface, 'dtype'):
                enhanced_iface.dtype = type('DataType', (), {
                    'base_types': ['UINT'],
                    'min_bits': 8,
                    'max_bits': 32
                })()
            
            # Add constraints attribute for test suite templates  
            if not hasattr(enhanced_iface, 'constraints'):
                enhanced_iface.constraints = type('Constraints', (), {
                    'parallelism': type('ParallelismConstraint', (), {
                        'default': 1,
                        'min': 1,
                        'max': 64
                    })()
                })()
            
            enhanced_interfaces.append(enhanced_iface)
        
        return enhanced_interfaces