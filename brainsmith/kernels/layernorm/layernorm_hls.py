"""
Layer Normalization HLS Backend

HLS backend implementation for layer normalization.
"""

from brainsmith.core.plugins import backend
from .layernorm import LayerNorm


@backend(
    name="LayerNormHLS", 
    kernel="LayerNorm",
    language="hls",
    default=True,  # Mark as default backend
    description="High-Level Synthesis backend for LayerNorm kernel",
    author="brainsmith-team",
    version="1.0.0"
)
class LayerNormHLS(LayerNorm):
    """
    HLS backend implementation for LayerNorm.
    
    Features a 3-stage pipeline architecture:
    1. Mean calculation
    2. Variance calculation
    3. Normalization
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize the HLS backend."""
        super().__init__(onnx_node, **kwargs)
    
    def get_nodeattr_types(self):
        """Inherit and extend attributes from base LayerNorm class."""
        my_attrs = {}
        my_attrs.update(LayerNorm.get_nodeattr_types(self))
        # Add HLS-specific attributes
        my_attrs.update({
            "pipeline_style": ("s", False, "flp"),  # Pipeline style: flp or stp
            "exec_mode": ("s", False, "rtlsim"),  # Execution mode
        })
        return my_attrs
    
    def global_includes(self):
        """Get HLS include files."""
        return [
            "#include <hls_vector.h>",
            '#include "layernorm.hpp"',
            '#include "bs_utils.hpp"'
        ]
    
    def get_hls_params(self):
        """Get parameters for HLS code generation."""
        idtype = self.get_input_datatype()
        odtype = self.get_output_datatype()
        
        return {
            "SIMD": self.get_nodeattr("SIMD"),
            "W": self.get_nodeattr("ifm_dim")[-1],  # Width dimension
            "epsilon": self.get_nodeattr("epsilon"),
            "TI": idtype.get_hls_datatype_str(),
            "TO": odtype.get_hls_datatype_str(),
        }
    
    def ipgen_extra_includes(self):
        """Add kernel-specific include paths."""
        import os
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(os.path.dirname(kernel_dir), 'utils')
        return f"-I{kernel_dir}/hls -I{utils_dir}"