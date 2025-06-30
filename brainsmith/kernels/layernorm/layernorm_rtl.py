"""
Layer Normalization RTL Backend

Optimized RTL backend implementation for layer normalization.
"""

from brainsmith.plugin.decorators import backend
from .layernorm import LayerNorm


@backend(
    name="LayerNormRTL",
    kernel="LayerNorm",
    backend_type="rtl",
    description="Optimized RTL backend for LayerNorm kernel",
    author="brainsmith-team",
    version="1.0.0"
)
class LayerNormRTL(LayerNorm):
    """
    RTL backend implementation for LayerNorm.
    
    Features:
    - Fully pipelined architecture
    - Configurable precision with fixed-point arithmetic
    - Optional fused operations for transformer blocks
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize the RTL backend."""
        super().__init__(onnx_node, **kwargs)
    
    def get_nodeattr_types(self):
        """Inherit and extend attributes from base LayerNorm class."""
        my_attrs = {}
        my_attrs.update(LayerNorm.get_nodeattr_types(self))
        # Add RTL-specific attributes
        my_attrs.update({
            "fixed_point_position": ("i", False, 8),  # Fixed-point precision
            "use_lut_invsqrt": ("i", False, 1),  # Use LUT for inverse sqrt
            "fuse_with_linear": ("i", False, 0),  # Fuse with following linear
        })
        return my_attrs
    
    def get_rtl_params(self):
        """Get parameters for RTL generation."""
        return {
            "DATA_WIDTH": self.get_input_datatype().bitwidth(),
            "FIXED_POINT": self.get_nodeattr("fixed_point_position"),
            "USE_LUT": self.get_nodeattr("use_lut_invsqrt"),
            "SIMD_WIDTH": self.get_nodeattr("SIMD"),
            "EPSILON": self.get_nodeattr("epsilon"),
        }