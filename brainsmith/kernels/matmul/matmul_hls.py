"""
Matrix Multiplication HLS Backend

HLS backend implementation for matrix multiplication.
"""

from brainsmith.plugin.decorators import backend
from .matmul import MatMul


@backend(
    name="MatMulHLS",
    kernel="MatMul",
    backend_type="hls",
    description="High-Level Synthesis backend for MatMul kernel",
    author="brainsmith-team",
    version="1.0.0"
)
class MatMulHLS(MatMul):
    """
    HLS backend implementation for MatMul.
    
    This backend generates Vivado HLS code for matrix multiplication
    with configurable parallelization and data types.
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize the HLS backend."""
        super().__init__(onnx_node, **kwargs)
    
    def get_nodeattr_types(self):
        """Inherit attributes from base MatMul class."""
        my_attrs = {}
        my_attrs.update(MatMul.get_nodeattr_types(self))
        # Add HLS-specific attributes if needed
        my_attrs.update({
            "hls_config": ("s", False, ""),  # Optional HLS configuration
        })
        return my_attrs
    
    def get_template_params(self):
        """Get parameters for HLS code generation."""
        return {
            "M": self.get_nodeattr("M"),
            "K": self.get_nodeattr("K"), 
            "N": self.get_nodeattr("N"),
            "PE": self.get_nodeattr("PE"),
            "SIMD": self.get_nodeattr("SIMD"),
            "TI": self.get_input_datatype(),
            "TW": self.get_weight_datatype(),
            "TO": self.get_output_datatype(),
        }