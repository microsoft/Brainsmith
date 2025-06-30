"""
Matrix Multiplication RTL Backend

Optimized RTL backend implementation for matrix multiplication.
"""

from brainsmith.plugin.decorators import backend
from .matmul import MatMul


@backend(
    name="MatMulRTL",
    kernel="MatMul",
    backend_type="rtl",
    description="Optimized RTL backend for MatMul kernel",
    author="brainsmith-team",
    version="1.0.0"
)
class MatMulRTL(MatMul):
    """
    RTL backend implementation for MatMul.
    
    This provides hand-optimized Verilog for maximum performance
    with custom systolic array architectures.
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize the RTL backend."""
        super().__init__(onnx_node, **kwargs)
    
    def get_nodeattr_types(self):
        """Inherit attributes from base MatMul class."""
        my_attrs = {}
        my_attrs.update(MatMul.get_nodeattr_types(self))
        # Add RTL-specific attributes
        my_attrs.update({
            "systolic_array": ("i", False, 0),  # Use systolic array
            "pipeline_depth": ("i", False, 4),  # Pipeline stages
        })
        return my_attrs
    
    def get_rtl_params(self):
        """Get parameters for RTL generation."""
        return {
            "USE_SYSTOLIC": self.get_nodeattr("systolic_array"),
            "PIPELINE_DEPTH": self.get_nodeattr("pipeline_depth"),
            "DATA_WIDTH": self.get_input_datatype().bitwidth(),
            "WEIGHT_WIDTH": self.get_weight_datatype().bitwidth(),
            "OUTPUT_WIDTH": self.get_output_datatype().bitwidth(),
        }