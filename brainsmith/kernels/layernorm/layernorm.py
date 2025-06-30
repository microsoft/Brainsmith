"""
Layer Normalization Kernel

Kernel definition for layer normalization operations.
"""

from brainsmith.plugin.core import kernel


@kernel(
    name="LayerNorm",
    description="Layer normalization kernel for transformer models",
    author="brainsmith-team",
    version="1.0.0"
)
class LayerNorm:
    """
    LayerNorm kernel definition.
    
    Supports both standard LayerNorm and RMSNorm variants.
    """
    
    def get_nodeattr_types(self):
        """Define the attributes/parameters for this kernel."""
        return {
            # Normalization parameters
            "epsilon": ("f", False, 1e-5),  # Small constant for numerical stability
            "elementwise_affine": ("i", False, 1),  # Apply learned scale/bias
            
            # Dimension parameters
            "ifm_dim": ("ints", True, []),  # Input feature map dimensions
            "axis": ("i", False, -1),  # Axis to normalize over
            
            # Parallelization
            "SIMD": ("i", False, 1),  # SIMD width for vectorization
            
            # Data types
            "inputDataType": ("s", True, ""),  # Input data type
            "outputDataType": ("s", True, ""),  # Output data type
            
            # Variant selection
            "variant": ("s", False, "layernorm"),  # "layernorm" or "rmsnorm"
        }
    
    def get_input_datatype(self):
        """Get the input data type."""
        return self.get_nodeattr("inputDataType")
    
    def get_output_datatype(self):
        """Get the output data type."""
        return self.get_nodeattr("outputDataType")
    
    def get_folded_input_shape(self):
        """Get the folded input shape based on SIMD."""
        ifm_dim = self.get_nodeattr("ifm_dim")
        simd = self.get_nodeattr("SIMD")
        # Fold the last dimension by SIMD
        folded_shape = list(ifm_dim[:-1]) + [ifm_dim[-1] // simd]
        return folded_shape
    
    def get_folded_output_shape(self):
        """Get the folded output shape (same as folded input)."""
        return self.get_folded_input_shape()