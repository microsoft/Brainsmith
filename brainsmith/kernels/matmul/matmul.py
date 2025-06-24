"""
Matrix Multiplication Kernel

Kernel definition for matrix multiplication operations.
"""

from brainsmith.plugin import kernel


@kernel(
    name="MatMul",
    description="Matrix multiplication kernel for linear algebra operations",
    author="brainsmith-team",
    version="1.0.0"
)
class MatMul:
    """
    MatMul kernel definition.
    
    This class defines the interface and parameters for matrix multiplication
    operations. The actual implementation is provided by backends.
    """
    
    def get_nodeattr_types(self):
        """Define the attributes/parameters for this kernel."""
        return {
            # Matrix dimensions
            "M": ("i", True, ""),  # Output rows
            "K": ("i", True, ""),  # Inner dimension
            "N": ("i", True, ""),  # Output columns
            
            # Parallelization parameters
            "PE": ("i", False, 1),  # Processing elements
            "SIMD": ("i", False, 1),  # SIMD width
            
            # Datatype parameters
            "inputDataType": ("s", True, ""),  # Input data type
            "weightDataType": ("s", True, ""),  # Weight data type
            "outputDataType": ("s", True, ""),  # Output data type
        }
    
    def get_input_datatype(self):
        """Get the input data type."""
        return self.get_nodeattr("inputDataType")
    
    def get_weight_datatype(self):
        """Get the weight data type."""
        return self.get_nodeattr("weightDataType")
    
    def get_output_datatype(self):
        """Get the output data type."""
        return self.get_nodeattr("outputDataType")