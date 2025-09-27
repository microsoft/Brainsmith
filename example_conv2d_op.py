#!/usr/bin/env python3
"""Example Conv2D operator using AutoHWCustomOp base class."""

from qonnx.core.datatype import DataType
from brainsmith.core.finn.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.core.dataflow import KernelSchema, InputSchema, OutputSchema


# First, define the kernel schema
class Conv2DSchema(KernelSchema):
    """Schema for a Conv2D kernel."""
    
    def __init__(self):
        super().__init__("conv2d")
        
        # Define input interfaces
        self.add_input(InputSchema(
            name="input",
            block_tiling=[":", "CHANNELS", ":", ":"],  # NCHW format - don't tile spatial dims
            stream_tiling=[1, "PE", 1, 1],  # Stream along channels
            datatype_attr="inputDataType",  # Custom attribute name
            is_weight=False
        ))
        
        self.add_input(InputSchema(
            name="weight", 
            block_tiling=["OC", "CHANNELS", ":", ":"],  # OIHW format - full kernel
            stream_tiling=["SIMD", "PE", 1, 1],  # Stream along both O and I
            is_weight=True,
            optional=False
        ))
        
        self.add_input(InputSchema(
            name="bias",
            block_tiling=["OC"],  # Just output channels
            stream_tiling=["SIMD"],  # Stream along channels
            is_weight=True,
            optional=True  # Bias is optional
        ))
        
        # Define output interface
        self.add_output(OutputSchema(
            name="output",
            block_tiling=[":", "OC", ":", ":"],  # Output shape
            datatype_attr="outputDataType"
        ))


# Now define the operator
class Conv2DHWCustomOp(AutoHWCustomOp):
    """Hardware Conv2D operator with automatic model management.
    
    This operator demonstrates:
    - Multiple inputs (data, weight, optional bias)
    - Parameter-based tiling templates
    - Custom datatype attributes
    - Resource estimation methods
    """
    
    # Set the kernel schema
    kernel_schema = Conv2DSchema()
    
    def get_nodeattr_types(self):
        """Define node attributes for Conv2D."""
        attrs = super().get_nodeattr_types()
        
        # Add Conv2D specific attributes
        conv_attrs = {
            # Kernel dimensions
            "KERNEL": ('i', True, 0),  # Kernel size (square)
            "STRIDE": ('i', False, 1),  # Stride (default 1)
            "PADDING": ('i', False, 0),  # Padding (default 0)
            "DILATION": ('i', False, 1),  # Dilation (default 1)
            
            # Channel dimensions
            "CHANNELS": ('i', True, 0),  # Input channels
            "OC": ('i', True, 0),  # Output channels
            
            # Parallelism
            "PE": ('i', False, 1),  # Input channel parallelism
            "SIMD": ('i', False, 1),  # Output channel parallelism
            
            # Datatypes
            "inputDataType": ('s', True, ""),
            "weightDataType": ('s', True, ""),
            "outputDataType": ('s', True, ""),
            
            # Performance
            "clock_freq_mhz": ('f', False, 100.0),
        }
        attrs.update(conv_attrs)
        
        return attrs
    
    def execute_node(self, context, graph):
        """Execute Conv2D in simulation.
        
        This would implement the actual convolution logic for simulation.
        """
        # Get the kernel model for execution
        model = self.get_kernel_model()
        
        # Access inputs
        input_tensor = context.get_tensor(self.onnx_node.input[0])
        weight_tensor = context.get_tensor(self.onnx_node.input[1])
        
        # Check for optional bias
        bias_tensor = None
        if len(self.onnx_node.input) > 2 and self.onnx_node.input[2]:
            bias_tensor = context.get_tensor(self.onnx_node.input[2])
        
        # TODO: Implement actual convolution logic here
        # For now, just create dummy output
        output_shape = model.outputs[0].tensor_dims
        output_tensor = np.zeros(output_shape)
        
        # Store result
        context.set_tensor(self.onnx_node.output[0], output_tensor)
    
    def bram_estimation(self):
        """Estimate BRAM usage for Conv2D."""
        model = self.get_kernel_model()
        
        # Get weight interface
        weight_input = model.get_input("weight")
        if weight_input:
            # Calculate weight storage needs
            weight_bits = prod(weight_input.tensor_dims) * weight_input.datatype.bitwidth()
            
            # Assume 36Kb BRAMs
            bram_bits = 36 * 1024
            brams_needed = math.ceil(weight_bits / bram_bits)
            
            return brams_needed
        
        return 0
    
    def lut_estimation(self):
        """Estimate LUT usage for Conv2D."""
        model = self.get_kernel_model()
        
        # Get parallelism factors
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        
        # Rough estimate: 100 LUTs per MAC unit
        macs = pe * simd
        luts = macs * 100
        
        return luts
    
    def get_exp_cycles(self):
        """Get expected cycles - overrides base to add conv2d specifics."""
        model = self.get_kernel_model()
        
        # Base initiation interval from streaming
        base_ii = model.initiation_interval
        
        # Add any Conv2D specific pipeline delays
        stride = self.get_nodeattr("STRIDE")
        if stride > 1:
            # Strided conv may need extra cycles
            base_ii = base_ii * stride
        
        return base_ii
    
    def infer_node_datatype(self):
        """Infer node datatypes - required by HWCustomOp."""
        # Set output datatype based on outputDataType attribute
        output_dtype = self.get_nodeattr("outputDataType")
        if output_dtype:
            self.set_nodeattr("outputDataType", output_dtype)
    
    def infer_module_name(self):
        """Return RTL module name."""
        return "Conv2D_rtl"


# Example usage
def demo_conv2d():
    """Demonstrate using the Conv2D operator."""
    
    print("=== Conv2D AutoHWCustomOp Example ===\n")
    
    # Show the schema
    schema = Conv2DSchema()
    print(f"Kernel: {schema.name}")
    print(f"Inputs: {[inp.name for inp in schema.inputs]}")
    print(f"Outputs: {[out.name for out in schema.outputs]}")
    
    print("\nKey features demonstrated:")
    print("- Multiple inputs with optional bias")
    print("- Parameter-based tiling (KERNEL, CHANNELS, PE, SIMD)")
    print("- Custom datatype attributes")
    print("- Resource estimation methods")
    print("- Cached kernel model with refresh via transforms")
    
    print("\nUsage pattern:")
    print("1. Define schema with tiling templates")
    print("2. Set kernel_schema class attribute")
    print("3. Define nodeattr types")
    print("4. Implement execute_node() for simulation")
    print("5. Implement resource estimation methods")
    print("6. Use get_kernel_model() to access cached model")
    
    print("\nThe kernel model is automatically:")
    print("- Created from current nodeattrs")
    print("- Cached for performance")
    print("- Refreshed by RefreshKernelModels transform")
    print("- Used to infer shapes and types")


import numpy as np
import math

def prod(shape):
    """Product of shape elements."""
    result = 1
    for x in shape:
        result *= x
    return result

if __name__ == "__main__":
    demo_conv2d()