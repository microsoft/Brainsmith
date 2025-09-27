#!/usr/bin/env python3
"""Minimal example of an elementwise operator using AutoHWCustomOp."""

from brainsmith.core.finn.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.core.dataflow import KernelSchema, InputSchema, OutputSchema


class ElementwiseAddSchema(KernelSchema):
    """Schema for elementwise addition."""
    
    def __init__(self):
        super().__init__("elementwise_add")
        
        # Two inputs with identical tiling
        self.add_input(InputSchema(
            name="a",
            block_tiling=[":", "CHANNELS", ":", ":"],
            stream_tiling=[1, "PE", 1, 1]
        ))
        
        self.add_input(InputSchema(
            name="b", 
            block_tiling=[":", "CHANNELS", ":", ":"],
            stream_tiling=[1, "PE", 1, 1]
        ))
        
        # Output matches inputs
        self.add_output(OutputSchema(
            name="output",
            block_tiling=[":", "CHANNELS", ":", ":"]
        ))


class ElementwiseAddOp(AutoHWCustomOp):
    """Simple elementwise addition operator.
    
    This minimal example shows:
    - Basic schema definition
    - Automatic model creation
    - How the base class handles most complexity
    """
    
    kernel_schema = ElementwiseAddSchema()
    
    def get_nodeattr_types(self):
        """Just need to define the tiling parameters."""
        attrs = super().get_nodeattr_types()
        attrs.update({
            "CHANNELS": ('i', True, 0),
            "PE": ('i', False, 1),
        })
        return attrs
    
    def execute_node(self, context, graph):
        """Simple elementwise add."""
        model = self.get_kernel_model()
        
        # Get inputs
        a = context.get_tensor(self.onnx_node.input[0])
        b = context.get_tensor(self.onnx_node.input[1])
        
        # Add them
        result = a + b
        
        # Store output
        context.set_tensor(self.onnx_node.output[0], result)
    
    def bram_estimation(self):
        """No weights, no BRAM needed."""
        return 0
    
    def lut_estimation(self):
        """Simple estimate based on parallelism."""
        pe = self.get_nodeattr("PE")
        bits = self.get_kernel_model().inputs[0].datatype.bitwidth()
        return pe * bits * 10  # ~10 LUTs per bit of adder


# Show the pattern
if __name__ == "__main__":
    print("=== Minimal AutoHWCustomOp Example ===")
    print()
    print("Steps:")
    print("1. Define schema with tiling patterns")
    print("2. Create operator with kernel_schema = YourSchema()")
    print("3. Define nodeattr_types for parameters")  
    print("4. Implement execute_node()")
    print("5. Implement resource estimations")
    print()
    print("The base class handles:")
    print("- Creating models from nodeattrs")
    print("- Caching for performance")
    print("- Shape/type inference")
    print("- Folding calculations")
    print("- SDIM configuration")
    print("- All the FINN interface methods")