############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Manual example of ThresholdingOp using AutoHWCustomOp and Kernel Modeling.

This module demonstrates how to create a HWCustomOp implementation manually
using the new AutoHWCustomOp base class and Kernel Modeling system.
It serves as a proof of concept and test case for the integration.
"""

from typing import Dict, Any, Tuple, Optional

# Kernel Modeling imports
from brainsmith.core.dataflow import (
    KernelDefinition,
    InputDefinition,
    OutputDefinition,
    RelationType,
    parameterized_tiles
)
from brainsmith.core.dataflow.base import ParameterBinding
from brainsmith.tools.hw_kernel_gen.data import DatatypeConstraintGroup

# AutoHWCustomOp import
from brainsmith.tools.hw_kernel_gen.auto_hw_custom_op_v2 import AutoHWCustomOp


def build_thresholding_kernel() -> KernelDefinition:
    """
    Build a KernelDefinition for thresholding operation.
    
    Thresholding compares input values against thresholds and outputs
    binary or multi-bit results. This definition captures:
    - Input with flexible INT/UINT datatypes (4-32 bits)
    - Output with narrow datatypes (1-8 bits)
    - Channel-wise parallelism
    
    Returns:
        KernelDefinition configured for thresholding
    """
    # Create kernel definition
    kernel_def = KernelDefinition(name="thresholding")
    
    # Add input interface
    # Supports signed and unsigned integers from 4 to 32 bits
    kernel_def.add_input(InputDefinition(
        name="input",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 4, 32),
            DatatypeConstraintGroup("UINT", 4, 32)
        ],
        block_dims_expr=parameterized_tiles("NumChannels"),
        optional=False
    ))
    
    # Add output interface
    # Typically 1-bit for binary thresholding, up to 8 bits for multi-threshold
    kernel_def.add_output(OutputDefinition(
        name="output",
        datatype_constraints=[
            DatatypeConstraintGroup("UINT", 1, 8)
        ],
        block_dims_expr=parameterized_tiles("NumChannels")
    ))
    
    # Add relationship - output shape matches input shape
    kernel_def.add_relationship(
        source_name="input",
        target_name="output",
        relationship_type=RelationType.EQUAL,
        description="Input and output have same dimensions"
    )
    
    return kernel_def


class ThresholdingOp(AutoHWCustomOp):
    """
    Thresholding operation implemented using AutoHWCustomOp.
    
    This class demonstrates manual implementation of a HWCustomOp
    using the Kernel Modeling system. It shows how to:
    1. Build a KernelDefinition in the constructor
    2. Extend node attributes for operation-specific parameters
    3. Override methods for custom behavior
    
    Attributes:
        NumChannels: Number of channels to process
        numSteps: Number of threshold steps (for multi-thresholding)
        ActVal: Activation value for multi-bit outputs
    """
    
    def __init__(self, onnx_node, **kwargs):
        """
        Initialize ThresholdingOp with kernel definition.
        
        Args:
            onnx_node: ONNX node for this operation
            **kwargs: Additional arguments for HWCustomOp
        """
        # Build kernel definition
        kernel_def = build_thresholding_kernel()
        
        # Initialize parent with kernel definition
        super().__init__(onnx_node, kernel_def, **kwargs)
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """
        Define node attributes for thresholding operation.
        
        Extends parent attributes with thresholding-specific parameters.
        
        Returns:
            Dictionary of attribute definitions
        """
        # Get parent attributes (includes datatype and SDIM attributes)
        attrs = super().get_nodeattr_types()
        
        # Add thresholding-specific attributes
        attrs.update({
            # Required: number of channels must match tensor dimensions
            "NumChannels": ("i", True, 0),
            
            # Optional: number of threshold steps (default 1 for binary)
            "numSteps": ("i", False, 1),
            
            # Optional: activation value for multi-bit outputs
            "ActVal": ("f", False, 1.0),
            
            # Optional: memory style for threshold storage
            "mem_mode": ("s", False, "const", {"const", "external"}),
            
            # Optional: runtime writeable thresholds
            "runtime_writeable_weights": ("i", False, 0),
        })
        
        return attrs
    
    def _extract_input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract input specifications for thresholding operation.
        
        For thresholding, we have a single input interface that processes
        data channel-wise. Shape is determined by NumChannels parameter.
        
        Returns:
            Dictionary with "input" interface specifications
        """
        # Get shape from node attributes
        num_channels = self.get_nodeattr("NumChannels")
        if num_channels is None or num_channels <= 0:
            raise ValueError("NumChannels must be specified and positive")
        
        # Assume batch size of 1 for hardware
        shape = (1, num_channels)
        
        # Get datatype using helper
        dtype = self._get_interface_datatype("input", is_input=True)
        
        return {
            "input": (shape, dtype)
        }
    
    def _extract_output_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract output specifications for thresholding operation.
        
        Output shape matches input shape for thresholding, but datatype
        may be different (often narrower for threshold outputs).
        
        Returns:
            Dictionary with "output" interface specifications
        """
        # Output shape matches input shape
        num_channels = self.get_nodeattr("NumChannels")
        shape = (1, num_channels)
        
        # Get output datatype using helper
        dtype = self._get_interface_datatype("output", is_input=False)
        
        return {
            "output": (shape, dtype)
        }
    
    def _extract_parameter_binding(self) -> Optional[ParameterBinding]:
        """
        Extract parameters needed by the KernelDefinition.
        
        For thresholding, we need:
        - NumChannels: Used in parameterized_tiles for block dimensions
        
        Returns:
            ParameterBinding with NumChannels parameter
        """
        num_channels = self.get_nodeattr("NumChannels")
        if num_channels is None:
            raise ValueError("NumChannels attribute is required for ThresholdingOp")
        
        return ParameterBinding({"NumChannels": num_channels})
    
    def execute_node(self, context, graph):
        """
        Execute thresholding in simulation.
        
        This is a simplified implementation for testing.
        Real implementation would perform actual thresholding.
        
        Args:
            context: Execution context with tensors
            graph: ONNX graph
        """
        node = self.onnx_node
        
        if len(node.input) > 0 and len(node.output) > 0:
            if node.input[0] in context:
                # Get input tensor
                input_tensor = context[node.input[0]]
                
                # For now, simple pass-through
                # Real implementation would threshold against stored values
                output_tensor = input_tensor.copy()
                
                # Store in context
                context[node.output[0]] = output_tensor
    
    def bram_estimation(self) -> int:
        """
        Estimate BRAM usage for threshold storage.
        
        Returns:
            Number of BRAM blocks needed
        """
        # Get base estimation from parent
        base_bram = super().bram_estimation()
        
        # Add threshold storage requirements
        num_channels = self.get_nodeattr("NumChannels")
        num_steps = self.get_nodeattr("numSteps")
        
        if num_channels > 0 and num_steps > 0:
            # Each threshold is same width as input datatype
            try:
                input_dtype = self.get_input_datatype(0)
                thresh_bits = input_dtype.bitwidth()
                
                # Total threshold memory
                total_thresh_bits = num_channels * num_steps * thresh_bits
                
                # Convert to BRAM blocks (18Kb each)
                bram_bits = 18 * 1024
                thresh_bram = (total_thresh_bits + bram_bits - 1) // bram_bits
                
                return base_bram + thresh_bram
            except:
                pass
        
        return base_bram
    
    def get_exp_cycles(self) -> int:
        """
        Get expected cycles for thresholding.
        
        Returns:
            Expected execution cycles
        """
        # Try parent implementation first
        base_cycles = super().get_exp_cycles()
        
        # For thresholding, cycles depend on folding
        try:
            # Get folded output shape
            folded_shape = self.get_folded_output_shape(0)
            
            # Cycles = number of folded output elements
            # (each folded element takes one cycle)
            if len(folded_shape) >= 2:
                # Product of num_blocks dimensions
                num_blocks_dims = len(folded_shape) // 2
                cycles = 1
                for i in range(num_blocks_dims):
                    cycles *= folded_shape[i]
                return cycles
        except:
            pass
        
        return base_cycles
    
    def verify_node(self) -> list:
        """
        Verify thresholding-specific configuration.
        
        Returns:
            List of verification messages
        """
        # Start with parent verification
        messages = super().verify_node()
        
        # Check NumChannels
        num_channels = self.get_nodeattr("NumChannels")
        if num_channels > 0:
            messages.append(f"✓ NumChannels set to {num_channels}")
        else:
            messages.append("✗ NumChannels must be > 0")
        
        # Check numSteps
        num_steps = self.get_nodeattr("numSteps")
        if num_steps > 0:
            messages.append(f"✓ numSteps set to {num_steps}")
        else:
            messages.append("✗ numSteps must be > 0")
        
        # Check output datatype compatibility
        try:
            output_dtype = self.get_output_datatype(0)
            output_bits = output_dtype.bitwidth()
            
            # For numSteps thresholds, need at least log2(numSteps+1) bits
            import math
            required_bits = math.ceil(math.log2(num_steps + 1))
            
            if output_bits >= required_bits:
                messages.append(
                    f"✓ Output datatype {output_dtype} sufficient for "
                    f"{num_steps} threshold steps"
                )
            else:
                messages.append(
                    f"✗ Output datatype {output_dtype} has {output_bits} bits, "
                    f"need at least {required_bits} for {num_steps} steps"
                )
        except Exception as e:
            messages.append(f"✗ Could not verify output datatype: {e}")
        
        return messages


# Example usage function
def create_thresholding_node_example():
    """
    Example of how to create and configure a thresholding node.
    
    This would typically be done by FINN during graph construction.
    """
    # Mock ONNX node class for demonstration
    class MockONNXNode:
        def __init__(self):
            self.name = "Threshold_0"
            self.op_type = "Thresholding"
            self.input = ["input_tensor"]
            self.output = ["output_tensor"]
            self.attribute = []
            
        def add_attribute(self, name, value):
            # Simplified attribute storage
            attr = type('obj', (object,), {'name': name, 'value': value})
            self.attribute.append(attr)
    
    # Create mock node
    node = MockONNXNode()
    
    # Add attributes
    node.add_attribute("backend", "fpgadataflow")
    node.add_attribute("NumChannels", 64)
    node.add_attribute("numSteps", 1)  # Binary thresholding
    node.add_attribute("input_dtype", "UINT8")
    node.add_attribute("output_dtype", "UINT1")
    node.add_attribute("SIMD", 8)  # Process 8 channels per cycle
    
    # Create operation
    op = ThresholdingOp(node)
    
    return op


if __name__ == "__main__":
    # Demonstrate the implementation
    print("ThresholdingOp Manual Implementation Example")
    print("=" * 50)
    
    # Create example
    op = create_thresholding_node_example()
    
    # Verify configuration
    print("\nVerification Results:")
    for msg in op.verify_node():
        print(f"  {msg}")
    
    print("\nNode Attributes:")
    for attr_name, attr_def in op.get_nodeattr_types().items():
        if attr_name in ["NumChannels", "numSteps", "input_dtype", "output_dtype", "SIMD"]:
            print(f"  {attr_name}: {attr_def}")