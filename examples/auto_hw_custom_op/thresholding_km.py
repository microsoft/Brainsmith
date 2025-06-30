############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Thresholding operation implemented using AutoHWCustomOp and Kernel Modeling.

This module demonstrates how to implement FINN's Thresholding operation using
the modern AutoHWCustomOp base class and Kernel Modeling system. It shows:
- How to define kernel operations with streaming parallelism
- Integration with FINN's datatype system
- Proper weight (threshold) handling
- Folded shape calculations using SDIM
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

# Kernel Modeling imports
from brainsmith.core.dataflow import (
    KernelDefinition,
    InputDefinition,
    OutputDefinition,
    RelationType
)
from brainsmith.core.dataflow.base import ParameterBinding
from brainsmith.tools.hw_kernel_gen.data import DatatypeConstraintGroup

# AutoHWCustomOp import
from brainsmith.tools.hw_kernel_gen.auto_hw_custom_op_v2 import AutoHWCustomOp


def build_thresholding_kernel() -> KernelDefinition:
    """
    Build a KernelDefinition for thresholding operation.
    
    Thresholding compares input values against per-channel thresholds and
    outputs quantized results. This definition captures:
    - Input data stream with NumChannels
    - Threshold weights (NumChannels × numSteps)
    - Output with same shape as input but potentially different datatype
    - PE parallelism for processing multiple channels per cycle
    
    Returns:
        KernelDefinition configured for thresholding
    """
    kernel_def = KernelDefinition(name="thresholding_km")
    
    # Input data interface
    # Note: Only the last dimension (NumChannels) can be tiled
    def input_tiling(tensor_dims, params, config):
        # Full tensor for all dimensions except the last
        tiles = list(tensor_dims[:-1])  # Keep all dimensions as-is
        # For NumChannels dimension, use the parameter value if available
        tiles.append(params.get("NumChannels", tensor_dims[-1]))
        return tuple(tiles)
    
    kernel_def.add_input(InputDefinition(
        name="input",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 4, 32),
            DatatypeConstraintGroup("UINT", 4, 32),
            DatatypeConstraintGroup("BIPOLAR", 1, 1)  # Support bipolar inputs
        ],
        block_dims_expr=input_tiling,
        optional=False
    ))
    
    # Threshold weights interface
    # Note: numSteps defines the number of threshold levels
    def threshold_tiling(tensor_dims, params, config):
        # For thresholds, we tile based on NumChannels and numSteps
        return (params.get("NumChannels", tensor_dims[0]), 
                params.get("numSteps", tensor_dims[1]))
    
    kernel_def.add_input(InputDefinition(
        name="thresholds",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 8, 32),
            DatatypeConstraintGroup("UINT", 8, 32)
        ],
        block_dims_expr=threshold_tiling,
        optional=False,
        is_weight=True  # Mark as weight input for FINN attribute mapping
    ))
    
    # Output interface
    # Output shape matches input shape
    def output_tiling(tensor_dims, params, config):
        # Same as input: full tensor for all dimensions except the last
        tiles = list(tensor_dims[:-1])  # Keep all dimensions as-is
        tiles.append(params.get("NumChannels", tensor_dims[-1]))
        return tuple(tiles)
    
    kernel_def.add_output(OutputDefinition(
        name="output",
        datatype_constraints=[
            DatatypeConstraintGroup("UINT", 1, 8),
            DatatypeConstraintGroup("BIPOLAR", 1, 1)  # Support bipolar outputs
        ],
        block_dims_expr=output_tiling
    ))
    
    # Add relationship for SDIM propagation between inputs
    kernel_def.add_relationship(
        source_name="input",
        target_name="thresholds",
        relationship_type=RelationType.EQUAL,
        source_dim=-1,  # Last dimension (NumChannels) of input
        target_dim=0,    # First dimension (NumChannels) of thresholds
        description="Input channels must match threshold channels for SDIM propagation"
    )
    
    return kernel_def


class ThresholdingHWCustomOp(AutoHWCustomOp):
    """
    Thresholding operation using AutoHWCustomOp and Kernel Modeling.
    
    This class implements FINN's Thresholding operation with modern
    Kernel Modeling system features:
    - SDIM-based parallelism (replacing PE)
    - Automatic shape and width calculations via KernelModel
    - Proper threshold tensor handling
    - Support for runtime writeable weights
    
    Key improvements over original:
    - Cleaner separation of definition vs runtime configuration
    - SDIM architecture for streaming dimensions
    - Better integration with datatype constraints
    """
    
    def __init__(self, onnx_node, **kwargs):
        """
        Initialize ThresholdingHWCustomOp with kernel definition.
        
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
        
        Returns:
            Dictionary of attribute definitions
        """
        # Get parent attributes (includes datatype and legacy attributes)
        attrs = super().get_nodeattr_types()
        
        # Add thresholding-specific attributes
        attrs.update({
            # Required: number of channels
            "NumChannels": ("i", True, 0),
            
            # Required: number of threshold steps
            "numSteps": ("i", True, 1),
            
            # Optional: PE parallelism (will be mapped to SDIM)
            "PE": ("i", False, 1),
            
            # Optional: activation bias value
            "ActVal": ("i", False, 0),
            
            # Optional: runtime writeable weights
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            
            # Optional: input shape for higher dimensions
            "numInputVectors": ("ints", False, [1]),
        })
        
        return attrs
    
    def _extract_input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract input specifications for thresholding.
        
        Returns:
            Dictionary with input interface specifications
        """
        # Get parameters
        num_channels = self.get_nodeattr("NumChannels")
        num_steps = self.get_nodeattr("numSteps")
        num_input_vecs = list(self.get_nodeattr("numInputVectors"))
        
        if num_channels is None or num_channels <= 0:
            raise ValueError("NumChannels must be specified and positive")
        if num_steps is None or num_steps <= 0:
            raise ValueError("numSteps must be specified and positive")
        
        # Input data shape
        input_shape = tuple(num_input_vecs + [num_channels])
        input_dtype = self._get_interface_datatype("input", is_input=True)
        
        # Threshold shape
        thresh_shape = (num_channels, num_steps)
        thresh_dtype = self._get_interface_datatype("thresholds", is_input=True)
        
        return {
            "input": (input_shape, input_dtype),
            "thresholds": (thresh_shape, thresh_dtype)
        }
    
    def _extract_output_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract output specifications for thresholding.
        
        Returns:
            Dictionary with output interface specifications
        """
        # Output shape matches input shape
        num_channels = self.get_nodeattr("NumChannels")
        num_input_vecs = list(self.get_nodeattr("numInputVectors"))
        
        output_shape = tuple(num_input_vecs + [num_channels])
        output_dtype = self._get_interface_datatype("output", is_input=False)
        
        return {
            "output": (output_shape, output_dtype)
        }
    
    def _extract_parameter_binding(self) -> Optional[ParameterBinding]:
        """
        Extract parameters needed by the KernelDefinition.
        
        Returns:
            ParameterBinding with required parameters
        """
        params = {}
        
        # NumChannels is used in parameterized_tiles
        num_channels = self.get_nodeattr("NumChannels")
        if num_channels is not None:
            params["NumChannels"] = num_channels
        
        # numSteps is used for threshold dimensions
        num_steps = self.get_nodeattr("numSteps")
        if num_steps is not None:
            params["numSteps"] = num_steps
        
        return ParameterBinding(params) if params else None
    
    def _apply_legacy_attributes(self):
        """
        Apply legacy attribute mappings for Thresholding.
        
        Maps:
        - PE → SDIM for the input channel dimension
        - numInputVectors → already handled in shape extraction
        
        Note: The relationship between input and thresholds ensures
        that PE is automatically propagated to the thresholds interface.
        """
        # Get PE value and apply as SDIM
        pe = self.get_nodeattr("PE")
        if pe and pe > 0:
            # Apply PE to the channel dimension of input
            # Since input shape is [..., NumChannels], PE applies to last dim
            # The relationship will propagate this to thresholds automatically
            sdim_config = {"input": pe}
            self._kernel_model.configure_sdim(sdim_config)
            self._sdim_config.update(sdim_config)
    
    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """
        Convert threshold matrix for hardware compatibility.
        
        Args:
            orig_thres_matrix: Original threshold tensor
            
        Returns:
            Reformatted tensor suitable for hardware
        """
        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        tmem = num_channels // pe
        
        assert num_channels % pe == 0, "NumChannels must be divisible by PE"
        assert orig_thres_matrix.ndim == 2, "Threshold matrix must be 2D"
        
        n_thres_steps = orig_thres_matrix.shape[1]
        assert n_thres_steps == self.get_nodeattr("numSteps"), "Mismatch in threshold steps"
        
        # Ensure unsigned thresholds for unsigned inputs
        if not self.get_input_datatype(0).signed():
            assert (orig_thres_matrix >= 0).all()
        
        ret = orig_thres_matrix
        
        # Duplicate single threshold across channels if needed
        if ret.shape[0] == 1:
            ret = np.tile(ret, (num_channels, 1))
        
        assert ret.shape[0] == num_channels
        
        # Distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        
        return ret.reshape(1, pe, tmem, n_thres_steps)
    
    def execute_node(self, context, graph):
        """
        Execute thresholding in simulation.
        
        Args:
            context: Execution context with tensors
            graph: ONNX graph
        """
        node = self.onnx_node
        inp_values = context[node.input[0]]
        th_val = context[node.input[1]]
        out_bias = self.get_nodeattr("ActVal")
        
        # Handle 4D tensors (NHWC format)
        is_4d = len(inp_values.shape) == 4
        if is_4d:
            inp_values = np.transpose(inp_values, (0, 3, 1, 2))
        
        # Apply multithreshold function
        y = multithreshold(inp_values, th_val, out_bias=out_bias)
        
        if is_4d:
            y = y.transpose(0, 2, 3, 1)
        
        # Handle bipolar output
        act = DataType[self.get_nodeattr("outputDataType")]
        if act == DataType["BIPOLAR"]:
            y = 2 * y - 1
        
        context[node.output[0]] = y
    
    def calc_tmem(self):
        """Calculate and return TMEM (threshold memory depth)."""
        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return num_channels // pe
    
    def minimize_accumulator_width(self, model):
        """
        Minimize threshold datatype width.
        
        Since thresholding doesn't have accumulators in the traditional sense,
        this method minimizes the threshold datatype instead.
        """
        idt = self.get_input_datatype(0)
        if str(idt).startswith("FLOAT") or self.get_nodeattr("weightDataType").startswith("FLOAT"):
            return DataType[self.get_nodeattr("weightDataType")]
        
        thresholds = model.get_initializer(self.onnx_node.input[1])
        threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
        
        min_threshold = thresholds.min()
        max_threshold = thresholds.max()
        min_input = idt.min()
        max_input = idt.max()
        
        # Get range required by threshold values
        tdt_min = min(min_input, min_threshold)
        tdt_max = max(max_input, max_threshold)
        
        if tdt_min < 0:
            if abs(tdt_min) > tdt_max:
                tdt = DataType.get_smallest_possible(tdt_min)
            else:
                tdt = DataType.get_smallest_possible(-tdt_max - 1)
        else:
            tdt = DataType.get_smallest_possible(tdt_max)
        
        assert np.vectorize(tdt.allowed)(threshold_tensor).all(), \
            f"Thresholds can't be expressed with type {tdt}"
        
        self.set_nodeattr("weightDataType", tdt.name)
        model.set_tensor_datatype(self.onnx_node.input[1], tdt)
        
        return tdt
    
    def verify_node(self) -> List[str]:
        """
        Verify thresholding-specific configuration.
        
        Returns:
            List of verification messages
        """
        # Start with parent verification
        messages = super().verify_node()
        
        # Check NumChannels
        num_channels = self.get_nodeattr("NumChannels")
        if num_channels and num_channels > 0:
            messages.append(f"✓ NumChannels set to {num_channels}")
        else:
            messages.append("✗ NumChannels must be > 0")
        
        # Check numSteps
        num_steps = self.get_nodeattr("numSteps")
        if num_steps and num_steps > 0:
            messages.append(f"✓ numSteps set to {num_steps}")
        else:
            messages.append("✗ numSteps must be > 0")
        
        # Check PE divisibility
        pe = self.get_nodeattr("PE")
        if pe and num_channels and num_channels % pe == 0:
            messages.append(f"✓ PE={pe} divides NumChannels={num_channels}")
        elif pe and num_channels:
            messages.append(f"✗ PE={pe} does not divide NumChannels={num_channels}")
        
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


# Example usage
def create_thresholding_example():
    """Create an example thresholding node."""
    # Mock ONNX node
    class MockONNXNode:
        def __init__(self):
            self.name = "Threshold_KM_0"
            self.op_type = "ThresholdingKM"
            self.input = ["input_tensor", "threshold_tensor"]
            self.output = ["output_tensor"]
            self.attribute = []
            
        def add_attribute(self, name, value):
            attr = type('obj', (object,), {
                'name': name,
                's': value if isinstance(value, str) else "",
                'i': value if isinstance(value, int) else 0,
                'ints': value if isinstance(value, list) else []
            })
            self.attribute.append(attr)
    
    # Create node
    node = MockONNXNode()
    
    # Add attributes
    node.add_attribute("backend", "fpgadataflow")
    node.add_attribute("NumChannels", 64)
    node.add_attribute("numSteps", 3)  # 4-level quantization
    node.add_attribute("PE", 16)  # Process 16 channels per cycle
    node.add_attribute("inputDataType", "UINT8")
    node.add_attribute("weightDataType", "UINT8")  # Threshold datatype
    node.add_attribute("outputDataType", "UINT2")  # 2-bit output
    node.add_attribute("ActVal", 0)
    node.add_attribute("numInputVectors", [1, 32, 32])  # E.g., image data
    
    # Create operation
    op = ThresholdingHWCustomOp(node)
    
    return op


if __name__ == "__main__":
    print("Thresholding with Kernel Modeling Example")
    print("=" * 50)
    
    # Create example
    op = create_thresholding_example()
    
    # Verify configuration
    print("\nVerification Results:")
    for msg in op.verify_node():
        print(f"  {msg}")
    
    print("\nShape Information:")
    print(f"  Normal input shape: {op.get_normal_input_shape(0)}")
    print(f"  Normal output shape: {op.get_normal_output_shape(0)}")
    
    # Force model creation to show folded shapes
    op._ensure_kernel_model()
    print(f"  Folded input shape: {op.get_folded_input_shape(0)}")
    print(f"  Folded output shape: {op.get_folded_output_shape(0)}")
    
    print("\nStream Widths:")
    print(f"  Input stream width: {op.get_instream_width(0)} bits")
    print(f"  Output stream width: {op.get_outstream_width(0)} bits")
    
    print("\nPerformance:")
    print(f"  Expected cycles: {op.get_exp_cycles()}")
    print(f"  TMEM depth: {op.calc_tmem()}")