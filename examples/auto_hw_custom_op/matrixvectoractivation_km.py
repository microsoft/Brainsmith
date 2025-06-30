############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
MatrixVectorActivation (MVAU) using AutoHWCustomOp and Kernel Modeling.

This module demonstrates how to implement FINN's MatrixVectorActivation
operation using the modern AutoHWCustomOp base class and Kernel Modeling system.
It shows:
- Matrix multiplication with optional activation
- Multi-dimensional SDIM for SIMD/PE parallelism
- Multiple memory modes (embedded, decoupled, external)
- Binary XNOR operations for quantized networks
- Integration with threshold-based activations
"""

import math
import numpy as np
import warnings
from typing import Dict, Any, Tuple, Optional, List
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.multithreshold import multithreshold
import qonnx.custom_op.general.xnorpopcount as xp
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    interleave_matrix_outer_dim_from_partitions,
)

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


def build_mvau_kernel() -> KernelDefinition:
    """
    Build a KernelDefinition for MatrixVectorActivation operation.
    
    MVAU performs matrix multiplication followed by optional activation:
    - Input vector/matrix multiplication with weight matrix
    - Optional threshold-based activation
    - Support for binary/bipolar operations via XNOR+popcount
    - Configurable parallelism in both input (SIMD) and output (PE) dimensions
    
    Returns:
        KernelDefinition configured for MVAU
    """
    kernel_def = KernelDefinition(name="mvau_km")
    
    # Input data interface (MW features)
    def input_tiling(tensor_dims, params, config):
        # Full tensor for all dimensions except the last (MW)
        tiles = list(tensor_dims[:-1])
        tiles.append(params.get("MW", tensor_dims[-1]))
        return tuple(tiles)
    
    kernel_def.add_input(InputDefinition(
        name="input",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 1, 32),
            DatatypeConstraintGroup("UINT", 1, 32),
            DatatypeConstraintGroup("BINARY", 1, 1),
            DatatypeConstraintGroup("BIPOLAR", 1, 1)
        ],
        block_dims_expr=input_tiling,
        optional=False
    ))
    
    # Weight matrix interface (MW × MH)
    def weight_tiling(tensor_dims, params, config):
        # Tile based on MW and MH parameters
        return (params.get("MW", tensor_dims[0]), 
                params.get("MH", tensor_dims[1]))
    
    kernel_def.add_input(InputDefinition(
        name="weights",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 1, 8),
            DatatypeConstraintGroup("UINT", 1, 8),
            DatatypeConstraintGroup("BINARY", 1, 1),
            DatatypeConstraintGroup("BIPOLAR", 1, 1)
        ],
        block_dims_expr=weight_tiling,
        optional=False,
        is_weight=True  # Mark as weight for FINN attribute mapping
    ))
    
    # Optional threshold interface for activation (MH × n_thres_steps)
    def threshold_tiling(tensor_dims, params, config):
        # Tile based on MH and n_thres_steps
        return (params.get("MH", tensor_dims[0]), 
                params.get("n_thres_steps", tensor_dims[1]))
    
    kernel_def.add_input(InputDefinition(
        name="thresholds",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 8, 32),
            DatatypeConstraintGroup("UINT", 8, 32)
        ],
        block_dims_expr=threshold_tiling,
        optional=True,  # Only needed when noActivation=0
        is_weight=True  # Thresholds are also weights
    ))
    
    # Output interface (MH features)
    def output_tiling(tensor_dims, params, config):
        # Full tensor for all dimensions except the last (MH)
        tiles = list(tensor_dims[:-1])
        tiles.append(params.get("MH", tensor_dims[-1]))
        return tuple(tiles)
    
    kernel_def.add_output(OutputDefinition(
        name="output",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 1, 32),
            DatatypeConstraintGroup("UINT", 1, 32),
            DatatypeConstraintGroup("BINARY", 1, 1),
            DatatypeConstraintGroup("BIPOLAR", 1, 1)
        ],
        block_dims_expr=output_tiling
    ))
    
    # Add relationships for SDIM propagation between inputs
    # Only needed between inputs since outputs don't have configurable SDIM
    kernel_def.add_relationship(
        source_name="input",
        target_name="weights",
        relationship_type=RelationType.EQUAL,
        source_dim=-1,  # Last dimension (MW) of input
        target_dim=0,   # First dimension (MW) of weights
        description="Input features must match weight matrix rows for SDIM propagation"
    )
    
    return kernel_def


class MVAUHWCustomOp(AutoHWCustomOp):
    """
    MatrixVectorActivation using AutoHWCustomOp and Kernel Modeling.
    
    This class implements FINN's MVAU operation with modern features:
    - Multi-dimensional SDIM for SIMD (input) and PE (output) parallelism
    - Automatic accumulator width calculation
    - Support for binary operations via XNOR+popcount
    - Multiple memory modes for weights
    - Optional threshold-based activation
    
    Key improvements over original:
    - Uses SDIM architecture instead of fixed SIMD/PE
    - Better datatype constraint handling
    - Cleaner weight and threshold management
    """
    
    def __init__(self, onnx_node, **kwargs):
        """
        Initialize MVAUHWCustomOp with kernel definition.
        
        Args:
            onnx_node: ONNX node for this operation
            **kwargs: Additional arguments for HWCustomOp
        """
        # Build kernel definition
        kernel_def = build_mvau_kernel()
        
        # Initialize parent with kernel definition
        super().__init__(onnx_node, kernel_def, **kwargs)
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """
        Define node attributes for MVAU operation.
        
        Returns:
            Dictionary of attribute definitions
        """
        # Get parent attributes
        attrs = super().get_nodeattr_types()
        
        # Add MVAU-specific attributes
        attrs.update({
            # Matrix dimensions
            "MW": ("i", True, 0),  # Input features
            "MH": ("i", True, 0),  # Output features
            
            # Parallelism (will be mapped to SDIM)
            "SIMD": ("i", False, 1),  # Input parallelism
            "PE": ("i", False, 1),    # Output parallelism
            
            # Operation configuration
            "noActivation": ("i", False, 0, {0, 1}),  # Skip activation
            "binaryXnorMode": ("i", False, 0, {0, 1}),  # Use XNOR for binary
            "ActVal": ("i", False, 0),  # Activation bias
            
            # Memory configuration
            "mem_mode": (
                "s", False, "internal_decoupled",
                {"internal_embedded", "internal_decoupled", "external"}
            ),
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            
            # Resource configuration
            "resType": ("s", False, "auto", {"auto", "lut", "dsp"}),
            
            # Input shape
            "numInputVectors": ("ints", False, [1]),
            
            # Accumulator datatype (auto-computed)
            "accDataType": ("s", False, "INT32"),
            
            # Optional: number of threshold steps for activation
            "n_thres_steps": ("i", False, 1),
        })
        
        return attrs
    
    def _extract_input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract input specifications for MVAU.
        
        Returns:
            Dictionary with input interface specifications
        """
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        num_input_vecs = list(self.get_nodeattr("numInputVectors"))
        no_act = self.get_nodeattr("noActivation")
        
        if mw is None or mw <= 0:
            raise ValueError("MW must be specified and positive")
        if mh is None or mh <= 0:
            raise ValueError("MH must be specified and positive")
        
        specs = {}
        
        # Input data
        input_shape = tuple(num_input_vecs + [mw])
        input_dtype = self._get_interface_datatype("input", is_input=True)
        specs["input"] = (input_shape, input_dtype)
        
        # Weights
        weight_shape = (mw, mh)
        weight_dtype = self._get_interface_datatype("weights", is_input=True)
        specs["weights"] = (weight_shape, weight_dtype)
        
        # Thresholds (if activation enabled)
        if no_act == 0:
            # Get number of threshold steps
            n_steps = self.get_nodeattr("n_thres_steps")
            if n_steps is None:
                n_steps = 1  # Default to binary threshold
            
            thresh_shape = (mh, n_steps)
            # Try to get threshold datatype, fall back to accumulator type
            try:
                thresh_dtype = self._get_interface_datatype("thresholds", is_input=True)
            except:
                # Use accumulator datatype as default
                thresh_dtype = DataType[self.get_nodeattr("accDataType")]
            
            specs["thresholds"] = (thresh_shape, thresh_dtype)
        
        return specs
    
    def _extract_output_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract output specifications for MVAU.
        
        Returns:
            Dictionary with output interface specifications
        """
        mh = self.get_nodeattr("MH")
        num_input_vecs = list(self.get_nodeattr("numInputVectors"))
        
        output_shape = tuple(num_input_vecs + [mh])
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
        
        # Matrix dimensions
        mw = self.get_nodeattr("MW")
        if mw is not None:
            params["MW"] = mw
        
        mh = self.get_nodeattr("MH")
        if mh is not None:
            params["MH"] = mh
        
        # Threshold steps (if using activation)
        if self.get_nodeattr("noActivation") == 0:
            n_steps = self.get_nodeattr("n_thres_steps")
            if n_steps is not None:
                params["n_thres_steps"] = n_steps
        
        return ParameterBinding(params) if params else None
    
    def _apply_legacy_attributes(self):
        """
        Apply legacy attribute mappings for MVAU.
        
        Maps:
        - SIMD → SDIM for input streaming (MW dimension)
        - PE → SDIM for output streaming (MH dimension)
        - numInputVectors → already handled in shape extraction
        
        For MVAU, SIMD/PE affect both input and weight interfaces:
        - Input uses SIMD for streaming MW features
        - Weights use [SIMD, PE] for 2D streaming
        """
        # Get SIMD and PE values
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        
        sdim_config = {}
        
        # Apply SIMD to input interface
        if simd and simd > 0:
            sdim_config["input"] = simd
        
        # Apply PE to weight interface (affects output dimension)
        if pe and pe > 0:
            # For weights, PE affects the MH dimension (second dim)
            # SIMD affects the MW dimension (first dim)
            if simd and simd > 0:
                sdim_config["weights"] = [simd, pe]
            else:
                sdim_config["weights"] = [1, pe]
        
        if sdim_config:
            self._kernel_model.configure_sdim(sdim_config)
            self._sdim_config.update(sdim_config)
    
    def calc_wmem(self):
        """Calculate weight memory depth."""
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        
        assert mh % pe == 0, "MH must be divisible by PE"
        assert mw % simd == 0, "MW must be divisible by SIMD"
        
        return mw * mh // (pe * simd)
    
    def calc_tmem(self):
        """Calculate threshold memory depth."""
        if self.get_nodeattr("noActivation") == 1:
            return 0
        else:
            mh = self.get_nodeattr("MH")
            pe = self.get_nodeattr("PE")
            return mh // pe
    
    def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
        """
        Convert weight matrix for hardware compatibility.
        
        Args:
            orig_weight_matrix: Original weight tensor
            
        Returns:
            Reformatted tensor suitable for hardware
        """
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wmem = self.calc_wmem()
        
        assert orig_weight_matrix.shape == (mw, mh), \
            f"Weight matrix shape {orig_weight_matrix.shape} doesn't match (MW={mw}, MH={mh})"
        assert mw % simd == 0, "MW must be divisible by SIMD"
        assert mh % pe == 0, "MH must be divisible by PE"
        
        # Transpose for hardware (ONNX vs hardware convention)
        ret = orig_weight_matrix.T
        
        # Convert bipolar to binary if needed
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            ret = (ret + 1) / 2
        
        # Interleave rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        
        # Reshape for hardware format
        ret = ret.reshape(1, pe, wmem, simd)
        
        # Reverse SIMD dimension
        ret = np.flip(ret, axis=-1)
        
        return ret
    
    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """
        Convert threshold matrix for hardware compatibility.
        
        Args:
            orig_thres_matrix: Original threshold tensor
            
        Returns:
            Reformatted tensor suitable for hardware
        """
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        tmem = mh // pe
        
        assert mh % pe == 0, "MH must be divisible by PE"
        assert orig_thres_matrix.ndim == 2, "Threshold matrix must be 2D"
        
        n_thres_steps = orig_thres_matrix.shape[1]
        
        # Check sign requirements for binary operations
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        
        # Reinterpret as bipolar if using binary XNOR mode
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        
        if inp_is_bipolar and wt_is_bipolar:
            assert (orig_thres_matrix >= 0).all()
            assert (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all()
        
        ret = orig_thres_matrix
        
        # Duplicate if single threshold
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        
        assert ret.shape[0] == mh
        
        # Distribute between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        
        return ret.reshape(1, pe, tmem, n_thres_steps)
    
    def execute_node(self, context, graph):
        """
        Execute MVAU in simulation.
        
        Args:
            context: Execution context with tensors
            graph: ONNX graph
        """
        node = self.onnx_node
        
        # Get inputs
        inp_a = context[node.input[0]]
        inp_a = inp_a.reshape(self.get_normal_input_shape(0))
        
        # Get weights (from initializer or context)
        mvau_w_init_list = [x for x in graph.initializer if x.name == node.input[1]]
        mvau_w_init = mvau_w_init_list[0] if mvau_w_init_list else None
        if mvau_w_init is not None:
            import onnx.numpy_helper as np_helper
            inp_b = np_helper.to_array(mvau_w_init)
        else:
            inp_b = context[node.input[1]]
        
        # Perform matrix multiplication
        if self.get_nodeattr("binaryXnorMode"):
            # Binary XNOR mode
            result = xp.xnorpopcountmatmul(inp_a, inp_b)
        elif (self.get_input_datatype(0) == DataType["BIPOLAR"] and
              self.get_input_datatype(1) == DataType["BIPOLAR"]):
            # Bipolar mode - convert to binary first
            result = xp.xnorpopcountmatmul((inp_a + 1) / 2, (inp_b + 1) / 2)
        else:
            # Regular matrix multiplication
            result = np.matmul(inp_a, inp_b)
        
        # Apply activation if enabled
        if self.get_nodeattr("noActivation") == 0:
            mvau_thr_init = [x for x in graph.initializer if x.name == node.input[2]][0]
            import onnx.numpy_helper as np_helper
            mvau_thr = np_helper.to_array(mvau_thr_init)
            
            odt_is_bipolar = self.get_output_datatype() == DataType["BIPOLAR"]
            out_scale = 2 if odt_is_bipolar else 1
            out_bias = -1 if odt_is_bipolar else self.get_nodeattr("ActVal")
            
            # Handle 4D tensors
            if result.ndim == 4:
                result = result.transpose((0, 3, 1, 2))
            
            result = multithreshold(result, mvau_thr, out_scale, out_bias)
            
            if result.ndim == 4:
                result = result.transpose((0, 2, 3, 1))
        
        # Store result
        oshape = context[node.output[0]].shape
        context[node.output[0]] = result.reshape(oshape)
    
    def minimize_accumulator_width(self, model):
        """
        Calculate minimum accumulator width based on weight values.
        
        Returns:
            Optimal accumulator DataType
        """
        weights = model.get_initializer(self.onnx_node.input[1])
        
        # Convert to bipolar if using binary XNOR mode
        if self.get_nodeattr("binaryXnorMode"):
            weights = 2 * weights - 1
        
        thresholds = None
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
        
        idt = self.get_input_datatype(0)
        
        # Calculate accumulator range
        (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)
        
        # Handle runtime-writeable weights
        if self.get_nodeattr("runtime_writeable_weights"):
            wdt = self.get_input_datatype(1)
            lower_worst = wdt.min() * np.ones_like(weights)
            lower_range = calculate_matvec_accumulator_range(lower_worst, idt)
            upper_worst = wdt.max() * np.ones_like(weights)
            upper_range = calculate_matvec_accumulator_range(upper_worst, idt)
            acc_min = min(min(lower_range), min(upper_range))
            acc_max = max(max(lower_range), max(upper_range))
        
        # Adjust range based on thresholds
        if thresholds is not None:
            threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
            min_threshold = thresholds.min()
            max_threshold = thresholds.max()
            
            # Clip thresholds if needed
            if max_threshold > acc_max or min_threshold < acc_min:
                warnings.warn(f"Clipping some thresholds in {self.onnx_node.name}")
                thresholds = np.clip(thresholds, acc_min, acc_max)
                model.set_initializer(self.onnx_node.input[2], thresholds)
                threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
                min_threshold = thresholds.min()
                max_threshold = thresholds.max()
            
            acc_min = min(min_threshold, acc_min)
            acc_max = max(max_threshold, acc_max)
        
        # Determine accumulator datatype
        if acc_min >= 0:
            acc_bit_width = np.log2(acc_max + 1)
            acc_bit_width = math.ceil(acc_bit_width)
            adt = DataType[f"UINT{acc_bit_width}"]
        else:
            _acc_max = max(-acc_min, 1 + acc_max)
            acc_bit_width = np.log2(_acc_max) + 1
            acc_bit_width = math.ceil(acc_bit_width)
            adt = DataType[f"INT{acc_bit_width}"]
        
        # Verify thresholds can be expressed with accumulator type
        if thresholds is not None:
            assert np.vectorize(adt.allowed)(threshold_tensor).all(), \
                f"Thresholds in {self.onnx_node.name} can't be expressed with type {adt}"
        
        # Update output datatype for no-activation mode
        if self.get_nodeattr("noActivation"):
            self.set_nodeattr("outputDataType", adt.name)
        
        self.set_nodeattr("accDataType", adt.name)
        return adt
    
    def verify_node(self) -> List[str]:
        """
        Verify MVAU-specific configuration.
        
        Returns:
            List of verification messages
        """
        # Start with parent verification
        messages = super().verify_node()
        
        # Check matrix dimensions
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        
        if mw and mw > 0:
            messages.append(f"✓ MW (input features) set to {mw}")
        else:
            messages.append("✗ MW must be > 0")
        
        if mh and mh > 0:
            messages.append(f"✓ MH (output features) set to {mh}")
        else:
            messages.append("✗ MH must be > 0")
        
        # Check divisibility
        if mw and simd and mw % simd == 0:
            messages.append(f"✓ SIMD={simd} divides MW={mw}")
        elif mw and simd:
            messages.append(f"✗ SIMD={simd} does not divide MW={mw}")
        
        if mh and pe and mh % pe == 0:
            messages.append(f"✓ PE={pe} divides MH={mh}")
        elif mh and pe:
            messages.append(f"✗ PE={pe} does not divide MH={mh}")
        
        # Check activation configuration
        no_act = self.get_nodeattr("noActivation")
        if no_act == 0:
            if len(self.onnx_node.input) >= 3:
                messages.append("✓ Activation enabled with threshold input")
            else:
                messages.append("✗ Activation enabled but no threshold input")
        else:
            messages.append("✓ No activation mode")
        
        # Check binary mode consistency
        if self.get_nodeattr("binaryXnorMode"):
            idt = self.get_input_datatype(0)
            wdt = self.get_input_datatype(1)
            if idt in [DataType["BINARY"], DataType["BIPOLAR"]] and \
               wdt in [DataType["BINARY"], DataType["BIPOLAR"]]:
                messages.append("✓ Binary XNOR mode with compatible datatypes")
            else:
                messages.append(f"✗ Binary XNOR mode but datatypes are {idt}, {wdt}")
        
        return messages


# Example usage
def create_mvau_example():
    """Create an example MVAU node."""
    # Mock ONNX node
    class MockONNXNode:
        def __init__(self):
            self.name = "MVAU_KM_0"
            self.op_type = "MVAUK"
            self.input = ["input_tensor", "weight_tensor", "threshold_tensor"]
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
    node.add_attribute("MW", 256)  # Input features
    node.add_attribute("MH", 128)  # Output features
    node.add_attribute("SIMD", 16)  # Process 16 inputs per cycle
    node.add_attribute("PE", 8)     # Produce 8 outputs per cycle
    node.add_attribute("inputDataType", "UINT8")
    node.add_attribute("weightDataType", "INT4")
    node.add_attribute("outputDataType", "UINT8")
    node.add_attribute("accDataType", "INT16")
    node.add_attribute("noActivation", 0)  # Use activation
    node.add_attribute("n_thres_steps", 255)  # 256-level quantization
    node.add_attribute("ActVal", 0)
    node.add_attribute("mem_mode", "internal_decoupled")
    node.add_attribute("numInputVectors", [1, 32, 32])  # E.g., conv layer
    
    # Create operation
    op = MVAUHWCustomOp(node)
    
    return op


if __name__ == "__main__":
    print("MatrixVectorActivation with Kernel Modeling Example")
    print("=" * 50)
    
    # Create example
    op = create_mvau_example()
    
    # Verify configuration
    print("\nVerification Results:")
    for msg in op.verify_node():
        print(f"  {msg}")
    
    print("\nShape Information:")
    print(f"  Normal input shape: {op.get_normal_input_shape(0)}")
    print(f"  Normal weight shape: {op.get_normal_input_shape(1)}")
    print(f"  Normal output shape: {op.get_normal_output_shape(0)}")
    
    # Force model creation to show folded shapes
    op._ensure_kernel_model()
    print(f"  Folded input shape: {op.get_folded_input_shape(0)}")
    print(f"  Folded output shape: {op.get_folded_output_shape(0)}")
    
    print("\nStream Widths:")
    print(f"  Input stream width: {op.get_instream_width(0)} bits")
    print(f"  Weight stream width: {op.get_instream_width(1)} bits")
    print(f"  Output stream width: {op.get_outstream_width(0)} bits")
    
    print("\nMemory Requirements:")
    print(f"  WMEM (weight memory): {op.calc_wmem()} entries")
    print(f"  TMEM (threshold memory): {op.calc_tmem()} entries")
    
    print("\nPerformance:")
    print(f"  Expected cycles: {op.get_exp_cycles()}")
    print(f"  BRAM estimation: {op.bram_estimation()} blocks")
    print(f"  LUT estimation: {op.lut_estimation()}")