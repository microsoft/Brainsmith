############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Migration to KernelOp by Microsoft Corporation
############################################################################

import math
import numpy as np
import warnings
from typing import Optional
from onnx import helper, NodeProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)
import qonnx.core.data_layout as DataLayout

from brainsmith.dataflow import KernelOp, FULL_DIM
from brainsmith.registry import kernel
import brainsmith.dataflow as df


# =============================================================================
# VectorVectorActivation Schema
# =============================================================================

VVAU_SCHEMA = df.KernelSchema(
    name="VectorVectorActivation",

    # =========================================================================
    # STRUCTURE: Input/Output interfaces
    # =========================================================================

    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM, FULL_DIM],  # Process full H, W dimensions
            stream_tiling=["nf", "sf", "SIMD", "PE"],  # Folded streaming: [nf, sf, SIMD*PE]
            required_layout="NHWC",
        ),
        df.InputSchema(
            name="weights",
            # Depthwise weights: (Channels, 1, k_h, k_w)
            block_tiling=[],  # No block tiling (static data)
            stream_tiling=[],  # Not streamed (static initializer or external stream)
        ),
        # Optional thresholds input (only if noActivation=0)
        df.InputSchema(
            name="thresholds",
            block_tiling=[],
            stream_tiling=[],
        ),
    ],

    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM, FULL_DIM],  # Same as input
            stream_tiling=["nf", "PE"],  # [nf, PE]
            datatype=None,  # From ONNX graph
            required_layout="NHWC",
        )
    ],

    # =========================================================================
    # KERNEL PARAMETERS
    # =========================================================================

    kernel_params={
        "PE": ("i", True, 1),
        "SIMD": ("i", False, 1),
        "Dim": ("ints", True, []),  # [H, W]
        "Channels": ("i", True, 0),
        "Kernel": ("ints", True, []),  # [k_h, k_w]
        "res_type": ("s", False, "auto"),
        "act_val": ("i", False, 0),
        "no_activation": ("i", False, 0),
        "binary_xnor_mode": ("i", False, 0),
        "mem_mode": ("s", False, "internal_decoupled"),
        "runtime_writeable_weights": ("i", False, 0),
        "ram_style": ("s", False, "auto"),

        # Accumulator datatype (kernel-specific parameter)
        "acc_dtype": ("s", False, "INT32"),
    },

    # =========================================================================
    # VALIDATION
    # =========================================================================

    constraints=[
        # Input must be dynamic, weights and thresholds must be static
        df.IsDynamic(("input",)),
        df.IsStatic(("weights",)),
        df.IsStatic(("thresholds",)),

        # PE must divide Channels
        df.DimensionDivisible("input", -1, "PE", hierarchy=df.ShapeHierarchy.STREAM),

        # Datatypes must be integer
        df.DatatypeInteger(("input", "output")),
    ],

    # =========================================================================
    # TRANSFORMATION
    # =========================================================================

)


# =============================================================================
# VectorVectorActivation Kernel Implementation
# =============================================================================

@kernel(
    description="Hardware Vector-Vector Activation Unit for depthwise convolutions (KernelOp-based)",
    author="Microsoft Corporation"
)
class VectorVectorActivation(KernelOp):
    """Modern VectorVectorActivation implementation using KernelOp system.

    Performs depthwise convolution with optional multi-threshold activation.
    Designed for post-Im2Col/ConvolutionInputGenerator processing.

    Key features:
    - Schema-driven design (no shape storage in nodeattrs)
    - Supports internal_embedded, internal_decoupled, and external memory modes
    - Parallelization via PE (channels) and SIMD (kernel elements)
    - Optional runtime-writable weights
    - Optional threshold activation

    Arete principles:
    - Shapes extracted from design_point (not nodeattrs)
    - Declarative constraints in schema
    - Two-phase construction (DesignSpace → Configuration)
    """

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> df.KernelSchema:
        """Build VectorVectorActivation schema (constant for all instances)."""
        return VVAU_SCHEMA

    # ================================================================
    # Inference (Static methods)
    # ================================================================

    @staticmethod
    def can_infer_from(node, model: ModelWrapper) -> bool:
        """Check if this node can be converted to VectorVectorActivation.

        Detects MatMul nodes with sparsity annotation indicating depthwise convolution.
        """
        if node.op_type != "MatMul":
            return False

        # Check for sparsity annotation
        sparsity = model.get_tensor_sparsity(node.input[1])
        if sparsity is None:
            return False

        # Check for depthwise convolution indicator
        try:
            k_h, k_w = sparsity["dw"]["kernel_shape"]
        except KeyError:
            return False

        # Get datatypes
        idt = model.get_tensor_datatype(node.input[0])
        wdt = model.get_tensor_datatype(node.input[1])

        # Must be integer types
        if not (idt.is_integer() and wdt.is_integer()):
            return False

        return True

    @staticmethod
    def infer_from(node, model: ModelWrapper, insert_index: int) -> df.TransformationResult:
        """Convert MatMul with sparsity to VectorVectorActivation.

        Extracts depthwise convolution structure from sparse MatMul and creates
        VVAU node. Optionally absorbs following MultiThreshold into VVAU.

        NOTE: Assumes input is already in NHWC layout. The global
        normalize_dataflow_layouts preprocessing pass ensures this.

        Args:
            node: MatMul ONNX node with sparsity annotation
            model: Model wrapper
            insert_index: Where to insert new node

        Returns:
            df.TransformationResult with new VVAU node
        """
        # Extract sparsity metadata
        sparsity = model.get_tensor_sparsity(node.input[1])
        try:
            k_h, k_w = sparsity["dw"]["kernel_shape"]
        except KeyError:
            raise Exception(
                f"{node.name}: sparsity annotation doesn't indicate depthwise convolution"
            )

        # Get input/output/weight info
        mm_input = node.input[0]
        mm_weight = node.input[1]
        mm_output = node.output[0]
        mm_in_shape = model.get_tensor_shape(mm_input)
        mm_out_shape = model.get_tensor_shape(mm_output)
        idt = model.get_tensor_datatype(mm_input)
        wdt = model.get_tensor_datatype(mm_weight)

        # Get weight matrix
        W = model.get_initializer(mm_weight)

        # Infer dense weight tensor from sparse weight matrix
        # Weight matrix has shape (k_h * k_w * Channels, Channels)
        # Need to reverse to (Channels, 1, k_h, k_w)
        channels = int(W.shape[1])

        # Transpose to (Channels, k_h * k_w * Channels)
        W = W.T
        # Reshape to (Channels, k_h, k_w, Channels)
        W = W.reshape(channels, k_h, k_w, channels)
        # Transpose to (Channels, Channels, k_h, k_w)
        W = W.transpose(0, 3, 1, 2)

        # Extract depthwise weights (diagonal elements)
        w_tensor = np.zeros((channels, 1, k_h, k_w), dtype=np.float32)
        for ch in range(channels):
            w_tensor[ch][0] = W[ch][ch]

        # Update weight initializer
        model.set_initializer(mm_weight, w_tensor)
        model.set_tensor_shape(mm_weight, (channels, 1, k_h, k_w))

        # Default PE = channels (fully parallel)
        pe = channels

        # Check for following MultiThreshold to absorb
        consumer = model.find_consumer(mm_output)
        nodes_to_remove = [node]
        nodes_to_insert = []

        if consumer is not None and consumer.op_type == "MultiThreshold":
            # Create VVAU with activation
            mt_output = consumer.output[0]
            mt_out_shape = model.get_tensor_shape(mt_output)
            mt_thres = consumer.input[1]
            T = model.get_initializer(mt_thres)

            assert T.shape[0] == 1 or T.shape[0] == channels, (
                f"{consumer.name}: First dimension of thresholds neither 1 nor Channels"
            )

            odt = model.get_tensor_datatype(mt_output)

            # Get MultiThreshold parameters
            from qonnx.custom_op.registry import getCustomOp
            mt_inst = getCustomOp(consumer)
            scale = mt_inst.get_nodeattr("out_scale")
            actval = mt_inst.get_nodeattr("out_bias")

            assert scale == 1.0, (
                f"{consumer.name}: out_scale must be 1.0 for HLS conversion"
            )
            assert int(actval) == actval, (
                f"{consumer.name}: out_bias must be integer for HLS conversion"
            )
            actval = int(actval)
            assert (not odt.signed()) or (actval < 0), (
                f"{consumer.name}: Signed output requires actval < 0"
            )

            model.set_tensor_shape(mm_input, mm_in_shape)
            model.set_tensor_shape(mt_output, mt_out_shape)

            # Create VVAU node with activation
            new_node = helper.make_node(
                "VectorVectorActivation",
                inputs=[mm_input, mm_weight, mt_thres],
                outputs=[mt_output],
                domain="brainsmith.kernels",
                backend="fpgadataflow",
                name=f"VectorVectorActivation_{node.name}",

                # Parameters
                Dim=[mm_in_shape[1], mm_in_shape[2]],
                Channels=channels,
                Kernel=[k_h, k_w],
                act_val=actval,
                no_activation=0,
            )

            nodes_to_remove.append(consumer)

        else:
            # No activation, matmul only
            odt = model.get_tensor_datatype(mm_output)
            model.set_tensor_shape(mm_input, mm_in_shape)
            model.set_tensor_shape(mm_output, mm_out_shape)

            # Create VVAU node without activation
            new_node = helper.make_node(
                "VectorVectorActivation",
                inputs=[mm_input, mm_weight],
                outputs=[mm_output],
                domain="brainsmith.kernels",
                backend="fpgadataflow",
                name=f"VectorVectorActivation_{node.name}",

                # Parameters
                Dim=[mm_in_shape[1], mm_in_shape[2]],
                Channels=channels,
                Kernel=[k_h, k_w],
                act_val=0,
                no_activation=1,
            )

        return df.TransformationResult(
            nodes_to_insert=[new_node],
            nodes_to_remove=nodes_to_remove,
            actual_layouts={}
        )

    # ================================================================
    # Shape Methods (Arete: Extract from design_point)
    # ================================================================

    def _infer_sparse_weight_tensor(self, W_conv, k_h, k_w, channels):
        """Helper to convert depthwise weights to sparse format for execution."""
        W_sparse = np.zeros((channels, channels, k_h, k_w), dtype=np.float32)
        for ch in range(channels):
            W_sparse[ch][ch] = W_conv[ch][0]
        W_conv = W_sparse.astype(np.float32)
        W_matmul = W_conv.transpose(0, 2, 3, 1)
        W_matmul = W_matmul.reshape(channels, channels * k_h * k_w)
        W_matmul = W_matmul.T
        return W_matmul

    def get_normal_input_shape(self, ind=0):
        """Get unfolded input shape from design_point."""
        if ind == 0:
            # Input data shape: (1, dim_h, dim_w, channels * k_h * k_w)
            dim_h, dim_w = self.get_nodeattr("Dim")
            channels = self.get_nodeattr("Channels")
            k_h, k_w = self.get_nodeattr("Kernel")
            return tuple([1, dim_h, dim_w, k_h * k_w * channels])
        elif ind == 1:
            # Weight shape: (channels, 1, k_h, k_w)
            channels = self.get_nodeattr("Channels")
            k_h, k_w = self.get_nodeattr("Kernel")
            return tuple([channels, 1, k_h, k_w])
        elif ind == 2:
            # Threshold shape (if exists)
            if self.get_nodeattr("no_activation") == 0:
                channels = self.get_nodeattr("Channels")
                # Threshold tensor from model
                thres = self.onnx_node.input[2] if len(self.onnx_node.input) > 2 else None
                if thres:
                    # Will be set during model processing
                    return None
            return None
        else:
            raise Exception(f"Invalid input index: {ind}")

    def get_normal_output_shape(self, ind=0):
        """Get unfolded output shape from design_point."""
        dim_h, dim_w = self.get_nodeattr("Dim")
        channels = self.get_nodeattr("Channels")
        return tuple([1, dim_h, dim_w, channels])

    def get_folded_input_shape(self, ind=0):
        """Get folded input shape with PE and SIMD parallelization.

        Folding pattern: (1, dim_h, dim_w, sf*nf, SIMD*PE)
        where sf = kernel_2 // SIMD, nf = channels // PE
        """
        k_h, k_w = self.get_nodeattr("Kernel")
        dim_h, dim_w = self.get_nodeattr("Dim")
        ch = self.get_nodeattr("Channels")
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        kernel_2 = k_h * k_w

        assert kernel_2 % simd == 0, (
            f"Requirement kernel (k_h * k_w) divisible by SIMD is violated"
        )
        sf = kernel_2 // simd

        assert ch % pe == 0, (
            f"Requirement Channels divisible by PE is violated"
        )
        nf = ch // pe

        if ind == 0:
            # Input data: (1, dim_h, dim_w, sf*nf, SIMD*PE)
            folded_input_shape = tuple([1, dim_h, dim_w, sf * nf, simd * pe])
        elif ind == 1 and self.get_nodeattr("mem_mode") == "external":
            # External weights: (1, sf*nf, PE)
            folded_input_shape = tuple([1, sf * nf, pe])
        else:
            raise Exception("Undefined input shape for requested input")

        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        """Get folded output shape: (1, dim_h, dim_w, nf, PE)."""
        ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        nf = ch // pe
        dim_h, dim_w = self.get_nodeattr("Dim")
        folded_output_shape = tuple([1, dim_h, dim_w, nf, pe])
        return folded_output_shape

    # ================================================================
    # Datatype Methods
    # ================================================================

    def get_accumulator_datatype(self):
        """Returns FINN DataType of accumulator."""
        return DataType[self.get_nodeattr("acc_dtype")]

    # ================================================================
    # Stream Width Methods
    # ================================================================

    def get_instream_width(self, ind=0):
        """Get input stream width in bits."""
        if ind == 0:
            i_bits = self.get_input_datatype(ind).bitwidth()
            simd = self.get_nodeattr("SIMD")
            pe = self.get_nodeattr("PE")
            width = i_bits * simd * pe
        elif ind == 1:
            mem_mode = self.get_nodeattr("mem_mode")
            if mem_mode in ["internal_decoupled", "external"]:
                simd = self.get_nodeattr("SIMD")
                pe = self.get_nodeattr("PE")
                wp = self.get_input_datatype(1).bitwidth()
                width = simd * pe * wp
            else:
                width = 0
        elif ind == 2:
            # Thresholds always embedded
            act = not self.get_nodeattr("no_activation")
            if act:
                width = 0
            else:
                raise Exception("Index out of range")
        else:
            raise Exception("Undefined input ind for this layer type")
        return width

    def get_outstream_width(self, ind=0):
        """Get output stream width in bits."""
        o_bits = self.get_output_datatype().bitwidth()
        out_width = o_bits * self.get_nodeattr("PE")
        return out_width

    # ================================================================
    # Memory Calculations
    # ================================================================

    def calc_wmem(self):
        """Calculates and returns WMEM (weight memory depth)."""
        ch = self.get_nodeattr("Channels")
        k_h, k_w = self.get_nodeattr("Kernel")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wmem = (k_h * k_w * ch // pe) // simd
        return wmem

    def calc_tmem(self):
        """Calculates and returns TMEM (threshold memory depth)."""
        if self.get_nodeattr("no_activation") == 1:
            return 0
        else:
            ch = self.get_nodeattr("Channels")
            pe = self.get_nodeattr("PE")
            return ch // pe

    # ================================================================
    # Performance/Resource Estimation
    # ================================================================

    def get_exp_cycles(self):
        """Expected cycles for execution."""
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        ch = self.get_nodeattr("Channels")
        dim_h, dim_w = self.get_nodeattr("Dim")
        k_h, k_w = self.get_nodeattr("Kernel")
        batch_size = 1
        mmv = 1
        exp_cycles = ((ch * k_h * k_w) / pe / simd) * batch_size * (dim_h * dim_w) / mmv
        return int(exp_cycles)

    def get_number_output_values(self):
        """Number of output values produced."""
        folded_oshape = self.get_folded_output_shape()
        return int(np.prod(folded_oshape[:-1]))

    # ================================================================
    # Bit Width Minimization
    # ================================================================

    def minimize_accumulator_width(self, model):
        """Minimize the accumulator bit width based on weight values and input datatypes."""
        weights = model.get_initializer(self.onnx_node.input[1])
        k_h, k_w = self.get_nodeattr("Kernel")
        fm = self.get_nodeattr("Channels")

        # Put weights into shape expected by calculate_matvec_accumulator_range
        weights = weights.reshape(fm, k_h * k_w).transpose()

        # Convert bipolar weights to binary if needed
        if self.get_nodeattr("binary_xnor_mode"):
            weights = 2 * weights - 1

        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
        else:
            thresholds = None

        idt = self.get_input_datatype(0)
        (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)

        # Handle runtime-writable weights
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

        # Validate thresholds fit in accumulator datatype
        if thresholds is not None:
            assert np.vectorize(adt.allowed)(threshold_tensor).all(), (
                f"Thresholds in {self.onnx_node.name} can't be expressed with type {str(adt)}"
            )

        # For no activation, output dt = acc dt
        if self.get_nodeattr("no_activation"):
            if model.find_direct_successors(self.onnx_node) is None:
                bw = roundup_to_integer_multiple(adt.bitwidth(), 8)
                new_adt_name = adt.name.replace(str(adt.bitwidth()), str(bw))
                adt = DataType[new_adt_name]
            self.set_nodeattr("output0Datatype", adt.name)

        self.set_nodeattr("acc_dtype", adt.name)
        return DataType[self.get_nodeattr("acc_dtype")]

    def minimize_weight_bit_width(self, model):
        """Minimize the weight bit width based on weight values."""
        if not self.get_nodeattr("runtime_writeable_weights"):
            weights = model.get_initializer(self.onnx_node.input[1])
            w_min = weights.min()
            w_max = weights.max()
            if w_min < 0:
                if abs(w_min) > w_max:
                    wdt = DataType.get_smallest_possible(w_min)
                else:
                    wdt = DataType.get_smallest_possible(-w_max - 1)
            else:
                wdt = DataType.get_smallest_possible(w_max)
            self.set_nodeattr("input1Datatype", wdt.name)
        return self.get_input_datatype(1)

    # ================================================================
    # HW-Compatible Tensor Conversion
    # ================================================================

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert threshold matrix to HW-compatible format.

        Returns: Reshaped tensor (1, PE, TMEM, n_thres_steps)
        """
        ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        tmem = self.calc_tmem()

        assert ch % pe == 0, "Requirement Channels divisible by PE is violated"
        assert orig_thres_matrix.ndim == 2, "Threshold matrix dimension is not as expected (2)"

        n_thres_steps = orig_thres_matrix.shape[1]

        # Check for bipolar/binary xnor mode
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.get_nodeattr("binary_xnor_mode") == 1
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)

        if inp_is_bipolar and wt_is_bipolar:
            assert (orig_thres_matrix >= 0).all()
            assert (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all()

        ret = orig_thres_matrix

        # Ensure channels match
        if ret.shape[0] == 1:
            ret = np.tile(ret, (ch, 1))
        assert ret.shape[0] == ch, "Channels of threshold matrix are not as expected"

        # Distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        assert ret.shape[0] == pe
        assert ret.shape[1] == tmem
        assert ret.shape[2] == n_thres_steps

        return ret.reshape(1, pe, tmem, n_thres_steps)

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
        """Convert weight matrix to HW-compatible format.

        Returns: Reshaped tensor (1, PE, WMEM, SIMD)
        """
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        ch = self.get_nodeattr("Channels")
        k_h, k_w = self.get_nodeattr("Kernel")
        wmem = self.calc_wmem()

        assert orig_weight_matrix.shape == (ch, 1, k_h, k_w), (
            "Weights matrix doesn't have expected shape (channels, 1, k_h, k_w)"
        )

        ret = orig_weight_matrix

        # Convert bipolar to binary if needed
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            ret = (ret + 1) / 2

        ret = ret.reshape(ch, k_h * k_w)

        # Distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        ret = ret.reshape(1, pe, wmem, simd)

        return ret

    # ================================================================
    # Execution
    # ================================================================

    def execute_node(self, context, graph):
        """Execute VVAU operation (Python reference implementation)."""
        node = self.onnx_node
        in_act = context[node.input[0]]
        (_, dim_h, dim_w, _) = in_act.shape
        (k_h, k_w) = self.get_nodeattr("Kernel")
        channels = self.get_nodeattr("Channels")

        # Check if producer is Im2Col or ConvolutionInputGenerator
        producer = [x for x in graph.node if x.output[0] == node.input[0]]
        if bool(producer) and producer[0].op_type in ["Im2Col", "ConvolutionInputGenerator"]:
            pe = channels
        else:
            pe = self.get_nodeattr("PE")

        # Reorder input activations (untangle PE interleaving)
        in_act = in_act.reshape(1, dim_h, dim_w, channels // pe, k_h * k_w, pe)
        in_act = in_act.transpose(0, 1, 2, 4, 3, 5)
        in_act = in_act.reshape(1, dim_h, dim_w, channels * k_h * k_w)

        # Reshape weights
        import onnx.numpy_helper as np_helper
        vvau_w_init = [x for x in graph.initializer if x.name == node.input[1]][0]
        vvau_w = np_helper.to_array(vvau_w_init)
        vvau_w_onnx = self._infer_sparse_weight_tensor(vvau_w, k_h, k_w, channels)

        # Compute matmul
        if (self.get_input_datatype(0) == DataType["BIPOLAR"] and
                self.get_input_datatype(1) == DataType["BIPOLAR"]):
            result = np.matmul(in_act, vvau_w_onnx)
            result = (result + k_h * k_w) / 2
        else:
            result = np.matmul(in_act, vvau_w_onnx)

        # Apply activation if present
        if self.get_nodeattr("no_activation") == 0:
            vvau_thr_init = [x for x in graph.initializer if x.name == node.input[2]][0]
            vvau_thr = np_helper.to_array(vvau_thr_init)
            odt_is_bipolar = self.get_output_datatype() == DataType["BIPOLAR"]
            out_scale = 2 if odt_is_bipolar else 1
            out_bias = -1 if odt_is_bipolar else self.get_nodeattr("act_val")

            # NHWC to NCHW for multithreshold
            result = result.transpose((0, 3, 1, 2))
            result = multithreshold(result, vvau_thr, out_scale, out_bias)
            # NCHW to NHWC
            result = result.transpose((0, 2, 3, 1))

        context[node.output[0]] = result

    # ================================================================
    # FINN Integration
    # ================================================================

    def infer_node_datatype(self, model):
        """Infer and update node datatypes."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])

        if idt != self.get_input_datatype(0):
            warn_str = (
                f"input0Datatype changing for {node.name}: "
                f"{self.get_input_datatype(0)} -> {idt}"
            )
            warnings.warn(warn_str)

        self.set_nodeattr("input0Datatype", idt.name)

        # Set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        """Verify node configuration."""
        info_messages = []

        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        return info_messages
