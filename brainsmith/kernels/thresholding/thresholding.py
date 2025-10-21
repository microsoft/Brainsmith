############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Migration to KernelOp by Microsoft Corporation
############################################################################

import numpy as np
import warnings
from typing import Callable, List
from onnx import helper

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions
import qonnx.core.data_layout as DataLayout
from qonnx.util.onnx import nchw_to_nhwc

from brainsmith.dataflow import KernelOp, FULL_DIM
from brainsmith.core.plugins import kernel
import brainsmith.dataflow as df
from typing import Optional
from onnx import NodeProto


# =============================================================================
# Thresholding Schema
# =============================================================================

THRESHOLDING_SCHEMA = df.KernelSchema(
    name="Thresholding",
    domain="brainsmith.kernels",

    # =========================================================================
    # STRUCTURE: Input/Output interfaces
    # =========================================================================

    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],       # Process full spatial dimensions
            stream_tiling=["PE"],           # Parallel channels with PE
            required_layout="NHWC",         # Hardware requires NHWC layout
            constraints=[df.IsDynamic(("input",))]  # Must be dynamic tensor
        ),
        df.InputSchema(
            name="thresholds",
            # Thresholds are constant weights (NumChannels x num_steps)
            # Not tiled or streamed - full tensor loaded as initializer
            block_tiling=[],  # No block tiling (static data)
            stream_tiling=[],  # Not streamed (static data)
            constraints=[df.IsStatic(("thresholds",))]  # Must be initializer
        ),
    ],

    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],                    # Same as input
            stream_tiling=[df.DerivedDim("input", -1)],  # Match input PE
            datatype=None,  # Datatype comes from ONNX graph (set via node attrs)
            required_layout="NHWC",
        )
    ],

    # =========================================================================
    # KERNEL PARAMETERS: Threshold-specific configuration
    # =========================================================================

    kernel_params={
        "num_steps": ("i", True, 1),       # Number of threshold steps (required)
        "act_val": ("i", False, 0),        # Activation bias value (ActVal)
        "num_input_vectors": ("ints", False, [1]),  # Batch/spatial dims (legacy)
        "runtime_writeable_weights": ("i", False, 0),  # AXI-lite writable (1/0)

        # Datatypes stored as nodeattrs for FINN backend compatibility
        # Note: These are also inferred from ONNX graph tensors
        "input_dtype": ("s", True, ""),
        "weight_dtype": ("s", True, ""),
        "output_dtype": ("s", True, ""),
    },

    # =========================================================================
    # VALIDATION: Constraints
    # =========================================================================

    constraints=[
        # PE must divide number of channels
        df.DimensionDivisible("input", -1, "PE", hierarchy=df.ShapeHierarchy.STREAM),

        # Datatypes must be integer (enforced in can_infer_from)
        df.DatatypeInteger(("input", "output")),
    ],

    # =========================================================================
    # TRANSFORMATION: Source op matching
    # =========================================================================

    source_ops=["MultiThreshold"],  # Matches MultiThreshold nodes

    # Parallelization
    initial_parallelization={"PE": 1},
)


# =============================================================================
# Thresholding Kernel Implementation
# =============================================================================

@kernel(
    description="Hardware multi-threshold activation (KernelOp-based)",
    author="Microsoft Corporation"
)
class Thresholding(KernelOp):
    """Modern Thresholding implementation using KernelOp system.

    This kernel applies multi-threshold activation functions to input tensors.
    It compares each input value against a set of thresholds to produce
    quantized outputs.

    Key features:
    - Schema-driven design (no shape storage)
    - Supports both internal_embedded and internal_decoupled memory modes
    - Parallelization via PE parameter
    - Optional runtime-writable thresholds (internal_decoupled mode)

    Arete principles:
    - Shapes extracted from kernel_instance (not nodeattrs)
    - Declarative constraints in schema
    - Two-phase construction (DesignSpace → Configuration)
    """

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> df.KernelSchema:
        """Build Thresholding schema (constant for all instances)."""
        return THRESHOLDING_SCHEMA

    # ================================================================
    # Inference (Static methods)
    # ================================================================

    @staticmethod
    def can_infer_from(node, model: ModelWrapper) -> bool:
        """Check if this node can be converted to Thresholding.

        Detects MultiThreshold nodes with:
        - Integer input/output datatypes
        - out_scale == 1.0
        - Integer out_bias (ActVal)
        """
        if node.op_type != "MultiThreshold":
            return False

        # Get datatypes
        idt = model.get_tensor_datatype(node.input[0])
        odt = model.get_tensor_datatype(node.output[0])

        # Must be integer types
        if not (idt.is_integer() and odt.is_integer()):
            return False

        # Get MultiThreshold instance to check scale/bias
        from qonnx.custom_op.registry import getCustomOp
        mt_inst = getCustomOp(node)

        # Check out_scale (must be 1.0 for HW conversion)
        scale = mt_inst.get_nodeattr("out_scale")
        if scale != 1.0:
            return False

        # Check out_bias is integer
        actval = mt_inst.get_nodeattr("out_bias")
        if int(actval) != actval:
            return False

        return True

    @staticmethod
    def infer_from(node, model: ModelWrapper, insert_index: int) -> df.TransformationResult:
        """Convert MultiThreshold node to Thresholding node.

        Extracts parameters from MultiThreshold and creates a Thresholding node
        with equivalent behavior.

        Args:
            node: MultiThreshold ONNX node
            model: Model wrapper
            insert_index: Where to insert new node

        Returns:
            df.TransformationResult with new Thresholding node
        """
        # Get input/output names and shapes
        thl_input = node.input[0]
        thl_threshold = node.input[1]
        thl_output = node.output[0]
        thl_in_shape = model.get_tensor_shape(thl_input)
        thl_thres_shape = model.get_tensor_shape(thl_threshold)

        # Get datatypes
        idt = model.get_tensor_datatype(thl_input)
        tdt = model.get_tensor_datatype(thl_threshold)
        odt = model.get_tensor_datatype(thl_output)

        # Track tensor renames for layout conversions
        original_input = thl_input
        original_output = thl_output
        nodes_to_remove = [node]
        nodes_to_insert = []

        # Check input layout and convert if needed
        thl_in_layout = model.get_tensor_layout(thl_input)
        if thl_in_layout == DataLayout.NCHW:
            thl_input = nchw_to_nhwc(thl_input, model, insert_index)
            insert_index += 1
            thl_in_shape = model.get_tensor_shape(thl_input)

        # Track where to insert the Thresholding node
        # (must be before any output layout conversion)
        thresholding_insert_index = insert_index

        # Check output layout and convert if needed
        thl_output_layout = model.get_tensor_layout(thl_output)
        if thl_output_layout == DataLayout.NCHW:
            thl_output = nchw_to_nhwc(thl_output, model, insert_index, reverse=True)
            insert_index += 1

        # Extract parameters from MultiThreshold
        from qonnx.custom_op.registry import getCustomOp
        mt_inst = getCustomOp(node)

        scale = mt_inst.get_nodeattr("out_scale")
        assert scale == 1.0, (
            f"{node.name}: MultiThreshold out_scale must be 1 for HLS conversion."
        )

        actval = mt_inst.get_nodeattr("out_bias")
        assert int(actval) == actval, (
            f"{node.name}: MultiThreshold out_bias must be integer for HLS conversion."
        )
        actval = int(actval)

        # For signed activation (except BIPOLAR), actval should be negative
        if odt != DataType["BIPOLAR"]:
            assert (not odt.signed()) or (actval < 0), (
                f"{node.name}: Signed output requires actval < 0"
            )

        # Extract NumChannels from input shape (last dimension)
        # After potential NCHW→NHWC conversion, channels are in last position
        num_channels = int(thl_in_shape[-1])

        # Create Thresholding node
        new_node = helper.make_node(
            "Thresholding",
            inputs=[thl_input, thl_threshold],
            outputs=[thl_output],
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name=f"Thresholding_{node.name}",

            # Template parameters
            PE=1,  # Start with no parallelization
            num_steps=thl_thres_shape[1],
            act_val=actval,
            input_dtype=idt.name,
            weight_dtype=tdt.name,
            output_dtype=odt.name,
            num_input_vectors=list(thl_in_shape[:-1]),
            runtime_writeable_weights=0,
        )

        # Return transformation result
        # Layout conversion nodes were already inserted via nchw_to_nhwc
        return df.TransformationResult(
            nodes_to_insert=[new_node],
            nodes_to_remove=nodes_to_remove,
            actual_layouts={}  # Layouts already handled by nchw_to_nhwc
        )

    def get_nodeattr_types(self):
        """Define nodeattrs for Thresholding kernel.

        Combines:
        - Schema-derived nodeattrs (template parameters + interface datatypes)
        - HWCustomOp execution nodeattrs (from parent)
        """
        my_attrs = super().get_nodeattr_types()
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        """Get unfolded input shape.

        Extracts from kernel_instance (Arete principle: no shape storage).
        Returns: tuple(numInputVectors + [NumChannels])
        """
        ki = self.kernel_instance
        if ind == 0:
            return tuple(ki.inputs["input"].tensor_shape)
        elif ind == 1:
            # Threshold tensor shape
            return tuple(ki.inputs["thresholds"].tensor_shape)
        else:
            raise Exception(f"Invalid input index: {ind}")

    def get_normal_output_shape(self, ind=0):
        """Get unfolded output shape.

        Same as input shape (pass-through).
        """
        ki = self.kernel_instance
        return tuple(ki.outputs["output"].tensor_shape)

    def get_folded_input_shape(self, ind=0):
        """Get folded input shape with PE parallelization.

        Transforms: (..., NumChannels) → (..., NumChannels//PE, PE)
        """
        if ind != 0:
            return self.get_normal_input_shape(ind)

        normal_ishape = list(self.get_normal_input_shape(0))
        pe = self.get_nodeattr("PE")
        num_channels = normal_ishape[-1]

        assert num_channels % pe == 0, (
            f"PE={pe} must divide NumChannels={num_channels}"
        )

        fold = num_channels // pe
        folded_ishape = normal_ishape[:-1] + [fold, pe]
        return tuple(folded_ishape)

    def get_folded_output_shape(self, ind=0):
        """Get folded output shape.

        Same as folded input shape.
        """
        return self.get_folded_input_shape(0)

    def get_number_output_values(self):
        """Number of output values produced.

        Returns: product of folded output shape except last dimension.
        """
        folded_oshape = self.get_folded_output_shape()
        return int(np.prod(folded_oshape[:-1]))

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            dt = DataType[self.get_nodeattr("input_dtype")]
        elif ind == 1:
            dt = DataType[self.get_nodeattr("weight_dtype")]
        else:
            raise Exception(f"Invalid input index: {ind}")
        return dt

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("output_dtype")]

    def get_instream_width(self, ind=0):
        """Get input stream width in bits.

        For ind=0 (data): inputDataType.bitwidth() * PE
        For ind=1 (thresholds, decoupled mode): PE * weightDataType.bitwidth() * numSteps
        """
        if ind == 0:
            ibits = self.get_input_datatype(0).bitwidth()
            pe = self.get_nodeattr("PE")
            return ibits * pe
        elif ind == 1:
            # For internal_decoupled mode only
            try:
                mem_mode = self.get_nodeattr("mem_mode")
            except AttributeError:
                mem_mode = "internal_embedded"

            if mem_mode == "internal_decoupled":
                pe = self.get_nodeattr("PE")
                wp = self.get_input_datatype(1).bitwidth()
                n_thres_steps = self.get_nodeattr("num_steps")
                return pe * wp * n_thres_steps
            else:
                return 0
        else:
            raise Exception(f"Invalid input index: {ind}")

    def get_outstream_width(self, ind=0):
        """Get output stream width in bits."""
        obits = self.get_output_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        return obits * pe

    def get_exp_cycles(self):
        """Expected cycles for execution.

        Returns: NumChannels/PE * batch_size * spatial_dimensions
        """
        folded_oshape = self.get_folded_output_shape()
        return int(np.prod(folded_oshape[:-1]))

    def calc_tmem(self):
        """Calculate TMEM (threshold memory depth).

        Returns: NumChannels // PE
        """
        ki = self.kernel_instance
        num_channels = ki.inputs["input"].tensor_shape[-1]
        pe = self.get_nodeattr("PE")
        return num_channels // pe

    def minimize_accumulator_width(self, model):
        """Minimize threshold width ('accumulator width' by convention).

        Analyzes threshold values to find the smallest datatype that can
        represent all threshold values.
        """
        idt = self.get_input_datatype(0)
        wdt_name = self.get_nodeattr("weight_dtype")

        # Skip minimization for floating-point types
        if str(idt).startswith("FLOAT") or wdt_name.startswith("FLOAT"):
            return DataType[wdt_name]

        # Get threshold tensor
        thresholds = model.get_initializer(self.onnx_node.input[1])
        min_threshold = thresholds.min()
        max_threshold = thresholds.max()

        # Get input range
        min_input = idt.min()
        max_input = idt.max()

        # Find required range
        tdt_min = min(min_input, min_threshold)
        tdt_max = max(max_input, max_threshold)

        # Find smallest datatype that fits
        if tdt_min < 0:
            if abs(tdt_min) > tdt_max:
                tdt = DataType.get_smallest_possible(tdt_min)
            else:
                tdt = DataType.get_smallest_possible(-tdt_max - 1)
        else:
            tdt = DataType.get_smallest_possible(tdt_max)

        # Validate thresholds fit in chosen datatype
        threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
        assert np.vectorize(tdt.allowed)(threshold_tensor).all(), (
            f"Thresholds can't be expressed with type {str(tdt)}"
        )

        # Update weight datatype
        self.set_nodeattr("weight_dtype", tdt.name)
        model.set_tensor_datatype(self.onnx_node.input[1], tdt)

        return tdt

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert threshold matrix to HW-compatible format.

        Ensures:
        - NumChannels % PE == 0
        - For unsigned inputs, thresholds are positive
        - Rows interleaved between PEs
        - Reshaped to (PE, TMEM, n_thres_steps)

        Args:
            orig_thres_matrix: Original threshold matrix (NumChannels, n_thres_steps)

        Returns:
            Reshaped threshold tensor (1, PE, TMEM, n_thres_steps)
        """
        ki = self.kernel_instance
        num_channels = ki.inputs["input"].tensor_shape[-1]
        pe = self.get_nodeattr("PE")
        tmem = num_channels // pe

        assert num_channels % pe == 0, (
            f"Requirement NumChannels={num_channels} divisible by PE={pe} is violated."
        )

        assert orig_thres_matrix.ndim == 2, (
            "Threshold matrix dimension is not as expected (2)."
        )

        n_thres_steps = orig_thres_matrix.shape[1]
        assert n_thres_steps == self.get_nodeattr("num_steps"), (
            "Mismatch in threshold steps"
        )

        # For unsigned inputs, ensure all thresholds are nonnegative
        if not self.get_input_datatype(0).signed():
            assert (orig_thres_matrix >= 0).all(), (
                "Unsigned input requires nonnegative thresholds"
            )

        ret = orig_thres_matrix

        # Ensure channels match NumChannels, duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (num_channels, 1))

        assert ret.shape[0] == num_channels, (
            f"Channels of threshold matrix ({ret.shape[0]}) don't match NumChannels ({num_channels})"
        )

        # Distribute rows between PEs (interleaving)
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)

        assert ret.shape[0] == pe, (
            f"First dimension after PE distribution ({ret.shape[0]}) != PE ({pe})"
        )
        assert ret.shape[1] == tmem, (
            f"Second dimension after PE distribution ({ret.shape[1]}) != TMEM ({tmem})"
        )
        assert ret.shape[2] == n_thres_steps, (
            f"Third dimension after PE distribution ({ret.shape[2]}) != numSteps ({n_thres_steps})"
        )

        return ret.reshape(1, pe, tmem, n_thres_steps)

    def execute_node(self, context, graph):
        """Execute thresholding operation using QONNX multithreshold.

        Applies multi-threshold activation to input tensor.
        """
        node = self.onnx_node
        inp_values = context[node.input[0]]
        th_val = context[node.input[1]]
        out_bias = self.get_nodeattr("act_val")

        # MultiThreshold expects inputs in (N,C,H,W) or (N,C) format
        # If 4D, input values in context are (N,H,W,C) and need transpose
        # If 2D, inputs can be passed directly
        is_4d = len(inp_values.shape) == 4

        if is_4d:
            inp_values = np.transpose(inp_values, (0, 3, 1, 2))

        # Apply multithreshold
        y = multithreshold(inp_values, th_val, out_bias=out_bias)

        if is_4d:
            y = y.transpose(0, 2, 3, 1)

        # Handle BIPOLAR output (binary to bipolar conversion)
        act = DataType[self.get_nodeattr("output_dtype")]
        if act == DataType["BIPOLAR"]:
            y = 2 * y - 1

        context[node.output[0]] = y.astype(np.float32)

    def verify_node(self):
        """Verify node configuration.

        Returns: List of verification messages
        """
        info_messages = []

        # Verify backend is fpgadataflow
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # Verify necessary attributes exist
        try:
            self.get_nodeattr("PE")
            self.get_nodeattr("num_steps")
            self.get_nodeattr("input_dtype")
            self.get_nodeattr("weight_dtype")
            self.get_nodeattr("output_dtype")
            self.get_nodeattr("act_val")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("Required Thresholding attributes do not exist")

        return info_messages

    def infer_node_datatype(self, model):
        """Infer and update node datatypes.

        Updates input_dtype from model and propagates output_dtype.
        """
        node = self.onnx_node

        # Check input datatype
        idt = model.get_tensor_datatype(node.input[0])
        current_idt = self.get_input_datatype(0)

        if idt != current_idt:
            warn_str = (
                f"input_dtype changing for {node.name}: "
                f"{current_idt.name} -> {idt.name}"
            )
            warnings.warn(warn_str)
            self.set_nodeattr("input_dtype", idt.name)

        # Set output datatype
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def make_shape_compatible_op(self, model):
        """Create a shape-compatible ONNX node.

        Used during shape inference to create a temporary node
        with explicit shape information.
        """
        in_shape = self.get_normal_input_shape(0)
        out_shape = self.get_normal_output_shape(0)

        return helper.make_node(
            "Thresholding",
            inputs=self.onnx_node.input,
            outputs=self.onnx_node.output,
            domain="brainsmith.kernels",
            input_shape=list(in_shape),
            output_shape=list(out_shape)
        )
