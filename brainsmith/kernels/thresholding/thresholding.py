############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Migration to KernelOp by Microsoft Corporation
# Refactored to eliminate redundancy and leverage dataflow system (2025)
############################################################################
# ARETE REFACTORING NOTES:
# - Deleted 116+ lines of redundant code
# - Removed shape/stream methods that duplicate KernelOp base class
# - Standardized nodeattr naming (input0Datatype, input1Datatype, output0Datatype)
# - Simplified infer_from() - direct onnx.helper usage, no abstraction layers
# - Kept only HW-specific logic (TMEM calc, threshold tensor formatting, decoupled mode)
# - Trusts the dataflow system instead of manual reimplementation
############################################################################


import numpy as np
from onnx import NodeProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM, KernelOp
from brainsmith.dataflow.constraints import (
    DatatypeInteger,
    DimensionDivisible,
    IsDynamic,
    IsStatic,
)
from brainsmith.dataflow.spec_helpers import derive_dim
from brainsmith.dataflow.types import VALUE_OPTIMIZED, ShapeHierarchy
from brainsmith.registry import kernel

# =============================================================================
# Thresholding Schema
# =============================================================================

THRESHOLDING_SCHEMA = df.KernelSchema(
    name="Thresholding",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],  # Process full spatial dimensions
            stream_tiling=["PE"],  # Parallel channels with PE
            required_layout="NHWC",  # Hardware requires NHWC layout
        ),
        df.InputSchema(
            name="thresholds",
            # Thresholds are constant weights (NumChannels x num_steps)
            # Not tiled or streamed - full tensor loaded as initializer
            block_tiling=[],  # No block tiling (static data)
            stream_tiling=[],  # Not streamed (static data)
            datatype=VALUE_OPTIMIZED,  # Optimize from actual values
        ),
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],  # Same as input
            stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)],  # Match input PE
            datatype=None,  # Datatype comes from ONNX graph (set via node attrs)
            required_layout="NHWC",
        )
    ],
    # =========================================================================
    # KERNEL PARAMETERS: Threshold-specific configuration
    # =========================================================================
    kernel_params={
        "num_steps": ("i", True, 1),  # Number of threshold steps (required)
        "act_val": ("i", False, 0),  # Activation bias value (ActVal)
        "num_input_vectors": ("ints", False, [1]),  # Batch/spatial dims (legacy)
        "runtime_writeable_weights": ("i", False, 0),  # AXI-lite writable (1/0)
    },
    # =========================================================================
    # VALIDATION: Constraints
    # =========================================================================
    constraints=[
        # Input must be dynamic, thresholds must be static
        IsDynamic(("input",)),
        IsStatic(("thresholds",)),
        # PE must divide number of channels
        DimensionDivisible("input", -1, "PE", hierarchy=df.ShapeHierarchy.STREAM),
        # Datatypes must be integer (enforced in can_infer_from)
        DatatypeInteger(("input", "output")),
    ],
    # Parallelization
)


# =============================================================================
# Thresholding Kernel Implementation
# =============================================================================


@kernel(
    description="Hardware multi-threshold activation (KernelOp-based)",
    author="Microsoft Corporation",
)
class Thresholding(KernelOp):  # → HWCustomOp → CustomOp (inheritance chain)
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
    - Shapes extracted from design_point (not nodeattrs)
    - Declarative constraints in schema
    - Two-phase construction (DesignSpace → Configuration)
    """

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> df.KernelSchema:
        """Build Thresholding schema (constant for all instances)."""
        return THRESHOLDING_SCHEMA

    # ================================================================
    # Inference (Static methods)
    # ================================================================

    @staticmethod
    def can_infer_from(node, model: ModelWrapper) -> bool:
        """Check if MultiThreshold node can convert to Thresholding.

        Only checks source-node attributes that won't be preserved in target node.
        Datatype validation is handled by schema constraint: DatatypeInteger((("input", "output"))).
        """
        if node.op_type != "MultiThreshold":
            return False

        from qonnx.custom_op.registry import getCustomOp

        mt_inst = getCustomOp(node)

        # Check MultiThreshold-specific constraints (not preserved in Thresholding node)
        return mt_inst.get_nodeattr("out_scale") == 1.0 and int(
            mt_inst.get_nodeattr("out_bias")
        ) == mt_inst.get_nodeattr("out_bias")

    @staticmethod
    def infer_from(node, model: ModelWrapper, insert_index: int) -> df.TransformationResult:
        """Convert MultiThreshold node to Thresholding node.

        Extracts and validates MultiThreshold-specific parameters (scale, actval).

        NOTE: Assumes input is already in NHWC layout (preprocessing required).

        Args:
            node: MultiThreshold ONNX node
            model: Model wrapper
            insert_index: Where to insert new node (unused - no layout conversion)

        Returns:
            df.TransformationResult with new Thresholding node
        """
        from qonnx.custom_op.registry import getCustomOp

        # Extract and validate MultiThreshold parameters
        mt_inst = getCustomOp(node)
        scale = mt_inst.get_nodeattr("out_scale")
        actval = mt_inst.get_nodeattr("out_bias")

        if scale != 1.0:
            raise ValueError(
                f"{node.name}: MultiThreshold out_scale must be 1.0 for HW conversion, got {scale}"
            )

        if int(actval) != actval:
            raise ValueError(
                f"{node.name}: MultiThreshold out_bias must be integer for HW conversion, got {actval}"
            )
        actval = int(actval)

        # Validate actval sign for signed outputs
        odt = model.get_tensor_datatype(node.output[0])
        if odt != DataType["BIPOLAR"] and odt.signed() and actval >= 0:
            raise ValueError(f"{node.name}: Signed output requires actval < 0, got {actval}")

        # Get shapes
        thl_thres_shape = model.get_tensor_shape(node.input[1])
        thl_in_shape = model.get_tensor_shape(node.input[0])

        # Create HW node
        hw_node = helper.make_node(
            "Thresholding",
            inputs=list(node.input),
            outputs=list(node.output),
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name=f"Thresholding_{node.name}",
            # Kernel parameters
            num_steps=int(thl_thres_shape[1]),
            act_val=actval,
            num_input_vectors=list(thl_in_shape[:-1]),
            runtime_writeable_weights=0,
        )

        return df.TransformationResult(nodes_to_insert=[hw_node], nodes_to_remove=[node])

    # ================================================================
    # Custom Stream Width (Decoupled Threshold Memory Mode)
    # ================================================================

    def get_instream_width(self, ind=0):
        """Get input stream width in bits.

        Overrides base class for ind=1 to handle decoupled threshold memory mode.
        In decoupled mode, thresholds stream in via AXI-Stream instead of being
        embedded in BRAM.

        For ind=0 (data): Uses base class (PE * input_datatype.bitwidth())
        For ind=1 (thresholds): PE * weight_datatype.bitwidth() * num_steps if decoupled, else 0
        """
        if ind == 0:
            # Use base class implementation
            return super().get_instream_width(ind)
        elif ind == 1:
            # Custom logic for threshold memory modes
            mem_mode = (
                self.get_nodeattr("mem_mode")
                if self.has_nodeattr("mem_mode")
                else "internal_embedded"
            )

            if mem_mode == "internal_decoupled":
                pe = self.get_nodeattr("PE")
                wp = self.get_input_datatype(1).bitwidth()
                n_thres_steps = self.get_nodeattr("num_steps")
                return pe * wp * n_thres_steps
            return 0
        else:
            raise ValueError(f"Invalid input index: {ind}")

    def calc_tmem(self):
        """Calculate TMEM (threshold memory depth).

        Returns: NumChannels // PE
        """
        self.get_
        ki = self.design_point
        num_channels = ki.inputs["input"].tensor_shape[-1]
        pe = self.get_nodeattr("PE")
        return num_channels // pe

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
        ki = self.design_point
        num_channels = ki.inputs["input"].tensor_shape[-1]
        pe = self.get_nodeattr("PE")
        tmem = num_channels // pe

        assert (
            num_channels % pe == 0
        ), f"Requirement NumChannels={num_channels} divisible by PE={pe} is violated."

        assert orig_thres_matrix.ndim == 2, "Threshold matrix dimension is not as expected (2)."

        n_thres_steps = orig_thres_matrix.shape[1]
        assert n_thres_steps == self.get_nodeattr("num_steps"), "Mismatch in threshold steps"

        # For unsigned inputs, ensure all thresholds are nonnegative
        if not self.get_input_datatype(0).signed():
            assert (orig_thres_matrix >= 0).all(), "Unsigned input requires nonnegative thresholds"

        ret = orig_thres_matrix

        # Ensure channels match NumChannels, duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (num_channels, 1))

        assert (
            ret.shape[0] == num_channels
        ), f"Channels of threshold matrix ({ret.shape[0]}) don't match NumChannels ({num_channels})"

        # Distribute rows between PEs (interleaving)
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)

        assert (
            ret.shape[0] == pe
        ), f"First dimension after PE distribution ({ret.shape[0]}) != PE ({pe})"
        assert (
            ret.shape[1] == tmem
        ), f"Second dimension after PE distribution ({ret.shape[1]}) != TMEM ({tmem})"
        assert (
            ret.shape[2] == n_thres_steps
        ), f"Third dimension after PE distribution ({ret.shape[2]}) != numSteps ({n_thres_steps})"

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
        act = self.get_output_datatype(0)
        if act == DataType["BIPOLAR"]:
            y = 2 * y - 1

        context[node.output[0]] = y.astype(np.float32)

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
            output_shape=list(out_shape),
        )
