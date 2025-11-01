# Portions derived from FINN project
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MVAU (Matrix Vector Activation Unit) kernel using modern KernelOp system.

This kernel implements matrix-vector multiplication with optional multi-threshold activation,
following modern Brainsmith patterns:
- MW/MH derived from weight tensor shape (not stored as nodeattrs)
- VALUE_OPTIMIZED for automatic weight datatype minimization
- Pattern C methods for hardware-specific operations
- Access shapes via ModelWrapper, not nodeattrs

Migrated from FINN's MatrixVectorActivation kernel.
"""

import math
import numpy as np
import warnings
from onnx import NodeProto, helper
from typing import Optional

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.custom_op.general.xnorpopcount as xp
from qonnx.custom_op.general.multithreshold import multithreshold
import onnx.numpy_helper as np_helper

from brainsmith.dataflow import KernelOp
import brainsmith.dataflow as df
from brainsmith.registry import kernel
from .mvau_schema import MVAU_SCHEMA


@kernel(
    description="Matrix-Vector Activation Unit (MVAU/MVTU) with PE/SIMD parallelism",
    author="Thomas Keller (migrated from AMD FINN)"
)
class MVAU(KernelOp):
    """Hardware kernel for matrix-vector multiplication with optional activation.

    Modern KernelOp implementation following Arete principles:
    - Shapes extracted from ModelWrapper (not stored)
    - VALUE_OPTIMIZED datatypes
    - Pattern C hardware methods (calc_wmem, minimize_accumulator_width, etc.)

    Modes:
    - noActivation=1: MV (matmul only, 2 inputs)
    - noActivation=0: MVTU (matmul + multi-threshold activation, 3 inputs)
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # ====================================================================
    # Schema (Required by KernelOp)
    # ====================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> df.KernelSchema:
        """Build MVAU schema (constant for all instances)."""
        return MVAU_SCHEMA

    # ====================================================================
    # Hardware-Specific Methods (Pattern C Retention)
    # ====================================================================

    def calc_wmem(self) -> int:
        """Calculate weight memory depth.

        HW-Specific Logic: WMEM = (MW × MH) / (PE × SIMD)
                         = cycles to stream one block

        Uses design_point.block_folding_factor (modern pattern).
        """
        weight_interface = self.design_point.inputs["weights"]
        return weight_interface.block_folding_factor

    def calc_tmem(self) -> int:
        """Calculate threshold memory depth.

        HW-Specific Logic: TMEM = MH / PE (only if noActivation=0)
                         = cycles to stream output channels

        Uses design_point.stream_cycles_shape (modern pattern).
        """
        if self.get_nodeattr("noActivation") == 1:
            return 0

        output_interface = self.design_point.outputs["output"]
        return output_interface.stream_cycles_shape[-1]

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
        """Convert weight matrix to HW-compatible format.

        HW-Specific Logic:
        1. Transpose (ONNX uses (MW, MH), HLS uses (MH, MW))
        2. Convert bipolar {-1,+1} to binary {0,1} if needed
        3. Interleave rows between PEs
        4. Reshape to (1, PE, WMEM, SIMD)
        5. Reverse SIMD dimension (HLS convention)

        Delegates to helper function with design_point-extracted parameters.

        Args:
            orig_weight_matrix: Weight matrix in ONNX format (MW, MH)

        Returns:
            Formatted weight tensor (1, PE, WMEM, SIMD)
        """
        from .mvau_helpers import format_weight_tensor_for_hw

        # Extract dimensions from design_point (modern pattern)
        dp = self.design_point
        pe = dp.outputs["output"].stream_shape[-1]
        simd = dp.inputs["weights"].stream_shape[0] if dp.inputs["weights"].stream_shape else 1
        wmem = self.calc_wmem()

        # Check if bipolar datatype
        is_bipolar = (self.get_input_datatype(1) == DataType["BIPOLAR"])

        # Delegate to helper (single source of truth)
        return format_weight_tensor_for_hw(
            orig_weight_matrix,
            pe=pe,
            simd=simd,
            wmem=wmem,
            is_bipolar=is_bipolar
        )

    def get_hw_compatible_threshold_tensor(self, orig_threshold_matrix):
        """Convert threshold matrix to HW-compatible format.

        HW-Specific Logic:
        1. Ensure positive thresholds for bipolar×bipolar (checked here)
        2. Broadcast if single channel
        3. Interleave rows between PEs
        4. Reshape to (1, PE, TMEM, n_thres_steps)

        Delegates to helper function with design_point-extracted parameters.

        Args:
            orig_threshold_matrix: Threshold matrix (MH or 1, n_thres_steps)

        Returns:
            Formatted threshold tensor (1, PE, TMEM, n_thres_steps)
        """
        from .mvau_helpers import format_threshold_tensor_for_hw

        # Extract dimensions from design_point (modern pattern)
        dp = self.design_point
        pe = dp.outputs["output"].stream_shape[-1]
        tmem = self.calc_tmem()
        mh = dp.outputs["output"].block_shape[-1]

        # Delegate to helper (single source of truth)
        return format_threshold_tensor_for_hw(
            orig_threshold_matrix,
            pe=pe,
            tmem=tmem,
            mh=mh
        )

    # Modern Pattern: Datatype computation moved to schema resolvers
    # - accDataType: Computed by _mvau_accumulator_datatype_resolver() in schema
    # - weightDataType: Computed by VALUE_OPTIMIZED in schema
    # - outputDataType: Computed by _mvau_output_datatype_resolver() in schema
    # No manual minimize_* methods needed - schema handles it declaratively!

    # ====================================================================
    # Execution (CPU Reference Implementation)
    # ====================================================================

    def execute_node(self, context, graph):
        """Execute MVAU on CPU for testing/validation.

        Performs matrix multiplication with optional multi-threshold activation.

        Modes:
        - noActivation=1: MV mode (matmul only)
        - noActivation=0: MVTU mode (matmul + multi-threshold activation)

        Args:
            context: Execution context with tensor values
            graph: ONNX graph (for accessing initializers)
        """
        node = self.onnx_node

        # Get input activations
        in_act = context[node.input[0]]

        # Get weights from initializer (static only)
        mvau_w_init = [x for x in graph.initializer if x.name == node.input[1]][0]
        mvau_w = np_helper.to_array(mvau_w_init)

        # Regular matmul (binaryXnorMode=0)
        # TODO Phase 3: Add bipolar/binary optimizations
        result = np.matmul(in_act, mvau_w)

        # Apply multi-threshold activation if enabled
        if self.get_nodeattr("noActivation") == 0:
            # Get thresholds from initializer
            thres_init = [x for x in graph.initializer if x.name == node.input[2]][0]
            thres = np_helper.to_array(thres_init)

            # Apply multi-threshold activation
            # multithreshold expects (data, thresholds) and returns quantized output
            out_dtype = self.get_output_datatype()
            result = multithreshold(result, thres)

            # Ensure output datatype is correct
            if out_dtype != DataType["INT32"]:
                result = result.astype(np.float32)

        # Store result
        oshape = context[node.output[0]].shape
        context[node.output[0]] = result.reshape(oshape)

    # ====================================================================
    # ONNX Inference (can_infer_from / infer_from)
    # ====================================================================

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if ONNX node can be converted to MVAU kernel."""
        if node.op_type != "MatMul":
            return False

        if len(node.input) != 2:
            return False

        return True

    @classmethod
    def infer_from(
        cls,
        node: NodeProto,
        model: ModelWrapper,
        insert_index: int
    ) -> df.TransformationResult:
        """Create MVAU HW node from ONNX MatMul node.

        Detects two patterns:
        1. MatMul only → MVAU with noActivation=1 (MV mode)
        2. MatMul → MultiThreshold → MVAU with noActivation=0 (MVTU mode)

        Note: MW, MH are NOT stored as nodeattrs - they're derived from
        the weight tensor shape at runtime.

        Args:
            node: ONNX MatMul node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes (unused)

        Returns:
            TransformationResult with MVAU node and removed nodes
        """
        # Check if MatMul output feeds into MultiThreshold
        consumers = model.find_consumers(node.output[0])
        has_multithreshold = False
        multithreshold_node = None

        if consumers and len(consumers) == 1:
            consumer = consumers[0]
            if consumer.op_type == "MultiThreshold":
                has_multithreshold = True
                multithreshold_node = consumer

        # Determine mode and parameters
        if has_multithreshold:
            # MVTU mode: MatMul + MultiThreshold
            hw_inputs = [
                node.input[0],                      # input activations
                node.input[1],                      # weights
                multithreshold_node.input[1],       # thresholds
            ]
            hw_outputs = [multithreshold_node.output[0]]
            nodes_to_remove = [node, multithreshold_node]
            no_activation = 0

            # Extract ActVal from MultiThreshold
            act_val = 0
            if hasattr(multithreshold_node, 'attribute'):
                for attr in multithreshold_node.attribute:
                    if attr.name == "out_dtype":
                        # ActVal = log2(max_value + 1)
                        out_dtype_str = attr.s.decode('utf-8')
                        out_dtype = DataType[out_dtype_str]
                        if out_dtype.bitwidth() > 0:
                            act_val = out_dtype.bitwidth()
                        break

        else:
            # MV mode: MatMul only
            hw_inputs = list(node.input)
            hw_outputs = list(node.output)
            nodes_to_remove = [node]
            no_activation = 1
            act_val = 0

        # Create MVAU node
        hw_node = helper.make_node(
            "MVAU",
            inputs=hw_inputs,
            outputs=hw_outputs,
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name=f"MVAU_{node.name}",

            # Kernel params (STRUCTURAL - fixed at inference)
            noActivation=no_activation,
            binaryXnorMode=0,                   # Phase 2: regular matmul
            mem_mode="internal_embedded",       # Phase 2: simplest mode
            dynamic_input=0,
            ActVal=act_val,
            runtime_writeable_weights=0,
            pumpedMemory=0,
        )

        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=nodes_to_remove,
        )

    # ====================================================================
    # Shape Inference Support
    # ====================================================================

    def make_shape_compatible_op(self, model):
        """Create shape-compatible ONNX node for shape inference.

        MVAU behaves like MatMul for shape purposes, so return a MatMul node.
        """
        node = self.onnx_node
        return helper.make_node(
            "MatMul",
            inputs=[node.input[0], node.input[1]],
            outputs=node.output,
            name=f"{node.name}_shape_compat"
        )
