############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Softmax - Hardware Softmax kernel using AutoHWCustomOp and Dataflow Modeling.

This implementation uses the modern Brainsmith Dataflow Modeling system for:
- Declarative KernelSchema with constraints and relationships
- Automatic shape inference via template resolution
- Intelligent two-level caching (tensor context + kernel model)
- Unified architecture with other Brainsmith kernels
"""

import numpy as np
from scipy.special import softmax
from onnx.helper import make_node

from brainsmith.core.finn import AutoHWCustomOp
from brainsmith.core.dataflow import (
    KernelSchema,
    InputSchema,
    OutputSchema,
    DimensionDivisible,
    ShapeHierarchy,
    DerivedDim
)
from brainsmith.core.plugins import kernel


# Module-level KernelSchema definition
SOFTMAX_SCHEMA = KernelSchema(
    name="Softmax",
    inputs=[
        InputSchema(
            name="input",
            block_tiling=[":"],               # One Softmax op: (1, 1, channels)
            stream_tiling=["SIMD"],           # Stream channels with SIMD parallelism
            datatype_attr="inputDataType",
            constraints=[
                # Validate SIMD divisibility on stream shape
                DimensionDivisible("input", -1, "SIMD", ShapeHierarchy.STREAM),
            ]
        )
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=[":"],               # Same as input: (1, 1, channels)
            stream_tiling=[DerivedDim("input", -1)],  # Output streams at same rate as input
            datatype_attr="outputDataType",
        )
    ],
    metadata={
        "description": "Hardware Softmax with 3-stage pipeline (max → exp → div)",
        "output_type": "FLOAT32",  # Softmax always outputs float probabilities
    }
)


@kernel(
    description="Hardware Softmax using AutoHWCustomOp and Dataflow Modeling",
    author="Thomas Keller"
)
class Softmax(AutoHWCustomOp):
    """Softmax implementation using AutoHWCustomOp base class.

    This kernel computes the softmax function over the last dimension (channels):
        output[i] = exp(input[i]) / sum(exp(input[j]) for all j)

    With numerical stability via max subtraction:
        output[i] = exp(input[i] - max) / sum(exp(input[j] - max) for all j)

    Hardware Implementation:
    - Streams data with SIMD parallelism on channel dimension
    - 3-stage pipeline: max extraction → exponentiation → division
    - Handles infinities explicitly (distributes probability among inf values)
    - Fixed-point accumulation for exponential sum
    - Each (batch, sequence_position) is an independent Softmax operation

    Shape Semantics:
    - TENSOR: Full logical tensor (e.g., [1, 128, 768] = batch × seq_len × channels)
    - BLOCK: One Softmax operation (e.g., [1, 1, 768] = all channels for one position)
    - STREAM: Channels folded by SIMD (e.g., [1, 1, 8] with SIMD=8)

    Processing:
    - Each block = one independent Softmax (all channels for one seq position)
    - Tensor folding factor = batch × seq_len (number of Softmax ops)
    - Block folding factor = channels / SIMD (cycles per Softmax op)
    - Total cycles = (batch × seq_len × channels / SIMD) + overhead
    """

    kernel_schema = SOFTMAX_SCHEMA

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Define node attributes for Softmax kernel.

        Returns:
            dict: Node attribute specifications
                - SIMD: Parallelization factor for channel dimension
                - inputDataType: FINN DataType for input (set by tensor context)
                - outputDataType: FINN DataType for output (set by tensor context)
                - exec_mode: Execution mode (python/cppsim/rtlsim)
        """
        my_attrs = super().get_nodeattr_types()
        my_attrs.update({
            "SIMD": ("i", True, 1),
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            "exec_mode": ("s", False, "python", {"python", "rtlsim", "cppsim"}),
        })
        return my_attrs

    def execute_node(self, context, graph):
        """Execute Softmax in specified execution mode.

        Args:
            context: Execution context containing tensor values
            graph: ONNX graph

        Raises:
            ValueError: If exec_mode is not supported
        """
        mode = self.get_nodeattr("exec_mode")

        if mode == "python":
            self._execute_python(context, graph)
        elif mode == "cppsim":
            self._execute_cppsim(context, graph)
        elif mode == "rtlsim":
            self._execute_rtlsim(context, graph)
        else:
            raise ValueError(f"Unsupported exec_mode: {mode}")

    def _execute_python(self, context, graph):
        """Python functional simulation using scipy.

        This provides a reference implementation for validation and testing.
        Applies softmax over the last dimension.

        Args:
            context: Execution context containing input/output tensors
            graph: ONNX graph
        """
        node = self.onnx_node
        input_data = context[node.input[0]]

        # scipy.special.softmax with axis=-1 (last dimension)
        # Handles numerical stability automatically
        output_data = softmax(input_data, axis=-1)

        # Store result as float32
        context[node.output[0]] = output_data.astype(np.float32)

    def _execute_cppsim(self, context, graph):
        """C++ simulation execution.

        This will be implemented by the HLS backend.

        Raises:
            NotImplementedError: cppsim requires HLS backend
        """
        raise NotImplementedError(
            "cppsim execution requires HLS backend. "
            "Use exec_mode='python' for functional simulation."
        )

    def _execute_rtlsim(self, context, graph):
        """RTL simulation execution.

        This will be implemented by the HLS/RTL backend.

        Raises:
            NotImplementedError: rtlsim requires RTL backend
        """
        raise NotImplementedError(
            "rtlsim execution requires RTL backend. "
            "Use exec_mode='python' for functional simulation."
        )

    def verify_node(self):
        """Verify node is properly configured.

        Note: Validation is automatic via declarative constraints in KernelSchema.
        The base class AutoHWCustomOp handles constraint checking during model
        building, so this method is a no-op.
        """
        pass

    def make_shape_compatible_op(self, model):
        """Create shape-compatible Softmax operation for InferShapes transformation.

        Overrides the default AutoHWCustomOp behavior (which returns RandomNormal)
        to return an actual Softmax ONNX node. This matches the manual implementation
        pattern and provides more semantically meaningful shape inference.

        Note: This is called during InferShapes BEFORE tensor context initialization,
        so it must access shapes directly from model rather than via kernel_model.

        Args:
            model: ModelWrapper instance

        Returns:
            ONNX Softmax node with same inputs/outputs as this node
        """
        # Create an ONNX Softmax node with the same I/O as this node
        # Softmax normalizes over axis=-1 by default
        # No need to get shape - ONNX Softmax infers it automatically
        return make_node(
            "Softmax",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            axis=-1  # Normalize over last dimension (channels)
        )

    def get_exp_cycles(self):
        """Expected cycles for Softmax computation.

        Formula: (batch * seq_len * channels / SIMD) + overhead

        With correct block tiling:
        - Block: (1, 1, channels) = one Softmax operation
        - We have (batch * seq_len) blocks to process
        - Each block takes (channels / SIMD) cycles to stream
        - Total: (batch * seq_len) * (channels / SIMD) + overhead

        This is equivalent to kernel_model.initiation_interval + overhead.

        The overhead accounts for:
        - Pipeline latency (3 stages: max → exp → div)
        - Data dependencies between stages
        - HLS synthesis optimizations

        Returns:
            int: Expected number of clock cycles
        """
        # Use the kernel_model's initiation interval
        # This correctly accounts for tensor folding and block folding
        base_cycles = self.kernel_model.initiation_interval

        # Overhead from HLS synthesis profiling
        # Softmax has 3-stage pipeline with dataflow
        # Latency depends on N (vector size) and SIMD
        # Conservative estimate: 3 * (N/SIMD) for pipeline fill/drain
        input_model = self.kernel_model.inputs[0]
        channels = input_model.tensor_shape[-1]
        simd = input_model.stream_shape[-1]

        # Pipeline overhead: ~3x the time for one vector to flow through
        overhead = 3 * (channels // simd)

        return int(base_cycles + overhead)
