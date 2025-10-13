############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
AutoLayerNorm - LayerNorm kernel using AutoHWCustomOp and Dataflow Modeling.

This implementation uses the modern Brainsmith Dataflow Modeling system for:
- Declarative KernelSchema with constraints and relationships
- Automatic shape inference via template resolution
- Intelligent two-level caching (tensor context + kernel model)
- Unified architecture with other Brainsmith kernels
"""

import torch
import numpy as np
import torch.nn.functional as F

from brainsmith.core.finn import AutoHWCustomOp
from brainsmith.core.dataflow import (
    KernelSchema,
    InputSchema,
    OutputSchema,
    DimensionDivisible,
    DatatypeConstraint,
    ShapeHierarchy,
    DerivedDim
)
from brainsmith.core.plugins import kernel


# Module-level KernelSchema definition
LAYERNORM_SCHEMA = KernelSchema(
    name="LayerNorm",
    inputs=[
        InputSchema(
            name="input",
            block_tiling=[":"],               # One LayerNorm op: (1, 1, channels)
            stream_tiling=["SIMD"],           # Stream channels with SIMD parallelism
            datatype_attr="inputDataType",
            constraints=[
                # Validate SIMD divisibility on stream shape
                DimensionDivisible("input", -1, "SIMD", ShapeHierarchy.STREAM),

                # Validate datatype (FLOAT32 only)
                DatatypeConstraint("input", "FLOAT", 32, 32),
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
        "description": "Hardware LayerNorm with epsilon-safe normalization",
        "epsilon_default": 1e-5,
    }
)


@kernel(
    description="Hardware LayerNorm using AutoHWCustomOp and Dataflow Modeling",
    author="Thomas Keller"
)
class AutoLayerNorm(AutoHWCustomOp):
    """LayerNorm implementation using AutoHWCustomOp base class.

    This kernel normalizes activations over the last dimension (channels) with:
        output = (input - mean) / sqrt(variance + epsilon)

    Where mean and variance are computed per-sample over the channel dimension.

    Hardware Implementation:
    - Streams data with SIMD parallelism on channel dimension
    - Single-pass streaming architecture (no weight storage)
    - Epsilon prevents division by zero
    - Each (batch, sequence_position) is an independent LayerNorm operation

    Shape Semantics:
    - TENSOR: Full logical tensor (e.g., [1, 128, 768] = batch × seq_len × channels)
    - BLOCK: One LayerNorm operation (e.g., [1, 1, 768] = all channels for one position)
    - STREAM: Channels folded by SIMD (e.g., [1, 1, 8] with SIMD=8)

    Processing:
    - Each block = one independent LayerNorm (all channels for one seq position)
    - Tensor folding factor = batch × seq_len (number of LayerNorm ops)
    - Block folding factor = channels / SIMD (cycles per LayerNorm op)
    - Total cycles = (batch × seq_len × channels / SIMD) + overhead
    """

    kernel_schema = LAYERNORM_SCHEMA

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Define node attributes for LayerNorm kernel.

        Returns:
            dict: Node attribute specifications
                - SIMD: Parallelization factor for channel dimension
                - epsilon: Small value to prevent division by zero
                - inputDataType: FINN DataType for input (set by tensor context)
                - outputDataType: FINN DataType for output (set by tensor context)
                - exec_mode: Execution mode (python/cppsim/rtlsim)
        """
        my_attrs = super().get_nodeattr_types()
        my_attrs.update({
            "SIMD": ("i", True, 1),
            "epsilon": ("f", True, 1e-5),
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            "exec_mode": ("s", False, "python", {"python", "rtlsim", "cppsim"}),
        })
        return my_attrs

    def execute_node(self, context, graph):
        """Execute LayerNorm in specified execution mode.

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
        """Python functional simulation using PyTorch.

        This provides a reference implementation for validation and testing.
        Normalizes over the last dimension using PyTorch's layer_norm.

        Args:
            context: Execution context containing input/output tensors
            graph: ONNX graph
        """
        node = self.onnx_node
        in_values = context[node.input[0]]

        # Get epsilon from nodeattr
        epsilon = self.get_nodeattr("epsilon")

        # PyTorch LayerNorm over last dimension
        # normalized_shape must be the dimensions to normalize over
        in_tensor = torch.from_numpy(in_values)
        out_tensor = F.layer_norm(
            in_tensor,
            normalized_shape=[in_values.shape[-1]],  # Normalize over channels
            eps=epsilon
        )

        # Store result
        context[node.output[0]] = out_tensor.numpy().astype(np.float32)

    def _execute_cppsim(self, context, graph):
        """C++ simulation execution.

        This will be implemented by the HLS backend in Phase 2.

        Raises:
            NotImplementedError: cppsim requires HLS backend
        """
        raise NotImplementedError(
            "cppsim execution requires HLS backend (Phase 2). "
            "Use exec_mode='python' for functional simulation."
        )

    def _execute_rtlsim(self, context, graph):
        """RTL simulation execution.

        This will be implemented by the HLS/RTL backend in Phase 2.

        Raises:
            NotImplementedError: rtlsim requires RTL backend
        """
        raise NotImplementedError(
            "rtlsim execution requires RTL backend (Phase 2). "
            "Use exec_mode='python' for functional simulation."
        )

    def verify_node(self):
        """Verify node is properly configured.

        Note: Validation is automatic via declarative constraints in KernelSchema.
        The base class AutoHWCustomOp handles constraint checking during model
        building, so this method is a no-op.
        """
        pass

    def get_exp_cycles(self):
        """Expected cycles for LayerNorm computation.

        Formula: (batch * seq_len * channels / SIMD) + overhead

        With correct block tiling:
        - Block: (1, 1, channels) = one LayerNorm operation
        - We have (batch * seq_len) blocks to process
        - Each block takes (channels / SIMD) cycles to stream
        - Total: (batch * seq_len) * (channels / SIMD) + overhead

        This is equivalent to kernel_model.initiation_interval + overhead.

        Returns:
            int: Expected number of clock cycles
        """
        # Use the kernel model's initiation interval
        # This correctly accounts for tensor folding and block folding
        base_cycles = self.kernel_model.initiation_interval

        # Overhead from HLS synthesis profiling
        # Includes: initial latency for mean/variance computation and final normalization
        overhead = 72

        return int(base_cycles + overhead)
