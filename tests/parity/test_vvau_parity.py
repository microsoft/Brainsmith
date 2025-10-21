"""Parity tests for VVAU: Legacy VVAU vs Brainsmith VectorVectorActivation.

This module validates equivalence between:
- Legacy implementation: VVAU (from vectorvectoractivation.py)
- Modern implementation: VectorVectorActivation (KernelOp-based)

The test validates the migration from legacy HWCustomOp pattern to modern
KernelOp system for depthwise convolution with activation.

Test Pattern:
    Input: [1, 8, 8, 64] (NHWC: batch × height × width × channels)
    Kernel: [3, 3] (depthwise 3×3 conv)
    Output: [1, 8, 8, 64] (same dimensions with depthwise filtering)
    PE: 4, SIMD: 8
    Datatype: INT8

Test Coverage:
- 25 base tests (shapes, datatypes, streams, execution, resources)
- 7 computational tests (memory, accumulators, tensors, files)
Total: 32 comprehensive parity tests
"""

import pytest
import numpy as np
from typing import Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.custom_op.registry import getCustomOp
from onnx import helper, TensorProto
import onnx.numpy_helper as np_helper

from tests.parity import ParityTestBase, ComputationalParityMixin
from brainsmith.kernels.vvau import VectorVectorActivation


class TestVVAUParity(ParityTestBase, ComputationalParityMixin):
    """Parity tests for VVAU kernel implementations.

    Validates equivalence between legacy VVAU and modern VectorVectorActivation
    using both ParityTestBase (25 tests) and ComputationalParityMixin (7 tests).

    Test Configuration:
    - Input shape: [1, 8, 8, 64] (NHWC format)
    - Kernel: 3×3 depthwise convolution
    - PE: 4 (channel parallelism)
    - SIMD: 8 (kernel element parallelism)
    - Input datatype: INT8
    - Weight datatype: INT8
    - Output datatype: INT8
    - Activation: Multi-threshold quantization
    """

    @property
    def manual_op_class(self):
        """Legacy VVAU implementation."""
        from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU
        return VVAU

    @property
    def auto_op_class(self):
        """Modern VectorVectorActivation (KernelOp) implementation."""
        return VectorVectorActivation

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX model for VVAU testing.

        Creates a model with depthwise convolution pattern that will be
        inferred as VVAU.

        Returns:
            Tuple of (ModelWrapper, node_name)
        """
        # Test configuration
        batch_size = 1
        dim_h, dim_w = 8, 8
        channels = 64
        k_h, k_w = 3, 3

        input_shape = [batch_size, dim_h, dim_w, channels]
        output_shape = [batch_size, dim_h, dim_w, channels]

        # Create weight tensor for depthwise conv (C, 1, K_H, K_W)
        # Depthwise conv has one filter per input channel
        weight_shape = [channels, 1, k_h, k_w]
        weights = np.random.randint(-128, 127, size=weight_shape, dtype=np.int8)

        # Create threshold tensor for multi-threshold activation
        # Shape: (C, T) where T is number of activation thresholds
        num_thresholds = 15  # For 4-bit output (2^4 - 1 thresholds)
        threshold_shape = [channels, num_thresholds]
        # Generate thresholds in ascending order per channel
        thresholds = np.sort(
            np.random.randint(-128, 127, size=threshold_shape, dtype=np.int32),
            axis=1
        )

        # Create ONNX model with Conv node (depthwise pattern)
        # This will be transformed to VVAU by inference
        conv_node = helper.make_node(
            "Conv",
            inputs=["input", "weights"],
            outputs=["conv_out"],
            domain="",
            name="Conv_depthwise",
            kernel_shape=[k_h, k_w],
            pads=[1, 1, 1, 1],  # SAME padding
            strides=[1, 1],
            group=channels  # Depthwise: one group per channel
        )

        # MultiThreshold node for activation
        mt_node = helper.make_node(
            "MultiThreshold",
            inputs=["conv_out", "thresholds"],
            outputs=["output"],
            domain="qonnx.custom_op.general",
            name="MultiThreshold_0",
            out_dtype="INT8",
            out_scale=1.0,
            out_bias=0.0
        )

        # Create graph
        graph = helper.make_graph(
            nodes=[conv_node, mt_node],
            name="vvau_parity_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
            ],
            initializer=[
                np_helper.from_array(weights.astype(np.float32), name="weights"),
                np_helper.from_array(thresholds.astype(np.float32), name="thresholds")
            ]
        )

        model_proto = helper.make_model(graph, producer_name="parity-test")
        model = ModelWrapper(model_proto)

        # Set tensor datatypes
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("weights", DataType["INT8"])
        model.set_tensor_datatype("conv_out", DataType["INT32"])
        model.set_tensor_datatype("thresholds", DataType["INT32"])
        model.set_tensor_datatype("output", DataType["INT8"])

        return model, "Conv_depthwise"

    def get_manual_transform(self):
        """Return transform for legacy VVAU creation.

        Note: Legacy VVAU uses ConvertToHWLayers which includes
        depthwise conv inference. For testing, we'll override
        setup_manual_op to create VVAU directly.
        """
        return None

    def get_auto_transform(self):
        """Return None - VectorVectorActivation uses unified inference."""
        return None

    def setup_manual_op(self):
        """Create legacy VVAU node directly for testing."""
        from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU

        # Test configuration
        batch_size = 1
        dim_h, dim_w = 8, 8
        channels = 64
        k_h, k_w = 3, 3
        pe = 4
        simd = 8

        input_shape = [batch_size, dim_h, dim_w, channels]
        output_shape = [batch_size, dim_h, dim_w, channels]

        # Create weight tensor (C, 1, K_H, K_W)
        weight_shape = [channels, 1, k_h, k_w]
        weights = np.random.randint(-128, 127, size=weight_shape, dtype=np.int8).astype(np.float32)

        # Create threshold tensor (C, T)
        num_thresholds = 15
        threshold_shape = [channels, num_thresholds]
        thresholds = np.sort(
            np.random.randint(-128, 127, size=threshold_shape, dtype=np.int32),
            axis=1
        ).astype(np.float32)

        # Create VVAU node
        vvau_node = helper.make_node(
            "VVAU",
            inputs=["input", "weights", "thresholds"],
            outputs=["output"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            name="VVAU_manual",

            # Legacy VVAU attributes
            PE=pe,
            SIMD=simd,
            Dim=[dim_h, dim_w],
            Channels=channels,
            Kernel=[k_h, k_w],
            resType="lut",
            ActVal=num_thresholds,
            inputDataType="INT8",
            weightDataType="INT8",
            outputDataType="INT8",
            accDataType="INT32",
            noActivation=0,
            mem_mode="internal_decoupled",
            runtime_writeable_weights=0,
            ram_style="auto"
        )

        # Create model
        graph = helper.make_graph(
            nodes=[vvau_node],
            name="vvau_parity_manual",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
            ],
            initializer=[
                np_helper.from_array(weights, name="weights"),
                np_helper.from_array(thresholds, name="thresholds")
            ]
        )

        model_proto = helper.make_model(graph, producer_name="parity-test")
        model = ModelWrapper(model_proto)

        # Set datatypes
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("weights", DataType["INT8"])
        model.set_tensor_datatype("thresholds", DataType["INT32"])
        model.set_tensor_datatype("output", DataType["INT8"])

        # Create op instance
        op = getCustomOp(vvau_node)

        return op, model

    def setup_auto_op(self):
        """Create modern VectorVectorActivation using direct instantiation."""
        # Test configuration (same as manual)
        batch_size = 1
        dim_h, dim_w = 8, 8
        channels = 64
        k_h, k_w = 3, 3
        pe = 4
        simd = 8

        input_shape = [batch_size, dim_h, dim_w, channels]
        output_shape = [batch_size, dim_h, dim_w, channels]

        # Create same weight and threshold tensors
        weight_shape = [channels, 1, k_h, k_w]
        weights = np.random.randint(-128, 127, size=weight_shape, dtype=np.int8).astype(np.float32)

        num_thresholds = 15
        threshold_shape = [channels, num_thresholds]
        thresholds = np.sort(
            np.random.randint(-128, 127, size=threshold_shape, dtype=np.int32),
            axis=1
        ).astype(np.float32)

        # Create VectorVectorActivation node (modern KernelOp)
        vvau_node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights", "thresholds"],
            outputs=["output"],
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name="VVAU_auto",

            # Modern attributes (schema-driven, no shape storage)
            PE=pe,
            SIMD=simd,
            Kernel=[k_h, k_w],
            resType="lut",
            noActivation=0,
            mem_mode="internal_decoupled",
            runtime_writeable_weights=0,
            ram_style="auto"
        )

        # Create model
        graph = helper.make_graph(
            nodes=[vvau_node],
            name="vvau_parity_auto",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
            ],
            initializer=[
                np_helper.from_array(weights, name="weights"),
                np_helper.from_array(thresholds, name="thresholds")
            ]
        )

        model_proto = helper.make_model(graph, producer_name="parity-test")
        model = ModelWrapper(model_proto)

        # Set tensor shapes
        model.set_tensor_shape("input", input_shape)
        model.set_tensor_shape("weights", weight_shape)
        model.set_tensor_shape("thresholds", threshold_shape)
        model.set_tensor_shape("output", output_shape)

        # Set datatypes
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("weights", DataType["INT8"])
        model.set_tensor_datatype("thresholds", DataType["INT32"])
        model.set_tensor_datatype("output", DataType["INT8"])

        # Note: We don't run InferShapes/InferDataTypes here because
        # VectorVectorActivation has 3 inputs with different shapes,
        # and KernelOp.make_shape_compatible_op() raises NotImplementedError
        # for this case. Since we're manually setting shapes and datatypes,
        # we can skip shape inference for testing.

        # Instantiate VectorVectorActivation directly
        op = VectorVectorActivation(vvau_node)

        # Note: We don't call build_design_space() for parity testing.
        # For unit tests, we just need the op instance to be functional
        # for shape/datatype/execution queries, which works without DSE.

        return op, model

    def configure_test_op(self, op, model, is_auto):
        """Configure op for testing.

        Both legacy and modern already have PE and SIMD set during
        node creation, so no additional configuration needed.
        """
        pass
