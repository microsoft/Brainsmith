"""Core parity testing for shapes, stream widths, and datatypes.

This module provides CoreParityTest, a focused framework for comparing
manual (FINN) vs auto (Brainsmith) implementations on structural properties:
- Shapes (normal, folded)
- Stream widths (regular, padded)
- Datatypes (input, output, inference)
- Shape compatibility

Design Philosophy:
- Focus on data structure properties, NOT execution correctness
- Compare manual vs auto implementations
- No golden reference needed (structural comparison only)
- Execution testing belongs in IntegratedPipelineTest

Usage:
    class TestMyKernelCoreParity(CoreParityTest):
        def make_test_model(self):
            # Create ONNX node
            return model, node_name

        def get_manual_transform(self):
            return InferMyKernelLayer  # FINN

        def get_auto_transform(self):
            return InferKernelList  # Brainsmith

        def get_num_inputs(self):
            return 2

        def get_num_outputs(self):
            return 1

Inherited Tests (7):
- test_normal_shapes_parity
- test_folded_shapes_parity
- test_stream_widths_parity
- test_stream_widths_padded_parity
- test_datatypes_parity
- test_datatype_inference_parity
- test_make_shape_compatible_op_parity
"""

import pytest
from abc import ABC, abstractmethod
from typing import Tuple, Type

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Import assertion helpers
from tests.parity.assertions import (
    assert_shapes_match,
    assert_widths_match,
    assert_values_match,
    assert_datatypes_match,
    assert_model_tensors_match,
)


class CoreParityTest(ABC):
    """Base class for testing structural parity between manual and auto kernels.

    Compares manual (FINN) vs auto (Brainsmith) implementations on:
    - Normal and folded shapes
    - Stream widths (regular and padded)
    - Input/output datatypes
    - Datatype inference behavior
    - Shape compatibility operations

    Does NOT test:
    - Execution correctness (use IntegratedPipelineTest)
    - Hardware estimation (use HWEstimationParityTest)

    Total: 7 structural parity tests
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement
    # ========================================================================

    @abstractmethod
    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create test ONNX model.

        Returns:
            (model, node_name): Model with ONNX node and its name
        """
        pass

    @abstractmethod
    def get_manual_transform(self) -> Type[Transformation]:
        """Return FINN's manual transform for this kernel.

        Returns:
            Transform class (e.g., InferAddStreamsLayer)
        """
        pass

    @abstractmethod
    def get_auto_transform(self) -> Type[Transformation]:
        """Return Brainsmith's unified transform.

        Returns:
            Transform class (typically InferKernelList)
        """
        pass

    @abstractmethod
    def get_num_inputs(self) -> int:
        """Return number of inputs for this kernel.

        Returns:
            Number of inputs
        """
        pass

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Return number of outputs for this kernel.

        Returns:
            Number of outputs
        """
        pass

    # ========================================================================
    # Optional Configuration Hooks
    # ========================================================================

    def configure_kernel_node(
        self, op: HWCustomOp, model: ModelWrapper, is_manual: bool
    ) -> None:
        """Configure kernel node before testing (optional).

        Override to set node attributes (PE, SIMD, etc.) for testing.

        IMPORTANT: If you change dimension parameters (PE, SIMD, etc.) on a KernelOp,
        you must call op._ensure_ready(model) afterwards to reconfigure the design point.

        Args:
            op: Kernel operator instance
            model: ModelWrapper containing the op
            is_manual: True if manual FINN implementation

        Example:
            def configure_kernel_node(self, op, model, is_manual):
                from brainsmith.dataflow.kernel_op import KernelOp

                op.set_nodeattr("PE", 8)

                # Reset design space after changing dimension parameters
                if isinstance(op, KernelOp):
                    op._ensure_ready(model)
        """
        pass

    # ========================================================================
    # Pipeline Execution Helpers
    # ========================================================================

    def run_manual_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run manual (FINN) pipeline: ONNX → HW node.

        Returns:
            (op, model): Manual operator and model
        """
        from qonnx.transformation.infer_shapes import InferShapes
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from finn.util.basic import getHWCustomOp

        # Create model
        model, node_name = self.make_test_model()

        # Standard preprocessing
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Manual kernel inference
        transform_class = self.get_manual_transform()
        model = model.transform(transform_class())

        # Get hardware node
        hw_node = model.get_node_from_name(node_name)
        if hw_node is None:
            # Transform may rename node
            hw_node = model.graph.node[0]

        op = getHWCustomOp(hw_node, model)

        # Configure if needed
        self.configure_kernel_node(op, model, is_manual=True)

        return op, model

    def run_auto_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run auto (Brainsmith) pipeline: ONNX → HW node.

        Returns:
            (op, model): Auto operator and model
        """
        from qonnx.transformation.infer_shapes import InferShapes
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from finn.util.basic import getHWCustomOp

        # Create model
        model, node_name = self.make_test_model()

        # Standard preprocessing
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Auto kernel inference
        transform_class = self.get_auto_transform()
        model = model.transform(transform_class())

        # Get hardware node
        hw_node = model.get_node_from_name(node_name)
        if hw_node is None:
            # Transform may rename node
            hw_node = model.graph.node[0]

        op = getHWCustomOp(hw_node, model)

        # Configure if needed
        self.configure_kernel_node(op, model, is_manual=False)

        return op, model

    # ========================================================================
    # Core Parity Tests (7 tests)
    # ========================================================================

    @pytest.mark.parity
    @pytest.mark.core
    def test_normal_shapes_parity(self):
        """Test normal input/output shapes match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input shapes
        for i in range(self.get_num_inputs()):
            manual_shape = manual_op.get_normal_input_shape(i)
            auto_shape = auto_op.get_normal_input_shape(i)
            assert_shapes_match(manual_shape, auto_shape, i, "normal input")

        # Output shapes
        for i in range(self.get_num_outputs()):
            manual_shape = manual_op.get_normal_output_shape(i)
            auto_shape = auto_op.get_normal_output_shape(i)
            assert_shapes_match(manual_shape, auto_shape, i, "normal output")

    @pytest.mark.parity
    @pytest.mark.core
    def test_folded_shapes_parity(self):
        """Test folded input/output shapes match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input shapes
        for i in range(self.get_num_inputs()):
            manual_shape = manual_op.get_folded_input_shape(i)
            auto_shape = auto_op.get_folded_input_shape(i)
            assert_shapes_match(manual_shape, auto_shape, i, "folded input")

        # Output shapes
        for i in range(self.get_num_outputs()):
            manual_shape = manual_op.get_folded_output_shape(i)
            auto_shape = auto_op.get_folded_output_shape(i)
            assert_shapes_match(manual_shape, auto_shape, i, "folded output")

    @pytest.mark.parity
    @pytest.mark.core
    def test_stream_widths_parity(self):
        """Test input/output stream widths match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input stream widths
        for i in range(self.get_num_inputs()):
            manual_width = manual_op.get_instream_width(i)
            auto_width = auto_op.get_instream_width(i)
            assert_widths_match(manual_width, auto_width, i, "Input")

        # Output stream widths
        for i in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width(i)
            auto_width = auto_op.get_outstream_width(i)
            assert_widths_match(manual_width, auto_width, i, "Output")

    @pytest.mark.parity
    @pytest.mark.core
    def test_stream_widths_padded_parity(self):
        """Test padded stream widths match (AXI alignment)."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input stream widths padded
        for i in range(self.get_num_inputs()):
            manual_width = manual_op.get_instream_width_padded(i)
            auto_width = auto_op.get_instream_width_padded(i)

            def format_width(w):
                return f"{w} bits (padded)"
            assert_values_match(
                manual_width, auto_width, f"Input {i} stream width", format_width
            )

        # Output stream widths padded
        for i in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width_padded(i)
            auto_width = auto_op.get_outstream_width_padded(i)

            def format_width(w):
                return f"{w} bits (padded)"
            assert_values_match(
                manual_width, auto_width, f"Output {i} stream width", format_width
            )

    @pytest.mark.parity
    @pytest.mark.core
    def test_datatypes_parity(self):
        """Test input/output datatypes match between implementations."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        # Input datatypes
        for i in range(self.get_num_inputs()):
            manual_dt = manual_op.get_input_datatype(i)
            auto_dt = auto_op.get_input_datatype(i)
            assert_datatypes_match(manual_dt, auto_dt, i, "Input")

        # Output datatypes
        for i in range(self.get_num_outputs()):
            manual_dt = manual_op.get_output_datatype(i)
            auto_dt = auto_op.get_output_datatype(i)
            assert_datatypes_match(manual_dt, auto_dt, i, "Output")

    @pytest.mark.parity
    @pytest.mark.core
    def test_datatype_inference_parity(self):
        """Test datatype inference produces matching results in models."""
        # Setup ops with fresh models
        manual_op, manual_model = self.run_manual_pipeline()
        auto_op, auto_model = self.run_auto_pipeline()

        # Run datatype inference
        manual_model_out = manual_op.infer_node_datatype(manual_model)
        auto_model_out = auto_op.infer_node_datatype(auto_model)

        # Use returned model if provided
        if manual_model_out is not None:
            manual_model = manual_model_out
        if auto_model_out is not None:
            auto_model = auto_model_out

        # Verify input datatypes in model
        for i in range(self.get_num_inputs()):
            input_name = manual_op.onnx_node.input[i]
            if not input_name:
                continue

            assert_model_tensors_match(
                manual_model, auto_model, input_name,
                f"After infer_node_datatype, input {i}"
            )

        # Verify output datatypes in model
        for i in range(self.get_num_outputs()):
            output_name = manual_op.onnx_node.output[i]

            assert_model_tensors_match(
                manual_model, auto_model, output_name,
                f"After infer_node_datatype, output {i}"
            )

    @pytest.mark.parity
    @pytest.mark.core
    def test_make_shape_compatible_op_parity(self):
        """Test shape-compatible ops preserve output structure.

        Note: make_shape_compatible_op() returns an ONNX NodeProto (per FINN API),
        not a wrapped HWCustomOp. This is used for shape inference.
        """
        manual_op, manual_model = self.run_manual_pipeline()
        auto_op, auto_model = self.run_auto_pipeline()

        # Returns ONNX NodeProto for shape inference
        manual_compat_node = manual_op.make_shape_compatible_op(manual_model)
        auto_compat_node = auto_op.make_shape_compatible_op(auto_model)

        # Verify output count matches (NodeProto.output is a list of output names)
        assert len(manual_compat_node.output) == len(auto_compat_node.output), (
            f"Shape-compatible op output count mismatch: "
            f"manual={len(manual_compat_node.output)}, "
            f"auto={len(auto_compat_node.output)}"
        )

        # Verify output names match (both should use same output names as original op)
        for i in range(self.get_num_outputs()):
            manual_output_name = manual_compat_node.output[i] if i < len(manual_compat_node.output) else None
            auto_output_name = auto_compat_node.output[i] if i < len(auto_compat_node.output) else None

            assert manual_output_name == manual_op.onnx_node.output[i], (
                f"Manual shape-compatible op output {i} name mismatch"
            )
            assert auto_output_name == auto_op.onnx_node.output[i], (
                f"Auto shape-compatible op output {i} name mismatch"
            )
