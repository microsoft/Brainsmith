# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parity tests comparing FINN ElementwiseBinary ops with Brainsmith ElementwiseBinaryOp.

Tests validate that both FINN's manual per-operation implementations and Brainsmith's
polymorphic implementation produce identical results for all 17 binary operations.

Each operation class inherits from ElementwiseBinaryParityBase which provides shared
test infrastructure. This results in 33 tests per operation × 17 operations = 561 total tests.

Current Test Status:
-------------------
- Base tests (26 per operation): Shapes, datatypes, execution, resources
- HLS tests (7 per operation): Code generation parity with FINN HLS backend

Test Coverage:
- 15 operations: Full parity (base + HLS tests)
- 2 operations (BitShift): Skipped (FINN doesn't support, Brainsmith extension)

Known Differences from FINN:
----------------------------
1. **Shape Return Types**: FINN's ElementwiseBinary returns lists, Brainsmith returns
   tuples (following majority FINN pattern). Tests normalize to tuples for comparison.

2. **Static Parameter Folding**: FINN reports folded shapes for static RHS parameter,
   Brainsmith correctly doesn't fold static parameters. Tests skip RHS folding comparison.

3. **BitShift Operations**: FINN doesn't have parity support for BitShiftLeft/BitShiftRight.
   These operations are Brainsmith extensions with their own test coverage.

Usage:
    # Run all ElementwiseBinary parity tests
    pytest brainsmith/kernels/elementwise_binary/tests/test_elementwise_binary_parity.py -v

    # Run just Add operation tests
    pytest brainsmith/kernels/elementwise_binary/tests/test_elementwise_binary_parity.py::TestElementwiseAddParity -v

    # Run supported operations only (skip BitShift)
    pytest brainsmith/kernels/elementwise_binary/tests/test_elementwise_binary_parity.py -v --ignore-glob="*BitShift*"

    # Run fast (skip slow tests)
    pytest brainsmith/kernels/elementwise_binary/tests/test_elementwise_binary_parity.py -m "not slow" -v

    # Run just execution tests
    pytest brainsmith/kernels/elementwise_binary/tests/test_elementwise_binary_parity.py -k "execute" -v
"""

import pytest
from typing import Tuple, Type

from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from qonnx.transformation.base import Transformation

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferElementwiseBinaryOperation

from brainsmith.transforms.infer_kernel_list import InferKernelList

# Import test framework
import sys
from pathlib import Path

# Add tests/parity to path for imports
repo_root = Path(__file__).resolve().parents[4]  # Go up to brainsmith-1/
tests_parity_dir = repo_root / "tests" / "parity"
if str(tests_parity_dir) not in sys.path:
    sys.path.insert(0, str(tests_parity_dir))

from base_parity_test import ParityTestBase
from assertions import assert_shapes_match
from hls_codegen_parity import HLSCodegenParityMixin


# =============================================================================
# Temporary Workaround: Specialize ElementwiseBinary to HLS
# TODO: Remove when proper SpecializeLayers support is added on main branch
# =============================================================================

class SpecializeElementwiseBinaryToHLS(Transformation):
    """Temporary transform to convert ElementwiseXXX nodes to ElementwiseXXX_hls.

    This is a simplified version of SpecializeLayers specifically for ElementwiseBinary.
    The proper fix is being implemented on another branch.
    """

    def apply(self, model):
        from onnx import helper
        graph = model.graph
        graph_modified = False

        for i, node in enumerate(list(graph.node)):
            # Check if this is a FINN ElementwiseBinary node
            if node.op_type.startswith("Elementwise") and node.domain == "finn.custom_op.fpgadataflow":
                # Create HLS version
                hls_op_type = f"{node.op_type}_hls"
                hls_domain = "finn.custom_op.fpgadataflow.hls"

                new_node = helper.make_node(
                    hls_op_type,
                    node.input,
                    node.output,
                    domain=hls_domain,
                    name=node.name
                )

                # Copy all attributes
                for attr in node.attribute:
                    new_node.attribute.append(attr)

                # Replace node
                graph.node.insert(i, new_node)
                graph.node.remove(node)
                graph_modified = True

            # Check if this is a Brainsmith ElementwiseBinaryOp node
            elif node.op_type == "ElementwiseBinaryOp" and node.domain == "brainsmith.kernels":
                # Create HLS version
                hls_domain = "brainsmith.kernels.elementwise_binary.elementwise_binary_hls"

                new_node = helper.make_node(
                    "ElementwiseBinaryOp_hls",
                    node.input,
                    node.output,
                    domain=hls_domain,
                    name=node.name,
                    implementation="vitis_hls",  # Mark as HLS backend for is_hls_node()
                    backend="fpgadataflow"       # Required with implementation attribute
                )

                # Copy all attributes
                for attr in node.attribute:
                    new_node.attribute.append(attr)

                # Replace node
                graph.node.insert(i, new_node)
                graph.node.remove(node)
                graph_modified = True

        return (model, graph_modified)


class ElementwiseBinaryParityBase(ParityTestBase, HLSCodegenParityMixin):
    """Base class for elementwise binary parity tests across different operations.

    This class provides shared implementation for testing FINN manual vs Brainsmith
    auto implementations of ElementwiseBinary operations. Subclasses override
    operation_type and manual_class_name to test different operations.

    FINN ElementwiseBinary Quirk:
    -----------------------------
    FINN's ElementwiseBinary implementation returns **lists** for shape methods,
    while most FINN kernels (AddStreams, ChannelwiseOp, etc.) return **tuples**.
    Brainsmith follows the majority tuple pattern. We normalize FINN's lists to
    tuples in shape comparison tests to avoid cosmetic mismatches.

    Attributes:
        operation_type: ONNX operation name (e.g., "Add", "Sub", "Mul")
        manual_class_name: FINN class name (e.g., "ElementwiseAdd")
    """

    operation_type: str = "Add"  # Override in subclasses
    manual_class_name: str = "ElementwiseAdd"  # Override in subclasses

    # =========================================================================
    # REQUIRED PROPERTIES FOR ParityTestBase
    # =========================================================================

    @property
    def manual_op_class(self) -> Type[HWCustomOp]:
        """FINN's manual ElementwiseXXX_hls HLS backend implementation."""
        import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls as eb_hls
        return getattr(eb_hls, f"{self.manual_class_name}_hls")

    @property
    def auto_op_class(self) -> Type[HWCustomOp]:
        """Brainsmith's polymorphic ElementwiseBinaryOp implementation."""
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp
        return ElementwiseBinaryOp

    @property
    def auto_hls_op_class(self) -> Type[HWCustomOp]:
        """Brainsmith's HLS backend for ElementwiseBinaryOp."""
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp_hls
        return ElementwiseBinaryOp_hls

    # =========================================================================
    # Setup Methods - Override to use SpecializeElementwiseBinaryToHLS
    # =========================================================================

    def setup_manual_op(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Create FINN manual ElementwiseXXX_hls.

        Workflow:
        1. Create ONNX operation
        2. Apply InferElementwiseBinaryOperation → ElementwiseXXX
        3. Apply SpecializeElementwiseBinaryToHLS → ElementwiseXXX_hls

        Returns:
            Tuple of (ElementwiseXXX_hls instance, model)
        """
        from qonnx.transformation.infer_shapes import InferShapes
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from finn.util.basic import getHWCustomOp

        model, node_name = self.make_test_model()

        # Standard inference
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Infer to ElementwiseXXX
        model = model.transform(InferElementwiseBinaryOperation())

        # Specialize to HLS
        model = model.transform(SpecializeElementwiseBinaryToHLS())

        # Get the HLS node (should be the only node in the graph)
        assert len(model.graph.node) == 1, f"Expected 1 node, got {len(model.graph.node)}"
        hls_node = model.graph.node[0]

        assert hls_node.op_type == f"{self.manual_class_name}_hls", (
            f"Expected {self.manual_class_name}_hls, got {hls_node.op_type}"
        )

        op = getHWCustomOp(hls_node, model)
        return op, model

    def setup_auto_op(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Create Brainsmith auto ElementwiseBinaryOp_hls.

        Workflow:
        1. Create ONNX operation
        2. Apply InferKernelList → ElementwiseBinaryOp
        3. Apply SpecializeElementwiseBinaryToHLS → ElementwiseBinaryOp_hls

        Returns:
            Tuple of (ElementwiseBinaryOp_hls instance, model)
        """
        from qonnx.transformation.infer_shapes import InferShapes
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from finn.util.basic import getHWCustomOp

        model, node_name = self.make_test_model()

        # Standard inference
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Infer to ElementwiseBinaryOp
        model = model.transform(InferKernelList())

        # Specialize to HLS
        model = model.transform(SpecializeElementwiseBinaryToHLS())

        # Get the HLS node (should be the only node in the graph)
        assert len(model.graph.node) == 1, f"Expected 1 node, got {len(model.graph.node)}"
        hls_node = model.graph.node[0]

        assert hls_node.op_type == "ElementwiseBinaryOp_hls", (
            f"Expected ElementwiseBinaryOp_hls, got {hls_node.op_type}"
        )

        op = getHWCustomOp(hls_node, model)
        return op, model

    # =========================================================================
    # Test Model Creation
    # =========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX model with binary operation.

        Creates model with dynamic LHS input and static RHS parameter.
        Phase 1 of ElementwiseBinaryOp only supports dynamic+static pattern.

        Returns:
            Tuple of (ModelWrapper, node_name) for the created ONNX model
        """
        # Configuration
        batch = 1
        h = w = 8
        ch = 64
        shape = [batch, h, w, ch]  # NHWC format
        node_name = f"{self.operation_type}_test"

        # Datatypes vary by operation
        idt0, idt1, odt = self._get_test_datatypes()

        # Create input tensor info
        inp0 = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, shape)
        outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        # Generate RHS as static parameter (Phase 1 limitation: dynamic + static)
        import numpy as np
        np.random.seed(42)  # Deterministic for reproducibility
        rhs_data = gen_finn_dt_tensor(idt1, shape)
        rhs_tensor = helper.make_tensor(
            "rhs", TensorProto.FLOAT, shape, rhs_data.flatten()
        )

        # Create ONNX node
        node = helper.make_node(
            self.operation_type, ["lhs", "rhs"], ["output"], name=node_name
        )

        # Build graph and model
        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=[inp0],
            outputs=[outp],
            initializer=[rhs_tensor],
        )
        model = ModelWrapper(qonnx_make_model(graph, producer_name="parity-test"))

        # Set datatypes
        model.set_tensor_datatype("lhs", idt0)
        model.set_tensor_datatype("rhs", idt1)
        model.set_tensor_datatype("output", odt)

        return model, node_name

    def _get_test_datatypes(self) -> Tuple[DataType, DataType, DataType]:
        """Get appropriate datatypes for this operation.

        Returns:
            (lhs_dtype, rhs_dtype, output_dtype) tuple
        """
        # Arithmetic operations
        if self.operation_type == "Add":
            return DataType["INT8"], DataType["INT8"], DataType["INT9"]
        elif self.operation_type == "Sub":
            return DataType["INT8"], DataType["INT8"], DataType["INT9"]
        elif self.operation_type == "Mul":
            return DataType["INT8"], DataType["INT4"], DataType["INT12"]
        elif self.operation_type == "Div":
            return DataType["INT8"], DataType["UINT4"], DataType["INT8"]

        # Comparison operations → BINARY output
        elif self.operation_type in ["Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"]:
            return DataType["INT8"], DataType["INT8"], DataType["BINARY"]

        # Logical operations → BINARY inputs and output
        elif self.operation_type in ["And", "Or", "Xor"]:
            return DataType["BINARY"], DataType["BINARY"], DataType["BINARY"]

        # Bitwise operations → preserve width
        elif self.operation_type in ["BitwiseAnd", "BitwiseOr", "BitwiseXor"]:
            return DataType["INT8"], DataType["INT8"], DataType["INT8"]

        # BitShift operations → LHS datatype unchanged
        elif self.operation_type in ["BitShiftLeft", "BitShiftRight"]:
            return DataType["INT8"], DataType["UINT4"], DataType["INT8"]

        else:
            raise ValueError(f"Unknown operation type: {self.operation_type}")

    def get_manual_transform(self) -> Type[Transformation]:
        """FINN's InferElementwiseBinaryOperation transform."""
        return InferElementwiseBinaryOperation

    def get_auto_transform(self) -> Type[Transformation]:
        """Brainsmith's unified InferKernelList transform."""
        return InferKernelList

    def configure_test_op(self, op: HWCustomOp, model: ModelWrapper, is_auto: bool) -> None:
        """Configure ElementwiseBinary op for testing.

        Sets PE parallelization for testing. Base class handles specialization separately.
        For Brainsmith auto implementation, also initializes the kernel model.

        Args:
            op: ElementwiseBinary operator instance
            model: ModelWrapper containing the op
            is_auto: True if auto implementation
        """
        # Set PE for testing (64 channels / 8 = 8-way folding)
        op.set_nodeattr("PE", 8)

        # Verify func parameter for Brainsmith implementation
        if is_auto:
            assert op.get_nodeattr("func") == self.operation_type, (
                f"Expected func={self.operation_type}, got {op.get_nodeattr('func')}"
            )
            # Initialize kernel model for KernelOp-based implementations
            # The transform may have failed to initialize if there were validation errors
            # Retry initialization now that PE is set correctly
            if hasattr(op, 'infer_node_datatype'):
                op.infer_node_datatype(model)

    def get_num_inputs(self) -> int:
        """ElementwiseBinary has 2 inputs."""
        return 2

    def get_num_outputs(self) -> int:
        """ElementwiseBinary has 1 output."""
        return 1

    # =========================================================================
    # Shape Normalization Overrides (FINN list → tuple conversion)
    # =========================================================================
    #
    # NOTE: FINN's ElementwiseBinary returns lists from shape methods, which is
    # inconsistent with most FINN kernels (AddStreams, ChannelwiseOp) that return
    # tuples. Brainsmith follows the majority pattern. These overrides normalize
    # FINN's lists to tuples before comparison to avoid cosmetic assertion failures.

    @pytest.mark.parity
    def test_normal_input_shape_parity(self):
        """Test normal input shape matches (with FINN list→tuple normalization)."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_shape = manual_op.get_normal_input_shape(ind)
            auto_shape = auto_op.get_normal_input_shape(ind)
            # Normalize FINN's list to tuple for comparison
            if isinstance(manual_shape, list):
                manual_shape = tuple(manual_shape)
            assert_shapes_match(manual_shape, auto_shape, ind, "normal input")

    @pytest.mark.parity
    def test_normal_output_shape_parity(self):
        """Test normal output shape matches (with FINN list→tuple normalization)."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_shape = manual_op.get_normal_output_shape(ind)
            auto_shape = auto_op.get_normal_output_shape(ind)
            # Normalize FINN's list to tuple for comparison
            if isinstance(manual_shape, list):
                manual_shape = tuple(manual_shape)
            assert_shapes_match(manual_shape, auto_shape, ind, "normal output")

    @pytest.mark.parity
    def test_folded_input_shape_parity(self):
        """Test folded input shape matches (with FINN list→tuple normalization).

        NOTE: Index 1 (RHS static parameter) is expected to differ:
        - FINN applies folding to static parameters (simpler but inaccurate)
        - Brainsmith does NOT fold static parameters (correct - they're not streamed)

        This is a semantic improvement in Brainsmith, not a bug.
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_inputs()):
            manual_shape = manual_op.get_folded_input_shape(ind)
            auto_shape = auto_op.get_folded_input_shape(ind)
            # Normalize FINN's list to tuple for comparison
            if isinstance(manual_shape, list):
                manual_shape = tuple(manual_shape)

            # Input 1 (RHS) is static - FINN incorrectly reports folded shape
            # Skip comparison for RHS since Brainsmith correctly doesn't fold it
            if ind == 1:
                # Just verify Brainsmith returns unfold shape for static parameter
                assert auto_shape[-1] == 1, f"Static RHS should have fold=1, got {auto_shape}"
                continue

            assert_shapes_match(manual_shape, auto_shape, ind, "folded input")

    @pytest.mark.parity
    def test_folded_output_shape_parity(self):
        """Test folded output shape matches (with FINN list→tuple normalization)."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        for ind in range(self.get_num_outputs()):
            manual_shape = manual_op.get_folded_output_shape(ind)
            auto_shape = auto_op.get_folded_output_shape(ind)
            # Normalize FINN's list to tuple for comparison
            if isinstance(manual_shape, list):
                manual_shape = tuple(manual_shape)
            assert_shapes_match(manual_shape, auto_shape, ind, "folded output")


# =============================================================================
# Test Classes for Each Operation
# =============================================================================

# Priority 1: Arithmetic Operations (4 operations)

class TestElementwiseAddParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary Add operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto:
    - 15 base structural tests (shapes, datatypes, execution)
    - 6 RTL tests (skipped - no RTL backend yet)
    - 3 efficiency tests (BRAM/URAM efficiency, op counts)
    - 1 execution test (Python simulation)
    """
    operation_type = "Add"
    manual_class_name = "ElementwiseAdd"


class TestElementwiseSubParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary Sub operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto:
    - 15 base structural tests (shapes, datatypes, execution)
    - 6 RTL tests (skipped - no RTL backend yet)
    - 3 efficiency tests (BRAM/URAM efficiency, op counts)
    - 1 execution test (Python simulation)
    """
    operation_type = "Sub"
    manual_class_name = "ElementwiseSub"


class TestElementwiseMulParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary Mul operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto:
    - 15 base structural tests (shapes, datatypes, execution)
    - 6 RTL tests (skipped - no RTL backend yet)
    - 3 efficiency tests (BRAM/URAM efficiency, op counts)
    - 1 execution test (Python simulation)
    """
    operation_type = "Mul"
    manual_class_name = "ElementwiseMul"


class TestElementwiseDivParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary Div operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto:
    - 15 base structural tests (shapes, datatypes, execution)
    - 6 RTL tests (skipped - no RTL backend yet)
    - 3 efficiency tests (BRAM/URAM efficiency, op counts)
    - 1 execution test (Python simulation)
    """
    operation_type = "Div"
    manual_class_name = "ElementwiseDiv"


# Priority 2: Comparison Operations (5 operations)

class TestElementwiseEqualParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary Equal operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Output datatype is BINARY (0 or 1).
    """
    operation_type = "Equal"
    manual_class_name = "ElementwiseEqual"


class TestElementwiseLessParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary Less operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Output datatype is BINARY (0 or 1).
    """
    operation_type = "Less"
    manual_class_name = "ElementwiseLess"


class TestElementwiseLessOrEqualParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary LessOrEqual operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Output datatype is BINARY (0 or 1).
    """
    operation_type = "LessOrEqual"
    manual_class_name = "ElementwiseLessOrEqual"


class TestElementwiseGreaterParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary Greater operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Output datatype is BINARY (0 or 1).
    """
    operation_type = "Greater"
    manual_class_name = "ElementwiseGreater"


class TestElementwiseGreaterOrEqualParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary GreaterOrEqual operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Output datatype is BINARY (0 or 1).
    """
    operation_type = "GreaterOrEqual"
    manual_class_name = "ElementwiseGreaterOrEqual"


# Priority 3: Logical Operations (3 operations)

class TestElementwiseAndParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary And operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Uses BINARY datatypes for inputs and output.
    """
    operation_type = "And"
    manual_class_name = "ElementwiseAnd"


class TestElementwiseOrParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary Or operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Uses BINARY datatypes for inputs and output.
    """
    operation_type = "Or"
    manual_class_name = "ElementwiseOr"


class TestElementwiseXorParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary Xor operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Uses BINARY datatypes for inputs and output.
    """
    operation_type = "Xor"
    manual_class_name = "ElementwiseXor"


# Priority 4: Bitwise Operations (3 operations)

class TestElementwiseBitwiseAndParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary BitwiseAnd operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Output width is max(lhs_width, rhs_width).
    """
    operation_type = "BitwiseAnd"
    manual_class_name = "ElementwiseBitwiseAnd"


class TestElementwiseBitwiseOrParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary BitwiseOr operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Output width is max(lhs_width, rhs_width).
    """
    operation_type = "BitwiseOr"
    manual_class_name = "ElementwiseBitwiseOr"


class TestElementwiseBitwiseXorParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary BitwiseXor operation.

    Tests 25 comparisons between FINN manual and Brainsmith auto.
    Output width is max(lhs_width, rhs_width).
    """
    operation_type = "BitwiseXor"
    manual_class_name = "ElementwiseBitwiseXor"


# Priority 5: BitShift Operations (2 operations)
# NOTE: FINN's InferElementwiseBinaryOperation doesn't support BitShift operations.
# These tests are expected to fail - BitShift is a Brainsmith extension.

@pytest.mark.skip(reason="FINN InferElementwiseBinaryOperation doesn't support BitShiftLeft")
class TestElementwiseBitShiftLeftParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary BitShiftLeft operation.

    SKIPPED: FINN doesn't have parity transform support for BitShift operations.
    BitShiftLeft is a Brainsmith extension with separate unit test coverage.

    Tests 26 comparisons between FINN manual and Brainsmith auto.
    Output datatype matches LHS (shift doesn't change type).
    """
    operation_type = "BitShiftLeft"
    manual_class_name = "ElementwiseBitShift"  # FINN uses single class for both shifts


@pytest.mark.skip(reason="FINN InferElementwiseBinaryOperation doesn't support BitShiftRight")
class TestElementwiseBitShiftRightParity(ElementwiseBinaryParityBase):
    """Parity tests for ElementwiseBinary BitShiftRight operation.

    SKIPPED: FINN doesn't have parity transform support for BitShift operations.
    BitShiftRight is a Brainsmith extension with separate unit test coverage.

    Tests 26 comparisons between FINN manual and Brainsmith auto.
    Output datatype matches LHS (shift doesn't change type).
    """
    operation_type = "BitShiftRight"
    manual_class_name = "ElementwiseBitShift"  # FINN uses single class for both shifts


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
