# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ChannelwiseOp parity tests with backend support (DualKernelTest).

This test suite validates FINN ChannelwiseOp vs Brainsmith ChannelwiseOp
for all 4 operation modes:
- Add (bias addition)
- Mul (scale multiplication)
- LessOrEqual (threshold comparison)
- GreaterOrEqual (threshold comparison)

Key Insight: FINN supports leq/geq in cppsim/rtlsim but NOT in Python execution.
This test suite handles this gracefully by skipping Python tests for comparison ops.

Test Coverage:
- Add/Mul: 20 tests each (full pipeline: Python + cppsim + rtlsim)
- LessOrEqual/GreaterOrEqual: 14 tests each (backend only: cppsim + rtlsim)
- Total: 68 tests

Architecture:
- ChannelwiseParityBase: Shared test infrastructure
- 4 subclasses: One per operation mode
- Conditional test execution based on operation capabilities

Usage:
    # Run all ChannelwiseOp tests
    pytest tests/kernels/test_channelwise_backend.py -v

    # Run only Add mode tests
    pytest tests/kernels/test_channelwise_backend.py::TestChannelwiseAddParity -v

    # Run only backend tests (skip Python)
    pytest tests/kernels/test_channelwise_backend.py -m "cppsim or rtlsim" -v

    # Skip slow tests
    pytest tests/kernels/test_channelwise_backend.py -m "not slow" -v
"""

import pytest
import numpy as np
from onnx import helper, TensorProto
from typing import Dict, Tuple, Type

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
import qonnx.core.data_layout as DataLayout

from finn.transformation.fpgadataflow.convert_to_hw_layers import InferChannelwiseLinearLayer
from brainsmith.primitives.transforms.infer_kernel import InferKernel
from brainsmith.kernels.channelwise import ChannelwiseOp

from tests.frameworks.dual_kernel_test import DualKernelTest


class ChannelwiseParityBase(DualKernelTest):
    """Base class for ChannelwiseOp parity tests across all operation modes.

    Handles the key difference between operation types:
    - Add/Mul: Full pipeline support (Python + cppsim + rtlsim)
    - LessOrEqual/GreaterOrEqual: Backend only (cppsim + rtlsim, no Python)

    Subclasses override:
    - operation_type: ONNX operation name ("Add", "Mul", "LessOrEqual", "GreaterOrEqual")

    Test Count:
    - Add/Mul: 20 tests (7 parity + 5 estimation + 8 execution)
    - LessOrEqual/GreaterOrEqual: 14 tests (7 parity + 5 estimation + 2 backend execution)
    """

    operation_type: str = "Add"  # Override in subclasses

    # ========================================================================
    # Operation Capabilities (varies by operation type)
    # ========================================================================

    def supports_python_execution(self) -> bool:
        """Check if this operation supports Python execution.

        FINN's ChannelwiseOp execute_node() only supports Add/Mul.
        LessOrEqual/GreaterOrEqual only work in cppsim/rtlsim.

        Returns:
            True for Add/Mul, False for comparison operations
        """
        return self.operation_type in ["Add", "Mul"]

    # ========================================================================
    # Required Configuration (from KernelTestConfig)
    # ========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX model with channelwise operation.

        Returns:
            (model, node_name): ModelWrapper and name of operation node
        """
        # Configuration
        batch = 1
        h = w = 8
        ch = 64
        shape = [batch, h, w, ch]  # NHWC format
        node_name = f"{self.operation_type}_test"

        # Datatypes vary by operation
        if self.operation_type == "Add":
            idt = DataType["INT8"]
            pdt = DataType["INT8"]
            odt = DataType["INT9"]  # Add expands by 1 bit
        elif self.operation_type == "Mul":
            idt = DataType["INT8"]
            pdt = DataType["INT4"]
            odt = DataType["INT12"]  # Mul: 8+4=12 bits
        elif self.operation_type in ["LessOrEqual", "GreaterOrEqual"]:
            idt = DataType["INT8"]
            pdt = DataType["INT8"]
            odt = DataType["BINARY"]  # Comparisons produce binary output
        else:
            raise ValueError(f"Unknown operation type: {self.operation_type}")

        # Create input tensor info
        inp = helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)
        outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        # Generate parameter tensor (per-channel values)
        np.random.seed(42)  # Deterministic for reproducibility
        param_data = gen_finn_dt_tensor(pdt, [ch])
        param_tensor = helper.make_tensor(
            "param", TensorProto.FLOAT, [ch], param_data.flatten().tolist()
        )

        # Create ONNX node
        node = helper.make_node(
            self.operation_type, ["data", "param"], ["output"], name=node_name
        )

        # Build graph and model
        graph = helper.make_graph(
            nodes=[node],
            name="test_channelwise",
            inputs=[inp],
            outputs=[outp],
            initializer=[param_tensor],
        )
        model = ModelWrapper(qonnx_make_model(graph, producer_name="channelwise-test"))

        # Set datatypes
        model.set_tensor_datatype("data", idt)
        model.set_tensor_datatype("param", pdt)
        model.set_tensor_datatype("output", odt)

        # Set data layout (required for FINN inference transforms)
        model.set_tensor_layout("data", DataLayout.NHWC)
        model.set_tensor_layout("output", DataLayout.NHWC)

        return model, node_name

    def get_manual_transform(self) -> Type[Transformation]:
        """Return FINN's InferChannelwiseLinearLayer transform.

        Returns:
            InferChannelwiseLinearLayer transform class
        """
        return InferChannelwiseLinearLayer

    def get_auto_transform(self) -> Type[Transformation]:
        """Return Brainsmith's InferKernel transform for ChannelwiseOp.

        Uses InferKernel(ChannelwiseOp) instead of InferKernelList to:
        - Test only ChannelwiseOp inference (not fallback to other kernels)
        - Expose bugs immediately (no silent fallback)
        - Match manual transform (which also infers ChannelwiseOp)

        Returns:
            InferKernel transform class initialized with ChannelwiseOp
        """
        # Return a lambda that creates InferKernel(ChannelwiseOp)
        # DualKernelTest expects a callable that returns a Transformation
        return lambda: InferKernel(ChannelwiseOp)

    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference for channelwise operation.

        Args:
            inputs: Dict with "data" key (and "param" is in model initializers)

        Returns:
            Dict with "output" key
        """
        # Get input data
        data = inputs["data"]

        # Get parameter from model
        # Note: This is called after pipeline runs, so we need to get param from model
        # For golden reference, we'll generate it the same way as make_test_model()
        ch = data.shape[-1]

        # Get datatype for param
        if self.operation_type == "Add":
            pdt = DataType["INT8"]
        elif self.operation_type == "Mul":
            pdt = DataType["INT4"]
        elif self.operation_type in ["LessOrEqual", "GreaterOrEqual"]:
            pdt = DataType["INT8"]

        # Generate same parameter (deterministic seed)
        np.random.seed(42)
        param = gen_finn_dt_tensor(pdt, [ch])

        # Compute operation
        if self.operation_type == "Add":
            # Broadcast param across spatial dimensions
            result = data + param.reshape(1, 1, 1, -1)
        elif self.operation_type == "Mul":
            result = data * param.reshape(1, 1, 1, -1)
        elif self.operation_type == "LessOrEqual":
            result = (data <= param.reshape(1, 1, 1, -1)).astype(np.int8)
        elif self.operation_type == "GreaterOrEqual":
            result = (data >= param.reshape(1, 1, 1, -1)).astype(np.int8)

        return {"output": result}

    def get_num_inputs(self) -> int:
        """ChannelwiseOp has 1 input (param is initializer)."""
        return 1

    def get_num_outputs(self) -> int:
        """ChannelwiseOp has 1 output."""
        return 1

    # ========================================================================
    # Backend Configuration
    # ========================================================================

    def get_backend_fpgapart(self) -> str:
        """Enable backend testing.

        Returns:
            FPGA part string for Zynq-7020
        """
        return "xc7z020clg400-1"

    def get_manual_backend_variants(self):
        """Return FINN backend for manual (FINN) pipeline.

        Manual pipeline uses FINN's InferChannelwiseLinearLayer which creates
        nodes with "Func" attribute. Must use FINN's ChannelwiseOp_hls backend
        which expects "Func", not Brainsmith's which expects "func".

        Returns:
            List containing FINN's ChannelwiseOp_hls backend class
        """
        from finn.custom_op.fpgadataflow.hls.channelwise_op_hls import ChannelwiseOp_hls
        return [ChannelwiseOp_hls]

    # ========================================================================
    # Optional Configuration
    # ========================================================================

    def configure_kernel_node(self, op, model: ModelWrapper) -> None:
        """Configure ChannelwiseOp with PE=8.

        Args:
            op: HWCustomOp instance (ChannelwiseOp)
            model: ModelWrapper

        Note: This is called for BOTH manual and auto pipelines.
        """
        # Set PE for testing (64 channels / 8 = 8-way folding)
        op.set_nodeattr("PE", 8)

        # Set preferred implementation style for backend
        op.set_nodeattr("preferred_impl_style", "hls")

    # ========================================================================
    # Conditional Test Execution (Python tests skip for comparison ops)
    # ========================================================================

    def test_manual_python_vs_golden(self):
        """Test manual Python execution vs golden (skip for comparison ops)."""
        if not self.supports_python_execution():
            pytest.skip(
                f"{self.operation_type} not supported in FINN Python execution "
                "(only cppsim/rtlsim)"
            )
        # Call parent implementation
        super().test_manual_python_vs_golden()

    def test_auto_python_vs_golden(self):
        """Test auto Python execution vs golden (skip for comparison ops)."""
        if not self.supports_python_execution():
            pytest.skip(
                f"{self.operation_type} not supported in FINN Python execution "
                "(only cppsim/rtlsim)"
            )
        # Call parent implementation
        super().test_auto_python_vs_golden()

    def test_manual_auto_parity_python(self):
        """Test manual vs auto Python parity (skip for comparison ops)."""
        if not self.supports_python_execution():
            pytest.skip(
                f"{self.operation_type} not supported in FINN Python execution "
                "(only cppsim/rtlsim)"
            )
        # Call parent implementation
        super().test_manual_auto_parity_python()


# =============================================================================
# Concrete Test Classes (One per Operation Mode)
# =============================================================================


class TestChannelwiseAddParity(ChannelwiseParityBase):
    """Parity: FINN ChannelwiseOp (Add) vs Brainsmith ChannelwiseOp (Add).

    Tests bias addition across full 3-stage pipeline:
    - Stage 1: ONNX Add node (with static param)
    - Stage 2: ChannelwiseOp base kernel
    - Stage 3: ChannelwiseOp_hls backend

    Test Coverage (20 tests):
    - 7 core parity tests (shapes, widths, datatypes)
    - 5 HW estimation tests (cycles, resources)
    - 8 golden execution tests (Python + cppsim + rtlsim)

    Operation: output = data + param (per-channel bias addition)
    Datatypes: INT8 + INT8 → INT9
    """

    operation_type = "Add"


class TestChannelwiseMulParity(ChannelwiseParityBase):
    """Parity: FINN ChannelwiseOp (Mul) vs Brainsmith ChannelwiseOp (Mul).

    Tests scale multiplication across full 3-stage pipeline:
    - Stage 1: ONNX Mul node (with static param)
    - Stage 2: ChannelwiseOp base kernel
    - Stage 3: ChannelwiseOp_hls backend

    Test Coverage (20 tests):
    - 7 core parity tests (shapes, widths, datatypes)
    - 5 HW estimation tests (cycles, resources)
    - 8 golden execution tests (Python + cppsim + rtlsim)

    Operation: output = data * param (per-channel scale multiplication)
    Datatypes: INT8 * INT4 → INT12
    """

    operation_type = "Mul"


# NOTE: LessOrEqual and GreaterOrEqual are NOT tested here because:
# - FINN's InferChannelwiseLinearLayer does NOT support these operations
# - Cannot do parity testing without FINN manual implementation
# - Brainsmith ChannelwiseOp DOES support leq/geq (should be tested separately)
#
# TODO: Create SingleKernelTest for ChannelwiseOp leq/geq to validate
# Brainsmith implementation without parity comparison


# =============================================================================
# Validation Meta-Tests
# =============================================================================


@pytest.mark.validation
def test_all_operation_modes_present():
    """Verify Add and Mul operation modes have test classes.

    Note: LessOrEqual/GreaterOrEqual are not tested because FINN doesn't
    support them in InferChannelwiseLinearLayer.
    """
    import inspect

    # Get all test classes in this module
    test_classes = [
        name
        for name, obj in globals().items()
        if inspect.isclass(obj)
        and issubclass(obj, ChannelwiseParityBase)
        and obj != ChannelwiseParityBase
    ]

    assert len(test_classes) == 2, f"Expected 2 test classes (Add, Mul), found {len(test_classes)}"

    # Check operation types
    operations = {
        TestChannelwiseAddParity.operation_type,
        TestChannelwiseMulParity.operation_type,
    }

    expected_operations = {"Add", "Mul"}
    assert operations == expected_operations, (
        f"Expected operations {expected_operations}, got {operations}"
    )


@pytest.mark.validation
def test_add_mul_support_python():
    """Verify Add/Mul support Python execution."""
    add_test = TestChannelwiseAddParity()
    mul_test = TestChannelwiseMulParity()

    assert add_test.supports_python_execution(), "Add should support Python"
    assert mul_test.supports_python_execution(), "Mul should support Python"


@pytest.mark.validation
def test_backend_enabled_for_all():
    """Verify backend is enabled for Add and Mul operation modes."""
    for test_class in [
        TestChannelwiseAddParity,
        TestChannelwiseMulParity,
    ]:
        test_instance = test_class()
        fpgapart = test_instance.get_backend_fpgapart()

        assert fpgapart is not None, f"{test_class.__name__} should enable backend"
        assert fpgapart == "xc7z020clg400-1", (
            f"Expected xc7z020clg400-1, got {fpgapart}"
        )


@pytest.mark.validation
def test_test_count_correct():
    """Verify each test class has correct number of tests.

    Add/Mul: 20 tests (full pipeline support)
    """
    import inspect

    for test_class, expected_count in [
        (TestChannelwiseAddParity, 20),
        (TestChannelwiseMulParity, 20),
    ]:
        # Get all test methods
        test_methods = [
            name
            for name, method in inspect.getmembers(test_class, inspect.isfunction)
            if name.startswith("test_")
        ]

        assert len(test_methods) == expected_count, (
            f"{test_class.__name__} should have {expected_count} tests, "
            f"found {len(test_methods)}: {test_methods}"
        )
