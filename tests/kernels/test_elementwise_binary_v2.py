"""Test elementwise binary operations (Add, Sub, Mul) using pytest-cases.

This file contains ONLY test logic - test data lives in test_elementwise_binary_cases.py.

Key improvements in restructured version:
1. Marks auto-propagated from case functions (no manual tracking)
2. Clear separation: test data vs test logic
3. Two selection modes: basic_validation (quick) vs full_sweep (comprehensive)
4. Reusable constants eliminate 90% code duplication
5. Debuggable: Set breakpoint on specific case function
6. Scalable: Add new cases without touching this file

Total tests: 34 base configs × 3 operations × 6 inherited tests = 612 tests
- 6 basic_validation configs (representative cases for quick feedback)
- 28 full_sweep configs (comprehensive testing)

Selection:
- pytest -m basic_validation                    # Quick validation (~108 tests)
- pytest -m "not basic_validation"              # Full sweep only (~504 tests)
- pytest                                        # All tests (612 tests)
"""

import numpy as np
import onnx.helper as helper
import pytest
from onnx import TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp
from brainsmith.primitives.transforms.infer_kernels import InferKernels
from tests.frameworks.single_kernel_test_v2 import SingleKernelTest
from tests.frameworks.test_config import KernelTestConfig, ModelStructure

# Import the cases module to get all case functions
from tests.kernels import test_elementwise_binary_cases


# Collect all case functions and call them to get base configs
_BASE_CONFIGS = []

for name in dir(test_elementwise_binary_cases):
    if name.startswith("case_"):
        case_func = getattr(test_elementwise_binary_cases, name)
        if callable(case_func):
            # Extract pytest marks (e.g., @pytest.mark.basic_validation)
            marks = []
            if hasattr(case_func, "pytestmark"):
                marks_attr = case_func.pytestmark
                marks = marks_attr if isinstance(marks_attr, list) else [marks_attr]

            # Call case function to get config
            try:
                result = case_func()
                _BASE_CONFIGS.append((result, marks))
            except TypeError:
                # Skip functions that require parameters
                pass


class TestElementwiseBinary(SingleKernelTest):
    """Test elementwise binary operations (Add, Sub, Mul).

    Test cases defined in test_elementwise_binary_cases.py for clarity.
    This class focuses purely on test logic, not test data.

    Architecture:
    1. Case functions (in cases file) return KernelTestConfig + marks
    2. Fixture multiplies cases by operations (Add, Sub, Mul)
    3. Marks auto-propagated from case functions
    4. Inherited test methods (from SingleKernelTest) run automatically

    Test Matrix:
    - 34 base configs (from case functions)
    - × 3 operations (Add, Sub, Mul)
    - × 6 inherited tests (pipeline + Python + cppsim + rtlsim)
    - = 612 total tests

    Selection Modes:
    - pytest -m basic_validation        # Quick validation (6 cases × 3 ops × 6 tests = 108)
    - pytest -m "not basic_validation"  # Full sweep only (28 cases × 3 ops × 6 tests = 504)
    - pytest                            # All tests (612 total)
    - pytest -k "broadcast"             # Run only broadcast cases
    """

    @pytest.fixture(
        params=[
            pytest.param(
                (operation, base_config),
                marks=marks,
                id=f"{operation.lower()}_{base_config.test_id}",
            )
            for operation in ["Add", "Sub", "Mul"]
            for base_config, marks in _BASE_CONFIGS
        ]
    )
    def kernel_test_config(self, request):
        """Generate final config by combining base config + operation.

        Uses pytest-cases to collect all case functions, then multiplies by operations.
        Marks from case functions are automatically propagated.

        Args:
            request: pytest request object containing (operation, base_config) tuple

        Returns:
            KernelTestConfig with operation set for this test
        """
        operation, base_config = request.param

        # Create final config with operation set
        return KernelTestConfig(
            test_id=f"{operation.lower()}_{base_config.test_id}",
            model=ModelStructure(
                operation=operation,
                input_shapes=base_config.input_shapes,
                input_dtypes=base_config.input_dtypes,
            ),
            design=base_config.design,
            platform=base_config.platform,
            validation=base_config.validation,
        )

    def make_test_model(self, kernel_test_config):
        """Create binary operation model from config.

        Supports both equal-shape and broadcasting scenarios.
        Output shape is computed via np.broadcast_shapes() to handle broadcasting.
        Operation type is extracted from config.operation (Add/Sub/Mul).

        Args:
            kernel_test_config: Test configuration from kernel_test_config fixture

        Returns:
            Tuple of (ModelWrapper, list of input names)
        """
        operation = kernel_test_config.operation
        input_shapes = kernel_test_config.input_shapes

        # Create input tensor infos with lhs/rhs naming
        lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, input_shapes["lhs"])
        rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, input_shapes["rhs"])

        # Compute output shape (handles broadcasting)
        output_shape = tuple(np.broadcast_shapes(input_shapes["lhs"], input_shapes["rhs"]))
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Create node and graph
        node = helper.make_node(operation, ["lhs", "rhs"], ["output"], name=f"{operation}_0")
        graph = helper.make_graph([node], f"test_{operation.lower()}", [lhs, rhs], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        return model, ["lhs", "rhs"]

    def get_kernel_inference_transform(self):
        """Convert ONNX binary op → ElementwiseBinaryOp kernel.

        Returns:
            Callable that returns InferKernels transform
        """
        return lambda: InferKernels([ElementwiseBinaryOp])
