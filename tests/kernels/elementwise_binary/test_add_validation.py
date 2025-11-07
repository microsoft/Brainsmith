"""Validation tests for Add elementwise operation (v5.0).

Tactical validation tests covering representative cases for CI/CD:
- Basic dtypes (INT8, UINT8, FLOAT32, INT16, UINT16, etc.)
- Parallelization (PE=8, PE=16)
- Broadcasting patterns (channel, scalar, spatial, bidirectional, rank mismatch)
- Mixed dtypes (mixed sign, mixed width)

Test count: ~28 cases × 6 inherited tests = ~168 tests

Usage:
    pytest tests/kernels/elementwise_binary/test_add_validation.py -v
    pytest -m validation_v5 -k add
"""

import pytest

from brainsmith.kernels.elementwise_binary.tests import ElementwiseBinaryTestBase
from tests.frameworks.test_config import KernelTestConfig

from .shared_cases import VALIDATION_CASES_BASE, make_elementwise_case

# ============================================================================
# Test Case Generation
# ============================================================================

# Build Add-specific validation cases from shared base
VALIDATION_CASES = [
    make_elementwise_case(
        test_id=f"add_{test_id}",
        operation="Add",
        input_shapes=shapes,
        input_dtypes=dtypes,
        design=design,
        platform=platform,
    )
    for test_id, shapes, dtypes, design, platform in VALIDATION_CASES_BASE
]

# ============================================================================
# Test Configuration Fixture
# ============================================================================


@pytest.fixture(
    params=[
        pytest.param(config, marks=[pytest.mark.validation_v5], id=config.test_id)
        for config in VALIDATION_CASES
    ]
)
def kernel_test_config(request) -> KernelTestConfig:
    """Parameterized test configuration for Add validation cases.

    Yields one test configuration per validation case.
    All cases marked with @pytest.mark.validation_v5.
    """
    return request.param


# ============================================================================
# Test Class
# ============================================================================


class TestAddValidation(ElementwiseBinaryTestBase):
    """Validation tests for Add elementwise operation.

    Inherits implementation from ElementwiseBinaryTestBase:
    - make_test_model(): Builds ONNX model with broadcasting support
    - get_kernel_op(): Returns ElementwiseBinaryOp

    Inherits 6 tests automatically from SingleKernelTest:
    1. test_stage1_model_structure - Validates ONNX model structure
    2. test_stage2_kernel_inference - Validates kernel inference transform
    3. test_python_execution_vs_golden - Python execution parity
    4. test_stage3_backend_specialization - Backend specialization validation
    5. test_cppsim_execution_vs_golden - HLS C++ simulation parity
    6. test_rtlsim_execution_vs_golden - RTL simulation parity

    All tests use session-scoped model_cache for computational reuse.
    """

    pass  # All implementation inherited from ElementwiseBinaryTestBase

    # ========================================================================
    # Inherited Tests (6 total)
    # ========================================================================
    # The following tests are inherited from SingleKernelTest and run
    # automatically for each test case in VALIDATION_CASES:
    #
    # 1. test_stage1_model_structure(kernel_test_config, stage1_model)
    #    - Validates Stage 1 model has correct inputs/outputs/dtypes
    #
    # 2. test_stage2_kernel_inference(kernel_test_config, stage2_model)
    #    - Validates kernel inference produces correct kernel type
    #
    # 3. test_python_execution_vs_golden(
    #        kernel_test_config, stage2_model, test_inputs, golden_outputs
    #    )
    #    - Validates Python execution matches QONNX golden reference
    #
    # 4. test_stage3_backend_specialization(kernel_test_config, stage3_model)
    #    - Validates backend specialization (HLS code generation)
    #    - Skipped if platform_config.fpgapart is None
    #
    # 5. test_cppsim_execution_vs_golden(
    #        kernel_test_config, stage3_model, test_inputs, golden_outputs
    #    )
    #    - Validates HLS C++ simulation matches golden reference
    #    - Skipped if platform_config.fpgapart is None
    #
    # 6. test_rtlsim_execution_vs_golden(
    #        kernel_test_config, stage3_model, test_inputs, golden_outputs
    #    )
    #    - Validates RTL simulation matches golden reference
    #    - Skipped if platform_config.fpgapart is None
    # ========================================================================
