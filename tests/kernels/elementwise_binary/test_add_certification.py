"""Certification tests for Add elementwise operation (v5.0).

Comprehensive certification sweep across all supported configurations:
- All integer dtypes (INT8, INT16, INT32, UINT8, UINT16, UINT32, BIPOLAR)
- Float dtypes (FLOAT32)
- Multiple shapes (2D, 4D)
- PE parallelization variants (1, 8, 16)

Test count: ~54 cases × 6 inherited tests = ~324 tests

Usage:
    pytest tests/kernels/elementwise_binary/test_add_certification.py -v
    pytest -m certification -k add
"""

import pytest

from brainsmith.kernels.elementwise_binary.tests import ElementwiseBinaryTestBase
from tests.fixtures.certification_sweeps import (
    DTYPES_ALL,
    SHAPES_2D,
    SHAPES_4D,
    build_certification_suite,
)
from tests.frameworks.test_config import KernelTestConfig

from .shared_cases import PLATFORM_ZYNQ7020, VALIDATION_STANDARD

# ============================================================================
# Test Case Generation (Comprehensive Sweep)
# ============================================================================

# Generate comprehensive certification cases using v5.0 helpers
_CERT_BASE = build_certification_suite(
    operation="Add",
    dtypes=DTYPES_ALL,  # 9 dtypes (8 integer + 1 float)
    shapes=[SHAPES_2D[0], SHAPES_4D[0]],  # 2 shapes: (1, 64), (1, 8, 8, 32)
    pe_variants=[1, 8, 16],  # 3 PE values
    input_names=("lhs", "rhs"),
)

# Convert to KernelTestConfig
CERTIFICATION_CASES = [
    KernelTestConfig(
        test_id=test_id,
        model=model,
        design=design,
        platform=PLATFORM_ZYNQ7020,  # All cert cases run on hardware
        validation=VALIDATION_STANDARD,
    )
    for test_id, model, design in _CERT_BASE
]

# ============================================================================
# Test Configuration Fixture
# ============================================================================


@pytest.fixture(
    params=[
        pytest.param(config, marks=[pytest.mark.certification], id=config.test_id)
        for config in CERTIFICATION_CASES
    ]
)
def kernel_test_config(request) -> KernelTestConfig:
    """Parameterized test configuration for Add certification cases.

    Yields one test configuration per certification case.
    All cases marked with @pytest.mark.certification.
    """
    return request.param


# ============================================================================
# Test Class
# ============================================================================


class TestAddCertification(ElementwiseBinaryTestBase):
    """Certification tests for Add elementwise operation.

    Comprehensive sweep across:
    - 9 datatypes (INT8, INT16, INT32, UINT8, UINT16, UINT32, BIPOLAR, FLOAT32, BINARY)
    - 2 shapes (2D: 1×64, 4D: 1×8×8×32)
    - 3 PE variants (1, 8, 16)
    - Total: 54 configurations

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
    # automatically for each test case in CERTIFICATION_CASES:
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
    #
    # 5. test_cppsim_execution_vs_golden(
    #        kernel_test_config, stage3_model, test_inputs, golden_outputs
    #    )
    #    - Validates HLS C++ simulation matches golden reference
    #
    # 6. test_rtlsim_execution_vs_golden(
    #        kernel_test_config, stage3_model, test_inputs, golden_outputs
    #    )
    #    - Validates RTL simulation matches golden reference
    # ========================================================================
