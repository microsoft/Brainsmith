"""Pytest configuration and fixture imports for Brainsmith tests."""

import os

# CRITICAL: Export Brainsmith config to environment BEFORE any other imports
# This ensures LD_LIBRARY_PATH is set early for XSI dynamic library loading
try:
    from brainsmith.config import get_config
    config = get_config()
    config.export_to_environment()
except Exception:
    pass  # Config might not be available during initial setup

# Plugin imports - REQUIRED for decorator side effects
# Without these, @kernel/@step/@transform test plugins won't be registered
import tests.fixtures.plugins.kernels     # TestKernel, TestKernel2
import tests.fixtures.plugins.steps       # test_step, test_step1-3
import tests.fixtures.plugins.transforms  # test_transform, test_transform2

# Import fixtures to make them available to all tests
from tests.fixtures.dse_fixtures import *
from tests.fixtures.model_utils import simple_onnx_model

# Import kernel test helpers for easy access in all kernel tests
# Use these for unit testing kernels (schema, inference, transformation)
# For parity testing (comparing implementations), use tests/parity/ParityTestBase
from tests.fixtures.kernel_test_helpers import (
    OnnxModelBuilder,
    make_binary_op_model,
    make_parametric_op_model,
    make_unary_op_model,
)


# Configure pytest plugins if needed
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "parity: marks tests comparing manual vs auto HWCustomOp implementations"
    )
    config.addinivalue_line(
        "markers", "cppsim: marks tests requiring C++ simulation with Vivado/Vitis HLS"
    )
    config.addinivalue_line(
        "markers", "hls: marks tests requiring HLS code generation"
    )
    config.addinivalue_line(
        "markers", "rtl: marks tests requiring RTL synthesis"
    )
    config.addinivalue_line(
        "markers", "rtlsim: marks tests requiring RTL simulation"
    )