"""Pytest configuration and fixture imports for Brainsmith tests."""

# Plugin imports - REQUIRED for decorator side effects
# Without these, @kernel/@step/@transform test plugins won't be registered
import tests.fixtures.plugins.kernels     # TestKernel, TestKernel2
import tests.fixtures.plugins.steps       # test_step, test_step1-3
import tests.fixtures.plugins.transforms  # test_transform, test_transform2

# Import fixtures to make them available to all tests
from tests.fixtures.dse_fixtures import *
from tests.fixtures.model_utils import simple_onnx_model


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