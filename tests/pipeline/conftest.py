"""Pytest configuration and fixtures for pipeline integration tests.

Provides shared fixtures for:
- FPGA part configuration
- Clock period configuration
- Golden reference tolerance settings
- Test data generation
"""

import pytest


@pytest.fixture
def default_fpga_part_hls():
    """Default FPGA part for HLS backend testing."""
    return "xcvu9p-flgb2104-2-i"  # UltraScale+ VU9P


@pytest.fixture
def default_fpga_part_rtl():
    """Default FPGA part for RTL backend testing."""
    return "xcvc1902-vsvd1760-2MP-e-S"  # Versal VC1902


@pytest.fixture
def default_clock_period():
    """Default clock period in nanoseconds for RTL synthesis."""
    return 3.0  # ~333MHz


@pytest.fixture
def golden_tolerance_python():
    """Tolerance settings for golden reference validation (Python execution).

    Returns:
        Dict with rtol (relative tolerance) and atol (absolute tolerance)
    """
    return {"rtol": 1e-7, "atol": 1e-9}


@pytest.fixture
def golden_tolerance_cppsim():
    """Tolerance settings for golden reference validation (C++ simulation).

    Returns:
        Dict with rtol (relative tolerance) and atol (absolute tolerance)
    """
    return {"rtol": 1e-5, "atol": 1e-6}


@pytest.fixture
def golden_tolerance_rtlsim():
    """Tolerance settings for golden reference validation (RTL simulation).

    Returns:
        Dict with rtol (relative tolerance) and atol (absolute tolerance)
    """
    return {"rtol": 1e-5, "atol": 1e-6}


def pytest_configure(config):
    """Register custom markers for pipeline integration tests."""
    config.addinivalue_line(
        "markers", "pipeline: Pipeline integration tests (full ONNX → Hardware flow)"
    )
    config.addinivalue_line("markers", "golden: Tests that validate against golden reference")
    config.addinivalue_line(
        "markers", "phase1: Phase 1 pipeline tests (pipeline + golden reference)"
    )
    config.addinivalue_line(
        "markers", "phase2: Phase 2 pipeline tests (cross-backend + parametric)"
    )
    config.addinivalue_line("markers", "phase3: Phase 3 pipeline tests (snapshots + properties)")
