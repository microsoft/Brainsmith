"""Design space fixtures for DSE testing.

Ported from OLD_FOR_REFERENCE_ONLY/fixtures/dse_fixtures.py
Real GlobalDesignSpace objects for integration testing.
"""

import pytest
from brainsmith.dse import GlobalDesignSpace, DSEConfig
from brainsmith.dse.types import OutputType
from tests.support.constants import (
    DSE_DEFAULT_CLOCK_PERIOD_NS,
    DSE_DEFAULT_PARALLEL_BUILDS,
    DSE_DEFAULT_MAX_COMBINATIONS
)


# Configuration Fixtures

@pytest.fixture
def blueprint_config() -> DSEConfig:
    """Create a basic DSEConfig for testing.

    Returns:
        DSEConfig with test defaults
    """
    return DSEConfig(
        clock_ns=DSE_DEFAULT_CLOCK_PERIOD_NS,
        output=OutputType.ESTIMATES,
        board="test_board",
        verify=False,
        parallel_builds=DSE_DEFAULT_PARALLEL_BUILDS,
        debug=False,
        save_intermediate_models=False
    )


@pytest.fixture
def base_finn_config() -> dict:
    """Create a base FINN configuration for testing.

    Returns:
        Dictionary with FINN configuration parameters
    """
    return {
        "output_dir": "output/test",
        "synth_clk_period_ns": DSE_DEFAULT_CLOCK_PERIOD_NS,
        "board": "test_board",
        "shell_flow_type": "test_flow",
        "generate_outputs": ["estimates"],
        "folding_config_file": None
    }


# Design Space Fixtures

@pytest.fixture
def simple_design_space(simple_onnx_model) -> GlobalDesignSpace:
    """Create a simple linear design space (no branches).

    Args:
        simple_onnx_model: Fixture providing ONNX model path

    Returns:
        GlobalDesignSpace with linear step sequence
    """
    return GlobalDesignSpace(
        model_path=str(simple_onnx_model),
        kernel_backends=[],
        steps=[
            "test_step",
            "test_step1",
            "test_step2"
        ],
        max_combinations=DSE_DEFAULT_MAX_COMBINATIONS
    )


@pytest.fixture
def branching_design_space(simple_onnx_model) -> GlobalDesignSpace:
    """Create a design space with a single branch point.

    Args:
        simple_onnx_model: Fixture providing ONNX model path

    Returns:
        GlobalDesignSpace with one branch point
    """
    return GlobalDesignSpace(
        model_path=str(simple_onnx_model),
        kernel_backends=[("TestKernel", ["TestKernel_hls", "TestKernel_rtl"])],
        steps=[
            "test_step",
            ["test_step1", "test_step2"],  # Branch point
            "test_step3"
        ],
        max_combinations=DSE_DEFAULT_MAX_COMBINATIONS
    )


@pytest.fixture
def multi_branch_design_space(simple_onnx_model) -> GlobalDesignSpace:
    """Create a design space with multiple branch levels.

    Args:
        simple_onnx_model: Fixture providing ONNX model path

    Returns:
        GlobalDesignSpace with multiple branches and kernel backends
    """
    return GlobalDesignSpace(
        model_path=str(simple_onnx_model),
        kernel_backends=[
            ("TestKernel", ["TestKernel_hls", "TestKernel_rtl"]),
            ("TestKernel2", ["TestKernel2_hls"])
        ],
        steps=[
            "test_step",
            ["test_step1", "test_step2"],  # First branch
            "test_step3",
            ["~", "test_step"],  # Second branch with skip option
            "test_step2"
        ],
        max_combinations=DSE_DEFAULT_MAX_COMBINATIONS
    )
