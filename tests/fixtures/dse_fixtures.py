"""Combined DSE fixtures for testing - design spaces and configurations."""

import pytest
from brainsmith.dse import GlobalDesignSpace, DSEConfig
from brainsmith.dse.types import OutputType
from tests.utils.test_constants import (
    DEFAULT_CLOCK_PERIOD_NS,
    DEFAULT_PARALLEL_BUILDS,
    DEFAULT_MAX_COMBINATIONS
)


# Configuration Fixtures

@pytest.fixture
def blueprint_config():
    """Create a basic DSEConfig for testing."""
    return DSEConfig(
        clock_ns=DEFAULT_CLOCK_PERIOD_NS,
        output=OutputType.ESTIMATES,
        board="test_board",
        verify=False,
        parallel_builds=DEFAULT_PARALLEL_BUILDS,
        debug=False,
        save_intermediate_models=False
    )


@pytest.fixture
def base_finn_config():
    """Create a base FINN configuration for testing."""
    return {
        "output_dir": "output/test",
        "synth_clk_period_ns": DEFAULT_CLOCK_PERIOD_NS,
        "board": "test_board",
        "shell_flow_type": "test_flow",
        "generate_outputs": ["estimates"],
        "folding_config_file": None
    }


# Design Space Fixtures

@pytest.fixture
def simple_design_space(simple_onnx_model):
    """Create a simple linear design space."""
    model_path = simple_onnx_model
    return GlobalDesignSpace(
        model_path=str(model_path),
        kernel_backends=[],
        steps=[
            "test_step",
            "test_step1", 
            "test_step2"
        ],
        max_combinations=DEFAULT_MAX_COMBINATIONS
    )


@pytest.fixture
def branching_design_space(simple_onnx_model):
    """Create a design space with branch points."""
    model_path = simple_onnx_model
    return GlobalDesignSpace(
        model_path=str(model_path),
        kernel_backends=[("TestKernel", ["TestKernel_hls", "TestKernel_rtl"])],
        steps=[
            "test_step",
            ["test_step1", "test_step2"],  # Branch point
            "test_step3"
        ],
        max_combinations=DEFAULT_MAX_COMBINATIONS
    )


@pytest.fixture
def multi_branch_design_space(simple_onnx_model):
    """Create a design space with multiple branch levels."""
    model_path = simple_onnx_model
    return GlobalDesignSpace(
        model_path=str(model_path),
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
        max_combinations=DEFAULT_MAX_COMBINATIONS
    )