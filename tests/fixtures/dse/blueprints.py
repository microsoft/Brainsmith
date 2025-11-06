"""Blueprint creation helpers for testing.

Ported from OLD_FOR_REFERENCE_ONLY/utils/blueprint_helpers.py
Programmatic YAML blueprint generation - avoids string templates.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pytest


# YAML Blueprint Templates

MINIMAL_BLUEPRINT = """
name: {name}
clock_ns: {clock_ns}
design_space:
  steps: {steps}
"""

FULL_BLUEPRINT = """
name: {name}
description: {description}
clock_ns: {clock_ns}
output: {output}
board: {board}{target_fps}
design_space:
  steps: {steps}
"""

EXTENDS_BLUEPRINT = """
name: {name}
extends: {extends}{clock_ns}
design_space:
  steps: {steps}
"""

BASE_STEPS_BLUEPRINT = """
name: {name}
clock_ns: {clock_ns}
design_space:
  steps: {steps}
"""

STEP_INSERT_AFTER_BLUEPRINT = """
name: {name}
extends: {extends}
design_space:
  steps:
    - after: {after_step}
      insert: {insert_step}
"""

STEP_INSERT_START_BLUEPRINT = """
name: {name}
extends: {extends}
design_space:
  steps:
    - at_start:
        insert: {insert_step}
"""

STEP_INSERT_END_BLUEPRINT = """
name: {name}
extends: {extends}
design_space:
  steps:
    - at_end:
        insert: {insert_step}
"""

STEP_REPLACE_BLUEPRINT = """
name: {name}
extends: {extends}
design_space:
  steps:
    - replace: {replace_step}
      with: {with_step}
"""

STEP_REMOVE_BLUEPRINT = """
name: {name}
extends: {extends}
design_space:
  steps:
    - remove: {remove_step}
"""

BRANCH_POINTS_BLUEPRINT = """
name: {name}
clock_ns: {clock_ns}
design_space:
  steps:
    - custom:test_step
    - [custom:test_step1, custom:test_step2, "~"]  # Branch with skip
    - custom:test_identity_step
"""

INHERITANCE_PARENT_BLUEPRINT = """
name: {name}
clock_ns: {clock_ns}{board}
design_space:
  steps: {steps}{kernels}
"""

INHERITANCE_GRANDPARENT_BLUEPRINT = """
name: {name}
clock_ns: {clock_ns}{board}
design_space:
  steps: {steps}
"""

STEP_RANGE_BLUEPRINT = """
name: {name}
clock_ns: {clock_ns}
start_step: {start_step}
stop_step: {stop_step}
design_space:
  steps: {steps}
"""


def create_blueprint_file(
    tmp_path: Path,
    template: str,
    name: str,
    clock_ns: float = 5.0,
    steps: Optional[List[str]] = None,
    **kwargs
) -> Path:
    """Create a blueprint YAML file from a template.

    Args:
        tmp_path: pytest tmp_path fixture for temporary files
        template: YAML template string with format placeholders
        name: Blueprint name
        clock_ns: Clock period in nanoseconds
        steps: List of step names (defaults to basic test steps)
        **kwargs: Additional template variables

    Returns:
        Path to the created blueprint file
    """
    if steps is None:
        steps = ["test_step1", "test_step2", "test_step3"]

    # Format lists as YAML arrays
    if isinstance(steps, list):
        steps_yaml = str(steps)
    else:
        steps_yaml = steps

    # Handle kernel_backends formatting
    if 'kernel_backends' in kwargs and isinstance(kwargs['kernel_backends'], list):
        kwargs['kernel_backends'] = str(kwargs['kernel_backends'])

    # Handle step_operations formatting
    if 'step_operations' in kwargs and isinstance(kwargs['step_operations'], list):
        kwargs['step_operations'] = str(kwargs['step_operations'])

    # Handle kernels formatting
    if 'kernels' in kwargs and isinstance(kwargs['kernels'], list):
        kwargs['kernels'] = str(kwargs['kernels'])

    # Format the template
    content = template.format(
        name=name,
        clock_ns=clock_ns,
        steps=steps_yaml,
        **kwargs
    )

    # Create file
    file_path = tmp_path / f"{name}.yaml"
    file_path.write_text(content)

    return file_path


def create_minimal_blueprint(tmp_path: Path, name: str = "test_minimal", **kwargs) -> Path:
    """Create a minimal blueprint with just required fields.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    return create_blueprint_file(tmp_path, MINIMAL_BLUEPRINT, name, **kwargs)


def create_full_blueprint(
    tmp_path: Path,
    name: str = "test_full",
    description: str = "Test blueprint",
    output: str = "bitfile",
    board: str = "V80",
    target_fps: Optional[int] = None,
    **kwargs
) -> Path:
    """Create a full blueprint with all common fields.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        description: Blueprint description
        output: Output type
        board: Target FPGA board
        target_fps: Target frames per second (optional, for FINN parallelization)
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    # Format target_fps as YAML (or omit if None)
    target_fps_yaml = f"\ntarget_fps: {target_fps}" if target_fps is not None else ""

    return create_blueprint_file(
        tmp_path,
        FULL_BLUEPRINT,
        name,
        description=description,
        output=output,
        board=board,
        target_fps=target_fps_yaml,
        **kwargs
    )


def create_extends_blueprint(
    tmp_path: Path,
    name: str = "test_child",
    extends: str = "parent.yaml",
    clock_ns: Optional[float] = None,
    **kwargs
) -> Path:
    """Create a blueprint that extends another (inheritance).

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        extends: Parent blueprint filename
        clock_ns: Clock period (optional - omit to inherit from parent)
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    # Format clock_ns as YAML (or omit if None to inherit from parent)
    clock_ns_yaml = f"\nclock_ns: {clock_ns}" if clock_ns is not None else ""

    return create_blueprint_file(
        tmp_path,
        EXTENDS_BLUEPRINT,
        name,
        extends=extends,
        clock_ns=clock_ns_yaml,
        **kwargs
    )


def create_base_steps_blueprint(
    tmp_path: Path,
    name: str = "base",
    steps: Optional[List[str]] = None,
    **kwargs
) -> Path:
    """Create a base blueprint for step operations testing.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        steps: List of step names
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    if steps is None:
        steps = ["custom:test_step", "custom:test_step1", "custom:test_step2"]

    return create_blueprint_file(
        tmp_path,
        BASE_STEPS_BLUEPRINT,
        name,
        steps=steps,
        **kwargs
    )


def create_step_insert_after_blueprint(
    tmp_path: Path,
    name: str = "test_after",
    extends: str = "base.yaml",
    after_step: str = "custom:test_step",
    insert_step: str = "custom:test_identity_step",
    **kwargs
) -> Path:
    """Create a blueprint with insert after operation.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        extends: Parent blueprint filename
        after_step: Step to insert after
        insert_step: Step to insert
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    return create_blueprint_file(
        tmp_path,
        STEP_INSERT_AFTER_BLUEPRINT,
        name,
        extends=extends,
        after_step=after_step,
        insert_step=insert_step,
        **kwargs
    )


def create_step_insert_start_blueprint(
    tmp_path: Path,
    name: str = "test_start",
    extends: str = "base.yaml",
    insert_step: str = "custom:test_transform_sequence_step",
    **kwargs
) -> Path:
    """Create a blueprint with insert at start operation.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        extends: Parent blueprint filename
        insert_step: Step to insert at start
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    return create_blueprint_file(
        tmp_path,
        STEP_INSERT_START_BLUEPRINT,
        name,
        extends=extends,
        insert_step=insert_step,
        **kwargs
    )


def create_step_insert_end_blueprint(
    tmp_path: Path,
    name: str = "test_end",
    extends: str = "base.yaml",
    insert_step: str = "custom:test_identity_step",
    **kwargs
) -> Path:
    """Create a blueprint with insert at end operation.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        extends: Parent blueprint filename
        insert_step: Step to insert at end
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    return create_blueprint_file(
        tmp_path,
        STEP_INSERT_END_BLUEPRINT,
        name,
        extends=extends,
        insert_step=insert_step,
        **kwargs
    )


def create_step_replace_blueprint(
    tmp_path: Path,
    name: str = "test_replace",
    extends: str = "base.yaml",
    replace_step: str = "custom:test_step1",
    with_step: str = "custom:test_identity_step",
    **kwargs
) -> Path:
    """Create a blueprint with replace operation.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        extends: Parent blueprint filename
        replace_step: Step to replace
        with_step: Replacement step
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    return create_blueprint_file(
        tmp_path,
        STEP_REPLACE_BLUEPRINT,
        name,
        extends=extends,
        replace_step=replace_step,
        with_step=with_step,
        **kwargs
    )


def create_step_remove_blueprint(
    tmp_path: Path,
    name: str = "test_remove",
    extends: str = "base.yaml",
    remove_step: str = "custom:test_step1",
    **kwargs
) -> Path:
    """Create a blueprint with remove operation.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        extends: Parent blueprint filename
        remove_step: Step to remove
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    return create_blueprint_file(
        tmp_path,
        STEP_REMOVE_BLUEPRINT,
        name,
        extends=extends,
        remove_step=remove_step,
        **kwargs
    )


def create_branch_points_blueprint(
    tmp_path: Path,
    name: str = "test_branches",
    **kwargs
) -> Path:
    """Create a blueprint with branch points and skip operators.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    return create_blueprint_file(
        tmp_path,
        BRANCH_POINTS_BLUEPRINT,
        name,
        **kwargs
    )


def create_inheritance_parent(
    tmp_path: Path,
    name: str = "parent",
    steps: Optional[List[str]] = None,
    kernels: Optional[List[str]] = None,
    board: Optional[str] = None,
    **kwargs
) -> Path:
    """Create a parent blueprint for inheritance testing.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        steps: List of step names
        kernels: List of kernel names
        board: Board name (optional)
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    if steps is None:
        steps = ["custom:test_step", "custom:test_step1"]

    if kernels is None:
        kernels = []  # Don't include kernels by default to avoid validation issues

    # Format board as YAML (or omit if None)
    board_yaml = f"\nboard: {board}" if board else ""

    # Format kernels as YAML array (or omit if empty)
    kernels_yaml = ""
    if kernels:
        kernels_yaml = "\n  kernels:"
        for kernel in kernels:
            kernels_yaml += f"\n    - {kernel}"

    return create_blueprint_file(
        tmp_path,
        INHERITANCE_PARENT_BLUEPRINT,
        name,
        steps=steps,
        board=board_yaml,
        kernels=kernels_yaml,
        **kwargs
    )


def create_inheritance_grandparent(
    tmp_path: Path,
    name: str = "grandparent",
    steps: Optional[List[str]] = None,
    board: Optional[str] = None,
    **kwargs
) -> Path:
    """Create a grandparent blueprint for inheritance testing.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        steps: List of step names
        board: Board name (optional)
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    if steps is None:
        steps = ["custom:test_step"]

    # Format board as YAML (or omit if None)
    board_yaml = f"\nboard: {board}" if board else ""

    return create_blueprint_file(
        tmp_path,
        INHERITANCE_GRANDPARENT_BLUEPRINT,
        name,
        steps=steps,
        board=board_yaml,
        **kwargs
    )


def create_step_range_blueprint(
    tmp_path: Path,
    name: str = "test_step_range",
    start_step: Optional[str] = None,
    stop_step: Optional[str] = None,
    steps: Optional[List[str]] = None,
    **kwargs
) -> Path:
    """Create a blueprint with step range control.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        start_step: Start step name (optional)
        stop_step: Stop step name (optional)
        steps: List of step names
        **kwargs: Additional template variables

    Returns:
        Path to created blueprint file
    """
    if steps is None:
        steps = ["test_step", "test_step1", "test_step2", "test_step3"]

    # Build template conditionally
    template = STEP_RANGE_BLUEPRINT
    if start_step is None:
        template = template.replace("start_step: {start_step}\n", "")
    if stop_step is None:
        template = template.replace("stop_step: {stop_step}\n", "")

    return create_blueprint_file(
        tmp_path,
        template,
        name,
        start_step=start_step or "",
        stop_step=stop_step or "",
        steps=steps,
        **kwargs
    )


# ============================================================================
# DSE Blueprint Fixtures - pytest fixtures for common blueprint patterns
# ============================================================================

# FINN Pipeline Presets
# These are standard FINN pipeline configurations validated against FINN's
# estimate_only_dataflow_steps (deps/finn/src/finn/builder/build_dataflow_config.py:132)

FINN_PIPELINE_MINIMAL = [
    'finn:streamline',  # Basic cleanup and optimization
    'finn:tidy_up',     # Additional cleanup
]

FINN_PIPELINE_ESTIMATES = [
    # Complete pipeline for resource/performance estimates
    'finn:streamline',                    # Model optimization
    'finn:convert_to_hw',                 # Convert to hardware layers
    'finn:create_dataflow_partition',     # Create dataflow partitions
    'finn:specialize_layers',             # Specialize to HLS/RTL variants
    'finn:target_fps_parallelization',    # Auto-set PE/SIMD from target_fps
    'finn:generate_estimate_reports',     # Generate resource estimates
]

FINN_PIPELINE_STITCHED_IP = [
    # Full pipeline for stitched IP generation (no synthesis)
    'finn:streamline',
    'finn:convert_to_hw',
    'finn:create_dataflow_partition',
    'finn:specialize_layers',
    'finn:target_fps_parallelization',
    'finn:hw_codegen',                    # Generate HLS code
    'finn:hw_ipgen',                      # Generate IP
    'finn:set_fifo_depths',               # Configure FIFOs
    'finn:create_stitched_ip',            # Stitch IP together
]


def create_finn_blueprint(
    tmp_path: Path,
    name: str = "finn_test",
    steps: Optional[List[str]] = None,
    clock_ns: float = 5.0,
    target_fps: int = 100,  # Required for estimate pipelines
    **kwargs
) -> Path:
    """Create blueprint with FINN steps for integration tests.

    IMPORTANT: This creates estimate-only pipelines by default. For estimates,
    you MUST provide target_fps (defaults to 100) to enable auto-parallelization.
    Without target_fps, generate_estimate_reports will fail with empty PE/SIMD.

    Default pipeline (FINN_PIPELINE_ESTIMATES):
    1. finn:streamline - Model optimization (~5-10s)
    2. finn:convert_to_hw - Convert to hardware layers (~5s)
    3. finn:create_dataflow_partition - Create dataflow partitions (~1s)
    4. finn:specialize_layers - Specialize to HLS/RTL variants (~2s)
    5. finn:target_fps_parallelization - Auto-set PE/SIMD from target_fps (~1s)
    6. finn:generate_estimate_reports - Generate resource estimates (~5s)

    Total execution time: ~20-30 seconds per segment

    Args:
        tmp_path: pytest tmp_path fixture
        name: Blueprint name
        steps: List of FINN step names (defaults to FINN_PIPELINE_ESTIMATES)
        clock_ns: Clock period in nanoseconds
        target_fps: Target frames per second (required for auto-parallelization)
                   Set to 100 for fast tests, higher for more aggressive parallelization
        **kwargs: Additional blueprint parameters

    Returns:
        Path to created blueprint file

    Example:
        # Use default estimate pipeline
        blueprint = create_finn_blueprint(tmp_path, name="test")

        # Use minimal pipeline (no estimates)
        blueprint = create_finn_blueprint(
            tmp_path, name="minimal",
            steps=FINN_PIPELINE_MINIMAL,
            target_fps=None  # Not needed for minimal
        )

        # Custom pipeline
        blueprint = create_finn_blueprint(
            tmp_path, name="custom",
            steps=['finn:streamline', 'finn:tidy_up'],
            target_fps=None
        )
    """
    if steps is None:
        steps = FINN_PIPELINE_ESTIMATES

    return create_full_blueprint(
        tmp_path,
        name=name,
        steps=steps,
        clock_ns=clock_ns,
        target_fps=target_fps,
        output="estimates",
        board="Pynq-Z1",  # Required for FINN
        **kwargs
    )


@pytest.fixture
def finn_blueprint(tmp_path) -> Path:
    """Minimal FINN blueprint fixture for integration tests.

    Uses fast FINN steps suitable for CI:
    - finn:streamline
    - finn:generate_estimate_reports

    Execution time: ~30-60 seconds
    """
    return create_finn_blueprint(tmp_path)


@pytest.fixture
def minimal_blueprint(tmp_path, simple_onnx_model) -> Path:
    """Bare minimum valid blueprint for DSE.

    Creates the simplest possible blueprint:
    - Uses simple_onnx_model fixture
    - 3 linear steps (no branches)
    - Estimates output only (fast, no FINN build)
    - Default clock period (5.0ns)

    Use for:
    - Blueprint parsing tests
    - Tree construction tests
    - Config validation tests
    - Fast unit tests (<1 second)

    Args:
        tmp_path: pytest tmp_path fixture
        simple_onnx_model: ONNX model fixture

    Returns:
        Path to minimal blueprint YAML file
    """
    return create_minimal_blueprint(
        tmp_path,
        name="minimal_dse",
        steps=["test_step", "test_step1", "test_step2"],
        clock_ns=5.0
    )


@pytest.fixture
def rtl_blueprint(tmp_path, quantized_onnx_model) -> Path:
    """Blueprint for RTL simulation (stub - requires FINN).

    Creates blueprint with RTL output type:
    - Uses quantized_onnx_model (FINN-compatible)
    - RTL simulation output
    - Board configuration required (Pynq-Z1)
    - Slower clock for RTL (10.0ns)

    WARNING: Actually running this blueprint requires:
    - Full FINN installation with Vivado HLS
    - RTL simulation tools (QuestaSim/Verilator)
    - Execution time: 30 minutes to hours

    Use @pytest.mark.rtlsim marker for tests using this fixture.

    This is a STUB for Phase 4 - creates valid YAML but not meant
    to be executed. Real RTL tests come in later phases.

    Args:
        tmp_path: pytest tmp_path fixture
        quantized_onnx_model: Quantized ONNX model fixture

    Returns:
        Path to RTL blueprint YAML file
    """
    return create_full_blueprint(
        tmp_path,
        name="rtl_sim",
        description="RTL simulation blueprint (stub)",
        output="rtl",
        board="Pynq-Z1",  # Common FINN development board
        clock_ns=10.0,    # Slower clock for RTL
        steps=["test_step1", "test_step2"]  # Simplified for stub
    )


@pytest.fixture
def bitfile_blueprint(tmp_path, quantized_onnx_model) -> Path:
    """Blueprint for bitfile generation (stub - requires Vivado).

    Creates blueprint with bitfile output:
    - Full hardware compilation to FPGA bitstream
    - Requires Xilinx Vivado toolchain
    - Board configuration required (Pynq-Z1)
    - Production-ready clock speed (5.0ns = 200MHz)

    WARNING: Actually running this blueprint requires:
    - Xilinx Vivado (commercial FPGA tools)
    - Valid board files and licenses
    - Execution time: hours to days

    Use @pytest.mark.bitfile and @pytest.mark.hardware markers
    for tests using this fixture. Only run on CI with FPGA access.

    This is a STUB for Phase 4 - creates valid YAML but not meant
    to be executed. Real bitfile tests require hardware CI runners.

    Args:
        tmp_path: pytest tmp_path fixture
        quantized_onnx_model: Quantized ONNX model fixture

    Returns:
        Path to bitfile blueprint YAML file
    """
    return create_full_blueprint(
        tmp_path,
        name="bitfile_gen",
        description="Bitfile generation blueprint (stub)",
        output="bitfile",
        board="Pynq-Z1",  # Common FINN deployment board
        clock_ns=5.0,     # Target 200MHz
        steps=["test_step1", "test_step2"]  # Simplified for stub
    )
