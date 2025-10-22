"""Blueprint creation helpers to eliminate YAML duplication in tests."""

from pathlib import Path
from typing import Dict, Any, List, Optional


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
board: {board}
save_intermediate_models: {save_intermediate_models}
design_space:
  steps: {steps}
"""

EXTENDS_BLUEPRINT = """
name: {name}
extends: {extends}
clock_ns: {clock_ns}
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
    - test_step
    - [test_step1, test_step2, "~"]  # Branch with skip
    - infer_kernels
    - [export_to_build, test_step]  # Branch without skip
    - test_step1
"""

INHERITANCE_PARENT_BLUEPRINT = """
name: {name}
clock_ns: {clock_ns}
board: V70
design_space:
  steps: {steps}
  kernels:
{kernels}
"""

INHERITANCE_GRANDPARENT_BLUEPRINT = """
name: {name}
clock_ns: {clock_ns}
board: V100
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
    """
    Create a blueprint YAML file from a template.

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
    # If kernel_backends is already a formatted YAML string, leave it as is

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
    """Create a minimal blueprint with just required fields."""
    return create_blueprint_file(tmp_path, MINIMAL_BLUEPRINT, name, **kwargs)


def create_full_blueprint(
    tmp_path: Path,
    name: str = "test_full",
    description: str = "Test blueprint",
    output: str = "bitfile",
    board: str = "V80",
    save_intermediate_models: bool = True,
    **kwargs
) -> Path:
    """Create a full blueprint with all common fields."""
    return create_blueprint_file(
        tmp_path,
        FULL_BLUEPRINT,
        name,
        description=description,
        output=output,
        board=board,
        save_intermediate_models=save_intermediate_models,
        **kwargs
    )


def create_extends_blueprint(
    tmp_path: Path,
    name: str = "test_child",
    extends: str = "parent.yaml",
    **kwargs
) -> Path:
    """Create a blueprint that extends another (inheritance)."""
    return create_blueprint_file(
        tmp_path,
        EXTENDS_BLUEPRINT,
        name,
        extends=extends,
        **kwargs
    )


def create_base_steps_blueprint(
    tmp_path: Path,
    name: str = "base",
    steps: Optional[List[str]] = None,
    **kwargs
) -> Path:
    """Create a base blueprint for step operations testing."""
    if steps is None:
        steps = ["test_step", "test_step1", "test_step2"]

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
    after_step: str = "test_step",
    insert_step: str = "infer_kernels",
    **kwargs
) -> Path:
    """Create a blueprint with insert after operation."""
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
    insert_step: str = "export_to_build",
    **kwargs
) -> Path:
    """Create a blueprint with insert at start operation."""
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
    insert_step: str = "infer_kernels",
    **kwargs
) -> Path:
    """Create a blueprint with insert at end operation."""
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
    replace_step: str = "test_step1",
    with_step: str = "infer_kernels",
    **kwargs
) -> Path:
    """Create a blueprint with replace operation."""
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
    remove_step: str = "test_step1",
    **kwargs
) -> Path:
    """Create a blueprint with remove operation."""
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
    """Create a blueprint with branch points and skip operators."""
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
    **kwargs
) -> Path:
    """Create a parent blueprint for inheritance testing."""
    if steps is None:
        steps = ["test_step1", "test_step2"]

    if kernels is None:
        kernels = ["TestKernel", "TestKernelWithBackends"]

    # Format kernels as YAML array
    kernels_yaml = ""
    for kernel in kernels:
        kernels_yaml += f"    - {kernel}\n"

    return create_blueprint_file(
        tmp_path,
        INHERITANCE_PARENT_BLUEPRINT,
        name,
        steps=steps,
        kernels=kernels_yaml,
        **kwargs
    )


def create_inheritance_grandparent(
    tmp_path: Path,
    name: str = "grandparent",
    steps: Optional[List[str]] = None,
    **kwargs
) -> Path:
    """Create a grandparent blueprint for inheritance testing."""
    if steps is None:
        steps = ["test_step1"]
    return create_blueprint_file(
        tmp_path,
        INHERITANCE_GRANDPARENT_BLUEPRINT,
        name,
        steps=steps,
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
    """Create a blueprint with step range control."""
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
