"""DSE (Design Space Exploration) testing utilities.

This module provides fixtures and utilities specifically for testing
Brainsmith's DSE engine:

- blueprints.py: Blueprint YAML generation and pytest fixtures
- design_spaces.py: GlobalDesignSpace fixtures and configurations

These utilities support:
- Blueprint parsing tests
- Tree construction tests
- Segment execution tests
- Cache behavior tests
- Full DSE integration tests

Usage:
    from tests.fixtures.dse import (
        create_finn_blueprint,
        simple_design_space,
        branching_design_space,
    )
"""

# Blueprint generation utilities and fixtures
from .blueprints import (
    # YAML Templates
    MINIMAL_BLUEPRINT,
    FULL_BLUEPRINT,
    EXTENDS_BLUEPRINT,
    BASE_STEPS_BLUEPRINT,
    STEP_INSERT_AFTER_BLUEPRINT,
    STEP_INSERT_START_BLUEPRINT,
    STEP_INSERT_END_BLUEPRINT,
    STEP_REPLACE_BLUEPRINT,
    STEP_REMOVE_BLUEPRINT,
    BRANCH_POINTS_BLUEPRINT,
    INHERITANCE_PARENT_BLUEPRINT,
    INHERITANCE_GRANDPARENT_BLUEPRINT,
    STEP_RANGE_BLUEPRINT,
    # Helper functions
    create_blueprint_file,
    create_minimal_blueprint,
    create_full_blueprint,
    create_extends_blueprint,
    create_base_steps_blueprint,
    create_step_insert_after_blueprint,
    create_step_insert_start_blueprint,
    create_step_insert_end_blueprint,
    create_step_replace_blueprint,
    create_step_remove_blueprint,
    create_branch_points_blueprint,
    create_inheritance_parent,
    create_inheritance_grandparent,
    create_step_range_blueprint,
    # FINN pipeline presets
    FINN_PIPELINE_MINIMAL,
    FINN_PIPELINE_ESTIMATES,
    FINN_PIPELINE_STITCHED_IP,
    create_finn_blueprint,
    # pytest fixtures
    finn_blueprint,
    minimal_blueprint,
    rtl_blueprint,
    bitfile_blueprint,
)

# Design space fixtures
from .design_spaces import (
    # Configuration fixtures
    blueprint_config,
    base_finn_config,
    # Design space fixtures
    simple_design_space,
    branching_design_space,
    multi_branch_design_space,
)

__all__ = [
    # Templates
    "MINIMAL_BLUEPRINT",
    "FULL_BLUEPRINT",
    "EXTENDS_BLUEPRINT",
    "BASE_STEPS_BLUEPRINT",
    "STEP_INSERT_AFTER_BLUEPRINT",
    "STEP_INSERT_START_BLUEPRINT",
    "STEP_INSERT_END_BLUEPRINT",
    "STEP_REPLACE_BLUEPRINT",
    "STEP_REMOVE_BLUEPRINT",
    "BRANCH_POINTS_BLUEPRINT",
    "INHERITANCE_PARENT_BLUEPRINT",
    "INHERITANCE_GRANDPARENT_BLUEPRINT",
    "STEP_RANGE_BLUEPRINT",
    # Blueprint creation
    "create_blueprint_file",
    "create_minimal_blueprint",
    "create_full_blueprint",
    "create_extends_blueprint",
    "create_base_steps_blueprint",
    "create_step_insert_after_blueprint",
    "create_step_insert_start_blueprint",
    "create_step_insert_end_blueprint",
    "create_step_replace_blueprint",
    "create_step_remove_blueprint",
    "create_branch_points_blueprint",
    "create_inheritance_parent",
    "create_inheritance_grandparent",
    "create_step_range_blueprint",
    # FINN pipelines
    "FINN_PIPELINE_MINIMAL",
    "FINN_PIPELINE_ESTIMATES",
    "FINN_PIPELINE_STITCHED_IP",
    "create_finn_blueprint",
    # Blueprint fixtures
    "finn_blueprint",
    "minimal_blueprint",
    "rtl_blueprint",
    "bitfile_blueprint",
    # Design space config
    "blueprint_config",
    "base_finn_config",
    # Design spaces
    "simple_design_space",
    "branching_design_space",
    "multi_branch_design_space",
]
