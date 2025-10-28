"""Assertion helpers for DSE integration tests.

Provides three assertion classes:
- TreeAssertions: Validate DSE tree structure and relationships
- ExecutionAssertions: Validate execution results and statistics
- BlueprintAssertions: Validate blueprint parsing results

Ported from OLD_FOR_REFERENCE_ONLY/utils/tree_assertions.py
Extended with execution and blueprint assertions for comprehensive test coverage.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from brainsmith.dse.tree import DSETree
from brainsmith.dse.design_space import GlobalDesignSpace
from brainsmith.dse.config import DSEConfig
from brainsmith.dse.types import TreeExecutionResult, SegmentStatus, OutputType
from tests.common.constants import (
    MIN_CHILDREN_FOR_BRANCH,
    NO_EFFICIENCY,
    EFFICIENCY_DECIMAL_PLACES,
    EFFICIENCY_PERCENTAGE_MULTIPLIER
)
from tests.common.assertions import AssertionHelper


# ============================================================================
# Data Classes for Expected Values
# ============================================================================

@dataclass
class ExpectedTreeStructure:
    """Expected structure for tree validation."""
    total_nodes: int
    total_leaves: int
    total_paths: int
    total_segments: int
    segment_efficiency: Optional[float] = None


@dataclass
class ExpectedExecutionLevel:
    """Expected execution level for validation."""
    level: int
    nodes: List[str]


@dataclass
class ExpectedExecutionStats:
    """Expected execution statistics."""
    total: int
    successful: int
    failed: int = 0
    cached: int = 0
    skipped: int = 0


# ============================================================================
# TreeAssertions - DSE Tree Structure Validation
# ============================================================================

class TreeAssertions(AssertionHelper):
    """Helper class for DSE tree assertions.

    Ported from OLD_FOR_REFERENCE_ONLY/utils/tree_assertions.py
    Validates tree structure, relationships, and execution order.
    Extends AssertionHelper for consistent error formatting.
    """

    @staticmethod
    def assert_tree_structure(tree: DSETree, expected: ExpectedTreeStructure):
        """Assert basic tree structure properties.

        Args:
            tree: The DSE tree to validate
            expected: Expected structure properties
        """
        stats = tree.get_statistics()
        assert stats['total_segments'] == expected.total_nodes, \
            f"Expected {expected.total_nodes} nodes, got {stats['total_segments']}"

        assert stats['total_paths'] == expected.total_leaves, \
            f"Expected {expected.total_leaves} leaves, got {stats['total_paths']}"
        assert stats['total_paths'] == expected.total_paths, \
            f"Expected {expected.total_paths} paths, got {stats['total_paths']}"

        assert stats['total_segments'] == expected.total_segments, \
            f"Expected {expected.total_segments} segments, got {stats['total_segments']}"

        if expected.segment_efficiency is not None:
            assert stats['segment_efficiency'] == expected.segment_efficiency, \
                f"Expected efficiency {expected.segment_efficiency}%, got {stats['segment_efficiency']}%"

    @staticmethod
    def assert_execution_order_structure(execution_order: List, tree: DSETree):
        """Assert basic execution order properties.

        Args:
            execution_order: List of nodes in execution order
            tree: The DSE tree
        """
        # Root should be first
        assert execution_order[0] == tree.root, \
            "Root node should be first in execution order"

        assert execution_order[0].segment_id == "root", \
            f"Root segment_id should be 'root', got '{execution_order[0].segment_id}'"

    @staticmethod
    def assert_parent_child_relationships(tree: DSETree):
        """Assert correct parent-child relationships throughout tree.

        Args:
            tree: The DSE tree to validate
        """
        def _check_node_relationships(node, expected_parent=None):
            """Recursively check relationships."""
            if expected_parent is not None:
                assert node.parent == expected_parent, \
                    f"Node {node.segment_id} has incorrect parent"

            # Check all children
            for child in node.children.values():
                assert child.parent == node, \
                    f"Child {child.segment_id} has incorrect parent"
                _check_node_relationships(child, node)

        # Start from root (which has no parent)
        _check_node_relationships(tree.root)

    @staticmethod
    def assert_leaf_properties(tree: DSETree):
        """Assert correct leaf node properties.

        Args:
            tree: The DSE tree to validate
        """
        def _check_leaf_consistency(node):
            """Check leaf property consistency."""
            has_children = bool(node.children)
            is_leaf_property = node.is_leaf

            # Leaf property should match absence of children
            assert (not has_children) == is_leaf_property, \
                f"Node {node.segment_id} leaf property ({is_leaf_property}) " \
                f"inconsistent with children ({has_children})"

            # Recursively check children
            for child in node.children.values():
                _check_leaf_consistency(child)

        _check_leaf_consistency(tree.root)

    @staticmethod
    def assert_branch_point_properties(tree: DSETree):
        """Assert correct branch point properties.

        Args:
            tree: The DSE tree to validate
        """
        def _check_branch_consistency(node):
            """Check branch point consistency."""
            has_multiple_children = len(node.children) > MIN_CHILDREN_FOR_BRANCH
            is_branch_property = node.is_branch_point

            # Branch point property should match multiple children
            assert has_multiple_children == is_branch_property, \
                f"Node {node.segment_id} branch property ({is_branch_property}) " \
                f"inconsistent with children count ({len(node.children)})"

            # Recursively check children
            for child in node.children.values():
                _check_branch_consistency(child)

        _check_branch_consistency(tree.root)

    @staticmethod
    def assert_complete_tree_validation(tree: DSETree, expected: ExpectedTreeStructure):
        """Perform comprehensive tree validation.

        Args:
            tree: The DSE tree to validate
            expected: Expected structure properties
        """
        # Basic structure
        TreeAssertions.assert_tree_structure(tree, expected)

        # Relationship consistency
        TreeAssertions.assert_parent_child_relationships(tree)

        # Property consistency
        TreeAssertions.assert_leaf_properties(tree)
        TreeAssertions.assert_branch_point_properties(tree)

        # Execution order basics
        execution_order = tree.get_execution_order()
        assert len(execution_order) == expected.total_nodes, \
            f"Execution order length {len(execution_order)} != total nodes {expected.total_nodes}"

        TreeAssertions.assert_execution_order_structure(execution_order, tree)


# ============================================================================
# ExecutionAssertions - DSE Execution Result Validation
# ============================================================================

class ExecutionAssertions(AssertionHelper):
    """Helper class for DSE execution result assertions.

    Validates execution results, segment statuses, and statistics.
    Extends AssertionHelper for consistent error formatting.
    """

    @staticmethod
    def assert_execution_stats(
        result: TreeExecutionResult,
        expected: ExpectedExecutionStats
    ):
        """Assert execution statistics match expected values.

        Args:
            result: The execution result to validate
            expected: Expected statistics
        """
        stats = result.compute_stats()

        assert stats['total'] == expected.total, \
            f"Expected {expected.total} total segments, got {stats['total']}"

        assert stats['successful'] == expected.successful, \
            f"Expected {expected.successful} successful segments, got {stats['successful']}"

        assert stats['failed'] == expected.failed, \
            f"Expected {expected.failed} failed segments, got {stats['failed']}"

        assert stats['cached'] == expected.cached, \
            f"Expected {expected.cached} cached segments, got {stats['cached']}"

        assert stats['skipped'] == expected.skipped, \
            f"Expected {expected.skipped} skipped segments, got {stats['skipped']}"

    @staticmethod
    def assert_segment_status(
        result: TreeExecutionResult,
        segment_id: str,
        expected_status: SegmentStatus
    ):
        """Assert specific segment has expected status.

        Args:
            result: The execution result
            segment_id: ID of segment to check
            expected_status: Expected status
        """
        segment_result = result.segment_results.get(segment_id)

        assert segment_result is not None, \
            f"Segment '{segment_id}' not found in execution results. " \
            f"Available segments: {list(result.segment_results.keys())}"

        assert segment_result.status == expected_status, \
            f"Segment '{segment_id}' expected status {expected_status.value}, " \
            f"got {segment_result.status.value}"

    @staticmethod
    def assert_execution_success(result: TreeExecutionResult):
        """Assert at least one successful build exists.

        Args:
            result: The execution result to validate

        Raises:
            AssertionError: If no successful or cached builds exist
        """
        stats = result.compute_stats()
        valid_builds = stats['successful'] + stats['cached']

        assert valid_builds > 0, \
            f"Expected at least one successful build, but got 0 successful and 0 cached. " \
            f"Failed: {stats['failed']}, Skipped: {stats['skipped']}"

    @staticmethod
    def assert_all_paths_executed(
        result: TreeExecutionResult,
        tree: DSETree
    ):
        """Assert all leaf paths were executed.

        Args:
            result: The execution result
            tree: The DSE tree that was executed
        """
        tree_stats = tree.get_statistics()
        execution_stats = result.compute_stats()

        # Total executed should at least equal number of paths
        # (may be more if non-leaf segments are included)
        assert execution_stats['total'] >= tree_stats['total_paths'], \
            f"Expected at least {tree_stats['total_paths']} executed segments, " \
            f"got {execution_stats['total']}"

    @staticmethod
    def assert_segment_output_exists(
        result: TreeExecutionResult,
        segment_id: str
    ):
        """Assert segment produced output model or directory.

        Args:
            result: The execution result
            segment_id: ID of segment to check
        """
        segment_result = result.segment_results.get(segment_id)

        assert segment_result is not None, \
            f"Segment '{segment_id}' not found in execution results"

        has_output = (
            segment_result.output_model is not None or
            segment_result.output_dir is not None
        )

        assert has_output, \
            f"Segment '{segment_id}' has no output_model or output_dir"

    @staticmethod
    def assert_no_failed_segments(result: TreeExecutionResult):
        """Assert no segments failed during execution.

        Args:
            result: The execution result to validate
        """
        stats = result.compute_stats()

        assert stats['failed'] == 0, \
            f"Expected 0 failed segments, got {stats['failed']}"

    @staticmethod
    def assert_segment_execution_time(
        result: TreeExecutionResult,
        segment_id: str,
        min_time: float = 0.0
    ):
        """Assert segment execution time is within expected range.

        Args:
            result: The execution result
            segment_id: ID of segment to check
            min_time: Minimum expected execution time in seconds
        """
        segment_result = result.segment_results.get(segment_id)

        assert segment_result is not None, \
            f"Segment '{segment_id}' not found in execution results"

        assert segment_result.execution_time >= min_time, \
            f"Segment '{segment_id}' execution time {segment_result.execution_time}s " \
            f"less than minimum {min_time}s"


# ============================================================================
# BlueprintAssertions - Blueprint Parsing Validation
# ============================================================================

class BlueprintAssertions(AssertionHelper):
    """Helper class for blueprint parsing assertions.

    Validates design space structure, configuration values, and inheritance.
    Extends AssertionHelper for consistent error formatting.
    """

    @staticmethod
    def assert_design_space_structure(
        design_space: GlobalDesignSpace,
        expected_steps: List[Union[str, List[str]]],
        expected_kernel_count: Optional[int] = None,
        expected_model_path: Optional[str] = None
    ):
        """Assert design space has expected structure.

        Args:
            design_space: The parsed design space
            expected_steps: Expected step sequence (including branches)
            expected_kernel_count: Expected number of kernel backends
            expected_model_path: Expected model path
        """
        assert design_space.steps == expected_steps, \
            f"Expected steps {expected_steps}, got {design_space.steps}"

        if expected_kernel_count is not None:
            actual_count = len(design_space.kernel_backends)
            assert actual_count == expected_kernel_count, \
                f"Expected {expected_kernel_count} kernel backends, got {actual_count}"

        if expected_model_path is not None:
            assert design_space.model_path == expected_model_path, \
                f"Expected model path '{expected_model_path}', got '{design_space.model_path}'"

    @staticmethod
    def assert_config_values(
        config: DSEConfig,
        expected_clock_ns: Optional[float] = None,
        expected_board: Optional[str] = None,
        expected_output: Optional[OutputType] = None,
        expected_save_intermediate: Optional[bool] = None,
        expected_verify: Optional[bool] = None,
        expected_parallel_builds: Optional[int] = None
    ):
        """Assert DSEConfig has expected values.

        Args:
            config: The parsed configuration
            expected_clock_ns: Expected clock period
            expected_board: Expected board name
            expected_output: Expected output type
            expected_save_intermediate: Expected save_intermediate_models flag
            expected_verify: Expected verify flag
            expected_parallel_builds: Expected parallel builds count
        """
        if expected_clock_ns is not None:
            assert config.clock_ns == expected_clock_ns, \
                f"Expected clock_ns {expected_clock_ns}, got {config.clock_ns}"

        if expected_board is not None:
            assert config.board == expected_board, \
                f"Expected board '{expected_board}', got '{config.board}'"

        if expected_output is not None:
            assert config.output == expected_output, \
                f"Expected output {expected_output.value}, got {config.output.value}"

        if expected_save_intermediate is not None:
            assert config.save_intermediate_models == expected_save_intermediate, \
                f"Expected save_intermediate_models {expected_save_intermediate}, " \
                f"got {config.save_intermediate_models}"

        if expected_verify is not None:
            assert config.verify == expected_verify, \
                f"Expected verify {expected_verify}, got {config.verify}"

        if expected_parallel_builds is not None:
            assert config.parallel_builds == expected_parallel_builds, \
                f"Expected parallel_builds {expected_parallel_builds}, " \
                f"got {config.parallel_builds}"

    @staticmethod
    def assert_step_sequence(
        design_space: GlobalDesignSpace,
        expected_sequence: List[str],
        ignore_branches: bool = False
    ):
        """Assert steps appear in expected sequence.

        Args:
            design_space: The parsed design space
            expected_sequence: Expected step names in order
            ignore_branches: If True, flatten branch lists for comparison
        """
        actual = design_space.steps

        if ignore_branches:
            # Flatten branch lists and skip "~" operators
            flattened = []
            for step in design_space.steps:
                if isinstance(step, list):
                    flattened.extend([s for s in step if s != "~"])
                else:
                    flattened.append(step)
            actual = flattened

        assert actual == expected_sequence, \
            f"Expected step sequence {expected_sequence}, got {actual}"

    @staticmethod
    def assert_branch_point(
        design_space: GlobalDesignSpace,
        index: int,
        expected_options: List[str]
    ):
        """Assert specific index contains branch point with expected options.

        Args:
            design_space: The parsed design space
            index: Index of the branch point in steps
            expected_options: Expected branch options (including "~" if present)
        """
        assert index < len(design_space.steps), \
            f"Index {index} out of range for steps (length {len(design_space.steps)})"

        step = design_space.steps[index]

        assert isinstance(step, list), \
            f"Expected branch point at index {index}, got single step '{step}'"

        assert set(step) == set(expected_options), \
            f"Expected branch options {set(expected_options)}, got {set(step)}"

    @staticmethod
    def assert_inheritance_applied(
        child_config: DSEConfig,
        parent_values: Dict[str, Any],
        child_overrides: Dict[str, Any]
    ):
        """Assert inheritance correctly merged parent and child values.

        Args:
            child_config: The child configuration after parsing
            parent_values: Values that should be inherited from parent
            child_overrides: Values that child should override
        """
        # Check parent values that weren't overridden
        for key, expected_value in parent_values.items():
            if key not in child_overrides:
                actual_value = getattr(child_config, key)
                assert actual_value == expected_value, \
                    f"Expected inherited value for '{key}': {expected_value}, " \
                    f"got {actual_value}"

        # Check child overrides
        for key, expected_value in child_overrides.items():
            actual_value = getattr(child_config, key)
            assert actual_value == expected_value, \
                f"Expected child override for '{key}': {expected_value}, " \
                f"got {actual_value}"

    @staticmethod
    def assert_step_operation_applied(
        design_space: GlobalDesignSpace,
        operation_type: str,
        expected_result: List[str]
    ):
        """Assert step operation produced expected result.

        Args:
            design_space: The parsed design space
            operation_type: Type of operation ("insert_after", "replace", "remove", etc.)
            expected_result: Expected flattened step sequence
        """
        # Flatten steps for comparison (skip branches and "~")
        actual_steps = []
        for step in design_space.steps:
            if isinstance(step, list):
                actual_steps.extend([s for s in step if s != "~"])
            else:
                actual_steps.append(step)

        assert actual_steps == expected_result, \
            f"Step operation '{operation_type}' failed. " \
            f"Expected {expected_result}, got {actual_steps}"

    @staticmethod
    def assert_step_range(
        config: DSEConfig,
        expected_start: Optional[str] = None,
        expected_stop: Optional[str] = None
    ):
        """Assert step range control has expected values.

        Args:
            config: The parsed configuration
            expected_start: Expected start_step value
            expected_stop: Expected stop_step value
        """
        if expected_start is not None:
            assert config.start_step == expected_start, \
                f"Expected start_step '{expected_start}', got '{config.start_step}'"

        if expected_stop is not None:
            assert config.stop_step == expected_stop, \
                f"Expected stop_step '{expected_stop}', got '{config.stop_step}'"


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_segment_efficiency(
    total_steps_with_segments: int,
    total_steps_without_segments: int
) -> float:
    """Calculate expected segment efficiency.

    Args:
        total_steps_with_segments: Total steps when using segments
        total_steps_without_segments: Total steps without segments

    Returns:
        Efficiency percentage rounded to 1 decimal place
    """
    if total_steps_without_segments == 0:
        return NO_EFFICIENCY

    efficiency = EFFICIENCY_PERCENTAGE_MULTIPLIER * (
        1 - total_steps_with_segments / total_steps_without_segments
    )
    return round(efficiency, EFFICIENCY_DECIMAL_PLACES)
