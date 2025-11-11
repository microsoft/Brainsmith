"""Test assertion utilities for all test suites.

This module consolidates assertion helpers from tests/common/, tests/parity/,
and tests/utils/ into a single organized file.

Provides:
- AssertionHelper: Base class with consistent error formatting
- ParityAssertion: Kernel parity testing (Manual vs Auto)
- TreeAssertions: DSE tree structure validation
- ExecutionAssertions: DSE execution result validation
- BlueprintAssertions: DSE blueprint parsing validation
- Specialized helpers: assert_shapes_match, assert_arrays_close, etc.

Organization:
1. Base assertion helper (common to all tests)
2. Kernel testing assertions (parity/manual vs auto)
3. DSE testing assertions (tree, execution, blueprints)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.dse.config import DSEConfig
from brainsmith.dse.design_space import GlobalDesignSpace
from brainsmith.dse.tree import DSETree
from brainsmith.dse.types import OutputType, SegmentStatus, TreeExecutionResult
from tests.support.constants import (
    EFFICIENCY_DECIMAL_PLACES,
    EFFICIENCY_PERCENTAGE_MULTIPLIER,
    MIN_CHILDREN_FOR_BRANCH,
    NO_EFFICIENCY,
)

# =============================================================================
# Base Assertion Helper (from tests/common/assertions.py)
# =============================================================================


class AssertionHelper:
    """Base class for all test assertion helpers.

    Provides consistent error message formatting across all test suites.
    Subclasses can use these utilities for domain-specific assertions.
    """

    @staticmethod
    def format_mismatch(
        description: str, expected: Any, actual: Any, formatter: Callable[[Any], str] | None = None
    ) -> str:
        """Format standard mismatch error message.

        Args:
            description: What is being compared (e.g., "Node count", "Input shape")
            expected: Expected value
            actual: Actual value
            formatter: Optional function to format values (default: str())

        Returns:
            Formatted error message string

        Example:
            >>> msg = AssertionHelper.format_mismatch(
            ...     "Total nodes", 5, 7
            ... )
            >>> print(msg)
            Total nodes mismatch:
              Expected: 5
              Actual:   7
        """
        fmt = formatter if formatter else str

        lines = [
            f"{description} mismatch:",
            f"  Expected: {fmt(expected)}",
            f"  Actual:   {fmt(actual)}",
        ]

        return "\n".join(lines)

    @staticmethod
    def format_comparison(
        description: str,
        value_a: Any,
        value_b: Any,
        label_a: str = "A",
        label_b: str = "B",
        formatter: Callable[[Any], str] | None = None,
    ) -> str:
        """Format comparison error message (for parity testing).

        Args:
            description: What is being compared
            value_a: First value (e.g., manual implementation)
            value_b: Second value (e.g., auto implementation)
            label_a: Label for first value (default: "A")
            label_b: Label for second value (default: "B")
            formatter: Optional function to format values

        Returns:
            Formatted error message string

        Example:
            >>> msg = AssertionHelper.format_comparison(
            ...     "Output shape", (1, 768), (1, 769),
            ...     label_a="Manual", label_b="Auto"
            ... )
            >>> print(msg)
            Output shape mismatch:
              Manual: (1, 768)
              Auto:   (1, 769)
        """
        fmt = formatter if formatter else str

        lines = [
            f"{description} mismatch:",
            f"  {label_a}: {fmt(value_a)}",
            f"  {label_b}: {fmt(value_b)}",
        ]

        return "\n".join(lines)

    @staticmethod
    def assert_equal(
        expected: Any, actual: Any, description: str, formatter: Callable[[Any], str] | None = None
    ) -> None:
        """Assert values are equal with consistent error formatting.

        Args:
            expected: Expected value
            actual: Actual value
            description: What is being compared
            formatter: Optional function to format values

        Raises:
            AssertionError: If values differ

        Example:
            >>> AssertionHelper.assert_equal(5, 5, "Node count")
            # Passes silently

            >>> AssertionHelper.assert_equal(5, 7, "Node count")
            AssertionError: Node count mismatch:
              Expected: 5
              Actual:   7
        """
        if expected != actual:
            msg = AssertionHelper.format_mismatch(description, expected, actual, formatter)
            raise AssertionError(msg)

    @staticmethod
    def assert_comparison(
        value_a: Any,
        value_b: Any,
        description: str,
        label_a: str = "A",
        label_b: str = "B",
        formatter: Callable[[Any], str] | None = None,
    ) -> None:
        """Assert two values match (for parity testing).

        Args:
            value_a: First value to compare
            value_b: Second value to compare
            description: What is being compared
            label_a: Label for first value
            label_b: Label for second value
            formatter: Optional function to format values

        Raises:
            AssertionError: If values differ
        """
        if value_a != value_b:
            msg = AssertionHelper.format_comparison(
                description, value_a, value_b, label_a, label_b, formatter
            )
            raise AssertionError(msg)


# =============================================================================
# Kernel Testing Assertions (from tests/parity/assertions.py)
# Parity Testing: Manual vs Auto Implementation Comparison
# =============================================================================


class ParityAssertion(AssertionHelper):
    """Parity-specific assertion helper.

    Extends AssertionHelper with "Manual vs Auto" comparison semantics.
    Provides consistent error message formatting for parity tests.
    """

    @staticmethod
    def format_mismatch(
        description: str,
        manual_value: Any,
        auto_value: Any,
        formatter: Callable[[Any], str] | None = None,
    ) -> str:
        """Format parity mismatch error message (Manual vs Auto).

        Args:
            description: What is being compared (e.g., "Input 0 normal shape")
            manual_value: Value from manual implementation
            auto_value: Value from auto implementation
            formatter: Optional function to format values (default: str())

        Returns:
            Formatted error message string

        Example:
            >>> msg = ParityAssertion.format_mismatch(
            ...     "Input 0 shape", (1, 768), (1, 769)
            ... )
            >>> print(msg)
            Input 0 shape mismatch:
              Manual: (1, 768)
              Auto:   (1, 769)
        """
        return AssertionHelper.format_comparison(
            description,
            manual_value,
            auto_value,
            label_a="Manual",
            label_b="Auto",
            formatter=formatter,
        )

    @staticmethod
    def assert_equal(
        manual_value: Any,
        auto_value: Any,
        description: str,
        formatter: Callable[[Any], str] | None = None,
    ) -> None:
        """Assert parity between manual and auto implementations.

        Args:
            manual_value: Value from manual implementation
            auto_value: Value from auto implementation
            description: What is being compared
            formatter: Optional function to format values

        Raises:
            AssertionError: If values differ
        """
        AssertionHelper.assert_comparison(
            manual_value,
            auto_value,
            description,
            label_a="Manual",
            label_b="Auto",
            formatter=formatter,
        )


# -----------------------------------------------------------------------------
# Specialized Assertion Helpers for Kernel Testing
# -----------------------------------------------------------------------------


def assert_shapes_match(
    manual_shape: tuple[int, ...], auto_shape: tuple[int, ...], index: int, kind: str
) -> None:
    """Assert tensor shapes match between implementations.

    Compares shape values only, ignoring container type (tuple vs list).
    This handles the FINN inconsistency where some methods return lists
    while others return tuples.

    Args:
        manual_shape: Shape from manual implementation
        auto_shape: Shape from auto implementation
        index: Input/output index
        kind: Description (e.g., "normal input", "folded output")

    Raises:
        AssertionError: If shapes differ

    Example:
        >>> assert_shapes_match((1, 768), (1, 768), index=0, kind="normal input")
        # Passes silently

        >>> assert_shapes_match((1, 768), [1, 768], index=0, kind="normal input")
        # Passes silently (same values, different container type)

        >>> assert_shapes_match((1, 768), (1, 769), index=0, kind="normal input")
        AssertionError: Input 0 normal input shape mismatch:
          Manual: (1, 768)
          Auto:   (1, 769)
    """
    # Compare values only, not container types (tuple vs list)
    # Convert to tuple for comparison to handle FINN's inconsistent return types
    if tuple(manual_shape) != tuple(auto_shape):
        # Capitalize first letter if kind starts with direction
        if kind.lower().startswith(("input", "output")):
            description = f"{kind.capitalize()} {index} shape"
        else:
            description = f"Index {index} {kind} shape"

        msg = ParityAssertion.format_mismatch(description, manual_shape, auto_shape)
        raise AssertionError(msg)


def assert_datatypes_match(
    manual_datatype: DataType, auto_datatype: DataType, index: int, direction: str
) -> None:
    """Assert datatypes match between implementations.

    Formats datatypes as "name (bitwidth bits)" for clarity.

    Args:
        manual_datatype: DataType from manual implementation
        auto_datatype: DataType from auto implementation
        index: Input/output index
        direction: "Input" or "Output"

    Raises:
        AssertionError: If datatypes differ

    Example:
        >>> from qonnx.core.datatype import DataType
        >>> dt1 = DataType['INT8']
        >>> dt2 = DataType['INT8']
        >>> assert_datatypes_match(dt1, dt2, index=0, direction="Input")
        # Passes silently

        >>> dt3 = DataType['INT16']
        >>> assert_datatypes_match(dt1, dt3, index=0, direction="Input")
        AssertionError: Input 0 datatype mismatch:
          Manual: INT8 (8 bits)
          Auto:   INT16 (16 bits)
    """
    if manual_datatype != auto_datatype:

        def format_dt(dt: DataType) -> str:
            if dt is None:
                return "None"
            return f"{dt.name} ({dt.bitwidth()} bits)"

        description = f"{direction} {index} datatype"
        msg = ParityAssertion.format_mismatch(
            description, manual_datatype, auto_datatype, format_dt
        )
        raise AssertionError(msg)


def assert_widths_match(
    manual_width: int, auto_width: int, index: int, direction: str, unit: str = "bits"
) -> None:
    """Assert stream widths match between implementations.

    Args:
        manual_width: Width from manual implementation
        auto_width: Width from auto implementation
        index: Input/output index
        direction: "Input" or "Output"
        unit: Unit string (default: "bits")

    Raises:
        AssertionError: If widths differ

    Example:
        >>> assert_widths_match(16, 16, index=0, direction="Input")
        # Passes silently

        >>> assert_widths_match(16, 32, index=0, direction="Input", unit="bits")
        AssertionError: Input 0 stream width mismatch:
          Manual: 16 bits
          Auto:   32 bits
    """
    if manual_width != auto_width:

        def format_width(width: int) -> str:
            return f"{width} {unit}"

        description = f"{direction} {index} stream width"
        msg = ParityAssertion.format_mismatch(description, manual_width, auto_width, format_width)
        raise AssertionError(msg)


def assert_values_match(
    manual_value: Any,
    auto_value: Any,
    description: str,
    formatter: Callable[[Any], str] | None = None,
) -> None:
    """Generic assertion for any value comparison.

    Args:
        manual_value: Value from manual implementation
        auto_value: Value from auto implementation
        description: What is being compared
        formatter: Optional function to format values for display

    Raises:
        AssertionError: If values differ

    Example:
        >>> assert_values_match(42, 42, "expected cycles")
        # Passes silently

        >>> assert_values_match(42, 43, "expected cycles")
        AssertionError: expected cycles mismatch:
          Manual: 42
          Auto:   43
    """
    ParityAssertion.assert_equal(manual_value, auto_value, description, formatter)


def assert_arrays_close(
    manual_array: np.ndarray,
    auto_array: np.ndarray,
    description: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Assert numpy arrays are numerically close.

    Provides detailed error messages showing shapes and value differences.

    Args:
        manual_array: Array from manual implementation
        auto_array: Array from auto implementation
        description: What is being compared (e.g., "Output 0 (out_V)")
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose

    Raises:
        AssertionError: If arrays differ

    Example:
        >>> a1 = np.array([1.0, 2.0, 3.0])
        >>> a2 = np.array([1.0, 2.0, 3.0])
        >>> assert_arrays_close(a1, a2, "output tensor")
        # Passes silently

        >>> a3 = np.array([1.0, 2.1, 3.0])
        >>> assert_arrays_close(a1, a3, "output tensor")
        AssertionError: output tensor differs between backends
        ...
    """
    # First check shapes
    if manual_array.shape != auto_array.shape:
        msg = (
            f"{description} shape mismatch:\n"
            f"  Manual: {manual_array.shape}\n"
            f"  Auto:   {auto_array.shape}"
        )
        raise AssertionError(msg)

    # Then check numerical equivalence
    np.testing.assert_allclose(
        manual_array,
        auto_array,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"\n"
            f"{'='*70}\n"
            f"{description} differs between backends\n"
            f"{'='*70}\n"
            f"Manual shape: {manual_array.shape}\n"
            f"Auto shape:   {auto_array.shape}\n"
            f"Tolerance: rtol={rtol}, atol={atol}\n"
            f"{'='*70}\n"
            f"\n"
            f"This indicates a numerical difference in outputs.\n"
            f"Check:\n"
            f"1. Datatype handling (accumulator precision, rounding)\n"
            f"2. Algorithm implementation (order of operations)\n"
            f"3. Parallelization effects (SIMD/PE folding)\n"
        ),
    )


def assert_model_tensors_match(
    manual_model: ModelWrapper, auto_model: ModelWrapper, tensor_name: str, description: str
) -> None:
    """Assert tensors in model graphs match.

    Checks if tensor exists and has matching datatype in both models.

    Args:
        manual_model: ModelWrapper from manual implementation
        auto_model: ModelWrapper from auto implementation
        tensor_name: Name of tensor to compare
        description: What is being compared

    Raises:
        AssertionError: If tensor datatypes differ
    """
    manual_dt = manual_model.get_tensor_datatype(tensor_name)
    auto_dt = auto_model.get_tensor_datatype(tensor_name)

    if manual_dt != auto_dt:

        def format_dt(dt):
            return f"{dt.name if dt else 'None'}"

        msg = ParityAssertion.format_mismatch(
            f"{description} ({tensor_name}) datatype in model", manual_dt, auto_dt, format_dt
        )
        raise AssertionError(msg)


# =============================================================================
# DSE Testing Assertions (from tests/utils/assertions.py)
# Design Space Exploration: Tree, Execution, and Blueprint Validation
# =============================================================================

# -----------------------------------------------------------------------------
# Data Classes for Expected Values
# -----------------------------------------------------------------------------


@dataclass
class ExpectedTreeStructure:
    """Expected structure for tree validation."""

    total_nodes: int
    total_leaves: int
    total_paths: int
    total_segments: int
    segment_efficiency: float | None = None


@dataclass
class ExpectedExecutionLevel:
    """Expected execution level for validation."""

    level: int
    nodes: list[str]


@dataclass
class ExpectedExecutionStats:
    """Expected execution statistics."""

    total: int
    successful: int
    failed: int = 0
    cached: int = 0
    skipped: int = 0


# -----------------------------------------------------------------------------
# TreeAssertions - DSE Tree Structure Validation
# -----------------------------------------------------------------------------


class TreeAssertions(AssertionHelper):
    """Helper class for DSE tree assertions.

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
        assert (
            stats["total_segments"] == expected.total_nodes
        ), f"Expected {expected.total_nodes} nodes, got {stats['total_segments']}"

        assert (
            stats["total_paths"] == expected.total_leaves
        ), f"Expected {expected.total_leaves} leaves, got {stats['total_paths']}"
        assert (
            stats["total_paths"] == expected.total_paths
        ), f"Expected {expected.total_paths} paths, got {stats['total_paths']}"

        assert (
            stats["total_segments"] == expected.total_segments
        ), f"Expected {expected.total_segments} segments, got {stats['total_segments']}"

        if expected.segment_efficiency is not None:
            assert (
                stats["segment_efficiency"] == expected.segment_efficiency
            ), f"Expected efficiency {expected.segment_efficiency}%, got {stats['segment_efficiency']}%"

    @staticmethod
    def assert_execution_order_structure(execution_order: list, tree: DSETree):
        """Assert basic execution order properties.

        Args:
            execution_order: List of nodes in execution order
            tree: The DSE tree
        """
        # Root should be first
        assert execution_order[0] == tree.root, "Root node should be first in execution order"

        assert (
            execution_order[0].segment_id == "root"
        ), f"Root segment_id should be 'root', got '{execution_order[0].segment_id}'"

    @staticmethod
    def assert_parent_child_relationships(tree: DSETree):
        """Assert correct parent-child relationships throughout tree.

        Args:
            tree: The DSE tree to validate
        """

        def _check_node_relationships(node, expected_parent=None):
            """Recursively check relationships."""
            if expected_parent is not None:
                assert (
                    node.parent == expected_parent
                ), f"Node {node.segment_id} has incorrect parent"

            # Check all children
            for child in node.children.values():
                assert child.parent == node, f"Child {child.segment_id} has incorrect parent"
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
            assert (not has_children) == is_leaf_property, (
                f"Node {node.segment_id} leaf property ({is_leaf_property}) "
                f"inconsistent with children ({has_children})"
            )

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
            assert has_multiple_children == is_branch_property, (
                f"Node {node.segment_id} branch property ({is_branch_property}) "
                f"inconsistent with children count ({len(node.children)})"
            )

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
        assert (
            len(execution_order) == expected.total_nodes
        ), f"Execution order length {len(execution_order)} != total nodes {expected.total_nodes}"

        TreeAssertions.assert_execution_order_structure(execution_order, tree)


# -----------------------------------------------------------------------------
# ExecutionAssertions - DSE Execution Result Validation
# -----------------------------------------------------------------------------


class ExecutionAssertions(AssertionHelper):
    """Helper class for DSE execution result assertions.

    Validates execution results, segment statuses, and statistics.
    Extends AssertionHelper for consistent error formatting.
    """

    @staticmethod
    def assert_execution_stats(result: TreeExecutionResult, expected: ExpectedExecutionStats):
        """Assert execution statistics match expected values.

        Args:
            result: The execution result to validate
            expected: Expected statistics
        """
        stats = result.compute_stats()

        assert (
            stats["total"] == expected.total
        ), f"Expected {expected.total} total segments, got {stats['total']}"

        assert (
            stats["successful"] == expected.successful
        ), f"Expected {expected.successful} successful segments, got {stats['successful']}"

        assert (
            stats["failed"] == expected.failed
        ), f"Expected {expected.failed} failed segments, got {stats['failed']}"

        assert (
            stats["cached"] == expected.cached
        ), f"Expected {expected.cached} cached segments, got {stats['cached']}"

        assert (
            stats["skipped"] == expected.skipped
        ), f"Expected {expected.skipped} skipped segments, got {stats['skipped']}"

    @staticmethod
    def assert_segment_status(
        result: TreeExecutionResult, segment_id: str, expected_status: SegmentStatus
    ):
        """Assert specific segment has expected status.

        Args:
            result: The execution result
            segment_id: ID of segment to check
            expected_status: Expected status
        """
        segment_result = result.segment_results.get(segment_id)

        assert segment_result is not None, (
            f"Segment '{segment_id}' not found in execution results. "
            f"Available segments: {list(result.segment_results.keys())}"
        )

        assert segment_result.status == expected_status, (
            f"Segment '{segment_id}' expected status {expected_status.value}, "
            f"got {segment_result.status.value}"
        )

    @staticmethod
    def assert_execution_success(result: TreeExecutionResult):
        """Assert at least one successful build exists.

        Args:
            result: The execution result to validate

        Raises:
            AssertionError: If no successful or cached builds exist
        """
        stats = result.compute_stats()
        valid_builds = stats["successful"] + stats["cached"]

        assert valid_builds > 0, (
            f"Expected at least one successful build, but got 0 successful and 0 cached. "
            f"Failed: {stats['failed']}, Skipped: {stats['skipped']}"
        )

    @staticmethod
    def assert_all_paths_executed(result: TreeExecutionResult, tree: DSETree):
        """Assert all leaf paths were executed.

        Args:
            result: The execution result
            tree: The DSE tree that was executed
        """
        tree_stats = tree.get_statistics()
        execution_stats = result.compute_stats()

        # Total executed should at least equal number of paths
        # (may be more if non-leaf segments are included)
        assert execution_stats["total"] >= tree_stats["total_paths"], (
            f"Expected at least {tree_stats['total_paths']} executed segments, "
            f"got {execution_stats['total']}"
        )

    @staticmethod
    def assert_segment_output_exists(result: TreeExecutionResult, segment_id: str):
        """Assert segment produced output model or directory.

        Args:
            result: The execution result
            segment_id: ID of segment to check
        """
        segment_result = result.segment_results.get(segment_id)

        assert segment_result is not None, f"Segment '{segment_id}' not found in execution results"

        has_output = (
            segment_result.output_model is not None or segment_result.output_dir is not None
        )

        assert has_output, f"Segment '{segment_id}' has no output_model or output_dir"

    @staticmethod
    def assert_no_failed_segments(result: TreeExecutionResult):
        """Assert no segments failed during execution.

        Args:
            result: The execution result to validate
        """
        stats = result.compute_stats()

        assert stats["failed"] == 0, f"Expected 0 failed segments, got {stats['failed']}"

    @staticmethod
    def assert_segment_execution_time(
        result: TreeExecutionResult, segment_id: str, min_time: float = 0.0
    ):
        """Assert segment execution time is within expected range.

        Args:
            result: The execution result
            segment_id: ID of segment to check
            min_time: Minimum expected execution time in seconds
        """
        segment_result = result.segment_results.get(segment_id)

        assert segment_result is not None, f"Segment '{segment_id}' not found in execution results"

        assert segment_result.execution_time >= min_time, (
            f"Segment '{segment_id}' execution time {segment_result.execution_time}s "
            f"less than minimum {min_time}s"
        )


# -----------------------------------------------------------------------------
# BlueprintAssertions - Blueprint Parsing Validation
# -----------------------------------------------------------------------------


class BlueprintAssertions(AssertionHelper):
    """Helper class for blueprint parsing assertions.

    Validates design space structure, configuration values, and inheritance.
    Extends AssertionHelper for consistent error formatting.
    """

    @staticmethod
    def assert_design_space_structure(
        design_space: GlobalDesignSpace,
        expected_steps: list[str | list[str]],
        expected_kernel_count: int | None = None,
        expected_model_path: str | None = None,
    ):
        """Assert design space has expected structure.

        Args:
            design_space: The parsed design space
            expected_steps: Expected step sequence (including branches)
            expected_kernel_count: Expected number of kernel backends
            expected_model_path: Expected model path
        """
        assert (
            design_space.steps == expected_steps
        ), f"Expected steps {expected_steps}, got {design_space.steps}"

        if expected_kernel_count is not None:
            actual_count = len(design_space.kernel_backends)
            assert (
                actual_count == expected_kernel_count
            ), f"Expected {expected_kernel_count} kernel backends, got {actual_count}"

        if expected_model_path is not None:
            assert (
                design_space.model_path == expected_model_path
            ), f"Expected model path '{expected_model_path}', got '{design_space.model_path}'"

    @staticmethod
    def assert_config_values(
        config: DSEConfig,
        expected_clock_ns: float | None = None,
        expected_board: str | None = None,
        expected_output: OutputType | None = None,
    ):
        """Assert DSEConfig has expected values.

        Args:
            config: The parsed configuration
            expected_clock_ns: Expected clock period
            expected_board: Expected board name
            expected_output: Expected output type
        """
        if expected_clock_ns is not None:
            assert (
                config.clock_ns == expected_clock_ns
            ), f"Expected clock_ns {expected_clock_ns}, got {config.clock_ns}"

        if expected_board is not None:
            assert (
                config.board == expected_board
            ), f"Expected board '{expected_board}', got '{config.board}'"

        if expected_output is not None:
            assert (
                config.output == expected_output
            ), f"Expected output {expected_output.value}, got {config.output.value}"

    @staticmethod
    def assert_step_sequence(
        design_space: GlobalDesignSpace, expected_sequence: list[str], ignore_branches: bool = False
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

        assert (
            actual == expected_sequence
        ), f"Expected step sequence {expected_sequence}, got {actual}"

    @staticmethod
    def assert_branch_point(
        design_space: GlobalDesignSpace, index: int, expected_options: list[str]
    ):
        """Assert specific index contains branch point with expected options.

        Args:
            design_space: The parsed design space
            index: Index of the branch point in steps
            expected_options: Expected branch options (including "~" if present)
        """
        assert index < len(
            design_space.steps
        ), f"Index {index} out of range for steps (length {len(design_space.steps)})"

        step = design_space.steps[index]

        assert isinstance(
            step, list
        ), f"Expected branch point at index {index}, got single step '{step}'"

        assert set(step) == set(
            expected_options
        ), f"Expected branch options {set(expected_options)}, got {set(step)}"

    @staticmethod
    def assert_inheritance_applied(
        child_config: DSEConfig, parent_values: dict[str, Any], child_overrides: dict[str, Any]
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
                assert actual_value == expected_value, (
                    f"Expected inherited value for '{key}': {expected_value}, "
                    f"got {actual_value}"
                )

        # Check child overrides
        for key, expected_value in child_overrides.items():
            actual_value = getattr(child_config, key)
            assert actual_value == expected_value, (
                f"Expected child override for '{key}': {expected_value}, " f"got {actual_value}"
            )

    @staticmethod
    def assert_step_operation_applied(
        design_space: GlobalDesignSpace, operation_type: str, expected_result: list[str]
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

        assert actual_steps == expected_result, (
            f"Step operation '{operation_type}' failed. "
            f"Expected {expected_result}, got {actual_steps}"
        )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def calculate_segment_efficiency(
    total_steps_with_segments: int, total_steps_without_segments: int
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
