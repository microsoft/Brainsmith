"""Shared assertion utilities for all test suites.

Provides a common base class for formatting assertion error messages
consistently across DSE, parity, and other test frameworks.

This module consolidates error message formatting logic that was previously
duplicated in tests/utils/assertions.py and tests/parity/assertions.py.
"""

from typing import Any, Callable, Optional


class AssertionHelper:
    """Base class for all test assertion helpers.

    Provides consistent error message formatting across all test suites.
    Subclasses can use these utilities for domain-specific assertions.
    """

    @staticmethod
    def format_mismatch(
        description: str,
        expected: Any,
        actual: Any,
        formatter: Optional[Callable[[Any], str]] = None
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
            f"  Actual:   {fmt(actual)}"
        ]

        return "\n".join(lines)

    @staticmethod
    def format_comparison(
        description: str,
        value_a: Any,
        value_b: Any,
        label_a: str = "A",
        label_b: str = "B",
        formatter: Optional[Callable[[Any], str]] = None
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
            f"  {label_b}: {fmt(value_b)}"
        ]

        return "\n".join(lines)

    @staticmethod
    def assert_equal(
        expected: Any,
        actual: Any,
        description: str,
        formatter: Optional[Callable[[Any], str]] = None
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
            msg = AssertionHelper.format_mismatch(
                description, expected, actual, formatter
            )
            raise AssertionError(msg)

    @staticmethod
    def assert_comparison(
        value_a: Any,
        value_b: Any,
        description: str,
        label_a: str = "A",
        label_b: str = "B",
        formatter: Optional[Callable[[Any], str]] = None
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
