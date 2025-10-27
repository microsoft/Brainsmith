"""Assertion helpers for parity testing.

This module provides consistent assertion formatting and error messages
for parity tests between manual and auto implementations.

Key Features:
- Centralized error message formatting
- Consistent "Manual vs Auto" comparison format
- Specialized helpers for common comparison types
- Clear, actionable error messages

Usage:
    from tests.parity.assertions import assert_shapes_match, assert_datatypes_match

    # In test methods:
    assert_shapes_match(manual_shape, auto_shape, index=0, kind="normal input")
    assert_datatypes_match(manual_dt, auto_dt, index=0, direction="Input")
"""

from typing import Any, Callable, Optional
import numpy as np
from qonnx.core.datatype import DataType


class ParityAssertion:
    """Base class for parity assertion helpers.

    Provides consistent error message formatting across all parity tests.
    """

    @staticmethod
    def format_mismatch(
        description: str,
        manual_value: Any,
        auto_value: Any,
        formatter: Optional[Callable[[Any], str]] = None
    ) -> str:
        """Format standard mismatch error message.

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
        fmt = formatter if formatter else str

        lines = [
            f"{description} mismatch:",
            f"  Manual: {fmt(manual_value)}",
            f"  Auto:   {fmt(auto_value)}"
        ]

        return "\n".join(lines)

    @staticmethod
    def assert_equal(
        manual_value: Any,
        auto_value: Any,
        description: str,
        formatter: Optional[Callable[[Any], str]] = None
    ) -> None:
        """Assert values are equal with consistent error formatting.

        Args:
            manual_value: Value from manual implementation
            auto_value: Value from auto implementation
            description: What is being compared
            formatter: Optional function to format values

        Raises:
            AssertionError: If values differ
        """
        if manual_value != auto_value:
            msg = ParityAssertion.format_mismatch(
                description, manual_value, auto_value, formatter
            )
            raise AssertionError(msg)


# =============================================================================
# Specialized Assertion Helpers
# =============================================================================

def assert_shapes_match(
    manual_shape: tuple,
    auto_shape: tuple,
    index: int,
    kind: str
) -> None:
    """Assert tensor shapes match between implementations.

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

        >>> assert_shapes_match((1, 768), (1, 769), index=0, kind="normal input")
        AssertionError: Input 0 normal input shape mismatch:
          Manual: (1, 768)
          Auto:   (1, 769)
    """
    if manual_shape != auto_shape:
        # Capitalize first letter if kind starts with direction
        if kind.lower().startswith(('input', 'output')):
            description = f"{kind.capitalize()} {index} shape"
        else:
            description = f"Index {index} {kind} shape"

        msg = ParityAssertion.format_mismatch(
            description, manual_shape, auto_shape
        )
        raise AssertionError(msg)


def assert_datatypes_match(
    manual_datatype: DataType,
    auto_datatype: DataType,
    index: int,
    direction: str
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
    manual_width: int,
    auto_width: int,
    index: int,
    direction: str,
    unit: str = "bits"
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
        msg = ParityAssertion.format_mismatch(
            description, manual_width, auto_width, format_width
        )
        raise AssertionError(msg)


def assert_values_match(
    manual_value: Any,
    auto_value: Any,
    description: str,
    formatter: Optional[Callable[[Any], str]] = None
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
    ParityAssertion.assert_equal(
        manual_value, auto_value, description, formatter
    )


def assert_arrays_close(
    manual_array: np.ndarray,
    auto_array: np.ndarray,
    description: str,
    rtol: float = 1e-5,
    atol: float = 1e-6
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
        )
    )


def assert_model_tensors_match(
    manual_model,
    auto_model,
    tensor_name: str,
    description: str
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
            f"{description} ({tensor_name}) datatype in model",
            manual_dt,
            auto_dt,
            format_dt
        )
        raise AssertionError(msg)
