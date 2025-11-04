############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Ordered dimension for Design Space Exploration (DSE) navigation.

This module provides OrderedDimension, a container for ordered sequences
of dimension values that supports navigation operations like stepping,
percentage-based indexing, and min/max access.

OrderedDimension is used for parallelization parameters (PE, SIMD, MW, MH)
and other explorable dimensions with natural ordering (depth, num_layers, etc.).

Contrasts with discrete (unordered) dimensions which are validated by FINN's
nodeattr system and stored as frozensets.
"""

import math
from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Tuple


@dataclass(frozen=True)
class OrderedDimension:
    """Ordered dimension for DSE navigation.

    Stores discrete values in sorted order, enabling navigation operations
    like stepping, percentage-based indexing, and min/max access.

    Used for parallelization parameters (PE, SIMD, MW, MH) and other
    explorable dimensions with natural ordering.

    Attributes:
        name: Dimension name (e.g., "SIMD", "PE", "depth")
        values: Sorted tuple of valid values
        default: Default value (None = minimum)

    Examples:
        >>> simd = OrderedDimension("SIMD", (1, 2, 4, 8, 16, 32, 64))
        >>> simd.min()
        1
        >>> simd.at_percentage(0.5)
        8
        >>> simd.step_up(8, n=2)
        32
    """

    name: str
    values: Tuple[int, ...]
    default: Optional[int] = None

    def __post_init__(self):
        """Validate invariants: sorted, unique, non-empty."""
        if not self.values:
            raise ValueError(f"OrderedDimension '{self.name}' has empty values")

        # Ensure tuple (not list)
        if not isinstance(self.values, tuple):
            object.__setattr__(self, 'values', tuple(self.values))

        # Validate sorted
        if self.values != tuple(sorted(self.values)):
            raise ValueError(
                f"OrderedDimension '{self.name}' values must be sorted ascending. "
                f"Got: {self.values}"
            )

        # Validate unique
        if len(self.values) != len(set(self.values)):
            duplicates = [v for v in self.values if self.values.count(v) > 1]
            raise ValueError(
                f"OrderedDimension '{self.name}' has duplicate values: {set(duplicates)}"
            )

        # Validate default (if specified)
        if self.default is not None and self.default not in self.values:
            raise ValueError(
                f"Default value {self.default} not in dimension '{self.name}'. "
                f"Valid values: {self.values}"
            )

    # =========================================================================
    # Core Properties
    # =========================================================================

    def get_default(self) -> int:
        """Get default value (explicit or minimum)."""
        return self.default if self.default is not None else self.values[0]

    def validate(self, value: int) -> bool:
        """Check if value is valid for this dimension."""
        return value in self.values

    # =========================================================================
    # Positional Access
    # =========================================================================

    def min(self) -> int:
        """Get minimum value (first in ordered sequence)."""
        return self.values[0]

    def max(self) -> int:
        """Get maximum value (last in ordered sequence)."""
        return self.values[-1]

    def at_index(self, idx: int) -> int:
        """Get value at index (supports negative indexing).

        Args:
            idx: Index position (0-based, supports negative like Python lists)

        Returns:
            Value at index

        Raises:
            IndexError: If index out of range

        Examples:
            >>> dim = OrderedDimension("PE", (1, 2, 4, 8, 16))
            >>> dim.at_index(0)
            1
            >>> dim.at_index(-1)
            16
            >>> dim.at_index(2)
            4
        """
        if not -len(self.values) <= idx < len(self.values):
            raise IndexError(
                f"Index {idx} out of range for dimension '{self.name}' "
                f"(length {len(self.values)})"
            )
        return self.values[idx]

    def index_of(self, value: int) -> int:
        """Get index of value in ordered sequence.

        Args:
            value: Value to find

        Returns:
            Zero-based index of value

        Raises:
            ValueError: If value not in dimension

        Examples:
            >>> dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16))
            >>> dim.index_of(4)
            2
            >>> dim.index_of(16)
            4
        """
        try:
            return self.values.index(value)
        except ValueError:
            raise ValueError(
                f"Value {value} not in dimension '{self.name}'. "
                f"Valid range: [{self.min()}, {self.max()}], "
                f"values: {self.values}"
            )

    # =========================================================================
    # Navigation (Relative Movement)
    # =========================================================================

    def step_up(self, current: int, n: int = 1) -> int:
        """Step up n positions from current value.

        Clamps at maximum if n steps would exceed bounds.

        Args:
            current: Current value (must be in dimension)
            n: Number of steps to move up (positive integer)

        Returns:
            New value n steps up (clamped at max)

        Raises:
            ValueError: If current value not in dimension or n < 0

        Examples:
            >>> dim = OrderedDimension("PE", (1, 2, 4, 8, 16, 32, 64))
            >>> dim.step_up(4, 1)
            8
            >>> dim.step_up(4, 2)
            16
            >>> dim.step_up(32, 10)
            64  # Clamped at max
        """
        if n < 0:
            raise ValueError(f"step_up requires n >= 0, got {n}")

        idx = self.index_of(current)
        new_idx = min(idx + n, len(self.values) - 1)
        return self.values[new_idx]

    def step_down(self, current: int, n: int = 1) -> int:
        """Step down n positions from current value.

        Clamps at minimum if n steps would go below bounds.

        Args:
            current: Current value (must be in dimension)
            n: Number of steps to move down (positive integer)

        Returns:
            New value n steps down (clamped at min)

        Raises:
            ValueError: If current value not in dimension or n < 0

        Examples:
            >>> dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16, 32, 64))
            >>> dim.step_down(16, 1)
            8
            >>> dim.step_down(16, 2)
            4
            >>> dim.step_down(4, 10)
            1  # Clamped at min
        """
        if n < 0:
            raise ValueError(f"step_down requires n >= 0, got {n}")

        idx = self.index_of(current)
        new_idx = max(idx - n, 0)
        return self.values[new_idx]

    # =========================================================================
    # Percentage-Based Access
    # =========================================================================

    def at_percentage(
        self,
        percentage: float,
        rounding: Literal['natural', 'down', 'up'] = 'natural'
    ) -> int:
        """Get value at percentage position in ordered sequence (0.0-1.0).

        Maps percentage to continuous index space, then rounds to discrete index.
        Useful for sweeping through dimension at regular intervals regardless
        of actual vector length.

        Args:
            percentage: Position in range [0.0, 1.0]
                - 0.0 → first value (min)
                - 1.0 → last value (max)
                - 0.5 → middle value
            rounding: How to round fractional indices
                - 'natural': Round to nearest (default, balanced)
                - 'down': Floor (conservative, prefer smaller values)
                - 'up': Ceiling (aggressive, prefer larger values)

        Returns:
            Value at percentage position

        Raises:
            ValueError: If percentage not in [0.0, 1.0] or invalid rounding mode

        Examples:
            >>> dim = OrderedDimension("PE", (1, 2, 4, 8, 16))  # 5 values
            >>> dim.at_percentage(0.0)
            1
            >>> dim.at_percentage(1.0)
            16
            >>> dim.at_percentage(0.5, rounding='natural')
            4  # Middle value (index 2 of 0-4)
            >>> dim.at_percentage(0.75, rounding='down')
            8  # 0.75 * 4 = 3.0 → floor(3.0) = 3 → values[3] = 8

            >>> # With 4 values, percentages map cleanly to indices
            >>> dim4 = OrderedDimension("X", (10, 20, 30, 40))
            >>> dim4.at_percentage(0.0)
            10  # 0.0 * 3 = 0
            >>> dim4.at_percentage(0.333, rounding='natural')
            20  # 0.333 * 3 ≈ 1.0 → round(1.0) = 1
            >>> dim4.at_percentage(1.0)
            40  # 1.0 * 3 = 3
        """
        if not 0.0 <= percentage <= 1.0:
            raise ValueError(
                f"Percentage must be in [0.0, 1.0], got {percentage}"
            )

        # Map percentage to continuous index space
        max_idx = len(self.values) - 1
        float_idx = percentage * max_idx

        # Apply rounding strategy
        if rounding == 'down':
            idx = int(math.floor(float_idx))
        elif rounding == 'up':
            idx = int(math.ceil(float_idx))
        elif rounding == 'natural':
            idx = round(float_idx)
        else:
            raise ValueError(
                f"Invalid rounding mode '{rounding}'. "
                f"Must be 'natural', 'down', or 'up'."
            )

        # Clamp to valid range (defensive, should be unnecessary)
        idx = max(0, min(idx, max_idx))

        return self.values[idx]

    # =========================================================================
    # Iteration
    # =========================================================================

    def __len__(self) -> int:
        """Number of valid values."""
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        """Iterate over values in order."""
        return iter(self.values)

    def __contains__(self, value: int) -> bool:
        """Check if value in dimension (for 'value in dim' syntax)."""
        return value in self.values

    # =========================================================================
    # Display
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        if len(self.values) <= 6:
            vals = str(self.values)
        else:
            vals = f"({self.values[0]}, {self.values[1]}, ..., {self.values[-1]})"

        default_str = f", default={self.default}" if self.default else ""
        return f"OrderedDimension('{self.name}', {vals}{default_str})"


__all__ = ['OrderedDimension']
