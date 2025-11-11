############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Broadcasting support for elementwise operations.

Implements ONNX multi-directional broadcasting semantics for hardware kernels.
Provides metadata and helper functions for handling broadcast patterns in
HLS code generation and design space exploration.

Key Concepts:
- ONNX Broadcasting: Multi-directional (e.g., [1,64,64,128] + [128])
- Broadcast Last Axis: Special handling when last dimension broadcasts
- PE (Processing Element) Parallelism: Last-dimension parallelization
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BroadcastInfo:
    """Metadata for broadcasting between two tensors.

    Encapsulates broadcasting analysis for hardware implementation:
    - Which dimensions broadcast (size 1 → size N)
    - Whether last axis broadcasts (affects PE parallelism)
    - Effective shapes for memory allocation

    Attributes:
        lhs_shape: Left-hand side tensor shape (tuple)
        rhs_shape: Right-hand side tensor shape (tuple)
        output_shape: Broadcasted output shape (tuple)
        broadcast_dims_lhs: Tuple of dimension indices where LHS broadcasts
        broadcast_dims_rhs: Tuple of dimension indices where RHS broadcasts
        broadcast_last_axis_lhs: True if LHS broadcasts in last dimension
        broadcast_last_axis_rhs: True if RHS broadcasts in last dimension

    Example:
        >>> info = BroadcastInfo.compute((1, 64, 64, 128), (128,))
        >>> info.output_shape
        (1, 64, 64, 128)
        >>> info.broadcast_dims_rhs
        (0, 1, 2)  # RHS broadcasts from rank-1 to rank-4
        >>> info.broadcast_last_axis_rhs
        False  # RHS does not broadcast in last dimension
    """

    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    broadcast_dims_lhs: tuple[int, ...]
    broadcast_dims_rhs: tuple[int, ...]
    broadcast_last_axis_lhs: bool
    broadcast_last_axis_rhs: bool

    @property
    def has_broadcast(self) -> bool:
        """True if any broadcasting occurs."""
        return len(self.broadcast_dims_lhs) > 0 or len(self.broadcast_dims_rhs) > 0

    @property
    def is_compatible_shapes(self) -> bool:
        """True if shapes are already compatible (no broadcasting needed)."""
        return not self.has_broadcast

    @property
    def broadcast_last_axis(self) -> bool:
        """True if either input broadcasts in last dimension (affects PE)."""
        return self.broadcast_last_axis_lhs or self.broadcast_last_axis_rhs

    @classmethod
    def compute(cls, lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> BroadcastInfo:
        """Compute broadcasting metadata from input shapes.

        Implements ONNX multi-directional broadcasting rules:
        1. Right-align shapes
        2. For each dimension, output size is max(lhs, rhs)
        3. Dimensions broadcast if size is 1 and other is not
        4. Missing dimensions (rank mismatch) implicitly broadcast

        Args:
            lhs_shape: Left-hand side tensor shape
            rhs_shape: Right-hand side tensor shape

        Returns:
            BroadcastInfo with computed metadata

        Raises:
            ValueError: If shapes are not broadcastable

        Example:
            >>> BroadcastInfo.compute((1, 64, 64, 128), (128,))
            BroadcastInfo(
                lhs_shape=(1, 64, 64, 128),
                rhs_shape=(128,),
                output_shape=(1, 64, 64, 128),
                broadcast_dims_lhs=(),
                broadcast_dims_rhs=(0, 1, 2),
                broadcast_last_axis_lhs=False,
                broadcast_last_axis_rhs=False
            )
        """
        # Validate shapes are broadcastable and compute output shape
        try:
            output_shape = tuple(np.broadcast_shapes(lhs_shape, rhs_shape))
        except ValueError as e:
            raise ValueError(
                f"Shapes {lhs_shape} and {rhs_shape} are not broadcastable: {e}"
            ) from e

        # Pad shapes to same rank (right-aligned)
        max_rank = max(len(lhs_shape), len(rhs_shape))
        lhs_padded = (1,) * (max_rank - len(lhs_shape)) + lhs_shape
        rhs_padded = (1,) * (max_rank - len(rhs_shape)) + rhs_shape

        # Find dimensions that broadcast (size 1 → size N)
        broadcast_dims_lhs = tuple(
            i for i in range(max_rank) if lhs_padded[i] == 1 and rhs_padded[i] > 1
        )
        broadcast_dims_rhs = tuple(
            i for i in range(max_rank) if rhs_padded[i] == 1 and lhs_padded[i] > 1
        )

        # Check if last axis broadcasts
        broadcast_last_axis_lhs = (
            len(broadcast_dims_lhs) > 0 and broadcast_dims_lhs[-1] == max_rank - 1
        )
        broadcast_last_axis_rhs = (
            len(broadcast_dims_rhs) > 0 and broadcast_dims_rhs[-1] == max_rank - 1
        )

        return cls(
            lhs_shape=lhs_shape,
            rhs_shape=rhs_shape,
            output_shape=output_shape,
            broadcast_dims_lhs=broadcast_dims_lhs,
            broadcast_dims_rhs=broadcast_dims_rhs,
            broadcast_last_axis_lhs=broadcast_last_axis_lhs,
            broadcast_last_axis_rhs=broadcast_last_axis_rhs,
        )

    def get_buffer_shape(self, input_name: str, pe: int) -> tuple[int, ...]:
        """Compute buffer shape for HLS memory allocation.

        For inputs that broadcast, buffer shape may differ from tensor shape:
        - Broadcast dimensions use size 1 (or omitted)
        - Last dimension is divided by PE (processing element parallelism)

        Args:
            input_name: "lhs" or "rhs"
            pe: Processing element parallelism factor (divides last dimension)

        Returns:
            Buffer shape for HLS memory allocation

        Raises:
            ValueError: If input_name is not "lhs" or "rhs"

        Example:
            >>> info = BroadcastInfo.compute((1, 64, 64, 128), (128,))
            >>> info.get_buffer_shape("rhs", pe=64)
            (2,)  # RHS is (128,) → (128/64,) = (2,)
        """
        if input_name == "lhs":
            shape = self.lhs_shape
            broadcast_dims = self.broadcast_dims_lhs
        elif input_name == "rhs":
            shape = self.rhs_shape
            broadcast_dims = self.broadcast_dims_rhs
        else:
            raise ValueError(f"input_name must be 'lhs' or 'rhs', got '{input_name}'")

        # Pad to output rank
        max_rank = len(self.output_shape)
        padded = (1,) * (max_rank - len(shape)) + shape

        # Replace broadcast dimensions with 1
        buffer_shape = tuple(1 if i in broadcast_dims else padded[i] for i in range(max_rank))

        # Divide last dimension by PE
        if buffer_shape:
            buffer_shape = buffer_shape[:-1] + (buffer_shape[-1] // pe,)

        return buffer_shape

    def needs_buffer(self, input_name: str) -> bool:
        """Check if input needs a buffer in HLS code.

        Streaming inputs (dynamic) don't need buffers.
        Static inputs with broadcasting or multi-dimensional access need buffers.

        Args:
            input_name: "lhs" or "rhs"

        Returns:
            True if input requires HLS buffer allocation
        """
        if input_name == "lhs":
            # LHS is typically streaming (dynamic) - no buffer needed
            return False
        elif input_name == "rhs":
            # RHS is typically static parameter - buffer needed if:
            # 1. Multi-dimensional (rank > 1), OR
            # 2. Broadcasting occurs
            return len(self.rhs_shape) > 1 or self.has_broadcast
        else:
            raise ValueError(f"input_name must be 'lhs' or 'rhs', got '{input_name}'")

    def should_read_new_value(self, input_name: str, loop_counters: tuple[str, ...]) -> str | None:
        """Generate condition for conditional reads in HLS loops.

        For static inputs with broadcasting, determine when to read new values
        vs. reuse existing values based on loop structure.

        Args:
            input_name: "lhs" or "rhs"
            loop_counters: Loop counter variable names (outermost to innermost)
                          e.g., ("rep", "spatial", "c", "pe")

        Returns:
            C++ boolean expression for read condition, or None if always read

        Example:
            >>> info = BroadcastInfo.compute((1, 1, 64, 128), (1, 64, 64, 128))
            >>> info.should_read_new_value("rhs", ("rep", "spatial", "c", "pe"))
            "spatial == 0"  # RHS broadcasts in spatial, only read when spatial=0
        """
        if input_name == "lhs":
            broadcast_dims = self.broadcast_dims_lhs
        elif input_name == "rhs":
            broadcast_dims = self.broadcast_dims_rhs
        else:
            raise ValueError(f"input_name must be 'lhs' or 'rhs', got '{input_name}'")

        if not broadcast_dims:
            return None  # No broadcasting, always read

        # Generate conditions for each broadcast dimension
        # Only read when broadcast dimension counters are 0
        conditions = []
        for dim_idx in broadcast_dims:
            if dim_idx < len(loop_counters):
                conditions.append(f"{loop_counters[dim_idx]} == 0")

        if not conditions:
            return None

        return " && ".join(conditions)

    def get_index_expression(
        self, input_name: str, loop_counters: tuple[str, ...], pe_variable: str = "pe"
    ) -> str:
        """Generate C++ indexing expression for buffer access.

        Creates proper indexing that accounts for:
        - Rank padding (implicit dimensions)
        - Broadcast dimensions (use index 0)
        - PE parallelism (divide last dimension)

        Args:
            input_name: "lhs" or "rhs"
            loop_counters: Loop counter variable names (outermost to innermost)
            pe_variable: Variable name for PE index (default "pe")

        Returns:
            C++ buffer indexing expression

        Example:
            >>> info = BroadcastInfo.compute((1, 64, 64, 128), (128,))
            >>> info.get_index_expression("rhs", ("rep", "spatial", "c"), "pe")
            "[pe]"  # RHS is rank-1, only index by PE

            >>> info = BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 64, 128))
            >>> info.get_index_expression("rhs", ("rep", "spatial", "c"), "pe")
            "[0][spatial][c][pe]"  # RHS broadcasts in first dimension
        """
        if input_name == "lhs":
            shape = self.lhs_shape
            broadcast_dims = self.broadcast_dims_lhs
        elif input_name == "rhs":
            shape = self.rhs_shape
            broadcast_dims = self.broadcast_dims_rhs
        else:
            raise ValueError(f"input_name must be 'lhs' or 'rhs', got '{input_name}'")

        # Pad to output rank
        max_rank = len(self.output_shape)
        rank_diff = max_rank - len(shape)

        # Build indexing expression
        indices = []
        for i in range(max_rank):
            if i in broadcast_dims:
                # Broadcast dimension: use 0
                indices.append("0")
            elif i < rank_diff:
                # Implicit dimension from rank padding: use 0
                indices.append("0")
            elif i == max_rank - 1:
                # Last dimension: use PE variable
                indices.append(pe_variable)
            else:
                # Regular dimension: use loop counter
                indices.append(loop_counters[i])

        # Format as C++ array indexing
        return "".join(f"[{idx}]" for idx in indices)


def compute_broadcast_info(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> BroadcastInfo:
    """Convenience function for computing broadcast info.

    Args:
        lhs_shape: Left-hand side tensor shape
        rhs_shape: Right-hand side tensor shape

    Returns:
        BroadcastInfo with computed metadata

    Raises:
        ValueError: If shapes are not broadcastable
    """
    return BroadcastInfo.compute(lhs_shape, rhs_shape)


__all__ = [
    "BroadcastInfo",
    "compute_broadcast_info",
]
