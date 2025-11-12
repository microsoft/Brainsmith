############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Shared utilities for the dataflow system."""

from collections.abc import Callable, Iterator
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from .kernel_op import KernelOp


def iter_valid_configurations(
    kernel_op: "KernelOp",
    model_w: "ModelWrapper",
    param_filters: dict[str, Callable[[int], bool]] | None = None,
) -> Iterator[dict[str, int]]:
    """Iterate over all valid parallelization configurations.

    Generates Cartesian product of valid parameter ranges from kernel's
    get_valid_ranges() method. Optionally filters by custom predicates.

    This utility is designed for Design Space Exploration (DSE), enabling
    systematic exploration of all valid kernel configurations. Use this
    instead of trial-and-error approaches with try-except blocks.

    Args:
        kernel_op: KernelOp instance to explore
        model_w: ModelWrapper containing the model
        param_filters: Optional per-parameter filter functions.
            Keys are parameter names (e.g., "SIMD", "PE").
            Values are predicates that take an int and return bool.
            Example: {"SIMD": lambda x: x >= 4 and x <= 128}
            Only values passing the filter will be included.

    Yields:
        Dict of parallelization parameters, e.g., {"SIMD": 64, "PE": 1}
        Yields configs in sorted order (by parameter names, then values)

    Example:
        >>> kernel_op = LayerNorm(node)
        >>> for config in iter_valid_configurations(kernel_op, model_w):
        ...     # Navigate design space to create candidate point
        ...     point = kernel_op.design_point
        ...     for param, value in config.items():
        ...         point = point.with_dimension(param, value)
        ...     # Profile or analyze design point
        ...     cycles = point.estimate_cycles()

    Example with filtering:
        >>> # Only explore SIMD values >= 4
        >>> filters = {"SIMD": lambda x: x >= 4}
        >>> for config in iter_valid_configurations(kernel_op, model_w, filters):
        ...     # Process config
        ...     pass

    Performance:
        - Uses get_valid_ranges() which is <1ms (cached after first call)
        - Iterator-based: memory efficient for large design spaces
        - Total iterations = product of |valid_ranges[param]| for all params
        - Example: SIMD has 18 divisors, PE has 4 → 72 total configs

    Notes:
        - Only generates valid configurations (no failed builds)
        - Explores full Cartesian product (exhaustive search)
        - For large design spaces, consider filtering to reduce search space
        - Each configuration is guaranteed to work with configure()
    """
    # Get valid ranges from kernel
    # Note: Returns Dict[str, Union[OrderedParameter, frozenset]] in new API
    dimensions = kernel_op.get_valid_ranges(model_w)

    # Early return if no parallelization parameters
    if not dimensions:
        return

    # Extract values from dimensions (handle both OrderedParameter and frozenset)
    # Import here to avoid circular dependency
    from .ordered_parameter import OrderedParameter

    valid_ranges = {}
    for name, dim in dimensions.items():
        if isinstance(dim, OrderedParameter):
            # OrderedParameter: extract values tuple
            valid_ranges[name] = set(dim.values)
        else:
            # frozenset: convert to set for filtering compatibility
            valid_ranges[name] = set(dim)

    # Apply filters if provided
    if param_filters:
        for param_name, filter_fn in param_filters.items():
            if param_name in valid_ranges:
                # Filter the valid range
                valid_ranges[param_name] = {v for v in valid_ranges[param_name] if filter_fn(v)}

    # Remove parameters with empty ranges (after filtering)
    valid_ranges = {k: v for k, v in valid_ranges.items() if v}

    # Early return if filtering eliminated all values
    if not valid_ranges:
        return

    # Generate Cartesian product
    # Sort keys and values for deterministic iteration order
    param_names = sorted(valid_ranges.keys())
    param_values = [sorted(valid_ranges[name]) for name in param_names]

    # Yield each configuration as a dict
    for value_tuple in product(*param_values):
        yield dict(zip(param_names, value_tuple))


__all__ = ["iter_valid_configurations"]
