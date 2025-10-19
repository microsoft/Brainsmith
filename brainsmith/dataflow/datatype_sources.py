############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Datatype derivation patterns for kernel schemas.

This module provides an extensible system for deriving datatype values from
other interfaces. All patterns inherit from the DatatypeSource ABC.

Common Patterns:
    - DerivedDatatype: Copy datatype from another interface
    - WidenedDatatype: Add bits for overflow protection
    - UnionDatatype: Compute datatype as union of input ranges
    - ComputedDatatype: Custom computation (escape hatch)

Example:
    >>> # Simple copy
    >>> DerivedDatatype("input")  # Output datatype = input datatype

    >>> # Widen for overflow (ElementwiseAdd)
    >>> WidenedDatatype("input0", extra_bits=1)

    >>> # Union of ranges (Concat)
    >>> UnionDatatype(("input0", "input1", "input2"))
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple
import math
import numpy as np

from qonnx.core.datatype import DataType, BaseDataType
from .utils import get_interface


@dataclass(frozen=True)
class DatatypeSource(ABC):
    """Base class for datatype derivation strategies.

    Subclass to add new datatype derivation patterns.
    All subclasses must be frozen dataclasses for immutability and hashability.

    The resolve() method is called during model building to compute the
    datatype from available interfaces and parameters.
    """

    @abstractmethod
    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable[[str], Any]) -> DataType:
        """Compute datatype from interfaces and parameters.

        Args:
            interfaces: Dict mapping interface name -> InterfaceModel
            param_getter: Function to retrieve nodeattr values

        Returns:
            Resolved DataType instance

        Raises:
            ValueError: If datatype cannot be resolved or is invalid
        """
        pass


@dataclass(frozen=True)
class DerivedDatatype(DatatypeSource):
    """Copy datatype from another interface.

    Most common pattern for type-preserving operations where output datatype
    should exactly match input datatype.

    Examples:
        LayerNorm output datatype = input datatype:
        >>> DerivedDatatype("input")

        Copy from specific input:
        >>> DerivedDatatype("input0")
    """
    source_interface: str

    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable) -> DataType:
        """Copy datatype from source interface or internal datatype."""
        source = get_interface(interfaces, self.source_interface, "DerivedDatatype")

        # Handle both DataType (internal datatypes stored directly) and models (interfaces)
        if isinstance(source, DataType):
            return source  # Internal datatype stored directly

        try:
            return source.datatype
        except AttributeError:
            raise ValueError(
                f"Source '{self.source_interface}' does not have "
                f"datatype attribute"
            )


@dataclass(frozen=True)
class WidenedDatatype(DatatypeSource):
    """Widen datatype by adding bits (for overflow protection).

    Common for accumulation operations (add, sum) where output needs extra
    bits to prevent overflow.

    Examples:
        ElementwiseAdd output = input + 1 bit for overflow:
        >>> WidenedDatatype("input0", extra_bits=1)

        Accumulator with 2 extra bits:
        >>> WidenedDatatype("input", extra_bits=2)
    """
    source_interface: str
    extra_bits: int

    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable) -> DataType:
        """Add extra bits to source datatype."""
        if self.extra_bits < 0:
            raise ValueError(f"extra_bits must be non-negative, got {self.extra_bits}")

        source = get_interface(interfaces, self.source_interface, "WidenedDatatype")

        # Handle both DataType (internal datatypes) and models (interfaces)
        if isinstance(source, DataType):
            base_dt = source
        else:
            base_dt = source.datatype

        new_width = base_dt.bitwidth() + self.extra_bits

        # Preserve signedness
        if base_dt.signed():
            return DataType[f"INT{new_width}"]
        else:
            return DataType[f"UINT{new_width}"]


@dataclass(frozen=True)
class UnionDatatype(DatatypeSource):
    """Compute datatype as union of input ranges.

    Common for concatenation operations where inputs may have different
    datatype ranges and output must accommodate all values.

    The output datatype is the minimum bitwidth that can represent the
    union of all input value ranges.

    Examples:
        Concat output = union of all input datatype ranges:
        >>> UnionDatatype(("input0", "input1", "input2"))

        Binary operation with different input ranges:
        >>> UnionDatatype(("input0", "input1"))
    """
    source_interfaces: Tuple[str, ...]

    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable) -> DataType:
        """Compute minimum datatype that covers all input ranges."""
        if not self.source_interfaces:
            raise ValueError("UnionDatatype requires at least one source interface")

        min_val = 0
        max_val = 0

        for name in self.source_interfaces:
            interface = get_interface(interfaces, name, "UnionDatatype")

            # Handle both DataType (internal datatypes) and models (interfaces)
            if isinstance(interface, DataType):
                dt = interface
            else:
                dt = interface.datatype

            min_val = min(min_val, dt.min())
            max_val = max(max_val, dt.max())

        # Compute bitwidth from range
        if min_val >= 0:
            # Unsigned: bitwidth = ceil(log2(max + 1))
            if max_val == 0:
                bitwidth = 1
            else:
                bitwidth = math.ceil(np.log2(max_val + 1))
            return DataType[f"UINT{bitwidth}"]
        else:
            # Signed: bitwidth = ceil(log2(max(-min, max+1))) + 1
            max_abs = max(-min_val, 1 + max_val)
            bitwidth = math.ceil(np.log2(max_abs) + 1)
            return DataType[f"INT{bitwidth}"]


@dataclass(frozen=True)
class ComputedDatatype(DatatypeSource):
    """Custom datatype computation.

    Escape hatch for complex datatype logic that doesn't fit built-in patterns.
    Provide clear documentation via the description field.

    Examples:
        MatMul accumulator datatype:
        >>> def compute_matmul_acc(ifs, pg):
        ...     input_dt = ifs["input"].datatype
        ...     weight_dt = ifs["weight"].datatype
        ...     mw = pg("MW")
        ...     # Compute accumulator range
        ...     acc_min = input_dt.min() * weight_dt.min() * mw
        ...     acc_max = input_dt.max() * weight_dt.max() * mw
        ...     # Compute bitwidth...
        ...     return DataType[f"INT{bitwidth}"]
        >>> ComputedDatatype(compute_matmul_acc, "MatMul accumulator")

        Custom range-based computation:
        >>> def custom_logic(interfaces, param_getter):
        ...     # Your custom datatype computation
        ...     return computed_datatype
        >>> ComputedDatatype(custom_logic, "Custom datatype logic")
    """
    compute_fn: Callable[[Dict[str, Any], Callable], DataType]
    description: str = ""  # Strongly encouraged for documentation

    def resolve(self, interfaces: Dict[str, Any], param_getter: Callable) -> DataType:
        """Call custom function to compute datatype."""
        try:
            result = self.compute_fn(interfaces, param_getter)
        except Exception as e:
            desc = f" ({self.description})" if self.description else ""
            raise ValueError(
                f"ComputedDatatype{desc} function raised exception: {e}"
            )

        if not isinstance(result, BaseDataType):
            desc = f" ({self.description})" if self.description else ""
            raise ValueError(
                f"ComputedDatatype{desc} function must return DataType, "
                f"got {type(result).__name__}"
            )

        return result

    def __repr__(self):
        """Show description in repr for better error messages."""
        if self.description:
            return f"ComputedDatatype({self.description})"
        return "ComputedDatatype(custom)"


__all__ = [
    'DatatypeSource',
    'DerivedDatatype',
    'WidenedDatatype',
    'UnionDatatype',
    'ComputedDatatype',
]
