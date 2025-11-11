"""Binary operation definitions for ElementwiseBinaryOp kernel.

Single source of truth for all 17 supported binary operations.
Eliminates duplication across npy_op and cpp_op properties.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryOperation:
    """Definition of a single binary operation.

    Attributes:
        name: ONNX operation name (e.g., "Add", "Mul")
        category: Operation category for grouping
        npy_op: NumPy ufunc for python execution
        cpp_template: C++ expression template with {0} and {1} placeholders
        description: Human-readable description
    """
    name: str
    category: str
    npy_op: Callable
    cpp_template: str
    description: str


class BinaryOperations:
    """Registry of all supported binary operations.

    Provides centralized operation definitions and query methods.
    Used by ElementwiseBinaryOp for execution and code generation.
    """

    # Arithmetic Operations (4)
    ADD = BinaryOperation(
        name="Add",
        category="arithmetic",
        npy_op=np.add,
        cpp_template="({0} + {1})",
        description="Addition"
    )
    SUB = BinaryOperation(
        name="Sub",
        category="arithmetic",
        npy_op=np.subtract,
        cpp_template="({0} - {1})",
        description="Subtraction"
    )
    MUL = BinaryOperation(
        name="Mul",
        category="arithmetic",
        npy_op=np.multiply,
        cpp_template="({0} * {1})",
        description="Multiplication"
    )
    DIV = BinaryOperation(
        name="Div",
        category="arithmetic",
        npy_op=np.divide,
        cpp_template="({0} / {1})",
        description="Division"
    )

    # Logical Operations (3)
    AND = BinaryOperation(
        name="And",
        category="logical",
        npy_op=np.logical_and,
        cpp_template="({0} && {1})",
        description="Logical AND"
    )
    OR = BinaryOperation(
        name="Or",
        category="logical",
        npy_op=np.logical_or,
        cpp_template="({0} || {1})",
        description="Logical OR"
    )
    XOR = BinaryOperation(
        name="Xor",
        category="logical",
        npy_op=np.logical_xor,
        cpp_template="(bool({0}) != bool({1}))",
        description="Logical XOR"
    )

    # Comparison Operations (5)
    EQUAL = BinaryOperation(
        name="Equal",
        category="comparison",
        npy_op=np.equal,
        cpp_template="({0} == {1})",
        description="Equality comparison"
    )
    LESS = BinaryOperation(
        name="Less",
        category="comparison",
        npy_op=np.less,
        cpp_template="({0} < {1})",
        description="Less than"
    )
    LESS_OR_EQUAL = BinaryOperation(
        name="LessOrEqual",
        category="comparison",
        npy_op=np.less_equal,
        cpp_template="({0} <= {1})",
        description="Less than or equal"
    )
    GREATER = BinaryOperation(
        name="Greater",
        category="comparison",
        npy_op=np.greater,
        cpp_template="({0} > {1})",
        description="Greater than"
    )
    GREATER_OR_EQUAL = BinaryOperation(
        name="GreaterOrEqual",
        category="comparison",
        npy_op=np.greater_equal,
        cpp_template="({0} >= {1})",
        description="Greater than or equal"
    )

    # Bitwise Operations (3)
    BITWISE_AND = BinaryOperation(
        name="BitwiseAnd",
        category="bitwise",
        npy_op=np.bitwise_and,
        cpp_template="({0} & {1})",
        description="Bitwise AND"
    )
    BITWISE_OR = BinaryOperation(
        name="BitwiseOr",
        category="bitwise",
        npy_op=np.bitwise_or,
        cpp_template="({0} | {1})",
        description="Bitwise OR"
    )
    BITWISE_XOR = BinaryOperation(
        name="BitwiseXor",
        category="bitwise",
        npy_op=np.bitwise_xor,
        cpp_template="({0} ^ {1})",
        description="Bitwise XOR"
    )

    # Bit Shift Operation (1)
    # ONNX standard: BitShift with direction attribute ("LEFT" or "RIGHT")
    BIT_SHIFT = BinaryOperation(
        name="BitShift",
        category="bitshift",
        npy_op=np.left_shift,  # Default to left shift; actual operation determined by direction attribute
        cpp_template="({0} << {1})",  # Default template; actual template determined by direction attribute
        description="Bit shift (direction determined by attribute)"
    )

    # Internal registry
    _ALL_OPERATIONS = [
        ADD, SUB, MUL, DIV,
        AND, OR, XOR,
        EQUAL, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL,
        BITWISE_AND, BITWISE_OR, BITWISE_XOR,
        BIT_SHIFT
    ]

    _OPERATION_MAP = {op.name: op for op in _ALL_OPERATIONS}

    @classmethod
    def get(cls, name: str) -> BinaryOperation:
        """Get operation by name.

        Args:
            name: ONNX operation name (e.g., "Add", "Mul")

        Returns:
            BinaryOperation instance

        Raises:
            KeyError: If operation name not found
        """
        return cls._OPERATION_MAP[name]

    @classmethod
    def all_operation_names(cls) -> set[str]:
        """Get set of all operation names for schema validation.

        Returns:
            Set of operation names (e.g., {"Add", "Sub", "Mul", ...})
        """
        return set(cls._OPERATION_MAP.keys())

    @classmethod
    def by_category(cls, category: str) -> list[BinaryOperation]:
        """Get all operations in a category.

        Args:
            category: Category name ("arithmetic", "logical", "comparison",
                                     "bitwise", "bitshift")

        Returns:
            List of operations in the category
        """
        return [op for op in cls._ALL_OPERATIONS if op.category == category]

    @classmethod
    def get_npy_op(cls, name: str) -> Callable:
        """Get NumPy operation for execution.

        Args:
            name: ONNX operation name

        Returns:
            NumPy ufunc (e.g., np.add, np.multiply)
        """
        return cls.get(name).npy_op

    @classmethod
    def get_cpp_template(cls, name: str) -> str:
        """Get C++ template for code generation.

        Args:
            name: ONNX operation name

        Returns:
            C++ expression template with {0} and {1} placeholders

        Example:
            >>> BinaryOperations.get_cpp_template("Add")
            "({0} + {1})"
        """
        return cls.get(name).cpp_template
